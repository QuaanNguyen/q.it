from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import locked_environment, profile_for
from qit_adapters.models import EmbeddingRuntime, load_embedding
from qit_adapters.registry import ADAPTERS


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def numel(self):
        if isinstance(self.data[0], list):
            return sum(len(row) for row in self.data)
        return len(self.data)

    def tolist(self):
        return self.data

    def unsqueeze(self, dimension):
        return self


class Score:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value


def maxsim(left, right, backend=None):
    assert backend == "torch"
    best = None
    for document_token in right.data:
        score = sum(
            query_value * document_value
            for query_value, document_value in zip(left.data[0], document_token)
        )
        best = score if best is None else max(best, score)
    return Score(best)


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float32 = "torch.float32"
        self.version = SimpleNamespace(cuda=None)
        self.threads = None
        self.interop = 1

    def set_num_threads(self, value):
        self.threads = value

    def get_num_interop_threads(self):
        return self.interop

    def set_num_interop_threads(self, value):
        self.interop = value

    def inference_mode(self):
        from contextlib import nullcontext

        return nullcontext()

    def as_tensor(self, value, device=None, dtype=None):
        return value if isinstance(value, FakeTensor) else FakeTensor(value)


def parameter(elements):
    return SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype="torch.float32",
        shape=[elements],
        numel=lambda: elements,
    )


class FakeModel:
    def __init__(self, vector_values, vectors, dimension=128):
        self.vector_values = vector_values
        self.vectors = vectors
        self.dimension = dimension
        self.max_seq_length = 512
        self.prompts = {"query": "query", "document": "document"}
        self.query_prefix = "Query: "
        self.document_prefix = "Document: "
        self.query_length = 32
        self.document_length = 180

    def float(self):
        return self

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def tokenizer(self, text, truncation=True):
        assert truncation is False
        return {"input_ids": list(range(len(text)))}

    def encode(
        self,
        texts,
        is_query=None,
        prompt_name=None,
        batch_size=None,
        convert_to_numpy=True,
        show_progress_bar=False,
        pool_factor=None,
        normalize_embeddings=None,
        prompt=None,
    ):
        if is_query is None:
            return SimpleNamespace(tolist=lambda: [[0.5] * 768 for _ in texts])
        values = self.vector_values if is_query else self.vectors[: len(texts)]
        return [FakeTensor([row]) for row in values]

    def named_parameters(self):
        return [("weight", parameter(4))]

    def named_buffers(self):
        return []


def vector(value):
    return [0.0] * 127 + [value]


def modules(dimension=128):
    torch = FakeTorch()
    pylate = ModuleType("pylate")
    pylate_models = ModuleType("pylate.models")
    pylate_scores = ModuleType("pylate.scores")
    pylate_scores.colbert_scores = maxsim
    model = FakeModel(
        [vector(1.0)], [vector(3.0), vector(2.0), vector(1.0)], dimension=dimension
    )

    class ColBERT:
        def __new__(cls, *args, **kwargs):
            return model

    pylate_models.ColBERT = ColBERT
    pylate.models = pylate_models
    pylate.scores = pylate_scores
    pylate.__path__ = []
    pylate_scores.__path__ = []
    sentence_transformers = ModuleType("sentence_transformers")

    class SentenceTransformer(ColBERT):
        pass

    sentence_transformers.SentenceTransformer = SentenceTransformer
    return {
        "torch": torch,
        "pylate": pylate,
        "pylate.models": pylate_models,
        "pylate.scores": pylate_scores,
        "sentence_transformers": sentence_transformers,
    }, model


def profile(adapter):
    name = {
        "uibk": "DataScience-UIBK/uibk-embedding-v1",
        "uibk-agent": "DataScience-UIBK/Agent-uibk-v1",
        "ultrabert": "dmis-lab/UltraBERT-Embed-150M",
    }[adapter]
    return profile_for(name)


def case(query="Which document answers the question?"):
    return {
        "id": "one",
        "input": query,
        "documents": ["alpha", "beta", "gamma"],
    }


class RetrievalTests(unittest.TestCase):
    def test_native_maxsim_retains_every_vector_with_document_order_ties(self):
        injected, model = modules()
        with patch.dict(sys.modules, injected):
            runtime = load_embedding(profile("uibk"), Path("/models/uibk"), 2)
            self.assertEqual(runtime.retrieval["method"], "maxsim")
            self.assertEqual(runtime.retrieval["dimensions"], 128)
            self.assertEqual(runtime.preparation["query_prompt"], "query")
            self.assertEqual(runtime.audit["parameter_count"], 4)
            observed = runtime.invoke(case(), 64)["retrieval"]
        self.assertEqual(observed["schema_version"], 1)
        self.assertEqual(observed["method"], "maxsim")
        self.assertEqual(len(observed["query"]), 1)
        self.assertEqual(len(observed["documents"]), 3)
        self.assertEqual(observed["scores"], [3.0, 2.0, 1.0])
        self.assertEqual(observed["ranking"], [0, 1, 2])
        self.assertEqual(observed["tie_break"], "document_order")

    def test_agent_uibk_requires_the_matching_reasoning_pack(self):
        injected, _ = modules()
        with patch.dict(sys.modules, injected):
            runtime = load_embedding(profile("uibk-agent"), Path("/models/agent"), 2)
            with self.assertRaisesRegex(ValueError, "AgentIR"):
                runtime.invoke(case("plain question"), 64)
            observed = runtime.invoke(
                case("Reasoning: the passage names the port.\n\nQuery: which port?"), 64
            )["retrieval"]
        self.assertEqual(observed["ranking"], [0, 1, 2])

    def test_role_preparation_owned_by_the_publisher_model_is_refused(self):
        injected, _ = modules()
        owned = case()
        owned["query_prefix"] = "Custom: "
        with patch.dict(sys.modules, injected):
            runtime = load_embedding(profile("uibk"), Path("/models/uibk"), 2)
            with self.assertRaisesRegex(
                ValueError, "owned by the pinned publisher model"
            ):
                runtime.invoke(owned, 64)

    def test_raw_observation_limit_is_enforced_before_scoring(self):
        injected, model = modules()
        model.vectors = [vector(1.0)] * 8200
        oversized = {
            "id": "many",
            "input": "question",
            "documents": [f"document {index}" for index in range(8200)],
        }
        with patch.dict(sys.modules, injected):
            runtime = load_embedding(profile("uibk"), Path("/models/uibk"), 2)
            with self.assertRaisesRegex(ValueError, "one million scalars"):
                runtime.invoke(oversized, 64)

    def test_ultrabert_returns_publisher_dense_embeddings_and_refuses_truncation(self):
        injected, model = modules(dimension=768)
        with patch.dict(sys.modules, injected):
            runtime = load_embedding(profile("ultrabert"), Path("/models/ultra"), 4)
            self.assertEqual(runtime.preparation["dimensions"], 768)
            self.assertFalse(hasattr(runtime, "retrieval"))
            embeddings = runtime.invoke(case(), 64)["embeddings"]
            self.assertEqual(len(embeddings), 4)
            self.assertEqual(len(embeddings[0]), 768)
            model.max_seq_length = 8
            with self.assertRaisesRegex(ValueError, "truncated"):
                runtime.invoke(case(), 64)

    def test_registry_routes_reviewed_embedding_sources(self):
        for adapter in ("ultrabert", "uibk", "uibk-agent"):
            self.assertTrue(ADAPTERS[adapter].reviewed_code)
            self.assertIs(ADAPTERS[adapter].load, load_embedding)

    def test_x86_only_uibk_refuses_arm64_with_its_declared_reason(self):
        row = profile("uibk")
        self.assertEqual(row["native_architectures"], ["x86_64"])
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="aarch64"),
            patch("platform.libc_ver", return_value=("glibc", "2.36")),
        ):
            with self.assertRaisesRegex(
                ValueError, "uibk has no retained lock for aarch64"
            ):
                locked_environment(row)

    def test_embedding_runtime_rejects_unknown_adapters(self):
        with self.assertRaisesRegex(ValueError, "unsupported native embedding adapter"):
            EmbeddingRuntime(
                {"adapter": "unknown", "precision": "float32"},
                Path("/none"),
                FakeTorch(),
            )


if __name__ == "__main__":
    unittest.main()
