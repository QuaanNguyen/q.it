import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qit_adapters.decisions import DecisionRuntime, distribution, load_decision
from qit_adapters.artifacts import lock_family, profile_for, profiles

CANDIDATES = [{"id": "second", "text": "Rain"}, {"id": "first", "text": "Clear"}]


def payload():
    return {
        "model": "fixture",
        "task": "decision_scoring",
        "case": {
            "id": "one",
            "decision": {
                "state": "The sky is blue",
                "question": "What fits?",
                "candidates": CANDIDATES,
            },
        },
    }


class ClassifierTests(unittest.TestCase):
    def test_catalog_has_complete_pinned_sub_billion_card_evidence(self):
        catalog = profiles()
        self.assertEqual(
            {row["adapter"] for row in catalog},
            {
                "gliclass",
                "gliner2",
                "intern",
                "liquid",
                "typed-gemma",
                "frida",
                "d3",
                "ultrabert",
                "uibk",
                "uibk-agent",
                "lighton",
                "indra",
            },
        )
        for row in catalog:
            self.assertEqual(profile_for(row["model"]), row)
            self.assertIs(type(row["model_card"]["parameter_count"]), int)
            self.assertGreater(row["model_card"]["parameter_count"], 0)
            self.assertLess(row["model_card"]["parameter_count"], 1_000_000_000)
            self.assertIn(row["revision"], row["model_card"]["url"])
            self.assertIn(row["revision"], row["model_card"]["count_source"])
            self.assertTrue(row["files"])
            for architecture in row.get("native_architectures", ("x86_64", "aarch64")):
                lock = (
                    Path(__file__).resolve().parents[1]
                    / "qit_adapters"
                    / "locks"
                    / f"{lock_family(row['adapter'])}-linux-{architecture}-py312.txt"
                )
                self.assertIn("--hash=sha256:", lock.read_text())

    def test_invalid_card_counts_are_refused_before_loading(self):
        row = profiles()[0]
        for count in (None, True, 0, -1, 1_000_000_000, 1_000_000_001):
            value = {**row, "model_card": {"parameter_count": count}}
            with patch("qit_adapters.artifacts.profiles", return_value=[value]):
                with self.assertRaisesRegex(ValueError, "model card"):
                    profile_for(value["model"])

    def test_loader_rejects_cuda_builds_and_non_float32_profiles(self):
        torch = SimpleNamespace(version=SimpleNamespace(cuda="12.8"))
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                load_decision({"precision": "float32"}, Path("/tmp"), 4)
            torch.version.cuda = None
            with self.assertRaisesRegex(ValueError, "float32"):
                load_decision({"precision": "bfloat16"}, Path("/tmp"), 4)

    def test_loader_audits_the_constructed_cpu_stack(self):
        parameter = SimpleNamespace(
            device=SimpleNamespace(type="cpu"),
            dtype="float32",
            shape=(10,),
            numel=lambda: 10,
        )
        model = SimpleNamespace(
            named_parameters=lambda: [("weights", parameter)], named_buffers=lambda: []
        )
        runtime = SimpleNamespace(model=model, extra_tensors={})
        threads = []
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            float32="float32",
            set_num_threads=threads.append,
            get_num_interop_threads=lambda: 1,
        )
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch("qit_adapters.decisions.DecisionRuntime", return_value=runtime),
        ):
            observed = load_decision({"precision": "float32"}, Path("/tmp"), 4)
        self.assertIs(observed, runtime)
        self.assertEqual(threads, [4])
        self.assertEqual(observed.audit["parameter_count"], 10)

    def test_native_scores_are_preserved_in_request_order_without_normalization(self):
        self.assertEqual(
            distribution(CANDIDATES, {"first": 0.75, "second": 0.25}, "first"),
            [
                {"id": "second", "probability": 0.25},
                {"id": "first", "probability": 0.75},
            ],
        )
        self.assertEqual(
            distribution(CANDIDATES, {"second": 0.5, "first": 0.5}, "second")[0][
                "probability"
            ],
            0.5,
        )
        for scores, selected in [
            ({"first": 1.0}, "first"),
            ({"second": 0.5, "first": 0.6}, "first"),
            ({"second": float("nan"), "first": 1.0}, "first"),
            ({"second": 0.25, "first": 0.75}, "second"),
        ]:
            with self.assertRaises(ValueError):
                distribution(CANDIDATES, scores, selected)

    def test_gliner_observes_complete_native_scores_without_changing_public_decoding(
        self,
    ):
        class Values:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [0.25, 0.75]

        class Native:
            def _compute_classification_probabilities(self):
                return ("choice", {"labels": ["second", "first"]}, Values())

            def _classification_schema(self, tasks):
                return tasks

            def _build_schema_dicts_and_metadata(self, schemas):
                return (schemas, [])

            def classify_text(self, state, tasks, **kwargs):
                self._compute_classification_probabilities()
                return {"choice": {"label": "first", "confidence": 0.75}}

        runtime = DecisionRuntime.__new__(DecisionRuntime)
        runtime.profile = {"adapter": "gliner2"}
        runtime.model = Native()
        original = runtime.model._compute_classification_probabilities
        runtime.model.processor = SimpleNamespace(
            _transform_record=lambda *args, **kwargs: SimpleNamespace(
                input_ids=[1, 2, 3]
            )
        )
        runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        runtime.configuration = {
            "calibration": "native",
            "selection_rule": "native",
            "preprocessing": "native",
        }
        observed = runtime.invoke(payload()["case"])
        self.assertEqual(
            observed["decision"]["candidates"],
            distribution(CANDIDATES, {"second": 0.25, "first": 0.75}, "first"),
        )
        self.assertEqual(runtime.model._compute_classification_probabilities, original)
        runtime.model.classify_text = lambda *args, **kwargs: {"choice": "first"}
        with self.assertRaisesRegex(
            ValueError, "complete native candidate distribution"
        ):
            runtime.invoke(payload()["case"])
        self.assertEqual(runtime.model._compute_classification_probabilities, original)
