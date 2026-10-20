import contextlib
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import profiles
from qit_adapters.backend import Backend
from qit_adapters.decisions import DecisionRuntime
from qit_adapters.registry import ADAPTERS


def case():
    return {
        "decision": {
            "state": "The sky is clear",
            "question": "What fits?",
            "candidates": [
                {"id": "rain", "text": "Rain"},
                {"id": "clear", "text": "Clear"},
            ],
        }
    }


def runtime(adapter):
    value = DecisionRuntime.__new__(DecisionRuntime)
    value.profile = {"adapter": adapter}
    value.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    value.configuration = {
        "calibration": "publisher-fixture",
        "selection_rule": "publisher-native-argmax",
        "preprocessing": "publisher-fixture",
    }
    return value


def answer():
    return {"probabilities": {"clear": 0.75, "rain": 0.25}, "choice": "clear"}


class PublisherDecisionTests(unittest.TestCase):
    def test_publisher_routes_require_explicit_review_before_runtime_loading(self):
        for profile in profiles():
            if profile["adapter"] not in {"intern", "liquid", "typed-gemma"}:
                continue
            self.assertTrue(ADAPTERS[profile["adapter"]].reviewed_code)
            source = {
                name: identity
                for name, identity in profile["files"].items()
                if name.endswith(".py")
            }
            self.assertTrue(source)
            self.assertTrue(
                all(len(identity["sha256"]) == 64 for identity in source.values())
            )
            with (
                tempfile.TemporaryDirectory() as directory,
                patch("qit_adapters.backend.verify", return_value={}),
                patch("qit_adapters.backend.locked_environment") as environment,
                patch("qit_adapters.backend.forbid_external_connections") as offline,
            ):
                with self.assertRaisesRegex(ValueError, "allow-reviewed-code"):
                    Backend(profile, directory)
                environment.assert_not_called()
                offline.assert_not_called()

    def test_intern_uses_publisher_request_and_preserves_complete_probabilities(self):
        value = runtime("intern")
        requests = []
        encoded = SimpleNamespace(tolist=lambda: [[1, 2, 3]])

        def encode(request):
            requests.append(copy.deepcopy(request))
            return (
                SimpleNamespace(messages=[{"role": "user", "content": "compiled"}]),
                {"input_ids": encoded},
                None,
            )

        value.engine = SimpleNamespace(
            backend=SimpleNamespace(encode=encode),
            predict=lambda request: {"answers": {"choice": answer()}},
        )
        observed = value.invoke(case())["decision"]
        self.assertEqual(requests[0]["state"], case()["decision"]["state"])
        self.assertEqual(
            list(requests[0]["questions"]["choice"]["criteria"]), ["rain", "clear"]
        )
        self.assertEqual(observed["selected_id"], "clear")
        self.assertEqual(
            observed["candidates"],
            [{"id": "rain", "probability": 0.25}, {"id": "clear", "probability": 0.75}],
        )
        self.assertEqual(observed["provenance"]["native"], answer())
        self.assertFalse(observed["provenance"]["input_truncated"])
        self.assertEqual(len(observed["provenance"]["input_digest"]), 64)

    def test_intern_preserves_publisher_overlength_rejection(self):
        value = runtime("intern")

        def reject(request):
            raise ValueError("above 8192; truncation is forbidden")

        value.engine = SimpleNamespace(backend=SimpleNamespace(encode=reject))
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            value.invoke(case())

    def test_typed_gemma_uses_native_answer_and_rejects_truncation(self):
        value = runtime("typed-gemma")
        encoded = SimpleNamespace(tolist=lambda: [[1, 2]])
        native = SimpleNamespace(
            probabilities=answer()["probabilities"], label="clear", to_dict=answer
        )
        value.tokenizer = lambda text, **kwargs: {"input_ids": text.split()}
        value.model = SimpleNamespace(
            config=SimpleNamespace(max_seq_length=8, prompt=""),
            _build_inputs=lambda *args: (
                {"input_ids": encoded},
                None,
                ["compiled text"],
            ),
            decide=lambda *args, **kwargs: {"choice": native},
        )
        observed = value.invoke(case())["decision"]
        self.assertEqual(observed["selected_id"], "clear")
        self.assertEqual(observed["candidates"][0], {"id": "rain", "probability": 0.25})
        self.assertEqual(observed["provenance"]["native"], answer())
        value.model.config.max_seq_length = 1
        with self.assertRaisesRegex(ValueError, "would be truncated"):
            value.invoke(case())

    def test_liquid_uses_native_preparation_and_rejects_each_truncation(self):
        value = runtime("liquid")
        value.tokenizer = lambda text, **kwargs: {"input_ids": text.split()}
        value.model = SimpleNamespace(
            config=SimpleNamespace(max_length=128),
            system_one=lambda **kwargs: {"answers": {"choice": answer()}},
        )
        prompt = SimpleNamespace(
            as_question=lambda request: SimpleNamespace(
                instructions=request["instructions"]
            ),
            render_options=lambda request: ["Rain", "Clear"],
            escape=str,
            serialize=str,
            encode=lambda *args: ([1, 2], [1]),
        )
        with patch(
            "qit_adapters.decisions.importlib.import_module", return_value=prompt
        ):
            observed = value.invoke(case())["decision"]
            self.assertEqual(observed["selected_id"], "clear")
            self.assertEqual(observed["provenance"]["native"], answer())
            long_question = case()
            long_question["decision"]["question"] = "word " * 100
            with self.assertRaisesRegex(ValueError, "question would be truncated"):
                value.invoke(long_question)
            prompt.render_options = lambda request: ["word " * 100]
            with self.assertRaisesRegex(
                ValueError, "candidate text would be truncated"
            ):
                value.invoke(case())
            prompt.render_options = lambda request: ["Rain", "Clear"]
            long_state = case()
            long_state["decision"]["state"] = "word " * 129
            with self.assertRaisesRegex(ValueError, "state would be truncated"):
                value.invoke(long_state)

    def test_intern_and_liquid_refuse_direct_image_execution(self):
        for adapter in ("intern", "liquid"):
            with self.assertRaisesRegex(ValueError, "image execution awaits"):
                runtime(adapter).invoke(
                    {**case(), "media": "data:image/png;base64,AAAA"}
                )
