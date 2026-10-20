import contextlib
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import fingerprint, profile_for
from qit_adapters.backend import Backend
from qit_adapters.decisions import load_decision
from qit_adapters.registry import ADAPTERS, Adapter


MODELS = [
    ("vllm-sr/d3-edge", 594321600, 594582720),
    ("vllm-sr/d3-lite", 852985920, 853247040),
]


def case():
    return {
        "decision": {
            "state": "The sky is blue",
            "question": "What fits?",
            "candidates": [
                {"id": "rain", "text": "Rain"},
                {"id": "clear", "text": "Clear"},
            ],
        }
    }


def tensor(elements, dtype):
    return SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        dtype=dtype,
        shape=(elements,),
        numel=lambda: elements,
    )


@contextlib.contextmanager
def publisher(stored):
    prepared = SimpleNamespace(
        errors=[], sequences=[[1, 2]], texts=["native prompt"], lengths=[2]
    )
    probabilities = {"choice": {"clear": 0.75, "rain": 0.25}}
    tokens = object()
    native = {"choice": "clear", "probabilities": probabilities["choice"]}
    runtime = SimpleNamespace(
        readout=tensor(261120, "float32"),
        prepare=Mock(return_value=prepared),
        run=Mock(return_value=(probabilities, tokens)),
        respond=Mock(return_value={"answers": {"choice": native}}),
    )
    weight = tensor(stored, "bfloat16")
    buffer = tensor(32, "float32")
    model = Mock()
    model.runtime = runtime
    model.to.return_value = model
    model.eval.return_value = model
    model.named_parameters.return_value = [("weight", weight)]
    model.named_buffers.return_value = [("buffer", buffer)]
    auto = SimpleNamespace(from_pretrained=Mock(return_value=model))
    tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=object()))
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        bfloat16="bfloat16",
        float32="float32",
        set_num_threads=Mock(),
        get_num_interop_threads=lambda: 1,
        inference_mode=contextlib.nullcontext,
    )
    with patch.dict(
        sys.modules,
        {
            "torch": torch,
            "transformers": SimpleNamespace(AutoModel=auto, AutoTokenizer=tokenizer),
        },
    ):
        yield SimpleNamespace(
            model=model,
            runtime=runtime,
            prepared=prepared,
            probabilities=probabilities,
            tokens=tokens,
            native=native,
            weight=weight,
            buffer=buffer,
            auto=auto,
            tokenizer=tokenizer,
            torch=torch,
        )


class D3Tests(unittest.TestCase):
    def test_native_construction_and_complete_audit_for_both_pins(self):
        for name, stored, loaded in MODELS:
            with self.subTest(model=name), publisher(stored) as source:
                value = profile_for(name)
                self.assertEqual(value["model_card"]["parameter_count"], stored)
                self.assertEqual(value["precision"], "bfloat16")
                runtime = load_decision(value, Path("/local/model"), 4)
                source.auto.from_pretrained.assert_called_once_with(
                    "/local/model",
                    local_files_only=True,
                    trust_remote_code=True,
                    device="cpu",
                    dtype="bfloat16",
                    verify="full",
                )
                source.tokenizer.from_pretrained.assert_called_once_with(
                    "/local/model", local_files_only=True
                )
                source.model.to.assert_called_once_with("cpu")
                source.model.eval.assert_called_once_with()
                source.model.float.assert_not_called()
                self.assertEqual(runtime.audit["parameter_count"], loaded)
                self.assertEqual(
                    [row["elements"] for row in runtime.audit["parameters"]],
                    [stored, 261120],
                )
                self.assertEqual(runtime.audit["parameters"][1]["dtype"], "float32")
                self.assertEqual(runtime.audit["buffers"][0]["elements"], 32)
                self.assertEqual(runtime.configuration["inputs"], ["text", "image"])

    def test_exact_native_pipeline_preserves_order_probabilities_and_image_digest(self):
        with publisher(MODELS[0][1]) as source:
            runtime = load_decision(profile_for(MODELS[0][0]), Path("/local/model"), 4)
            text = runtime.invoke(case())["decision"]
            source.runtime.run.assert_called_once_with(source.prepared)
            source.runtime.respond.assert_called_once_with(
                source.prepared, source.probabilities, source.tokens
            )
            source.runtime.prepare.assert_called_once_with(
                "The sky is blue",
                {
                    "choice": {
                        "type": "choice",
                        "instructions": "What fits?",
                        "criteria": {"rain": "Rain", "clear": "Clear"},
                    }
                },
                images=None,
            )
            self.assertEqual(
                text["candidates"],
                [
                    {"id": "rain", "probability": 0.25},
                    {"id": "clear", "probability": 0.75},
                ],
            )
            self.assertIs(text["provenance"]["native"], source.native)
            self.assertEqual(
                text["provenance"]["input_digest"],
                fingerprint(
                    {
                        "sequences": [[1, 2]],
                        "texts": ["native prompt"],
                        "lengths": [2],
                        "media": None,
                    }
                ),
            )
            request = case()
            request["media"] = "data:image/png;base64,AAAA"
            image = runtime.invoke(request)["decision"]
            self.assertEqual(
                source.runtime.prepare.call_args.kwargs["images"], [request["media"]]
            )
            self.assertNotEqual(
                text["provenance"]["input_digest"], image["provenance"]["input_digest"]
            )
            request["media"] = "data:image/jpeg;base64,BBBB"
            self.assertNotEqual(
                image["provenance"]["input_digest"],
                runtime.invoke(request)["decision"]["provenance"]["input_digest"],
            )
            source.prepared.errors = ["input too long"]
            source.runtime.run.reset_mock()
            source.runtime.respond.reset_mock()
            with self.assertRaisesRegex(ValueError, "native input preparation failed"):
                runtime.invoke(case())
            source.runtime.run.assert_not_called()
            source.runtime.respond.assert_not_called()

    def test_readout_precision_and_full_stack_audit_reject_invalid_observed_tensors(
        self,
    ):
        for mutation in (
            "readout-bf16",
            "readout-cuda",
            "weight-f16",
            "weight-cuda",
            "buffer-f16",
            "billion",
        ):
            with self.subTest(mutation=mutation), publisher(MODELS[0][1]) as source:
                if mutation == "readout-bf16":
                    source.runtime.readout.dtype = "bfloat16"
                elif mutation == "readout-cuda":
                    source.runtime.readout.device.type = "cuda"
                elif mutation == "weight-f16":
                    source.weight.dtype = "float16"
                elif mutation == "weight-cuda":
                    source.weight.device.type = "cuda"
                elif mutation == "buffer-f16":
                    source.buffer.dtype = "float16"
                    source.buffer.is_floating_point = lambda: True
                else:
                    source.weight.numel = lambda: 1_000_000_000 - 261120
                with self.assertRaisesRegex(
                    ValueError, "readout|unsupported|sub-billion"
                ):
                    load_decision(profile_for(MODELS[0][0]), Path("/local/model"), 4)
        with publisher(MODELS[0][1]):
            value = profile_for(MODELS[0][0])
            value["precision"] = "float32"
            with self.assertRaisesRegex(ValueError, "bfloat16"):
                load_decision(value, Path("/local/model"), 4)

    def test_source_gate_precedes_runtime_and_metadata_fingerprints_native_identity(
        self,
    ):
        for name, stored, loaded in MODELS:
            with (
                self.subTest(model=name),
                tempfile.TemporaryDirectory() as directory,
                publisher(stored) as source,
            ):
                value = profile_for(name)
                with (
                    patch("qit_adapters.backend.verify", return_value={}),
                    patch(
                        "qit_adapters.backend.locked_environment",
                        return_value={"packages": {}},
                    ),
                    patch("qit_adapters.backend.forbid_external_connections"),
                ):
                    with self.assertRaisesRegex(ValueError, "allow-reviewed-code"):
                        Backend(value, directory)
                    source.auto.from_pretrained.assert_not_called()
                    backend = Backend(value, directory, allow_reviewed_code=True)
                    configuration = backend.metadata()["configuration"]
                    self.assertEqual(configuration["parameter_count"], loaded)
                    self.assertEqual(configuration["dtype"], "bfloat16")
                    self.assertEqual(
                        configuration["reviewed_source_revision"], value["revision"]
                    )
                    self.assertEqual(
                        configuration["precision_admission"],
                        "d3-native-bf16-backbone-fp32-readout-v1",
                    )
                    identity = {
                        key: item
                        for key, item in configuration.items()
                        if key != "configuration_id"
                    }
                    self.assertEqual(
                        configuration["configuration_id"], fingerprint(identity)
                    )
                    del identity["precision_admission"]
                    self.assertNotEqual(
                        configuration["configuration_id"], fingerprint(identity)
                    )
                    request = {
                        "model": name,
                        "task": "decision_scoring",
                        "case": case(),
                    }
                    backend.validate_request(request)
                    request["case"]["media"] = "data:image/png;base64,AAAA"
                    backend.validate_request(request)
                    request["case"]["media"] = "https://example.invalid/image.png"
                    with self.assertRaisesRegex(ValueError, "inline PNG or JPEG"):
                        backend.validate_request(request)
        value = profile_for(MODELS[0][0])
        loader = Mock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(ADAPTERS, {"d3": Adapter(loader, reviewed_code=True)}),
        ):
            with self.assertRaisesRegex(ValueError, "model not installed"):
                Backend(value, directory, allow_reviewed_code=True)
            loader.assert_not_called()

    def test_modified_publisher_source_is_refused_before_import(self):
        value = profile_for(MODELS[0][0])
        source = b"publisher source"
        value["files"] = {
            "modeling_d3.py": {
                "bytes": len(source),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            publisher(MODELS[0][1]) as native,
        ):
            path = Path(directory) / "modeling_d3.py"
            path.write_bytes(b"modified source!")
            with self.assertRaisesRegex(ValueError, "size|digest"):
                Backend(value, directory, allow_reviewed_code=True)
            native.auto.from_pretrained.assert_not_called()
