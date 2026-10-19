import contextlib
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import profile_for, profiles, verify
from qit_adapters.backend import Backend, audit_modules
from qit_adapters.decisions import DecisionRuntime, distribution
from qit_adapters.server import CpuServer


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


class FixtureRuntime:
    def invoke(self, case):
        candidates = case["decision"]["candidates"]
        return {
            "type": "result",
            "decision": {
                "schema_version": 1,
                "candidates": distribution(
                    candidates, {"second": 0.25, "first": 0.75}, "first"
                ),
                "selected_id": "first",
                "provenance": {
                    "calibration": "native",
                    "selection_rule": "first-max",
                    "preprocessing": "fixture",
                    "input_digest": "a" * 64,
                    "input_truncated": False,
                    "native": {"scores": [0.25, 0.75]},
                },
            },
        }


class AdapterTests(unittest.TestCase):
    def test_offline_startup_blocks_external_network_before_any_connection(self):
        command = subprocess.run(
            [
                sys.executable,
                "-c",
                "from qit_adapters.offline import forbid_external_connections; import socket; forbid_external_connections(); socket.create_connection(('203.0.113.1',80),timeout=1)",
            ],
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(command.returncode, 0)
        self.assertIn("offline serving forbids external connections", command.stderr)

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
                return "choice", {"labels": ["second", "first"]}, Values()

            def _classification_schema(self, tasks):
                return tasks

            def _build_schema_dicts_and_metadata(self, schemas):
                return schemas, []

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

    def test_expected_answers_and_generation_settings_are_refused(self):
        backend = Backend.__new__(Backend)
        backend.profile = {
            "model": "fixture",
            "adapter": "gliclass",
            "task": "decision_scoring",
        }
        backend.validate_request(payload())
        for extra in ["expected", "expected_index"]:
            request = payload()
            request["case"][extra] = "first"
            with self.assertRaises(ValueError):
                backend.validate_request(request)
        request = payload()
        request["max_output_tokens"] = 32
        with self.assertRaises(ValueError):
            backend.validate_request(request)

    def test_nondecision_routes_validate_their_actual_task_and_output_limit(self):
        backend = Backend.__new__(Backend)
        backend.profile = {
            "model": "ocr",
            "adapter": "lighton",
            "task": "image_to_text",
        }
        backend.validate_request(
            {
                "model": "ocr",
                "task": "image_to_text",
                "case": {"prompt": "<OCR>"},
                "max_output_tokens": 1024,
            }
        )
        with self.assertRaises(ValueError):
            backend.validate_request(
                {
                    "model": "ocr",
                    "task": "decision_scoring",
                    "case": {},
                    "max_output_tokens": 10,
                }
            )

    def test_artifacts_are_verified_before_runtime_import_and_never_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"original pinned publisher content"
            identity = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            profile = {"files": {"model.safetensors": identity}, "adapter": "gliclass"}
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                with self.assertRaisesRegex(ValueError, "model not installed"):
                    Backend(profile, directory)
            path = root / "model.safetensors"
            path.write_bytes(data)
            self.assertEqual(
                verify(profile, root)["model.safetensors"]["sha256"], identity["sha256"]
            )
            path.write_bytes(b"modified pinned publisher content")
            with self.assertRaises(ValueError):
                verify(profile, root)
            path.unlink()
            path.symlink_to(root / "other")
            with self.assertRaises(ValueError):
                verify(profile, root)

    def test_loaded_audit_counts_shared_parameters_once_and_includes_unregistered_readout(
        self,
    ):
        def tensor(elements, dtype="float32"):
            return SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                dtype=dtype,
                shape=(elements,),
                numel=lambda: elements,
            )

        parameter = tensor(10)
        readout = tensor(3)
        fixed = tensor(100)
        module = SimpleNamespace(
            named_parameters=lambda: [("weight", parameter)],
            named_buffers=lambda: [("fixed", fixed)],
        )
        torch = SimpleNamespace(float32="float32", bfloat16="bfloat16")
        audit = audit_modules(
            {"a": module, "b": module}, {"readout": readout}, torch, "float32"
        )
        self.assertEqual(audit["parameter_count"], 13)
        self.assertEqual(audit["buffers"][0]["elements"], 100)
        parameter.dtype = "bfloat16"
        with self.assertRaises(ValueError):
            audit_modules({"a": module}, {}, torch, "float32")
        self.assertEqual(
            audit_modules({"a": module}, {"readout": readout}, torch, "bfloat16")[
                "parameter_count"
            ],
            13,
        )

    def test_all_candidates_have_immutable_original_artifact_identities_and_honest_status(
        self,
    ):
        catalog = profiles()
        self.assertEqual(len(catalog), 17)
        self.assertEqual(len({profile["model"] for profile in catalog}), len(catalog))
        for profile in catalog:
            self.assertEqual(profile_for(profile["model"]), profile)
            self.assertEqual(len(profile["revision"]), 40)
            self.assertEqual(profile["qualification"], "unverified")
            self.assertIn(profile["implementation"], ("experimental", "pending"))
            self.assertTrue(profile["files"])
            for identity in profile["files"].values():
                self.assertTrue(identity.get("sha256") or identity.get("git_blob"))
        command = subprocess.run(
            [
                sys.executable,
                "-m",
                "qit_adapters.cli",
                "serve",
                "--model",
                catalog[0]["model"],
                "--model-path",
                "/nonexistent/qit-model",
            ],
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True,
            text=True,
        )
        self.assertEqual(command.returncode, 1)
        self.assertIn("model not installed", command.stderr)

    def test_optional_server_uses_the_existing_wire_contract(self):
        backend = Backend.__new__(Backend)
        backend.profile = {
            "model": "fixture",
            "adapter": "gliclass",
            "task": "decision_scoring",
        }
        backend.runtime = FixtureRuntime()
        backend.configuration = {
            "device": "cpu",
            "dtype": "float32",
            "task": "decision_scoring",
            "parameter_count": 10,
            "revision": "fixture",
        }
        with CpuServer(("127.0.0.1", 0), backend) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("POST", "/v1/qit/invoke", json.dumps(payload()))
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            result = json.loads(response.read())
            self.assertEqual(result["decision"]["candidates"][1]["probability"], 0.75)
            self.assertNotIn("output_tokens", result)
            connection.close()
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    unittest.main()
