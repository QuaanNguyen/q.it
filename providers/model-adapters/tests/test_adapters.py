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

from qit_adapters.artifacts import (
    environment,
    fingerprint,
    profile_for,
    profiles,
    verify,
)
from qit_adapters.backend import Backend, audit_modules
from qit_adapters.registry import ADAPTERS, Adapter
from qit_adapters.server import CpuServer


def profile():
    return {
        "model": "fixture/model",
        "adapter": "fixture",
        "revision": "a" * 40,
        "task": "decision_scoring",
        "precision": "float32",
        "license": "MIT",
        "requirements": {},
        "files": {},
    }


def payload():
    return {
        "model": "fixture/model",
        "task": "decision_scoring",
        "case": {
            "id": "one",
            "decision": {
                "state": "The sky is clear",
                "question": "What fits?",
                "candidates": [
                    {"id": "clear", "text": "Clear"},
                    {"id": "rain", "text": "Rain"},
                ],
            },
        },
    }


class FixtureRuntime:
    audit = {"parameter_count": 10, "parameters": [], "buffers": []}
    configuration = {
        "schema_version": 1,
        "inputs": ["text"],
        "calibration": "native",
        "selection_rule": "first-max",
        "preprocessing": "fixture",
    }

    def invoke(self, case):
        return {
            "type": "result",
            "decision": {
                "schema_version": 1,
                "candidates": [
                    {"id": "clear", "probability": 0.75},
                    {"id": "rain", "probability": 0.25},
                ],
                "selected_id": "clear",
                "provenance": {
                    "calibration": "native",
                    "selection_rule": "first-max",
                    "preprocessing": "fixture",
                    "input_digest": fingerprint(case),
                    "input_truncated": False,
                    "native": {},
                },
            },
        }


@contextlib.contextmanager
def backend(runtime=None, reviewed_code=False, allow_reviewed_code=False):
    runtime = runtime or FixtureRuntime()
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(
            ADAPTERS,
            {"fixture": Adapter(lambda *args: runtime, reviewed_code)},
        ),
        patch("qit_adapters.backend.locked_environment", return_value={"packages": {}}),
        patch("qit_adapters.backend.forbid_external_connections"),
    ):
        yield Backend(profile(), directory, allow_reviewed_code=allow_reviewed_code)


class AdapterTests(unittest.TestCase):
    def test_package_has_no_routes_or_runtime_dependencies(self):
        self.assertEqual(profiles(), [])
        self.assertEqual(ADAPTERS, {})
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            profile_for("fixture/model")

    def test_profile_requires_an_immutable_revision(self):
        value = profile()
        with patch("qit_adapters.artifacts.profiles", return_value=[value]):
            self.assertEqual(profile_for(value["model"]), value)
            value["revision"] = "main"
            with self.assertRaisesRegex(ValueError, "immutable"):
                profile_for(value["model"])

    def test_artifacts_are_verified_before_loading_and_never_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"original publisher content"
            identity = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            value = {**profile(), "files": {"model.bin": identity}}
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                with self.assertRaisesRegex(ValueError, "model not installed"):
                    Backend(value, root)
            path = root / "model.bin"
            path.write_bytes(data)
            self.assertEqual(verify(value, root)["model.bin"], identity)
            path.write_bytes(b"modified publisher content")
            with self.assertRaisesRegex(ValueError, "digest"):
                verify(value, root)
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "size"):
                verify(value, root)
            path.unlink()
            path.symlink_to(root / "other")
            with self.assertRaisesRegex(ValueError, "inside"):
                verify(value, root)
            value["files"] = {"../outside": identity}
            with self.assertRaisesRegex(ValueError, "inside"):
                verify(value, root)

    def test_git_blob_artifact_identity_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            data = b"publisher configuration"
            (Path(directory) / "config.json").write_bytes(data)
            value = {
                "files": {
                    "config.json": {
                        "bytes": len(data),
                        "git_blob": hashlib.sha1(
                            f"blob {len(data)}\0".encode() + data
                        ).hexdigest(),
                    }
                }
            }
            self.assertEqual(
                verify(value, directory)["config.json"]["sha256"],
                hashlib.sha256(data).hexdigest(),
            )

    def test_environment_requires_exact_versions(self):
        with patch("importlib.metadata.version", return_value="2.9.1+cpu"):
            self.assertEqual(environment({"torch": "2.9.1"}), {"torch": "2.9.1+cpu"})
            with self.assertRaisesRegex(ValueError, "unsupported runtime"):
                environment({"torch": "2.9.1+cuda"})

    def test_unknown_adapters_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported adapter"):
                Backend(profile(), directory)

    def test_reviewed_code_requires_explicit_admission(self):
        with self.assertRaisesRegex(ValueError, "allow-reviewed-code"):
            with backend(reviewed_code=True):
                pass
        with backend(reviewed_code=True, allow_reviewed_code=True) as loaded:
            self.assertEqual(
                loaded.metadata()["configuration"]["reviewed_source_revision"], "a" * 40
            )

    def test_backend_retains_configuration_and_rejects_oversized_stacks(self):
        with backend() as loaded:
            metadata = loaded.metadata()
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["configuration"]["parameter_count"], 10)
            self.assertEqual(len(metadata["configuration"]["configuration_id"]), 64)
        for count in (0, -1, True, 1_000_000_000):
            runtime = FixtureRuntime()
            runtime.audit = {"parameter_count": count}
            with self.assertRaisesRegex(ValueError, "fewer than one billion"):
                with backend(runtime):
                    pass

    def test_expected_answers_and_generation_settings_are_refused(self):
        with backend() as loaded:
            loaded.validate_request(payload())
            for extra in ("expected", "expected_index"):
                request = payload()
                request["case"][extra] = "clear"
                with self.assertRaises(ValueError):
                    loaded.validate_request(request)
            request = payload()
            request["max_output_tokens"] = 32
            with self.assertRaises(ValueError):
                loaded.validate_request(request)
            with self.assertRaisesRegex(ValueError, "JSON object"):
                loaded.validate_request([])

    def test_nondecision_routes_require_task_and_bounded_output_limit(self):
        with backend() as loaded:
            loaded.profile = {**profile(), "task": "image_to_text"}
            request = {
                "model": "fixture/model",
                "task": "image_to_text",
                "case": {},
                "max_output_tokens": 1024,
            }
            loaded.validate_request(request)
            for limit in (0, True, 32769, None):
                with self.assertRaises(ValueError):
                    loaded.validate_request({**request, "max_output_tokens": limit})
            with self.assertRaises(ValueError):
                loaded.validate_request({**request, "task": "embedding"})

    def test_loaded_audit_counts_shared_parameters_and_extra_readout_once(self):
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
        parameter.dtype = "float32"
        parameter.device.type = "cuda"
        with self.assertRaises(ValueError):
            audit_modules({"a": module}, {}, torch, "float32")

    def test_offline_startup_blocks_external_connections(self):
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

    def test_server_uses_existing_metadata_and_ndjson_contract(self):
        with (
            backend() as loaded,
            CpuServer(("127.0.0.1", 0), loaded, "secret") as server,
        ):
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:

                def request(method, path, body=None, key="secret", headers=None):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=5
                    )
                    connection.request(
                        method,
                        path,
                        body,
                        headers or {"Authorization": "Bearer " + key},
                    )
                    response = connection.getresponse()
                    result = (
                        response.status,
                        response.getheader("Content-Type"),
                        response.read(),
                    )
                    connection.close()
                    return result

                self.assertEqual(request("GET", "/health", key="wrong")[0], 401)
                status, kind, body = request("GET", "/v1/qit/model")
                self.assertEqual((status, kind), (200, "application/json"))
                self.assertEqual(json.loads(body)["model"], "fixture/model")
                status, kind, body = request(
                    "POST", "/v1/qit/invoke", json.dumps(payload())
                )
                self.assertEqual((status, kind), (200, "application/x-ndjson"))
                self.assertTrue(body.endswith(b"\n"))
                self.assertEqual(json.loads(body)["decision"]["selected_id"], "clear")
                self.assertEqual(request("POST", "/v1/qit/invoke", "[]")[0], 400)
                self.assertEqual(request("POST", "/v1/qit/invoke", "{")[0], 400)
                self.assertEqual(
                    request(
                        "POST",
                        "/v1/qit/invoke",
                        b"",
                        headers={
                            "Authorization": "Bearer secret",
                            "Content-Length": str(64 * 1024 * 1024 + 1),
                        },
                    )[0],
                    400,
                )
                with server.inference:
                    self.assertEqual(
                        request("POST", "/v1/qit/invoke", json.dumps(payload()))[0], 409
                    )
                self.assertEqual(request("GET", "/unknown")[0], 404)
            finally:
                server.shutdown()
                worker.join()

    def test_server_binds_only_to_loopback(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            CpuServer(("0.0.0.0", 0), None)


if __name__ == "__main__":
    unittest.main()
