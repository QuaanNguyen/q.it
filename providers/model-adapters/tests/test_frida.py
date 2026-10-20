import builtins
import contextlib
import copy
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import locked_environment, profile_for, profiles
from qit_adapters.backend import Backend
from qit_adapters.decisions import DecisionRuntime, load_decision
from qit_adapters.registry import ADAPTERS, Adapter
from qit_adapters.server import CpuServer


PACKAGE = Path(__file__).resolve().parents[1]


def profile():
    return profile_for("ai-forever/FRIDA-Decisions")


def case():
    return {
        "decision": {
            "state": "Сейчас идёт дождь",
            "question": "Что взять?",
            "candidates": [
                {"id": "umbrella", "text": "Зонт"},
                {"id": "glasses", "text": "Очки"},
            ],
        }
    }


def native_runtime():
    runtime = DecisionRuntime.__new__(DecisionRuntime)
    runtime.profile = profile()
    runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    runtime.configuration = {
        "calibration": "native-frida",
        "selection_rule": "publisher-native-argmax",
        "preprocessing": "native-frida",
    }
    answer = {
        "choice": "umbrella",
        "probabilities": {"glasses": 0.125, "umbrella": 0.875},
    }
    requests = []

    def compile_request(request):
        requests.append(copy.deepcopy(request))
        question = request["questions"]["choice"]
        candidates = [
            SimpleNamespace(
                state=request["state"], instruction=question["instructions"], text=text
            )
            for text in question["criteria"].values()
        ]
        tokenized = SimpleNamespace(
            state=request["state"],
            questions=question["instructions"],
            options=list(question["criteria"].values()),
        )
        return None, candidates, tokenized

    runtime.engine = SimpleNamespace(
        compile=Mock(side_effect=compile_request),
        judge=Mock(return_value={"usage": {}, "answers": {"choice": answer}}),
        text=SimpleNamespace(count=lambda text: len(text.split())),
        config=SimpleNamespace(instruction_max_tokens=4, option_max_tokens=4),
    )
    return runtime, requests


@contextlib.contextmanager
def installed_source():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "frida_decisions"
        root.mkdir()
        content = b"raise RuntimeError('publisher import sentinel')\n"
        (root / "__init__.py").write_bytes(content)
        value = profile()
        value["files"] = {}
        value["requirements"] = {"frida-decisions": "0.1.0"}
        value["runtime_source"]["python_files"] = {
            "__init__.py": hashlib.sha256(content).hexdigest()
        }
        source = {
            "url": f"https://github.com/ai-forever/FRIDA-Decisions/archive/{value['runtime_source']['commit']}.tar.gz",
            "archive_info": {
                "hashes": {"sha256": value["runtime_source"]["archive_sha256"]}
            },
        }
        distribution = SimpleNamespace(
            locate_file=lambda name: Path(directory) / name,
            read_text=lambda name: json.dumps(source)
            if name == "direct_url.json"
            else "record",
        )
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch("platform.libc_ver", return_value=("glibc", "2.28")),
            patch("sys.version_info", (3, 12)),
            patch(
                "qit_adapters.artifacts.environment",
                side_effect=lambda required: {
                    name: version for name, version in required.items()
                },
            ),
            patch("importlib.metadata.distribution", return_value=distribution),
        ):
            yield value, root, source


class FridaTests(unittest.TestCase):
    def test_catalog_and_archive_pins_retain_one_new_route(self):
        self.assertEqual(
            set(ADAPTERS),
            {"gliclass", "gliner2", "intern", "liquid", "typed-gemma", "frida"},
        )
        self.assertEqual(len(profiles()), 6)
        self.assertFalse(ADAPTERS["frida"].reviewed_code)
        value = profile()
        self.assertEqual(len(value["files"]), 7)
        self.assertEqual(value["model_card"]["parameter_count"], 823401216)
        source = value["runtime_source"]
        pin = f"https://github.com/ai-forever/FRIDA-Decisions/archive/{source['commit']}.tar.gz#sha256={source['archive_sha256']}"
        for path in [
            PACKAGE / "pyproject.toml",
            *sorted((PACKAGE / "qit_adapters/locks").glob("frida-*.txt")),
        ]:
            self.assertIn(pin, path.read_text())
        self.assertEqual(
            len(list((PACKAGE / "qit_adapters/locks").glob("frida-*.txt"))), 2
        )

    def test_native_request_probabilities_and_compiled_digest(self):
        runtime, requests = native_runtime()
        observed = runtime.invoke(case())["decision"]
        self.assertEqual(
            list(requests[0]["questions"]["choice"]["criteria"]),
            ["umbrella", "glasses"],
        )
        self.assertEqual(
            observed["candidates"],
            [
                {"id": "umbrella", "probability": 0.875},
                {"id": "glasses", "probability": 0.125},
            ],
        )
        self.assertEqual(observed["selected_id"], "umbrella")
        self.assertEqual(
            observed["provenance"]["native"],
            runtime.engine.judge.return_value["answers"]["choice"],
        )
        self.assertEqual(runtime.engine.judge.call_args.args[0], requests[0])
        for field in ("state", "question", "candidates"):
            changed = case()
            if field == "candidates":
                changed["decision"][field][0]["text"] += " новый"
            else:
                changed["decision"][field] += " новый"
            self.assertNotEqual(
                runtime.invoke(changed)["decision"]["provenance"]["input_digest"],
                observed["provenance"]["input_digest"],
            )

    def test_each_native_limit_refuses_before_judging(self):
        for field, length in (("state", 513), ("question", 5), ("candidates", 4)):
            runtime, _ = native_runtime()
            changed = case()
            if field == "candidates":
                changed["decision"][field][0]["text"] = "слово " * length
            else:
                changed["decision"][field] = "слово " * length
            with self.assertRaisesRegex(ValueError, "would be truncated"):
                runtime.invoke(changed)
            runtime.engine.judge.assert_not_called()
        runtime, _ = native_runtime()
        changed = case()
        changed["decision"]["state"] = "English only"
        with self.assertRaisesRegex(ValueError, "Russian decision pack"):
            runtime.invoke(changed)
        runtime.engine.compile.assert_not_called()
        runtime.engine.judge.return_value["usage"]["state_truncated"] = True
        with self.assertRaisesRegex(ValueError, "reported a truncated state"):
            runtime.invoke(case())

    def test_constructor_audits_registered_encoder_and_scalar_head(self):
        def tensor(count):
            return SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                dtype="float32",
                shape=(count,),
                numel=lambda: count,
            )

        encoder, head = tensor(823401216), tensor(1537)
        model = SimpleNamespace(
            named_parameters=lambda: [
                ("encoder.weight", encoder),
                ("head.weight", head),
            ],
            named_buffers=lambda: [],
        )
        judge = Mock(return_value=SimpleNamespace(model=model))
        tokenizer = Mock()
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            float32="float32",
            set_num_threads=Mock(),
            get_num_interop_threads=lambda: 1,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "transformers": SimpleNamespace(
                        AutoModel=Mock(),
                        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
                    ),
                    "frida_decisions": SimpleNamespace(
                        Judge=SimpleNamespace(from_pretrained=judge)
                    ),
                },
            ),
        ):
            runtime = load_decision(profile(), Path(directory).resolve(), 4)
            judge.assert_called_once_with(
                str(Path(directory).resolve()),
                device="cpu",
                dtype="float32",
                state_max=512,
                state_cache_mb=0,
            )
            self.assertIs(runtime.model, model)
            self.assertEqual(runtime.audit["parameter_count"], 823402753)
            self.assertEqual(
                [item["name"] for item in runtime.audit["parameters"]],
                ["model.encoder.weight", "model.head.weight"],
            )
            head.numel = lambda: 176598784
            with self.assertRaisesRegex(ValueError, "sub-billion limit"):
                load_decision(profile(), Path(directory).resolve(), 4)
            head.dtype = "bfloat16"
            with self.assertRaisesRegex(ValueError, "unsupported device or precision"):
                load_decision(profile(), Path(directory).resolve(), 4)

    def test_verified_runtime_identity_is_in_existing_environment_metadata(self):
        with installed_source() as (value, _, source):
            environment = locked_environment(value)
            self.assertEqual(
                environment["source_verification"]["commit"],
                value["runtime_source"]["commit"],
            )
            self.assertEqual(
                environment["source_verification"]["installed_python_sha256"],
                value["runtime_source"]["python_files"],
            )
            source["archive_info"] = {}
            self.assertEqual(
                locked_environment(value)["source_verification"],
                environment["source_verification"],
            )

    def test_source_tampering_refuses_startup_before_publisher_import(self):
        original_import = builtins.__import__
        for mutation in (
            "modified",
            "missing",
            "added",
            "symlink",
            "root-symlink",
            "directory-symlink",
            "url",
            "hash",
            "legacy-hash",
        ):
            with (
                self.subTest(mutation=mutation),
                installed_source() as (value, root, source),
            ):
                path = root / "__init__.py"
                if mutation == "modified":
                    path.write_text("modified")
                elif mutation == "missing":
                    path.unlink()
                elif mutation == "added":
                    (root / "extra.py").write_text("extra")
                elif mutation == "symlink":
                    outside = root.parent / "outside.py"
                    outside.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(outside)
                elif mutation == "directory-symlink":
                    outside = root.parent / "outside"
                    outside.mkdir()
                    (outside / "injected.py").write_text("injected")
                    (root / "nested").symlink_to(outside, target_is_directory=True)
                elif mutation == "root-symlink":
                    moved = root.parent / "moved"
                    root.rename(moved)
                    root.symlink_to(moved, target_is_directory=True)
                elif mutation == "url":
                    source["url"] = "https://example.com/unreviewed.tar.gz"
                elif mutation == "hash":
                    source["archive_info"]["hashes"]["sha256"] = "0" * 64
                else:
                    source["archive_info"]["hash"] = "sha256=" + "0" * 64
                imported = []

                def import_sentinel(name, *args, **kwargs):
                    if name == "frida_decisions":
                        imported.append(name)
                        raise AssertionError("publisher import reached")
                    return original_import(name, *args, **kwargs)

                loader = Mock(side_effect=lambda *args: __import__("frida_decisions"))
                with (
                    patch.dict(ADAPTERS, {"frida": Adapter(loader)}),
                    patch("builtins.__import__", side_effect=import_sentinel),
                ):
                    with self.assertRaisesRegex(ValueError, "FRIDA"):
                        Backend(value, root.parent)
                loader.assert_not_called()
                self.assertEqual(imported, [])

    def test_native_fixture_serves_existing_http_contract_and_metadata(self):
        runtime, requests = native_runtime()
        runtime.audit = {"parameter_count": 823402753, "parameters": [], "buffers": []}
        runtime.configuration["inputs"] = ["text"]
        with (
            installed_source() as (value, root, _),
            patch.dict(ADAPTERS, {"frida": Adapter(lambda *args: runtime)}),
            patch("qit_adapters.backend.forbid_external_connections"),
        ):
            loaded = Backend(value, root.parent)
            configuration = loaded.metadata()["configuration"]
            self.assertIsNone(configuration["reviewed_source_revision"])
            self.assertEqual(
                configuration["environment"]["source_verification"]["commit"],
                value["runtime_source"]["commit"],
            )
            self.assertEqual(configuration["dtype"], "float32")
            request = {
                "model": value["model"],
                "task": "decision_scoring",
                "case": case(),
            }
            with CpuServer(("127.0.0.1", 0), loaded, "secret") as server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=5
                )
                try:
                    connection.request(
                        "POST",
                        "/v1/qit/invoke",
                        json.dumps(request),
                        {"Authorization": "Bearer secret"},
                    )
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(
                        response.getheader("Content-Type"), "application/x-ndjson"
                    )
                    self.assertEqual(
                        json.loads(response.read())["decision"]["selected_id"],
                        "umbrella",
                    )
                    self.assertNotIn("expected", json.dumps(requests))
                finally:
                    connection.close()
                    server.shutdown()
                    worker.join()
            request["case"]["media"] = "data:image/png;base64,AAAA"
            with self.assertRaisesRegex(ValueError, "image-backed execution"):
                loaded.validate_request(request)

    def test_original_russian_pack_contains_three_text_cases(self):
        pack = PACKAGE / "packs/russian-choice"
        records = [
            json.loads(line) for line in (pack / "cases.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [record["id"] for record in records], ["weather", "temperature", "reading"]
        )
        for record in records:
            self.assertTrue(
                any(
                    "\u0400" <= letter <= "\u04ff"
                    for letter in record["decision"]["state"]
                )
            )
            self.assertNotIn("media", record)
