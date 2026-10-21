import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_adapters.artifacts import lock_family
from qit_adapters.nemotron import archive_inventory
from qit_adapters.registry import ADAPTERS
from qit_adapters.speech import SpeechRuntime, load_speech

try:
    import numpy
    import soundfile
except ImportError:
    numpy = None


class FakeTensor:
    def __init__(self, values, dtype, device="cpu", mode="gemm8", panels=8):
        self.values = values
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.mode = mode
        self._dense = None
        self._panels = SimpleNamespace(nbytes=panels)
        self._scales32 = self


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float32 = "torch.float32"
        self.version = SimpleNamespace(cuda=None)

    def set_num_threads(self, value):
        pass

    def get_num_interop_threads(self):
        return 1

    def equal(self, left, right):
        return left is right


class TernaryLinear:
    def __init__(self, weight, out_features=3, in_features=2, group_size=2):
        self.weight = weight
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.scales = weight

    def named_parameters(self):
        return []

    def named_buffers(self):
        return []


def kestrel_modules(model):
    package = ModuleType("kestrel")
    models = ModuleType("kestrel.models")
    parakeet = ModuleType("kestrel.models.parakeet_tdt")
    weights = ModuleType("kestrel.models.parakeet_tdt.weights")
    features = ModuleType("kestrel.models.parakeet_tdt.features")
    weights.load_parakeet_tdt = lambda *args, **options: SimpleNamespace(
        model=model, tokenizer=object()
    )
    features.parakeet_features = lambda samples: (object(), object())
    package.models = models
    models.parakeet_tdt = parakeet
    parakeet.weights = weights
    parakeet.features = features
    return {
        "kestrel": package,
        "kestrel.models": models,
        "kestrel.models.parakeet_tdt": parakeet,
        "kestrel.models.parakeet_tdt.weights": weights,
        "kestrel.models.parakeet_tdt.features": features,
    }


class ReduxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.torch = FakeTorch()

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self, manifest, weight):
        (self.directory / "ternary.json").write_text(
            json.dumps({"quantized_modules": manifest})
        )
        model = SimpleNamespace(
            named_modules=lambda: [("encoder.linear", TernaryLinear(weight))]
        )
        with patch.dict(sys.modules, kestrel_modules(model)):
            return SpeechRuntime(
                {"adapter": "redux", "precision": "native_ternary"},
                self.directory,
                self.torch,
            )

    def test_redux_reconciles_logical_weights_with_the_complete_manifest(self):
        weight = FakeTensor([1.0], "torch.float32")
        runtime = self.runtime([{"in_features": 2, "out_features": 3}], weight)
        self.assertEqual(runtime.logical_parameters[0]["resident_format"], "gemm8")
        self.assertEqual(
            runtime.preparation["weight_format"], "publisher native ternary"
        )

    def test_redux_refuses_a_mismatched_resident_format_or_manifest(self):
        with self.assertRaisesRegex(ValueError, "native CPU GEMM8 panels"):
            self.runtime(
                [{"in_features": 2, "out_features": 3}],
                FakeTensor([1.0], "torch.float32", mode="dense"),
            )
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            self.runtime(
                [{"in_features": 10, "out_features": 10}],
                FakeTensor([1.0], "torch.float32"),
            )

    def test_float32_route_refuses_a_ternary_checkpoint(self):
        model = SimpleNamespace(is_ternary=True)
        with patch.dict(sys.modules, kestrel_modules(model)):
            with self.assertRaisesRegex(ValueError, "must not load a ternary"):
                SpeechRuntime({"adapter": "photon"}, self.directory, self.torch)

    def test_lock_family_and_registry_route_both_speech_runtimes(self):
        self.assertEqual(lock_family("redux"), "photon")
        self.assertIs(ADAPTERS["photon"].load, load_speech)
        self.assertIs(ADAPTERS["redux"].load, load_speech)
        self.assertFalse(ADAPTERS["nemotron"].reviewed_code)


class SpeechAuditTests(unittest.TestCase):
    def test_logical_weights_join_the_admitted_parameter_count(self):
        runtime = SimpleNamespace(
            model=object(), extra_tensors={}, logical_parameters=[{"elements": 5}]
        )
        with (
            patch.dict(sys.modules, {"torch": FakeTorch()}),
            patch("qit_adapters.speech.SpeechRuntime", return_value=runtime),
            patch(
                "qit_adapters.backend.audit_modules",
                return_value={"parameter_count": 10},
            ),
        ):
            admitted = load_speech(
                {"adapter": "redux", "precision": "native_ternary"}, Path("/models"), 2
            )
        self.assertEqual(admitted.audit["parameter_count"], 15)
        self.assertEqual(admitted.audit["logical_quantized_weights"], [{"elements": 5}])

    def test_a_logical_stack_over_the_limit_is_refused(self):
        runtime = SimpleNamespace(
            model=object(),
            extra_tensors={},
            logical_parameters=[{"elements": 2_000_000_000}],
        )
        with (
            patch.dict(sys.modules, {"torch": FakeTorch()}),
            patch("qit_adapters.speech.SpeechRuntime", return_value=runtime),
            patch(
                "qit_adapters.backend.audit_modules",
                return_value={"parameter_count": 10},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "under-one-billion"):
                load_speech(
                    {"adapter": "redux", "precision": "native_ternary"},
                    Path("/models"),
                    2,
                )


@unittest.skipIf(numpy is None, "install the optional audio test dependencies")
class AudioTests(unittest.TestCase):
    def wav(self, seconds, rate=16000):
        import io

        import numpy

        buffer = io.BytesIO()
        soundfile.write(
            buffer,
            numpy.zeros(int(seconds * rate), dtype="float32"),
            rate,
            format="WAV",
        )
        import base64

        return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()

    def test_audio_is_resampled_and_bounded(self):
        from qit_adapters.media import audio_from_case

        samples = audio_from_case({"media": self.wav(0.01, 8000)}, 16000)
        self.assertEqual(samples.dtype, numpy.float32)
        self.assertEqual(len(samples), 160)
        with self.assertRaisesRegex(ValueError, "thirty seconds"):
            audio_from_case({"media": self.wav(31, 8000)}, 16000)
        with self.assertRaisesRegex(ValueError, "inline audio/ data URL"):
            audio_from_case({"media": "data:image/png;base64,AAAA"}, 16000)


class ArchiveTests(unittest.TestCase):
    def archive(self, members):
        path = Path(self.temporary.name) / "model.nemo"
        with tarfile.open(path, "w") as archive:
            for name, content in members:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.type = tarfile.REGTYPE
                archive.addfile(info, __import__("io").BytesIO(content))
        return path

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary.cleanup()

    def test_inventory_records_regular_member_checksums(self):
        path = self.archive([("weights.bin", b"abc")])
        inventory = archive_inventory(path)
        self.assertEqual(inventory[0]["bytes"], 3)
        self.assertEqual(
            inventory[0]["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_unsafe_duplicate_and_nonregular_members_are_refused(self):
        for members, expected in [
            ([("../escape", b"x")], "unsafe or duplicate"),
            ([("a", b"x"), ("a", b"y")], "unsafe or duplicate"),
        ]:
            with self.assertRaisesRegex(ValueError, expected):
                archive_inventory(self.archive(members))
        path = Path(self.temporary.name) / "link.nemo"
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
        with self.assertRaisesRegex(ValueError, "nonregular"):
            archive_inventory(path)


if __name__ == "__main__":
    unittest.main()
