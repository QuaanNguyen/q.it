import base64
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from qit_adapters.artifacts import profile_for
from qit_adapters.backend import Backend
from qit_adapters.media import image_from_case, media_bytes
from qit_adapters.models import ImageRuntime, load_image
from qit_adapters.registry import ADAPTERS, Adapter


def png(width=8, height=8):
    output = io.BytesIO()
    Image.new("RGB", (width, height), (12, 34, 56)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def case(prompt="<OCR>"):
    return {"id": "one", "prompt": prompt, "media": png()}


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def tolist(self):
        return self.values

    def __getitem__(self, key):
        row, part = key
        assert row == 0
        return FakeTensor(self.values[part])

    def __len__(self):
        return len(self.values)


class FakeProcessor:
    def apply_chat_template(self, messages, **options):
        assert messages[0]["content"][0]["type"] == "image"
        assert options["enable_thinking"] is False
        return FakeInputs()

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "recognized page text"


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeTensor([1, 2, 3, 4, 5]))

    def to(self, device):
        assert device == "cpu"
        return self


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **options):
        assert options["do_sample"] is False
        assert options["max_new_tokens"] == 32
        return FakeTensor([1, 2, 3, 4, 5, 6, 7])

    def named_parameters(self):
        return [
            (
                "weight",
                SimpleNamespace(
                    device=SimpleNamespace(type="cpu"),
                    dtype="torch.float32",
                    shape=[4],
                    numel=lambda: 4,
                ),
            )
        ]

    def named_buffers(self):
        return []


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float32 = "torch.float32"
        self.version = SimpleNamespace(cuda=None)

    def set_num_threads(self, value):
        pass

    def get_num_interop_threads(self):
        return 1

    def inference_mode(self):
        from contextlib import nullcontext

        return nullcontext()


def modules():
    torch = FakeTorch()
    transformers = ModuleType("transformers")
    model = FakeModel()
    processor = FakeProcessor()

    class Qwen3_5ForConditionalGeneration:
        @staticmethod
        def from_pretrained(directory, **options):
            assert options["local_files_only"] is True
            assert options["attn_implementation"] == "eager"
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(directory, **options):
            assert options["local_files_only"] is True
            return processor

    transformers.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration
    transformers.AutoProcessor = AutoProcessor
    return {"torch": torch, "transformers": transformers}


class MediaTests(unittest.TestCase):
    def test_inline_media_rejects_wrong_prefix_and_unbounded_encoding(self):
        with self.assertRaisesRegex(ValueError, "inline image/ data URL"):
            media_bytes({"media": "data:audio/wav;base64,AAAA"}, "image/")
        with self.assertRaisesRegex(ValueError, "bounded base64 data"):
            media_bytes({"media": "data:image/png,not-base64"}, "image/")
        with self.assertRaisesRegex(ValueError, "bounded base64 data"):
            media_bytes({"media": "data:image/png;base64,***"}, "image/")

    def test_images_are_converted_to_rgb_and_bounded(self):
        image = image_from_case({"media": png()})
        self.assertEqual(image.mode, "RGB")
        with self.assertRaisesRegex(ValueError, "sixteen million pixels"):
            image_from_case({"media": png(4001, 4000)})


class LightOnTests(unittest.TestCase):
    def test_lighton_requires_the_ocr_directive_and_returns_text(self):
        injected = modules()
        with patch.dict(sys.modules, injected):
            runtime = load_image(
                profile_for("lightonai/LightOnOCR-3-0.8B"), Path("/models/lighton"), 2
            )
            self.assertEqual(runtime.audit["parameter_count"], 4)
            self.assertEqual(runtime.preparation["image_longest_edge"], 2048)
            with self.assertRaisesRegex(ValueError, "only the <OCR>"):
                runtime.invoke(case("Describe this image"), 32)
            result = runtime.invoke(case(), 32)
        self.assertEqual(result["output_text"], "recognized page text")
        self.assertEqual(result["input_tokens"], 5)
        self.assertEqual(result["output_tokens"], 2)

    def test_unsupported_image_adapter_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported native image adapter"):
            ImageRuntime(
                {"adapter": "unknown", "precision": "float32"},
                Path("/none"),
                FakeTorch(),
            )

    def test_registry_routes_ocr_without_reviewed_publisher_source(self):
        self.assertFalse(ADAPTERS["lighton"].reviewed_code)
        self.assertFalse(ADAPTERS["indra"].reviewed_code)
        self.assertIs(ADAPTERS["lighton"].load, load_image)


class IndraAdmissionTests(unittest.TestCase):
    def backend(self, bound):
        runtime = SimpleNamespace(
            audit={
                "parameter_count": None,
                "parameter_count_upper_bound": bound,
                "parameter_count_basis": "onnx-full-static-constant-upper-bound-v1",
            },
            configuration=None,
            preparation={"route": "publisher OrtOCR full profile including math"},
        )
        with (
            patch("qit_adapters.backend.verify", return_value={}),
            patch(
                "qit_adapters.backend.locked_environment",
                return_value={"packages": {}},
            ),
            patch("qit_adapters.backend.forbid_external_connections"),
            patch.dict(ADAPTERS, {"indra": Adapter(lambda *args: runtime)}),
        ):
            return Backend(profile_for("datalab-to/indra-ocr"), Path("/models/indra"))

    def test_indra_reports_its_static_bound_instead_of_an_exact_count(self):
        backend = self.backend(300_000_000)
        configuration = backend.metadata()["configuration"]
        self.assertIsNone(configuration["parameter_count"])
        self.assertEqual(configuration["parameter_count_upper_bound"], 300_000_000)
        self.assertEqual(
            configuration["parameter_count_basis"],
            "onnx-full-static-constant-upper-bound-v1",
        )

    def test_indra_refuses_an_unbounded_or_missing_bound(self):
        with self.assertRaisesRegex(ValueError, "sub-billion limit"):
            self.backend(2_000_000_000)
        with self.assertRaisesRegex(ValueError, "sub-billion limit"):
            self.backend(None)


if __name__ == "__main__":
    unittest.main()
