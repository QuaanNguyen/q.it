from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .decisions import load_decision
from .models import load_embedding, load_image
from .nemotron import load_nemotron
from .onnx_ocr import load_onnx_ocr
from .speech import load_speech


@dataclass(frozen=True)
class Adapter:
    load: Callable[[dict, Path, int], object]
    reviewed_code: bool = False


ADAPTERS: dict[str, Adapter] = {
    "d3": Adapter(load_decision, reviewed_code=True),
    "frida": Adapter(load_decision),
    "gliclass": Adapter(load_decision),
    "gliner2": Adapter(load_decision),
    "intern": Adapter(load_decision, reviewed_code=True),
    "liquid": Adapter(load_decision, reviewed_code=True),
    "typed-gemma": Adapter(load_decision, reviewed_code=True),
    "ultrabert": Adapter(load_embedding, reviewed_code=True),
    "uibk": Adapter(load_embedding, reviewed_code=True),
    "uibk-agent": Adapter(load_embedding, reviewed_code=True),
    "lighton": Adapter(load_image),
    "indra": Adapter(load_onnx_ocr),
    "photon": Adapter(load_speech),
    "redux": Adapter(load_speech),
    "nemotron": Adapter(load_nemotron),
}
