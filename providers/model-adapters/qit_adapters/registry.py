from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .decisions import load_decision


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
}
