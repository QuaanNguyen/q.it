from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Adapter:
    load: Callable[[dict, Path, int], object]
    reviewed_code: bool = False


ADAPTERS: dict[str, Adapter] = {}
