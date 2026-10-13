import json
import re
from importlib.resources import files


def profiles():
    return json.loads(files("qit_cpu").joinpath("models.json").read_text())


def profile_for(model):
    for profile in profiles():
        if profile["model"] == model:
            if not re.fullmatch(r"[a-f0-9]{40}", profile["revision"]):
                raise ValueError(
                    "model profile must pin an immutable checkpoint revision"
                )
            return profile
    raise ValueError(f"unknown CPU model {model!r}; use qit-cpu list")


def image_settings(case):
    value = case.get("image_settings") or {
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance_scale": 7.5,
        "seed": 42,
    }
    if set(value) != {"width", "height", "steps", "guidance_scale", "seed"}:
        raise ValueError(
            "image settings require width, height, steps, guidance_scale, and seed"
        )
    for name in ("width", "height"):
        size = value[name]
        if type(size) is not int or not 64 <= size <= 1024 or size % 8:
            raise ValueError(
                "image dimensions must be multiples of eight between 64 and 1024"
            )
    if type(value["steps"]) is not int or not 1 <= value["steps"] <= 100:
        raise ValueError("image steps must be between one and one hundred")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**63:
        raise ValueError("image seed must be a nonnegative signed 64-bit integer")
    if (
        type(value["guidance_scale"]) not in (int, float)
        or not 0 <= value["guidance_scale"] <= 30
    ):
        raise ValueError("image guidance must be between zero and thirty")
    return value


def prepared_inputs(profile, case):
    query = required_text(case, "input")
    documents = case.get("documents")
    if (
        not isinstance(documents, list)
        or not documents
        or not all(isinstance(d, str) and d.strip() for d in documents)
    ):
        raise ValueError("documents must be a nonempty list of text passages")
    query_prefix = case.get("query_prefix")
    document_prefix = case.get("document_prefix")
    if query_prefix is None:
        query_prefix = profile.get("query_prefix", "")
    if document_prefix is None:
        document_prefix = profile.get("document_prefix", "")
    if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
        raise ValueError("retrieval prefixes must be strings")
    return [query_prefix + query] + [document_prefix + d for d in documents]


def required_text(case, name):
    value = case.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case requires nonempty {name}")
    return value
