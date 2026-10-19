import hashlib
import importlib.metadata
from importlib.resources import files
import json
from pathlib import Path
import re


IMPLEMENTED_ADAPTERS = {
    "gliclass",
    "gliner2",
    "intern",
    "liquid",
    "typed-gemma",
    "d3",
    "frida",
    "lighton",
    "ultrabert",
    "uibk",
    "uibk-agent",
    "photon",
    "redux",
    "nemotron",
    "indra",
}
REVIEWED_SOURCE_ADAPTERS = {
    "intern",
    "liquid",
    "typed-gemma",
    "d3",
    "ultrabert",
    "uibk",
    "uibk-agent",
}


def profiles():
    catalog = json.loads(files("qit_adapters").joinpath("models.json").read_text())
    for profile in catalog:
        profile["implementation"] = (
            "experimental" if profile["adapter"] in IMPLEMENTED_ADAPTERS else "pending"
        )
    return catalog


def profile_for(model):
    for profile in profiles():
        if profile["model"] == model:
            if not re.fullmatch(r"[0-9a-f]{40}", profile["revision"]):
                raise ValueError("checkpoint revision must be an immutable commit")
            return profile
    raise ValueError(f"unsupported model {model!r}; use qit-adapter list")


def verify(profile, directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(
            "model not installed: supply an existing local model directory"
        )
    observed = {}
    for name, identity in profile["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(
                f"artifact path must stay inside the model directory: {name}"
            )
        if not path.is_file():
            raise ValueError(
                f"model not installed: missing {name}; install the pinned snapshot yourself"
            )
        size = path.stat().st_size
        if size != identity["bytes"]:
            raise ValueError(
                f"artifact size differs from the pinned publisher file: {name}"
            )
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
        expected = identity.get("sha256")
        if (expected and sha256.hexdigest() != expected) or (
            not expected and git_blob.hexdigest() != identity.get("git_blob")
        ):
            raise ValueError(
                f"artifact digest differs from the pinned publisher file: {name}"
            )
        observed[name] = {"sha256": sha256.hexdigest(), "bytes": size}
    return observed


def environment(required):
    installed = {}
    for name, expected in required.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(
                f"runtime not installed: {name}=={expected}; use its isolated optional extra"
            ) from error
        if (
            version != expected
            if "+" in expected
            else version.split("+", 1)[0] != expected
        ):
            raise ValueError(
                f"unsupported runtime: {name} {version}; requires {expected}"
            )
        installed[name] = version
    return installed


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def locked_environment(profile):
    import platform
    import sys

    if (
        platform.system() != "Linux"
        or platform.machine() not in {"x86_64", "aarch64"}
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError(
            "unsupported environment: optional-adapter locks cover Linux CPython 3.12 on the explicitly available architecture; native Raspberry Pi qualification remains pending"
        )
    libc, libc_version = platform.libc_ver()
    if libc != "glibc" or tuple(int(part) for part in libc_version.split(".")[:2]) < (
        2,
        28,
    ):
        raise ValueError(
            "unsupported libc: optional CPU wheels require glibc 2.28 or later"
        )
    family = {"uibk-agent": "uibk", "redux": "photon"}.get(
        profile["adapter"], profile["adapter"]
    )
    resource = files("qit_adapters").joinpath(
        f"locks/{family}-linux-{platform.machine()}-py312.txt"
    )
    if not resource.is_file():
        raise ValueError(
            f"unsupported native dependencies: {family} has no retained lock for {platform.machine()}; no runtime or precision fallback is permitted"
        )
    content = resource.read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\\\s;]+)", content, re.MULTILINE))
    installed = environment(pins)
    for name, expected in profile["requirements"].items():
        installed.update(environment({name: expected}))
    records = {}
    sources = {}
    for name in installed:
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        records[name] = hashlib.sha256(record.encode()).hexdigest() if record else None
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            sources[name] = json.loads(direct_url)
    if family == "frida":
        source = sources.get("frida-decisions", {})
        if (
            source.get("archive_info", {}).get("hashes", {}).get("sha256")
            != "9a927dbd9958b18772ec0b35f6240f0a865e3f4fbb6df3f5dabfbe0a74608e8b"
            or source.get("url")
            != "https://github.com/ai-forever/FRIDA-Decisions/archive/00c8b5f0a88312e969d65b01a076bd9e447de2db.tar.gz"
        ):
            raise ValueError(
                "FRIDA runtime must retain the exact publisher Git source identity"
            )
    return {
        "packages": installed,
        "lock_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "installed_records": records,
        "installed_sources": sources,
    }
