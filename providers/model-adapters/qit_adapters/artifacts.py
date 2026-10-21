import hashlib
import importlib.metadata
from importlib.resources import files
import json
from pathlib import Path
import re


LOCK_FAMILIES = {"uibk-agent": "uibk", "redux": "photon"}


def profiles():
    catalog = json.loads(files("qit_adapters").joinpath("models.json").read_text())
    return catalog


def profile_for(model):
    for profile in profiles():
        if profile["model"] == model:
            if not re.fullmatch(r"[0-9a-f]{40}", profile["revision"]):
                raise ValueError("checkpoint revision must be an immutable commit")
            count = profile.get("model_card", {}).get("parameter_count")
            if type(count) is not int or not 0 < count < 1_000_000_000:
                raise ValueError(
                    "model card must establish fewer than one billion parameters"
                )
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


def verify_runtime_source(profile, distribution):
    expected = profile["runtime_source"]
    package = Path(distribution.locate_file("frida_decisions"))
    root = package.resolve()
    if package.is_symlink() or not root.is_dir():
        raise ValueError("FRIDA source path escapes its installed package")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("FRIDA source path escapes its installed package")
    installed = {path.relative_to(root).as_posix(): path for path in root.rglob("*.py")}
    if installed.keys() != expected["python_files"].keys():
        raise ValueError(
            "FRIDA installed Python inventory differs from the pinned publisher source"
        )
    observed = {}
    for name, digest in expected["python_files"].items():
        path = installed[name]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("FRIDA source path escapes its installed package")
        observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed[name] != digest:
            raise ValueError(
                f"FRIDA installed source differs from the reviewed commit: {name}"
            )
    return {
        "reviewed_archive_sha256": expected["archive_sha256"],
        "commit": expected["commit"],
        "installed_python_sha256": observed,
    }


def lock_family(adapter):
    return LOCK_FAMILIES.get(adapter, adapter)


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
    family = lock_family(profile["adapter"])
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
    source_verification = None
    if family == "frida":
        expected = profile["runtime_source"]
        source = sources.get("frida-decisions", {})
        url = f"https://github.com/ai-forever/FRIDA-Decisions/archive/{expected['commit']}.tar.gz"
        if source.get("url") != url:
            raise ValueError(
                "FRIDA runtime must retain the exact publisher Git source identity"
            )
        archive = source.get("archive_info", {})
        recorded = archive.get("hashes", {}).get("sha256")
        legacy = archive.get("hash", "")
        if (recorded is not None and recorded != expected["archive_sha256"]) or (
            legacy.startswith("sha256=")
            and legacy != "sha256=" + expected["archive_sha256"]
        ):
            raise ValueError(
                "FRIDA runtime archive hash differs from the reviewed source"
            )
        source_verification = verify_runtime_source(
            profile, importlib.metadata.distribution("frida-decisions")
        )
    return {
        "packages": installed,
        "adapter_source": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files("qit_adapters").iterdir()
            if path.is_file()
            and (path.name.endswith(".py") or path.name == "models.json")
        },
        "lock_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "installed_records": records,
        "installed_sources": sources,
        "source_verification": source_verification,
    }
