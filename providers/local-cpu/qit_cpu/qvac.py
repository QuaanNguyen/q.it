import json
import hashlib
from importlib.resources import files
from pathlib import Path
import tempfile
import urllib.error
import urllib.request


def profiles():
    return json.loads(files("qit_cpu").joinpath("qvac-models.json").read_text())


def profile_for(model):
    for profile in profiles():
        if profile["model"] == model:
            return profile
    raise ValueError(f"unsupported NeMo model {model!r}; use qit-nemo list")


def required_files(profile, route):
    if route not in profile["routes"]:
        raise ValueError(
            f"route_unsupported: {route!r} is not supported by {profile['model']}"
        )
    names = [
        profile["archive"],
        "inference_gated.py" if route == "automatic" else "inference.py",
    ]
    if route in ("multilingual", "automatic"):
        names.append("adapters.safetensors")
    if route == "automatic":
        names.append("language_gate.safetensors")
    return names


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def artifact_path(directory, name):
    root = Path(directory).resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(
            f"artifact_path: {name} must be a regular file inside the model directory"
        )
    return path


def verify(profile, route, directory):
    verified = {}
    for name in required_files(profile, route):
        path = artifact_path(directory, name)
        if not path.is_file():
            raise ValueError(
                f"artifact_missing: {name}; run qit-nemo fetch explicitly before serving"
            )
        checksum = digest(path)
        if checksum != profile["files"][name]:
            raise ValueError(
                f"artifact_digest: {name} differs from the pinned publisher revision"
            )
        verified[name] = {"sha256": checksum, "bytes": path.stat().st_size}
    return verified


def fetch(profile, route, directory):
    names = required_files(profile, route)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = artifact_path(root, name)
        if destination.exists():
            if (
                not destination.is_file()
                or digest(destination) != profile["files"][name]
            ):
                raise ValueError(
                    f"artifact_digest: existing {name} differs from the pinned publisher revision"
                )
            continue
        url = f"https://huggingface.co/{profile['model']}/resolve/{profile['revision']}/{name}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 2 * 1024**3:
                            raise ValueError(
                                f"artifact_size: {name} exceeds the bounded download size"
                            )
                        output.write(chunk)
            if digest(temporary) != profile["files"][name]:
                raise ValueError(
                    f"artifact_digest: downloaded {name} differs from the pinned publisher revision"
                )
            temporary.replace(destination)
        except (urllib.error.URLError, OSError) as error:
            raise ValueError(f"artifact_download: {name}: {error}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify(profile, route, directory)
