import argparse
import hashlib
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "qit-runtime" / "src" / "model-catalog.generated.json"
LOCAL_CPU = ROOT / "providers" / "local-cpu"
ADAPTERS = ROOT / "providers" / "model-adapters"
OLLAMA = ROOT / "providers" / "model-routes" / "ollama.json"
HUGGINGFACE = ROOT / "providers" / "model-routes" / "huggingface.json"


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_catalog(module_path, package):
    sys.path.insert(0, str(module_path))
    try:
        module = __import__(f"{package}.catalog", fromlist=["profiles"])
        return module.profiles()
    finally:
        sys.path.pop(0)


def adapter_catalog():
    sys.path.insert(0, str(ADAPTERS))
    try:
        module = __import__("qit_adapters.artifacts", fromlist=["profiles"])
        registry = __import__("qit_adapters.registry", fromlist=["ADAPTERS"])
        return [
            {**record, "implementation": "experimental" if record["adapter"] in registry.ADAPTERS else "pending"}
            for record in module.profiles()
        ]
    finally:
        sys.path.pop(0)


def require_revision(record):
    revision = record.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(
            f"model {record.get('model')!r} must pin a forty character revision"
        )
    return revision


def route(provider, serve_as, task, implementation, adapter=None, **extra):
    entry = {
        "provider": provider,
        "serve_as": serve_as,
        "adapter": adapter,
        "tasks": [task],
        "implementation": implementation,
    }
    for name in ("precision", "qualification"):
        if extra.get(name) is not None:
            entry[name] = extra[name]
    return entry


def build():
    sources = []
    models = {}

    def model(record, parameters=None, license=None):
        identifier = record.get("model")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("every catalog entry needs a model identifier")
        entry = models.setdefault(
            identifier,
            {"id": identifier, "parameters": None, "license": None, "revision": None, "routes": []},
        )
        for name, value in (("parameters", parameters), ("license", license), ("revision", record.get("revision"))):
            if value is None:
                continue
            if entry[name] is not None and entry[name] != value:
                raise ValueError(
                    f"model {identifier!r} declares conflicting {name}: {entry[name]!r} and {value!r}"
                )
            entry[name] = value
        return entry

    for record in provider_catalog(LOCAL_CPU, "qit_cpu"):
        require_revision(record)
        entry = model(record, record.get("parameters"), record.get("license"))
        entry["routes"].append(route("local-cpu", record["model"], record["task"], "direct"))
    sources.append({"name": "local-cpu", "path": "providers/local-cpu/qit_cpu/models.json", "sha256": digest(LOCAL_CPU / "qit_cpu" / "models.json")})

    for record in adapter_catalog():
        require_revision(record)
        parameters = record.get("model_card", {}).get("parameter_count", record.get("parameters"))
        entry = model(record, parameters, record.get("license"))
        entry["routes"].append(
            route(
                "local-cpu",
                record["model"],
                record["task"],
                record.get("implementation", "pending"),
                adapter=record["adapter"],
                precision=record.get("precision"),
                qualification=record.get("qualification"),
            )
        )
    sources.append({"name": "model-adapters", "path": "providers/model-adapters/qit_adapters/models.json", "sha256": digest(ADAPTERS / "qit_adapters" / "models.json")})

    for record in read_json(OLLAMA)["routes"]:
        entry = model(record, record.get("parameters"), record.get("license"))
        entry["routes"].append(route("ollama", record["package"], record["task"], "direct"))
    sources.append({"name": "ollama", "path": "providers/model-routes/ollama.json", "sha256": digest(OLLAMA)})

    for record in read_json(HUGGINGFACE)["routes"]:
        entry = model(record, record.get("parameters"), record.get("license"))
        entry["routes"].append(
            route(
                record["provider"],
                record["model"],
                record["task"],
                "direct",
                precision=record.get("precision"),
                qualification=record.get("qualification"),
            )
        )
    sources.append({"name": "huggingface", "path": "providers/model-routes/huggingface.json", "sha256": digest(HUGGINGFACE)})

    seen = set()
    for entry in models.values():
        if entry["revision"] is None:
            raise ValueError(f"model {entry['id']!r} has no pinned revision")
        for item in entry["routes"]:
            key = (item["provider"], item["adapter"], item["serve_as"])
            if key in seen:
                raise ValueError(f"duplicate route {key}")
            seen.add(key)
        entry["routes"].sort(key=lambda item: (item["provider"], item["adapter"] or "", item["serve_as"]))
    ordered = sorted(
        models.values(),
        key=lambda entry: (
            entry["parameters"] is None,
            entry["parameters"] or 0,
            entry["id"],
        ),
    )
    return {"schema_version": 1, "sources": sources, "models": ordered}


def render(document):
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    document = build()
    text = render(document)
    if arguments.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != text:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=str(OUTPUT),
                tofile="generated",
            )
            sys.stderr.writelines(diff)
            print("model catalog is stale; run python3 scripts/generate-model-catalog.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(document['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
