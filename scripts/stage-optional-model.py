import argparse
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

from qit_adapters.artifacts import profile_for, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    profile = profile_for(args.model)
    plan = {
        "model": profile["model"],
        "revision": profile["revision"],
        "bytes": sum(entry["bytes"] for entry in profile["files"].values()),
        "files": profile["files"],
    }
    if not args.download:
        print(json.dumps(plan, indent=2))
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in profile["files"]:
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("artifact output escapes the staging directory")
        if path.exists():
            verify({"files": {name: profile["files"][name]}}, root)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/"
            + profile["model"]
            + "/resolve/"
            + profile["revision"]
            + "/"
            + urllib.parse.quote(name)
        )
        request = urllib.request.Request(url)
        if os.environ.get("HF_TOKEN"):
            request.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
        temporary = path.with_name(path.name + ".partial")
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("xb") as output,
            ):
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
            verify(
                {"files": {temporary.name: profile["files"][name]}}, temporary.parent
            )
            temporary.rename(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    print(
        json.dumps(
            {"model": profile["model"], "artifacts": verify(profile, root)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
