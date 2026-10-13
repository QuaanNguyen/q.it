import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from huggingface_hub import get_token

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description="Run bounded real-checkpoint CPU protocol checks, with explicit model downloads"
)
parser.add_argument("models", nargs="*")
parser.add_argument("--exclude", action="append", default=[])
parser.add_argument(
    "--output", type=Path, default=root / ".scratch/local-cpu-validation"
)
parser.add_argument("--cache", type=Path, default=root / ".scratch/cpu-model-cache")
parser.add_argument("--qit", type=Path, default=root / "target/debug/qit")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--trust-publisher-code", action="store_true")
args = parser.parse_args()
if not args.qit.is_file():
    parser.error("build q.it first or provide --qit with an existing executable")
if not 1 <= args.threads <= 64:
    parser.error("threads must be between one and sixty-four")
results = args.output.resolve()
results.mkdir(parents=True, exist_ok=True)
profiles = json.loads((root / "providers/local-cpu/qit_cpu/models.json").read_text())
selected = [
    profile
    for profile in profiles
    if profile["model"] not in args.exclude
    and (not args.models or profile["model"] in args.models)
]
unknown = set(args.models + args.exclude) - {profile["model"] for profile in profiles}
if unknown:
    parser.error("unknown canonical models: " + ", ".join(sorted(unknown)))
if not selected:
    parser.error("no models selected")
if (
    any(profile["trust_publisher_code"] for profile in selected)
    and not args.trust_publisher_code
):
    parser.error(
        "selected models require --trust-publisher-code to enable their pinned publisher code"
    )
base = root / "qit-runtime/benchmarks/core"
source_digest = hashlib.sha256(
    b"".join(
        path.read_bytes()
        for path in sorted((root / "providers/local-cpu/qit_cpu").glob("*.py"))
    )
    + (root / "providers/local-cpu/qit_cpu/models.json").read_bytes()
).hexdigest()
binary_digest = hashlib.sha256(args.qit.read_bytes()).hexdigest()
validation_digest = hashlib.sha256(
    Path(__file__).read_bytes()
    + b"".join(path.read_bytes() for path in sorted(base.glob("*.jsonl")))
).hexdigest()
report = []
for profile in selected:
    label = profile["model"].replace("/", "--")
    directory = results / label
    directory.mkdir(exist_ok=True)
    previous = directory / "result.json"
    if (
        previous.exists()
        and json.loads(previous.read_text()).get("status") == "succeeded"
        and json.loads(previous.read_text()).get("source_digest") == source_digest
        and json.loads(previous.read_text()).get("binary_digest") == binary_digest
        and json.loads(previous.read_text()).get("validation_digest")
        == validation_digest
        and json.loads(previous.read_text()).get("threads") == args.threads
    ):
        report.append(json.loads(previous.read_text()))
        continue
    env = dict(
        os.environ,
        HF_HOME=str(args.cache.resolve()),
        HF_HUB_DISABLE_XET="1",
        QIT_HOME=str(results / "state"),
        TOKENIZERS_PARALLELISM="false",
    )
    if env.get("QIT_CPU_API_KEY"):
        env["QIT_API_KEY"] = env["QIT_CPU_API_KEY"]
    else:
        env.pop("QIT_API_KEY", None)
    if profile["model"].startswith("google/"):
        token = get_token()
        if token:
            env["HF_TOKEN"] = token
    log = (directory / "server.log").open("w")
    command = [
        sys.executable,
        "-m",
        "qit_cpu",
        "serve",
        "--model",
        profile["model"],
        "--port",
        "0",
        "--threads",
        str(args.threads),
    ]
    if profile["trust_publisher_code"]:
        command.append("--trust-publisher-code")
    server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    record = {
        "model": profile["model"],
        "revision": profile["revision"],
        "task": profile["task"],
        "implementation": profile["implementation"],
        "status": "failed",
        "source_digest": source_digest,
        "binary_digest": binary_digest,
        "validation_digest": validation_digest,
        "threads": args.threads,
        "checked_at_ms": int(time.time() * 1000),
    }
    print("START", profile["model"], flush=True)
    try:
        deadline = time.monotonic() + 900
        endpoint = None
        while time.monotonic() < deadline and server.poll() is None:
            lines = (directory / "server.log").read_text(errors="replace").splitlines()
            endpoints = [
                line.removeprefix("q.it CPU provider listening on ")
                for line in lines
                if line.startswith("q.it CPU provider listening on ")
            ]
            if endpoints:
                endpoint = endpoints[-1]
                break
            time.sleep(1)
        if not endpoint:
            raise RuntimeError("CPU server did not load; inspect server.log")
        headers = (
            {"Authorization": "Bearer " + env["QIT_CPU_API_KEY"]}
            if env.get("QIT_CPU_API_KEY")
            else {}
        )
        metadata = json.load(
            urlopen(Request(endpoint + "/v1/qit/model", headers=headers), timeout=10)
        )
        (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
        configuration = metadata["configuration"]
        if configuration["device"] != "cpu" or configuration["dtype"] != "float32":
            raise RuntimeError("checkpoint must run entirely on CPU in float32")
        if (
            configuration["revision"] != profile["revision"]
            or not 0 < configuration["parameter_count"] < 1000000000
        ):
            raise RuntimeError(
                "loaded checkpoint must match its pinned revision and parameter limit"
            )
        task = profile["task"]
        mode = profile["implementation"]
        names = {
            "text_generation": "text-generation-smoke",
            "embedding": "embedding-retrieval",
            "reranking": "reranking-relevance",
            "image_to_text": "image-to-text-smoke",
            "speech_to_text": "speech-to-text-smoke",
            "text_to_image": "text-to-image-smoke",
        }
        case = json.loads((base / (names[task] + ".jsonl")).read_text().splitlines()[0])
        if mode in ("git", "blip"):
            case["prompt"] = "a photograph of"
            case.pop("expected_contains", None)
        if mode == "florence":
            case["prompt"] = "<CAPTION>"
            case.pop("expected_contains", None)
        if task == "text_to_image":
            case["image_settings"] = {
                "width": 64,
                "height": 64,
                "steps": 2,
                "guidance_scale": 7.5,
                "seed": 42,
            }
        pack = directory / "pack"
        pack.mkdir(exist_ok=True)
        (pack / "cases.jsonl").write_text(json.dumps(case) + "\n")
        pack_id = "local-cpu-validation-" + mode.replace("_", "-")
        (pack / "pack.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": pack_id,
                    "name": "Local CPU protocol validation",
                    "version": "1.0.0",
                    "description": "One bounded case; not an official evaluation",
                    "publisher": "q.it",
                    "license": "CC0-1.0",
                    "source": "Bundled q.it synthetic cases",
                    "benchmarks": [
                        {
                            "id": mode.replace("_", "-"),
                            "name": mode + " CPU transport check",
                            "description": "One synthetic CPU case, no warmup, sixteen-token limit",
                            "task": task,
                            "cases": "cases.jsonl",
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        qit = str(args.qit.resolve())
        subprocess.run(
            [qit, "benchmark", "remove", pack_id],
            env=env,
            capture_output=True,
            text=True,
        )
        install = subprocess.run(
            [qit, "benchmark", "install", str(pack)],
            env=env,
            capture_output=True,
            text=True,
        )
        if install.returncode:
            raise RuntimeError(install.stderr)
        output = subprocess.run(
            [
                qit,
                "run",
                pack_id + "/" + mode.replace("_", "-"),
                "--provider",
                "local-cpu",
                "--model",
                profile["model"],
                "--base-url",
                endpoint,
                "--warmups",
                "0",
                "--iterations",
                "1",
                "--max-tokens",
                "16",
                "--timeout",
                "600",
                "--pid",
                str(server.pid),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=660,
        )
        (directory / "qit.log").write_text(output.stdout + output.stderr)
        export = subprocess.run(
            [qit, "export", "--format", "jsonl"],
            env=env,
            capture_output=True,
            text=True,
        )
        all_records = [json.loads(line) for line in export.stdout.splitlines()]
        matches = [
            row for row in all_records if row["run"]["model"] == profile["model"]
        ]
        if matches:
            row = max(matches, key=lambda row: row["run"]["started_at_ms"])
            (directory / "result.jsonl").write_text(json.dumps(row) + "\n")
            record["run_id"] = row["run"]["id"]
        if output.returncode:
            raise RuntimeError(output.stdout + output.stderr)
        if (
            not matches
            or row["run"]["provider_metadata"]["configuration"]["device"] != "cpu"
        ):
            raise RuntimeError("CPU run was not retained in its JSONL export")
        record.update(
            status="succeeded", parameter_count=configuration["parameter_count"]
        )
    except Exception as error:
        record["error"] = str(error)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()
    previous.write_text(json.dumps(record, indent=2) + "\n")
    report.append(record)
    (results / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        record["status"].upper(), profile["model"], record.get("error", ""), flush=True
    )

all_results = [
    json.loads(path.read_text()) for path in sorted(results.glob("*/result.json"))
]
(results / "report.json").write_text(json.dumps(all_results, indent=2) + "\n")
print(
    f"Selected checks: {sum(row['status'] == 'succeeded' for row in report)}/{len(report)} succeeded",
    flush=True,
)
sys.exit(0 if all(row["status"] == "succeeded" for row in report) else 1)
