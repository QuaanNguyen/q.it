import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--qit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pack")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-reviewed-code", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--native-reference")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "QIT_HOME": str(output / "state")}
    report = {
        "model": args.model,
        "benchmark": args.benchmark,
        "status": "failed",
        "scope": "installed-command CPU smoke; dataset and Raspberry Pi qualification are separate",
    }
    command = [
        args.adapter,
        "serve",
        "--model",
        args.model,
        "--model-path",
        args.model_path,
        "--threads",
        str(args.threads),
        "--port",
        "0",
    ]
    if args.allow_reviewed_code:
        command.append("--allow-reviewed-code")
    server = None
    try:
        if args.pack:
            subprocess.run(
                [args.qit, "benchmark", "install", args.pack],
                env=environment,
                check=True,
            )
        with (
            (output / "server.stdout").open("w+") as stdout,
            (output / "server.stderr").open("w") as stderr,
        ):
            server = subprocess.Popen(
                command, env=environment, stdout=stdout, stderr=stderr
            )
            endpoint = None
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                stdout.seek(0)
                for line in stdout:
                    if line.startswith("q.it optional adapter listening on "):
                        endpoint = line.strip().split(" on ", 1)[1]
                        break
                if endpoint or server.poll() is not None:
                    break
                time.sleep(0.2)
            if endpoint is None:
                raise ValueError(
                    "adapter startup failed or timed out; retain server.stderr for the precise native diagnostic"
                )
            request = urllib.request.Request(endpoint + "/v1/qit/model")
            if environment.get("QIT_API_KEY"):
                request.add_header(
                    "Authorization", "Bearer " + environment["QIT_API_KEY"]
                )
            with urllib.request.urlopen(request, timeout=10) as response:
                metadata = json.load(response)
            (output / "metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            run_command = [
                args.qit,
                "run",
                args.benchmark,
                "--provider",
                "local-cpu",
                "--model",
                args.model,
                "--base-url",
                endpoint,
                "--iterations",
                "1",
                "--warmups",
                "0",
                "--timeout",
                "600",
                "--pid",
                str(server.pid),
            ]
            with (
                (output / "run.stdout").open("w") as run_stdout,
                (output / "run.stderr").open("w") as run_stderr,
            ):
                run = subprocess.run(
                    run_command,
                    env=environment,
                    stdout=run_stdout,
                    stderr=run_stderr,
                    timeout=3600,
                )
            report["run_exit_code"] = run.returncode
            with (output / "successful.jsonl").open("w") as exported:
                subprocess.run(
                    [args.qit, "export", "--format", "jsonl"],
                    env=environment,
                    stdout=exported,
                    check=True,
                )
            if run.returncode:
                raise ValueError(
                    "benchmark failed; retain the state database and run.stderr"
                )
            report["status"] = "cpu-smoke-passed"
            report["publisher_parity"] = (
                "not checked; requires a separately produced publisher-native reference"
            )
            if args.native_reference:
                reference = json.loads(Path(args.native_reference).read_text())
                rows = [
                    json.loads(line)
                    for line in (output / "successful.jsonl").read_text().splitlines()
                ]
                samples = [sample for row in rows for sample in row["samples"]]
                observed = {
                    sample["case_id"]: sample.get("decision")
                    or sample.get("retrieval")
                    or sample.get("speech")
                    or sample.get("output_excerpt")
                    for sample in samples
                }
                (output / "observed.json").write_text(
                    json.dumps(observed, indent=2, allow_nan=False) + "\n"
                )
                if reference != observed:
                    raise ValueError(
                        "publisher-native reference differs from retained observations"
                    )
                report["publisher_parity"] = "exact retained-observation match"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        database = output / "state/results.db"
        if database.exists():
            with sqlite3.connect(database) as connection:
                connection.row_factory = sqlite3.Row
                for table in ("benchmark_runs", "benchmark_samples"):
                    rows = [
                        dict(row)
                        for row in connection.execute("SELECT * FROM " + table)
                    ]
                    (output / (table + ".json")).write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "cpu-smoke-passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
