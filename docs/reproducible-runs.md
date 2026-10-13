# Reproducing benchmark runs

Reproducibility means another run uses the same important conditions and can therefore be compared fairly.
The checklist below is intended for normal Raspberry Pi users rather than benchmark-framework developers.

## Before the run

Record or keep constant:

- Raspberry Pi model and RAM size.
- Cooling method, such as the official active cooler or a fan case.
- Power supply.
- Raspberry Pi OS version and kernel.
- Model-server name and version.
- Exact model name, file, revision, and quantization.
- Benchmark-pack version.
- Warmup count, measured-pass count, and output limit.
- Any special server setting, such as thread count or context size.

Stop unrelated heavy work when possible.
A file copy, software update, browser session, or another model request can change CPU, storage, and memory results.

q.it reads and saves the Ollama server version before running any benchmark cases.
Ollama embedding runs require version 0.40.3 or later, including when the server runs on another machine.
Earlier versions and unverifiable versions fail preflight and remain in run history.
See the [Ollama 0.40.3 release notes](https://github.com/ollama/ollama/releases/tag/v0.40.3) for the embedding performance change.
The local CPU provider also records its server version, package versions, checkpoint revisions, precision, thread count, and model-specific preparation.
Other provider versions remain unrecorded until their integrations can report them reliably.

Allow the Pi to return to a similar idle temperature before runs that will be compared closely.
Use the same cooling and power setup because heat or undervoltage can reduce performance.

## Use explicit run settings

Write important choices in the command instead of relying on defaults:

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model gemma3:1b \
  --warmups 1 \
  --iterations 5 \
  --max-tokens 64 \
  --timeout 120
```

Use the same command when comparing two runs.
Change only the item you are trying to study, such as the model or quantization.

## Keep warm and cold tests separate

The current built-in workflow measures a warmed-up model.
The warmup request is not included in the reported measurements.

Do not compare that result with a first request that also had to load the model into memory.
Model loading and steady generation answer different questions.

## Understand CPU and memory scope

The default CPU and RAM values describe the whole Pi during each request.
They can include unrelated programs and system services.

If you know the model server's process number, pass it to q.it:

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model gemma3:1b \
  --pid 1234
```

This adds memory sampling for that process.
It does not turn the host-wide CPU measurement into a process-only measurement.

## Save and share the result

Every attempt is already stored locally.
Export successful runs as JSONL when another person needs the detailed samples:

```bash
qit export --format jsonl --output results.jsonl
```

Use CSV for a compact table that opens easily in spreadsheet software:

```bash
qit export --format csv --output results.csv
```

Share the command, exported result, Pi model, cooling setup, power supply, operating-system version, server version, and exact model identity together.

## Compare like with like

Do not compare scores from different benchmark-pack versions as if they used the same questions.
Do not compare image or audio results when preprocessing settings differ.
Do not combine failed runs with successful runs.
Do not treat the built-in smoke checks as full quality leaderboards.

The dashboard keeps model comparisons with different recorded Ollama versions in separate machine and run setups.
Historical runs without a saved server version remain available in their own setup.
The dashboard marks a latency-and-memory point as Pareto-efficient only among successful runs using the same benchmark, pack version, host name, and recorded provider metadata.
You should still confirm that the remaining hardware and software conditions match.
