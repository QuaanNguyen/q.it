# Issue 81 Sol CPU validation

Inspected on October 16, 2026, for [issue 81](https://github.com/QuaanNguyen/q.it/issues/81).
The previously submitted setup job and all three text-model array tasks completed successfully on October 15.
Their retained q.it exports contain nine successful benchmark runs and thirty-nine measured sample results.

## Completed text checks

| Canonical checkpoint | Loaded parameter count | Ollama weight format | Sol node | Array task |
|---|---:|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 494,032,768 | Q4_K_M | sc012 | 65695474_0 |
| HuggingFaceTB/SmolLM2-135M-Instruct | 134,515,008 | F16 | sc012 | 65695474_1 |
| HuggingFaceTB/SmolLM2-360M-Instruct | 361,821,120 | F16 | sc015 | 65695474_2 |

Each model completed the three-case built-in text-generation smoke benchmark and the existing five-case GSM8K and MMLU-Pro packs.
Each run used one warmup, one measured pass, a 128-token output limit, and a 300-second request timeout.
All samples retain output excerpts, request latency, time to first token, exact token throughput, host CPU and memory observations, and model-process peak RSS.
Ollama 0.40.3 is recorded with every run.
The saved loaded-model responses confirm zero VRAM use, and the runner process records confirm four allowed CPU cores.
The compute nodes use AMD EPYC 7713 processors and Linux x86_64.

These are CPU transport and timing checks using the original Ollama weight formats.
They do not satisfy the separate CPU float32 qualification profile.
Every GSM8K and MMLU-Pro sample has an absent quality score, so successful execution does not establish dataset accuracy.
The models also ran on different host snapshots, and these results are not a shared model-ranking cohort.
They establish no Raspberry Pi compatibility or performance claim.

## Retained evidence

The original experiment remains at `/scratch/qnguye19/qit-issue81-20261013` on Sol.
It retains the submitted scripts, source archive and checksum, model manifests and digests, setup checks, scheduler logs, per-model SQLite history, JSONL exports, CSV summaries, and allocation records.
The source archive records base revision `48f1ed84c8f848582895dd10605386f4ebff7b15`; its separate checksum identifies the submitted snapshot, which included then-uncommitted changes.

The three completed result directories are:

- `results/65700355-qit81-qwen25-05b`
- `results/65700356-qit81-smollm2-135m`
- `results/65695474-qit81-smollm2-360m`

The first two directory names retain scheduler-assigned task job IDs; the array task identities are listed above.
Each directory contains a completion marker and three successful outcomes.
Scheduler accounting independently reports exit code zero for setup job `65695473` and all tasks of array `65695474`.
The local inspection copy is under `.scratch/sol-recovery-20261016/previous` and excludes SQLite contents and verbose logs.

## Completed float32 qualification

The isolated experiment at `/scratch/qnguye19/qit-cpu-20261016` used repository revision `aeea343`, Python 3.12.14, CPU-only PyTorch 2.9.1, torchvision 0.24.1, and the provider's pinned model dependencies.
Setup job `65784478` completed successfully after source-checksum verification, Python protocol tests, Rust formatting, workspace tests, clippy, and executable-checksum retention.
The resolved Python environment is retained in `setup/python-lock.txt`.

Array `65784479` completed all forty-three tasks on CPU-only Zen 3 nodes, with four cores and sixteen GiB allocated per task and at most two concurrent checks.
Forty-two models passed and one failed during model acquisition.
Every passing record was independently checked for its pinned checkpoint revision, complete loaded CPU float32 parameter audit below one billion, saved successful q.it sample, and provider-process RSS.

| Task | Passed | Selected |
|---|---:|---:|
| Text generation | 6 | 6 |
| Embeddings | 9 | 10 |
| Reranking | 6 | 6 |
| Image to text | 8 | 8 |
| Speech to text | 8 | 8 |
| Text to image | 5 | 5 |

The failed checkpoint was `google/embeddinggemma-300m` at its pinned revision.
The publisher endpoint returned HTTP 401 for its gated repository because the Sol environment lacks authenticated approved access.
This is an access prerequisite, not evidence of a CPU inference incompatibility.
The failed attempt and server diagnosis remain retained, and it is excluded from successful results.
No credentials were transferred from another device.

These checks used the existing synthetic one-case profile, no warmup, one measured pass, four threads, and a sixteen-token generation limit.
Image generators used the explicitly reduced 64 by 64, two-step loading profile.
Pinned publisher code was explicitly enabled where required.
The results establish loading and transport across all six tasks, without official dataset-quality scores or native Raspberry Pi qualification.

Summary job `65785142` completed successfully and wrote `summary.json` and `summary.md` in the experiment root.
The scripts, source archive and checksum, per-model logs and metadata, SQLite state, JSONL exports, allocation records, and setup checks remain on Sol.
The local compact inspection copy is under `.scratch/sol-recovery-20261016/completed`.
The original experiment was not overwritten.

## Fresh installation and remaining work

The installed-package check passed on local macOS ARM64, including benchmark discovery, HTTP health, and the embedded dashboard and JavaScript assets from a temporary installation outside the checkout.
The first direct CPU-only Sol check exposed a verification-script bug: packaging honored an external Cargo build directory, while archive extraction assumed the checkout's default directory.
The script now gives packaging its own temporary build directory and reads the archive from that same location.
CI now runs the installed-package check with an external build directory to retain that regression coverage.
The corrected Sol check uses a separately retained source archive and patch under `/scratch/qnguye19/qit-install-20261016`.
Direct allocation `65789529` completed successfully on a CPU-only Zen 3 node.
The installed command discovered bundled benchmarks, served a healthy HTTP endpoint, and delivered its embedded dashboard and JavaScript assets from outside the source checkout.
Allocation evidence, the source archive and patch, install log, and passing result marker remain with that experiment.
A separate direct CPU-only allocation, `65789846`, also passed a real SmolLM2 135M request through a freshly installed q.it executable, then repeated the dashboard checks.
The provider used its separately installed optional environment and previously acquired pinned model cache.
Its retained audit confirms 134,515,008 CPU float32 parameters and four threads, with the successful sample and JSONL export under `real-model/`.
The verification runner and its checksum, allocation evidence, install log, and passing marker remain in the install experiment.
This verifies the documented separated install and serving path on Sol; the new guided setup workflow itself remains implementation work.

The user selected guided setup for q.it and optional provider dependencies, with model acquisition remaining explicit.
The immediate device target is a CPU-only ASU Sol allocation.
The [fresh-device and model-qualification wayfinder map](https://github.com/QuaanNguyen/q.it/issues/90) tracks guided setup, the new candidate batches, runtime and license decisions, scoring gaps, and qualification evidence.
Gated-access recovery, dataset-backed scoring, and native Raspberry Pi qualification remain incomplete.
