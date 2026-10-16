# QVAC NeMo CPU provider

The optional NeMo provider serves the original QVAC EU and EU-sub checkpoints through q.it's existing Local CPU speech protocol.
It uses a separate environment and command so installing NeMo does not replace the existing Transformers provider stack.
Its retained dependency lock currently targets Linux x86_64, CPython 3.12, and glibc 2.28 or later.
Other platforms receive an explicit unsupported-platform diagnostic; native Raspberry Pi ARM64 qualification remains separate.

## Install the isolated environment

From a source checkout, install the complete hashed CPU dependency set and then the provider wheel:

```bash
uv venv --python 3.12 .venv-nemo
uv pip sync --python .venv-nemo/bin/python --require-hashes --torch-backend cpu providers/local-cpu/qit_cpu/locks/nemo-linux-x86_64-py312.txt
uv build providers/local-cpu --wheel --out-dir /tmp/qit-nemo-wheels
uv pip install --python .venv-nemo/bin/python --no-deps /tmp/qit-nemo-wheels/qit_local_cpu-0.1.0-py3-none-any.whl
.venv-nemo/bin/qit-nemo list
```

The optional package extra pins NeMo ASR 3.0.0 and matching Torchaudio 2.9.1 alongside the existing Torch 2.9.1 base dependency.
The generated lock selects CPU-only Torch and Torchaudio wheels and pins every resolved transitive dependency with artifact hashes.
Serving validates the installed versions against that lock and records dependency identities, installed wheel-record digests, provider source digests, Python ABI, architecture, kernel, and libc.
The existing model extra and its Transformers pins remain unchanged.

Regenerate the lock only when deliberately qualifying a new environment:

```bash
uv pip compile providers/local-cpu/pyproject.toml --extra nemo --python-version 3.12 --python-platform x86_64-manylinux_2_28 --torch-backend cpu --generate-hashes --no-header -o providers/local-cpu/qit_cpu/locks/nemo-linux-x86_64-py312.txt
```

## Acquire models explicitly

Acquisition is a separate command and never happens during serving or a measured request:

```bash
.venv-nemo/bin/qit-nemo fetch --model qvac/TranscribePsy-250M-EU --route english --model-dir /path/to/qvac-eu
.venv-nemo/bin/qit-nemo verify --model qvac/TranscribePsy-250M-EU --route english --model-dir /path/to/qvac-eu
```

Each download uses the original publisher repository at an immutable revision and must match the pinned per-file SHA-256 digest before activation.
Existing files with incorrect digests are rejected rather than silently replaced.
Symbolic links and files escaping the model directory are rejected.
Only the NeMo archive is acquired for the trunk; its separate safetensors export is not downloaded or counted again.

| Checkpoint | Pinned revision | Supported configurations |
| --- | --- | --- |
| qvac/TranscribePsy-250M-EU | `49d2b21e72018798a755c0cbe5ebb4fefe2384a1` | English, multilingual, automatic |
| qvac/TranscribePsy-250M-EU-sub | `ccf70de9ff9848f5cd5975adda1370ce4addf6f1` | Shared trunk for its eight supported languages |

English EU uses the bare trunk.
Multilingual EU also loads the publisher's six residual adapters for explicitly non-English input.
Automatic EU loads those adapters and the learned language gate, runs the native preliminary gate pass inside the invocation, and records which route was selected.
EU-sub uses its shared trunk and has no adapters or learned gate.
A route incompatible with the selected checkpoint is rejected before downloading or loading anything.

To acquire the other configurations:

```bash
.venv-nemo/bin/qit-nemo fetch --model qvac/TranscribePsy-250M-EU --route multilingual --model-dir /path/to/qvac-eu
.venv-nemo/bin/qit-nemo fetch --model qvac/TranscribePsy-250M-EU --route automatic --model-dir /path/to/qvac-eu
.venv-nemo/bin/qit-nemo fetch --model qvac/TranscribePsy-250M-EU-sub --route shared --model-dir /path/to/qvac-eu-sub
```

## Start the provider

Publisher code is loaded only after verifying its pinned digest and receiving an explicit trust flag:

```bash
.venv-nemo/bin/qit-nemo serve --model qvac/TranscribePsy-250M-EU --route english --model-dir /path/to/qvac-eu --trust-publisher-code --threads 4 --port 8082
```

Run an already installed speech benchmark through q.it:

```bash
qit run my-speech-pack/my-benchmark --provider local-cpu --model qvac/TranscribePsy-250M-EU --base-url http://127.0.0.1:8082 --iterations 1 --warmups 1
```

Use the provider process identifier printed in its metadata with q.it's process telemetry option when collecting provider RSS on the same host.
The provider binds only to loopback, preserves optional API-key authentication, and rejects overlapping requests.
Model acquisition, restore, tokenizer preparation, dependency auditing, and component auditing occur before the endpoint becomes available.
Serving enables Hugging Face offline modes and blocks external Python socket connections; it does not initiate hidden model downloads.

On a shared cluster, a job-local Python bytecode cache can avoid slow writes to shared storage:

```bash
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/qit-nemo-pycache"
```

## Reproducibility and learned components

The provider restores the verified original archive onto CPU, uses float32 throughout the learned stack, enters evaluation mode, and invokes transcription under inference mode.
Decoding is fixed to greedy batch mode with CUDA graph decoding disabled and fused batch size minus one.
Unsupported decoding configuration fails startup rather than inheriting an archive's beam or accelerator settings.
The actual decoder and preprocessor configuration are retained in metadata.

Every parameter and buffer in the trunk, each loaded adapter, and the optional gate is audited for its name, shape, element count, dtype, device, and persistence.
The parameter limit counts unique learned parameters across all loaded components, including adapters held in ordinary Python lists and attached through hooks.
Persistent and nonpersistent buffers are reported separately from learned parameters.
The restored state-dictionary element count is reconciled with the publisher's safetensors headline, rather than treating that headline as a learned-parameter count.
Archive members retain per-file digests, and the restored tokenizer's serialized model must match a verified archive member.

Adapter feature guards reject shapes that the publisher helper would otherwise silently skip.
The provider verifies that all native hooks remain attached and records actual adapter-module execution counts for adapted requests.
Automatic routing includes the publisher's gate pass in invocation latency, preserves its native probability and threshold, and records the gate's audio window, which is at most twenty seconds.
The final transcription still receives the complete accepted audio case.
Route, pinned artifacts, decoder, tokenizer, preparation, provider source, and dependency lock contribute to a stable configuration identity, so dashboard comparisons retain separate English, multilingual, and automatic configurations.

## Audio, results, and diagnostics

Speech requests use inline audio and the existing inert benchmark packs.
Audio is decoded to finite float32 samples, downmixed to mono, resampled to 16 kHz, and limited to thirty seconds.
No generation-token budget truncates NeMo transcription.

A successful result returns the complete transcript, actual audio duration, and native routing evidence.
These values survive persistence, JSONL export, the run-detail response, and pack removal.
The dashboard exposes the full transcript and routing details, while unavailable generation-token counts, first-token timing, and token throughput remain absent.
Malformed durations and malformed routing evidence fail the invocation and remain in failed history.
Historical results without the new speech observation remain readable.

Startup diagnostics distinguish unsupported model or route, Python/platform/libc mismatch, missing or mismatched packages, missing or corrupt artifacts, unsafe archive paths, native imports, publisher imports, archive restore, decoder configuration, adapter/gate setup, tensor dtype/device, component counts, and tokenizer identity.
Inference diagnostics preserve gate, hook, feature-shape, audio, and decoder failures.
The provider does not switch checkpoints, runtimes, precision, or devices to hide a failure.

Synthetic tones validate execution and transport only.
A versioned spoken-audio case can establish a single transcription result, while dataset-wide quality claims require the separate scoring and qualification work.
Sol Linux x86_64 results do not establish Raspberry Pi behavior.

The [execution evidence](research/qvac-nemo-cpu-validation.md) records the clean packaged installation and real English/German CPU checks for both original checkpoints and all four configurations.

The [publisher-route audit](https://github.com/QuaanNguyen/q.it/blob/a55dd5d14fa03a82adb0ba557ecf2f27a3824edb/docs/research/parakeet-qvac-cpu-routes.md) retains the original source-backed rationale.
The [QVAC EU source](https://huggingface.co/qvac/TranscribePsy-250M-EU/tree/49d2b21e72018798a755c0cbe5ebb4fefe2384a1), [EU-sub source](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub/tree/ccf70de9ff9848f5cd5975adda1370ce4addf6f1), and [NeMo release metadata](https://pypi.org/project/nemo-toolkit/3.0.0/) identify the upstream artifacts used here.
