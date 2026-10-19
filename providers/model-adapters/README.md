# Optional model adapters

Install one runtime in its own environment.
You install the original model files yourself, separately from q.it.
Installing q.it, opening its interfaces, and starting an adapter never downloads a model.

```bash
python3.12 -m venv .venv-adapter
.venv-adapter/bin/pip install uv==0.9.5
.venv-adapter/bin/uv pip sync --python .venv-adapter/bin/python --torch-backend cpu --require-hashes providers/model-adapters/qit_adapters/locks/gliclass-linux-aarch64-py312.txt
.venv-adapter/bin/uv pip install --python .venv-adapter/bin/python --no-deps ./providers/model-adapters
.venv-adapter/bin/qit-adapter list
.venv-adapter/bin/qit-adapter verify --model knowledgator/gliclass-v4-edge --model-path /absolute/path/to/pinned-model
.venv-adapter/bin/qit-adapter serve --model knowledgator/gliclass-v4-edge --model-path /absolute/path/to/pinned-model
```

Use the `linux-x86_64` lock on Sol.
The retained locks require Linux, CPython 3.12, and a matching 64-bit architecture.
Most require glibc 2.28 or later; Photon and Redux on ARM64 require glibc 2.34 or later.
UIBK has no retained Linux ARM64 environment because its pinned FastPLAID release has no compatible wheel.
A resolved dependency lock is not evidence of a working native installation or Raspberry Pi execution.

Run a decision smoke pack from another terminal:

```bash
qit benchmark install providers/model-adapters/packs/english-choice
qit run english-choice-smoke/choice --provider local-cpu --model knowledgator/gliclass-v4-edge --base-url http://127.0.0.1:8766 --warmups 0 --iterations 1
qit tui
```

These three original cases test transport and scoring, without a dataset-quality claim.
Use the versioned `russian-choice` pack for FRIDA.
The FRIDA adapter rejects non-Russian states; it must not be scored on English packs.
An Agent UIBK pack must retain the publisher's reasoning-and-query format.
OCR routes require an OCR pack whose prompt is `<OCR>`; chat and VQA packs are rejected.
Speech qualification needs actual spoken audio, such as the pinned LibriSpeech case retained with the Sol evidence, rather than the core tone transport case.

## Explicit artifact staging

`qit-adapter verify` compares every required file against the exact revision and byte identity in the installed manifest.
Missing files report `model not installed` before importing a runtime.
Symlinks, modified artifacts, unsafe paths, unexpected Indra graphs, and incompatible environments fail without a fallback.

The separate repository helper prints an acquisition plan by default:

```bash
.venv-adapter/bin/python scripts/stage-optional-model.py --model knowledgator/gliclass-v4-edge --output /absolute/path/to/pinned-model
```

Adding `--download` explicitly acquires those original pinned files and verifies them.
This helper is outside the benchmark runner and serving lifecycle.
An authentication token, if required, comes from the environment and is never written to results.
Normal serving sets offline and telemetry-disable settings before importing model code, rejects external Python socket connections, and uses direct local publisher loading.
Native network behavior still requires a traced qualification run.

## Adapter status

Every new checkpoint is experimental and remains unqualified until its real installed-command, publisher-parity, CPU audit, and appropriate benchmark checks pass.
No candidate checkbox is completed by dependency resolution or fixture tests.

| Runtime extra / lock | Models | Method and current boundary |
| --- | --- | --- |
| `gliclass` | GLiClass v4 edge | Native complete single-label probabilities. |
| `gliner2` | GLiNER 2.5 Decide | Observes the pinned runtime's complete native classification probabilities before public top-choice decoding. |
| `intern` | Intern Decision 0.8B | Native calibrated text decisions; image execution remains unsupported. |
| `liquid` | d1 omni 600M | Native text decisions with the full loaded stack; image and audio decisions remain unsupported. |
| `d3` | d3 edge, then lite | Separately admitted native BF16 backbone with float32 readout; full text and image choice distributions. |
| `typed-gemma` | embeddinggemma 2 typed decision | Native single-choice `decide()` with complete probabilities. |
| `frida` | FRIDA Decisions | Russian single-choice PyTorch path; pinned and hashed original publisher source archive. |
| `lighton` | LightOnOCR 3 0.8B | Publisher OCR prompt and deterministic transcription. |
| `ultrabert` | UltraBERT Embed 150M | Publisher CLS pooling, query instruction, normalized 768-dimensional embeddings. |
| `uibk` | UIBK embedding v1, Agent UIBK v1 | Full PyLate token vectors and native MaxSim, with all branches retained. |
| `photon` | Parakeet Ultra, Redux | Direct Kestrel full-utterance TDT; Redux retains native GEMM8 panels, ternary logical shapes, scales, and activation quantization. |
| `indra` | Indra OCR | Torch-free full ten-graph ONNX pipeline, including math; conservative full static-constant bound, with exact learned count explicitly unknown. |
| `nemotron` | Nemotron 3.5 ASR streaming 0.6B | Separate NeMo full-utterance route, pinned archive, English language prompt, decoder, chunk and cache identity. |
| Pending | Whistle | The inspected native interface cannot verify effective decoder depth or the complete logical packed-weight audit. |

The NeMo environment is an independent copy of the compatible #112 lock.
It does not change the QVAC profiles or their recorded validation.
Whistle never invokes a wrapper that downloads an engine or sends publisher telemetry.
Its pinned artifact manifest is retained, but serving remains a precise unsupported path under #87.

## Source review and identity

Intern, Liquid, d3, typed Gemma, UltraBERT, and both UIBK models execute pinned publisher Python source.
Review those exact files before adding `--allow-reviewed-code` to `serve`.
Their manifest includes file sizes, original Git blob or LFS identities, and explicit SHA-256 identities for reviewed Python files.
Changing a revision or a source file requires a new review and qualification.

The result metadata retains model and source artifacts, the dependency lock digest, installed distribution record digests, package versions, native preparation, loaded tensor device and precision, and original license links.
FRIDA additionally validates the pinned source archive's installed provenance.
NeMo inventories and hashes archive members and matches the restored tokenizer back to an archive member.
The Indra audit walks nested graphs, sparse tensors, tensor attributes, scalar/list constants, and model-local functions, and rejects unresolved attribute references, external tensor data, quantized operators, and unsupported constant precision.
Its bound includes duplicated exports and fixed constants, so it establishes a conservative sub-billion ceiling without claiming an exact learned-parameter count.

Native d3 BF16 and Redux quantized runs have distinct precision and configuration identities.
Neither is relabelled as float32.
Indra output metadata retains the original modified OpenRAIL license link, research scope, attribution, change disclosure, and applicable output obligations accepted by the owner.
Model weights and native engine binaries are never included in this package.

## Retained qualification evidence

```bash
python3 scripts/qualify-optional-adapter.py --adapter /absolute/path/to/qit-adapter --qit /absolute/path/to/qit --model knowledgator/gliclass-v4-edge --model-path /absolute/path/to/pinned-model --pack providers/model-adapters/packs/english-choice --benchmark english-choice-smoke/choice --output /absolute/path/to/new-evidence-directory
```

The helper starts the already installed adapter, records metadata, runs the installed q.it command with the provider PID, and retains logs, the complete state database, every run and sample row, and successful JSONL exports.
Startup, import, model-audit, and inference failures remain in its evidence directory.
A separately produced publisher-native reference can be supplied with `--native-reference`; no parity is claimed when it is absent.
Full transcripts, audio duration, native speech routing, and all accepted retrieval token vectors survive history and exports.
Oversized retrieval observations fail rather than being truncated or collapsed to one vector.
Generation throughput and first-token metrics remain absent for these non-generation tasks.
