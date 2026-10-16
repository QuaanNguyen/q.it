# Publisher CPU routes for Moondream Parakeet and QVAC TranscribePsy

Research date: October 16, 2026.
This note answers [Decide portable CPU runtimes for the new speech checkpoints](https://github.com/QuaanNguyen/q.it/issues/91) from public publisher cards, source, and package metadata.
No models were downloaded, packages installed, or inference executed.
A source-supported route and a published platform wheel do not establish a working installation or a qualified Raspberry Pi.

## Result

The existing NeMo implementation ticket combines two different runtimes.
QVAC publishes a direct NeMo CPU float32 path, while Moondream explicitly publishes Photon, whose underlying engine is Kestrel. [QVAC inference][qvac-infer], [Moondream Redux card][redux-card], [Moondream Ultra card][ultra-card], [Kestrel weight loader][kestrel-weights]
Ultra has a CPU float32 loading route in publisher source, although its published accuracy and throughput measurements use a GPU. [Ultra card][ultra-card], [Kestrel weight loader][kestrel-weights]
Redux remains ternary on CPU even when activations use float32, so it cannot silently enter the current float32 comparison setup. [Redux manifest][redux-manifest], [Kestrel source distribution][kestrel-source]

| Candidate | Immutable publisher revision | Documented route | Admission boundary |
| --- | --- | --- | --- |
| moondream/parakeet-redux | `2bf128600aac4b16946f7ed8372e56117fe5e23b` | Photon CPU with packed ternary weights | Owner precision decision required |
| moondream/parakeet-ultra | `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` | Photon/Kestrel; source accepts explicit CPU float32 | Clean installation and actual execution remain unverified |
| qvac/TranscribePsy-250M-EU | `49d2b21e72018798a755c0cbe5ebb4fefe2384a1` | NeMo CPU float32, explicit language routing | Audit adapters and optional learned gate separately |
| qvac/TranscribePsy-250M-EU-sub | `ccf70de9ff9848f5cd5975adda1370ce4addf6f1` | NeMo CPU float32 | One trunk, no language adapters or gate |

Revisions and published tensor summaries were read from the original [Redux metadata][redux-api], [Ultra metadata][ultra-api], [QVAC EU metadata][qvac-api], and [QVAC EU-sub metadata][sub-api].
The model cards mark Moondream as CC-BY-4.0 and QVAC as Apache-2.0; QVAC also records attribution to its CC-BY-4.0 Parakeet teacher. [Redux card][redux-card], [Ultra card][ultra-card], [QVAC card][qvac-card], [QVAC EU-sub card][sub-card]

## QVAC integration and complete-stack audit

The publisher restores a local NeMo archive with explicit CPU placement, converts it to float32, switches to evaluation mode, and invokes transcription under inference mode. [QVAC inference][qvac-infer], [EU-sub inference][sub-infer]
Its decoding helper selects greedy batch decoding unless beam decoding was already configured, disables the CUDA graph decoder by default, and sets fused batch size to minus one. [QVAC inference][qvac-infer]
Implementation should record the actual decoder configuration and choose one fixed comparison configuration rather than inherit whatever a restored archive happens to contain.

QVAC EU's English path loads the trunk alone.
Non-English audio requires six residual CAFE adapters attached at encoder layers 3, 7, 11, 15, 19, and 23; choosing the wrong route silently changes accuracy. [QVAC inference][qvac-infer]
The English LibriSpeech route can therefore explicitly declare English and omit both adapters and the optional gate.
EU-sub supports its eight languages with the same trunk and no routing. [EU-sub inference][sub-infer]
Automatic EU routing adds a learned language gate, performs a preliminary trunk pass, and conditionally attaches the adapters for the transcription pass. [Gated inference][qvac-gated]
That is a separate model configuration and its routing work belongs inside invocation timing.

Both Hub summaries list 253,954,838 float32 elements for the trunk, which is not the EU multilingual full-stack count. [QVAC EU metadata][qvac-api], [QVAC EU-sub metadata][sub-api]
The card reports approximately 0.50M additional adapter parameters and another approximately 0.50M for the optional gate. [QVAC card][qvac-card]
Exact sidecar totals remain to be verified from tensor metadata during qualification; no sidecar weights or headers were fetched in this research.
Count the unique trunk parameters plus every adapter used by the configured route and, when enabled, the learned gate.
Do not count the NeMo archive and its safetensors copy twice.

The publisher retains adapter modules in an ordinary Python list and installs them through forward hooks. [Adapter helper][qvac-adapter]
Consequently, iterating only the restored trunk's registered parameters will miss the adapters.
The provider must audit the trunk, adapter modules, and gate explicitly, checking CPU placement and float32 throughout.
It must also reject missing hooks or incompatible feature shapes, because the publisher helper's shape guard can skip an adapter without raising. [Adapter helper][qvac-adapter]

The publisher supplies checksums for the archive, sidecars, and inference source. [QVAC checksum manifest][qvac-sha], [EU-sub checksum manifest][sub-sha]
Pin source revision and individual file digests before startup, retain tokenizer and preprocessor identity from the restored archive, and record the active route and every learned component in provider metadata.

## Optional dependency environments and platform boundary

Keep NeMo ASR and Photon/Kestrel in separate environments from the existing Transformers provider and from each other.
The current q.it provider pins Torch 2.9.1 and Transformers 4.57.6, while NeMo ASR adds a substantial independent dependency graph and Photon pins native extension packages. [q.it package manifest][qit-package], [NeMo package metadata][nemo-meta], [Moondream package metadata][md-meta], [Kestrel package metadata][kestrel-meta]

A concrete initial NeMo qualification candidate is CPython 3.11, NeMo ASR 3.0.0, CPU Torch 2.9.1, matching Torchaudio 2.9.1, and the existing float32 audio preparation packages.
This is a proposed direct-dependency baseline, not a solved or tested lock.
NeMo 3.0.0 requires Python at least 3.10 and Torch at least 2.6; its ASR extra brings librosa, lhotse, sentencepiece, Hydra, Lightning, and other packages, and non-macOS base metadata also requires CUDA bindings. [NeMo package metadata][nemo-meta]
Do not request NeMo's CUDA, compiled, TTS, or all-extras installations for this CPU task.
Resolve the complete transitive graph and retain its hashes before declaring the optional setup supported.
An import or a dependency requirement involving CUDA does not by itself prove that CPU transcription is impossible or that a GPU was used.

| Route | Linux x86_64 evidence | Linux ARM64 evidence | Honest current status |
| --- | --- | --- | --- |
| QVAC NeMo | Publisher script explicitly selects CPU float32; universal NeMo wheel and matching Torch/Torchaudio wheels exist | Matching Torch/Torchaudio ARM64 wheels exist; NeMo's universal wheel alone does not establish all ASR dependencies or Pi compatibility | Candidate for Sol validation; ARM64 full-stack qualification pending |
| Photon/Kestrel | Redux card names x86 CPU support; native and kernel wheels exist | Redux card names ARM NEON; required native packages publish aarch64 wheels | CPU route present in publisher source; actual installation and execution pending |

Wheel evidence comes from [Torch 2.9.1 metadata][torch-meta], [Torchaudio 2.9.1 metadata][audio-meta], [Kestrel Native 0.1.8 metadata][native-meta], and [Kestrel Kernels 0.7.4 metadata][kernels-meta].
The Python 3.11 Torch and Torchaudio Linux wheels are tagged manylinux 2.28 for both architectures.
Kestrel Native has a manylinux 2.28 ARM64 wheel, but Kestrel Kernels and its required ARM64 bundles are tagged manylinux 2.34/2.35, which excludes older glibc installations from this prebuilt route. [Native metadata][native-meta], [Kernel metadata][kernels-meta], [Bundle A metadata][bundle-a], [Bundle B metadata][bundle-b]
A package's wheel tag is an installation boundary, not evidence of instruction-set compatibility or successful inference on every CPU of that architecture.
Record glibc, CPU instruction features, kernel, architecture, Python ABI, and package hashes in each qualified environment.

## Photon facts that change the plan

Moondream 2.6.1 pins Kestrel 0.9.1.
Kestrel 0.9.1 pins Kestrel Native 0.1.8 and Kestrel Kernels 0.7.4 and requires Torch C DLPack Extension at least 0.1.3. [Moondream package metadata][md-meta], [Kestrel package metadata][kestrel-meta]
The inspected source archive SHA-256 is `52083e88c71ccd6023096714033403d73a7cfb7a83b72493509bfb09163f344a`. [Kestrel source distribution][kestrel-source]

In that archive, the Parakeet weight loader accepts a local checkpoint directory and explicit CPU and float32 arguments, strictly loads the checkpoint, attaches its learned VAD head when present, and casts full-precision weights to the requested dtype.
The runtime passes its configured device and dtype into that loader and accepts explicit CPU thread limits. [Kestrel weight loader][kestrel-weights], [Kestrel CPU runtime][kestrel-runtime]
Thus Ultra has a source-backed CPU float32 implementation route, but the model's config and architecture alone are insufficient proof that the existing Transformers provider can load it.
Its config names a development Transformers 5.6 version, newer than q.it's pinned 4.57.6. [Ultra config][ultra-config], [q.it package manifest][qit-package]
The Ultra Hub metadata reports 627,270,663 stored F16/F32 tensor elements; the loaded audit must still reconcile learned parameters, buffers, and its VAD head. [Ultra metadata][ultra-api], [Kestrel weight loader][kestrel-weights]
Do not silently convert it to NeMo or substitute an upstream NVIDIA checkpoint.

For Redux, the manifest declares 603,979,776 logical ternary weights and 23,290,911 dense entries, totaling 627,270,687 before reconciling scale, buffer, and learned-parameter classifications. [Redux manifest][redux-manifest]
The Hub's 148,923,399 total includes 120,913,920 packed U8 elements and is a storage-element count, not a complete logical parameter audit. [Redux metadata][redux-api]
Qualification must reconcile the manifest against loaded logical shapes, learned scales, decoder, front end, and the 213,377-parameter VAD head, without double counting components already included in a manifest total. [Redux manifest][redux-manifest]
Do not relabel ternary weights as float32 because the config advertises a floating activation dtype.

Photon's ordinary client creates a runtime that can automatically fetch model files and report usage telemetry.
Its own checkpoints default to a moving main revision in the loader, and the package documentation does not provide a telemetry opt-out in its listed environment settings. [Moondream source distribution][md-source], [Kestrel source distribution][kestrel-source]
A pinned local checkpoint path avoids moving model identity, but the new provider design must still establish an offline execution route and account for telemetry before measurement.
The publisher's lower-level loader, feature preparation, and generation code are available for inspection, so this can be investigated without implementing a new model architecture. [Kestrel weight loader][kestrel-weights], [Kestrel Parakeet model][kestrel-model]

## Concrete implementation and diagnostic requirements

For QVAC, add an optional NeMo ASR backend behind the existing loopback provider protocol, retain the existing speech task and inert benchmark packs, and decode q.it audio to 16 kHz mono before transcription.
Return the actual transcript and audio duration through the existing result event.
The present local provider exposes its own model and invocation routes, so the implementation ticket's mention of a transcription endpoint should not imply that q.it currently offers an OpenAI transcription server. [q.it server][qit-server], [q.it ASR backend][qit-backend]

Persist runtime identity, archive and sidecar digests, inference-source revision, route, decoder settings, thread configuration, full-stack audit, and the complete environment lock.
Keep initialization and explicit model acquisition outside measured requests.
Batch ASR does not provide streamed first-token timing through this route; retain absent metrics unless a real protocol measurement supplies them.

Report a precise unsupported path for a missing optional runtime, incompatible Python/glibc/architecture, failed native import, absent local artifact, missing adapter or gate, incompatible feature shape, non-float32 learned component, and unsupported decoding setting.
Preserve the failure stage and original diagnostic rather than attempting CUDA, automatic quantization, another model, or a hidden download.
Actual CPU smoke transcription, a real LibriSpeech case, installed-package verification, memory use, and native ARM64 behavior remain execution work.

## Map implications

The facts resolve the runtime-family decision: rewrite [Add a CPU ASR provider path for NeMo-style Parakeet checkpoints](https://github.com/QuaanNguyen/q.it/issues/85) around QVAC EU and EU-sub, including explicit English, adapter, and optional gate configurations.
Remove Moondream as its assumed NeMo dependency.
Create a separate question ticket for a pinned, offline Photon/Kestrel CPU integration, including telemetry behavior and complete native dependency locks, before a Photon implementation ticket is considered ready.
That question can build on the source-backed Ultra float32 loader found here; no new architecture research is required to begin it.
Extend [Decide admission policy for exceptional model candidates](https://github.com/QuaanNguyen/q.it/issues/95) to ask separately about Redux's native ternary precision, in addition to Whistle.
The owner must decide both exceptions explicitly; this research grants neither one.
QVAC's supported-language routing and the corrected Redux count are facts that can update the catalog tickets now.
All runtime passes, exact sidecar counts, offline behavior, and Raspberry Pi qualification remain unverified.

[qvac-card]: https://huggingface.co/qvac/TranscribePsy-250M-EU/blob/49d2b21e72018798a755c0cbe5ebb4fefe2384a1/README.md
[qvac-infer]: https://huggingface.co/qvac/TranscribePsy-250M-EU/blob/49d2b21e72018798a755c0cbe5ebb4fefe2384a1/inference.py
[qvac-gated]: https://huggingface.co/qvac/TranscribePsy-250M-EU/blob/49d2b21e72018798a755c0cbe5ebb4fefe2384a1/inference_gated.py
[qvac-adapter]: https://huggingface.co/qvac/TranscribePsy-250M-EU/blob/49d2b21e72018798a755c0cbe5ebb4fefe2384a1/eval/adapter_wrapper.py
[qvac-api]: https://huggingface.co/api/models/qvac/TranscribePsy-250M-EU/revision/49d2b21e72018798a755c0cbe5ebb4fefe2384a1
[qvac-sha]: https://huggingface.co/qvac/TranscribePsy-250M-EU/blob/49d2b21e72018798a755c0cbe5ebb4fefe2384a1/SHA256SUMS
[sub-card]: https://huggingface.co/qvac/TranscribePsy-250M-EU-sub/blob/ccf70de9ff9848f5cd5975adda1370ce4addf6f1/README.md
[sub-infer]: https://huggingface.co/qvac/TranscribePsy-250M-EU-sub/blob/ccf70de9ff9848f5cd5975adda1370ce4addf6f1/inference.py
[sub-api]: https://huggingface.co/api/models/qvac/TranscribePsy-250M-EU-sub/revision/ccf70de9ff9848f5cd5975adda1370ce4addf6f1
[sub-sha]: https://huggingface.co/qvac/TranscribePsy-250M-EU-sub/blob/ccf70de9ff9848f5cd5975adda1370ce4addf6f1/SHA256SUMS
[redux-card]: https://huggingface.co/moondream/parakeet-redux/blob/2bf128600aac4b16946f7ed8372e56117fe5e23b/README.md
[redux-api]: https://huggingface.co/api/models/moondream/parakeet-redux/revision/2bf128600aac4b16946f7ed8372e56117fe5e23b
[redux-manifest]: https://huggingface.co/moondream/parakeet-redux/blob/2bf128600aac4b16946f7ed8372e56117fe5e23b/ternary.json
[ultra-card]: https://huggingface.co/moondream/parakeet-ultra/blob/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b/README.md
[ultra-api]: https://huggingface.co/api/models/moondream/parakeet-ultra/revision/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b
[ultra-config]: https://huggingface.co/moondream/parakeet-ultra/blob/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b/config.json
[nemo-meta]: https://pypi.org/pypi/nemo_toolkit/3.0.0/json
[md-meta]: https://pypi.org/pypi/moondream/2.6.1/json
[kestrel-meta]: https://pypi.org/pypi/kestrel/0.9.1/json
[native-meta]: https://pypi.org/pypi/kestrel-native/0.1.8/json
[kernels-meta]: https://pypi.org/pypi/kestrel-kernels/0.7.4/json
[bundle-a]: https://pypi.org/pypi/kestrel-kernels-bundle-a/0.7.4/json
[bundle-b]: https://pypi.org/pypi/kestrel-kernels-bundle-b/0.7.4/json
[torch-meta]: https://pypi.org/pypi/torch/2.9.1/json
[audio-meta]: https://pypi.org/pypi/torchaudio/2.9.1/json
[kestrel-source]: https://files.pythonhosted.org/packages/6d/26/d6a3530617148deff50f92323425601b31006393d3b9f74e1cd3340a5792/kestrel-0.9.1.tar.gz
[md-source]: https://files.pythonhosted.org/packages/a0/9d/5b6b3b8fe0f78698a53c40f205eaa9c475b83f42ceab369f7af0bb4a5a52/moondream-2.6.1.tar.gz
[qit-package]: https://github.com/QuaanNguyen/q.it/blob/aeea34321998f04b313b0ca82382b3db32da4809/providers/local-cpu/pyproject.toml
[qit-backend]: https://github.com/QuaanNguyen/q.it/blob/aeea34321998f04b313b0ca82382b3db32da4809/providers/local-cpu/qit_cpu/backend.py
[qit-server]: https://github.com/QuaanNguyen/q.it/blob/aeea34321998f04b313b0ca82382b3db32da4809/providers/local-cpu/qit_cpu/server.py

[kestrel-weights]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/weights.py
[kestrel-runtime]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/runtime.py
[kestrel-model]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/model.py
