# Pinned offline Photon CPU integration

Research date: October 16, 2026.
This note resolves the design question in [Decide pinned offline Photon CPU integration](https://github.com/QuaanNguyen/q.it/issues/101), following the [publisher-route audit](parakeet-qvac-cpu-routes.md).
Only existing publisher source and public package metadata were inspected.
No runtime, native engine, model payload, package installation, or inference was executed or downloaded for this follow-up.

## Selected route

Wrap the publisher's lower-level Kestrel Parakeet implementation in q.it's optional CPU provider, using an isolated environment and staged local Ultra files.
Use the publisher's loader, feature extraction, generation, and tokenizer directly; this requires no reimplementation of the model.
The source-backed route is:

1. Validate that an absolute local checkpoint directory exists and contains the pinned configuration, tokenizer, and weights before importing or loading the runtime.
2. Call `kestrel.models.parakeet_tdt.weights.load_parakeet_tdt` with that directory, `device="cpu"`, `dtype=torch.float32`, and `local_files_only=True`.
3. Pass q.it's prepared 16 kHz mono float32 samples through `kestrel.models.parakeet_tdt.features.parakeet_features`.
4. Under Torch inference mode, call the loaded model's `generate` with the resulting features and attention mask and an explicit output-token limit.
5. Slice the first output sequence using its returned length and call the loaded tokenizer's `decode` on those token IDs.

The exact loader accepts an existing directory without calling Hub download machinery, strictly loads the tensor state, attaches the checkpoint's VAD head when present, and casts dense weights to the requested CPU dtype. [Weight loader][weights], [Checkpoint resolver][resolver]
The feature helper requires at least 320 samples and computes the publisher's 128-bin features; the model's generation output includes sequences and lengths, and its tokenizer removes blank and pad tokens. [Feature source][features], [Model source][model], [Tokenizer source][tokenizer]

Limit this initial provider path to q.it's existing bounded audio cases, currently at most thirty seconds, and one complete utterance per request. [q.it audio preparation][qit-media]
This does not promise Photon streaming, long-file segmentation, word timestamps, translation, language forcing, or exact equivalence with the higher-level orchestration.
Those behaviors are outside this narrow integration.
Keep model loading, validation, and warmup outside measurement; include feature preparation and generation in invocation timing.
Audit every loaded learned component, including VAD, for CPU placement, float32, and the sub-billion boundary before reporting readiness.

## Avoiding telemetry and acquisition

Do not instantiate the Moondream Photon client, Kestrel `RuntimeConfig`, or `InferenceEngine` for this route.
The ordinary client constructs the engine, and a configuration without an explicit local model path can trigger acquisition. [Moondream source distribution][md-source], [Runtime configuration][config]
Engine initialization starts a background Hub configuration probe and constructs a Photon telemetry reporter; omitting an API key does not disable this behavior. [Kestrel 0.9.1 source distribution, engine/core.py][package-source]
The reporter performs network requests during validation and periodic flushes. [Reporter source][reporter]
No supported telemetry opt-out switch was identified in these inspected entry points.

Avoid the telemetry-owning objects instead of mutating their private state, patching network calls, or assuming offline cache flags disable reporting.
The selected model loader, feature helper, model generation methods, and tokenizer contain no reporter construction or remote inference path. [Weight loader][weights], [Feature source][features], [Model source][model], [Tokenizer source][tokenizer]
Importing a Kestrel submodule still imports its package initializer, which imports engine classes, so the installed-package acceptance check must verify import and native-library initialization with outbound networking unavailable. [Package initializer][initializer]
Python source inspection does not establish the behavior of every compiled dependency.

Stage Ultra revision `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` explicitly and verify all three file digests.
The original model-weight SHA-256 is `c9608f36d0ab956c14bfcc525479b0746b3b42a56f6949ec85c14eb7466717dc`. [Publisher artifact metadata][ultra-tree]
Reject a missing local path before calling the loader, and reject a ternary manifest in this float32 Ultra configuration.
Use Hub offline mode as an additional guard against accidental Hub access, while retaining a direct local path and explicit local-files-only loading.
A missing artifact must fail with its identity and preparation step, without downloading it during provider startup or a benchmark.

## Sol glibc 2.28 and native packages

The initial version baseline is Kestrel 0.9.1 with CPU Torch 2.9.1, Kestrel Native 0.1.8, Kestrel Kernels 0.7.4, and Torch C DLPack Extension 0.1.3.
Moondream 2.6.1 also pins this Kestrel version but is unnecessary for the selected low-level entry point. [Kestrel metadata][package-meta], [Moondream metadata][md-meta]
Kestrel Kernels additionally requires bundles A and B on Linux and bundles C through H on Linux x86_64; these are mandatory package dependencies even for a CPU invocation. [Kernel metadata][kernels]
Do not bypass those requirements with a partial installation.

For the proposed CPython 3.11 x86_64 environment, each mandatory native package has a compatible wheel-tag route for Sol's reported glibc 2.28:

| Package | Applicable published Linux wheel tag |
| --- | --- |
| Torch 2.9.1 | manylinux 2.28 x86_64 |
| Kestrel Native 0.1.8 | manylinux 2.17 x86_64 |
| Torch C DLPack Extension 0.1.3 | manylinux 2.24 / 2.28 x86_64 |
| Kestrel Kernels 0.7.4 | manylinux 2.24 / 2.31 x86_64 |
| Each Kestrel Kernels bundle A through H, 0.7.4 | manylinux 2.24 / 2.31 x86_64 |

These are alternative compatibility tags on published wheels, so the presence of a second 2.31 tag does not impose a 2.31 minimum when the same wheel advertises 2.24. [Torch metadata][torch], [Native metadata][native], [DLPack metadata][dlpack], [Kernel metadata][kernels], [Bundle A][bundle-a], [Bundle B][bundle-b], [Bundle C][bundle-c], [Bundle D][bundle-d], [Bundle E][bundle-e], [Bundle F][bundle-f], [Bundle G][bundle-g], [Bundle H][bundle-h]
Thus Sol glibc 2.28 is admissible at the package-tag level; it is not yet a tested Photon/Kestrel host.
Pin the compatible 2.24 wheel artifacts and hashes rather than a same-version 2.34 build.
The CPython 3.11 kernel wheel SHA-256 is `ac45724d1faa300af5994078ef6209322c7c53c8f3c68179761ee126497ef771`. [Kernel metadata][kernels]
A different Python version needs its own wheel selection.

The metadata has not established a complete transitive lock.
Resolve and hash the remaining dependencies, including NumPy, tokenizers, safetensors, httpx, huggingface-hub, packaging, Pillow if required by the chosen environment, and their transitive/native requirements before an installation claim. [Kestrel metadata][package-meta], [Kernel metadata][kernels]
The CPU kernel dispatcher, native shared-library dependencies, Torch ABI compatibility, CPU instruction support, and absence of accelerator initialization still require a clean import and actual CPU execution.
A GPU-oriented package name or bundled GPU kernels alone does not demonstrate GPU execution.

## ARM64 boundary

Kestrel Native 0.1.8 has a manylinux 2.28 aarch64 wheel, while Kestrel Kernels 0.7.4 and its required ARM64 bundles A and B advertise manylinux 2.34 / 2.35. [Native metadata][native], [Kernel metadata][kernels], [Bundle A][bundle-a], [Bundle B][bundle-b]
This makes glibc 2.34 the observed prebuilt ARM64 floor for this package set.
ARM64 bundles C through H are excluded by the package's Linux architecture markers. [Kernel metadata][kernels]
Do not infer Raspberry Pi compatibility from Sol or from Jetson GPU documentation.
A native ARM64 installation must resolve its complete lock, import CPU-native packages, prove CPU float32 execution and offline behavior, and establish memory fit before qualification.

## Map consequence and remaining gates

The narrow design decision is answered: use the publisher's direct local Parakeet seam, avoid the client/engine lifecycle, and retain separate optional environment locks per architecture.
Sol's glibc version does not block the documented x86_64 wheel route.
Implementation can now be specified without another architectural decision.
Its acceptance gates remain a fully hashed installable lock, installed-package import with networking unavailable, real speech transcription and output equivalence checks, complete parameter/dtype audit, CPU-only evidence, and memory measurements.
ARM64 additionally needs a compatible glibc host and native qualification.
Redux stays outside this float32 route pending its separate owner precision decision.

[weights]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/weights.py
[resolver]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/asr/checkpoint.py
[features]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/features.py
[model]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/model.py
[tokenizer]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/tokenizer.py
[config]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/config.py
[reporter]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/photon.py
[initializer]: https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/__init__.py
[package-source]: https://files.pythonhosted.org/packages/6d/26/d6a3530617148deff50f92323425601b31006393d3b9f74e1cd3340a5792/kestrel-0.9.1.tar.gz
[md-source]: https://files.pythonhosted.org/packages/a0/9d/5b6b3b8fe0f78698a53c40f205eaa9c475b83f42ceab369f7af0bb4a5a52/moondream-2.6.1.tar.gz
[ultra-tree]: https://huggingface.co/api/models/moondream/parakeet-ultra/tree/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b?recursive=true
[package-meta]: https://pypi.org/pypi/kestrel/0.9.1/json
[md-meta]: https://pypi.org/pypi/moondream/2.6.1/json
[torch]: https://pypi.org/pypi/torch/2.9.1/json
[native]: https://pypi.org/pypi/kestrel-native/0.1.8/json
[dlpack]: https://pypi.org/pypi/torch-c-dlpack-ext/0.1.3/json
[kernels]: https://pypi.org/pypi/kestrel-kernels/0.7.4/json
[bundle-a]: https://pypi.org/pypi/kestrel-kernels-bundle-a/0.7.4/json
[bundle-b]: https://pypi.org/pypi/kestrel-kernels-bundle-b/0.7.4/json
[bundle-c]: https://pypi.org/pypi/kestrel-kernels-bundle-c/0.7.4/json
[bundle-d]: https://pypi.org/pypi/kestrel-kernels-bundle-d/0.7.4/json
[bundle-e]: https://pypi.org/pypi/kestrel-kernels-bundle-e/0.7.4/json
[bundle-f]: https://pypi.org/pypi/kestrel-kernels-bundle-f/0.7.4/json
[bundle-g]: https://pypi.org/pypi/kestrel-kernels-bundle-g/0.7.4/json
[bundle-h]: https://pypi.org/pypi/kestrel-kernels-bundle-h/0.7.4/json
[qit-media]: https://github.com/QuaanNguyen/q.it/blob/aeea34321998f04b313b0ca82382b3db32da4809/providers/local-cpu/qit_cpu/media.py
