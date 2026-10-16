# Whistle CPU format and portability boundary

Research date: October 16, 2026.
This note answers [Decide the Whistle CPU format and portability boundary](https://github.com/QuaanNguyen/q.it/issues/92) using original Cactus artifacts, source, and package metadata.
No model payloads or native engines were downloaded, packages installed, or inference executed.
Published binaries establish an available route, not a runtime pass or Raspberry Pi qualification.

## Result

The publisher's supported production path consumes the native quantized Whistle container through the Needle CPU engine.
There is no documented Cactus engine path that directly loads the original float32 safetensors checkpoint as an all-float32 inference model. [Whistle card][card], [Python inference source][inference], [Export source][export]
The publisher also provides JAX/Flax reference model code with float32 configuration and unquantized forward methods, so the original checkpoint is not intrinsically unusable outside quantized deployment.
That reference code is distinct from the production engine and is not evidence that the current engine supports float32 transcription. [Reference model][reference]

Linux x86_64 and Linux ARM64 deployment artifacts both exist in the publisher's engine repository.
The Python loader explicitly recognizes x86_64 and aarch64, including glibc and musl variants. [Engine artifact metadata][engine-tree], [Native loader and platform selection][fetch]
This clears the old macOS-only uncertainty at the distribution level; native Raspberry Pi execution and memory fit remain unverified.
The owner still must decide whether quantized Whistle belongs in an explicitly separate comparison setup.

## Pinned identity and licensing

| Component | Identity | Metadata-only evidence |
| --- | --- | --- |
| Whistle model repository | `ca5287601bef25af26dcf2e1b2bdc0843a7c19e5` | Original publisher revision |
| Native container | `whistle.cact`, 16,919,407 bytes | SHA-256 `b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb` |
| Original checkpoint | `checkpoints/whistle.safetensors`, 220,618,620 bytes | SHA-256 `5fd58c246f522ecee2b568b306598a3befdc274c225e3a7376abce277ab3d107` |
| Engine artifact repository | `93e88ede867e4311c39151285e74df513e4b835b` | Cactus-Compute/needle3 |
| Python wrapper | cactus-needle 3.2.3 | Python at least 3.9, base dependency huggingface_hub |
| Audited public runtime source | `21aab0d878ef513c621946aa40bebcce203b207b` | Original cactus-compute/needle source |

Model file identities were read from [original model tree metadata][model-tree].
The model, original engine repository, and Python package declare Apache-2.0. [Whistle license][license], [Needle repository license][engine-license], [PyPI package metadata][package-meta]
Retain the publisher's attribution and license with distributed artifacts; this note makes no broader legal judgment.

The Hub does not return a safetensors parameter total for this repository. [Whistle model metadata][model-api]
The catalog ticket's 55,151,433 is therefore a previously reported header count, not a count independently reproduced here.
Qualification must audit the original checkpoint's learned tensors and map the deployed container's logical tensors back to that architecture, including the audio front end, convolutional stem, encoder, decoder, embedding/engram tables, and any learned heads.
Packed byte count, file size, and selected decoder depth are not substitutes for full-stack logical parameter identity.
If a smaller ladder depth is selected, record the complete shipped model audit and active subnetwork configuration separately rather than silently changing the model's reported size.
The publisher describes the full Whistle architecture and laddered decoder in its card. [Whistle card][card]

## Why the original checkpoint is not an engine float32 option

The Python production wrapper passes the selected file's bytes into the native model loader and invokes the engine's transcription function.
It does not provide a safetensors checkpoint loader, a weight-dtype selector, or an all-float32 option. [Python inference source][inference]
Float32 audio samples accepted by that interface describe input samples, not learned-weight precision.

The publisher's exporter writes quantized matrices for the decoder and audio encoder/stem, alongside floating tensors used for other components.
Its quantized packing routine accepts the deployment quantization setting rather than an arbitrary 32-bit weight selection. [Export source][export]
The card explicitly compares native Whistle at 2 to 4 bits with CPU Whisper float32 and Moonshine int8. [Whistle card][card]
The `.cact` container's ability to hold some floating records does not imply that the publisher supplies a complete float32 Whistle export or an engine mode for one.
The supported deployment route examined here therefore requires native quantized weights.
This is a boundary of the published path, not proof that a future engine implementation could never support float32.

The JAX reference model defaults to float32 and implements feature preparation, encoding, decoding, and logits with quantization disabled by default.
It loads the original checkpoint through the publisher's checkpoint reader. [Reference model][reference], [Checkpoint reader][checkpoint]
However, the published fine-tuning evaluator explicitly quantizes parameters and requests quantized execution before its autoregressive scoring loop. [Fine-tuning source][finetune]
Establishing an unquantized CPU transcription provider from the reference model would require another investigation of tokenizer assets, decoding semantics, package locks, and output equivalence.
Do not call that an existing engine option or undertake it as an incidental workaround for an unanswered owner decision.

## Linux distribution and isolated installation

The engine repository publishes both CLI/static-library folders and separate Python shared-library wheels for Linux x86_64 and ARM64. [Engine artifact metadata][engine-tree]
The Python package's wheel is platform-neutral because the native engine is obtained separately through its fetch machinery. [Package metadata][package-meta], [Native loader and platform selection][fetch]
A successful installation of the wrapper alone is not a complete offline installation.

A concrete initial environment candidate is CPython 3.11 with cactus-needle 3.2.3, a fully pinned huggingface_hub dependency lock, and the chosen native engine artifact staged beforehand.
Use a separate optional provider environment; the production transcription path needs neither Torch nor NeMo nor the package's training, GPU, microphone, or comparison extras. [Package metadata][package-meta], [Python inference source][inference]
q.it already prepares mono audio, so feed 16 kHz float samples directly and avoid adding microphone/resampling dependencies solely for this path.

The inspected wrapper pins engine version 3.3.1 for Needle 3, which also serves Whistle; this differs from the wrapper's own 3.2.3 version. [Native loader and platform selection][fetch]
At the pinned engine repository revision, the relevant glibc-wheel digests are:

| Engine wheel | SHA-256 |
| --- | --- |
| cactus_needle-3.3.1-py3-none-manylinux2014_x86_64.whl | `3de03a183f56e28f8f1945fbbc5048efca0699a7f3ed0d66a244ca7e30442c22` |
| cactus_needle-3.3.1-py3-none-manylinux2014_aarch64.whl | `3128efab8a8cf9f8381a6ea38eee3db66d85746260489563513160940dcbaf63` |

Those identities come from [publisher artifact tree metadata][engine-tree].
The manylinux2014 tags indicate a glibc 2.17 distribution baseline; importing and executing on the actual CPU remains a required check.
The repository also contains musllinux 1.2 wheels, while a 32-bit ARM CLI folder does not establish a supported 32-bit Python shared-library route. [Engine artifact metadata][engine-tree], [Package loader source][package-source]
The wrapper diagnoses unsupported architectures and distinguishes Android Bionic from ordinary Linux to avoid selecting an incompatible Linux shared object. [Package loader source][package-source]

Explicit acquisition should stage the pinned model and engine, verify hashes, and set a local engine-library path before starting the provider.
Pass the explicit local Whistle container path instead of relying on first-use fetching.
The default wrapper fetches missing weights and native libraries, so a benchmark integration must validate local availability before accepting requests. [Package loader source][package-source]
Disable both Python and native engine telemetry before startup using the publisher's documented telemetry controls. [Telemetry source][telemetry], [Needle deployment documentation][deploy]
Record that telemetry is disabled in the environment manifest and verify that offline startup and invocation do not trigger acquisition.

## Integration and comparison metadata

Reuse the existing speech benchmark task and inert packs through an isolated loopback provider backend.
The native wrapper holds a process-global model and documents that it is not thread-safe, so serialize inference and keep one selected Whistle model per process. [Python inference source][inference]
q.it's existing local CPU server already serializes invocation. [q.it server][qit-server]

Return transcript and input duration, preserving the native engine's language and timing fields as additional metadata when available.
The wrapper exposes first-token milliseconds and decoder tokens per second, but these have engine-specific scope and must not be relabelled as a streamed q.it measurement without checking the actual endpoint semantics. [Python inference source][inference]
Silence can legitimately yield an empty transcript, which requires a speech-specific output-validation policy; do not invent words to make a transport check succeed. [Whistle card][card]

The card exposes a decoder-depth option on the native engine, but the inspected Python Whistle constructor and transcription method do not expose it. [Whistle card][card], [Python inference source][inference]
Choose either the pinned CLI/C interface that accepts depth or an explicitly recorded full-depth Python route.
An integration must verify the effective depth instead of recording a requested setting that the Python wrapper ignores.
The exact native default depth and complete decode configuration remain qualification facts to obtain from the staged engine.

Each run must retain wrapper and engine versions separately, engine artifact digest and platform tag, container revision and digest, original-checkpoint identity, logical parameter audit, active decoder depth, native weight quantization, activation/cache precision where reported, language mode, beam/keyword settings, sample rate, thread/affinity settings, and q.it/host identity.
Do not describe mixed native 2-to-4-bit weights as float32 or make up an exact bit layout when it has not been parsed from the container.
Any accepted quantized setup must have a distinct comparison identity, including precision and engine configuration, rather than depending on a label only visible in free-text notes.

## Unsupported paths and remaining evidence

Fail before measurement with the actual stage and diagnostic for a missing optional wrapper, missing staged engine/model, incompatible architecture/libc/shared library, unsupported container, missing depth control, a failed parameter audit, or an inadmissible quantized precision setup.
Retain the attempted identity and failure without falling back to another model, cloud transcription, a GPU, or automatic acquisition.
A transport pass, real English speech transcription, installed-package setup, parameter audit, engine depth/defaults, process memory, and native ARM64 run remain unverified.

## Map implications

The source facts resolve the present format and published-platform questions: the supported engine route is native quantized `.cact`, and both Linux x86_64 and ARM64 artifacts exist.
[Add a cactus-needle CPU provider path for speech-to-text](https://github.com/QuaanNguyen/q.it/issues/87) should retain an explicit dependency on [Decide admission policy for exceptional model candidates](https://github.com/QuaanNguyen/q.it/issues/95).
No precision exception is granted by this report.
If the owner accepts native quantized Whistle, implementation can proceed with a separate setup, pinned engine acquisition, the depth-control decision, and actual CPU validation.
If the owner requires float32, leave this native provider candidate unadmitted and create a separate research question only if the owner wants investigation of the JAX reference path or future publisher engine support.
Moondream Redux independently needs a ternary-precision admission decision, as detailed in [Publisher CPU routes for Moondream Parakeet and QVAC TranscribePsy](parakeet-qvac-cpu-routes.md).
The two owner decisions must not be inferred from each other.

[card]: https://huggingface.co/Cactus-Compute/whistle/blob/ca5287601bef25af26dcf2e1b2bdc0843a7c19e5/README.md
[license]: https://huggingface.co/Cactus-Compute/whistle/blob/ca5287601bef25af26dcf2e1b2bdc0843a7c19e5/LICENSE
[model-api]: https://huggingface.co/api/models/Cactus-Compute/whistle/revision/ca5287601bef25af26dcf2e1b2bdc0843a7c19e5
[model-tree]: https://huggingface.co/api/models/Cactus-Compute/whistle/tree/ca5287601bef25af26dcf2e1b2bdc0843a7c19e5?recursive=true
[engine-tree]: https://huggingface.co/api/models/Cactus-Compute/needle3/tree/93e88ede867e4311c39151285e74df513e4b835b?recursive=true
[package-meta]: https://pypi.org/pypi/cactus-needle/3.2.3/json
[package-source]: https://files.pythonhosted.org/packages/97/e1/a64541188e38bc5b1ee3ab4cf90f967a2428f9e7c6feadddfd4cb50a90dc/cactus_needle-3.2.3.tar.gz
[inference]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/agent/whistle.py
[export]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/model/export.py
[reference]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/model/whistle.py
[checkpoint]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/model/checkpoints.py
[finetune]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/model/whistle_finetune.py
[telemetry]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/_telemetry.py
[deploy]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/README.md
[engine-license]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/LICENSE
[qit-server]: https://github.com/QuaanNguyen/q.it/blob/aeea34321998f04b313b0ca82382b3db32da4809/providers/local-cpu/qit_cpu/server.py

[fetch]: https://github.com/cactus-compute/needle/blob/21aab0d878ef513c621946aa40bebcce203b207b/needle/agent/fetch.py
