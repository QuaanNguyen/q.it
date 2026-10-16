# Indra OCR: audit and CPU deployment boundary

Research date: 2026-10-16.
This note answers [Decide the Indra OCR audit and CPU deployment boundary](https://github.com/QuaanNguyen/q.it/issues/94) for [Add an ONNX Runtime CPU provider path for multi-stage OCR pipelines](https://github.com/QuaanNguyen/q.it/issues/88) and [Catalog candidates: sub-1B batch 2026-10-16](https://github.com/QuaanNguyen/q.it/issues/89).
Only publisher source, license text, package metadata, and model-file metadata were read.
No model weights were downloaded, no packages installed, no inference run, and no license accepted.

## Proposed route

Keep Indra outside the qualified catalog until owner review of the weight license and a complete graph audit permit the next steps.
Prepare a separate, explicitly selected ONNX CPU serving environment, retaining q.it's provider boundary and declarative OCR benchmark pack.
Use the original publisher pipeline rather than adapting Indra into a conversational Transformers model.
These are technical recommendations, not implementation or a determination of legal permission.
The current local CPU package requires Torch even for its base installation, so a Torch-free ONNX route needs dependency separation before it can satisfy the publisher's lightweight installation premise.
See [local CPU packaging](../../providers/local-cpu/pyproject.toml) and [the provider and pack decision](../adr/0005-benchmark-packs-and-provider-adapters.md).

## Reproducible publisher identity

The source inspected is commit `3d773aeedc5d79d1d0a2a04007a9075cd71a59f4`, whose package metadata declares `indra-ocr` version `0.3.2` and Python `>=3.10`.
Its dependencies are NumPy, headless OpenCV, ONNX Runtime, pypdfium2, and Hugging Face Hub, without pinned versions.
See [publisher package metadata](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/pyproject.toml).

The source selects model tag `v3.1.1`, which resolved during research to immutable revision `ed345c7f158eddbc7dee0c6a7269afb75a94f7e8`.
The automatic downloader permits ONNX files broadly, omits math files unless math is requested, and periodically checks its cached revision.
A local model directory bypasses downloading, making a validated, immutable directory the recommended measurement input.
See [publisher model resolution](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/models.py) and [publisher revision metadata](https://huggingface.co/api/models/datalab-to/indra-ocr/revision/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8?blobs=true).

PyPI identifies the `indra_ocr-0.3.2-py3-none-any.whl` SHA-256 as `d55ad2be54c45df80e977626bcf0d64cf6df7aa20eecf608f90669c90057eed2` and the source archive as `672963005be6fe9986889b48d4540a50ba4241da8e601cfa4665227cb5bf91fc`.
These are registry metadata identities, not locally verified package bytes.
The published wheel has not been checked for equality with the inspected source commit, so implementation must choose and record one source provenance.
See [PyPI version metadata](https://pypi.org/pypi/indra-ocr/0.3.2/json).

## Pipeline and required stages

The publisher image pipeline performs orientation and deskew, line detection, line recognition with optional script routing, then reading order, layout, and output assembly.
The language classifier routes Chinese-character lines to the CJK recognizer and other lines to the Latin recognizer.
Optional math classifies line crops, encodes selected lines, initializes decoder state, and repeatedly executes the decoder step before conditionally replacing OCR text with LaTeX.
See [image pipeline](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/pipeline.py) and [math pipeline](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/math.py).

| Profile stage | Files | Qualification requirement |
| --- | --- | --- |
| Detection and Latin recognition | `det.onnx`, `rec.onnx`, `meta.json` | Required even for the publisher's minimal local directory. |
| Order and layout | `order.onnx` | Require for the full default pipeline rather than accepting geometric fallback. |
| Orientation | `orient.onnx` | Require for the full default pipeline rather than silently disabling orientation. |
| Script routing and CJK recognition | `script.onnx`, `rec_cjk.onnx`, `meta.json` | Require both graphs and vocabulary for the full default pipeline. |
| Optional math profile | `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`, `math_dec_step.onnx`, `math_meta.json` | Require all five files when math is enabled. |

The table follows the [publisher session construction](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/pipeline.py), [local directory validation](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/models.py), and [math loader](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/math.py).
The source also recognizes optional `mask.onnx` and automatically prefers `rec.int8.onnx` when present, although neither file is in this pinned repository inventory.
Recommend rejecting unmanifested graph files and explicitly disabling the int8 choice for the float32 profile.
Missing optional graphs must fail full-profile qualification rather than silently change the compared pipeline.

For OCR benchmarking, use images directly or the publisher's forced OCR PDF mode.
Its default PDF mode can extract trusted PDF text and bypass some recognition, making that a distinct document-conversion configuration.
Keep OCR text accuracy separate from VQA and chat scores, and record page rendering settings and text-layer use if PDF conversion is later included.
See [publisher PDF conversion modes](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/document.py).

## File manifest

The following byte sizes and SHA-256 identities are publisher LFS metadata at `ed345c7f158eddbc7dee0c6a7269afb75a94f7e8`.
They have not been verified against downloaded weights.
The ten ONNX files sum to 38,025,105 bytes; the six default non-math graphs sum to 18,158,743 bytes.
This explains why neither the repository's overall storage figure nor its small recognizer headline represents a complete parameter audit.
See [publisher file metadata](https://huggingface.co/api/models/datalab-to/indra-ocr/revision/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8?blobs=true).

| File | Bytes | Publisher SHA-256 |
| --- | ---: | --- |
| `det.onnx` | 389,948 | `46b0050ce4c4692f887a1a9c038480223af1c7a91b3fcced2657d9cf00290dc4` |
| `math_dec_init.onnx` | 1,356,364 | `919417fde9962c3318aeb478c1fe3955dd2bb8cd34476c68debed2ba39e5dcd6` |
| `math_dec_step.onnx` | 8,429,418 | `5ff6e6cbaf0dea70314f66222a1469736a8c5e7994fab06a5b4e53c90469239f` |
| `math_enc.onnx` | 9,889,575 | `d4e4a6c40dad1ee988d32c6403b1590ca42864d4483bbe1e671c5dbb999a5a97` |
| `mathcls.onnx` | 191,005 | `818b1556e9ef337958db6be0c9791299d3f74a4575586b0cd469aaae9534ff96` |
| `order.onnx` | 2,973,138 | `23635dd2324b4ccc00c408af961273a2dedd2e4f0386bc54dba678c41bcf6b46` |
| `orient.onnx` | 371,414 | `8d7bfce5d08bcad55955f41ca494a7f9d9d3ed5d5b6da1f6f276d5c509590729` |
| `rec.onnx` | 4,977,797 | `e243be069b357e56a0e867a21d7482e498df42fb19678bfb67a25f48b1f06151` |
| `rec_cjk.onnx` | 9,255,441 | `4cec840b55e1bb7f375daf30a2975404a867a71ac1a5bc5b1629c4909ce16baa` |
| `script.onnx` | 191,005 | `5ba37030e817affede0097a7503b6829624104166c6e0b6aacecacf919fc7d11` |

Small publisher metadata and license files were fetched as text and hashed locally.
The exact inputs are [configuration](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/config.json), [OCR metadata](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/meta.json), [math metadata](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/math_meta.json), and [weight license](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/LICENSE).

| File | Locally computed SHA-256 |
| --- | --- |
| `config.json` | `cd143c66015620865d9d8361b1bb6f9781363b8e010d78c49b10d69009807d1d` |
| `meta.json` | `ff6ec58f91c93a475ae9505ad21c73c3ecda0b378f1d889dd83408e10832d1bb` |
| `math_meta.json` | `85d9715fb8a77791b55ef89f57fb41e72eb3170ec58a7c3bee3b394268bf2b10` |
| `LICENSE` | `50c88a018a7ee322e89f6b09401b237beb0cd331685879405b3db48993950a5f` |

## Complete parameter audit method

The publisher explicitly states that `model.safetensors` contains only the Latin recognizer's float weights, exists for the Hub count, and is unused by the package.
The Hub metadata reports 1,236,366 float32 parameters for that file.
It cannot be used as the full pipeline count.
See [publisher configuration](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/config.json) and [model metadata](https://huggingface.co/api/models/datalab-to/indra-ocr/revision/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8?blobs=true).

ONNX stores graph initializers, sparse initializers, tensor attributes, model-local functions, and external tensor data, so a dense root-initializer sum is incomplete in the general case.
Exported inference graphs also do not identify every tensor as a learned parameter.
See [ONNX protobuf specification](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto), [ONNX intermediate representation](https://onnx.ai/onnx/repo-docs/IR.html), and [external data specification](https://onnx.ai/onnx/repo-docs/ExternalData.html).

Recommended audit procedure:

1. After owner review permits acquisition, validate all ten pinned graphs and their dependencies, not merely the subset selected by one test image.
2. Enumerate every graph scope, nested graph attribute, reachable local function, initializer, sparse initializer, and tensor-valued or scalar/list constant.
3. Record tensor name and scope, dimensions, data type, element count, content digest, storage location, and learned-weight versus structural-constant classification with evidence.
4. Resolve external tensor files only inside the validated bundle and hash each referenced file or segment; reject missing data and escaping paths.
5. Report per-stage stored tensor elements separately from the evidenced learned-parameter count, including buffers and structural constants separately.
6. Count references to the same tensor definition once inside a graph, but do not merge independent tensors merely because their bytes match.
7. Reconcile duplicated exports or tied weights, especially decoder initialization and step graphs, only with publisher architecture or training evidence; otherwise retain a conservative total and document the uncertainty.
8. Publish the total for the full available bundle and the total for the selected profile, retaining stage execution traces per benchmark case.
9. Validate floating learned weights as float32 and check graph operators and constants for quantization or mixed precision; integer token indices and dimensions are not themselves reduced-precision learned weights.
10. Record ONNX IR and opset versions, CPU kernel availability, audit tool version, and audit output digest before qualifying the under-one-billion limit.

These are audit requirements proposed from the specifications, not a claim that the actual graph audit has passed.
The exact full learned-parameter count, graph precision, export sharing, external-data status, and CPU operator compatibility remain unverified because weights were not acquired.
Serialized file size must not substitute for a parameter count.

## CPU installation and isolation

The official CPU Python distribution is `onnxruntime`; GPU distributions are separate.
Recommend an isolated Python 3.11 environment for this provider with fully pinned dependency versions and wheel hashes, avoiding the existing Transformers environment.
See [official CPU installation instructions](https://onnxruntime.ai/docs/install/).

A concrete release candidate is `onnxruntime==1.31.0`, whose registry metadata requires Python `>=3.11` and lists the following CPython 3.11 Linux wheels.
Its base dependencies include FlatBuffers, NumPy, Packaging, and Protobuf.
See [release metadata](https://pypi.org/pypi/onnxruntime/1.31.0/json).

| Architecture | Wheel | Publisher SHA-256 |
| --- | --- | --- |
| x86_64 | `onnxruntime-1.31.0-cp311-cp311-manylinux_2_28_x86_64.whl` | `d4092b78fc5bab77ce6522393098cdb2535423045ecdcff15cc0d022162d6b66` |
| ARM64 | `onnxruntime-1.31.0-cp311-cp311-manylinux_2_28_aarch64.whl` | `37c7dfe398550afdf9670a29315dbb88e49d8afc473ffaf1f410376efbb9c80a` |

The wheel tags require the matching CPython ABI, a matching 64-bit architecture, and glibc 2.28 or newer.
A Raspberry Pi needs a compatible 64-bit Linux userland; a 32-bit ARM userland does not match the listed ARM64 wheel.
Wheel availability alone does not prove Raspberry Pi 5 execution or dependency compatibility.
See [Python wheel platform tag specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).

Preserve the publisher's CPU-only sessions, explicit thread settings, full graph optimization setting, and disabled CPU memory arena.
The source documents memory-arena growth on varying OCR shapes as the reason for disabling it.
Keep one worker inside the scheduler CPU allocation, record both package versions and thread counts, and validate every created session uses only the CPU execution provider.
Use end-to-end request timing for comparison, retaining publisher stage timings as supplementary diagnostics.
See [publisher session setup](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/indra_ocr/pipeline.py) and [ONNX Runtime session API](https://onnxruntime.ai/docs/api/python/api_summary.html).

Implementation must resolve and hash all transitive dependencies on both architectures, verify imports without model downloads, and then run approved OCR inputs on the pinned bundle.
No complete dependency lock or successful install is claimed here.
Sol x86_64 can validate its own CPU route but cannot establish ARM64 qualification.

## License facts for owner review

Code is Apache-2.0; weights carry a modified OpenRAIL-M license.
The weight terms treat downloading and using as acceptance.
Attachment A restricts affiliated entities above $20 million prior-year gross revenue or total funding, with personal/research exceptions for those two clauses.
Its competitor clause has no stated research exception and includes affiliates' products/services.
Sections 4 and 5 require downstream restrictions and license notices; Sections 6 and 7 address output responsibility and attribution, including a model link, license copy, and change disclosure.
Section 8 expressly extends share-alike to outputs and derivatives.
Attachment A also restricts unlawful uses and third-party intellectual-property violations.
See the [original weight license](https://huggingface.co/datalab-to/indra-ocr/blob/ed345c7f158eddbc7dee0c6a7269afb75a94f7e8/LICENSE) and [original code license](https://github.com/datalab-to/indra/blob/3d773aeedc5d79d1d0a2a04007a9075cd71a59f4/LICENSE).

Owner review must establish the relevant individual/entity and affiliations, research applicability, competitive status, publication/output obligations, and whether separate publisher permission is required.
This research makes no determination that benchmarking or public results satisfy the terms.
The catalog's short license label does not capture all these obligations.

## Required unsupported-path diagnostics

Recommend rejecting setup or qualification with a specific reason and remedy for an incompatible Python/ABI, 32-bit userland, old glibc, absent compatible wheel, native-library import failure, missing CPU execution provider, unsupported graph operator/opset, or a missing or mismatched model file.
Also distinguish unresolved license review, incomplete parameter audit, non-float32 learned weights, absent requested math stages, and attempted automatic downloads.
Preserve failed benchmark attempts in run history with those diagnostics rather than reporting a successful empty OCR response.
These are implementation acceptance requirements proposed for the [ONNX provider ticket](https://github.com/QuaanNguyen/q.it/issues/88), not implemented behavior.

## Remaining gates

The technical route is specified, but owner license review, full graph/precision audit, immutable package locking, provider implementation, a scored OCR benchmark pack, and actual CPU execution remain outstanding.
ARM64 remains explicitly unverified until a real compatible ARM64 host completes the install and OCR path.
