# Vision and embedding candidate serving routes

Research date: 2026-10-16.
Resolves [Decide serving routes for the new vision and embedding candidates](https://github.com/QuaanNguyen/q.it/issues/93), supporting [Catalog candidates: sub-1B batch 2026-10-15](https://github.com/QuaanNguyen/q.it/issues/86) and [Catalog candidates: sub-1B batch 2026-10-16](https://github.com/QuaanNguyen/q.it/issues/89).
This investigation inspected publisher cards, pinned configuration and inference source, package metadata, and HTTP byte ranges containing safetensors JSON headers only.
It did not download tensor payloads, install packages, load checkpoints, run inference, or decide license eligibility.
Every CPU route below remains a qualification candidate, and Raspberry Pi execution remains unverified.

## Decision

Keep the existing provider environment unchanged and qualify these candidates in isolated environments.
LightOnOCR has a valid OCR route and UltraBERT has a valid dense retrieval route after dependency and integration work.
Liquid d1 and Intern-Decision require a decision-task design before either can enter truthful benchmark comparisons.
UIBK requires a separate late-interaction route; dense-only extraction would be a separately named experimental configuration.
None of the five is a catalog-only addition to the current pinned provider.
The route evidence and missing integrations are detailed below.

## Canonical identities and count audit

Each revision came from the canonical publisher's model API on the research date.
Header totals sum tensor shape products, while learned counts exclude fixed buffers identified in publisher source.
These are static checkpoint audits, not a substitute for counting unique learned parameters in the eventually loaded CPU stack.
The [safetensors format](https://github.com/huggingface/safetensors#format) separates its length-prefixed JSON header from tensor data, permitting this audit without downloading weights.

| Canonical publisher checkpoint | Immutable revision | HF floating scalar total | Learned full-stack parameters | Count evidence |
| --- | --- | ---: | ---: | --- |
| lightonai/LightOnOCR-3-0.8B | `36d8636925327c671bd8e479c517a004f7566bd9` | 852,985,920 | 852,985,920 | [Metadata](https://huggingface.co/api/models/lightonai/LightOnOCR-3-0.8B), [single checkpoint](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/model.safetensors) |
| LiquidAI/d1-omni-600M | `02b55d7076f15129e59ab3f94783f32c4b088674` | 587,161,089 | 587,143,681 | [Metadata](https://huggingface.co/api/models/LiquidAI/d1-omni-600M), [single checkpoint](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/model.safetensors), [audio source](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/audio.py) |
| dmis-lab/UltraBERT-Embed-150M | `11315b8ad88a0110a8ffc256fe92219a98c45cba` | 151,032,576 | 151,032,576 | [Metadata](https://huggingface.co/api/models/dmis-lab/UltraBERT-Embed-150M), [single checkpoint](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/model.safetensors) |
| internlm/Intern-Decision-0.8B | `85a0cc5a99d67ea8d56dfe98115689212867171d` | 852,985,920 | 852,985,920 | [Metadata](https://huggingface.co/api/models/internlm/Intern-Decision-0.8B), [three-component index](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/model.safetensors.index.json) |
| DataScience-UIBK/uibk-embedding-v1 | `b18df82e77306006a1dd0162199f6dfeacb5492e` | 149,015,808 at root only | 179,017,728 | [Metadata](https://huggingface.co/api/models/DataScience-UIBK/uibk-embedding-v1), [full file tree](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/tree/b18df82e77306006a1dd0162199f6dfeacb5492e), [output source](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/hybrid_tokens.py) |

Liquid's file contains 587,161,106 stored scalars: 587,161,089 F32 values plus 17 integer batch counters.
Its 17 BatchNorm layers also store 17,408 F32 running-statistic values, giving 587,143,681 learned parameters after excluding all 17,425 buffers.
The learned breakdown is text encoder 354,483,968, decision head 26,248,193, vision 94,234,880, and audio 112,176,640.
All four components must remain in its full-stack audit even for an image-only benchmark.
Sources: [checkpoint header source](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/model.safetensors), [component construction](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py), and [BatchNorm construction](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/audio.py).

UIBK's three F32 files contain 149,015,808 root scalars, 20,060,928 dense-branch scalars, and 10,249,856 output-module scalars.
Their sum, 179,326,592, is the ticket's stored-scalar count.
The output module registers two fixed rotation buffers with shapes 768 by 381 and 128 by 127, together 308,864 scalars.
Subtracting these leaves 9,940,992 learned output parameters and 179,017,728 learned parameters overall.
Sources: [root checkpoint](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/model.safetensors), [dense branch](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/dense_branch.safetensors), [output checkpoint](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/1_HybridTokens/model.safetensors), and [buffer declarations](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/hybrid_tokens.py).

Intern-Decision's three components contain 752,393,024 language, 12,588,544 projector, and 88,004,352 vision parameters.
The checkpoint includes 2,592 F32 values and 852,983,328 BF16 values, so a future float32 benchmark must explicitly convert the complete loaded stack.
LightOn's checkpoint is BF16, while UltraBERT and UIBK are F32.
Sources: the pinned checkpoint links and canonical metadata in the table above.

## Current environment and isolation

The inspected q.it revision is `aeea34321998f04b313b0ca82382b3db32da4809`.
Its [provider dependencies](../../providers/local-cpu/pyproject.toml) pin Python 3.11 through 3.13, PyTorch 2.9.1, Transformers 4.57.6, Sentence Transformers 5.3.0, Pillow 12.0.0, and supporting numeric packages.
Its [provider implementation](../../providers/local-cpu/qit_cpu/backend.py) uses generative image chat, dense Sentence Transformers embeddings, and CrossEncoder reranking; it has no typed decision or token-vector MaxSim path.

| Candidate | Why the current pins are insufficient | Isolated qualification route |
| --- | --- | --- |
| LightOnOCR | Requires Qwen3.5 classes absent from Transformers 4.57.6 | Transformers 5.5.4 as its publisher recipe, or a separately validated later pin; CPU float32 Qwen3.5 conditional generation |
| Liquid d1 | Publisher requires Transformers at least 5.15 and torchvision is missing | A Transformers 5.16.1 candidate environment, PyTorch 2.9.1 and torchvision 0.24.1; pinned publisher decision API |
| UltraBERT | Requires Sentence Transformers at least 6 and Transformers at least 5; saved module paths use the new Sentence Transformers layout | Sentence Transformers 6.0.1 and Transformers 5.16.1 match its saved configuration; retain PyTorch 2.9.1 only as a metadata-admissible candidate |
| Intern-Decision | Publisher pins Transformers 5.14.1 and torchvision 0.24.1 | Publisher environment with PyTorch 2.9.1 and a separate decision integration |
| UIBK | Publisher code requires Transformers at least 5.3 and imports PyLate | PyLate 1.6.0, Sentence Transformers 5.3.0, Transformers 5.3.0, PyTorch 2.9.0 and FastPLAID 1.4.6.290 are an exact metadata-admissible research environment, pending platform and execution checks |

Sources: [LightOn recipe](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/README.md), [Liquid recipe](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md), [UltraBERT saved versions](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/config_sentence_transformers.json), [UltraBERT modules](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/modules.json), [Intern requirements](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/requirements.txt), [UIBK source](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/branched_transformer.py), [PyLate metadata](https://pypi.org/pypi/pylate/1.6.0/json), and [FastPLAID metadata](https://pypi.org/pypi/fast-plaid/1.4.6.290/json).

Transformers 4.57.6 has ModernBERT but neither Qwen3.5 nor UltraBERT in its native architecture mapping.
Sentence Transformers 6.0.1 requires Transformers at least 5.0 and PyTorch at least 2.2; Transformers 5.16.1's torch extra requires PyTorch at least 2.5, so those declarations admit PyTorch 2.9.1 without proving remote-code API compatibility.
Sources: [4.57.6 mapping](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/configuration_auto.py), [Sentence Transformers metadata](https://pypi.org/pypi/sentence-transformers/6.0.1/json), and [Transformers metadata](https://pypi.org/pypi/transformers/5.16.1/json).

PyLate 1.6.0 caps Transformers at 5.3.0 and pins Sentence Transformers 5.3.0, preventing one shared environment with UltraBERT or Liquid d1's publisher requirements.
Every released FastPLAID version in PyLate's permitted 1.4.6.270 through 1.4.6.2110 range pins a different exact PyTorch version: 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.10.0, or 2.11.0.
None admits q.it's exact PyTorch 2.9.1 pin.
Sources: [PyLate constraints](https://pypi.org/pypi/pylate/1.6.0/json) and [FastPLAID release metadata](https://pypi.org/pypi/fast-plaid/json), with version-specific endpoints for [2.9.0](https://pypi.org/pypi/fast-plaid/1.4.6.290/json), [2.10.0](https://pypi.org/pypi/fast-plaid/1.4.6.2100/json), and [2.11.0](https://pypi.org/pypi/fast-plaid/1.4.6.2110/json).

## Task and preparation contracts

### LightOnOCR

Use the checkpoint's processor, chat template, Qwen3.5 conditional-generation class, and thinking disabled.
Transcription uses an image with an empty text instruction; grounding uses the literal `grounding` instruction.
Other instructions are outside the publisher's supported training distribution, so generic visual-question benchmarks would misrepresent this checkpoint.
For document comparisons, preserve aspect ratio and record the publisher's 400 DPI rendering and 2048-pixel longest-edge preparation.
Use an OCR transcription pack, and keep grounding as a separate benchmark requiring bounding-box-aware expectations.
Source: [publisher usage and prompt contract](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/README.md).

### Liquid d1 and Intern-Decision

Liquid d1 scores named options over a state and optional images or audio, returning typed probabilities with zero generated output tokens.
Its image text budget is 896 tokens; audio is mono 16 kHz and capped at 30 seconds; one request accepts images or audio, never both.
It is neither OCR transcription nor ASR nor an open-ended visual assistant.
Its native source constructs all modality encoders and exposes its own decision methods rather than a conventional text-generation call.
Sources: [publisher task contract](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md), [API source](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py), and [prompt compiler](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/prompt.py).

Intern-Decision also uses a single forward pass, a complete assistant JSON skeleton with decision placeholders, candidate-symbol logits, and checkpoint-specific probability calibration.
Its supported engine does not call text generation; auto-generated Hugging Face pipeline examples do not establish a valid alternative benchmark route.
Preserve question and option order, the checkpoint tokenizer and processor, the empty thinking block, and the 0.8B engine's default calibration temperature 2.747760550703.
Its usage output count denotes scored fields, so it must not become generation token throughput.
Sources: [publisher inference contract](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md) and [actual inference source](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/inference.py).

Both candidates therefore need a new decision about task schema, typed provider transport, quality scoring, and meaningful latency metrics before implementation.
Their names and architecture sizes do not justify mapping them onto q.it's existing generative image task.
This is a route conclusion drawn from the publisher contracts and the existing provider interfaces cited above.

### UltraBERT

Load the complete pinned Sentence Transformers checkpoint with pinned trusted publisher code.
Its saved configuration supplies the query instruction `Instruct: Given a question, retrieve passages that answer the question\nQuery: ` and an empty document prompt.
Its saved pooling configuration selects CLS, includes the prompt, and returns 768 dimensions; use normalized vectors for cosine retrieval.
The checkpoint's 32,768-token limit and English text/code scope should be recorded with benchmark preparation.
Sources: [saved prompts](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/config_sentence_transformers.json), [pooling](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/1_Pooling/config.json), [encoder source](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/modeling_ultrabert.py), and [publisher card](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/README.md).

### UIBK

The supported checkpoint is a branched ModernBERT encoder followed by a custom hybrid-token module, with both custom modules required.
Use PyLate query/document roles, required `query: ` and `document: ` prompts, query length 256, document length 2048, and the stored role markers and masking configuration.
Output is a variable-length matrix of 128-dimensional vectors, including three dense slots; score with MaxSim while preserving all vectors.
Flattening, averaging, or selecting one output vector would change the model's advertised retrieval behavior.
Sources: [module manifest](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/modules.json), [role and length configuration](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/config_sentence_transformers.json), [output source](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/hybrid_tokens.py), and [publisher usage](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/README.md).

The custom output source exposes a normalized 768-dimensional dense representation internally, but still imports PyLate and depends on the learned dense branch and latent-attention module.
A dense-only experiment would need explicit integration, checkpoint-wide loading validation, a distinct configuration identity, and its own quality evidence.
The ordinary root encoder with CLS or mean pooling is not that dense branch.
Source: [dense extraction and pooling source](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/hybrid_tokens.py).

## Linux CPU and ARM64 evidence

PyTorch 2.9.1 and torchvision 0.24.1 publish Linux aarch64 wheels for Python 3.11 through 3.13, tagged manylinux 2.28.
The matching torchvision package requires exactly PyTorch 2.9.1.
These artifacts support a dependency route on sufficiently recent 64-bit Linux; they do not prove model execution, Raspberry Pi memory suitability, or an entire resolved environment.
Sources: [PyTorch files and requirements](https://pypi.org/pypi/torch/2.9.1/json) and [torchvision files and requirements](https://pypi.org/pypi/torchvision/0.24.1/json).

| Candidate | CPU evidence | ARM64 limit |
| --- | --- | --- |
| LightOnOCR | Qwen3.5 has a slower PyTorch fallback when optional fast kernels are unavailable | No checkpoint-specific Raspberry Pi execution evidence was established; float32 CPU processor and generation must be exercised |
| Liquid d1 | Publisher explicitly chooses float32 for CPU; shipped model uses PyTorch and NumPy implementations | ARM64 packages exist for the proposed base, but exact custom vision/audio execution remains untested |
| UltraBERT | Shipped source contains an eager attention implementation rather than an unconditional CUDA-only call | Remote source imports evolving Transformers helpers; package acceptance and CPU/ARM64 execution remain separate checks |
| Intern-Decision | Engine accepts explicit CPU, float32, and attention implementation; CUDA synchronization is conditional | Publisher latency evidence is RTX 4090 only; CPU and ARM64 success must not be inferred from those numbers |
| UIBK | Custom branches use PyTorch operations and latent pooling uses scaled dot-product attention | All inspected FastPLAID releases permitted by PyLate lack Linux aarch64 wheels, blocking the ordinary binary-only install route |

Sources: [Qwen3.5 fallback documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [Liquid CPU example](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md), [UltraBERT eager source](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/blob/11315b8ad88a0110a8ffc256fe92219a98c45cba/modeling_ultrabert.py), [Intern engine](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/inference.py), [Intern latency context](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md), and [FastPLAID compatible build files](https://pypi.org/pypi/fast-plaid/1.4.6.290/json).

TEI currently documents aarch64 CPU support and a CPU ARM64 image, but its supported architecture list does not include UltraBERT or UIBK's custom hybrid modules.
ModernBERT support alone is insufficient evidence for the latter's branched encoder and token output.
Do not advertise these two checkpoints as TEI-compatible without explicit implementation evidence and execution.
Source: [TEI supported models and hardware](https://huggingface.co/docs/text-embeddings-inference/supported_models).

## Revision integrity and owner decisions

Pin model files, processor/tokenizer files, and every trusted code module to the revisions in the identity table.
Liquid's lazy tokenizer loader calls the model's stored name or path without passing an explicit revision, so a pinned local snapshot is the safer integration boundary for preserving tokenizer identity.
UIBK's extra dense-branch download forwards a revision through model arguments, which must also be verified when integrating a Hub-loaded checkpoint.
Sources: [Liquid tokenizer loader](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py) and [UIBK branch loader](https://huggingface.co/DataScience-UIBK/uibk-embedding-v1/blob/b18df82e77306006a1dd0162199f6dfeacb5492e/branched_transformer.py).

LightOnOCR, UltraBERT, UIBK, and Intern-Decision declare Apache-2.0; Intern also ships an upstream Qwen license that must be preserved with applicable notices when redistributing.
No special owner eligibility decision was identified for those four from the inspected publisher materials.
Sources: [LightOn license statement](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/README.md), [UltraBERT metadata](https://huggingface.co/api/models/dmis-lab/UltraBERT-Embed-150M), [UIBK metadata](https://huggingface.co/api/models/DataScience-UIBK/uibk-embedding-v1), and [Intern upstream license](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/LICENSE-QWEN).

Liquid d1 uses LFM Open License v1.0, which defines commercial use broadly, sets a $10 million annual-revenue threshold, and includes a qualified nonprofit research provision plus redistribution conditions.
The owner must determine the applicable entity and whether intended benchmarking and result publication are covered before execution or redistribution.
This report records that unresolved requirement without accepting terms or deciding eligibility.
Source: [exact pinned Liquid license](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE).

The next map questions should choose the typed-decision benchmark contract, the UIBK full-versus-dense scope and ARM64 dependency strategy, and the isolated environment policy before catalog implementation.
LightOnOCR and UltraBERT can then proceed to explicit CPU load audits, smoke runs, and task-correct dataset benchmarks in their own dependency cohorts.
