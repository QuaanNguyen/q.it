# Local CPU serving for issue 81

Research checked on 2026-10-13 against publisher model cards, checkpoint configuration files, official implementation source, and package metadata.
This note supplies implementation prerequisites for the remaining forty candidates in [the text shortlist](sub-billion-text-models.md) and [the multimodal shortlist](sub-billion-multimodal-models.md).
It establishes API and configuration evidence; actual successful checkpoint inference must be recorded separately.

## Dependency choices

A conservative shared environment can pin PyTorch 2.9.1, Transformers 4.57.6, Sentence Transformers 5.3.0, and Diffusers 0.36.0 on Python 3.10 or later.
The published package dependency declarations admit that combination, but dependency resolution alone does not prove every publisher's remote code executes successfully.
Sources: [Transformers 4.57.6 metadata](https://pypi.org/pypi/transformers/4.57.6/json), [Sentence Transformers 5.3.0 metadata](https://pypi.org/pypi/sentence-transformers/5.3.0/json), [Diffusers 0.36.0 metadata](https://pypi.org/pypi/diffusers/0.36.0/json), and [PyTorch 2.9.1 metadata](https://pypi.org/pypi/torch/2.9.1/json).

Sentence Transformers 6.1.0 requires Transformers at least 5.0.0 and therefore cannot be combined with Transformers 4.57.6.
The current releases inspected were Transformers 5.19.0, Sentence Transformers 6.1.0, Diffusers 0.41.0, and PyTorch 2.14.1.
Sources: [Sentence Transformers metadata](https://pypi.org/pypi/sentence-transformers/json), [Transformers metadata](https://pypi.org/pypi/transformers/json), [Diffusers metadata](https://pypi.org/pypi/diffusers/json), and [PyTorch metadata](https://pypi.org/pypi/torch/json).

Transformers 4.57.6 has native configuration mappings for Granite, Qwen3, Moonshine, ModernBERT, and SmolVLM.
Nomic's text architecture requires its publisher code with that Transformers version; its card says native text-only loading becomes available with Transformers 5.5.0 and Sentence Transformers 5.3.0.
Sources: [4.57.6 architecture mappings](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/configuration_auto.py) and [Nomic card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

Use the official CPU wheel index for Linux, because the ordinary PyPI Linux PyTorch distributions declare accelerator package dependencies.
macOS uses the ordinary supported PyTorch distribution, with CPU placement explicitly selected even on Apple Silicon.
Float32 and eager attention provide a portable baseline, and every learned model component must remain on CPU.
Sources: [PyTorch installation instructions](https://pytorch.org/get-started/locally/) and [PyTorch package dependencies](https://pypi.org/pypi/torch/2.9.1/json).

## Text generation

Qwen3 requires its tokenizer chat template with `enable_thinking=False` for a strictly non-thinking baseline.
The default enables thinking, and a textual `/no_think` instruction does not provide the same hard guarantee.
The publisher recommends temperature 0.7, top-p 0.8, and top-k 20 for non-thinking sampling; a deterministic greedy smoke test should be identified as a different decoding configuration.
Source: [Qwen3 publisher card](https://huggingface.co/Qwen/Qwen3-0.6B).

Granite 4.0 350M is the dense canonical checkpoint and can use native causal generation with the checkpoint's own chat template.
Source: [IBM Granite publisher card](https://huggingface.co/ibm-granite/granite-4.0-350m).

Pythia 410M is a pretrained completion model rather than an instruction model.
Pass a raw prompt to its tokenizer and causal generator, return only newly generated tokens, and keep its completion results in their own cohort.
Source: [EleutherAI Pythia card](https://huggingface.co/EleutherAI/pythia-410m).

## Embedding preparation

Load complete Sentence Transformers checkpoints so their pooling, normalization, and any projection modules survive.
Select an explicit query or document prompt for each role, and prevent a checkpoint's default prompt from being applied to the wrong role.
The inspected public configurations for Qwen3 Embedding and Snowflake have no default prompt name.
Sources: [Qwen3 Sentence Transformers configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config_sentence_transformers.json) and [Snowflake configuration](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/blob/e58a8f756156a1293d763f17e3aae643474e9b8a/config_sentence_transformers.json).

| Checkpoint family | Query preparation | Document preparation | Pooling and output |
| --- | --- | --- | --- |
| Qwen3 Embedding 0.6B | Stored `query` prompt: `Instruct: Given a web search query, retrieve relevant passages that answer the query` followed by a newline and `Query:` | Stored `document` prompt is empty | Last-token pooling, normalized 1024 dimensions |
| EmbeddingGemma 300M | `task: search result \| query: ` | `title: none \| text: ` when no title is supplied | Retain complete Sentence Transformers projection stack, normalized 768 dimensions |
| BGE small/base English v1.5 | `Represent this sentence for searching relevant passages: ` | Empty prefix | CLS pooling, normalized 384/768 dimensions |
| E5 small/base v2 | `query: ` | `passage: ` | Attention-mask-aware mean pooling, normalized 384/768 dimensions |
| Jina embeddings v2 base English | Empty prefix | Empty prefix | Attention-mask-aware mean pooling, normalized 768 dimensions |
| Nomic text v1.5 | `search_query: ` | `search_document: ` | Attention-mask-aware mean pooling, normalize full 768 dimensions |
| GTE ModernBERT base | Empty prefix | Empty prefix | CLS pooling, normalize 768 dimensions |
| Snowflake Arctic embed m v1.5 | Stored `query` prompt: `Represent this sentence for searching relevant passages: ` | Empty prefix | CLS pooling, normalized 768 dimensions |

Sources: [Qwen3 card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [EmbeddingGemma card](https://huggingface.co/google/embeddinggemma-300m), [BGE card](https://huggingface.co/BAAI/bge-small-en-v1.5), [E5 card](https://huggingface.co/intfloat/e5-small-v2), [Jina card](https://huggingface.co/jinaai/jina-embeddings-v2-base-en), [Nomic card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [GTE card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base), and [Snowflake card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5).

BGE, E5, Jina, Nomic, and GTE have no stored role prompt maps in the inspected configurations, so the explicit preparation above is necessary where the publisher prescribes prefixes.
Nomic and GTE module manifests contain Transformer and Pooling without a Normalize module, so request normalized embeddings explicitly.
Sources: [BGE configuration](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config_sentence_transformers.json), [E5 modules](https://huggingface.co/intfloat/e5-small-v2/blob/ffb93f3bd4047442299a41ebb6fa998a38507c52/modules.json), [Jina configuration](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/322d4d7e2f35e84137961a65af894fda0385eb7a/config_sentence_transformers.json), [Nomic modules](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/e9b6763023c676ca8431644204f50c2b100d9aab/modules.json), and [GTE modules](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/blob/e7f32e3c00f91d699e8c43b53106206bcc72bb22/modules.json).

Nomic's reduced-dimension example applies layer normalization before truncating and then performs L2 normalization.
Any dimensionality reduction must follow that recipe rather than merely slicing an already normalized baseline vector.
Source: [Nomic dimensionality example](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

Jina requires pinned trusted publisher code to preserve its ALiBi architecture; loading ordinary BERT would discard required checkpoint weights.
EmbeddingGemma remains gated, so an approved authenticated download is required and a missing authorization must be reported as a loading prerequisite rather than a benchmark failure.
Sources: [Jina troubleshooting](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) and [EmbeddingGemma access requirements](https://huggingface.co/google/embeddinggemma-300m).

## Reranking

All six selected rerankers can use native sequence classification of query/document pairs and a single output score per pair.
Preserve raw logits for score transport and rank by descending relevance, because an optional sigmoid changes score calibration but not order.
Avoid Sentence Transformers' convenience activation defaults when exact published raw-score behavior is desired.
Sources: [BGE reranking example](https://huggingface.co/BAAI/bge-reranker-base), [MiniLM L6 card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), and [GTE ModernBERT reranker card](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base).

## Image inference

SmolVLM and SmolVLM2 use their own multimodal processor and checkpoint chat template with an image plus text message.
The original SmolVLM card uses `AutoModelForVision2Seq`; SmolVLM2 uses `AutoModelForImageTextToText` and supports still images.
Use CPU float32 and eager attention, and slice the input-token prefix off decoder-only generation before decoding the answer.
Sources: [SmolVLM card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), [SmolVLM2 card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct), and [SmolVLM architecture documentation](https://huggingface.co/docs/transformers/model_doc/smolvlm).

Canonical Microsoft Florence checkpoints contain custom configuration and model mappings that differ from the converted checkpoints used by native Transformers Florence examples.
The native documentation's `florence-community` checkpoint does not establish direct compatibility for the pinned canonical Microsoft weights.
Use pinned Microsoft processor/model code through `AutoProcessor` and `AutoModelForCausalLM` with trusted publisher code, or explicitly verify an audited conversion before using a native class.
The publisher model source depends on `einops` and `timm`, and its inner language model explicitly inherits `GenerationMixin` while the outer model delegates generation.
Sources: [canonical Florence configuration](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/config.json), [canonical Florence source](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/modeling_florence2.py), and [native Florence documentation](https://huggingface.co/docs/transformers/model_doc/florence2).

Florence receives an explicit task token such as `<CAPTION>` or `<OCR>`, decodes with special tokens retained, and passes the result through processor postprocessing with the task and original image size.
Do not replace an arbitrary vision question with captioning while presenting its score as question answering.
Source: [Microsoft Florence publisher example](https://huggingface.co/microsoft/Florence-2-base).

GIT base and BLIP image-captioning large should use caption-specific benchmark cases and their own processors.
GIT's image-only caption generation uses `AutoProcessor` and `AutoModelForCausalLM` with image pixel values, while BLIP uses `BlipProcessor` and `BlipForConditionalGeneration` with an image and optional caption prefix.
Neither canonical checkpoint is a generic conversational visual assistant.
Sources: [GIT architecture documentation](https://huggingface.co/docs/transformers/model_doc/git), [GIT publisher card](https://huggingface.co/microsoft/git-base), and [BLIP CPU example](https://huggingface.co/Salesforce/blip-image-captioning-large).

## Audio inference

Whisper and Distil-Whisper load through `AutoModelForSpeechSeq2Seq` and `AutoProcessor`, or the automatic speech recognition pipeline.
Decode uploaded audio to a mono floating waveform, resample to the processor's required sampling rate, and pass the actual rate to preprocessing.
Use CPU float32 and transcription rather than translation, with the same English speech subset for comparisons to the English-only models.
Sources: [Whisper publisher card](https://huggingface.co/openai/whisper-tiny) and [Distil-Whisper CPU example](https://huggingface.co/distil-whisper/distil-small.en).

Moonshine uses `MoonshineForConditionalGeneration` with an `AutoProcessor` that receives raw audio rather than Whisper log-Mel features.
The publisher caps total generated sequence length at approximately 6.5 tokens per audio second to avoid hallucination loops.
Preserve that duration-based limit with a small nonzero minimum for very short inputs and record it with decoding settings.
Source: [Moonshine publisher example](https://huggingface.co/moonshine-ai/moonshine-tiny).

## Diffusion inference

All five BK-SDM checkpoints declare `StableDiffusionPipeline` and `PNDMScheduler`.
Their scheduler configurations use scaled-linear betas from 0.00085 to 0.012, 1000 training timesteps, skipped PRK steps, `set_alpha_to_one=False`, and a one-step offset.
The two v2 checkpoints explicitly specify epsilon prediction, so do not infer velocity prediction from their SD2 ancestry.
Sources: [BK-SDM base scheduler](https://huggingface.co/nota-ai/bk-sdm-base/blob/0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0/scheduler/scheduler_config.json), [small scheduler](https://huggingface.co/nota-ai/bk-sdm-small/blob/572238db7ed3a10858900803f3fc8cca53e893e0/scheduler/scheduler_config.json), [tiny scheduler](https://huggingface.co/nota-ai/bk-sdm-tiny/blob/0364108e53b7f7f4d2585e817a0b7a83dc261cfa/scheduler/scheduler_config.json), [v2 small scheduler](https://huggingface.co/nota-ai/bk-sdm-v2-small/blob/ad82c51a3b568e855a7f79c5b893ea2f9ebb197f/scheduler/scheduler_config.json), and [v2 tiny scheduler](https://huggingface.co/nota-ai/bk-sdm-v2-tiny/blob/68277af553777858cd47e133f92e4db47321bc74/scheduler/scheduler_config.json).

Load the checkpoint's scheduler configuration intact and set the complete pipeline to CPU float32.
Use a CPU generator with a recorded seed, batch size one, explicit image dimensions, guidance, and denoising steps.
Sources: [BK-SDM v2 example](https://huggingface.co/nota-ai/bk-sdm-v2-tiny) and [Diffusers loading documentation](https://huggingface.co/docs/diffusers/using-diffusers/loading).

The v1 checkpoint indexes include a learned safety checker, which must be explicitly disabled for the audited sub-billion stack.
The v2 indexes already declare no checker and `requires_safety_checker=False`.
Override the checker and unused feature extractor with `None` and explicitly set `requires_safety_checker=False`, then count unique learned parameters across U-Net, text encoder, and the full VAE.
Sources: [v1 model index](https://huggingface.co/nota-ai/bk-sdm-base/blob/0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0/model_index.json), [v2 model index](https://huggingface.co/nota-ai/bk-sdm-v2-tiny/blob/68277af553777858cd47e133f92e4db47321bc74/model_index.json), and [Stable Diffusion component contract](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img).

## Current text checkpoint pins

The following immutable revisions were returned by each canonical publisher's [Hugging Face model metadata endpoint](https://huggingface.co/docs/hub/api) on the research date.
The prior multimodal audit already contains immutable revisions for its twenty-one checkpoints.

| Canonical checkpoint | Revision |
| --- | --- |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) | `7ae557604adf67be50417f59c2c2f167def9a775` |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951) | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| [ibm-granite/granite-4.0-350m](https://huggingface.co/ibm-granite/granite-4.0-350m/tree/bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2) | `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2` |
| [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m/tree/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f) | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m/tree/57c266a740f537b4dc058e1b0cda161fd15afa75) | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a) | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52) | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd) | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/tree/322d4d7e2f35e84137961a65af894fda0385eb7a) | `322d4d7e2f35e84137961a65af894fda0385eb7a` |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/e9b6763023c676ca8431644204f50c2b100d9aab) | `e9b6763023c676ca8431644204f50c2b100d9aab` |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22) | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` |
| [Snowflake/snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/tree/e58a8f756156a1293d763f17e3aae643474e9b8a) | `e58a8f756156a1293d763f17e3aae643474e9b8a` |
| [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/tree/7b0235231ca2674cb8ca8f022859a6eba2b1c968) | `7b0235231ca2674cb8ca8f022859a6eba2b1c968` |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base/tree/2cfc18c9415c912f9d8155881c133215df768a70) | `2cfc18c9415c912f9d8155881c133215df768a70` |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large/tree/55611d7bca2a7133960a6d3b71e083071bbfc312) | `55611d7bca2a7133960a6d3b71e083071bbfc312` |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/tree/f7481e6055501a30fb19d090657df9ec1f79ab2c) | `f7481e6055501a30fb19d090657df9ec1f79ab2c` |
