# Additional CPU models for text benchmarks

Research checked on 2026-10-09 against canonical publisher model cards, Hugging Face metadata, Ollama listings, and serving documentation.
This shortlist proposes six text-generation models, ten text-embedding models, and six rerankers for future validation.
It excludes the previously requested EmbeddingGemma 2, Qwen3.5 0.8B, Llama3.2 1B, and Gemma3 1B.
None of these proposed models has been tested on Sol or Raspberry Pi 5 as part of this research.

The limit is strictly fewer than one billion total learned parameters, including embeddings and output heads, rather than an active-expert count or compressed file size.
The selected checkpoints are official releases from established model teams or the original Sentence Transformers maintainers.
An Ollama package is an optional serving artifact for a named canonical model, rather than an additional model choice.

## Text generation

All six models have native Hugging Face Transformers implementations that can execute on CPU.
Five are instruction models suitable for the existing text-generation smoke test and imported MMLU-Pro or GSM8K prompt packs.
Pythia is a deliberately labeled pretrained baseline and requires a completion-oriented prompt path before comparison with chat models.

| Canonical checkpoint and source | Publisher | Total model size | License | CPU serving route and prerequisite | Benchmark role |
| --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Alibaba Qwen | About 600M in the model card | Apache-2.0 | Transformers with a CPU PyTorch build, or [Ollama qwen3:0.6b](https://ollama.com/library/qwen3:0.6b); pin thinking behavior | Modern multilingual instruction and reasoning baseline |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Alibaba Qwen | 494.0M | Apache-2.0 | Transformers with CPU placement, or [Ollama qwen2.5:0.5b](https://ollama.com/library/qwen2.5:0.5b) | Smaller instruction baseline with mathematics training |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Hugging Face | 134.5M | Apache-2.0 | Transformers CPU is explicitly documented, or [Ollama smollm2:135m](https://ollama.com/library/smollm2:135m) | Very small instruction model for latency and memory tradeoffs |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Hugging Face | 361.8M | Apache-2.0 | Transformers CPU, or [Ollama smollm2:360m](https://ollama.com/library/smollm2:360m) | Same training family at a larger edge budget |
| [ibm-granite/granite-4.0-350m](https://huggingface.co/ibm-granite/granite-4.0-350m) | IBM Research | 352.4M | Apache-2.0 | CPU Transformers; select the dense checkpoint, whose card explicitly describes CPU placement | Instruction, extraction, question answering, and tool-use baseline |
| [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m) | EleutherAI | 405.3M learned floating-point elements, marketed as 410M | Apache-2.0 | CPU Transformers generation; requires raw completion support or a documented server chat template | Reproducible pretrained control, separate from instruction rankings |

The current q.it Transformers provider sends `/v1/chat/completions`, while [Transformers Serve](https://huggingface.co/docs/transformers/serve-cli/serving) also documents a separate completion endpoint.
Do not silently invent a Pythia chat template or treat base-model prompt failures as instruction-model quality scores.
CPU-only execution should use CPU PyTorch, explicit CPU placement, float32 where needed, and an ordinary attention implementation rather than a CUDA-only Flash Attention dependency.
Qwen3 reasoning settings affect visible answers, generated tokens, and TTFT, so record and compare the same setting for every pass.
An Ollama run must prove zero GPU offload from its loaded-model state and record its exact artifact digest and quantization.

## Text embeddings

These ten canonical embedding checkpoints fit semantic search and the existing embedding-retrieval check or imported SciFact retrieval pack.
All use CPU-capable Transformers or Sentence Transformers implementations.
TEI supports their architecture families, and its [supported model list](https://huggingface.co/docs/text-embeddings-inference/supported_models) explicitly names Qwen3 Embedding, EmbeddingGemma, and Nomic Embed.
The [TEI CPU guide](https://huggingface.co/docs/text-embeddings-inference/local_cpu) documents CPU execution, while [Hugging Face Serve](https://github.com/huggingface/hf-serve) provides the separate Sentence Transformers serving route used by q.it.
Select a CPU build for the actual machine architecture and validate `/embed` or `/v1/embeddings` before scheduling the dataset run.

| Canonical checkpoint and source | Publisher | Total model size | License | CPU serving route | Required retrieval behavior |
| --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Alibaba Qwen | About 600M; 595.8M root checkpoint floating-point elements | Apache-2.0 | TEI CPU, or Hugging Face Serve with Sentence Transformers | Last-token pooling, normalized vectors, query instruction, 1024 dimensions by default |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | Google DeepMind | About 300M, with projection modules retained | Gemma terms; gated download | TEI CPU or Hugging Face Serve, with approved Hub access and float32 | Distinct query and document prompts; retain Sentence Transformers projection modules and 768 dimensions |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Beijing Academy of Artificial Intelligence | 33.4M | MIT | TEI CPU or Hugging Face Serve | CLS pooling, 384 dimensions; recommended query-only instruction |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Beijing Academy of Artificial Intelligence | 109.5M | MIT | TEI CPU or Hugging Face Serve | CLS pooling, 768 dimensions; matched larger BGE control |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | Microsoft E5 research team | 33.4M | MIT | TEI CPU or Hugging Face Serve | Mean pooling, 384 dimensions, `query: ` and `passage: ` prefixes |
| [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | Microsoft E5 research team | 109.5M | MIT | TEI CPU or Hugging Face Serve | Mean pooling, 768 dimensions, the same asymmetric prefixes |
| [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) | Jina AI | 137.4M | Apache-2.0 | TEI JinaBERT CPU; Sentence Transformers path requires its documented custom-code support | Mean pooling, 768 dimensions, ALiBi long-context encoder |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Nomic AI | 136.7M | Apache-2.0 | TEI CPU or Hugging Face Serve; modern Transformers supports the text architecture without custom code | `search_query: ` and `search_document: ` prefixes; 768 dimensions unless explicitly changed |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Alibaba Tongyi Lab | 149.0M | Apache-2.0 | TEI with ModernBERT support, or Hugging Face Serve with Transformers at least 4.48 | CLS pooling, 768 dimensions; optional GPU Flash Attention is unnecessary for CPU |
| [Snowflake/snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) | Snowflake | 108.9M | Apache-2.0 | TEI BERT CPU or Hugging Face Serve | CLS pooling, query-only retrieval instruction, 768 dimensions for the baseline |

q.it currently sends the query and candidate passages together without adding model-specific prompts.
For fair quality measurements, preserve the original dataset and add explicit, recorded query/document preparation before invoking a server.
The embedding route must preserve each model's published pooling and projection behavior.
Applying one default prompt to both roles is insufficient for E5, Nomic, Qwen3 Embedding, and EmbeddingGemma retrieval.
BGE v1.5 can work without an instruction, but its publisher recommends a query-only instruction for short-query passage retrieval.
EmbeddingGemma explicitly forbids float16 activations, so use float32 as the portable CPU baseline.
Jina's native Sentence Transformers example requires `trust_remote_code`; pin and inspect that publisher code, or use the supported TEI implementation.
Nomic's card states that Transformers 5.5 and Sentence Transformers 5.3 remove that custom-code requirement for its text-only architecture.

## Reranking

These six models are single-score cross-encoders suitable for the existing reranking-relevance check and a future SciFact reranking pack using identical candidate passages.
They are original maintained relevance models rather than embedding models repurposed as rerankers.
q.it already supports the TEI `/rerank` request shape.
The [TEI reranking guide](https://huggingface.co/docs/text-embeddings-inference/quick_tour#re-rankers-and-sequence-classification) documents single-class sequence classification and explicitly uses BGE Reranker Large.

| Canonical checkpoint and source | Publisher | Total model size | License | CPU serving route and prerequisite | Benchmark role |
| --- | --- | --- | --- | --- | --- |
| [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | Original Sentence Transformers maintainers, using Microsoft MiniLM | 22.7M | Apache-2.0 | TEI CPU BERT classifier path verified in source; checkpoint smoke run pending | Fast, widely used English relevance baseline |
| [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | Original Sentence Transformers maintainers, using Microsoft MiniLM | 33.4M | Apache-2.0 | TEI CPU BERT classifier path verified in source; checkpoint smoke run pending | Same family at twice the encoder depth |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | Beijing Academy of Artificial Intelligence | 278.0M | MIT | TEI CPU XLM-RoBERTa sequence classification | English and Chinese relevance baseline |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | Beijing Academy of Artificial Intelligence | 559.9M | MIT | TEI CPU XLM-RoBERTa; documented TEI reranking example | Larger matched BGE control |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Beijing Academy of Artificial Intelligence | 567.8M | Apache-2.0 | TEI CPU XLM-RoBERTa sequence classification; float32 portable baseline | Newer multilingual relevance model |
| [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | Alibaba Tongyi Lab | 149.6M checkpoint floating-point elements, advertised as 149M | Apache-2.0 | Explicit first-party TEI CPU container example; its example targets Linux AMD64 | ModernBERT relevance model with longer input support |

Both MiniLM configurations specify BERT sequence classification, one output label, and absolute positions: [L6 configuration](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/config.json) and [L12 configuration](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/blob/7b0235231ca2674cb8ca8f022859a6eba2b1c968/config.json).
At TEI revision `56964585069318d01ddeeec9454a952a66f1d3a2`, the [router identifies classification architectures](https://github.com/huggingface/text-embeddings-inference/blob/56964585069318d01ddeeec9454a952a66f1d3a2/router/src/lib.rs#L407) and exposes single-label models as rerankers.
The [Candle backend dispatches BERT on CPU](https://github.com/huggingface/text-embeddings-inference/blob/56964585069318d01ddeeec9454a952a66f1d3a2/backends/candle/src/lib.rs#L287), and its [BERT implementation loads the classification head](https://github.com/huggingface/text-embeddings-inference/blob/56964585069318d01ddeeec9454a952a66f1d3a2/backends/candle/src/models/bert.rs#L375) and implements score prediction.
This verifies the implementation path for both MiniLM candidates; loading their actual checkpoints and calling `/rerank` remain required validation steps.
Sentence Transformers CrossEncoder on CPU is the reference path if a selected TEI release fails, with a compatible reranking HTTP route required before q.it can use that fallback.
CPU inference is available through standard PyTorch classification for all six, but that alone does not prove an HTTP server accepts the checkpoint correctly.
Before the q.it run, validate TEI health, the loaded model identity, a three-document rerank, output order, and the configured truncation length.
Avoid the CUDA container shown in some generic examples and select the actual CPU build.
The Alibaba AMD64 container example is evidence for Sol CPU serving, not evidence for an ARM64 Raspberry Pi binary.
The published MiniLM throughput table was measured on a V100 GPU and must not be copied into a CPU or Pi result table.

## Parameter audit

The table sizes above follow publisher model cards or rounded learned floating-point checkpoint counts, rather than quantized storage size.
The public Hugging Face model API was checked for every listed checkpoint using `https://huggingface.co/api/models/<canonical-checkpoint>`.
Every inspected root checkpoint has fewer than one billion stored tensor elements, even before excluding integer buffers or duplicate tied tensors.
This metadata is a cross-check and does not replace counting unique learned parameters across all model modules at execution time.

For example, [Qwen3 metadata](https://huggingface.co/api/models/Qwen/Qwen3-0.6B) reports 751,632,384 stored BF16 elements, while its publisher advertises approximately 600M model parameters and its configuration ties input/output embeddings.
The issue should preserve the published model count and require an actual unique-parameter count during setup, instead of relabeling that model as a 751M independently parameterized architecture.
[Pythia metadata](https://huggingface.co/api/models/EleutherAI/pythia-410m) reports 405,334,208 F16 elements plus 100,663,296 U8 elements, so the latter must not be counted as learned weights.
Small BERT checkpoints also include 512-element integer position buffers.
For EmbeddingGemma, count the complete Sentence Transformers model with its projection modules rather than only the root backbone file.

## Execution acceptance criteria

1. Pin canonical checkpoint revision, license, tokenizer, chat template or retrieval prompts, pooling, dimensions, server version, dependency lock, precision, and model artifact digest.
2. Count complete unique learned parameters and reject anything at or above one billion parameters, regardless of quantization or active MoE size.
3. Run smoke checks first on CPU-only Sol allocations, then the matching larger dataset packs using identical case IDs and candidate order.
4. Keep CPU thread count, CPU affinity, warmups, measured passes, token limits, timeout, precision, and cache conditions consistent within each comparison cohort.
5. Confirm no accelerator was used and record the loaded-model/runtime evidence with each result.
6. Record request latency and resource use for every task, streaming TTFT and exact token throughput for generation, retrieval/reranking MRR where relevant, and dataset scores only when the scorer actually exists.
7. Save failed runs and diagnostics without ranking them as successful samples, and separate pretrained Pythia from instruction-quality comparisons.
8. Perform native ARM64 Raspberry Pi 5 validation before making compatibility, memory-fit, or throughput claims for that device.
