# Issue 81 local CPU validation

Validated on October 13, 2026, on local macOS ARM64.
All forty remaining canonical checkpoints completed a real q.it request through the optional local CPU provider.
This covers three text generators, ten embedding models, six rerankers, eight vision models, eight speech models, and five image generators.
The three previously submitted text checkpoints remain outside this local batch.

## Scope and procedure

Every checkpoint was loaded before measurement using its immutable publisher revision.
All inference parameters were audited as CPU float32, with four CPU threads and fewer than one billion unique learned parameters across the complete required stack.
The checked libraries were PyTorch 2.9.1, Transformers 4.57.6, Sentence Transformers 5.3.0, and Diffusers 0.36.0.
The check used one synthetic case, zero warmups, one measured iteration, and a sixteen-token generation limit.
GIT and BLIP received caption prefixes, while Florence received its explicit caption task token.
Pythia used a separate completion pack.
The speech case was a generated tone, which validates transport without establishing recognition accuracy.
The five diffusion checks used explicitly recorded 64 by 64 images, two steps, guidance 7.5, and seed 42, with the optional learned safety checker disabled.
A separate browser-started built-in image run also completed the normal 512 by 512, twenty-step profile with guidance 7.5 and seed 42.
Its PNG was decoded and checked at the requested dimensions, then retained with its sample and JSONL export.
These are compatibility checks and do not establish official dataset scores, comparative speed, or Raspberry Pi performance.

## Checkpoints

| Task | Canonical checkpoint and pinned revision | Loaded parameters | Outcome |
|---|---|---:|---|
| text generation | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | 596,049,920 | Passed |
| text generation | [ibm-granite/granite-4.0-350m](https://huggingface.co/ibm-granite/granite-4.0-350m/tree/bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2) | 352,379,904 | Passed |
| text generation | [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m/tree/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f) | 405,334,016 | Passed |
| embedding | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | 595,776,512 | Passed |
| embedding | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m/tree/57c266a740f537b4dc058e1b0cda161fd15afa75) | 307,581,696 | Passed |
| embedding | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | 33,360,000 | Passed |
| embedding | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a) | 109,482,240 | Passed |
| embedding | [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52) | 33,360,000 | Passed |
| embedding | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd) | 109,482,240 | Passed |
| embedding | [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en/tree/322d4d7e2f35e84137961a65af894fda0385eb7a) | 137,368,320 | Passed |
| embedding | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/e9b6763023c676ca8431644204f50c2b100d9aab) | 136,731,648 | Passed |
| embedding | [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22) | 149,014,272 | Passed |
| embedding | [Snowflake/snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/tree/e58a8f756156a1293d763f17e3aae643474e9b8a) | 108,891,648 | Passed |
| reranking | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | 22,713,601 | Passed |
| reranking | [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/tree/7b0235231ca2674cb8ca8f022859a6eba2b1c968) | 33,360,385 | Passed |
| reranking | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base/tree/2cfc18c9415c912f9d8155881c133215df768a70) | 278,044,417 | Passed |
| reranking | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large/tree/55611d7bca2a7133960a6d3b71e083071bbfc312) | 559,891,457 | Passed |
| reranking | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) | 567,755,777 | Passed |
| reranking | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/tree/f7481e6055501a30fb19d090657df9ec1f79ab2c) | 149,605,633 | Passed |
| image to text | [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964) | 256,484,928 | Passed |
| image to text | [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/tree/a7da5b986cb59b408707209984f360a5f4ad7e47) | 507,482,304 | Passed |
| image to text | [HuggingFaceTB/SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/tree/067788b187b95ebe7b2e040b3e4299e342e5b8fd) | 256,484,928 | Passed |
| image to text | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467) | 507,482,304 | Passed |
| image to text | [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base/tree/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac) | 231,414,016 | Passed |
| image to text | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594) | 776,465,408 | Passed |
| image to text | [microsoft/git-base](https://huggingface.co/microsoft/git-base/tree/1f7fe8444292beb4a259e3a5b6eba440cd5999d4) | 176,619,066 | Passed |
| image to text | [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large/tree/353689b859fcf0523410b1806dace5fb46ecdf41) | 446,290,492 | Passed |
| speech to text | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af) | 37,760,640 | Passed |
| speech to text | [openai/whisper-base](https://huggingface.co/openai/whisper-base/tree/e37978b90ca9030d5170a5c07aadb050351a65bb) | 72,593,920 | Passed |
| speech to text | [openai/whisper-small](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) | 241,734,912 | Passed |
| speech to text | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e) | 763,857,920 | Passed |
| speech to text | [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en/tree/9e4a67ca4569c30be43a3fe7fba1621e504f0093) | 166,132,224 | Passed |
| speech to text | [distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en/tree/6e61418885eaf4d5cc9f64e508e80ac5b4c052b7) | 394,375,168 | Passed |
| speech to text | [moonshine-ai/moonshine-tiny](https://huggingface.co/moonshine-ai/moonshine-tiny/tree/390624ed33d594443aa4aa221f5b9f283b545b5a) | 27,092,736 | Passed |
| speech to text | [moonshine-ai/moonshine-base](https://huggingface.co/moonshine-ai/moonshine-base/tree/7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf) | 61,513,920 | Passed |
| text to image | [nota-ai/bk-sdm-base](https://huggingface.co/nota-ai/bk-sdm-base/tree/0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0) | 786,099,307 | Passed |
| text to image | [nota-ai/bk-sdm-small](https://huggingface.co/nota-ai/bk-sdm-small/tree/572238db7ed3a10858900803f3fc8cca53e893e0) | 689,061,227 | Passed |
| text to image | [nota-ai/bk-sdm-tiny](https://huggingface.co/nota-ai/bk-sdm-tiny/tree/0364108e53b7f7f4d2585e817a0b7a83dc261cfa) | 530,099,307 | Passed |
| text to image | [nota-ai/bk-sdm-v2-small](https://huggingface.co/nota-ai/bk-sdm-v2-small/tree/ad82c51a3b568e855a7f79c5b893ea2f9ebb197f) | 909,829,227 | Passed |
| text to image | [nota-ai/bk-sdm-v2-tiny](https://huggingface.co/nota-ai/bk-sdm-v2-tiny/tree/68277af553777858cd47e133f92e4db47321bc74) | 750,867,307 | Passed |

## Corrections verified by real requests

SmolVLM2 requires the pinned number-to-words dependency.
Canonical Florence uses greedy decoding because the text streamer cannot operate with beam search.
Its pinned publisher code expects a legacy cache interface, so this profile disables the incompatible generation cache and records that setting.
Florence outputs pass through the publisher processor with the original task token and image dimensions.
Moonshine receives right-hand zero padding only when a clip is shorter than its convolutional minimum, while preserving the original duration.
Public checkpoints load without cached authentication, so an expired unrelated login does not block them.
EmbeddingGemma was checked after the user enabled the required gated access.
Text timing uses native token callbacks before text buffering, with exact generated token counts.

## Reproduce and inspect

Install the optional serving environment described in the [CPU provider guide](../../providers/local-cpu/README.md), then run:

```bash
cargo build --locked -p qit
python scripts/check-local-cpu.py --trust-publisher-code \
  --exclude Qwen/Qwen2.5-0.5B-Instruct \
  --exclude HuggingFaceTB/SmolLM2-135M-Instruct \
  --exclude HuggingFaceTB/SmolLM2-360M-Instruct
```

The [checker](../../scripts/check-local-cpu.py) saves model metadata, server logs, installed synthetic packs, q.it logs, per-model JSONL run exports, and a combined JSON report under `.scratch/local-cpu-validation`.
These local artifacts remain ignored rather than embedding weights, credentials, or large generated outputs in the repository.
Per-model records retain the checkpoint revision, provider source hash, validated executable hash, and run identifier.
The complete Python environment is recorded locally in `.scratch/local-cpu-environment.txt`.
A separate dashboard can inspect the isolated results:

```bash
QIT_HOME=.scratch/local-cpu-validation/state qit dashboard --port 2473
```
