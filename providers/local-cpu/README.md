# Local CPU provider

This optional server exposes the 43 canonical checkpoints from [issue 81](https://github.com/QuaanNguyen/q.it/issues/81) through q.it's local CPU provider.
It covers text generation, text embeddings, reranking, image questions, captioning and OCR, speech transcription, and the five BK-SDM image generators.
Starting this server explicitly loads and downloads the selected checkpoint before any benchmark measurement begins.

The server pins each checkpoint revision and counts the complete loaded inference stack.
It rejects stacks at or above one billion parameters and keeps every inference parameter on CPU in float32.
Each server process loads one model and accepts one inference request at a time.
Model-specific input preparation, package versions, checkpoint revisions, CPU thread settings, and loaded parameter counts are recorded with benchmark runs.

## Install

Use Python 3.11, 3.12, or 3.13.
From the repository root, create a dedicated environment:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
```

On Linux, install the CPU-only PyTorch wheels first:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, the normal PyTorch wheels are appropriate because the provider explicitly selects CPU.
Install the serving package and its pinned model libraries:

```bash
python -m pip install -e 'providers/local-cpu[models]'
```

List every pinned checkpoint and its task:

```bash
qit-cpu list
```

## Text generation and raw completion

Start a model server in one terminal:

```bash
qit-cpu serve --model Qwen/Qwen3-0.6B --threads 4
```

Run q.it from another terminal:

```bash
qit run core/text-generation-smoke --provider local-cpu --model Qwen/Qwen3-0.6B
```

Qwen3 uses its chat template with thinking explicitly disabled.
Instruction checkpoints use their publisher's chat templates.
Pythia receives raw completion text and should use a separate completion benchmark rather than an instruction ranking.
Text and image-question generation use explicitly recorded greedy decoding, stream text, and report exact generated token counts.
Token timing comes from the generation callbacks before text buffering; throughput measures the interval between the first and last generated tokens.
Time to first token and token throughput remain absent when the endpoint cannot supply the required observations.

## Embeddings and reranking

Start an embedding server:

```bash
qit-cpu serve --model intfloat/e5-small-v2
qit run core/embedding-retrieval --provider local-cpu --model intfloat/e5-small-v2
```

The two commands above run in separate terminals.
Sentence Transformers retains each checkpoint's pooling and projection modules.
The provider applies the recorded query and document instructions and normalizes the complete vectors without reducing their dimensions.
Original benchmark text and document order are preserved.
A pack can explicitly override the query or document prefix, including supplying an empty prefix.

For reranking, start the desired cross-encoder and select the reranking benchmark:

```bash
qit-cpu serve --model cross-encoder/ms-marco-MiniLM-L6-v2
qit run core/reranking-relevance --provider local-cpu --model cross-encoder/ms-marco-MiniLM-L6-v2
```

Embedding and reranking runs do not report generation-token throughput or time to first token.
Google's EmbeddingGemma checkpoint requires approved Hugging Face access before downloading its gated weights.
The provider uses the normal Hugging Face authentication mechanism without saving credentials in run metadata.

## Images, captions, and OCR

SmolVLM and SmolVLM2 instruction models accept the existing image-question benchmark:

```bash
qit-cpu serve --model HuggingFaceTB/SmolVLM-256M-Instruct
qit run core/image-to-text-smoke --provider local-cpu --model HuggingFaceTB/SmolVLM-256M-Instruct
```

GIT and BLIP require a captioning pack whose text is an explicit caption prefix, such as `a photograph of`.
Florence requires a pack with an explicit supported task token, such as `<CAPTION>` or `<OCR>`.
The provider rejects generic image-question prompts for Florence instead of changing the benchmark question.
Canonical Florence generation disables the incompatible modern key-value cache and records this decoding setting.
Its publisher processor postprocesses caption and OCR tokens using the original image dimensions.
Caption, OCR, and conversational image-question quality scores belong in separate benchmarks.

Jina, Nomic, and Florence require pinned publisher code in this environment.
Start those checkpoints with `--trust-publisher-code` to explicitly enable their pinned code revisions.
Both the model revision and the code repository revision are pinned in the profile.

## Speech

Start any of the listed Whisper, Distil-Whisper, or Moonshine checkpoints and use the speech provider:

```bash
qit-cpu serve --model openai/whisper-tiny
qit run core/speech-to-text-smoke --provider local-cpu --model openai/whisper-tiny
```

Audio is decoded locally, converted to mono, and resampled to the processor's required sampling rate.
The current server accepts audio cases up to thirty seconds long.
Whisper multilingual checkpoints are configured for English transcription rather than translation.
Moonshine uses a duration-based decoding cap within the requested token limit.
Very short clips receive recorded right-hand zero padding to satisfy its convolutional input minimum while retaining the original audio duration.
The built-in tone checks transport only; use spoken audio and reference transcripts in a separate pack for recognition quality.

## Image generation

Start one of the five BK-SDM profiles:

```bash
qit-cpu serve --model nota-ai/bk-sdm-tiny
qit run core/text-to-image-smoke --provider local-cpu --model nota-ai/bk-sdm-tiny --timeout 600
```

The complete U-Net, text encoder, and VAE are included in the loaded parameter audit.
The optional learned safety checker is disabled in this explicitly recorded benchmark profile.
The publisher's scheduler configuration is retained, including the epsilon prediction setting of the v2 checkpoints.

Default generation uses one 512 by 512 image, twenty denoising steps, guidance 7.5, and CPU seed 42.
A benchmark case can supply explicit image settings to define another reproducible profile.
q.it validates the actual PNG and its requested dimensions, stores it with the sample, includes it in JSONL exports, and shows a downloadable preview in run details.
Image generation does not invent a quality score, token count, or time to first token.

## Local checks

The normal test suite uses loopback protocol fixtures and never downloads model weights:

```bash
python -m unittest discover -s providers/local-cpu/tests
cargo test --locked --workspace
```

Real checkpoint checks are a separate explicit operation and require the serving environment and model downloads.
The following command checks all canonical models one at a time:

```bash
cargo build --locked -p qit
python scripts/check-local-cpu.py --trust-publisher-code
```

Pass one or more canonical model IDs to check a subset, or repeat `--exclude` to omit models.
The checker uses isolated local state, preserves per-model logs and JSONL run exports, and writes a combined report under `.scratch/local-cpu-validation`.
It checks one synthetic case per model with four CPU threads, no warmup, one iteration, and a sixteen-token limit.
Image generators use an explicitly reduced 64 by 64, two-step profile for this loading and transport check.
Successful checks are reused only when the provider source, q.it executable, validation cases, checker, and thread settings match.
A successful smoke request establishes local loading and transport behavior, not an official evaluation score or Raspberry Pi performance.
Linux ARM64 package availability and real Pi memory fit require separate qualification.
