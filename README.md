# q.it

q.it benchmarks edge models on a Raspberry Pi 5 and stores the results for later comparison.
It works with models already served by Ollama, Hugging Face Transformers Serve, Hugging Face Serve, or Hugging Face Text Embeddings Inference.

q.it does not download a model or start a model server during a benchmark.
This keeps each run predictable and avoids including model setup time by accident.

## Current status

The repository includes six small, self-made benchmarks that confirm a model and server work correctly.
They exercise text generation, text embeddings, image input, audio upload, and reranking without downloading a large dataset.

These built-in checks are smoke tests, not official model evaluations.
Official datasets such as MMLU-Pro, GSM8K, LibriSpeech, OCRBench, MTEB, and BEIR are intended to be separate benchmark packs.
The local pack installer is ready, but an online pack registry and downloadable official packs are not included yet.

See [Built-in benchmarks](docs/built-in-benchmarks.md) for exactly what each bundled check does.

The [CPU model validation plan](https://github.com/QuaanNguyen/q.it/issues/81) proposes 43 additional models below one billion parameters across all six tasks.
See the [text-model research](docs/research/sub-billion-text-models.md) and [multimodal-model research](docs/research/sub-billion-multimodal-models.md) for parameter audits, canonical checkpoints, CPU serving requirements, and pending integrations.

## Quick start on Raspberry Pi OS

### 1. Install q.it

Install a current Rust toolchain on the Pi.

Until q.it is published to crates.io, clone this repository and install it from the checkout:

```bash
git clone https://github.com/QuaanNguyen/q.it.git
cd q.it
cargo install --path qit
```

After a crates.io release is available, installation will be:

```bash
cargo install qit
```

To keep results in a specific directory, set `QIT_HOME` before running q.it:

```bash
export QIT_HOME="$HOME/qit-data"
```

Without that setting, Raspberry Pi OS stores results under `~/.local/share/qit`.

### 2. Start a model server

For a first run, Ollama is the simplest option for a supported GGUF or Ollama model.

Start Ollama in one terminal:

```bash
ollama serve
```

Make a small model available from another terminal:

```bash
ollama pull gemma3:1b
```

### 3. Check compatibility

List the available server types:

```bash
qit provider list
```

List benchmarks that Ollama can run:

```bash
qit benchmark list --provider ollama
```

### 4. Run a benchmark

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model gemma3:1b
```

The default run performs one warmup and three measured passes over every case.
The warmup is excluded from the reported measurements.

### 5. Inspect the result

```bash
qit results
```

Start the optional dashboard when you want charts:

```bash
qit dashboard
```

Open [http://127.0.0.1:2471](http://127.0.0.1:2471) on the Pi.

When working over SSH, forward the dashboard port from your computer:

```bash
ssh -L 2471:127.0.0.1:2471 pi@raspberrypi.local
```

Then open [http://127.0.0.1:2471](http://127.0.0.1:2471) on your computer.

The overview compares models in a grid of latency, time-to-first-token, throughput, quality, CPU, and memory charts.
Choose a benchmark version and machine setup to compare the latest successful run for each model configuration.
Tradeoff charts highlight quality versus latency and provider memory versus latency, while the trend chart tracks repeated runs.
Run history retains every outcome and opens the saved cases and outputs for inspection.

Use **Start benchmark** to choose an installed benchmark, model, provider, and endpoint, then click **Start benchmark**.
The model server must already be running with the model available.
The page shows saved sample progress and retains failures in history.
One browser-started benchmark runs at a time to reduce measurement interference.
CPU and host-memory charts measure the machine running q.it; supplying the model server's local process ID enables a separate process-memory chart.
Keep the dashboard on loopback and use the SSH tunnel when working from another computer.

The ASCII theme uses the q.it wordmark and locally installed [Michelangelus](https://www.microsoft.com/en-us/download/details.aspx?id=108856) for headings.
Install the font on the device running your browser, including your laptop when viewing the Pi through SSH.
Its license prohibits bundling the font files, so devices without it use a serif fallback.
See [the dashboard design research](docs/research/dashboard-ranking-design.md) for the Arena-inspired ranking choices and scoring limits.

## Built-in benchmarks

| Benchmark | Purpose | Supported server |
|-----------|---------|------------------|
| `core/text-generation-smoke` | Checks short text generation, basic instruction following, TTFT, and TPS. | Ollama or Transformers Serve |
| `core/embedding-retrieval` | Checks whether related text is placed near a query in embedding space. | Ollama, Hugging Face Serve, or Text Embeddings Inference |
| `core/image-to-text-smoke` | Sends a generated two-color image and asks the model to identify its colors. | Ollama or Transformers Serve |
| `core/speech-to-text-smoke` | Confirms that an audio file can travel through the transcription endpoint. | Transformers Serve |
| `core/reranking-relevance` | Checks whether the most relevant passage is ranked first. | Text Embeddings Inference |
| `core/text-to-image-smoke` | Reserves the task shape for a future image-generation server. | Not runnable yet |

The speech check uses a generated tone, so it does not measure transcription accuracy.
The text-to-image check has no active server integration and is shown as reserved.

## Supported model servers

### Ollama

Use Ollama for compatible text-generation, text-embedding, and vision models.
It is the preferred path for GGUF models because q.it can use Ollama's exact token count and generation time.

Run a model served on another machine by supplying its address:

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model my-model \
  --base-url http://192.168.1.20:11434
```

### Hugging Face Transformers Serve

The Hugging Face command-line tool downloads and manages model files.
Transformers Serve is the server used by q.it for compatible text, image-to-text, and speech-to-text models.

```bash
transformers serve --port 8000
qit run core/image-to-text-smoke \
  --provider transformers \
  --model organization/model-name
```

Model support varies on Linux ARM64.
A successful connection check means the server responded, not that every Transformers model will fit or run correctly on a Pi.

### Hugging Face Serve

Use [Hugging Face Serve](https://github.com/huggingface/hf-serve) for text embeddings from compatible SentenceTransformers models, including EmbeddingGemma 2.
Install the server with its official CPU dependency setup and a Transformers version that supports the chosen model.
For EmbeddingGemma 2, use Transformers 5.19.0 and SentenceTransformers 6.1.0.
The Hugging Face Serve 0.1.8 lockfile pins Transformers 5.17.0, which does not recognize this model.
Upgrade Transformers inside the server's environment before starting it, and launch the installed server directly so a frozen sync does not restore the older dependency.

Start the server in one terminal:

```bash
MODEL_ID=google/embeddinggemma-2 hf-serve \
  --task feature-extraction --device cpu --dtype float32 \
  --host 127.0.0.1 --port 8080
```

Run the check from another terminal:

```bash
qit run core/embedding-retrieval \
  --provider hf-serve \
  --model google/embeddinggemma-2 \
  --base-url http://127.0.0.1:8080
```

q.it uses the server's OpenAI-compatible embedding endpoint and checks that the requested model appears in its model list.
Setting the model through the server's environment also ensures Hugging Face Serve 0.1.8 reports its identity when SentenceTransformers metadata does not contain it.
The model name must match the name reported by the server, including when serving a local model directory.
The integration currently covers text embeddings.
Each request contains at most four texts to limit the server's CPU memory use.
For larger retrieval cases, q.it combines the batches in input order and measures the total time for the case.
The [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) recommends float32 on most CPUs; float16 can produce invalid embeddings.
CPU validation on another Linux machine does not establish Raspberry Pi compatibility or performance.

### Hugging Face Text Embeddings Inference

Use Text Embeddings Inference for supported text-embedding and reranking models.

```bash
qit run core/embedding-retrieval \
  --provider tei \
  --model organization/embedding-model \
  --base-url http://127.0.0.1:8080
```

## What q.it measures

| Measurement | Plain-language meaning |
|-------------|------------------------|
| Request latency | How long the complete request took. |
| TTFT | How long the user waited before the first meaningful generated text appeared. |
| TPS | How many output tokens the model generated per second after output began. |
| CPU use | How busy the whole Pi was during the measured request. |
| RAM use | The highest total used memory observed during the measured request. |
| Process memory | The highest memory observed for a supplied model-server process. |
| Quality score | Whether the output matched the simple expectation for that case. |

Ollama supplies an exact output-token count and generation time.
Transformers Serve TPS is recorded only when the server supplies an exact completion-token count.
q.it never treats one network message as one model token.

CPU and RAM are measured for the whole Pi because an already-running server may also be doing unrelated work.
If you know the model server's process number, add it to collect that process's memory separately:

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model gemma3:1b \
  --pid 1234
```

## Successful and failed runs

Every attempted run is kept in the local database.
This includes connection failures, timeouts, malformed responses, interrupted runs, and successful runs.

A wrong model answer lowers the quality score but does not mean the benchmark tool failed.
A run is marked failed when q.it cannot complete or validate every required request.

Only successful runs appear in analysis exports:

```bash
qit export --format jsonl --output results.jsonl
qit export --format csv --output results.csv
```

JSONL keeps the detailed result for every measured case.
CSV gives one summary row per successful run.

## Reproducing a result

A comparison is useful only when the important conditions stay the same.
Use the same:

- Raspberry Pi model and RAM size.
- Cooling and power supply.
- Operating system and model-server version.
- Exact model and quantization.
- Benchmark-pack version.
- Warmup count, measured-pass count, and output limit.
- Background workload and server settings.

Use an explicit command instead of relying on defaults when sharing a result:

```bash
qit run core/text-generation-smoke \
  --provider ollama \
  --model gemma3:1b \
  --warmups 1 \
  --iterations 5 \
  --max-tokens 64
```

Export JSONL when another person needs enough detail to inspect or reuse the result.
See [Reproducing benchmark runs](docs/reproducible-runs.md) for a practical checklist.

## Benchmark add-ons

A benchmark pack is a local directory containing its description, cases, expected answers, and optional media.
Packs contain data only and cannot execute programs on the Pi.

Check a downloaded or locally authored pack before installing it:

```bash
qit benchmark verify /path/to/pack
qit benchmark install /path/to/pack
```

List the newly available benchmarks:

```bash
qit benchmark list
```

Remove a pack without deleting its historical results:

```bash
qit benchmark remove pack-name
```

There is currently no online q.it pack registry.
A pack must already exist as a local directory before it can be installed.

See [Benchmark pack guide](docs/benchmark-packs.md) for the supported layout and a small example.

## Multimodal model scope

The current server integrations cover text generation, text embeddings, image-to-text, speech-to-text, and text reranking.
Some future model families need a dedicated integration before their extra modalities can be measured honestly.

EmbeddingGemma 2 text embeddings can run through Hugging Face Serve.
Its image, audio, and video embedding modes still need a suitable server integration and benchmark packs for each modality, output size, and retrieval task.

The current release does not claim multimodal EmbeddingGemma 2 support.
See [the benchmark landscape](docs/research/pi5-edge-benchmark-landscape.md) for the researched expansion plan.

## Development

Run the Rust test suite:

```bash
cargo test
```

Build the dashboard:

```bash
cd qit-web
npm install
npm run build
```

Verify that the packaged command can be installed and started from a clean temporary directory:

```bash
scripts/test-packaged-product.sh
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making a change.
