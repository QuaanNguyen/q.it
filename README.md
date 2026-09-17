# q.it

A local catalog and capacity planner for the open-weight models already on your machine.

## What It Does

q.it scans local GGUF artifacts and supported Transformers packages.
It shows what is present, what fits the configured capacity budget, and what is ready to serve.
It can start a local worker, proxy generation through one local HTTP and SSE control plane, and keep pins, sessions, settings, and measurements in local app state.

q.it never downloads model files, Python, PyTorch, or catalog data at runtime.

## Run It

From the repository root:

```bash
QIT_HOME="$PWD/.qit-data" QIT_MODELS_DIR="$HOME/models/gguf" cargo run -p qit-runtime
```

`QIT_HOME` keeps q.it state in a local writable directory.
This avoids macOS Application Support permission issues in terminals with restricted filesystem access.

Open [http://127.0.0.1:2471](http://127.0.0.1:2471).

## Local Models

q.it discovers GGUF artifacts below `QIT_MODELS_DIR`.
With `QIT_MODELS_DIR=$HOME/models/gguf`, it discovers supported Transformers packages below `$HOME/models/transformers`.
The current package recipes support local Qwen 3.5 and Gemma 4 layouts with their config, tokenizer metadata, model card, and safetensors weights.
Other checkpoint layouts are not offered for serving until q.it has a tested package recipe and worker contract for them.

GGUF Start and Try use `llama-server`.
Install it with `brew install llama.cpp`, or set `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` to its executable.

Transformers text-chat packages need an OpenAI-compatible worker configured through `QIT_TRANSFORMERS_WORKER_PATH`.
q.it starts the worker privately on loopback, waits for `/health`, and proxies `/v1/chat/completions` as q.it SSE.

Capacity estimates are currently calibrated for Apple Silicon unified memory.
On other hardware, q.it may report an unknown capacity budget while still cataloging local files.

## Use It

Choose Scan library after adding or changing local files.
The Catalog groups alternatives by family and recommends the smallest ready package that fits the stable budget.
Packages with missing files, an unavailable runtime, or insufficient memory are not ready.
Packages that fail the stable-budget check cannot start a worker.

The control plane is available under `/api`.
Use `/api/catalog` for artifacts and packages, `/api/capacity` for reservations and sessions, `/api/sessions` to start or stop a package, and `POST /api/generate` for SSE text generation against a Loaded session.

## Configuration

- `QIT_HOME` - app state root, defaulting to `~/Library/Application Support/q.it`.
- `QIT_MODELS_DIR` - GGUF library root, defaulting to `$QIT_HOME/models/gguf`.
- `QIT_OS_RESERVE_BYTES` - stable-budget operating-system reserve override.
- `QIT_PORT` - HTTP listen port, defaulting to `2471`.
- `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` - `llama-server` executable.
- `QIT_TRANSFORMERS_WORKER_PATH` - external OpenAI-compatible Transformers worker executable.
- `QIT_WEB_DIST` - optional built web asset directory.

## Development

Run the full runtime suite:

```bash
cargo test
```

Work on the web UI:

```bash
cd qit-web
npm install
npm run dev
```

Vite proxies `/api` to the local runtime.

## Contributing and Agent Navigation

Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.
Agents must also read [AGENTS.md](AGENTS.md), [CONTEXT.md](CONTEXT.md), and the relevant decisions in [docs/adr](docs/adr/).
The project-specific agent notes live in [docs/agents/project.md](docs/agents/project.md).
