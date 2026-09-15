# q.it

q.it is an offline package catalog and capacity planner for local open-weight inference on Apple Silicon.
It scans local GGUF artifacts and supported local Transformers packages, calculates fit from a stable memory budget, and supervises workers behind one local HTTP and SSE control plane.

## Supported packages

GGUF artifacts are discovered below `QIT_MODELS_DIR`.
q.it offers its bundled low-memory Qwen GGUF package when the required local file is present.

The bundled Qwen Transformers package resolves below the sibling `transformers` directory.
For example, `QIT_MODELS_DIR=$HOME/models/gguf` makes q.it look for it at `$HOME/models/transformers/Qwen/Qwen2.5-0.5B-Instruct`.
It requires `config.json`, `tokenizer.json`, `tokenizer_config.json`, `README.md`, `model.safetensors.index.json`, and the two safetensor shards named by that index.
Other local checkpoint directories are not cataloged or offered for serving until q.it owns a tested package recipe for them.
The catalog reports each package's file roles, sizes, SHA-256 hashes, source provenance, capabilities, fit, and one readiness reason.

The verified host platform is macOS on Apple Silicon.
Other platforms report unknown hardware capacity and are not a supported local-serving target.

## Run

```bash
QIT_MODELS_DIR="$HOME/models/gguf" cargo run -p qit-runtime
```

Open http://127.0.0.1:2471.

GGUF Start and Try use `llama-server`.
Install it with `brew install llama.cpp`, or set `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` to an executable.

Transformers text-chat packages need an executable OpenAI-compatible worker set through `QIT_TRANSFORMERS_WORKER_PATH`.
q.it starts the worker on a private loopback port with `--host`, `--port`, `--model`, and `--context-length`.
It waits for `/health` before marking the session Loaded, proxies `/v1/chat/completions` as q.it SSE, records failure details and logs, and stops the worker on session cleanup.

q.it never downloads model files, Python, PyTorch, or catalog data at runtime.

## Use

Choose Scan library after adding or changing local files.
The Catalog groups alternatives by family and recommends the smallest ready package that fits the current stable budget.
Packages with missing files, an unavailable runtime, or insufficient memory are not ready.
Packages that fail the stable-budget check cannot start a worker.

The control plane is available under `/api`.
Use `/api/catalog` to inspect artifacts and packages, `/api/capacity` to inspect reservations and sessions, `/api/sessions` to start or stop a package, and `POST /api/generate` for SSE text generation against a Loaded session.

## Configuration

- `QIT_HOME` - app state root, defaulting to `~/Library/Application Support/q.it`
- `QIT_MODELS_DIR` - GGUF library root, defaulting to `$QIT_HOME/models/gguf`
- `QIT_OS_RESERVE_BYTES` - stable-budget operating-system reserve override
- `QIT_PORT` - HTTP listen port, defaulting to `2471`
- `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` - `llama-server` executable
- `QIT_TRANSFORMERS_WORKER_PATH` - external OpenAI-compatible Transformers worker executable
- `QIT_WEB_DIST` - optional built web asset directory

## Web development

```bash
cd qit-web
npm install
npm run dev
```

Vite proxies `/api` to the local runtime.
