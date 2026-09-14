# q.it

Local catalog and capacity planner for GGUF artifacts on Apple Silicon.

## Run

```bash
QIT_MODELS_DIR="$HOME/models/gguf" cargo run -p qit-runtime
```

Open http://127.0.0.1:2471

Start and Try need `llama-server` (`brew install llama.cpp` is auto-discovered on Apple Silicon). Override with `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` if needed.
Transformers packages need an OpenAI-compatible worker configured with `QIT_TRANSFORMERS_WORKER_PATH`.
q.it launches it with `--host`, `--port`, `--model`, and `--context-length`, then waits for `/health` before proxying `/v1/chat/completions`.

Optional:

- `QIT_HOME` — app state (default `~/Library/Application Support/q.it`)
- `QIT_OS_RESERVE_BYTES` — planner OS reserve; overrides the value saved on the Settings page
- `QIT_PORT` — listen port (default 2471)
- `QIT_WORKER_PATH` or `LLAMA_SERVER_PATH` — llama-server binary for real inference
- `QIT_TRANSFORMERS_WORKER_PATH` - external Transformers worker binary

Web UI in development:

```bash
cd qit-web && npm install && npm run dev
```

Vite proxies `/api` to the runtime.
