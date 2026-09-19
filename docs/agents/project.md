# q.it agent project notes

Supplements `CONTEXT.md` and ADRs with repo-specific facts agents need repeatedly.

## Product direction

Local catalog, capacity planner, and serving control plane for open-weight artifacts and model packages.
Users browse **artifacts** and **model packages**, see **fit** against a **stable budget**, **pin** or **what-if** capacity, start and stop worker children, and Try an artifact in a multi-turn window whose transcript lives only in the browser.
Model packages are grouped by family, but fit, readiness, files, capabilities, and actions stay package-specific.
Not a chat product; agents and Hub install are later milestones on the same control plane.
Capacity estimates are calibrated for Apple Silicon unified memory, but the catalog and local control plane do not require an Apple Silicon-only product claim.

Parent spec: GitHub issue [#1](https://github.com/QuaanNguyen/q.it/issues/1). Tracer bullets [#2–#7](https://github.com/QuaanNguyen/q.it/issues/2). Local copies: `.scratch/milestone-1-catalog-capacity/`.

## Repo layout

| Package | Role |
|---------|------|
| `qit` | Publishable user shell: installs the `qit` command and starts the runtime library |
| `qit-runtime` | Rust daemon: probe, scan, planner, supervisor, HTTP+SSE API, SQLite, embedded or proxied UI |
| `qit-web` | React + Vite + TypeScript SPA; dev proxies `/api` to runtime |

Shipped UX: one `qit` process on `127.0.0.1:2471`.
The publishable shell uses `qit-runtime` as a library and serves the embedded production browser assets.
Development remains `cargo run -p qit-runtime` plus `cd qit-web && npm run dev`.

## Test seam (locked)

**One seam:** `qit-runtime` HTTP+SSE control plane.

Control-plane integration tests live in `qit-runtime/tests/control_plane.rs`.
Use temp `QIT_HOME`, injectable hardware snapshot (`FixedProbe`), and the stub worker (`qit-stub-worker` via `StubBinLauncher`).
Packaged-product smoke tests may live in `qit/tests/` when they must stage the user-facing executable, but they still assert behavior through HTTP and SSE.
No real Metal, no real llama.cpp, no Hub in CI.

Do not add a second production test seam unless the control plane cannot express the behavior.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `QIT_HOME` | App state root (default `~/Library/Application Support/q.it/`) |
| `QIT_MODELS_DIR` | GGUF library scan root (default `$QIT_HOME/models/gguf`; maintainer uses `~/models/gguf`) |
| `QIT_OS_RESERVE_BYTES` | Planner OS reserve override |
| `QIT_PORT` | Listen port (default `2471`) |
| `QIT_WORKER_PATH` / `LLAMA_SERVER_PATH` | Executable `llama-server` for local inference |
| `QIT_TRANSFORMERS_WORKER_PATH` | External OpenAI-compatible Transformers worker |
| `QIT_WEB_DIST` | Optional path to built `qit-web/dist` for production UI |

## Milestone 1 status

Tracer bullets #2–#6 verified on the maintainer's Mac with Nemotron ([#23](https://github.com/QuaanNguyen/q.it/issues/23)). Closure work ([#8](https://github.com/QuaanNguyen/q.it/issues/8)): **done** [#15](https://github.com/QuaanNguyen/q.it/issues/15)–[#22](https://github.com/QuaanNguyen/q.it/issues/22) and [#7](https://github.com/QuaanNguyen/q.it/issues/7) (including peak RSS via `wait4` `ru_maxrss`). Research notes live in `docs/research/`. Homebrew `llama-server` at `/opt/homebrew/bin/llama-server` is auto-discovered; `/health` 503 while loading, 200 when ready.

## Generate API

`POST /api/generate` takes `messages: [{role, content}]` (or a one-message `prompt`), optional `max_tokens` (default 512), and streams SSE `token` events. `done` carries `{prompt_tokens, completion_tokens, n_ctx}` from the worker's usage chunk; the UI's context square reads it. One generate at a time: a concurrent request gets `409 generate in flight`.

## Offline package catalog

GGUF artifacts are scanned below `QIT_MODELS_DIR`.
Supported Transformers package recipes are discovered below the sibling `transformers` library, currently covering local Qwen 3.5 and Gemma 4 layouts.
The scan records local file size and modification time, then exposes package capabilities, catalog planner-hint provenance, readiness, and one clear readiness reason through `/api/catalog`.
Other local checkpoint directories are not catalogued until q.it owns a tested package recipe and external-worker contract for them.

## UI notes

Catalog is the Cards layout from `prototype/catalog-ui` (variant B), with no variant switcher. Rows carry Start/Stop with an inline status icon (spinner, green check, red cross; hover for the error and an **Inspect** link to Capacity) and a **Try** button that opens the Try window under the card, auto-starting the session when needed. Transcript is browser memory only. Palette and type live in `qit-web/src/styles.css`.

## Coding conventions

- No comments in source code unless the user explicitly asks. Names and types carry intent.
- Deep modules at the HTTP control-plane seam; inject probes and worker launchers for tests.
