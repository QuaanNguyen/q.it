# q.it agent project notes

Supplements `CONTEXT.md` and ADRs with repo-specific facts agents need repeatedly.

## Product direction

Local inference runtime for Apple Silicon (v1). Users browse **artifacts**, see **fit** against a **stable budget**, **pin** or **what-if** capacity, **start/stop** worker children, and **Try** an artifact in a multi-turn window whose transcript lives only in the browser. Not a chat product; agents and Hub install are later milestones on the same control plane.

Parent spec: GitHub issue [#1](https://github.com/QuaanNguyen/q.it/issues/1). Tracer bullets [#2–#7](https://github.com/QuaanNguyen/q.it/issues/2). Local copies: `.scratch/milestone-1-catalog-capacity/`.

## Repo layout

| Package | Role |
|---------|------|
| `qit-runtime` | Rust daemon: probe, scan, planner, supervisor, HTTP+SSE API, SQLite, embedded or proxied UI |
| `qit-web` | React + Vite + TypeScript SPA; dev proxies `/api` to runtime |

Shipped UX: one process (`qit-runtime`) on `127.0.0.1:2471`. Dev: `cargo run -p qit-runtime` plus `cd qit-web && npm run dev`.

## Test seam (locked)

**One seam:** `qit-runtime` HTTP+SSE control plane.

Integration tests live in `qit-runtime/tests/control_plane.rs`. Use temp `QIT_HOME`, injectable hardware snapshot (`FixedProbe`), stub worker (`qit-stub-worker` via `StubBinLauncher`). No real Metal, no real llama.cpp, no Hub in CI.

Do not add a second production test seam unless the control plane cannot express the behavior.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `QIT_HOME` | App state root (default `~/Library/Application Support/q.it/`) |
| `QIT_MODELS_DIR` | GGUF library scan root (default `$QIT_HOME/models/gguf`; maintainer uses `~/models/gguf`) |
| `QIT_OS_RESERVE_BYTES` | Planner OS reserve override |
| `QIT_PORT` | Listen port (default `2471`) |
| `QIT_WORKER_PATH` / `LLAMA_SERVER_PATH` | Real `llama-server` for local inference |
| `QIT_WEB_DIST` | Optional path to built `qit-web/dist` for production UI |

## Milestone 1 status

Tracer bullets #2–#6 verified on the maintainer's Mac with Nemotron ([#23](https://github.com/QuaanNguyen/q.it/issues/23)). Closure work ([#8](https://github.com/QuaanNguyen/q.it/issues/8)): **done** [#15](https://github.com/QuaanNguyen/q.it/issues/15) worker discovery/readiness, [#16](https://github.com/QuaanNguyen/q.it/issues/16) session rows, [#17](https://github.com/QuaanNguyen/q.it/issues/17) single-flight generate, [#18](https://github.com/QuaanNguyen/q.it/issues/18) context presets, [#19](https://github.com/QuaanNguyen/q.it/issues/19) settings table, the cancel test on [#7](https://github.com/QuaanNguyen/q.it/issues/7). **Open:** peak RSS on #7 (blocked on [#11](https://github.com/QuaanNguyen/q.it/issues/11)), [#20–#22](https://github.com/QuaanNguyen/q.it/issues/20) blocked on research [#10–#14](https://github.com/QuaanNguyen/q.it/issues/10). Homebrew `llama-server` at `/opt/homebrew/bin/llama-server` is auto-discovered; `/health` 503 while loading, 200 when ready.

## Generate API

`POST /api/generate` takes `messages: [{role, content}]` (or a one-message `prompt`), optional `max_tokens` (default 512), and streams SSE `token` events. `done` carries `{prompt_tokens, completion_tokens, n_ctx}` from the worker's usage chunk; the UI's context square reads it. One generate at a time: a concurrent request gets `409 generate in flight`.

## UI notes

Catalog rows carry Start/Stop with an inline status icon (spinner, green check, red cross; hover for the error and an **Inspect** link to Capacity) and a **Try** button that opens the Try window under the row, auto-starting the session when needed. Transcript is browser memory only. Palette and type live in `qit-web/src/styles.css`; throwaway layout variants live on branch `prototype/catalog-ui`.

## Coding conventions

- No comments in source code unless the user explicitly asks. Names and types carry intent.
- Deep modules at the HTTP control-plane seam; inject probes and worker launchers for tests.
