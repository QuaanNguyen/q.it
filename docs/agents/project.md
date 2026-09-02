# q.it agent project notes

Supplements `CONTEXT.md` and ADRs with repo-specific facts agents need repeatedly.

## Product direction

Local inference runtime for Apple Silicon (v1). Users browse **artifacts**, see **fit** against a **stable budget**, **pin** or **what-if** capacity, **start/stop** worker children, and **smoke-test** generation. Not a chat product; agents and Hub install are later milestones on the same control plane.

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

Core tracer bullets are implemented on `main`. Known spec gaps (see code-review notes): peak RSS, live memory pressure, instruct-only Generate filter, model-max context presets, session/settings SQLite tables, single-flight generate.

## Coding conventions

- No comments in source code unless the user explicitly asks. Names and types carry intent.
- Deep modules at the HTTP control-plane seam; inject probes and worker launchers for tests.
