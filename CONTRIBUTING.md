# Contributing to q.it

q.it is a local catalog, capacity planner, and serving control plane for open-weight artifacts and model packages.
Good changes preserve that boundary and make the user-visible workflow more reliable.

## Before You Change Anything

Read [AGENTS.md](AGENTS.md) for repository-wide instructions.
Read [CONTEXT.md](CONTEXT.md) for the project vocabulary.
Read the relevant ADRs in [docs/adr](docs/adr/) before changing an established boundary.
Read [docs/agents/project.md](docs/agents/project.md) for the current product and test-seam facts.

For a bug, reproduce it through the HTTP and SSE control plane as closely as possible to the user workflow before changing code.
For a feature, find the relevant GitHub issue and verify that the requested behavior is not already implemented.

## Development Workflow

Keep changes small and scoped to one behavior.
Use the terms from `CONTEXT.md` in code, tests, issues, and documentation.
Do not add source-code comments.
Use names, types, and structure to explain intent.

The integration seam is `qit-runtime`'s HTTP and SSE control plane.
Use the control-plane harness, fixed hardware probes, and stub workers instead of real model processes in automated tests.
Do not add a second production test seam unless the control plane cannot express the behavior.
Distribution smoke tests that stage the `qit` executable live in `qit/tests/` and still assert product behavior through the HTTP and SSE control plane.

Run focused tests while working and the full suite before handing work off:

```bash
cargo test
```

For UI work, run the runtime and Vite development server together:

```bash
QIT_HOME="$PWD/.qit-data" QIT_MODELS_DIR="$HOME/models/gguf" cargo run -p qit-runtime
cd qit-web && npm install && npm run dev
```

## Issues and Reviews

Issues live on GitHub at `QuaanNguyen/q.it`.
Use the repository issue-tracker guidance in [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).

Before committing, check the diff, run the relevant tests, and review the change against the issue acceptance criteria.
Do not manually edit generated files or `CHANGELOG.md`.
