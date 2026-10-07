# q.it agent project notes

Supplements `CONTEXT.md`, ADRs, and the research note with repository facts agents need repeatedly.

## Product direction

q.it is a Raspberry Pi 5-first benchmark runner for edge models served through explicit provider endpoints.
It does not catalog local model files, plan capacity, download models, or manage provider processes in the current release.

The execution workflow is command-line first because users operate the Pi over SSH.
The optional dashboard is read only and analyzes persisted results.

## Repository layout

| Package | Role |
|---------|------|
| `qit` | Publishable shell that installs the `qit` command. |
| `qit-runtime` | Benchmark engine, provider adapters, telemetry, SQLite store, pack loader, exports, and embedded dashboard server. |
| `qit-web` | React and TypeScript analysis dashboard compiled into `qit-runtime/web-dist`. |

## Active seams

**Execution seam**: the `qit` command.

**Analysis seam**: the read-only HTTP interface under `/api` and its embedded browser application.

**Provider seam**: built-in Ollama native, Transformers Serve, and Text Embeddings Inference adapters.

**Extension seam**: declarative benchmark packs with a versioned JSON manifest and JSONL cases.

Tests should exercise these interfaces with temporary state and fake loopback providers.
No real model weights, model downloads, containers, or external network calls belong in the normal suite.

## State

`QIT_HOME` overrides the state directory.
Linux otherwise uses `$XDG_DATA_HOME/qit` or `$HOME/.local/share/qit`.

The directory contains:

- `results.db` for all benchmark runs and sample results.
- `benchmark-packs/` for installed declarative add-ons.

Interrupted running rows become failed the next time the store opens.
Pack removal never deletes historical results.

## Run behavior

A run binds its benchmark and pack version, task, provider, endpoint, model string, iteration settings, host snapshot, and optional provider process identifier.
Every required case must complete and persist for the run to succeed.
Wrong answers are valid scored outputs.
Transport, protocol, timeout, output-validation, scorer, and persistence errors fail the run.

The successful run dataset is a query over succeeded rows rather than a copied database.
JSONL exports retain samples, while CSV exports contain flat run summaries.

## Provider facts

Ollama uses `/api/show` for capability preflight, `/api/generate` for NDJSON text or vision generation, and `/api/embed` for text embeddings.
Its exact output count and evaluation duration drive token throughput.

Transformers Serve uses its OpenAI-compatible model, chat-completion, and audio-transcription routes.
The `hf` command is acquisition tooling and is not a provider.

Text Embeddings Inference uses its native health, embedding, and reranking routes.
It is not treated as evidence that an arbitrary Transformers embedding model works on ARM64.

## Dashboard notes

The dashboard is responsive, dependency-light, and uses inline SVG and CSS for charts.
Pareto status compares only succeeded runs with the same benchmark, pack version, and host.
Generated assets are written to `qit-runtime/web-dist` by `npm run build`.

## Coding conventions

- Do not add source-code comments.
- Keep protocol-specific behavior inside provider adapters.
- Keep packs inert and validate all paths before activation.
- Persist raw sample values so summary formulas can evolve without rerunning models.
- Keep unavailable metrics absent instead of inventing approximations.
