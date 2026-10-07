---
status: accepted
---

# Declarative benchmark packs and built-in provider adapters

q.it is now a Raspberry Pi-first benchmark runner instead of a model catalog, capacity planner, or serving control plane.
Benchmark packs are inert JSON and JSONL add-ons, while built-in provider adapters own Ollama, OpenAI-compatible, and Hugging Face Text Embeddings Inference protocol behavior.
This split lets users install new evaluation data without executing pack code, keeps timing and telemetry comparable across packs, and leaves protocol-specific streaming details in one deep module.

The command-line interface is the execution interface and the optional loopback dashboard is a read-only analysis interface over the same SQLite result store.
All run attempts remain in history, but only succeeded runs enter the exportable analysis dataset.

This decision supersedes ADR-0001 through ADR-0004 because their catalog, capacity, session, worker, and runtime-pack concepts belong to the previous product.
