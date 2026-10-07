# q.it

q.it benchmarks edge models on Raspberry Pi 5 through Ollama, Hugging Face Transformers Serve, and Hugging Face Text Embeddings Inference.

The six bundled benchmarks are small, self-made checks for confirming that the model-serving path works.
Larger official evaluations are installed separately as benchmark packs.

Install the command with:

```bash
cargo install qit
```

Discover a provider and its compatible built-in benchmarks:

```bash
qit provider list
qit benchmark list --provider ollama
```

Run a benchmark, inspect the durable result, and start the optional dashboard:

```bash
qit run core/text-generation-smoke --provider ollama --model gemma3:1b
qit results
qit dashboard
```

See the [project repository](https://github.com/QuaanNguyen/q.it) for provider setup, benchmark-pack authoring, metric definitions, exports, and development instructions.
