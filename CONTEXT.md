# q.it

Raspberry Pi-first benchmarking for local and network-served edge models.
Users run repeatable benchmark packs against provider endpoints, retain every outcome, and compare successful runs through a local analysis dashboard.

## Language

**Benchmark**:
A named, repeatable evaluation over one task and a fixed set of cases.
_Avoid_: test, workload, plugin

**Benchmark case**:
One input and its optional quality expectation within a benchmark.
_Avoid_: prompt, sample, fixture

**Benchmark pack**:
A versioned, installable add-on containing benchmark definitions and their case data.
Packs are declarative and never contain executable code.
_Avoid_: plugin, suite, extension

**Task**:
The model capability exercised by a benchmark, such as text to text, image to text, speech to text, embedding, reranking, or text to image.
_Avoid_: modality, capability

**Provider**:
The serving protocol used to invoke a model, such as Ollama, an OpenAI-compatible Transformers server, or Hugging Face Text Embeddings Inference.
_Avoid_: runtime, backend, host

**Provider adapter**:
The protocol-specific translation between benchmark cases and a provider, including streaming and usage metadata.
_Avoid_: benchmark plugin, runtime recipe

**Benchmark run**:
One durable attempt to execute a benchmark against a specific model, provider, host, and run configuration.
A run is either succeeded, failed, or running.
_Avoid_: session, job, measurement

**Sample result**:
The observed outcome for one benchmark case in one measured iteration, including latency, quality, and sampled resource use.
_Avoid_: benchmark run, data point

**Successful run dataset**:
The queryable and exportable set of succeeded benchmark runs and their summaries.
Failed runs remain in run history but never enter this dataset.
_Avoid_: results database, leaderboard

**Host telemetry**:
CPU and memory observations sampled from the machine executing the provider during a benchmark case.
When a provider process identifier is supplied, its resident memory is sampled separately.
_Avoid_: model usage, profiler

**Time to first token**:
Elapsed time from starting a streamed generation request until the first non-empty output token event arrives.
_Avoid_: startup time, first response time

**Token throughput**:
Completion tokens divided by provider-reported generation time when available, otherwise by elapsed time after the first token when an exact completion-token count is available.
_Avoid_: requests per second, characters per second

**Analysis dashboard**:
The optional local web interface for filtering run history and comparing successful runs with charts and a Pareto frontier.
It does not serve models or author benchmark packs.
_Avoid_: control plane, catalog, admin UI
