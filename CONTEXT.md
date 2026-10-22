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
The model capability exercised by a benchmark, such as text to text, image to text, speech to text, text to speech, embedding, reranking, or text to image.
_Avoid_: modality, capability

**Model catalog**:
The generated list of known models, their provider routes, tasks, adapters, and pinned identifiers.
It describes what q.it can invoke, never what is installed or running on this machine.
_Avoid_: registry, model zoo, download list

**Route**:
One way to serve a model: a provider, an optional adapter, a served model name, and the tasks it can run.
A model may have several routes with different providers or adapters.
_Avoid_: deployment, endpoint, variant

**Selection**:
The benchmark, provider, and model choices made together in the terminal workspace.
Each choice narrows the others when routes cannot serve the selected task.
_Avoid_: form, filter state

**Released choice**:
A previously selected benchmark, provider, or model that was cleared because another choice made it incompatible.
The workspace reports the cleared value, the reason, and moves focus to that field.
_Avoid_: reset, invalidation error

**Availability probe**:
A short check that the selected endpoint is reachable and lists the selected model.
An unprobed route stays allowed; a failed probe blocks the start gate with the observed reason.
_Avoid_: health check, ping

**Decision scoring**:
Assessing a fixed set of candidate answers to a named question about a supplied state, with a probability for each candidate and a selected answer.
_Avoid_: generated answer, token generation, free-form prediction

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

**Time to first audio**:
Elapsed time from starting a speech synthesis request until the first non-empty audio chunk arrives.
_Avoid_: startup time, first response time

**Real-time factor**:
Wall-clock synthesis time divided by the synthesized audio duration.
A value below one means synthesis ran faster than playback.
_Avoid_: speed, latency ratio

**Analysis dashboard**:
The optional local web interface for starting benchmark runs, inspecting run history, and comparing successful runs with charts and a Pareto frontier.
It does not serve models or author benchmark packs.
_Avoid_: control plane, catalog, admin UI

**Model comparison**:
A ranking of model configurations on one metric within the same benchmark version, machine snapshot, and measured run setup.
Each configuration retains its model, provider, and endpoint identity.
_Avoid_: overall intelligence score, Arena rating
