# Raspberry Pi 5 edge-model benchmark landscape

Research date: 2026-10-06.

This note evaluates how q.it can become a Raspberry Pi 5-first benchmark runner for GGUF and Hugging Face models served locally or over a network.
It uses the domain language in [CONTEXT.md](../../CONTEXT.md) and follows the declarative benchmark-pack decision in [ADR-0005](../adr/0005-benchmark-packs-and-provider-adapters.md).
All third-party behavior described below is grounded in a linked first-party specification, repository, paper, model card, or dataset page.

## Executive recommendation

The practical product shape is a small Rust command installed with Cargo, built-in provider adapters, inert benchmark packs installed separately, one durable SQLite store, and an optional loopback analysis dashboard.

The initial provider adapters should be:

1. Ollama's native API for [GGUF import](https://docs.ollama.com/import), text to text, [image to text](https://docs.ollama.com/capabilities/vision), [tool calling](https://docs.ollama.com/capabilities/tool-calling), and [documented text embeddings](https://docs.ollama.com/api/embed).
2. [OpenAI-compatible Transformers Serve](https://huggingface.co/docs/transformers/serve-cli/serving) for text, image, audio, video, tool calling, and speech transcription.
3. [Hugging Face Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/quick_tour) for text embeddings, reranking, and sequence classification using its [CPU ARM64 container](https://huggingface.co/docs/text-embeddings-inference/en/supported_models#supported-hardware).

A later Cactus provider adapter is justified for [Cactus models](https://github.com/cactus-compute/cactus), [Needle tool calling and extraction](https://github.com/cactus-compute/needle), Whistle speech recognition, and other edge-specific behavior that is not fully represented by the first three protocols.
A later q.it-owned local embedding provider is also necessary for models such as [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma) whose multimodal embedding interface is not exposed by Ollama's documented text embedding route, Transformers Serve, or today's TEI model set.

The benchmark runner should always collect timing, run outcome, host telemetry, model identity, provider identity, and environment metadata.
Quality scoring should be added through installable benchmark packs selected by task and provider capabilities.
Benchmark packs should contain data and definitions only, never native libraries, Python, shell commands, or model repository code.

Cold-load and warm-inference results must be separate populations.
CPU and RAM attribution is trustworthy only when q.it can identify or own the provider process tree, ideally through a fresh cgroup v2 for each benchmark run.
Host-wide telemetry is still useful for an external provider, but it must not be labeled as model process usage.

## Product boundary

The pivot should retain the useful shell, SQLite, and local browser foundations while removing catalog, capacity-planning, and model-serving assumptions from the earlier product.
ADR-0005 already makes the command-line interface the execution interface, the dashboard a read-only analysis surface, provider adapters built in, and benchmark packs declarative.

That split has several consequences:

- Installing q.it through Cargo installs the trusted runner, provider adapters, telemetry collector, scorers, result store, and dashboard assets.
- Installing a benchmark pack adds only a versioned manifest, case data, media assets, and expected answers.
- Adding a new scorer or provider protocol requires a q.it release because a declarative pack cannot safely add executable behavior.
- A pack can require a scorer family that q.it already implements, such as exact match, multiple choice, VQA consensus, word error rate, retrieval ranking, or caption metrics.
- Large media packs can be represented by a small signed manifest plus separately fetched, checksummed asset shards.
- Every installed pack remains usable offline after all declared assets are present.

The user-facing word for these add-ons should remain **benchmark pack**, not plugin.
This keeps the install experience extensible without creating a native plugin ABI or an arbitrary-code execution path.

## Raspberry Pi 5 as a benchmark host

Raspberry Pi 5 has a 2.4 GHz quad-core 64-bit Arm Cortex-A76 processor, 512 KB of L2 cache per core, 2 MB of shared L3 cache, LPDDR4X-4267 memory, and memory variants from 1 GB through 16 GB. [Raspberry Pi 5 specifications](https://www.raspberrypi.com/products/raspberry-pi-5/)
The official specification recommends a 5 V, 5 A USB-C power source, and Raspberry Pi warns that an underpowered supply can cause problems. [Raspberry Pi 5 specifications](https://www.raspberrypi.com/products/raspberry-pi-5/)
Raspberry Pi documents that sustained heavy load can cause thermal throttling, with throttling beginning at 80 degrees Celsius and increasing at 85 degrees Celsius. [Raspberry Pi thermal testing](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)
Raspberry Pi recommends active cooling for best performance, and its sustained-load testing found that passive cooling can still reach throttling conditions. [Raspberry Pi hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#add-heatsinks)

These facts make the environment part of every result rather than incidental metadata.
Each benchmark run should retain:

- Raspberry Pi model and RAM capacity.
- OS release, kernel, architecture, and firmware version.
- Online CPU count and CPU affinity.
- CPU frequency governor, minimum and maximum policy frequencies, and sampled current frequency.
- Cooling description and an optional ambient-temperature note.
- Power source description when known.
- Swap configuration and swap use.
- Model storage device and filesystem when load time is compared.
- Provider version, provider launch arguments when known, and thread count.
- q.it version, benchmark-pack version, and scorer version.

Linux CPUFreq exposes governor and frequency policy data below the CPU policy sysfs directories, and it distinguishes requested frequency from hardware-derived current frequency where the driver provides both. [Linux CPUFreq documentation](https://docs.kernel.org/admin-guide/pm/cpufreq.html)
Raspberry Pi documents `vcgencmd measure_temp` as its accurate instantaneous SoC temperature source and documents `measure_clock arm` for the Arm clock. [Raspberry Pi hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#measure-temperatures) [Raspberry Pi vcgencmd documentation](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd)
Raspberry Pi's `get_throttled` result identifies current and historical undervoltage, frequency capping, throttling, and soft temperature-limit conditions. [Raspberry Pi vcgencmd documentation](https://www.raspberrypi.com/documentation/computers/os.html#get_throttled)

A comparative benchmark should be marked environmentally invalid when it experiences current throttling, an undervoltage event, an out-of-memory event, unexpected swap activity, or severe unrelated host contention.
The partial data should still be retained for diagnosis, but the run should be failed and excluded from the successful run dataset.

## Common measurement contract

### Timing boundaries

All client timing should use a monotonic clock.
The adapter should record raw timestamps or raw durations before computing summaries.

| Measurement | Definition | Applicability |
|---|---|---|
| Request latency | Time immediately before dispatching prepared request bytes through receipt of the provider's documented terminal response. | Every provider request. |
| Case latency | Time from beginning case preparation through validated output, excluding offline quality aggregation. | Every benchmark case. |
| Model load latency | Time from an explicitly unloaded or stopped state until the provider reports the model ready. | Cold-load profiles. |
| Time to first byte | Time from request dispatch to the first response byte. | Diagnostic only. |
| Time to first token | Time from request dispatch to the first non-empty semantic output event. | Streamed text-producing tasks. |
| Time to first visible answer | Time from dispatch to the first visible answer content, excluding a separate reasoning stream. | Reasoning models when the provider separates reasoning. |
| Completion latency | Time from first non-empty semantic output to the terminal response. | Streamed text-producing tasks. |
| Time per output token | Completion latency divided by one less than the exact output-token count. | Text generation with at least two exact output tokens. |
| Provider decode rate | Provider-reported output-token count divided by provider-reported decode duration. | Providers exposing both values. |
| Client decode rate | One less than the exact output-token count divided by completion latency. | Streamed text generation with an exact count. |
| End-to-end token rate | Exact output-token count divided by request latency. | Text generation with an exact count. |

MLCommons defines TTFT as latency until the first generated token and TPOT as the average latency between consecutive output tokens. [MLCommons endpoint client terminology](https://github.com/mlcommons/endpoints/blob/main/docs/ENDPOINT_CLIENT.md#terminology--acronyms)
MLCommons' current endpoint rules trigger TTFT on the first non-empty text fragment in visible output, tool-call content, or reasoning content, which is a useful protocol-level convention for q.it. [MLCommons endpoint rules](https://github.com/mlcommons/endpoints-submission-docs/blob/main/_sources/policies-endpoints_rules.md#28-output-token-counting-and-ttft)
The dashboard can label decode rate as tokens per second, or TPS, while the stored metric retains its precise client-observed or provider-reported definition.

An SSE or NDJSON chunk is not guaranteed to contain exactly one model token.
Chunk arrival intervals therefore must not be presented as inter-token latency.
Time per output token and client decode rate should be absent unless the adapter receives an exact provider count or can apply the exact tokenizer associated with the model revision.
Every stored token count should include its source, such as provider, reference tokenizer, approximate tokenizer, or unavailable.

For a reasoning model, q.it should retain both the first semantic output time and the first visible answer time when the protocol exposes reasoning separately.
This prevents a model that streams reasoning early from appearing to have the same user-visible latency as a model that begins its answer early.

### Distributions and repetitions

A single run summary should report count, minimum, median, mean, p90, p95, maximum, and standard deviation for repeated latency measurements.
Raw sample results should remain available so future summary formulas do not require rerunning the model.

The default profiles should be:

- Smoke: a fixed, versioned case subset and one measured iteration for compatibility checks.
- Standard: a fixed, versioned representative subset, one warmup iteration, and at least three measured iterations where runtime permits.
- Full: the benchmark's canonical evaluation split and declared evaluation protocol.

Case selection must be deterministic and stored as explicit case identifiers.
A randomly limited run is not comparable unless the seed, selection algorithm, source order, and resulting case identifiers are all part of the run identity.

### Cold and warm runs

Cold and warm measurements answer different questions and should never be pooled.

- A cold-load profile begins from a documented unloaded provider state and includes model readiness or the first request's load component.
- A warm profile explicitly preloads the model, performs warmup work that is excluded from summaries, and verifies that the intended model remains resident.
- A cache profile records whether prompt or input caches are permitted and whether the selected cases intentionally reuse prefixes.
- A sustained profile runs long enough to reveal thermal throttling and reports its full CPU, memory, temperature, frequency, and pressure trace.

Ollama accepts a zero keep-alive duration to unload a model immediately and reports load duration in its final response. [Ollama generate API](https://docs.ollama.com/api/generate)
Transformers Serve exposes a model-loading SSE stream whose terminal state is either ready or error and whose stages distinguish processor, configuration, download, and weight loading. [Transformers Serve model loading](https://huggingface.co/docs/transformers/serve-cli/serving#loading-models)

### Metrics for non-generative tasks

| Task | Performance metrics | Quality metrics |
|---|---|---|
| Speech to text | Request latency, real-time factor, audio seconds per wall second, optional first token, and peak memory. | Word error rate and character error rate with a declared normalizer. |
| Image to text | Request latency, optional time to first token, decode rate, pixels or encoded bytes per case, and peak memory. | Accuracy, exact match, VQA consensus, OCR score, or caption metrics. |
| Audio or video to text | Request latency, real-time factor, media seconds per wall second, sampling policy, optional time to first token, and peak memory. | Accuracy, exact match, task-specific score, or caption metrics. |
| Embedding | Latency per item, items per second, input tokens or media seconds per second, batch size, output dimension, and peak memory. | Retrieval, similarity, classification, clustering, or bitext score. |
| Reranking | Latency per query, candidate pairs per second, candidate count, truncation policy, and peak memory. | nDCG, mean reciprocal rank, mean average precision, recall, or precision. |
| Tool calling | Request latency, optional time to first tool content, tokens, and peak memory. | Function-name accuracy, argument exact match, AST or JSON validity, and sequence accuracy. |

For speech, q.it should define real-time factor as wall seconds divided by input-audio seconds, where lower is better.
It should also store the reciprocal audio processing rate because users often find "audio seconds processed per second" easier to read.

## CPU, RAM, and contention measurement

### Preferred local-provider method

When q.it starts the provider, it should create a fresh cgroup v2 for the benchmark run and place the provider plus every descendant in it.
The cgroup should be unique per run so peak counters and failure events have an unambiguous lifetime.
An unprivileged process can manage only a suitably delegated cgroup subtree, so the runner must feature-detect delegation and use the procfs fallback when no writable subtree is available. [Linux cgroup v2 delegation](https://docs.kernel.org/admin-guide/cgroup-v2.html#delegation)

Linux cgroup v2 `cpu.stat` always provides usage, user, and system time in microseconds for the cgroup and its descendants. [Linux cgroup v2 CPU interface](https://docs.kernel.org/admin-guide/cgroup-v2.html#cpu-interface-files)
Linux cgroup v2 exposes current memory, peak memory, swap use, memory type breakdowns, and memory-event counters for a cgroup and its descendants. [Linux cgroup v2 memory interface](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)

The core derived CPU values should be:

- CPU core-seconds equal the cgroup CPU usage delta divided by one million.
- Average occupied cores equal the cgroup CPU usage delta divided by wall-clock microseconds.
- Host-normalized CPU percent equals average occupied cores divided by online logical CPUs and multiplied by 100.
- One-core percent equals average occupied cores multiplied by 100, so a fully occupied four-core Pi can approach 400 percent.

The store should retain average occupied cores because it does not depend on a monitoring tool's percentage convention.

The memory values should be:

- Current cgroup memory.
- Peak cgroup memory.
- Peak anonymous memory when available from sampled breakdowns.
- Peak file-cache memory when available from sampled breakdowns.
- Peak swap use.
- Provider-process RSS and high-water RSS when a stable process identifier is known.
- Host memory available at start and minimum host memory available during the run.

The current kernel documentation defines `memory.peak` on non-root cgroups and allows reset through an open file descriptor, so q.it should feature-detect it and prefer a fresh cgroup instead of relying on a global reset. [Linux cgroup v2 memory peak](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files)
If `memory.peak` is unavailable, q.it should sample `memory.current` and report the maximum observed value with the sampling period attached.

### Fallback and external-provider method

Linux `/proc/PID/status` exposes current RSS and RSS high-water mark, but the kernel documentation warns that these values can be imprecise. [Linux proc process status](https://docs.kernel.org/filesystems/proc.html#proc-pid-status)
Linux `/proc/PID/smaps_rollup` provides accumulated mapping data at higher collection cost. [Linux proc mapping statistics](https://docs.kernel.org/filesystems/proc.html#the-proc-pid-map-files)
Linux `/proc/meminfo` exposes host total memory, estimated available memory, cache, and swap values. [Linux proc memory information](https://docs.kernel.org/filesystems/proc.html#meminfo)

These sources support a fallback hierarchy:

1. Fresh cgroup v2 containing the provider process tree.
2. Known provider process plus descendants, sampled through procfs.
3. Host-wide CPU and memory telemetry only.
4. No server resource metrics for a remote endpoint without a cooperating remote collector.

An already running shared Ollama or Transformers daemon cannot be attributed precisely to one q.it run if it serves other traffic.
The result must label the resource scope as cgroup, process tree, process, host, remote unavailable, or provider reported.
Provider-reported RAM should be retained separately from kernel-observed RAM rather than substituted for it.

### Pressure and environmental trace

Linux pressure stall information reports partial and full CPU, memory, and I/O stalls through rolling averages and cumulative microseconds. [Linux PSI documentation](https://docs.kernel.org/accounting/psi.html)
The runner should capture system and cgroup pressure deltas when available.
It should sample CPU, memory, temperature, Arm clock, throttling bits, and pressure at a configurable interval, with 250 ms as a reasonable low-overhead default for long Pi runs.

The run record should retain both a compact summary and the downsampled time series.
This supports a dashboard trace that can explain a late-run throughput collapse instead of showing only an unexplained mean.

## Provider adapter landscape

### Adapter matrix

| Provider | Practical Pi tasks | Formats and deployment | Native timing | Important boundary | Recommendation |
|---|---|---|---|---|---|
| [Ollama native](https://docs.ollama.com/api/introduction) | Text to text, [image to text](https://docs.ollama.com/capabilities/vision), [tool calling](https://docs.ollama.com/capabilities/tool-calling), [structured text](https://docs.ollama.com/capabilities/structured-outputs), and [text embeddings](https://docs.ollama.com/api/embed). | [Official Linux ARM64 build](https://docs.ollama.com/linux), [GGUF import, and architecture-dependent Safetensors import](https://docs.ollama.com/import). | [Load, prompt evaluation, output evaluation, total duration, and token counts](https://docs.ollama.com/api/usage). | Published embedding input is text, and the [published OpenAPI](https://docs.ollama.com/openapi.yaml) has no stable audio input route. | First release. |
| [Transformers Serve](https://huggingface.co/docs/transformers/serve-cli/serving) | Text, image, audio, and video to text, tool calling, and speech transcription. | Hugging Face cache and Transformers-compatible models, with model-specific dependencies and memory behavior. | Client timing and documented final usage counts. | No documented embedding or reranking endpoint, and Pi support must be qualified per model and dependency set. | First release with an explicit compatibility matrix. |
| [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/quick_tour) | Text embedding, reranking, similarity, and sequence classification. | [Official CPU ARM64 container](https://huggingface.co/docs/text-embeddings-inference/en/supported_models#supported-hardware) for supported model families. | [Response timing headers](https://github.com/huggingface/text-embeddings-inference/blob/main/router/src/http/server.rs), [Prometheus metrics](https://huggingface.github.io/text-embeddings-inference/openapi.json), and client timing. | [Current model-family support](https://huggingface.co/docs/text-embeddings-inference/en/supported_models) is narrower than all Transformers models and does not establish multimodal EmbeddingGemma 2 support. | First release for embedding and reranking. |
| [Cactus and Needle](https://github.com/cactus-compute/cactus) | Text, vision, speech, tool calling, extraction, and embeddings depending on model and API. | Edge-native Cactus artifacts and [Linux ARM64 Needle builds](https://github.com/cactus-compute/needle). | Native Cactus and Needle results expose several latency and throughput values. | Cloud handoff and telemetry must be disabled for a strictly local benchmark, and provider-specific semantics need a dedicated adapter. | Second release. |
| [TGI llama.cpp backend](https://huggingface.co/docs/text-generation-inference/backends/llamacpp) | GGUF text generation on CPU is technically documented. | Native-built llama.cpp container can target ARM CPU features. | TGI metrics and client timing. | The [TGI repository](https://github.com/huggingface/text-generation-inference) was archived in March 2026 and now recommends other engines. | Do not add unless a user has an existing TGI deployment. |

### Hugging Face CLI is not the provider

The `hf` CLI is the Hugging Face Hub command-line client for authentication, downloading, uploading, caching, and repository operations. [Hugging Face Hub CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
The official lightweight local server is `transformers serve`, not `hf serve`. [Transformers Serve documentation](https://huggingface.co/docs/transformers/serve-cli/serving)

q.it should therefore model two independent operations:

- Model acquisition can use an already populated cache or instructions involving the `hf` CLI.
- Benchmark execution talks to a provider endpoint or a q.it-owned local provider process.

Model download should never happen as a hidden side effect of starting a measured run.
A preflight may report missing artifacts and offer an explicit acquisition step before measurement begins.

### Ollama native API

Ollama publishes a Linux ARM64 package, which establishes architecture compatibility but is not a Pi 5 performance certification. [Ollama Linux installation](https://docs.ollama.com/linux)
Ollama can import GGUF directly and can import Safetensors only for supported architectures, while imported GGUF must already be quantized. [Ollama model import](https://docs.ollama.com/import)

The adapter should use Ollama's native API instead of its OpenAI compatibility layer because the native terminal response carries richer model timing and cache data.

`/api/generate` and `/api/chat` stream by default and return total duration, load duration, prompt tokens, cached prompt tokens, uncached prompt-evaluation duration, output tokens, output-evaluation duration, terminal state, and terminal reason. [Ollama generate API](https://docs.ollama.com/api/generate) [Ollama chat API](https://docs.ollama.com/api/chat)
Ollama states that all of these timing values are nanoseconds and puts usage fields in the final streamed chunk. [Ollama usage fields](https://docs.ollama.com/api/usage)

The adapter can derive:

- Native decode tokens per second from output tokens and output-evaluation duration.
- Native uncached prompt tokens per second from uncached prompt tokens and prompt-evaluation duration.
- Native load latency from load duration.
- Client TTFT from dispatch to the first non-empty response or thinking content.
- Client wall latency independently from Ollama's total duration.

Ollama streams newline-delimited JSON rather than SSE. [Ollama streaming API](https://docs.ollama.com/api/streaming)
An Ollama error can arrive as an NDJSON object after a successful HTTP status has already been sent. [Ollama error handling](https://docs.ollama.com/api/errors)
An Ollama request is therefore successful only after the adapter sees no error object and sees the documented terminal marker.

`/api/tags` inventories installed models and includes digest, size, format, family, parameter-size label, and quantization. [Ollama list-models API](https://docs.ollama.com/api/tags)
`/api/show` is the stronger compatibility probe because it exposes capabilities, format, family, parameter size, quantization, context metadata, and model metadata. [Ollama show-model API](https://docs.ollama.com/api-reference/show-model-details)
`/api/ps` reports loaded models, expiry, context length, size, and VRAM size, but it is not a CPU or RSS accounting interface. [Ollama running-models API](https://docs.ollama.com/api/ps)

Ollama's documented vision path accepts base64 images for compatible models. [Ollama vision documentation](https://docs.ollama.com/capabilities/vision)
Ollama's documented embedding endpoint accepts one text or an array of texts and returns vectors, total duration, load duration, and input-token count. [Ollama embedding API](https://docs.ollama.com/api/embed)
The current published OpenAPI does not define an audio request field, so q.it must not infer speech or audio support from a model name. [Ollama OpenAPI](https://docs.ollama.com/openapi.yaml)

### Transformers Serve

Transformers Serve is documented as a lightweight local or self-hosted OpenAI-compatible server for evaluation, experimentation, and moderate load. [Transformers Serve documentation](https://huggingface.co/docs/transformers/serve-cli/serving)
It documents chat completions for text, image, audio, and video, legacy text completions, Responses API requests, audio transcription, model listing, and model-loading progress. [Transformers Serve endpoints](https://huggingface.co/docs/transformers/serve-cli/serving)
Its model-loading stream ends with exactly one ready or error event and distinguishes processor, configuration, download, and weights stages. [Transformers Serve model loading](https://huggingface.co/docs/transformers/serve-cli/serving#loading-models)

The adapter should support both chat completions and the dedicated audio-transcription route.
It should wait for the first non-empty content or reasoning delta instead of the first SSE frame because the server can emit structural frames before semantic content. [Transformers chat completion source](https://github.com/huggingface/transformers/blob/main/src/transformers/cli/serving/chat_completion.py)
Transformers Serve returns standard usage counts and finish reasons but does not document Ollama-like prompt and decode duration fields. [Transformers Serve documentation](https://huggingface.co/docs/transformers/serve-cli/serving)

Each supported model must pass an aarch64 CPU qualification test that covers import, processor construction, media decoding, one warm inference, one cold load, memory ceiling, and clean termination.
Video support requires an additional media dependency, so video-capable packs should expose a precise missing-dependency reason rather than fail during a run. [Transformers Serve video documentation](https://huggingface.co/docs/transformers/serve-cli/serving#video-based-completions)

GGUF handling is version-sensitive in Transformers, and the documented path can differ by architecture and release. [Transformers GGUF documentation](https://huggingface.co/docs/transformers/main/quantization/gguf)
The stored model identity must therefore include the Transformers version, model revision, selected GGUF file when applicable, load dtype, and observed in-memory footprint.

### Text Embeddings Inference

TEI currently documents embeddings for several text model families plus rerankers and sequence-classification models. [TEI supported models](https://huggingface.co/docs/text-embeddings-inference/en/supported_models)
TEI publishes a CPU ARM64 container and explicitly lists aarch64 CPU support. [TEI supported hardware](https://huggingface.co/docs/text-embeddings-inference/en/supported_models#supported-hardware)
Its documented quick tour covers native embeddings, OpenAI-compatible embeddings, reranking, classification, and batching. [TEI quick tour](https://huggingface.co/docs/text-embeddings-inference/quick_tour)

The adapter should support native embedding and reranking routes rather than force both through one OpenAI shape.
TEI's OpenAPI includes health, information, embedding, reranking, prediction, similarity, tokenization, decoding, OpenAI embeddings, and metrics routes. [TEI OpenAPI](https://huggingface.github.io/text-embeddings-inference/openapi.json)
TEI's router source exposes total, tokenization, queue, and inference timing through response headers, plus compute token or character counts. [TEI router source](https://github.com/huggingface/text-embeddings-inference/blob/main/router/src/http/server.rs)

The benchmark result should retain those server timings alongside client wall latency.
Batch size, input count, input-token count, output dimension, truncation policy, normalization, pooling, and prompt prefix must all be part of an embedding result's identity.

### Cactus, Needle, and Whistle

Cactus exposes edge-oriented text, speech, vision, transcription, embedding, tool-calling, and RAG behavior and includes ARM NEON kernels. [Cactus repository](https://github.com/cactus-compute/cactus)
Its native completion result includes success state, cloud-handoff state, time to first token, total time, prefill rate, decode rate, RAM usage, and token counts. [Cactus engine response](https://github.com/cactus-compute/cactus#cactus-engine)
Its local server is OpenAI-compatible, but q.it still needs provider-specific parsing to retain native metrics and reject cloud-handoff results. [Cactus command reference](https://github.com/cactus-compute/cactus#using-this-repo)

Needle 3 targets small edge devices for tool calls, structured extraction, and text embeddings, and its repository publishes a Linux ARM64 build path. [Needle repository](https://github.com/cactus-compute/needle)
Whistle shares the Needle runtime, accepts 16 kHz mono audio, reports time to first token and decode rate, and supports speech transcription and audio embeddings. [Needle Whistle documentation](https://github.com/cactus-compute/needle#whistle)
Needle states that telemetry is enabled by default in its binary. [Needle deployment documentation](https://github.com/cactus-compute/needle#deploy)

A valid edge benchmark must disable Cactus cloud handoff and Needle telemetry, then verify that the response indicates local execution.
The provider adapter should preserve native confidence and timing fields without using provider-reported RAM as a replacement for kernel telemetry.

### Why TGI is not an initial target

The former TGI llama.cpp backend documents GGUF, CPU inference, native builds, and an ARM CPU architecture option. [TGI llama.cpp backend](https://huggingface.co/docs/text-generation-inference/backends/llamacpp)
However, Hugging Face archived the TGI repository on 2026-03-21, placed it in maintenance mode, and recommends other inference engines going forward. [TGI repository](https://github.com/huggingface/text-generation-inference)

Supporting it would add operational weight and a fourth text-generation protocol while providing little Pi-specific value beyond Ollama or a direct llama.cpp-compatible provider.
q.it should accept a generic OpenAI-compatible TGI endpoint only if that generic behavior already works, without promising TGI-specific lifecycle support.

## Capability-driven benchmark selection

Provider and benchmark compatibility should be structural rather than based on model names.

Each provider probe should produce a normalized descriptor containing:

- Accepted input kinds: text, image, audio, video, or interleaved media.
- Produced output kinds: text, transcript, embedding, rerank score, classification score, image, audio, or video.
- Streaming support and streamed semantic categories.
- Tool-calling and structured-output support.
- Batch support and limits.
- Exact token-usage availability.
- Native load, prompt, queue, inference, and decode timing availability.
- Model context and media limits when discoverable.
- Embedding dimensions and dimension override support when discoverable.
- Local process ownership and telemetry scope.

Each benchmark definition should declare requirements using the same vocabulary.
The command that lists available benchmarks should show only compatible packs by default and explain every rejected requirement on request.
Unknown capability should never be treated as supported.

Protocol probing is still not enough for newer custom models.
A one-case preflight should validate the exact request shape, terminal response, output type, scorer compatibility, and declared resource ceiling before a full run begins.

## Benchmark-pack candidates

The catalog below separates practical first-wave packs from larger or more specialized packs.
Each pack should offer smoke, standard, and full profiles where the source dataset permits them.
Smoke and standard profiles must use stable case identifiers rather than a mutable first-N slice.

### Text to text

| Benchmark pack | What it measures | Scoring | Pi practicality | Primary source |
|---|---|---|---|---|
| Fixed-shape text performance | Prefill, TTFT, decode rate, context scaling, concurrency, and memory with synthetic text of fixed token lengths. | No semantic quality score. | Essential first-wave pack. | q.it-owned definition informed by [MLCommons endpoint metric terminology](https://github.com/mlcommons/endpoints/blob/main/docs/ENDPOINT_CLIENT.md#terminology--acronyms). |
| MMLU-Pro | Broad academic knowledge and reasoning across ten-choice questions. | Accuracy by subject and overall. | Strong standard pack with small deterministic subsets. | [Official MMLU-Pro repository](https://github.com/TIGER-AI-Lab/MMLU-Pro). |
| GSM8K | Multi-step grade-school mathematics. | Final numeric exact match after canonical extraction. | Strong standard pack. | [Official GSM8K repository](https://github.com/openai/grade-school-math). |
| IFEval | Verifiable instruction following. | Strict and loose rule compliance. | Strong standard pack because scoring is deterministic. | [Official IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval). |
| GPQA | Graduate-level, expert-written science questions. | Multiple-choice accuracy. | Useful optional hard-reasoning pack. | [Official GPQA repository](https://github.com/idavidrein/gpqa). |
| HumanEval | Code synthesis. | Pass at k through executable tests. | Later opt-in pack only because generated code requires a robust sandbox. | [Official HumanEval repository](https://github.com/openai/human-eval). |

The lm-evaluation-harness is useful as a behavior reference because it supports reusable task definitions and local OpenAI-compatible model interfaces. [lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness)
q.it should not embed the harness as its runtime because doing so would make Python and arbitrary task code part of the trusted execution path.
Instead, q.it should port and test only the scorer and prompt behavior needed by each declarative pack.

### Tool calling and structured extraction

| Benchmark pack | What it measures | Scoring | Pi practicality | Primary source |
|---|---|---|---|---|
| BFCL | Function selection, arguments, parallel calls, and multi-turn tool behavior. | Category-specific AST or executable-call accuracy. | High-value second-wave pack for Ollama, Transformers Serve, Cactus, and Needle. | [Official Berkeley Function Calling Leaderboard repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard). |
| Needle extraction | Typed field extraction and constrained structured output. | Field micro-F1 and parse validity. | High-value Cactus and Needle pack after their adapter exists. | [Needle benchmark description](https://github.com/cactus-compute/needle#benchmarks). |

Benchmark tools must never invoke real user functions.
Tool calls should be evaluated against inert schemas and expected call records.

### Image to text and visual reasoning

| Benchmark pack | What it measures | Scoring | Pi practicality | Primary source |
|---|---|---|---|---|
| VQAv2 | General visual question answering with multiple human references. | Official consensus accuracy and normalization. | Strong first-wave image pack with a stable smoke subset. | [Official VQA evaluation specification](https://visualqa.org/evaluation.html). |
| COCO Caption | Image caption generation. | BLEU, METEOR, ROUGE-L, CIDEr, and optionally SPICE. | Useful, but the historical scorer has old Python and Java dependencies that should be replaced by tested built-in implementations. | [Official COCO caption scorer](https://github.com/tylin/coco-caption). |
| OCRBench | Text recognition, document QA, key information extraction, and OCR reasoning. | Category-specific exact or normalized matching. | High-value edge vision pack. | [Official OCRBench repository](https://github.com/qywh2023/OCRbench). |
| ChartQA | Chart understanding and visual arithmetic. | Relaxed and exact accuracy according to the official protocol. | Good standard visual reasoning pack. | [Official ChartQA repository](https://github.com/vis-nlp/ChartQA). |
| MMMU | Multi-discipline visual knowledge and reasoning over diverse image types. | Multiple-choice and open-answer accuracy. | Full suite is large, but the official validation split supports a deterministic standard profile. | [Official MMMU repository](https://github.com/MMMU-Benchmark/MMMU). |

Image preprocessing is part of the benchmark identity.
The result must record original dimensions, transmitted dimensions, encoding, compression quality when applicable, number of images, and whether the provider or q.it resized them.

### Speech, audio, and video to text

| Benchmark pack | What it measures | Scoring | Pi practicality | Primary source |
|---|---|---|---|---|
| LibriSpeech test-clean and test-other | English read-speech transcription under clean and harder conditions. | Word error rate with a pinned normalization. | Essential first-wave speech pack, with source test archives of roughly 346 MB and 328 MB. | [Official LibriSpeech distribution](https://www.openslr.org/12). |
| FLEURS language profiles | Multilingual speech transcription and language coverage. | Word error rate and character error rate per language. | Install only selected language profiles because the dataset exposes 103 language configurations. | [Official Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs). |
| AIR-Bench foundation | Speech, natural sound, music, and mixed-audio understanding through single-choice questions. | Accuracy across 19 task groups. | Practical second-wave audio-language pack because its foundation score does not require an external judge. | [Official AIR-Bench repository](https://github.com/OFA-Sys/AIR-Bench). |
| AIR-Bench chat | Open-ended audio-language conversation. | External model judging in the reference workflow. | Do not enable in the offline default dataset because judge choice changes the score and may require a cloud service. | [Official AIR-Bench repository](https://github.com/OFA-Sys/AIR-Bench). |
| Dynamic-SUPERB profiles | Speech, music, and general-sound tasks across classification, regression, and generation. | Task-specific metrics. | Broad later pack family, split into small task-specific installs rather than one 180-task download. | [Official Dynamic-SUPERB repository](https://github.com/dynamic-superb/dynamic-superb). |
| Video-MME v2 | Video understanding with fixed-frame or frame-rate sampling and optional subtitles. | Accuracy and grouped score. | Later pack with a small fixed smoke profile because media decode and long inputs are expensive on a Pi. | [Official Video-MME v2 repository](https://github.com/MME-Benchmarks/Video-MME-v2). |

The NIST SCTK repository provides the canonical `sclite` speech-scoring tool and related statistical utilities. [NIST SCTK repository](https://github.com/usnistgov/SCTK)
q.it should implement and fixture-test the normalization and edit-distance behavior it needs in Rust, while retaining the declared normalizer name and version in every score.

Audio and video results must record duration, sample rate, channels, codec, decoded sample count, frame-sampling policy, frames sent, subtitle mode, and media preprocessing time.
Without those fields, two models or providers can appear comparable while receiving materially different inputs.

### Text and multimodal embeddings

| Benchmark pack | What it measures | Scoring | Pi practicality | Primary source |
|---|---|---|---|---|
| MTEB text profiles | Text retrieval, reranking, classification, clustering, semantic similarity, pair classification, and bitext mining. | Task-specific official metrics. | Essential, but ship small named profiles rather than the whole catalog. | [Official MTEB repository](https://github.com/embeddings-benchmark/mteb). |
| BEIR profiles | Heterogeneous zero-shot information retrieval and reranking. | nDCG, MAP, recall, precision, and MRR. | SciFact and other small corpora are practical; the largest corpora should be optional. | [Official BEIR repository](https://github.com/beir-cellar/beir). |
| MIEB lite | Image and image-text embedding quality. | Classification, retrieval, similarity, and other task-specific metrics. | Good second-wave image-embedding profile. | [Official MIEB paper and code location](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_MIEB_Massive_Image_Embedding_Benchmark_ICCV_2025_paper.pdf). |
| MSEB retrieval | Sound and audio-text embedding quality. | Retrieval and other sound-embedding metrics. | Good second-wave audio-embedding profile. | [Official MSEB repository](https://github.com/google-research/mseb). |
| MAEB profiles | Audio embedding quality across speech, music, environmental sound, and cross-modal tasks. | Classification, retrieval, clustering, and related task metrics. | Later profile family because the complete benchmark is broad. | [Official MAEB paper](https://arxiv.org/abs/2602.16008) and [MTEB implementation](https://github.com/embeddings-benchmark/mteb). |
| MMEB v3 profiles | Text, image, video, audio, visual-document, tool, GUI, and memory retrieval. | Task-specific retrieval metrics. | Most extensive later pack family, divided by input type and asset size. | [Official VLM2Vec and MMEB repository](https://github.com/TIGER-AI-Lab/VLM2Vec). |

MTEB is now a multimodal evaluation toolbox rather than only a text benchmark, and its repository documents task, benchmark, model, and result interfaces across languages and modalities. [MTEB repository](https://github.com/embeddings-benchmark/mteb)
q.it should use its task taxonomy as input when designing packs while keeping q.it's pack payload declarative.

Retrieval and reranking packs must pin corpus revision, query revision, relevance judgments, distance function, normalization, candidate depth, rerank depth, and every prompt prefix.
Index-building time, query-encoding time, corpus-encoding time, search time, and reranking time should be separate measurements.
The dashboard should never present only combined query latency when an expensive index build was excluded.

### Output-media tasks

Text-to-speech, text-to-image, image-to-image, and text-to-video should be reserved task names even though the initial providers above do not offer a common stable interface for them.
The first pack for any generated-media task should be a performance and validity pack that records first output byte or first playable chunk, total latency, output duration or dimensions, and resource use.
Automated perceptual quality should not enter the successful run dataset until q.it pins an evaluator model, evaluator revision, preprocessing contract, and calibration protocol.
This avoids silently turning a benchmark of one edge model into a benchmark dominated by another untracked evaluator model.

## EmbeddingGemma 2 profile

EmbeddingGemma 2 is a 740M-parameter model that maps text, image, audio, and video into one 768-dimensional space and is intended to run locally on consumer CPUs and GPUs. [Google EmbeddingGemma documentation](https://ai.google.dev/gemma/docs/embeddinggemma)
It supports Matryoshka output dimensions of 128, 256, 512, and 768 and an 8,192-token shared context. [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
Its text path is approximately 270M parameters, the text-plus-image path approximately 440M, the text-plus-audio path approximately 570M, and the full model 740M because the vision and audio encoders can be omitted independently. [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2#2-selective-encoder-loading)
Google's model card warns against float16 for this model and recommends float32 on most CPUs. [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2#4-numerical-precision)
The model card documents default costs of 280 tokens per image, 140 tokens per video frame, and 25 tokens per audio second, with configurable video sampling and vision token budget. [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2#context-limits)

This model needs a purpose-built benchmark matrix rather than one aggregate score:

| Axis | Required profiles |
|---|---|
| Active encoders | Text only, text plus vision, text plus audio, and full multimodal. |
| Output dimension | 128, 256, 512, and 768. |
| Pair type | Text-text, text-image, image-image, text-audio, audio-audio, text-video, video-video, and interleaved input where a benchmark defines it. |
| Batch shape | One item and small batches that fit 4 GB, 8 GB, and 16 GB boards. |
| Precision | Float32 by default on Pi, with any other verified dtype stored explicitly. |
| Quality | MTEB, MIEB lite, MMEB, MSEB, and MAEB profiles as appropriate. |
| Performance | Load time, encode latency, items per second, media seconds per second, peak memory, and vector dimension. |

Ollama's published embedding API accepts text, and TEI's current supported-model page lists text EmbeddingGemma 1 rather than multimodal EmbeddingGemma 2. [Ollama embedding API](https://docs.ollama.com/api/embed) [TEI supported models](https://huggingface.co/docs/text-embeddings-inference/en/supported_models)
Transformers Serve's documented endpoint list has no embeddings route. [Transformers Serve endpoints](https://huggingface.co/docs/transformers/serve-cli/serving)
The initial implementation path should therefore be a q.it-owned local provider built around the official Transformers or SentenceTransformers interface, not a special executable hidden inside an EmbeddingGemma benchmark pack. [Official SentenceTransformers guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)

## Benchmark-pack contract

Each installed benchmark pack should declare:

- Stable pack identifier, version, publisher, and source URL.
- Task and required provider capabilities.
- Case schema version and scorer family version.
- Source dataset revision, split, license notice, and attribution.
- Prompt template and chat role mapping.
- Generation or inference settings.
- Output parser and normalization identifier.
- Quality metric names and aggregation rules.
- Stable smoke, standard, and full case lists.
- Asset URLs, byte sizes, cryptographic hashes, archive layout, and installed hashes.
- Per-case timeout and whole-run timeout.
- Retry policy, with retries retained as separate attempts rather than overwritten.
- Media preprocessing policy.
- Expected download size and expected installed size.
- Minimum q.it version and scorer support.

The pack installer should verify schema, hashes, task support, license metadata, case references, and total extracted size before activation.
Activation should be atomic so an interrupted download cannot become an available benchmark.
Pack removal should never delete benchmark results or the stored identity of the removed pack.

Because packs are inert, custom user benchmarks can still be extensive.
Users can author JSON and JSONL cases against existing task and scorer families without compiling Rust.
The authoring command should validate a pack, show resolved case counts and asset sizes, run scorer fixtures, and create a deterministic archive.

## Run identity, success, and failure

### Immutable run identity

A benchmark run should bind these dimensions before it changes from running to a terminal state:

- Benchmark-pack identifier, version, profile, case identifiers, and scorer version.
- Provider adapter and adapter version.
- Endpoint and whether it is local or remote.
- Model name, digest or immutable revision, format, quantization, dtype, and selected files when discoverable.
- Prompt template and inference parameters.
- Cold, warm, cache, concurrency, repetition, and timeout settings.
- Hardware and software environment snapshot.
- Telemetry scope and sampling period.
- Random seed and any deterministic shuffling details.

Ollama exposes a model digest and model-format details through its model inventory and show routes. [Ollama list-models API](https://docs.ollama.com/api/tags) [Ollama show-model API](https://docs.ollama.com/api-reference/show-model-details)
Hugging Face model acquisition should resolve a branch-like name to an immutable Hub revision before measurement whenever the provider permits it. [Hugging Face Hub download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)

### Terminal classification

The top-level state should remain exactly running, succeeded, or failed, matching the project glossary.

A run succeeds only when:

- Preflight validated the provider and selected task.
- Every required benchmark case reached the pack's defined terminal condition.
- Every required sample result and raw response metadata was persisted.
- Required quality summaries are finite and persisted.
- No environment-invalidating condition occurred.
- The run transaction committed its terminal summary.

A wrong answer is a valid sample result and lowers quality; it does not fail the run.
A parse-invalid answer can be a scored zero when the pack explicitly defines that behavior; it does not automatically indicate a harness failure.
A missing case after retries, provider crash, timeout, lost terminal marker, scorer error, out-of-memory event, or invalid environment fails the run.

Recommended failure classes are:

- Unsupported task or feature.
- Pack or asset validation.
- Provider unavailable.
- Model discovery or model load.
- Transport.
- Protocol or missing terminal marker.
- Timeout.
- Provider process exit.
- Out of memory.
- Output validation.
- Scorer.
- Environment invalid.
- User cancellation.
- Persistence.

Every failure record should retain phase, case identifier when applicable, attempt number, HTTP status, provider error code, provider message, process exit status, timeout kind, OOM counter delta, last valid stream event, and the bounded tail of provider logs.
Secrets, authorization headers, and raw private prompts must be redacted before persistence.

An Ollama HTTP 200 is not enough to mark success because its documented mid-stream errors cannot change the already sent status. [Ollama error handling](https://docs.ollama.com/api/errors)
A TEI adapter should preserve its distinct HTTP validation, payload-size, backend, and overload failures rather than collapse them into one provider error. [TEI OpenAPI](https://huggingface.github.io/text-embeddings-inference/openapi.json)

## Storage and successful run dataset

SQLite is sufficient for a single-Pi runner if raw media is stored as content-addressed files rather than database blobs.

The durable model should separate:

- Benchmark-run identity and terminal outcome.
- Model and provider snapshot.
- Host environment snapshot.
- Sample results and retry attempts.
- Scalar measurements with units, source, and aggregation scope.
- Telemetry time series stored in compressed chunks.
- Quality summaries by benchmark, category, and subject.
- Failure details.
- Asset and pack provenance.

All attempted runs remain in run history.
The successful run dataset should be a database view or equivalent query restricted to succeeded runs rather than a second copied database.
This prevents drift while satisfying the rule that failed runs never enter analytical comparisons or exports.

Exports should include JSONL for lossless exchange and CSV for flat summaries.
Every exported row should carry units and enough identity fields to reject accidental comparisons across different packs, profiles, dimensions, media policies, or telemetry scopes.

## Analysis dashboard

The dashboard should be optional, read only, loopback-bound by default, and able to start after benchmark execution has finished.

The highest-value views are:

- Pareto scatterplots for quality versus TTFT, quality versus decode rate, quality versus peak RAM, and decode rate versus peak RAM.
- Concurrency curves for system token throughput, per-request TTFT, and per-user decode rate.
- Grouped bars for one comparable metric across model, quantization, provider, and benchmark profile.
- Box plots or percentile bands for repeated latency measurements.
- CPU, RAM, temperature, frequency, swap, and pressure traces aligned to case boundaries.
- Quality breakdowns by subject, language, task family, and input type.
- Failure counts and failure-class heatmaps.
- Cold versus warm paired comparisons.
- Embedding dimension versus quality, latency, and memory tradeoffs.

Pareto charts must compare only runs with compatible task, pack version, profile, scorer, input policy, and environment-valid state.
The UI should show why a point is dominated and which dimensions differ when two runs cannot be compared.
It should not average unrelated benchmark scores into a universal model-quality number.

## Recommended delivery sequence

### First release

- Cargo-installed q.it command.
- Benchmark-pack list, add, remove, verify, and offline-import flows.
- Ollama native, Transformers Serve, and TEI provider adapters.
- Capability probe and one-case preflight.
- Linux host telemetry, optional process telemetry, and cgroup v2 support when q.it owns the provider.
- Run history with strict succeeded or failed classification.
- Successful run dataset and JSONL or CSV export.
- Fixed-shape text performance, MMLU-Pro, GSM8K, IFEval, VQAv2, OCRBench, LibriSpeech, small FLEURS language profiles, small MTEB profiles, and small BEIR profiles.
- Optional local dashboard with Pareto, bar, percentile, trace, and failure views.

### Second release

- Cactus, Needle, and Whistle provider behavior.
- BFCL and structured extraction.
- AIR-Bench foundation profiles.
- COCO caption, ChartQA, MMMU validation profiles, and Video-MME v2 smoke profile.
- MIEB lite, MSEB, MAEB, and multimodal embedding profiles.
- q.it-owned local Transformers or SentenceTransformers embedding provider for EmbeddingGemma 2.
- Sustained thermal profiles and concurrency sweeps.

### Later releases

- MMEB v3 profile families split by task and asset size.
- Dynamic-SUPERB task families.
- Sandboxed code evaluation.
- Generated-audio, generated-image, and generated-video performance contracts after stable provider APIs exist.
- Cooperative remote telemetry for a provider running on another host.

## Decisions to preserve during implementation

1. Use native Ollama rather than its OpenAI compatibility layer when collecting benchmark data.
2. Treat `hf` as acquisition and `transformers serve` as serving.
3. Keep provider-reported and client-observed timing side by side.
4. Never infer a task from a model name when a capability probe can establish it.
5. Never equate a streamed chunk with one token.
6. Keep cold, warm, sustained, and concurrency profiles separate.
7. Label every CPU and RAM value with its attribution scope.
8. Keep incorrect model answers distinct from failed benchmark execution.
9. Store all failed attempts, but expose only succeeded runs through the successful run dataset.
10. Keep benchmark packs declarative and move every executable scorer or protocol adapter into reviewed q.it code.
11. Pin cases, assets, prompts, normalization, scorer behavior, model revision, and environment before calling two runs comparable.
12. Make thermal throttling, undervoltage, OOM, swap, and severe contention visible instead of allowing them to silently distort Pi 5 results.
