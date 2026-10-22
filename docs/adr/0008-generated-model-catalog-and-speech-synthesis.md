---
status: accepted
---

# Generated model catalog and speech synthesis

The terminal workspace needs one compatibility source for benchmarks, providers, and models.
Handwritten lists drifted between the TUI, the CLI, and the dashboard, and they could not express Hugging Face direct routes or optional adapters.
The catalog is therefore generated from the four installed metadata sources: the local CPU checkpoint list, the optional adapter manifests, the Ollama route map, and the Hugging Face direct route map.
`scripts/generate-model-catalog.py` joins them into `qit-runtime/src/model-catalog.generated.json`, and `--check` fails when the committed file drifts from those sources.
The runtime parses the embedded JSON, rejects provider and task combinations that have no transport, and exposes the same route facts to the TUI, the CLI, and the dashboard.

A selection is a benchmark, a provider, and a model route.
Each choice narrows the other two by task, so selecting a benchmark hides providers and models that cannot run it, and selecting a model hides benchmarks it cannot serve.
Choosing an incompatible value clears the other field with a typed cause, keeps the field the user just chose, and moves focus to the cleared field.
The dashboard still returns a derived provider list for each benchmark, preserving the existing web contract without storing a second handwritten list.

Availability is not compatibility.
Catalog membership means a route could be invoked, not that the model is installed or the endpoint is running.
An unprobed route remains selectable, while a failed probe blocks the start gate with the observed endpoint and model identity.
Probes are scoped to the exact route and base URL, so editing an endpoint requires a fresh probe.

Text to speech is a first-class task rather than a decision or transcription side effect.
The wire contract streams base64 16-bit PCM chunks and a terminal result carrying the sample rate, byte count, and synthesized duration.
q.it records time to first audio and audio duration with each sample, then computes the real-time factor during summarization instead of trusting a provider-reported ratio.
The first catalog candidates are the pinned Phonon 31m and Phonon 40m multilingual checkpoints.
Their publisher-runtime prototype remains isolated from serving until native dependency locks and a complete loaded parameter audit are available.
No model is bundled or downloaded during serving, and the adapter converts the publisher float32 stream into the validated PCM contract.

The catalog describes only reviewed routes.
Running an unlisted model remains possible through the CLI, but it is not offered by the picker until its metadata and pinned revision are added to one of the sources.
Adding a route requires regenerating the catalog, which keeps provider protocol support and selection behavior checked in one place.
