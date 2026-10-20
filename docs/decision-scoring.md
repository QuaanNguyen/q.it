# Single-choice decision benchmarks

Decision scoring asks a model to select one answer from an ordered set of candidates and return a probability for every candidate.
The task uses an explicitly started, compatible Local CPU endpoint.
The [optional adapter package](../providers/model-adapters/README.md) loads cataloged decision models from verified local files in separate runtimes.

## Pack format

Keep text-only and image-backed cases in separate benchmarks so their results cannot enter the same comparison.
Both can live in one pack, with separate case files and stable benchmark identities:

```json
{
  "schema_version": 1,
  "id": "decisions",
  "name": "Decision examples",
  "version": "1.0.0",
  "description": "Synthetic single-choice examples.",
  "publisher": "Your name",
  "license": "CC0-1.0",
  "source": "Local examples",
  "benchmarks": [
    {
      "id": "text-choice",
      "name": "Text choices",
      "description": "Text-only examples.",
      "task": "decision_scoring",
      "decision_input": "text",
      "cases": "text.jsonl"
    },
    {
      "id": "image-choice",
      "name": "Image choices",
      "description": "Image-backed examples.",
      "task": "decision_scoring",
      "decision_input": "image",
      "cases": "image.jsonl"
    }
  ]
}
```

Each case contains a nonempty state, one nonempty question, and two to ten ordered candidates.
Candidate identifiers must be unique, at most eighty ASCII letters, digits, underscores, or hyphens; candidate text must be nonempty.
The optional expected answer references one candidate identifier:

```json
{"id":"weather","decision":{"state":"The sky is clear and blue.","question":"Which description fits?","candidates":[{"id":"rain","text":"Rainy"},{"id":"clear","text":"Clear"}],"expected":"clear"}}
```

Image-backed cases use the same structure with an additional image:

```json
{"id":"signal","media":"signal.png","media_type":"image/png","decision":{"state":"An intersection is shown.","question":"What should the driver do?","candidates":[{"id":"stop","text":"Stop"},{"id":"go","text":"Proceed"}],"expected":"stop"}}
```

Images must be valid PNG or JPEG files inside the pack, or matching base64 data URLs.
Remote URLs, escaping paths, and symbolic links are rejected.
Images are limited to sixteen MiB of encoded file bytes, 8192 pixels per dimension, and a sixty-four MiB decoder allocation budget.
Declared media types must match the image format and any data URL media type.
Every image benchmark case must have an image, and every text benchmark case must omit it.
Generation prompts, retrieval fields, image-generation settings, and their scoring fields are rejected for this task.
Increase the pack version whenever cases, labels, media, or input preparation change.

Verify, install, and run through the normal command:

```bash
qit benchmark verify /path/to/decisions
qit benchmark install /path/to/decisions
qit run decisions/text-choice --provider local-cpu --model my-decision-model --base-url http://127.0.0.1:8766 --iterations 3 --warmups 1
```

Decision runs reject generation token limits.
Unlabelled cases still produce durable observations and latency measurements, with accuracy and probability-quality metrics absent.

## Provider contract

The Local CPU protocol uses `GET /v1/qit/model` and NDJSON from `POST /v1/qit/invoke`.
Preflight requires the requested model, protocol version one, a nonempty server version, CPU execution, a nonempty checkpoint revision, and a loaded parameter count greater than zero and below one billion.
Decision precision is float32 except for two pinned native D3 configurations.
D3 Edge at `c454ee39b862248d70abd7f84195d2f54ffe6d43` and D3 Lite at `03560c26f5199c3c04129a2ccb38960c913635e0` admit `bfloat16` only with `precision_admission` equal to `d3-native-bf16-backbone-fp32-readout-v1`.
The adapter emits that identity after observing a CPU FP32 learned readout and auditing the complete loaded stack, including its native BF16 backbone.
All other model and revision combinations retain float32 admission.
D3 supports text and image cases through the same protocol.
[Retained D3 CPU checks](research/d3-cpu-checks.md) document the historical execution scope separately from current fixtures.
The decision configuration declares supported inputs and stable identities for calibration, native selection rules, and input preparation:

```json
{
  "schema_version": 1,
  "version": "adapter-1.0.0",
  "model": "my-decision-model",
  "configuration": {
    "device": "cpu",
    "dtype": "float32",
    "task": "decision_scoring",
    "parameter_count": 123456789,
    "revision": "pinned-checkpoint-revision",
    "decision": {
      "schema_version": 1,
      "inputs": ["text", "image"],
      "calibration": "native-v1",
      "selection_rule": "native-first-maximum",
      "preprocessing": "pinned-template-and-image-preparation"
    }
  }
}
```

Record publisher code revisions, dependency versions, threads, and model-specific preparation in the surrounding configuration as appropriate.
The full configuration is retained in history.
Dashboard model identity retains decision configuration changes while excluding the operational process identifier.
The existing comparison scope still separates benchmark versions, hosts, run settings, and serving environments.

Invocation contains only the model, task, case identifier, state, question, ordered candidates, and optional normalized image data URL.
Expected labels and generation token limits are never sent:

```json
{"model":"my-decision-model","task":"decision_scoring","case":{"id":"weather","decision":{"state":"The sky is clear and blue.","question":"Which description fits?","candidates":[{"id":"rain","text":"Rainy"},{"id":"clear","text":"Clear"}]}}}
```

Return exactly one terminal result event, followed by the end of the response:

```json
{"type":"result","decision":{"schema_version":1,"candidates":[{"id":"rain","probability":0.25},{"id":"clear","probability":0.75}],"selected_id":"clear","provenance":{"calibration":"native-v1","selection_rule":"native-first-maximum","preprocessing":"pinned-template-and-image-preparation","input_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","input_truncated":false,"native":{"example":true}}}}
```

The digest above is illustrative.
Providers must report the SHA-256 hexadecimal digest of the actual rendered input and retain relevant native preparation, calibration, and tie evidence in the native evidence object.
q.it validates the digest shape and preserves it; it cannot independently reconstruct a provider's internal model input.
Truncated inputs fail rather than silently entering full-input comparisons.
Calibration, selection, and preprocessing identities must match preflight.

Returned candidates must match the complete requested identifier order exactly, without omissions, additions, or duplicates.
Probabilities must be finite, lie between zero and one, and sum to one within absolute tolerance `0.000001`.
q.it does not renormalize them.
The selected candidate must have maximal probability; exact ties preserve the provider's native selection.
Each candidate may include a finite native `logit`; unavailable logits remain absent.
q.it does not fabricate logits or recalibrate scores.
Missing results, duplicate terminal events, generation deltas, malformed distributions, and inconsistent provenance fail the run.

## Scores and retained evidence

Correct-choice accuracy is the proportion of labelled measured cases whose selected identifier matches the expected identifier.
Higher accuracy ranks first, and equal accuracy receives the same rank.
Probability-quality metrics appear alongside accuracy and never form a composite ranking.

For probabilities `p[k]` and a one-hot expected answer `y[k]`, multiclass Brier score is `sum((p[k] - y[k])^2)` over all candidates, without dividing by candidate count.
Negative log likelihood is `-ln(p[expected])`, using the natural logarithm without clipping.
Lower values are better for both metrics.
Run summaries average each metric over labelled measured samples, excluding warmups.
For an expected probability of zero, the sample and aggregate log loss use `{"status":"infinite"}`, and the summary counts such samples explicitly.
Finite losses use `{"status":"finite","value":0.2876820724517809}`.
CSV writes infinite loss as `infinite`, and the dashboard displays it as `∞`.

A valid wrong answer is a successful invocation with accuracy zero.
Protocol and transport failures remain in run history and are excluded from successful-run exports.
JSONL exports and run-detail responses retain the original decision case, expected label, actual image data URL, ordered probabilities, optional logits, selection, provenance, and independent metrics.
Pack removal does not delete this evidence.
CSV adds flat Brier, log-loss, and zero-gold-probability summary columns.
Historical runs without decision fields remain readable.

Latency covers the whole adapter invocation, including media preparation, request, response, parsing, and validation.
Existing CPU, host-memory, and optional provider-RSS observations remain available.
Generation token counts, time to first token, and token throughput remain absent.

Synthetic protocol checks establish transport and product behavior only.
Dataset quality scores require licensed, versioned evaluation packs and real models, and native Raspberry Pi qualification requires execution on that hardware.
Multi-field decisions, native boolean or ordered-score answer modes, and audio decisions remain outside this first implementation.
