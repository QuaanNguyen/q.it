---
status: accepted
---

# Single-choice decision scoring

Decision models return candidate distributions rather than generated text, so their first common task is single-choice decision scoring over text-only and image-backed cases in separate versioned benchmarks.
The Local CPU adapter accepts ordered candidates and preserves complete native probabilities, optional logits, selected answers, calibration, selection rules, input preparation, and truncation evidence.
Expected answers stay inside q.it's scorer and never enter provider requests.
Malformed distributions and truncated input fail the run, while valid wrong decisions remain successful scored outputs.

Correct-choice accuracy determines ranking, with multiclass Brier score and natural-log negative log likelihood displayed independently alongside it.
Zero expected probability produces explicit infinite loss and a separate count, without clipping or renormalizing native scores.
The original case and image are retained with measured outputs so removing a pack preserves auditable history.
Whole-invocation latency and existing resource measurements apply; generation-token counts, time to first token, and token throughput remain absent.

The contract extends the declarative pack and provider boundaries in ADR-0005 and the scoped comparisons in ADR-0006.
Model-specific adapters and real-model qualification remain separate from synthetic protocol verification.
Multi-field decisions, native boolean or ordered-score modes, and audio are deferred by the owner's initial scope decision.
The [decision-scoring guide](../decision-scoring.md) defines the public pack, protocol, validation bounds, and metric formulas.
