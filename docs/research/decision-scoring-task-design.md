# Proposed decision-scoring task

Research date: 2026-10-16.
Answers [Design a task-correct decision scoring route](https://github.com/QuaanNguyen/q.it/issues/106).
The owner approved planning decision benchmarking and selected shared single-choice text/image cases for the first rollout in [Choose the initial decision benchmark scope](https://github.com/QuaanNguyen/q.it/issues/107).
Multi-field decisions and Liquid-only audio are deferred.
The owner selected correct-choice accuracy as the main ranking, with probability-quality scores alongside it.
This is a proposed contract, not implemented support or approval to execute a restricted-license model.
It builds on the pinned source and environment audits in [Vision and embedding candidate serving routes](vision-embedding-candidate-routes.md).

## Common contract

Decision scoring evaluates explicit candidate answers about a supplied state, preserving their distributions and answer semantics.
It belongs in a distinct task because current [task and result types](../../qit-runtime/src/model.rs), [provider dispatch](../../qit-runtime/src/provider.rs), and [pack guide](../benchmark-packs.md) represent generation, retrieval, and media tasks without typed candidate distributions.
Extend the existing pack, provider, sample persistence, export, and comparison seams rather than placing executable publisher code inside a pack.
That preserves [the accepted declarative-pack boundary](../adr/0005-benchmark-packs-and-provider-adapters.md).

Each case supplies a stable case identity, a text state, an ordered list of named questions, and an ordered list of candidates per question.
Each question has instructions and a declared answer shape: single choice, boolean, or ordered score.
Each candidate has a stable identifier and its descriptive text; ordered scores additionally have explicit numeric values.
Use arrays for order at the pack and transport boundary, then construct publisher-native ordered mappings inside each adapter.
Expected labels belong only to the benchmark scorer and must never be sent to the provider.
Optional media must use the existing pack-local path and data-URL validation rules; image order is part of the case identity.
Structured state serialization, if later included, must be fixed by the pack version rather than varying with a provider's JSON defaults.

The common result retains ordered candidate identifiers, native probabilities, the selected candidate or expected numeric score, native confidence where available, and score provenance.
Retain native answer fields alongside the common representation so translation remains auditable.
Validate candidate identity, cardinality, question coverage, finite probabilities in the closed interval from zero to one, and distribution sums within an explicit versioned tolerance.
Never silently renormalize malformed results, reorder candidates by score, or insert missing probability entries.
Malformed responses fail the run; valid wrong answers remain successful scored samples, following the [existing run contract](../agents/project.md).

Logits are optional native evidence, not a required substitute for probabilities.
If exposed without another model call, preserve candidate logits with their extraction position, candidate mapping, calibration stage, and source revision.
Otherwise mark logits unavailable; taking logarithms of probabilities does not recover native logits.
Model-specific probability semantics and calibration remain distinct model configurations rather than being homogenized by a benchmark-level softmax.

## Publisher translation

Liquid d1 uses its pinned decision API and native option-marker compiler, with the full text, decision, vision, and audio stack retained.
Its native probability method returns boolean candidates in yes/no order although its compiler renders false/true order; the adapter must map by semantics rather than positional assumptions.
Its compact boolean answer exposes only the probability of yes, so obtaining the full distribution should use the same native scoring invocation, never a second measured model call.
Text probabilities use saved per-question-type and option-count temperatures; image and audio probabilities use the native uncalibrated softmax.
Its answer-selection behavior preserves the first candidate on an exact choice tie.
Sources: [pinned model and probability path](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py), [candidate compiler and answer conversion](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/prompt.py), and [saved calibration](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/config.json).

Intern-Decision uses its pinned compiler, tokenizer, processor, assistant skeleton, empty thinking block, and decision-placeholder positions.
Its forward pass reads candidate-symbol logits immediately before each placeholder, requires each candidate symbol to tokenize to exactly one token, and softmaxes only over permitted symbols.
Preserve its native two-step calibration and checkpoint-specific temperature 2.747760550703; the engine uses lexical candidate-value tie-breaking.
It returns a count of scored fields under its output-token label, which must be retained as publisher metadata without becoming generated-token usage.
Source: [pinned engine, compiler, and calibration](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/inference.py).

For common accuracy, preserve a publisher's explicit selected decision and tie rule.
Where a native shape supplies only a probability or expected score, an adapter-derived discrete selection must identify its benchmark rule separately; propose first candidate on a probability tie, pending the scope discussion.
Record exact ties so selection-rule differences are visible.
Retain each native expected score; for comparable ordered-score cases, propose consecutive values starting at zero, which both publishers can represent without changing Liquid's index-based expectation.
Arbitrary numeric scales, boolean threshold changes, and multi-question requests require explicit future acceptance rather than implicit coercion.

## Reproducibility and validity

Comparison identity must include canonical checkpoint and code revisions, processor and tokenizer revisions, native compiler version, candidate and question order, calibration settings, tie rule, precision, threads, attention mode, package lock, input limits, preprocessing, and the pack version.
Also retain model-specific rendered-input hashes and truncation evidence while keeping the original case and media hashes available.
Different token counts are expected because the native compilers differ; equal case content does not imply equal tokenization or forward-pass shape.

Intern rejects input exceeding its configured context rather than truncating it.
Liquid truncates state and per-option text in its compiler and has a smaller 896-token text budget when images are present.
Reject cases from a shared full-input benchmark if either adapter detects loss of required state, question, or option content; do not silently compare truncated and complete cases.
The first implementation needs observable preprocessing evidence or a conservative validated bound before it can claim full-input comparability.
Sources: [Intern input validation](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/inference.py), [Liquid truncation](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/prompt.py), and [Liquid media budgets](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py).

## Selected quality presentation and metric semantics

For the selected single-choice benchmark, use one gold candidate per case and top-choice accuracy as the primary quality score: one for a selected gold candidate, zero otherwise.
Rank by accuracy with higher values better, and show multiclass Brier score and negative log likelihood separately with lower values better.
Do not combine accuracy and probability-quality metrics into a composite ranking.
Report the arithmetic mean over all measured case repetitions, with benchmark passes and case sets fixed for comparisons.
Missing labels produce absent quality metrics rather than an assumed correct answer.

Preserve two probability-sensitive metrics beside accuracy: negative log likelihood of the gold candidate using the natural logarithm, and multiclass Brier score as the sum of squared differences between candidate probabilities and the one-hot gold distribution.
Do not divide Brier score by the number of candidates; it ranges from zero to two, and lower is better.
Negative log likelihood is zero for probability one and positive infinity for a gold probability of zero.
Do not clamp zero silently: retain zero-gold-probability counts and represent the aggregate as explicitly infinite using a structured status, since JSON cannot encode infinity as a numeric value.
Keep metric names, direction, aggregation, and version explicit; a single quality-score field cannot honestly encode all three meanings.
Do not multiply probability metrics by an arbitrary factor to fit the current dashboard's score display.

Boolean cases can use the same two-candidate metrics once the discrete answer and tie rule are chosen.
For later ordered-score cases, propose mean absolute error between the native probability-weighted numeric answer and the gold numeric value, alongside candidate Brier score and log likelihood when a gold level exists.
Do not infer categorical correctness from rounded expected scores.
For later multi-field cases, propose separate per-field results plus an equal-case mean of field accuracy and an all-fields-correct case metric; do not let cases with more fields silently dominate the benchmark.
These later shapes and their primary ranking metrics remain proposals for the human scope ticket.

Calibration assessment must use a fixed labeled split and preserve the native calibration regime in result identity.
Brier score and log likelihood provide directly specified probability-quality evidence; do not fit temperatures on benchmark labels or call accuracy alone evidence of calibrated probabilities.
Reliability plots or expected calibration error would require a separately versioned binning contract before becoming comparable metrics.

## Timing, telemetry, and qualification

Measure one full provider invocation with a monotonic clock, covering native preprocessing, media processing where applicable, forward scoring, calibration, and result transport.
Load the model before warmups, retain measured and warmup counts separately, and exclude benchmark-side quality scoring from invocation latency.
Native forward-only timings may be retained as additional publisher telemetry, clearly separated from end-to-end latency.
Report existing latency percentiles and host CPU, host memory, and provider-process RSS using the [runner](../../qit-runtime/src/runner.rs) and [telemetry seam](../../qit-runtime/src/telemetry.rs).
Record sampling interval and preserve unavailable observations, since very short decisions may finish between resource samples.
Never emit time to first token, generated-token count, or token throughput for decision scoring.
Record native input positions and decision-field count only with their true meanings; do not rank unlike multi-field and single-field requests by an invented token metric.

The static full-stack learned counts are 587,143,681 for Liquid d1 and 852,985,920 for Intern-Decision.
Qualification must independently verify unique loaded learned counts, complete component identity, CPU placement, and float32 across every learned component, including unused modality encoders retained in the audited stack.
Use isolated, locked environments: Liquid requires Transformers at least 5.15, while Intern's publisher pins Transformers 5.14.1, PyTorch 2.9.1, and torchvision 0.24.1.
The [prior audit](vision-embedding-candidate-routes.md) records exact revisions, static counts, ARM64 limitations, and the unresolved Liquid license gate.
Successful Sol CPU execution would establish that host's qualification only; Raspberry Pi support still requires separate ARM64 installation and execution evidence.

## Selected first scope and proposed acceptance gates

The selected first benchmark scope is shared single-choice text/image cases for both models.
Include text-only cases and image-backed cases with one named question and one selected candidate answer per case.
Keep text-only and image-backed measurements in separate versioned benchmarks so their quality and latency are not silently averaged together.
Two to ten explicitly ordered candidates and one gold answer remain the proposed bounded case contract.
Multi-field decisions and Liquid-only audio are deferred by the owner's scope choice.
Native boolean and ordered-score answer shapes remain outside this first implementation; a two-option case can still use the common single-choice shape.

1. Apply the selected single-choice text/image scope and accuracy-first ranking, retaining native calibration and reporting probability-quality evidence alongside accuracy.
2. Extend pack verification and provider result validation with data-only decision cases, expected-answer isolation, ordered identities, typed distributions, and explicit metric semantics.
3. Use fake provider tests to cover order preservation, exact ties, missing or duplicated candidates, malformed distributions, zero-probability gold labels, truncation, persistence/export round trips, and absent generated-token metrics.
4. Validate each adapter's common result against its pinned publisher-native result on the same approved cases, without adding another forward pass to the measured invocation.
5. Complete license eligibility and trusted-code review separately, then perform explicit full-stack CPU audits and smoke runs in the locked environments.
6. Run a versioned, licensed, fixed dataset benchmark and retain every failure as well as successful quality evidence before advertising model support.
7. Require a separate clean ARM64 install and real-device execution gate before claiming Raspberry Pi support.

The initial answer/media scope and primary quality presentation are settled by the owner's live answers.
Preserve each publisher's native calibration and provenance; any later change needs a distinct comparison configuration rather than fitting temperatures on benchmark labels.
The map's standing independent-work preference permits each model's readiness to be evaluated separately while keeping comparisons within genuinely compatible configurations.
License eligibility, approved model acquisition, dataset rights, actual implementation, and model qualification remain independent gates.
