# Optional classifier CPU checks

These are installed-command CPU float32 smoke checks and independent publisher-output comparisons on Sol Linux x86_64.
They establish compatibility for the pinned checkpoint and runtime combinations below.
They do not establish dataset-wide quality, performance rankings, native ARM64 installation, or Raspberry Pi qualification.

| Route | Pinned checkpoint | Original publisher card | Hugging Face card count | Loaded learned count | Retained result |
| --- | --- | --- | ---: | ---: | --- |
| GLiClass v4 edge | `90be5550327b138635db1f63531f9dfa15e624da` | [Original card](https://huggingface.co/knowledgator/gliclass-v4-edge/blob/90be5550327b138635db1f63531f9dfa15e624da/README.md) | 439,790,595 | 439,790,596 | Three English cases succeeded and publisher probabilities matched. |
| GLiNER2.5 Decide | `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90` | [Original card](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md) | 486,444,053 | 486,444,053 | Three English cases succeeded and the publisher's public winner and confidence matched. |

The numeric card counts are the original repositories' Hugging Face safetensors metadata, verified on 2026-10-20 at the exact pinned revisions.
GLiClass's prose card rounds its size to 0.4B, while GLiNER's prose card states 340M.
The larger exact stored and loaded counts above remain strictly below one billion.
The loaded audit is independent of stored metadata and is enforced again at startup.

The original draft retained these results before this extraction:

- GLiClass evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/0/text` on Sol.
Its adapter source was `7c5964ec8fe13b259b7b37a8e1e9c941e0e76082`.
- GLiNER evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/4/outcomes/1/text` on Sol.
Its adapter source was `7494884bb843cc2c80e3aa4e655a75b1654a037c`.
- Both used q.it binary source `3d112569a9d7f28f7185b25d28e2f2fd2b8c4b8f` and four CPU threads.
Each result directory retains metadata, parameter audits, raw observations, successful exports, the SQLite store, and publisher-comparison output.

GLiNER's public API returns only the winning label and its confidence.
The adapter obtains the complete probability vector from the pinned native classification function before public decoding and restores the original method after invocation.
Its fixture checks verify ordering, complete-vector requirements, failure handling, and restoration.
The retained public-output comparison does not independently verify every GLiNER option probability.

The extraction uses the same constructors, input preparation, native scoring, and artifact identities as the corresponding draft routes.
Its loader uses the shared package registry and attaches the existing complete CPU float32 audit.
Fixture and installed-package checks validate that integration separately from the historical real-model runs.
No new model run is implied by those fixture checks.
