# Optional classifier CPU checks

These are installed-command CPU float32 smoke checks and independent publisher-output comparisons on Sol Linux x86_64.
They establish compatibility for the pinned checkpoint and runtime combinations below.
They do not establish dataset-wide quality, performance rankings, native ARM64 installation, or Raspberry Pi qualification.

| Route | Pinned checkpoint | Original publisher card | Hugging Face card count | Loaded learned count | Retained result |
| --- | --- | --- | ---: | ---: | --- |
| GLiClass v4 edge | `90be5550327b138635db1f63531f9dfa15e624da` | [Original card](https://huggingface.co/knowledgator/gliclass-v4-edge/blob/90be5550327b138635db1f63531f9dfa15e624da/README.md) | 439,790,595 | 439,790,596 | Three English cases succeeded and publisher probabilities matched. |
| GLiNER2.5 Decide | `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90` | [Original card](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md) | 486,444,053 | 486,444,053 | Three English cases succeeded and the publisher's public winner and confidence matched. |
| Liquid d1 omni | `02b55d7076f15129e59ab3f94783f32c4b088674` | [Original card](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md) | 587,161,089 | 587,143,681 | Three English text cases succeeded and publisher probabilities matched. |
| Intern Decision | `85a0cc5a99d67ea8d56dfe98115689212867171d` | [Original card](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/README.md) | 852,985,920 | 852,985,920 | Three English text cases succeeded and publisher probabilities matched. |
| Typed-decision Gemma | `8403ff7fddc75b6994130e581a832f875fb5bd0e` | [Original card](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision/blob/8403ff7fddc75b6994130e581a832f875fb5bd0e/README.md) | 272,120,348 | 272,120,324 | Three English cases succeeded and publisher probabilities matched. |

The numeric card counts are the original repositories' Hugging Face safetensors metadata, verified on 2026-10-20 at the exact pinned revisions.
GLiClass's prose card rounds its size to 0.4B, while GLiNER's prose card states 340M.
The larger exact stored and loaded counts above remain strictly below one billion.
The loaded audit is independent of stored metadata and is enforced again at startup.
The stored metadata and learned-tensor audit are separate counts and are not substituted for one another.

The original draft retained these results before this extraction:

- GLiClass evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/0/text` on Sol.
Its adapter source was `7c5964ec8fe13b259b7b37a8e1e9c941e0e76082`.
- GLiNER evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/4/outcomes/1/text` on Sol.
Its adapter source was `7494884bb843cc2c80e3aa4e655a75b1654a037c`.
- Both used q.it binary source `3d112569a9d7f28f7185b25d28e2f2fd2b8c4b8f` and four CPU threads.
Each result directory retains metadata, parameter audits, raw observations, successful exports, the SQLite store, and publisher-comparison output.

The publisher-code routes retained the same three-case English coverage, four CPU threads, and q.it binary source:

- Liquid evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/4/outcomes/2/text`.
Its adapter source was `7494884bb843cc2c80e3aa4e655a75b1654a037c`.
- Intern evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/3/text`.
Its adapter source was `7c5964ec8fe13b259b7b37a8e1e9c941e0e76082`.
- Typed Gemma evidence is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/4/outcomes/6/text`.
Its adapter source was `7494884bb843cc2c80e3aa4e655a75b1654a037c`.

Their pinned Python source identities are verified before loading, and startup requires explicit reviewed-code admission.
Intern's pinned encoder rejects inputs above 8192 tokens without tokenizer truncation.
Liquid and typed Gemma preserve their native preparation limits and the adapter's explicit truncation refusals.
Intern and Liquid image execution remains unsupported even though the complete multimodal stack is counted.

GLiNER's public API returns only the winning label and its confidence.
The adapter obtains the complete probability vector from the pinned native classification function before public decoding and restores the original method after invocation.
Its fixture checks verify ordering, complete-vector requirements, failure handling, and restoration.
The retained public-output comparison does not independently verify every GLiNER option probability.

The extraction uses the same constructors, input preparation, native scoring, and artifact identities as the corresponding draft routes.
Its loader uses the shared package registry and attaches the existing complete CPU float32 audit.
Fixture and installed-package checks validate that integration separately from the historical real-model runs.
No new model run is implied by those fixture checks.

## FRIDA Russian text evidence

FRIDA-Decisions retained three successful original Russian text cases and matching complete native candidate distributions and choices on Sol Linux x86_64.
Its checkpoint revision is `e8fc105679af2ab44e4c2e52b65d29db58166083`.
The [pinned publisher card](https://huggingface.co/ai-forever/FRIDA-Decisions/blob/e8fc105679af2ab44e4c2e52b65d29db58166083/README.md) labels the route 823M.
The [pinned Hugging Face safetensors metadata](https://huggingface.co/api/models/ai-forever/FRIDA-Decisions/revision/e8fc105679af2ab44e4c2e52b65d29db58166083?blobs=true), verified on 2026-10-20, reports 823,401,216 stored encoder parameters.
The retained CPU float32 audit counts 823,402,753 learned parameters, including the 1,537-parameter scalar head.
Startup independently audits the complete loaded stack and requires strictly fewer than one billion parameters.

The retained result directory is `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/7/text`.
It used four CPU threads, draft adapter source `7c5964ec8fe13b259b7b37a8e1e9c941e0e76082`, and q.it binary source `3d112569a9d7f28f7185b25d28e2f2fd2b8c4b8f`.
The independent publisher comparison matched the full distributions and choices for weather, temperature, and reading.
The installed publisher runtime was pinned to commit `00c8b5f0a88312e969d65b01a076bd9e447de2db` and archive SHA-256 `9a927dbd9958b18772ec0b35f6240f0a865e3f4fbb6df3f5dabfbe0a74608e8b`.
The extraction retains the original constructor, compile and judge calls, truncation refusals, source inventory, and artifact identities.
Installed dependency source verification runs before publisher import and records the runtime commit separately from the checkpoint revision.

These historical checks establish compatibility of that original source and runtime combination.
The extraction's fixture and installed-package checks establish integration behavior separately.
No fresh model run, dataset quality, performance ranking, aarch64 installation, or Raspberry Pi qualification is claimed.
