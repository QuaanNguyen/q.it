# Optional model adapter package

This package provides local artifact verification, CPU parameter audits, offline startup, and the existing q.it local CPU serving protocol.
The catalog currently contains five optional text decision routes.
GLiClass v4 edge and GLiNER2.5 Decide use their classifier libraries.
Intern Decision, Liquid d1 omni, and typed-decision Gemma use pinned publisher code.
Their model files and isolated runtimes are installed separately by the user.
Installing it does not download a model or install a model runtime.

```bash
python3.12 -m venv /tmp/qit-adapter
/tmp/qit-adapter/bin/pip install ./providers/model-adapters
/tmp/qit-adapter/bin/qit-adapter list
```

The list command reports the available routes without importing their runtimes.
Verification and serving reject models absent from the catalog.

## Run a text classifier

Use Linux CPython 3.12 and install one runtime in a separate environment.
Choose the lock matching the runtime and host architecture.

```bash
python3.12 -m venv .venv-adapter
.venv-adapter/bin/pip install uv==0.9.5
.venv-adapter/bin/uv pip sync --python .venv-adapter/bin/python --torch-backend cpu --require-hashes providers/model-adapters/qit_adapters/locks/gliclass-linux-aarch64-py312.txt
.venv-adapter/bin/uv pip install --python .venv-adapter/bin/python --no-deps ./providers/model-adapters
.venv-adapter/bin/qit-adapter verify --model knowledgator/gliclass-v4-edge --model-path /absolute/path/to/pinned-model
.venv-adapter/bin/qit-adapter serve --model knowledgator/gliclass-v4-edge --model-path /absolute/path/to/pinned-model
```

Use the `linux-x86_64` lock on Sol.
GLiNER uses the `gliner2` lock and the original `fastino/GLiNER2.5-Decide` checkpoint.
The original artifact manifest pins each required file and refuses missing or modified files before loading.
There is no automatic acquisition or runtime fallback.

Run the original three-case decision pack against the serving endpoint.

```bash
qit benchmark install providers/model-adapters/packs/english-choice
qit run english-choice-smoke/choice --provider local-cpu --model knowledgator/gliclass-v4-edge --base-url http://127.0.0.1:8766 --warmups 0 --iterations 1
```

Both routes retain the publisher's complete ordered probabilities and refuse generation settings.
GLiClass refuses truncated input and duplicate candidate text.
GLiNER observes native complete probabilities before the publisher's public top-choice decoding and restores the original method after every request.
The loaded stack is audited as CPU float32 independently of the model-card count.

The [retained CPU checks](../../docs/research/optional-classifier-cpu-checks.md) describe original installed-command smoke runs and publisher comparisons.
The three cases test transport and scoring rather than dataset-wide quality.
Dependency locks and these Sol checks do not establish native Raspberry Pi qualification.

## Run a publisher decision model

Install the matching `intern`, `liquid`, or `typed-gemma` lock in a separate environment.
The three checkpoints require different pinned Transformers versions and must not share a runtime environment.
Review the pinned publisher Python files before opting in to execution.

```bash
.venv-adapter/bin/qit-adapter serve --model internlm/Intern-Decision-0.8B --model-path /absolute/path/to/pinned-model --allow-reviewed-code
```

The matching checkpoints are `LiquidAI/d1-omni-600M` and `mrm8488/embeddinggemma-2-ft-typed-decision` for the other two locks.
The adapter checks required file identities before applying the publisher-code gate and importing a model runtime.
Changing the pinned source requires a new source review and CPU verification.
All three retain their publisher's complete single-choice probabilities and loaded CPU float32 audit.
Intern and Liquid accept text cases only; their multimodal model components are still included in the loaded-stack count.
Image and audio decision execution are not included in these routes.
Use the existing three-case English pack to check transport and scoring.

## Adding a route

A route supplies a catalog entry with a model identifier, immutable 40-character checkpoint revision, task, precision, license, exact dependency requirements, and required file identities.
Each file identity contains its byte size and either its SHA-256 digest or original Git blob identity.
Model-card evidence must establish a size strictly below one billion parameters before a checkpoint is admitted.
The loaded-stack audit is a separate runtime check.

Register a loader in `qit_adapters/registry.py` for the catalog entry's adapter name.
The loader receives the profile, resolved local model directory, and CPU thread count.
It returns a runtime with an `audit` dictionary containing the complete learned parameter count, a decision `configuration` dictionary or `None`, and an `invoke` method.
Decision invocation receives the case; other tasks also receive the output limit.
The runtime returns a result event using the existing local CPU protocol.
Optional preparation metadata is retained when present.

Torch loaders can use the shared tensor auditor to check CPU placement and precision, count shared tensors once, and include learned components outside registered modules.
Loaders own CPU runtime initialization and must audit the complete model stack before returning.
Publisher-code loaders require the registry's reviewed-code flag and explicit source-review admission at startup.

A route also supplies its hash-pinned Linux CPython 3.12 dependency lock for each supported architecture.
Startup verifies the lock's installed package versions and retains lock, adapter-source, and installed-package provenance.
Required model artifacts are verified before loading, and startup blocks external Python socket connections before invoking the loader.
Native network behavior requires separate traced verification.
The server binds to IPv4 loopback and permits one inference request at a time.

## Checks

```bash
python3 -m unittest discover -s providers/model-adapters/tests
```

The tests use fixture-only profiles and runtimes, local HTTP requests, and temporary files.
They establish package and protocol behavior without executing a model.
They do not qualify any checkpoint, dataset, runtime installation, or Raspberry Pi performance.
