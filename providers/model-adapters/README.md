# Optional model adapter package

This package provides local artifact verification, CPU parameter audits, offline startup, and the existing q.it local CPU serving protocol.
The catalog is empty, and this package does not include any model routes, model files, runtime extras, or dependency locks.
Installing it does not download a model or install a model runtime.

```bash
python3.12 -m venv /tmp/qit-adapter
/tmp/qit-adapter/bin/pip install ./providers/model-adapters
/tmp/qit-adapter/bin/qit-adapter list
```

The list command succeeds without output until a model route is added.
Verification and serving reject models absent from the catalog.

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
