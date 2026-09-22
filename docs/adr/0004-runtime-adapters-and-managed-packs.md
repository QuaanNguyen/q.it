# Runtime adapters select executable material

Model-package recipes declare runtime compatibility, but they do not resolve executables or construct worker commands.
The runtime registry selects one adapter for a runtime recipe, and that same adapter determines readiness and launches the worker.
An adapter owns serve-profile validation, executable resolution, command translation, health and generation protocol endpoints, and runtime-specific diagnostics.
The supervisor owns session and child-process lifecycle without selecting executable paths.

Managed runtime packs are immutable, versioned executable material selected by runtime adapters.
The runtime-pack module owns compatibility evaluation, trusted archive acquisition, archive and installed-content verification, atomic activation, discovery, selection, repair, update, removal, and offline archive provisioning.
Readiness and launch resolve the same selected runtime adapter and runtime pack.
Installing or updating a managed pack is always an explicit user operation and never occurs as a side effect of Start or Try.
The pack repository uses TUF-compatible signed metadata with a q.it-shipped Trust root and threshold root-key rotation.
The first production pack is a managed Apple Silicon Transformers pack for the curated Qwen 3.5 recipe.
Gemma 4 joins that pack only after it meets the same control-plane and real-device qualification contract.
The existing worker-path environment variables remain development and diagnostic overrides after managed packs ship.

The browser, CLI adapters, and integration tests continue to use the HTTP and SSE control-plane seam from ADR-0001.
Runtime-pack installation does not authorize model-package downloads or execution of code from model repositories.
