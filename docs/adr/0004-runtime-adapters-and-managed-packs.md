# Runtime adapters select executable material

Model-package recipes declare runtime compatibility, but they do not resolve executables or construct worker commands.
The runtime registry selects one adapter for a runtime recipe, and that same adapter determines readiness and launches the worker.
An adapter owns serve-profile validation, executable resolution, command translation, health and generation protocol endpoints, and runtime-specific diagnostics.
The supervisor owns session and child-process lifecycle without selecting executable paths.

Managed runtime packs will provide immutable, versioned executable material to runtime adapters.
Installing or updating a managed pack is always an explicit user operation and never occurs as a side effect of Start or Try.
The existing worker-path environment variables remain development and diagnostic overrides.

The browser, CLI adapters, and integration tests continue to use the HTTP and SSE control-plane seam from ADR-0001.
Runtime-pack installation does not authorize model-package downloads or execution of code from model repositories.
