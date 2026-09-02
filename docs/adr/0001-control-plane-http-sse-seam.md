# Control plane at HTTP+SSE

The browser and all integration tests interact with `qit-runtime` only through a resource HTTP API under `/api` and SSE for smoke-test generation. Child worker ports bind loopback and are reached by the runtime, not the UI. This single seam is the test surface: inject hardware snapshots and stub workers rather than testing internal crates directly.
