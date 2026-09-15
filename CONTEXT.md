# q.it

Offline-capable catalog and capacity planner for local open-weights artifacts and model packages on Apple Silicon.
Users see what fits this machine, reserve capacity, and run inference through supervised worker processes without managing ports or backends manually.

## Language

**Artifact**:
A single scanned GGUF file plus metadata (org, filename, bytes, architecture, context length, confidence, **kind**).
Existing artifact workflows remain runnable while the catalog adds model packages.
_Avoid_: model (when meaning a file)

**Model package**:
A specific servable unit in the catalog, including its family, format, required files, capabilities, memory estimate, and readiness.
A model package can require one or more local files.
_Avoid_: model, artifact, package family

**Package family**:
Editorial metadata that groups related model packages without owning fit or readiness.
_Avoid_: model package, runnable unit

**Readiness**:
A binary model-package state indicating whether q.it can serve it now.
When readiness is No, one **readiness reason** explains the blocker.
_Avoid_: availability, installed

**Readiness reason**:
The single current blocker that keeps a model package from being ready, such as missing required files, insufficient memory, or a missing runtime.
_Avoid_: status message, warning

**Runtime recipe**:
The package-selected worker integration that validates a serve profile and launches a worker.
_Avoid_: backend setting, engine configuration

**Serve profile**:
The context length and runtime-specific settings for one target.
_Avoid_: session tuple, universal runtime knobs

**Target identity**:
The artifact or model package selected for a reservation or session.
_Avoid_: artifact id, package id

**Artifact kind**:
A scan-time label from GGUF headers: instruct, base, embedding, rerank, vision_projector, or unknown. **Try** is offered only for instruct.
_Avoid_: model type, chat-capable

**Backend**:
An inference engine implementation behind a common session interface. Milestone 1 implements GGUF via a child `llama-server` process; MLX and others are future backends.
_Avoid_: engine, provider

**Capacity planner**:
The subsystem that computes stable **budget**, **headroom**, and **fit** badges from hardware probe data, artifact estimates, **pins**, **what-if** reservations, and **loaded** sessions.
_Avoid_: fit checker, memory calculator

**Control plane**:
The `qit-runtime` HTTP API under `/api` plus SSE for generation. The browser and tests talk only here; worker ports stay on loopback behind the runtime.
_Avoid_: API layer (alone), server

**Fit**:
A planner assessment of whether estimated bytes fit remaining **headroom**, based on **stable budget** rather than currently free RAM.
Artifacts display Fits / Tight / No while model packages expose a binary yes or no.
_Avoid_: compatible, runs

**Headroom**:
Bytes remaining in the **stable budget** after pins, what-ifs, and loaded-session estimates are subtracted.
_Avoid_: free memory, available RAM

**Loaded session**:
A **session** whose worker child process is running (Starting, Loaded, Stopping, or Failed). Only Loaded sessions spawn inference traffic.
_Avoid_: running model, active job

**Pin**:
A persisted reservation that consumes planner **headroom** and survives daemon restart. Does not start a worker by itself.
_Avoid_: bookmark, favorite

**Session**:
A capacity and runtime row identified by a target identity and serve profile.
Two sessions with different context lengths or runtime settings are distinct.
_Avoid_: slot, job

**Stable budget**:
Planner memory ceiling: unified memory minus OS reserve, capped by Metal `recommendedMaxWorkingSetSize` when available. Does not use live free RAM.
_Avoid_: available memory, system RAM

**Try**:
A multi-turn trial conversation against one **loaded session** to verify an artifact works. The transcript lives only in browser memory and is gone on reload or close; nothing is stored, no multi-model, no agents. Not a chat product.
_Avoid_: chat, smoke test, demo

**What-if reservation**:
An ephemeral reservation used on the Capacity page to simulate loading an artifact. Cleared on daemon restart; does not persist in SQLite.
_Avoid_: draft pin, preview

**Worker**:
A child inference process (e.g. `llama-server`) started by the supervisor for a **loaded session**. Bundled or overridden via env; never downloaded at runtime in v1.
_Avoid_: server process, llama
