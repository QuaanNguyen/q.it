# q.it

Local catalog and capacity planner for open-weights GGUF artifacts on Apple Silicon. Users see what fits this machine, reserve capacity, and run inference through supervised worker processes without managing ports or backends manually.

## Language

**Artifact**:
A single installable GGUF file plus metadata (org, filename, bytes, architecture, context length, confidence, **kind**). The runnable unit in the catalog; not a model family name alone.
_Avoid_: model (when meaning a file), package

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
A planner label (Fits / Tight / No) for whether an artifact's estimated bytes fit remaining **headroom** at a given context length. Based on **stable budget**, not currently free RAM.
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
A capacity and runtime row identified by `(artifact_id, n_ctx, n_gpu_layers, n_parallel)`. Drives spawn flags and KV estimates. Two sessions with different `n_ctx` are distinct.
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
