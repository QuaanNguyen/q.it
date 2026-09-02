# KV estimate for hybrid Mamba/attention GGUFs (nemotron_h)

Resolves [#14](https://github.com/QuaanNguyen/q.it/issues/14). Sources are llama.cpp `master` at commit [`b81c99b`](https://github.com/ggml-org/llama.cpp/commit/b81c99b479d4c24e5eeca10de99032ebd343ef8f) (2026-09-02); every code citation below is a permalink to that commit. The GGUF header dump was produced by a read-only Python parser over the header/KV section of the local file (no tensor data read, no `llama-server` run).

## Question

For hybrid Mamba2/attention GGUFs such as `nemotron_h`:

1. How does llama.cpp decide which layers are attention (KV cache) vs recurrent (SSM state), and which GGUF keys carry that split?
2. How does llama.cpp size the KV cache for a hybrid model?
3. How does llama.cpp size the recurrent state?
4. What does the local Nemotron 3 Nano 4B Q4_K_M header contain, and what is the corrected estimate for `n_ctx=4096`, `n_parallel=1`?

## Summary

- `nemotron_h.attention.head_count_kv` and `nemotron_h.feed_forward_length` are **per-layer uint32 arrays** of length `block_count`. There is no separate layer-type key. A layer is recurrent iff `head_count_kv[il] == 0 && feed_forward_length[il] == 0`; attention iff `head_count_kv[il] > 0`; MLP-only otherwise.
- Attention KV per layer is `n_ctx_seq * n_stream * head_count_kv[il] * (key_length + value_length) * bytes(type)`, default `f16` (2 bytes). `n_ctx_seq = pad256(n_ctx) / n_seq_max` when not unified, so `n_ctx_seq * n_stream == n_ctx` for the planner's purposes.
- Recurrent state per recurrent layer is `n_seq_max * (1 + n_rs_seq) * (n_embd_r + n_embd_s) * 4` bytes (always `f32`), where `n_embd_r = (ssm.conv_kernel - 1) * (ssm.inner_size + 2 * ssm.group_count * ssm.state_size)` and `n_embd_s = ssm.state_size * ssm.inner_size`. It does not scale with `n_ctx`.
- Nemotron 3 Nano 4B: 42 blocks = 4 attention + 21 Mamba2 + 17 MLP. Corrected estimate at 4k/1 slot: **2.989 GB** (file 2.837 GB + KV 64 MiB + recurrent state 81 MiB), vs planner 4.98 GB and observed RSS 3.14 GB.
- The planner's 4.98 GB is reproduced exactly by the current bug: the Rust parser skips array values, so `head_count_kv` falls back to `head_count = 40`, `head_dim` becomes `3136 / 40 = 78` instead of `key_length = 128`, and all 42 layers are charged.

## Findings

### 1. Layer typing: which layers hold KV vs SSM state

**Architecture classification.** `nemotron_h` is registered as `LLM_ARCH_NEMOTRON_H` ([llama-arch.cpp#L93](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-arch.cpp#L93)). It is **not** in `llm_arch_is_recurrent()` ([llama-arch.cpp#L1048-L1060](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-arch.cpp#L1048-L1060)); it **is** in `llm_arch_is_hybrid()` ([llama-arch.cpp#L1062-L1085](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-arch.cpp#L1062-L1085)), which routes it to `llama_memory_hybrid` (attention KV cache + recurrent state side by side).

**Per-layer arrays.** `llama_model::load_hparams` zero-fills `n_head_arr`, `n_head_kv_arr`, `n_ff_arr` and then reads `feed_forward_length`, `attention.head_count` and `attention.head_count_kv` with `get_key_or_arr`, which accepts either a scalar (broadcast to every layer) or an array of exactly `n_layer_all` elements ([llama-model.cpp#L1269-L1296](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L1269-L1296); loader at [llama-model-loader.cpp#L456-L497](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model-loader.cpp#L456-L497)). `head_count_kv` defaults to a copy of `head_count` when absent ([llama-model.cpp#L1293-L1294](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L1293-L1294)). Per-layer accessors are `n_head_kv(il)`, `n_ff(il)` ([llama-hparams.cpp#L58-L72](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L58-L72)); storage is `std::array<uint32_t, LLAMA_MAX_LAYERS>` ([llama-hparams.h#L91-L93](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.h#L91-L93)).

**The recurrent rule for nemotron_h.** There is no separate layer-type key. `llama_model_nemotron_h::load_arch_hparams` reads the five `ssm.*` keys and then sets, for every layer:

```
is_recr_impl[i] = i < n_layer() && n_head_kv(i) == 0 && n_ff(i) == 0
```

([models/nemotron-h.cpp#L3-L14](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/models/nemotron-h.cpp#L3-L14); accessor `is_recr(il)` at [llama-hparams.cpp#L235-L241](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L235-L241), storage `is_recr_impl` at [llama-hparams.h#L171](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.h#L171)). The tensor loader uses the same split: `is_recr(i)` -> SSM tensors; else `n_ff(i) == 0` -> attention tensors; else FFN tensors ([models/nemotron-h.cpp#L70-L119](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/models/nemotron-h.cpp#L70-L119)).

**Memory filters.** When building memory for `LLM_ARCH_NEMOTRON_H`, the attention cache gets layers where `!is_recr(il) && n_ff(il) == 0` and the recurrent store gets layers where `is_recr(il) && n_ff(il) == 0` ([llama-model.cpp#L2466-L2472](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L2466-L2472)). MLP-only layers (`n_ff > 0`) get neither. So exactly the layers with `head_count_kv[il] > 0` are charged KV.

**Who writes the zeros.** The converter emits `head_count_kv` as a list with the real KV head count on attention layers and `0` elsewhere ([conversion/granite.py#L505-L512](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/conversion/granite.py#L505-L512), inherited by `NemotronHModel`), and `feed_forward_length` as a list with `n_ff` on MLP layers and `0` elsewhere ([conversion/nemotron.py#L335-L339](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/conversion/nemotron.py#L335-L339)). Layer roles come from the HF config's `hybrid_override_pattern` / `layers_block_type` (`M` = Mamba2, `*` = attention, `-` = MLP) ([conversion/nemotron.py#L238-L250](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/conversion/nemotron.py#L238-L250), [#L271-L279](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/conversion/nemotron.py#L271-L279)). Key name constants: `{arch}.attention.head_count_kv`, `{arch}.attention.key_length`, `{arch}.attention.value_length` ([gguf-py constants.py#L182-L187](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/gguf-py/gguf/constants.py#L182-L187)); `{arch}.ssm.conv_kernel`, `inner_size`, `state_size`, `time_step_rank`, `group_count` ([constants.py#L278-L282](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/gguf-py/gguf/constants.py#L278-L282)).

**Spec note.** The GGUF spec documents `[llm].attention.head_count_kv` as a scalar `uint64` and does not mention per-layer arrays or `ssm.group_count` ([ggml docs/gguf.md, "LLM" metadata section](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)). The array form and `group_count` are llama.cpp conventions carried by `gguf-py` and `get_key_or_arr`; any GGUF parser targeting llama.cpp models must accept both scalar and array for `head_count`, `head_count_kv`, and `feed_forward_length`.

### 2. KV cache sizing for attention layers

**Per-layer tensor shapes.** For each layer that passes `has_kv(il)` and the layer filter, `llama_kv_cache` allocates `K = [n_embd_k_gqa(il), kv_size, n_stream]` and `V = [n_embd_v_gqa(il), kv_size, n_stream]` ([llama-kv-cache.cpp#L165-L174](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L165-L174), [#L209-L210](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L209-L210), [#L233-L234](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L233-L234)). Total bytes are the sum of `ggml_nbytes` over those tensors ([llama-kv-cache.cpp#L1894-L1910](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L1894-L1910)). `has_kv(il)` is true for all layers unless `n_layer_kv_from_start` is set (Gemma3n-style), which nemotron_h does not use ([llama-hparams.cpp#L295-L306](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L295-L306)).

**`n_embd_k_gqa` / `n_embd_v_gqa`.** `n_embd_k_gqa(il) = n_embd_head_k(il) * n_head_kv(il)` and `n_embd_v_gqa(il) = n_embd_head_v(il) * n_head_kv(il)` ([llama-hparams.cpp#L131-L141](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L131-L141)). Because `n_head_kv(il) == 0` on recurrent and MLP layers, these are 0 there; but those layers are already excluded by the filter, so they never allocate a tensor.

**`n_embd_head_k` vs `n_embd_head_v`.** Both default to `n_embd / n_head` and are overridden by `attention.key_length` / `attention.value_length` when present ([llama-model.cpp#L1327-L1335](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L1327-L1335); accessors at [llama-hparams.cpp#L115-L129](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L115-L129)). For Nemotron 3 Nano, `n_embd / n_head = 3136 / 40 = 78.4`, so the planner must use `key_length = value_length = 128` and cannot derive head_dim from `embedding_length`. The converter always writes both keys for nemotron_h ([conversion/nemotron.py#L326-L330](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/conversion/nemotron.py#L326-L330)). When flash attention is off and V sizes vary across layers, V is padded to `n_embd_v_gqa_max()` ([llama-kv-cache.cpp#L158-L161](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L158-L161), [#L210](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L210)); for nemotron_h all attention layers share one V size, so this has no effect.

**Cell count and streams (`n_ctx_seq`, `n_seq_max`, `kv_unified`).** The hybrid memory is built with `attn_kv_size = cparams.n_ctx_seq`, `n_seq_max = cparams.n_seq_max`, `unified = cparams.kv_unified` ([llama-model.cpp#L2531-L2548](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L2531-L2548); forwarded to `llama_kv_cache` at [llama-memory-hybrid.cpp#L34-L53](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-memory-hybrid.cpp#L34-L53)). `n_stream = unified ? 1 : n_seq_max` ([llama-kv-cache.cpp#L84](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-kv-cache.cpp#L84)). The context derives `n_ctx = pad256(n_ctx)`; unified: `n_ctx_seq = n_ctx`; otherwise `n_ctx_seq = pad256(n_ctx / n_seq_max)` and `n_ctx` is rounded down to `n_ctx_seq * n_seq_max` ([llama-context.cpp#L288-L303](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-context.cpp#L288-L303)). Either way the cell budget per layer is `n_ctx_seq * n_stream ~= pad256(n_ctx)`; `n_seq_max` only changes how it is partitioned. `llama-server` maps `-np` to `n_seq_max` ([common.cpp#L1723](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/common/common.cpp#L1723)); `kv_unified` defaults off and is enabled by default only when `-np` is left at auto ([arg.cpp#L1721-L1727](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/common/arg.cpp#L1721-L1727), [llama-context.cpp#L3640](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-context.cpp#L3640)). qit-runtime always passes `--parallel N` explicitly, so it is in the non-unified branch.

**Element type.** `type_k = type_v = GGML_TYPE_F16` by default ([llama-context.cpp#L3631-L3632](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-context.cpp#L3631-L3632)), 2 bytes per element. qit-runtime does not pass `-ctk`/`-ctv`.

**Resulting formula (per attention layer, f16):**

```
kv_bytes(il) = n_ctx_seq * n_stream * head_count_kv[il] * (key_length + value_length) * 2
             ~= pad256(n_ctx) * head_count_kv[il] * (key_length + value_length) * 2
```

Summed over layers with `head_count_kv[il] > 0`. The ticket's stated form `n_ctx * n_head_kv * head_dim * 2 (K,V) * bytes(f16)` is confirmed, with `head_dim` split into `key_length` for K and `value_length` for V.

### 3. Recurrent state sizing

**Construction.** For hybrid archs, `llama_memory_recurrent` is created with `type_r = type_s = GGML_TYPE_F32`, `rs_size = max(1, n_seq_max)`, `n_seq_max`, `n_rs_seq` ([llama-model.cpp#L2540-L2544](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-model.cpp#L2540-L2544) via [llama-memory-hybrid.cpp#L54-L65](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-memory-hybrid.cpp#L54-L65)). It has `mem_size = rs_size` cells, i.e. one cell per sequence, not per token ([llama-memory-recurrent.cpp#L20-L39](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-memory-recurrent.cpp#L20-L39)).

**Per-layer tensors.** For each layer passing the recurrent filter:

```
n_rows = mem_size * (1 + n_rs_seq)
r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows)
s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows)
```

([llama-memory-recurrent.cpp#L101-L107](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-memory-recurrent.cpp#L101-L107)). Bytes are summed with `ggml_nbytes` in `size_r_bytes()` / `size_s_bytes()` ([llama-memory-recurrent.cpp#L730-L751](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-memory-recurrent.cpp#L730-L751)).

**`n_embd_r` (conv state) for Mamba/Mamba2:**

```
n_embd_r = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
```

([llama-hparams.cpp#L183-L209](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L183-L209); the RWKV / LFM2 / KDA branches above it do not apply to nemotron_h because `wkv_head_size`, `n_shortconv_l_cache`, `n_embd_head_kda` are all 0 for it).

**`n_embd_s` (SSM state):**

```
n_embd_s = ssm_d_state * ssm_d_inner
```

([llama-hparams.cpp#L211-L233](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.cpp#L211-L233)).

**Key mapping.** `ssm_d_conv <- {arch}.ssm.conv_kernel`, `ssm_d_inner <- ssm.inner_size`, `ssm_d_state <- ssm.state_size`, `ssm_dt_rank <- ssm.time_step_rank`, `ssm_n_group <- ssm.group_count` ([models/nemotron-h.cpp#L4-L8](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/models/nemotron-h.cpp#L4-L8); fields at [llama-hparams.h#L174-L178](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-hparams.h#L174-L178)). `time_step_rank` is used for tensor shapes only, not state size.

**`n_rs_seq`.** Defaults to 0 ([llama-context.cpp#L3611](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-context.cpp#L3611)); `common` sets it from `speculative.need_n_rs_seq()`, which is non-zero only for MTP/EAGLE3/DFlash/DSpark drafting ([common.cpp#L1724](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/common/common.cpp#L1724), [common.h#L394-L400](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/common/common.h#L394-L400)), and is clamped to 0 for archs that do not support rollback ([llama-context.cpp#L104-L108](https://github.com/ggml-org/llama.cpp/blob/b81c99b479d4c24e5eeca10de99032ebd343ef8f/src/llama-context.cpp#L104-L108)). qit-runtime does not enable speculative decoding, so `n_rs_seq = 0`.

**Resulting formula (per recurrent layer, f32):**

```
rs_bytes(il) = n_seq_max * (1 + n_rs_seq) * (n_embd_r + n_embd_s) * 4
```

Summed over layers with `head_count_kv[il] == 0 && feed_forward_length[il] == 0`. Independent of `n_ctx`.

### 4. Nemotron 3 Nano 4B Q4_K_M header dump

File: `/Users/quan/models/gguf/nvidia/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`, 2,837,072,864 bytes. GGUF v3, 263 tensors, 36 KV pairs. Relevant keys (GGUF value type in brackets; 4 = uint32, 8 = string, 9 = array):

```
general.architecture                     [8] = nemotron_h
general.file_type                        [4] = 15   (Q4_K_M)
nemotron_h.block_count                   [4] = 42
nemotron_h.context_length                [4] = 1048576
nemotron_h.embedding_length              [4] = 3136
nemotron_h.attention.head_count          [4] = 40                (scalar)
nemotron_h.attention.head_count_kv       [9] = uint32[42]:
  [0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0]
nemotron_h.feed_forward_length           [9] = uint32[42]:
  [0,12544,0,12544,0,12544,0,0,12544,0,12544,0,0,12544,0,12544,0,0,12544,0,12544,0,12544,
   0,0,12544,0,12544,0,12544,0,0,0,12544,0,0,0,12544,0,12544,0,12544]
nemotron_h.attention.key_length          [4] = 128
nemotron_h.attention.value_length        [4] = 128
nemotron_h.rope.dimension_count          [4] = 78
nemotron_h.ssm.conv_kernel               [4] = 4
nemotron_h.ssm.state_size                [4] = 128
nemotron_h.ssm.group_count               [4] = 8
nemotron_h.ssm.inner_size                [4] = 7680
nemotron_h.ssm.time_step_rank            [4] = 96
nemotron_h.vocab_size                    [4] = 131072
```

Derived layer split (applying the nemotron_h rule from section 1):

| Role | Rule | Count | Layers |
|------|------|-------|--------|
| Attention (KV cache) | `head_count_kv[il] > 0` | 4 | 12, 17, 24, 32 |
| Mamba2 (recurrent state) | `head_count_kv[il] == 0 && n_ff[il] == 0` | 21 | 0, 2, 4, 6, 7, 9, 11, 14, 16, 19, 21, 23, 26, 28, 30, 31, 34, 35, 36, 38, 40 |
| MLP only (no state) | `n_ff[il] > 0` | 17 | remaining |

4 + 21 + 17 = 42 = `block_count`.

### Worked estimate: `n_ctx = 4096`, `n_parallel = 1`

Context params: `n_seq_max = 1`, `kv_unified = false`, `n_ctx = pad256(4096) = 4096`, `n_ctx_seq = 4096`, `n_stream = 1`, `n_rs_seq = 0`, `type_k = type_v = f16`.

Attention KV (4 layers, `head_count_kv = 8`, `key_length = value_length = 128`):

```
per layer = 4096 * 1 * 8 * (128 + 128) * 2 = 16,777,216 B = 16 MiB
total     = 4 * 16 MiB = 67,108,864 B = 64 MiB
```

Recurrent state (21 layers, f32):

```
n_embd_r = (4 - 1) * (7680 + 2 * 8 * 128) = 3 * 9728 = 29,184
n_embd_s = 128 * 7680                       = 983,040
per layer = 1 * (29,184 + 983,040) * 4      = 4,048,896 B ~= 3.86 MiB
total     = 21 * 4,048,896                  = 85,026,816 B ~= 81.1 MiB  (R 2.3 MiB, S 78.8 MiB)
```

Total:

```
file        2,837,072,864
+ KV           67,108,864
+ RS           85,026,816
= estimate  2,989,208,544 B  = 2.989 GB (2.784 GiB)
```

Comparison:

| Quantity | Bytes | GB |
|----------|-------|----|
| Current planner estimate | 4,984,032,224 | 4.98 |
| Corrected estimate (this doc) | 2,989,208,544 | 2.99 |
| Observed loaded-worker RSS | ~3,140,000,000 | 3.14 |

The current planner number is reproduced exactly by `2 * 42 * 40 * 78 * 4096 * 1 * 2 = 2,146,959,360` on top of the file size, which is what `kv_cache_bytes` in `qit-runtime/src/estimate.rs` computes once `head_count_kv` is `None` (the Rust reader discards array values via `skip_array`) and `head_dim = embedding_length / head_count = 3136 / 40 = 78`. Three compounding errors: 42 layers instead of 4, 40 KV heads instead of 8, head_dim 78 instead of 128 (the last one under-counts, the first two over-count).

The ~150 MB gap between the corrected estimate and the observed RSS is consistent with llama.cpp's compute/scheduler buffers for the graph, output logits (`131072` vocab), and Metal runtime overhead, none of which the estimator models. The corrected estimate is a lower bound for RSS; the ticket's observation that the planner over-charges by ~1.85 GB is explained entirely by the KV mis-estimate.

Context scaling with the corrected formula (recurrent state is constant at 81 MiB):

| n_ctx | KV | Total estimate |
|-------|----|----------------|
| 4,096 | 64 MiB | 2.99 GB |
| 8,192 | 128 MiB | 3.06 GB |
| 32,768 | 512 MiB | 3.46 GB |
| 131,072 | 2 GiB | 5.07 GB |

## Recommendation for qit-runtime's estimator and scan

### Keys to parse (in `gguf.rs`)

Accept both scalar and array (GGUF type 9 with element type 4/10) for the per-layer keys; store arrays as `Vec<u32>` and broadcast scalars to `block_count` entries:

| Key | Form | Purpose |
|-----|------|---------|
| `{arch}.block_count` | scalar | layer count `L` |
| `{arch}.attention.head_count` | scalar or `u32[L]` | fallback for `head_count_kv`, head_dim fallback |
| `{arch}.attention.head_count_kv` | scalar or `u32[L]` | per-layer KV heads; `0` marks non-attention layer |
| `{arch}.feed_forward_length` | scalar or `u32[L]` | distinguishes MLP-only layers (`> 0`) from recurrent (`== 0`) |
| `{arch}.attention.key_length` | scalar, optional | K head dim; default `embedding_length / head_count` |
| `{arch}.attention.value_length` | scalar, optional | V head dim; default `embedding_length / head_count` |
| `{arch}.embedding_length` | scalar | only for the head_dim fallback |
| `{arch}.ssm.conv_kernel` | scalar, optional | `d_conv` |
| `{arch}.ssm.inner_size` | scalar, optional | `d_inner` |
| `{arch}.ssm.state_size` | scalar, optional | `d_state` |
| `{arch}.ssm.group_count` | scalar, optional (default 0 for Mamba1) | `n_group` |

`ssm.time_step_rank` is not needed for sizing. Persist these in the artifact row (or a JSON column) so the planner does not re-open files.

### Formula (in `estimate.rs`)

```
n_ctx_eff  = round_up(n_ctx, 256)
head_dim_k = key_length   or embedding_length / head_count
head_dim_v = value_length or embedding_length / head_count

for il in 0..block_count:
    kv_heads = head_count_kv[il]            (array, or scalar broadcast, or head_count[il] if absent)
    n_ff     = feed_forward_length[il]      (array, or scalar broadcast, or 0 if absent)
    if kv_heads > 0:
        kv  += n_ctx_eff * kv_heads * (head_dim_k + head_dim_v) * 2      # f16
    else if n_ff == 0 and ssm.inner_size > 0:
        n_embd_r = (conv_kernel - 1) * (inner_size + 2 * group_count * state_size)
        n_embd_s = state_size * inner_size
        rs  += n_parallel * (n_embd_r + n_embd_s) * 4                     # f32

estimate = file_bytes + kv + rs
```

Notes:

- `n_parallel` does not multiply KV: `n_ctx_seq * n_stream == n_ctx_eff` in both unified and non-unified modes. The current formula's `* n_parallel` factor on KV is wrong for llama-server semantics (the context is split across slots, not replicated). It does multiply recurrent state, which is per-sequence.
- For non-hybrid transformer archs the loop reduces to the existing formula with `head_dim` taken from `key_length`/`value_length` when present; this also fixes any dense model whose `head_dim != embedding_length / head_count`.
- Pure recurrent archs (`mamba`, `mamba2`, `rwkv*`) have no `head_count_kv` and would fall into the `rs` branch for every layer, which matches `llm_arch_is_recurrent` handling.
- The estimate remains a floor for RSS; a fixed or per-arch compute-buffer allowance (~150-300 MB observed here) is a separate concern and should stay separate from the KV/RS terms so the fit label's inputs remain auditable.

### Scan/test implications

- `write_test_gguf` in `gguf.rs` should gain array-valued `head_count_kv`/`feed_forward_length` and the `ssm.*` keys so a `nemotron_h`-shaped fixture can assert the 4-attention / 21-recurrent split and the 2.989 GB figure.
- Existing artifacts scanned with the old reader have `head_count_kv = NULL` for nemotron_h; a rescan (or a schema bump) is required for the corrected numbers to appear.
