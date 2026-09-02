# Live memory pressure and free RAM on macOS

Research for [#12](https://github.com/QuaanNguyen/q.it/issues/12). Feeds the `memory_pressure` and `free_ram_bytes` fields of `HardwareSnapshot`, which the control plane already exposes on `/api/hardware` but `SystemProbe` currently leaves `None`. Per [ADR 0003](../adr/0003-stable-budget-not-free-ram.md) these are display-only and must not feed **fit** or **stable budget**.

## Question

What are the primary macOS APIs for (1) the current memory pressure level and (2) free/available RAM, callable from a plain unentitled arm64 binary? How do they map onto the Normal/Warn/Critical levels Activity Monitor shows, and which is cheapest to poll on each `/api/hardware` request?

## Sources

Primary sources only. XNU, system_cmds and libdispatch are cited at the `main` commit fetched on 2026-09-02.

| Tag | Source |
|-----|--------|
| [XNU] | `apple-oss-distributions/xnu` @ `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea` (latest tag at that time `xnu-12377.121.6`) |
| [SYSCMDS] | `apple-oss-distributions/system_cmds` @ `408bba7453608006b89772db185defbac8fe2fd0` |
| [DISPATCH] | `apple-oss-distributions/libdispatch` @ `2361ffb78a76f7ee488cd052eb0bc5c767118bf9` |
| [SDK] | Command Line Tools `MacOSX.sdk` headers on this machine: `mach/vm_statistics.h`, `mach/host_info.h`, `mach/mach_host.h`, `dispatch/source.h` |
| [MAN] | `man 1 memory_pressure`, `man 1 vm_stat`, `man 3 sysctl`, `man 1 top` on this machine |
| [ADEV] | developer.apple.com: `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`, `dispatch_source_memorypressure_flags_t`, `DISPATCH_MEMORYPRESSURE_WARN`, `DispatchSource.MemoryPressureEvent`, `host_statistics64` |
| [ASUP] | Apple Support, "View memory usage in Activity Monitor on Mac" (`actmntr1004`) |
| [LOCAL] | Observed on this Mac: macOS 26.6.2 (25G83), arm64, 16 GiB, uid 501, no root, no entitlements |

## Findings

### 1. The kernel's pressure state machine

The kernel keeps one global `vm_pressure_level_t memorystatus_vm_pressure_level`, initialised to `kVMPressureNormal` [XNU `bsd/kern/kern_memorystatus_notify.c:151`]. The enum is

```c
typedef enum vm_pressure_level {
	kVMPressureNormal             = 0,
	kVMPressureWarning            = 1,
	kVMPressureUrgent             = 2,
	kVMPressureCritical           = 3,
	kVMPressureForegroundJetsam   = 4,
	kVMPressureBackgroundJetsam   = 5,
} vm_pressure_level_t;
```

[XNU `bsd/sys/event_private.h:328-337`]. It is a private header; it is not in the public SDK.

Transitions happen in `vm_pressure_response()` [XNU `osfmk/vm/vm_pageout.c:4716-4790`]:

- Normal -> Critical if `VM_PRESSURE_WARNING_TO_CRITICAL()`, else Normal -> Warning if `VM_PRESSURE_NORMAL_TO_WARNING()`.
- Warning/Urgent -> Normal if `VM_PRESSURE_WARNING_TO_NORMAL()`, else -> Critical if `VM_PRESSURE_WARNING_TO_CRITICAL()`.
- Critical -> Normal if `VM_PRESSURE_WARNING_TO_NORMAL()`, else -> Warning if `VM_PRESSURE_CRITICAL_TO_WARNING()`.
- If stuck in a non-normal level for `vm_pressure_level_transition_threshold` (default 30 min) it re-broadcasts the same level [XNU `vm_pageout.c:4702-4714, 4771-4786`].

With the compressor active (always on shipping macOS), the predicates are ratios of `AVAILABLE_NON_COMPRESSED_MEMORY` to `AVAILABLE_MEMORY` [XNU `vm_pageout.c:10299-10362`]:

```c
AVAILABLE_NON_COMPRESSED_MEMORY = vm_page_active_count + vm_page_inactive_count + vm_page_free_count + vm_page_speculative_count
AVAILABLE_MEMORY                = AVAILABLE_NON_COMPRESSED_MEMORY + VM_PAGE_COMPRESSOR_COUNT
```

[XNU `osfmk/vm/vm_page.h:1528-1529`]

| Transition | Condition (compressor active) | Source |
|-----------|-------------------------------|--------|
| Normal -> Warning | `AVAILABLE_NON_COMPRESSED_MEMORY < VM_PAGE_COMPRESSOR_COMPACT_THRESHOLD` | `vm_pageout.c:10320` |
| Warning -> Critical | `vm_compressor_low_on_space()` or `AVAILABLE_NON_COMPRESSED_MEMORY < 1.2 * VM_PAGE_COMPRESSOR_SWAP_UNTHROTTLE_THRESHOLD` | `vm_pageout.c:10332` |
| Warning -> Normal | `AVAILABLE_NON_COMPRESSED_MEMORY > 1.2 * VM_PAGE_COMPRESSOR_COMPACT_THRESHOLD` | `vm_pageout.c:10348` |
| Critical -> Warning | `AVAILABLE_NON_COMPRESSED_MEMORY > 1.4 * VM_PAGE_COMPRESSOR_SWAP_UNTHROTTLE_THRESHOLD` | `vm_pageout.c:10360` |

where `VM_PAGE_COMPRESSOR_COMPACT_THRESHOLD = AVAILABLE_MEMORY * 10 / vm_compressor_minorcompact_threshold_divisor` and `VM_PAGE_COMPRESSOR_SWAP_UNTHROTTLE_THRESHOLD = AVAILABLE_MEMORY * 10 / vm_compressor_unthrottle_threshold_divisor` [XNU `osfmk/vm/vm_compressor_xnu.h:444-447`]. The divisors are kernel tunables, so user space cannot reproduce the exact thresholds; it should read the level, not recompute it.

Consequence: pressure is a function of how much of physical memory the compressor occupies relative to uncompressed pageable memory, with hysteresis. It is not a function of `free_count` alone, and "Pages free" from `vm_stat` will look alarmingly small (tens of MB) on a healthy machine [LOCAL: 8489 pages = 132 MiB free at Normal].

### 2. `sysctl kern.memorystatus_vm_pressure_level` (the level; recommended)

Declared as a read-only `SYSCTL_PROC` [XNU `kern_memorystatus_notify.c:1902-1912`]. Two facts matter:

1. **It returns the dispatch-encoded level, not the enum above.** The handler calls `convert_internal_pressure_level_to_dispatch_level()` and copies out that value [XNU `kern_memorystatus_notify.c:1884-1900`]. The conversion is

   | `memorystatus_vm_pressure_level` | Sysctl value | Constant |
   |---|---|---|
   | `kVMPressureNormal` (0) | `1` | `NOTE_MEMORYSTATUS_PRESSURE_NORMAL` (0x01) |
   | `kVMPressureWarning` (1), `kVMPressureUrgent` (2) | `2` | `NOTE_MEMORYSTATUS_PRESSURE_WARN` (0x02) |
   | `kVMPressureCritical` (3) | `4` | `NOTE_MEMORYSTATUS_PRESSURE_CRITICAL` (0x04) |

   [XNU `kern_memorystatus_notify.c:1775-1803`; constants at `bsd/sys/event_private.h:301-303`]. The `NOTE_MEMORYSTATUS_PRESSURE_*` values are numerically identical to `DISPATCH_MEMORYPRESSURE_NORMAL/WARN/CRITICAL = 0x01/0x02/0x04` [SDK `dispatch/source.h:255-257`]. Apple's own `memory_pressure` tool compares the sysctl directly against `DISPATCH_MEMORYPRESSURE_*` [SYSCMDS `memory_pressure/memory_pressure.c:210-213, 625-632`]. A value of `0` or `3` is therefore never returned; treat anything outside {1, 2, 4} as unknown.

2. **On release kernels it is `CTLFLAG_MASKED`**, so it is hidden from `sysctl -a` but still readable by name [XNU `kern_memorystatus_notify.c:1909-1910`]. Confirmed [LOCAL]: `sysctl -a | grep memorystatus` does not list it, yet `sysctl kern.memorystatus_vm_pressure_level` prints `1`, and `sysctlbyname()` from a plain C binary returns `1` with 4-byte length.

3. **No privilege needed on macOS.** The `priv_check_cred(..., PRIV_VM_PRESSURE, 0)` gate is compiled only for `!XNU_TARGET_OS_OSX` [XNU `kern_memorystatus_notify.c:1888-1896`]. Confirmed as uid 501 [LOCAL].

Caveat: the Cursor agent sandbox (a `sandbox-exec` profile) denied the `sysctl` tool with `Operation not permitted`; a normal user process has no such restriction. App Sandboxed binaries may differ; `qit-runtime` is not sandboxed.

Cost: ~2 µs per `sysctlbyname()` call, ~1 µs via a pre-resolved MIB with `sysctl()` [LOCAL, 1000-iteration loop]. `man 3 sysctl` recommends `sysctlnametomib()` + `sysctl()` for repeated queries ("runs in about a third the time") [MAN].

### 3. `sysctl kern.memorystatus_level` (percentage; not a level)

A read-only int [XNU `bsd/kern/kern_memorystatus.c:875`], computed in `vm_pressure_response()` as

```c
memorystatus_level = (available_memory * 100) / total_pages;   // total_pages = atop_64(max_mem) [- secluded]
```

[XNU `vm_pageout.c:4732-4738`], where `available_memory` is `memorystatus_get_available_page_count()`. On macOS without `CONFIG_JETSAM` that count is refreshed with `AVAILABLE_NON_COMPRESSED_MEMORY` [XNU `vm_page.h:1547-1549`, `vm_pageout.c:3849-3851`]. So it is "percent of RAM that is active + inactive + free + speculative", i.e. everything not wired and not compressor-owned. It counts busy active pages as "available", so it is not free RAM. Observed 58-59 % while `vm_stat` showed 132 MiB free and 4.1 GiB in the compressor; a user-space recomputation from `host_statistics64` (active + inactive + free_count) gave 57 %, confirming the formula [LOCAL].

This is the number `memory_pressure` prints as "System-wide memory free percentage" (via the private `memorystatus_get_level` syscall, same variable) [SYSCMDS `memory_pressure.c:128-140, 685-686`; XNU `kern_memorystatus.c:876-889`]. It is not in the SDK's `man 3 sysctl` table; only `vm.swapusage` and `vm.loadavg` are documented under `CTL_VM` [MAN]. Not `CTLFLAG_MASKED`, so it appears in `sysctl -a` [LOCAL]. Also readable unprivileged.

Related sysctls seen in `sysctl -a` [LOCAL]: `kern.memorystatus_purge_on_warning: 2`, `kern.memorystatus_purge_on_urgent: 5`, `kern.memorystatus_purge_on_critical: 8` (purgeable-memory purge aggressiveness per level), `vm.page_free_count`, `vm.page_speculative_count`, `vm.page_pageable_internal_count`, `vm.page_pageable_external_count`, `vm.compressor_bytes_used`, `vm.swapusage`. `vm.memory_pressure` is a pageout-scan counter (observed 1685), not a level.

### 4. `host_statistics64(HOST_VM_INFO64)` (page counts; recommended for free RAM)

Public Mach API, macOS 10.6+ [ADEV `host_statistics64`; SDK `mach/mach_host.h:278-290`]. Call with `mach_host_self()`, flavor `HOST_VM_INFO64 = 4`, a `vm_statistics64_data_t`, and `count = HOST_VM_INFO64_COUNT` [SDK `mach/host_info.h:181-182, 205-217`]. No privilege; unprivileged `vm_stat` and `memory_pressure` both use exactly this call [SYSCMDS `vm_stat/vm_stat.c:262-264`, `memory_pressure.c:145-153`].

Field semantics, from the SDK header and the kernel filler `vm_stats()` [SDK `mach/vm_statistics.h:142-174`; XNU `osfmk/kern/host.c:801-881`]:

| Field | Kernel value | Meaning |
|-------|--------------|---------|
| `free_count` | `vm_page_free_count + vm_page_speculative_count` | Free pages **including** speculative read-ahead pages. Header: "speculative pages are already accounted for in free_count". |
| `speculative_count` | `vm_page_speculative_count` | File data read ahead but never used; reclaimable instantly. |
| `active_count` | `vm_page_active_count` + per-CPU local queues | Recently used pageable pages. |
| `inactive_count` | `vm_page_inactive_count` | Pageable pages on the inactive list; candidates for reclaim/compression. |
| `wire_count` | `vm_page_wire_count + throttled + lopage_free` (macOS) | Cannot be paged out. |
| `purgeable_count` | `vm_page_purgeable_count` | Volatile purgeable memory the kernel may drop on demand (counted inside the queues above). |
| `external_page_count` | `vm_page_pageable_external_count` + local | File-backed (non-swap) pageable pages. |
| `internal_page_count` | `vm_page_pageable_internal_count` + local | Anonymous pageable pages. |
| `compressor_page_count` | `compressor_object->resident_page_count` | Physical pages occupied by compressed data. |
| `total_uncompressed_pages_in_compressor` | `c_segment_pages_compressed` | Logical pages stored in the compressor. |
| `swapped_count` (rev2) | `vm_page_swapped_count` | Compressor-stored pages currently on disk. |

`vm_stat` prints "Pages free" as `free_count - speculative_count`, and "Pages speculative" separately [SYSCMDS `vm_stat.c:132-135`; MAN `vm_stat(1)`]. `memory_pressure` does the same [SYSCMDS `memory_pressure.c:157`]. Page size for arithmetic is `vm_kernel_page_size` / `hw.pagesize` = 16384 on Apple Silicon [LOCAL; MAN `sysctl(3)` `hw.pagesize`].

Rate limit: for non-platform binaries the kernel serves at most a random 2-10 `host_statistics*` requests per flavor per 1-second window; beyond that it returns a cached copy from the last real read [XNU `host.c:575-578, 751-798`]. Polling once per `/api/hardware` request is well inside this. Cost measured ~1 µs per call [LOCAL].

Struct size: the SDK's `HOST_VM_INFO64_COUNT` is 62 ints (rev3 adds MTE tag-storage fields); the kernel fills up to the count it knows and tells you how many it filled. Always pass `HOST_VM_INFO64_COUNT` and only depend on fields up to `swapped_count` [SDK `host_info.h:209-217`; XNU `host.c:855-879`].

### 5. Activity Monitor and `top`

Apple documents the categories but not the formulas [ASUP]:

- "Memory Pressure: Graphically represents how efficiently your memory is serving your processing needs. Memory pressure is determined by the amount of free memory, swap rate, wired memory, and file cached memory."
- "Memory Used" = App Memory + Wired Memory + Compressed.
- "Cached Files: The size of files cached by the system into unused memory to improve performance."
- "Swap Used".

Activity Monitor is closed source, so the field-level mapping below is an inference from the `vm_statistics64` field definitions in section 4, not a cited fact:

| Activity Monitor | Probable `vm_statistics64` expression |
|---|---|
| Wired Memory | `wire_count * page` |
| Compressed | `compressor_page_count * page` |
| App Memory | `(internal_page_count - purgeable_count) * page` |
| Cached Files | `(external_page_count + purgeable_count) * page` |
| Memory Used | Wired + Compressed + App Memory |
| Pressure graph colour | the kernel level in section 2 (green = Normal, yellow = Warn, red = Critical); the "determined by ..." sentence matches the compressor/available-memory predicates in section 1 |

On this machine those expressions gave Wired 2.07 GiB, Compressed 4.12 GiB, App 6.20 GiB, Cached Files 2.87 GiB, which sum to ~15.3 GiB of 16 GiB, consistent with Activity Monitor's layout [LOCAL]. `top` reports `PhysMem` "broken into wired, active, inactive, used, and free components" [MAN `top(1)`].

### 6. The `memory_pressure` CLI

`man memory_pressure` describes it as a tool to "apply real or simulate memory pressure" [MAN]. Read-only behaviour, from source [SYSCMDS `memory_pressure.c:608-690`]:

- With no flags it prints `hw.memsize`, a `host_statistics64` dump (`print_vm_statistics`, suppressed by `-Q`), and `System-wide memory free percentage: N%` where N is `kern.memorystatus_level` (section 3), then `return 0`.
- It never prints the pressure level in read-only mode; the level is only read internally when `-l` is used to decide when to stop allocating (`current_level >= desired_level`, comparing dispatch-encoded 1/2/4) [SYSCMDS `memory_pressure.c:203-217`].
- Exit codes: `0` on the read-only path and on `-l` with an invalid level name; `exit(-1)` (shell sees 255) if any `sysctlbyname`/`memorystatus_get_level` fails [SYSCMDS `memory_pressure.c:104-108, 133-138, 634-637`].
- `-l`, `-p`, `-S` allocate or fake pressure; `-S` writes `kern.memorypressure_manual_trigger`, which needs root [SYSCMDS `memory_pressure.c:692-745`]. Do not use these in the probe.

Observed [LOCAL]: `memory_pressure` and `memory_pressure -Q` both exit 0 and print `System-wide memory free percentage: 58%`.

Verdict: it adds nothing over the two sysctls plus `host_statistics64`, and costs a process spawn (~ms) instead of ~1-2 µs.

### 7. `dispatch_source` memory-pressure events

`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` (macOS 10.9+) takes handle 0 and a mask of `DISPATCH_MEMORYPRESSURE_NORMAL (0x01) | WARN (0x02) | CRITICAL (0x04)` [SDK `dispatch/source.h:135-146, 232-260`; ADEV]. Apple's guidance: "Elevated memory pressure is a system-wide condition that applications registered for this source should react to by changing their future memory use behavior"; `WARN` means "Apps should release memory that they do not need right now" [SDK `source.h:244-252`; ADEV `DISPATCH_MEMORYPRESSURE_WARN`].

Under the hood it is a kevent on `EVFILT_MEMORYSTATUS (-14)` with fflags `NOTE_MEMORYSTATUS_PRESSURE_*` [DISPATCH `src/event/event_kevent.c:2771-2790`; XNU `event_private.h:81, 301-303`]. The kernel delivers the same converted level as the sysctl (`convert_internal_pressure_level_to_dispatch_level`) [XNU `kern_memorystatus_notify.c:1145, 1775-1803`], so the sysctl is a synchronous read of exactly the state the dispatch source pushes. On desktop, `WARN | CRITICAL` registrations fire only for system-level pressure; per-process limit warnings use the separate `NOTE_MEMORYSTATUS_PROC_LIMIT_*` flags [XNU `kern_memorystatus_notify.c:507-522`].

Notes for a Rust daemon:

- It is push, not pull: there is no "current level" getter on the source; you learn the level only when it changes (plus the 30-minute re-broadcast). A daemon would have to seed its state from the sysctl anyway.
- Requires linking libdispatch and running a queue; `qit-runtime` has no dispatch dependency today.
- Useful later if the supervisor should react (e.g. refuse new sessions on Critical), not for `/api/hardware` reads.

### 8. Level mapping summary

| Kernel enum | `kern.memorystatus_vm_pressure_level` | `DISPATCH_MEMORYPRESSURE_*` / `NOTE_MEMORYSTATUS_PRESSURE_*` | Activity Monitor graph | `memory_pressure -l` name |
|---|---|---|---|---|
| `kVMPressureNormal` 0 | 1 | `NORMAL` 0x01 | green | `normal` |
| `kVMPressureWarning` 1, `kVMPressureUrgent` 2 | 2 | `WARN` 0x02 | yellow | `warn` |
| `kVMPressureCritical` 3 | 4 | `CRITICAL` 0x04 | red | `critical` |

`kVMPressureUrgent` is folded into Warn everywhere user space can see [XNU `kern_memorystatus_notify.c:1787-1791`]. The Jetsam levels 4/5 are only broadcast on non-macOS or via the debug sysctl.

### 9. Cost comparison for per-request polling

| Method | Measured cost [LOCAL] | Notes |
|---|---|---|
| `sysctl()` with cached MIB | ~1 µs | Cheapest for the level. |
| `sysctlbyname()` | ~2 µs | Fine; string lookup each call. |
| `host_statistics64(HOST_VM_INFO64)` | ~1 µs | Rate-limited to 2-10 real reads/s for third-party binaries, then served from cache. |
| Spawning `sysctl -n`, `vm_stat`, `memory_pressure -Q` | ~1.5-3 ms each | Current `sysctl_n()` helper does this; 3 orders of magnitude slower, plus PATH and parsing risk. |
| `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` | n/a (event) | No synchronous read; needs libdispatch. |

## Recommendation for qit-runtime's SystemProbe

**Level:** `sysctlbyname("kern.memorystatus_vm_pressure_level")` via `libc::sysctlbyname` (add `libc` as a direct macOS dependency; it is already in `Cargo.lock` at 0.2.189 transitively, and that version exports `sysctlbyname`, `host_statistics64`, `mach_host_self`, `vm_statistics64`, `HOST_VM_INFO64`, `HOST_VM_INFO64_COUNT`). Map `1 -> "normal"`, `2 -> "warn"`, `4 -> "critical"`, anything else or an error -> `None`. Store as the existing `memory_pressure: Option<String>`. If the probe is ever called more than a few times per second, resolve the MIB once with `sysctlnametomib` and call `sysctl`.

**Free RAM:** one `host_statistics64(mach_host_self(), HOST_VM_INFO64, ..., HOST_VM_INFO64_COUNT)` per probe, page size from `hw.pagesize` (or `vm_kernel_page_size`), and

```text
free_ram_bytes = (free_count + inactive_count + purgeable_count) * page_size
```

Rationale: `free_count` already includes speculative pages (do not add `speculative_count` again); inactive pages are what the pageout daemon reclaims or compresses next and are the bulk of "available" on a healthy Mac (4.5 GiB vs 0.08 GiB truly free on this machine); purgeable pages are dropped on demand at every pressure level (`kern.memorystatus_purge_on_*`). This is the closest user-space analogue to what the kernel itself treats as reclaimable without touching the compressor, and it keeps `free_ram_bytes` from reading ~100 MiB on an idle 16 GiB machine. Expose the strict figure too if the UI wants it: `strictly_free_bytes = (free_count - speculative_count) * page_size`, which is `vm_stat`'s "Pages free".

Optional extra fields, all from the same call and cheap to add to `HardwareSnapshot` later: `compressed_bytes = compressor_page_count * page`, `wired_bytes = wire_count * page`, `cached_files_bytes = (external_page_count + purgeable_count) * page`, and `kern.memorystatus_level` as `available_percent`.

**Do not:** shell out to `memory_pressure`, `vm_stat` or `sysctl`; recompute the level from page counts (thresholds depend on kernel tunables); use any of these numbers in `budget_bytes()` (ADR 0003).

**Replace `sysctl_n()`** for `hw.memsize` and `machdep.cpu.brand_string` with `libc::sysctlbyname` at the same time, since the process spawn is the dominant cost of `/api/hardware` today and the same helper will serve all four keys.

### Faking it in tests

`HardwareSnapshot` already carries `memory_pressure: Option<String>` and `free_ram_bytes: Option<u64>`, and `FixedProbe { snapshot }` returns it verbatim through `/api/hardware`. `qit-runtime/tests/control_plane.rs` already builds snapshots via `probe_with_free(Some(99))` and asserts `hw["free_ram_bytes"] == 99` while `budget_bytes` stays fixed (`stable_budget_ignores_free_ram`). To cover the new field, extend that helper (or add a sibling) to set `memory_pressure: Some("warn".into())` and assert `hw["memory_pressure"] == "warn"` and that `budget_bytes`/fit badges are unchanged across `"normal"`, `"warn"`, `"critical"`, and `None`. No real sysctl or Mach call runs in CI; `SystemProbe` is exercised only when the daemon runs on a Mac.

## Open items

- Activity Monitor's exact "Memory Used"/"Cached Files" formulas are inferred (section 5), not documented by Apple. If parity with Activity Monitor matters for the UI, verify side by side on a live machine before labelling a number "Cached Files".
- Whether to also subscribe to `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` for supervisor policy (e.g. block new sessions at Critical) is a separate decision; the sysctl is sufficient for `/api/hardware`.
