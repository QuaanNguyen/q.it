# QVAC NeMo CPU execution evidence

This evidence supports [Add a CPU ASR provider path for QVAC NeMo checkpoints](https://github.com/QuaanNguyen/q.it/issues/85).
The [provider guide](../qvac-nemo-cpu.md) describes the isolated installation and explicit model acquisition commands.

## Execution boundary

ASU Sol allocation `65805890` completed with exit `0:0` on October 16, 2026.
It requested four CPU cores and 24 GiB of memory on Linux x86_64 node `sc007`, with no GPU allocation.
Torch and Torchaudio were `2.9.1+cpu`, NeMo was `3.0.0`, and all audited learned tensors used CPU float32.
The installed environment matched the complete hashed dependency lock retained in this change.
Model acquisition and provider startup occurred before measured invocations.

The provider was installed from a wheel into a fresh isolated environment.
The q.it command was installed from the packaged Rust crates into an isolated installation prefix, rather than invoked from the source build directory.
That executable passed the installed-command speech regression, the native spoken requests, JSONL/CSV export, dashboard API, and embedded-asset checks.
The speech regression uses a synthetic loopback provider and is separate from the native QVAC results below.

An earlier allocation, `65805289`, exited before model loading because the shell utility `rg` was unavailable.
It is an orchestration failure and supplies no model result.
An initial restored-model audit also exposed the difference between learned parameters and the publisher's state-dictionary element headline.
The corrected audit explicitly reconciles both counts and retains buffers separately.

## Original models and component counts

| Model or configuration | Original publisher revision | Unique learned parameters |
| --- | --- | ---: |
| EU English trunk | `49d2b21e72018798a755c0cbe5ebb4fefe2384a1` | 253,921,542 |
| EU multilingual trunk and six adapters | Same EU revision | 254,417,286 |
| EU automatic trunk, six adapters, and language gate | Same EU revision | 254,912,647 |
| EU-sub shared trunk | `ccf70de9ff9848f5cd5975adda1370ce4addf6f1` | 253,921,542 |

The six adapters contribute 495,744 learned parameters in total, and the optional language gate contributes 495,361.
Each trunk has 253,954,838 state-dictionary elements, including persistent buffers, matching the pinned publisher tensor-export headline.
Independent safetensors header inspection confirmed the adapter and gate counts.
Startup metadata retains every component's parameter and buffer names, shapes, counts, dtypes, devices, and persistence, as well as archive-member and restored-tokenizer digests.

## Spoken cases

Every row below used one warmup and one measured invocation through the freshly installed q.it command.
All six requests succeeded and retained their complete transcript, actual audio duration, and routing evidence.
Token counts, first-token timing, and token throughput remained absent.

| Configuration | Case | Audio seconds | Word accuracy | Invocation milliseconds | Peak provider RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| EU English | English | 5.855 | 1.0000 | 1204.545 | 2815340544 |
| EU multilingual | German | 17.940 | 0.9444 | 1973.216 | 2949042176 |
| EU automatic | English | 5.855 | 1.0000 | 2201.438 | 2861023232 |
| EU automatic | German | 17.940 | 0.9444 | 3863.290 | 2996039680 |
| EU-sub shared | English | 5.855 | 1.0000 | 1243.795 | 2867593216 |
| EU-sub shared | German | 17.940 | 0.9444 | 2034.086 | 2940465152 |

The automatic English request selected the bare trunk with non-English probability `0.018775705248117447`.
The automatic German request selected the adapted route with probability `0.968200147151947`.
Both used the publisher threshold `0.8582`, and the preliminary gate pass was included in invocation latency.
Every adapted invocation recorded one actual execution of each of the six adapter modules.
The German outputs differed from the reference in the spelling of Kowloon.

The English case is `1272-128104-0000` from the clean validation split of [the LibriSpeech test fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a), pinned at `5be91486e11a2d616f4ec5db8d3fd248585ac07a`.
Its parquet SHA-256 is `4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603`, and its extracted audio SHA-256 is `4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41`.
[The original LibriSpeech distribution](https://www.openslr.org/12) identifies the CC BY 4.0 license.

The German case is row `2003` of `de_de/test/0000.parquet` from [the FLEURS parquet conversion](https://huggingface.co/datasets/google/fleurs/tree/168de341b3db6859a9bac1c50a2ef5e3b47647e0/de_de/test), pinned at `168de341b3db6859a9bac1c50a2ef5e3b47647e0`.
The [original dataset card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md), pinned at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, identifies CC BY 4.0.
Its parquet SHA-256 is `3b40c214cabd603f1ddbf3da0499f7780975d6c56b75240f07a8990dbb7a6c07`, and its extracted audio SHA-256 is `1603ed8fcea31a7f32f51d137eaa3516d15aca947898b81a136f4a9732f08ad2`.
Both inert single-case packs retain provenance alongside their reference transcripts on Sol.

## Retained evidence

The experiment root is `/scratch/qnguye19/qit-qvac-nemo-20261016`.
Allocation evidence is under `attempts/65805890`, including `allocation.txt`, `node.txt`, dependency installation/import logs, provider metadata and logs for all four configurations, the six route logs, `route-exits.tsv`, `runs.jsonl`, `runs.csv`, `history.txt`, `validation-summary.json`, and the packaged-install and dashboard markers.
`packaged-install-result`, `dashboard-api-result`, `qualification-result`, and `result` all contain `passed`.
The independent evidence verifier reconciles sidecar headers and loaded counts, CPU tensor placement, durations, absent generation metrics, process RSS, both gate outcomes, and actual adapter execution.
The result database contains seven successful runs because an initial English request preceded the final six-case verification with process telemetry.
It remains retained rather than being deleted from history.

The executed source archive SHA-256 is `533cdfea9ba04fe0a783ab03453e6cd94e020d92bf1faba78d8ceedfa3786319`.
It contains the runtime, provider, generated dashboard assets, and tests based on fresh-install commit `10b674b576e8140b508e31346fba5be5413ea99c` plus this implementation.
Documentation additions after the snapshot do not change the executed implementation.
The installed provider wheel SHA-256 is `ff1f8414d642c4bb985a3938241ac9c82cd7ad365dd53033c390aefc64d2ce29`.
Weights and source audio remain on Sol; only result and audit evidence was copied for local dashboard review.

These are reproducible single-case CPU execution checks, not dataset-wide quality estimates, stable performance rankings, or hardware capacity requirements.
Whole-host memory includes other processes and is not the provider RSS reported above.
Native Raspberry Pi ARM64 support remains unverified, and the current retained NeMo environment explicitly rejects unsupported architectures.
