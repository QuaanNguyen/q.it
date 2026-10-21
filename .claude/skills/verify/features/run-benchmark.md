# Run a benchmark

## Sub-features

- Select a benchmark pack id such as `core/embedding-retrieval`.
- Select a provider protocol; verification uses `local-cpu` over a loopback endpoint.
- Set iterations, warmups, output limit, timeout, and CPU thread target.
- Stream per-case progress and exit non-zero when any case fails.
- Persist every sample, including failures, with host and process telemetry.

## How to get to it (user POV)

```sh
target/debug/qit run core/embedding-retrieval --provider local-cpu \
  --model maxsim-fixture --base-url http://127.0.0.1:<port>/maxsim-fixture \
  --iterations 2 --warmups 1
```

The installed core packs cover text, embedding retrieval, reranking, decisions, OCR, speech, and image generation.

## Driving it with the CLI

`helpers/drive-retrieval.sh` starts the fixture provider, resolves its port, runs the command above under an isolated `QIT_HOME`, and records exit code plus both streams. `run.status` must be `0`, and `run.out` must show all three cases.

For a failure path, use model `maxsim-inconsistent`: the provider returns a retrieval observation whose scores disagree with its vectors. The command must exit `2` and print `native retrieval score differs from MaxSim over the retained vectors`.

## Gotchas

- `qit run` prints its final `Failure: ...` line on stdout, so assert on both streams.
- Warmups go through the same validation as measured samples; keep `--warmups 1` so a broken contract cannot hide until iteration one.
- The provider must report `device: cpu`, a sub-billion `parameter_count`, and `task: embedding` for the Local CPU preflight to accept it.
- Provider output is one JSON object per line; the terminal `result` event must carry either `retrieval` or `embeddings`, never both.
