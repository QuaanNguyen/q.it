---
name: verify
description: Drive the q.it benchmark runner and dashboard end to end against a loopback Local CPU provider, then prove that measurements, native retrieval observations, exports, and dashboard responses survive the run. Use before claiming a runtime, scoring, storage, export, or dashboard change works.
---

# Verify q.it

## Launch

Build the command once from the repository root:

```sh
cargo build --locked -p qit
```

The binary is `target/debug/qit`. Every drive gets its own data directory through `QIT_HOME`, so no run touches `~/.qit` or another agent's state.

Start the loopback provider the drives use:

```sh
.claude/skills/verify/helpers/fixture_provider.py --port-file /tmp/qit-fixture-port
```

It prints nothing and writes `127.0.0.1:<port>` to the port file when it is ready. The provider path is `http://127.0.0.1:<port>/<model>`; the model name selects behavior (`maxsim-fixture` returns a valid observation, `maxsim-inconsistent` returns one whose scores disagree with its vectors).

For the dashboard, start it with `qit dashboard --port 0` and wait for the line `q.it dashboard listening on <base>` on stdout. Ready means that line, not merely a live process.

Teardown kills only the PIDs the drive recorded. Never kill by process name.

## Doctor

```sh
.claude/skills/verify/helpers/doctor.sh
```

It is read-only. It reports the binary path and whether it is older than the Rust sources, the Python version the provider needs, the checkout revision, and the evidence root. Run it whenever a drive fails before any benchmark starts.

## Drive

One command per mapped feature runs the full journey, asserts the result, and writes evidence:

```sh
.claude/skills/verify/helpers/drive-retrieval.sh
.claude/skills/verify/helpers/drive-ocr.sh
.claude/skills/verify/helpers/drive-speech.sh
```

All three build, start the fixture provider, run the pack with one warmup and two iterations, record `qit results`, export JSONL, read the same run from `qit dashboard`'s `/api/runs/<id>`, then drive an invalid provider answer to prove the CLI refuses it with exit code 2. The OCR and speech drives additionally prove that a failed run stays in history. Individual steps stay separated in the evidence directory so a failure names the step that broke.

To drive a different feature by hand, keep the same shape: isolated `QIT_HOME`, explicit provider base URL, capture exit code plus both streams.

## Evidence

Each drive writes `.qit-evidence/verify/<UTC timestamp>/`:

| File | Proves |
|------|--------|
| `run.out`, `run.err`, `run.status` | the command a user runs succeeded |
| `results.out` | the run appears in history |
| `export.jsonl` | retained samples carry the observation |
| `dashboard.json` | the analysis API returns the same observation |
| `invalid.out`, `invalid.status` | inconsistent provider output is refused |
| `summary.md` | what was asserted, with counts and the run ID |

The retrieval assertions recompute MaxSim from the retained vectors, check that ranking follows score order with document-order ties, and check that `quality_score` equals `1 / (position + 1)` for the case's expected document. They read the expectation from the installed benchmark pack, not from the run output.

The OCR assertions check the expected colors score 1.0, that retained output evidence names the route and stage calls, and that the dashboard serves the exported evidence unchanged.

The speech assertions check that the stored duration equals the duration of the case audio rather than a value the provider invented, that the transcript and routing survive the dashboard, and that a negative duration is refused.

Mocks are limited to the provider endpoint, which is a real process boundary. The command, store, export, and dashboard are the production paths.

Cleanup deletes only the scratch `QIT_HOME` and stops recorded PIDs. Evidence survives cleanup.

## Cleanup

```sh
.claude/skills/verify/helpers/drive-retrieval.sh --cleanup
```

stops any process whose PID is recorded in the newest evidence directory. The drive script also registers an EXIT trap, so a failed attempt never strands a provider or dashboard.

## Helpers

| Helper | Purpose |
|--------|---------|
| `helpers/fixture_provider.py` | loopback Local CPU provider: `maxsim-fixture`, `maxsim-inconsistent`, and `datalab-to/indra-ocr`, with `--mode invalid-metadata` |
| `helpers/doctor.sh` | read-only readiness check |
| `helpers/assert_retrieval.py` | recomputes MaxSim from the evidence and writes `summary.md` |
| `helpers/assert_ocr.py` | checks OCR evidence, history, and dashboard parity |
| `helpers/assert_speech.py` | checks transcript, measured duration, history, and dashboard parity |
| `helpers/drive-retrieval.sh` | full retrieval journey plus cleanup mode |
| `helpers/drive-ocr.sh` | full OCR journey plus cleanup mode |
| `helpers/drive-speech.sh` | full speech journey plus cleanup mode |
