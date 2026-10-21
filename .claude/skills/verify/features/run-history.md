# Review run history

## Sub-features

- `qit results` lists recent runs with status, provider, model, and timestamps.
- `qit results --succeeded` hides failed runs.
- The TUI `Results` screen opens a run and shows per-case detail, including retrieval ranking, decision output, and saved media.
- Failed runs stay in history and are excluded from successful exports.

## How to get to it (user POV)

```sh
target/debug/qit results --limit 5
target/debug/qit tui
```

In the TUI, press `4` for Results, arrows to select a run, Enter to open it.

## Driving it with the CLI

The drive records `results.out` right after the run and requires the run id and `succeeded` to appear. For failure coverage it runs `maxsim-inconsistent` and requires the failed run to appear with the provider error.

The TUI is driven by `scripts/test-terminal-ui.py`, which runs the real binary in a PTY:

```sh
cargo build --locked -p qit
python3 scripts/test-terminal-ui.py
```

## Gotchas

- History reads the same SQLite store as exports; a run that is missing from `qit results` will also be missing from exports.
- Sample order in exports is `iteration, case_id`, so case ids appear alphabetically within one iteration.
- The TUI needs a terminal size of at least 80x24; the PTY helper sets it explicitly.
