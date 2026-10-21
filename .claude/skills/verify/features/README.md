# q.it verification feature map

Each file describes one user-facing feature: what it is, how a user reaches it, how the harness drives it, and the observable end state that proves it works.
Start from `SKILL.md` for launch, doctor, evidence, and cleanup.

| Feature | Primary surface | Proof |
|---------|-----------------|-------|
| [Run a benchmark](run-benchmark.md) | `qit run` | exit 0, succeeded samples, retained measurements |
| [Review run history](run-history.md) | `qit results`, TUI screens | the run and its failures appear in history |
| [Analyze in the dashboard](dashboard-analysis.md) | `qit dashboard` and `qit-web` | `/api/runs/<id>` returns retained samples |
| [Export results](export-results.md) | `qit export` | JSONL and CSV carry every retained field |

Features not yet mapped, worth adding on the next pass: benchmark pack install and verify, provider discovery (`qit provider list`), and comparison views (`qit compare`, TUI screen 5).
