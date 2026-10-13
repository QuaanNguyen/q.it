---
status: accepted
---

# Dashboard benchmark execution and scoped model comparisons

The dashboard can start an installed benchmark against an existing provider endpoint through the same engine used by the CLI, extending the read-only dashboard boundary in ADR-0005 at the user's request.
An accepted run is persisted before returning, executes in a background worker with a separate SQLite connection, and exposes saved sample progress through the existing run-detail interface so chart and history reads remain responsive.
One browser-started run is admitted at a time, browser requests must originate from the dashboard, and provider credentials stay in memory rather than run history.
The dashboard continues to leave model acquisition, model serving, and benchmark-pack authorship outside its scope.

Model comparisons show the latest succeeded run for each model, provider, and endpoint within one benchmark version, machine snapshot, measured-pass count, warmup count, and output limit.
Repeated successful runs remain available in history and trend charts, while failures and unavailable measurements never become ranked zero values.
