# Model ranking dashboard design

Research date: 2026-10-09.

This note separates Arena's documented evaluation methods from design recommendations for q.it's measured benchmark results.
Sources are first-party Arena pages, Microsoft documentation, the license packaged with Microsoft's font download, and q.it's current implementation.

## Arena patterns that transfer

Arena's [text leaderboard](https://arena.ai/leaderboard/chat/text) puts models at the center and exposes category filters, rank, score, uncertainty, vote counts, and an update date.
Its [agent comparison](https://arena.ai/leaderboard/agent/pareto) offers ranking and Pareto views, model-level resource figures, and explicit measurement units.
Arena explains that [category rankings and resource tradeoffs](https://arena.ai/blog/agent-categories-and-cost) help users choose models for a particular task, and its frontier highlights options whose measured performance cannot improve without spending more.
That article also exposes percentile choices and requires a comparable unit of work before comparing resource costs.

Recommended q.it adaptation:

| Pattern | q.it application |
|---------|------------------|
| Category scope | Select one benchmark and pack version before ranking models. |
| Comparable environment | Scope comparisons to one host and disclose provider and run settings. |
| Metric ranking | Use separate model rankings for latency, time to first token, token throughput, quality, and memory. |
| Measurement direction | Label latency, time to first token, and memory as lower is better; throughput and the implemented quality scores as higher is better. |
| Evidence counts | Show successful measured samples, benchmark runs, and the latest measurement date. |
| Resource tradeoff | Plot benchmark quality against latency or measured provider memory. |
| Percentile detail | Show median and 95th-percentile latency with clear units. |
| Model identity | Use model names as chart labels and retain provider identity as metadata. |

These are design recommendations rather than claims that q.it shares Arena's evaluation method.
An overview grid can contain independent model ranking charts for latency, time to first token, throughput, quality, provider memory, and host CPU use.
Keeping separate metrics preserves the distinction between model answer quality and the performance of the machine and serving setup.
Retain run history and the benchmark list as their own pages, and expose a manual benchmark launch page through the same trusted runner.
The launch page is a user-requested extension of the read-only dashboard boundary currently documented in [ADR-0005](../adr/0005-benchmark-packs-and-provider-adapters.md).

## Honest limits

Arena's [FAQ](https://arena.ai/faq) says its Bradley-Terry ratings come from pairwise human preference votes, and only votes collected before model identities are revealed contribute to official rankings.
Its [voting workflow](https://arena.ai/how-it-works) compares two anonymous responses before revealing their identities.
q.it has no equivalent vote population, so it should not invent an Elo rating, Arena score, preference win rate, or rank-confidence interval.
The 95th percentile describes a distribution of observed latency and is not a 95% confidence interval.
Do not fabricate error bars from a single run summary or use them to imply statistical significance.

q.it's [scorers](../../qit-runtime/src/runner.rs) currently measure expected text coverage, word accuracy, or reciprocal retrieval rank depending on the task.
The quality chart should identify the benchmark so a high score on a small smoke benchmark is not presented as broad model intelligence.
Do not combine unrelated benchmarks into an overall quality score.
Do not average latency across different hosts, pack versions, output limits, or serving settings without showing those differences.
Failed runs remain visible in history and are excluded from successful metric rankings.
Missing time-to-first-token or throughput values, especially for embeddings, should remain absent rather than become zero.
Provider resident memory and host-wide memory are different measurements and require different labels.
The dashboard has no measured monetary cost or energy use, so Arena's dollar-cost axes cannot be copied directly.

## Michelangelus font

The requested [Michelangelus family](https://learn.microsoft.com/en-us/typography/font-list/michelangelus) is a proportional serif with regular, bold, italic, and bold italic styles.
Microsoft's [download page](https://www.microsoft.com/en-us/download/details.aspx?id=108856) provides version 1.10 and describes local installation on systems that support OpenType fonts.
The official [font archive](https://download.microsoft.com/download/e8912dd7-c7e6-4ca1-a36b-c121d3fb238f/Michelangelus%20Fonts.zip) contains four TrueType fonts and `Michelangelus Fonts EULA.rtf`.
The inspected archive has SHA-256 `eef25b7cf00c24f63a59aea159671968e7490b233c58dc00c2cdcfdba049dc84`.

The packaged license permits installing and using copies on the user's devices, but explicitly grants no distribution or sublicensing rights and prohibits publishing the software for others to copy.
The license does not grant permission to redistribute the fonts inside q.it or serve them as downloadable web fonts.
Microsoft's [font FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq#web) separately allows CSS font-stack references to locally installed fonts and distinguishes those references from copying or converting font files for web hosting.

Use a CSS reference to locally installed Michelangelus with a serif fallback, and direct users to Microsoft's download if it is absent.
The font needs to be installed on the device running the browser, including a laptop used to view a Pi dashboard over SSH.
Keep the font files outside the repository and packaged dashboard assets.
An ASCII-inspired theme can use restrained colors, straight borders, bracketed navigation, text labels, and monospace numeric details while using Michelangelus for the wordmark and prose.
