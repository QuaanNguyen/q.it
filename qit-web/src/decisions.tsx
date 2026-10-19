import type { DecisionObservation, Run } from "./api";
import { logLoss, providerLabel } from "./api";
import { decisionRankings } from "./analysis";

export function DecisionRankings({ runs, rankings }: { runs: Run[]; rankings?: { run_id: string; rank: number }[] }) {
  const ranked = rankings ? [
    ...rankings.flatMap((entry) => runs.filter((run) => run.id === entry.run_id).map((run) => ({ run, rank: entry.rank as number | null }))),
    ...runs.filter((run) => !rankings.some((entry) => entry.run_id === run.id)).map((run) => ({ run, rank: null })),
  ] : decisionRankings(runs);
  return (
    <section className="panel decision-rankings" aria-label="Decision accuracy ranking">
      <div className="section-title"><h2>Choice accuracy</h2></div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Rank</th><th>Model</th><th>Accuracy ↑</th><th>Brier ↓</th><th>Log loss ↓</th><th>Zero-probability gold</th></tr></thead>
          <tbody>{ranked.map(({ run, rank }) => (
            <tr key={run.id}>
              <td>{rank ?? "-"}</td>
              <td><a href={`#/runs?run=${encodeURIComponent(run.id)}`}>{run.model}</a><small>{providerLabel(run.provider)}</small></td>
              <td>{run.summary.quality_score_mean === null ? "-" : `${(run.summary.quality_score_mean * 100).toFixed(1)}%`}</td>
              <td>{run.summary.decision_metrics?.brier_score_mean.toFixed(3) ?? "-"}</td>
              <td>{logLoss(run.summary.decision_metrics?.negative_log_likelihood)}</td>
              <td>{run.summary.decision_metrics?.zero_gold_probability_count ?? "-"}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  );
}

export function DecisionOutput({ observation }: { observation: DecisionObservation }) {
  const { result, metrics } = observation;
  return (
    <details className="decision-output">
      <summary>Candidate probabilities</summary>
      <table>
        <thead><tr><th>Candidate</th><th>Probability</th><th>Native logit</th></tr></thead>
        <tbody>{result.candidates.map((candidate) => (
          <tr key={candidate.id}>
            <td>{candidate.id}{candidate.id === result.selected_id ? " ✓" : ""}</td>
            <td>{candidate.probability.toPrecision(6)}</td>
            <td>{candidate.logit ?? "-"}</td>
          </tr>
        ))}</tbody>
      </table>
      <details><summary>Benchmark input</summary><p>{observation.case.state}</p><p>{observation.case.question}</p>{observation.media && <img className="sample-image" src={observation.media} alt="Benchmark input" />}<pre>{JSON.stringify(observation.case.candidates, null, 2)}</pre><p>Expected: {observation.case.expected ?? "unlabelled"}</p></details>
      {metrics && <div className="run-config"><span>Brier {metrics.brier_score.toFixed(3)}</span><span>Log loss {logLoss(metrics.negative_log_likelihood)}</span></div>}
      <details><summary>Score provenance</summary><pre>{JSON.stringify(result.provenance, null, 2)}</pre></details>
    </details>
  );
}
