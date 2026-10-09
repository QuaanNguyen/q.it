import type { Run, Task } from "./api";

export function benchmarkKey(run: Run): string {
  return JSON.stringify([run.benchmark_id, run.pack_version]);
}

export function setupKey(run: Run): string {
  return JSON.stringify([
    run.host_name,
    run.host.operating_system,
    run.host.kernel,
    run.host.architecture,
    run.host.cpu,
    run.host.logical_cpu_count,
    run.host.total_memory_bytes,
    run.host.qit_version,
    run.iterations,
    run.warmups,
    run.max_output_tokens,
  ]);
}

export function modelKey(run: Run): string {
  return JSON.stringify([run.model, run.provider, run.base_url]);
}

export function latestModels(runs: Run[]): Run[] {
  const latest = new Map<string, Run>();
  for (const run of [...runs].sort(
    (left, right) =>
      right.started_at_ms - left.started_at_ms ||
      left.id.localeCompare(right.id),
  )) {
    if (run.status !== "succeeded") continue;
    if (!latest.has(modelKey(run))) latest.set(modelKey(run), run);
  }
  return [...latest.values()].sort((left, right) =>
    left.model.localeCompare(right.model),
  );
}

export function comparisonGroups(runs: Run[]): { key: string; runs: Run[] }[] {
  const groups = new Map<string, Run[]>();
  for (const run of runs) {
    if (run.status !== "succeeded") continue;
    const key = benchmarkKey(run);
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  return [...groups]
    .map(([key, runs]) => ({ key, runs }))
    .sort((left, right) => {
      const count = (runs: Run[]) =>
        Math.max(
          ...[...new Set(runs.map(setupKey))].map(
            (setup) =>
              latestModels(runs.filter((run) => setupKey(run) === setup))
                .length,
          ),
        );
      const scored = (runs: Run[]) =>
        runs.filter((run) => run.summary.quality_score_mean !== null).length;
      return (
        count(right.runs) - count(left.runs) ||
        scored(right.runs) - scored(left.runs) ||
        Math.max(...right.runs.map((run) => run.started_at_ms)) -
          Math.max(...left.runs.map((run) => run.started_at_ms))
      );
    });
}

export function frontier(
  runs: Run[],
  resource: (run: Run) => number | null,
  score: (run: Run) => number | null,
  higherScore: boolean,
): Set<string> {
  const usable = runs.filter(
    (run) => resource(run) !== null && score(run) !== null,
  );
  return new Set(
    usable
      .filter(
        (run) =>
          !usable.some((other) => {
            const resourceBetter = resource(other)! <= resource(run)!;
            const scoreBetter = higherScore
              ? score(other)! >= score(run)!
              : score(other)! <= score(run)!;
            return (
              resourceBetter &&
              scoreBetter &&
              (resource(other)! < resource(run)! ||
                score(other)! !== score(run)!)
            );
          }),
      )
      .map((run) => run.id),
  );
}

export function qualityLabel(task: Task): string {
  if (task === "embedding" || task === "reranking")
    return "Mean reciprocal rank";
  if (task === "speech_to_text") return "Word accuracy";
  return "Expected-text coverage";
}

export const modelColors = [
  "#9fd57b",
  "#e2b869",
  "#86becb",
  "#c4a5de",
  "#d89581",
  "#8dc5b0",
  "#c8ca7c",
  "#94abd6",
];

export function modelPalette(models: string[]): Map<string, string> {
  return new Map(
    [...new Set(models)]
      .sort()
      .map((model, index) => [
        model,
        modelColors[index] ??
          `hsl(${(((index - modelColors.length) * 137.508) % 360).toFixed(3)} 58% 72%)`,
      ]),
  );
}
