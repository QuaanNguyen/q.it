import assert from "node:assert/strict";
import test from "node:test";
import {
  benchmarkKey,
  comparisonGroups,
  frontier,
  latestModels,
  modelKey,
  modelPalette,
  qualityLabel,
  setupKey,
} from "../src/analysis.ts";

function run(overrides = {}) {
  return {
    id: "first",
    benchmark_id: "core/text-generation-smoke",
    pack_version: "1.0.0",
    model: "gemma3:1b",
    provider: "ollama",
    base_url: "http://localhost:11434",
    host_name: "pi5",
    host: {
      operating_system: "linux",
      kernel: "6.12",
      architecture: "aarch64",
      cpu: "Cortex-A76",
      logical_cpu_count: 4,
      total_memory_bytes: 8589934592,
      qit_version: "0.1.0",
    },
    status: "succeeded",
    started_at_ms: 1000,
    iterations: 3,
    warmups: 1,
    max_output_tokens: 32,
    summary: { latency_ms_p50: 100, quality_score_mean: 0.9 },
    ...overrides,
  };
}

test("model comparisons select the latest successful run, retaining separate serving configurations", () => {
  const original = run();
  const latest = run({ id: "latest", started_at_ms: 2000 });
  const failed = run({ id: "failed", started_at_ms: 3000, status: "failed" });
  const otherServer = run({
    id: "other-server",
    base_url: "http://localhost:11435",
  });
  const models = latestModels([original, failed, otherServer, latest]);
  assert.deepEqual(
    new Set(models.map((item) => item.id)),
    new Set(["latest", "other-server"]),
  );
  assert.notEqual(modelKey(original), modelKey(otherServer));
});

test("rankings never combine different benchmark versions, machine snapshots, or run setup", () => {
  const original = run();
  assert.notEqual(
    benchmarkKey(original),
    benchmarkKey(run({ pack_version: "2.0.0" })),
  );
  for (const update of [
    { host_name: "sol" },
    { warmups: 0 },
    { iterations: 1 },
    { max_output_tokens: 64 },
    { host: { ...original.host, architecture: "x86_64" } },
    { host: { ...original.host, logical_cpu_count: 8 } },
  ]) {
    assert.notEqual(setupKey(original), setupKey(run(update)));
  }
  const groups = comparisonGroups([
    original,
    run({ id: "llama", model: "llama3.2:1b" }),
    run({ id: "embedding", benchmark_id: "core/embedding-retrieval" }),
    run({ status: "failed", benchmark_id: "failed-only" }),
  ]);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].runs.length, 2);
});

test("Pareto tradeoffs preserve ties, reject dominated models, and skip missing measurements", () => {
  const models = [
    run(),
    run({ id: "tie" }),
    run({
      id: "slow",
      summary: { latency_ms_p50: 200, quality_score_mean: 0.8 },
    }),
    run({
      id: "quality",
      summary: { latency_ms_p50: 150, quality_score_mean: 1 },
    }),
    run({
      id: "unknown",
      summary: { latency_ms_p50: null, quality_score_mean: 1 },
    }),
  ];
  const optimal = frontier(
    models,
    (item) => item.summary.latency_ms_p50,
    (item) => item.summary.quality_score_mean,
    true,
  );
  assert.deepEqual(optimal, new Set(["first", "tie", "quality"]));
  assert.equal(qualityLabel("embedding"), "Mean reciprocal rank");
  assert.equal(qualityLabel("text_generation"), "Expected-text coverage");
});

test("each model has a distinct color shared across all chart labels and marks", () => {
  const models = Array.from({ length: 30 }, (_, index) => `model-${index}`);
  const palette = modelPalette([...models, models[0]]);
  assert.equal(palette.size, 30);
  assert.equal(new Set(palette.values()).size, 30);
  assert.deepEqual(palette, modelPalette([...models].reverse()));
});
