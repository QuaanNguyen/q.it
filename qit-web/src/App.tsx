import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { api, bytes, metric, providerLabel, taskLabel } from "./api";
import type {
  Benchmark,
  Host,
  Provider,
  ProviderInfo,
  Run,
  RunDetail,
  RunsResponse,
} from "./api";
import {
  comparisonGroups,
  latestModels,
  qualityLabel,
  setupKey,
} from "./analysis";
import {
  ModelChartColors,
  RankingChart,
  ScatterChart,
  TrendChart,
} from "./charts";

type View = "overview" | "runs" | "benchmarks" | "start";
type DashboardData = {
  host: Host;
  benchmarks: Benchmark[];
  providers: ProviderInfo[];
  history: RunsResponse;
};

function currentView(): View {
  const value = location.hash.replace("#/", "").split("?")[0];
  return value === "runs" || value === "benchmarks" || value === "start"
    ? value
    : "overview";
}

function hashValue(name: string): string | null {
  return new URLSearchParams(location.hash.split("?")[1]).get(name);
}

const titles: Record<View, string> = {
  overview: "Model performance",
  runs: "Run history",
  benchmarks: "Benchmarks",
  start: "Start a benchmark",
};

export default function App() {
  const [view, setView] = useState<View>(currentView);
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const [host, benchmarks, providers, history] = await Promise.all([
        api.host(),
        api.benchmarks(),
        api.providers(),
        api.runs(),
      ]);
      setData({ host, benchmarks, providers, history });
      setUpdatedAt(new Date());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRefreshing(false);
    }
  }, []);
  useEffect(() => {
    void load();
    const onHash = () => setView(currentView());
    window.addEventListener("hashchange", onHash);
    if (!location.hash) location.hash = "#/overview";
    return () => window.removeEventListener("hashchange", onHash);
  }, [load]);
  useEffect(() => {
    const timer = window.setInterval(
      () => void load(),
      data?.history.running ? 2000 : 15000,
    );
    return () => window.clearInterval(timer);
  }, [load, data?.history.running]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#/overview" aria-label="q.it overview">
          q.it
        </a>
        <p className="brand-caption">EDGE MODEL BENCHMARKS</p>
        <div className="nav-caption">[ workspace ]</div>
        <nav aria-label="Main navigation">
          {(["overview", "runs", "benchmarks", "start"] as View[]).map(
            (item, index) => (
              <a
                key={item}
                href={`#/${item}`}
                className={item === view ? "active" : ""}
                aria-current={item === view ? "page" : undefined}
              >
                <span className="nav-index">0{index + 1}</span>
                {item === "start"
                  ? "Start benchmark"
                  : item === "overview"
                    ? "Overview"
                    : titles[item]}
                <span className="nav-arrow">{item === view ? "<" : ""}</span>
              </a>
            ),
          )}
        </nav>
        <div className="sidebar-foot">
          <span className="connection">
            [ {error ? "offline" : data ? "connected" : "connecting"} ]
          </span>
          <strong>{data?.host.host_name ?? "Local workspace"}</strong>
          <span>
            {data?.host.raspberry_pi
              ? "Raspberry Pi"
              : (data?.host.architecture ?? "")}
          </span>
          <a
            href="https://www.microsoft.com/en-us/download/details.aspx?id=108856"
            target="_blank"
            rel="noreferrer"
          >
            Michelangelus font ↗
          </a>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <p className="eyebrow">
              q.it / {view === "start" ? "execution" : "workspace"}
            </p>
            <h1>{titles[view]}</h1>
          </div>
          <div className="topbar-actions">
            <span className="updated">
              {updatedAt
                ? `SYNC ${updatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`
                : "CONNECTING"}
            </span>
            <button
              onClick={() => void load()}
              disabled={refreshing}
              type="button"
            >
              {refreshing ? "[ syncing ]" : "[ refresh ]"}
            </button>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}
        {!data && !error && (
          <div className="loading">[ loading benchmark workspace... ]</div>
        )}
        {data && view === "overview" && <Overview data={data} />}
        {data && view === "runs" && <RunHistory runs={data.history.runs} />}
        {data && view === "benchmarks" && (
          <Benchmarks benchmarks={data.benchmarks} />
        )}
        {data && view === "start" && (
          <StartBenchmark data={data} onStarted={load} />
        )}
        <footer className="page-footer">
          <span>q.it / local benchmark workspace</span>
          <span>PERFORMANCE + QUALITY</span>
        </footer>
      </main>
    </div>
  );
}

function Overview({ data }: { data: DashboardData }) {
  const groups = useMemo(
    () => comparisonGroups(data.history.runs),
    [data.history.runs],
  );
  const [selectedBenchmark, setSelectedBenchmark] = useState("");
  const [selectedSetup, setSelectedSetup] = useState("");
  const group =
    groups.find((item) => item.key === selectedBenchmark) ?? groups[0];
  const setups = useMemo(
    () =>
      [...new Set((group?.runs ?? []).map(setupKey))]
        .map((key) => ({
          key,
          runs: group!.runs.filter((run) => setupKey(run) === key),
        }))
        .sort(
          (left, right) =>
            latestModels(right.runs).length - latestModels(left.runs).length,
        ),
    [group],
  );
  const setup = setups.find((item) => item.key === selectedSetup) ?? setups[0];
  const runs = setup?.runs ?? [];
  const models = latestModels(runs);
  const reference = models[0];
  const benchmark = data.benchmarks.find(
    (item) => item.id === reference?.benchmark_id,
  );
  const quality = reference
    ? qualityLabel(reference.task)
    : "Benchmark quality";
  return (
    <>
      <div className="page-intro">
        <p>Compare models on the same benchmark and machine.</p>
        <a className="text-link" href="#/start">
          [ + new benchmark ]
        </a>
      </div>
      {groups.length > 0 ? (
        <>
          <section
            className="comparison-controls"
            aria-label="Comparison filters"
          >
            <Field label="Benchmark / pack version">
              <select
                value={group.key}
                onChange={(event) => {
                  setSelectedBenchmark(event.target.value);
                  setSelectedSetup("");
                }}
              >
                {groups.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.runs[0].benchmark_name} / v{item.runs[0].pack_version}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Machine / run setup">
              <select
                value={setup.key}
                onChange={(event) => setSelectedSetup(event.target.value)}
              >
                {setups.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.runs[0].host_name} / {item.runs[0].host.architecture}{" "}
                    / {item.runs[0].iterations} passes / {item.runs[0].warmups}{" "}
                    warmups / {item.runs[0].max_output_tokens} tokens
                  </option>
                ))}
              </select>
            </Field>
          </section>
          <div className="comparison-meta">
            <span>
              <b>{models.length}</b> model configurations
            </span>
            <span>
              <b>{runs.length}</b> successful runs
            </span>
            <span>
              <b>
                {models.reduce(
                  (sum, run) => sum + run.summary.successful_sample_count,
                  0,
                )}
              </b>{" "}
              ranked samples
            </span>
            <span className="meta-note">
              Latest successful run per model configuration
            </span>
          </div>
          <ModelChartColors runs={data.history.runs}>
            <section
              className="chart-grid"
              aria-label="Model comparison charts"
            >
              <RankingChart
                title="Latency"
                number="01"
                detail="Total request time · lower is better"
                runs={models}
                value={(run) => run.summary.latency_ms_p50}
                tail={(run) => run.summary.latency_ms_p95}
                format={time}
                unit="ms"
              />
              <RankingChart
                title="Time to first token"
                number="02"
                detail="First streamed output · lower is better"
                runs={models}
                value={(run) => run.summary.ttft_ms_p50}
                tail={(run) => run.summary.ttft_ms_p95}
                format={time}
                unit="ms"
                empty="TTFT is measured for streamed generation, when available."
              />
              <RankingChart
                title="Token throughput"
                number="03"
                detail="Median decode rate · higher is better"
                runs={models}
                value={(run) => run.summary.tokens_per_second_p50}
                format={(value) => `${value.toFixed(1)} tok/s`}
                unit="tok/s"
                higher
                empty="This task has no exact completion-token throughput."
              />
              <RankingChart
                title="Benchmark quality"
                number="04"
                detail={`${quality} · higher is better`}
                runs={models}
                value={(run) => run.summary.quality_score_mean}
                format={(value) => `${(value * 100).toFixed(1)}%`}
                unit="score"
                higher
                max={1}
                empty="This benchmark has no scored quality expectation."
              />
              <RankingChart
                title="Provider memory"
                number="05"
                detail="Peak process RSS · lower is better"
                runs={models}
                value={(run) => run.summary.process_rss_bytes_peak}
                format={(value) => bytes(value)}
                unit="RSS"
                empty="Supply the model server's process ID when starting a run."
              />
              <RankingChart
                title="Host CPU use"
                number="06"
                detail="Mean whole-machine utilization · lower is lighter"
                runs={models}
                value={(run) => run.summary.host_cpu_percent_mean}
                format={(value) => `${value.toFixed(1)}%`}
                unit="CPU"
                max={100}
                empty="CPU telemetry is available when benchmarking on Linux."
              />
              <ScatterChart
                title="Quality / latency"
                number="07"
                detail="Upper left favors quality and speed"
                runs={models}
                xValue={(run) => run.summary.latency_ms_p50}
                yValue={(run) => run.summary.quality_score_mean}
                xFormat={time}
                yFormat={(value) => `${(value * 100).toFixed(0)}%`}
                xLabel="Median latency (ms)"
                yLabel={quality}
                higherY
              />
              <ScatterChart
                title="Memory / latency"
                number="08"
                detail="Lower left favors speed and smaller memory use"
                runs={models}
                xValue={(run) => run.summary.latency_ms_p50}
                yValue={(run) => run.summary.process_rss_bytes_peak}
                xFormat={time}
                yFormat={(value) => bytes(value)}
                xLabel="Median latency (ms)"
                yLabel="Provider RSS"
              />
              <TrendChart runs={runs} />
              <RankingChart
                title="Host memory"
                number="10"
                detail="Peak whole-machine RAM · includes other processes"
                runs={models}
                value={(run) => run.summary.host_memory_used_bytes_peak}
                format={(value) => bytes(value)}
                unit="RAM"
                empty="Host memory telemetry is available on Linux."
              />
            </section>
          </ModelChartColors>
          <details className="methodology">
            <summary>[ how to read these charts ]</summary>
            <p>
              Rankings use each model configuration's latest successful run with
              the same benchmark version, machine snapshot, measured passes,
              warmups, and output limit. Different serving endpoints remain
              separate configurations. Failed runs stay in history.
            </p>
            <p>
              Bars show p50; the dashed extension marks p95 where available.
              Ties share a rank. Quality measures this benchmark's scoring rule.
              Memory and CPU describe the measured machine; a remote endpoint's
              resource use requires running q.it on that machine.
            </p>
            <p>
              {benchmark?.description ??
                "Historical benchmark data remains available even after removing its pack."}
            </p>
          </details>
        </>
      ) : (
        <section className="empty-state">
          <p className="eyebrow">[ no successful benchmark runs ]</p>
          <h2>Your model comparisons start here.</h2>
          <p>
            Choose an installed benchmark and a model server to collect
            performance and quality measurements.
          </p>
          <a className="primary-button" href="#/start">
            [ start a benchmark → ]
          </a>
        </section>
      )}
    </>
  );
}

export function time(value: number): string {
  return value >= 1000
    ? `${(value / 1000).toFixed(2)} s`
    : `${value.toFixed(1)} ms`;
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

function RunHistory({ runs }: { runs: Run[] }) {
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(() =>
    hashValue("run"),
  );
  useEffect(() => {
    const update = () => setSelected(hashValue("run"));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  const visible = runs.filter(
    (run) =>
      (status === "all" || run.status === status) &&
      `${run.model} ${run.benchmark_id} ${run.provider} ${run.host_name}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="page-intro">
        <p>Every attempt, including failed and running benchmarks.</p>
        <span className="count-label">[ {runs.length} runs ]</span>
      </div>
      <section className="panel history-panel">
        <div className="history-tools">
          <div className="segmented" aria-label="Run status filter">
            {["all", "succeeded", "failed", "running"].map((value) => (
              <button
                key={value}
                aria-pressed={status === value}
                className={status === value ? "active" : ""}
                onClick={() => setStatus(value)}
              >
                {value}
              </button>
            ))}
          </div>
          <input
            aria-label="Search runs"
            placeholder="Search model, benchmark, or host"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <RunTable runs={visible} />
      </section>
      {selected && (
        <div className="detail-section">
          <div className="detail-heading">
            <h2>Run details</h2>
            <a href="#/runs">[ close ]</a>
          </div>
          <RunProgress id={selected} />
        </div>
      )}
    </>
  );
}

function RunTable({ runs }: { runs: Run[] }) {
  if (!runs.length)
    return <div className="table-empty">[ no matching runs ]</div>;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Outcome</th>
            <th>Model</th>
            <th>Benchmark</th>
            <th>Latency p50</th>
            <th>TTFT p50</th>
            <th>Throughput</th>
            <th>Quality</th>
            <th>Host / started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <a
                  href={`#/runs?run=${encodeURIComponent(run.id)}`}
                  className={`run-status ${run.status}`}
                >
                  [ {run.status} ]
                </a>
              </td>
              <td className="model-cell">
                {run.model}
                <small>{providerLabel(run.provider)}</small>
              </td>
              <td>
                {run.benchmark_name}
                <small>
                  v{run.pack_version} · {taskLabel(run.task)}
                </small>
              </td>
              <td>
                {run.summary.latency_ms_p50 === null
                  ? "-"
                  : time(run.summary.latency_ms_p50)}
              </td>
              <td>
                {run.summary.ttft_ms_p50 === null
                  ? "-"
                  : time(run.summary.ttft_ms_p50)}
              </td>
              <td>{metric(run.summary.tokens_per_second_p50, " tok/s")}</td>
              <td>
                {run.summary.quality_score_mean === null
                  ? "-"
                  : `${(run.summary.quality_score_mean * 100).toFixed(1)}%`}
              </td>
              <td>
                {run.host_name}
                <small>
                  {new Date(run.started_at_ms).toLocaleString([], {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </small>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunProgress({ id, expected }: { id: string; expected?: number }) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let stopped = false;
    let timer: number;
    setDetail(null);
    const load = async () => {
      try {
        const result = await api.detail(id);
        if (stopped) return;
        setDetail(result);
        setError(null);
        if (result.run.status === "running")
          timer = window.setTimeout(() => void load(), 1000);
      } catch (reason) {
        if (!stopped) {
          setError(reason instanceof Error ? reason.message : String(reason));
          timer = window.setTimeout(() => void load(), 3000);
        }
      }
    };
    void load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [id]);
  if (!detail)
    return (
      <div className="panel progress-panel">
        {error ?? "[ loading run... ]"}
      </div>
    );
  const { run, samples } = detail;
  const progress = expected
    ? Math.min(100, (samples.length / expected) * 100)
    : run.status === "running"
      ? null
      : 100;
  return (
    <section className="panel progress-panel" aria-label="Benchmark progress">
      <div className="detail-heading">
        <div>
          <p className="eyebrow">[ benchmark run ]</p>
          <h2>{run.model}</h2>
          <p>
            {run.benchmark_name} / {run.host_name}
          </p>
        </div>
        <span className={`run-status ${run.status}`} role="status">
          [ {run.status} ]
        </span>
      </div>
      {error && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}
      {run.error && (
        <p className="error-banner" role="alert">
          {run.error}
        </p>
      )}
      <div className="progress-label">
        <span>
          {samples.length}
          {expected ? ` / ${expected}` : ""} measured samples saved
        </span>
        <span>
          {run.status === "running" && !samples.length
            ? "Connecting / warming up"
            : `${samples.filter((sample) => sample.succeeded).length} completed successfully`}
        </span>
      </div>
      <progress
        aria-label="Measured sample progress"
        value={progress ?? undefined}
        max={100}
      />
      <div className="run-config">
        <span>{providerLabel(run.provider)}</span>
        <span>{run.iterations} passes</span>
        <span>{run.warmups} warmups</span>
        <span>{run.max_output_tokens} token limit</span>
        <span>{bytes(run.summary.process_rss_bytes_peak)} process RSS</span>
      </div>
      <details>
        <summary>[ samples + outputs ]</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Pass / case</th>
                <th>Outcome</th>
                <th>Latency</th>
                <th>Output / error</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((sample) => (
                <tr key={sample.id}>
                  <td>
                    {sample.iteration} / {sample.case_id}
                  </td>
                  <td>{sample.succeeded ? "succeeded" : "failed"}</td>
                  <td>
                    {sample.latency_ms === null ? "-" : time(sample.latency_ms)}
                  </td>
                  <td className="sample-output">
                    {sample.error ?? sample.output_excerpt ?? "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      {run.status === "succeeded" && (
        <a className="text-link" href="#/overview">
          [ compare results → ]
        </a>
      )}
    </section>
  );
}

function Benchmarks({ benchmarks }: { benchmarks: Benchmark[] }) {
  const [task, setTask] = useState("all");
  const tasks = [...new Set(benchmarks.map((benchmark) => benchmark.task))];
  const visible = benchmarks.filter(
    (benchmark) => task === "all" || benchmark.task === task,
  );
  return (
    <>
      <div className="page-intro">
        <p>
          {benchmarks.length} installed benchmarks. Fixed cases, repeatable
          measurements.
        </p>
        <Field label="Task">
          <select
            value={task}
            onChange={(event) => setTask(event.target.value)}
          >
            <option value="all">All tasks</option>
            {tasks.map((value) => (
              <option key={value} value={value}>
                {taskLabel(value)}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <section className="benchmark-grid">
        {visible.map((benchmark, index) => (
          <article className="panel benchmark-card" key={benchmark.id}>
            <div className="benchmark-head">
              <span className="eyebrow">
                [ {String(index + 1).padStart(2, "0")} /{" "}
                {taskLabel(benchmark.task)} ]
              </span>
              <small>
                {benchmark.built_in ? "BUILT IN" : benchmark.pack_name}
              </small>
            </div>
            <h2>{benchmark.name}</h2>
            <p>{benchmark.description}</p>
            <div className="tags">
              {benchmark.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <dl>
              <div>
                <dt>Cases</dt>
                <dd>{benchmark.case_count}</dd>
              </div>
              <div>
                <dt>Pack version</dt>
                <dd>{benchmark.pack_version}</dd>
              </div>
            </dl>
            <div className="provider-row">
              {benchmark.supported_providers.length ? (
                benchmark.supported_providers.map((provider) => (
                  <span key={provider}>{providerLabel(provider)}</span>
                ))
              ) : (
                <span>Provider support pending</span>
              )}
            </div>
            {benchmark.supported_providers.length > 0 && (
              <a
                className="text-link"
                href={`#/start?benchmark=${encodeURIComponent(benchmark.id)}`}
              >
                [ run this benchmark → ]
              </a>
            )}
          </article>
        ))}
      </section>
    </>
  );
}

function StartBenchmark({
  data,
  onStarted,
}: {
  data: DashboardData;
  onStarted: () => Promise<void>;
}) {
  const initial =
    data.benchmarks.find(
      (benchmark) => benchmark.id === hashValue("benchmark"),
    ) ??
    data.benchmarks.find(
      (benchmark) => benchmark.id === "core/text-generation-smoke",
    ) ??
    data.benchmarks[0];
  const [benchmarkId, setBenchmarkId] = useState(initial?.id ?? "");
  const [provider, setProvider] = useState<Provider>(
    initial?.supported_providers[0] ?? "ollama",
  );
  const [model, setModel] = useState("");
  const [endpoint, setEndpoint] = useState(
    data.providers.find((item) => item.id === provider)?.default_base_url ?? "",
  );
  const [apiKey, setApiKey] = useState("");
  const [iterations, setIterations] = useState(3);
  const [warmups, setWarmups] = useState(1);
  const [outputLimit, setOutputLimit] = useState("");
  const [timeout, setTimeout] = useState(120);
  const [pid, setPid] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState<Run | null>(null);
  const benchmark = data.benchmarks.find((item) => item.id === benchmarkId);
  const active = data.history.runs.find((run) => run.status === "running");
  const supported = data.providers.filter((item) =>
    benchmark?.supported_providers.includes(item.id),
  );
  const knownModels = [
    ...new Set(
      data.history.runs
        .filter((run) => run.provider === provider)
        .map((run) => run.model),
    ),
  ];
  const changeProvider = (value: Provider) => {
    setProvider(value);
    setEndpoint(
      data.providers.find((item) => item.id === value)?.default_base_url ?? "",
    );
    setApiKey("");
  };
  const changeBenchmark = (value: string) => {
    setBenchmarkId(value);
    setOutputLimit("");
    const next = data.benchmarks.find((item) => item.id === value);
    if (
      next?.supported_providers.length &&
      !next.supported_providers.includes(provider)
    )
      changeProvider(next.supported_providers[0]);
  };
  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (!benchmark || busy || active) return;
    setBusy(true);
    setError(null);
    try {
      const run = await api.start({
        benchmark_id: benchmark.id,
        provider,
        model: model.trim(),
        base_url: endpoint.trim(),
        api_key: apiKey || null,
        iterations,
        warmups,
        max_output_tokens: outputLimit ? Number(outputLimit) : null,
        timeout_seconds: timeout,
        target_pid: pid ? Number(pid) : null,
      });
      setStarted(run);
      setApiKey("");
      await onStarted();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };
  const monitored = active ?? started;
  const monitoredBenchmark = data.benchmarks.find(
    (item) => item.id === monitored?.benchmark_id,
  );
  return (
    <>
      <div className="page-intro">
        <p>
          Run an installed benchmark against a model that is already being
          served.
        </p>
        <span className="count-label">[ on {data.host.host_name} ]</span>
      </div>
      <div className="execution-grid">
        <form
          className="panel run-form"
          onSubmit={(event) => void start(event)}
        >
          <div className="form-heading">
            <span className="eyebrow">[ 01 / configure ]</span>
            <h2>Benchmark setup</h2>
          </div>
          <Field label="Benchmark">
            <select
              value={benchmarkId}
              onChange={(event) => changeBenchmark(event.target.value)}
              required
            >
              {data.benchmarks.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                  disabled={!item.supported_providers.length}
                >
                  {item.name} / {taskLabel(item.task)}
                </option>
              ))}
            </select>
          </Field>
          {benchmark && (
            <p className="field-description">{benchmark.description}</p>
          )}
          <div className="form-row">
            <Field label="Provider">
              <select
                value={provider}
                onChange={(event) =>
                  changeProvider(event.target.value as Provider)
                }
                disabled={!supported.length}
              >
                {supported.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field
              label="Model"
              hint="Exact name reported by the model server."
            >
              <input
                value={model}
                onChange={(event) => setModel(event.target.value)}
                list="known-models"
                placeholder="e.g. gemma3:1b"
                required
                autoComplete="off"
              />
            </Field>
          </div>
          <datalist id="known-models">
            {knownModels.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <Field label="Server endpoint">
            <input
              value={endpoint}
              onChange={(event) => setEndpoint(event.target.value)}
              type="url"
              required
            />
          </Field>
          <div className="form-row">
            <Field label="Measured passes">
              <input
                type="number"
                min={1}
                max={10000}
                value={iterations}
                onChange={(event) => setIterations(Number(event.target.value))}
                required
              />
            </Field>
            <Field label="Warmup requests">
              <input
                type="number"
                min={0}
                max={10000}
                value={warmups}
                onChange={(event) => setWarmups(Number(event.target.value))}
                required
              />
            </Field>
          </div>
          <details className="advanced-settings">
            <summary>[ advanced settings ]</summary>
            <div className="form-row">
              <Field
                label="Output token limit"
                hint={`Benchmark default: ${benchmark?.default_max_output_tokens ?? "-"}`}
              >
                <input
                  type="number"
                  min={1}
                  max={4294967295}
                  value={outputLimit}
                  onChange={(event) => setOutputLimit(event.target.value)}
                  placeholder="Use benchmark default"
                />
              </Field>
              <Field label="Request timeout (seconds)">
                <input
                  type="number"
                  min={1}
                  max={86400}
                  value={timeout}
                  onChange={(event) => setTimeout(Number(event.target.value))}
                  required
                />
              </Field>
            </div>
            <Field
              label="Model server process ID"
              hint="Optional. Collects process RSS when the server runs on this machine."
            >
              <input
                type="number"
                min={1}
                max={2147483647}
                value={pid}
                onChange={(event) => setPid(event.target.value)}
                placeholder="Optional process ID"
              />
            </Field>
            <Field
              label="API key"
              hint="Sent to the selected server for this run. Never saved to run history."
            >
              <input
                type="password"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                autoComplete="off"
              />
            </Field>
          </details>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <div className="form-submit">
            <span>
              {(benchmark?.case_count ?? 0) * iterations} measured samples
            </span>
            <button
              type="submit"
              className="primary-button"
              disabled={busy || !!active || !supported.length}
            >
              {busy
                ? "[ starting... ]"
                : active
                  ? "[ benchmark running ]"
                  : "[ start benchmark → ]"}
            </button>
          </div>
        </form>
        <aside className="execution-guide">
          <div className="panel guide-panel">
            <span className="eyebrow">[ 02 / measure ]</span>
            <h2>A repeatable run.</h2>
            <ol>
              <li>Start your model server and make the model available.</li>
              <li>
                Choose the benchmark, exact model name, and serving endpoint.
              </li>
              <li>
                Start the run. q.it checks support, warms up, and records each
                measured case.
              </li>
            </ol>
            <div className="guide-summary">
              <span>{taskLabel(benchmark?.task ?? "text_generation")}</span>
              <span>
                {benchmark?.case_count ?? 0} cases / v
                {benchmark?.pack_version ?? "-"}
              </span>
              <span>
                {iterations} passes + {warmups} warmups
              </span>
            </div>
            <p>
              Latency and quality come from the selected model server. CPU and
              memory come from {data.host.host_name}.
            </p>
          </div>
          <div className="guide-note">
            [ one browser benchmark at a time ]
            <p>
              Keep the machine idle for comparable performance measurements.
              Successful runs appear in the comparison charts; all outcomes
              remain in history.
            </p>
          </div>
        </aside>
      </div>
      {monitored && (
        <div className="detail-section">
          <RunProgress
            id={monitored.id}
            expected={
              monitoredBenchmark
                ? monitoredBenchmark.case_count * monitored.iterations
                : undefined
            }
          />
        </div>
      )}
    </>
  );
}
