import { useEffect, useMemo, useState } from "react";
import {
  AnalysisPoint,
  api,
  Benchmark,
  bytes,
  Host,
  metric,
  providerLabel,
  Run,
  RunsResponse,
  taskLabel,
} from "./api";

type View = "overview" | "runs" | "benchmarks";

type DashboardData = {
  host: Host;
  benchmarks: Benchmark[];
  history: RunsResponse;
  points: AnalysisPoint[];
};

function currentView(): View {
  const value = location.hash.replace("#/", "");
  return value === "runs" || value === "benchmarks" ? value : "overview";
}

export default function App() {
  const [view, setView] = useState<View>(currentView);
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const load = async () => {
    try {
      const [host, benchmarks, history, analysis] = await Promise.all([
        api.host(),
        api.benchmarks(),
        api.runs(),
        api.analysis(),
      ]);
      setData({ host, benchmarks, history, points: analysis.points });
      setUpdatedAt(new Date());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  useEffect(() => {
    void load();
    const onHash = () => setView(currentView());
    window.addEventListener("hashchange", onHash);
    if (!location.hash) location.hash = "#/overview";
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#/overview" aria-label="q.it overview">
          <span className="brand-mark">q</span>
          <span>
            q.it
            <small>edge model lab</small>
          </span>
        </a>
        <nav>
          <NavLink view="overview" active={view} label="Overview" icon="⌁" />
          <NavLink view="runs" active={view} label="Run history" icon="≡" />
          <NavLink view="benchmarks" active={view} label="Benchmarks" icon="◫" />
        </nav>
        <div className="sidebar-foot">
          <span className="status-dot" />
          <span>{data?.host.host_name ?? "Connecting"}</span>
          <small>{data?.host.raspberry_pi ? "Raspberry Pi" : "Local host"}</small>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <p className="eyebrow">Raspberry Pi benchmark workspace</p>
            <h1>{view === "overview" ? "Analysis" : view === "runs" ? "Run history" : "Benchmark packs"}</h1>
          </div>
          <div className="topbar-actions">
            {updatedAt && <span className="updated">Updated {updatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}
            <button className="refresh" onClick={() => void load()} type="button">
              Refresh
            </button>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}
        {!data && !error && <Loading />}
        {data && view === "overview" && <Overview data={data} />}
        {data && view === "runs" && <RunHistory runs={data.history.runs} />}
        {data && view === "benchmarks" && <Benchmarks benchmarks={data.benchmarks} />}
      </main>
    </div>
  );
}

function NavLink({ view, active, label, icon }: { view: View; active: View; label: string; icon: string }) {
  return (
    <a href={`#/${view}`} className={view === active ? "active" : ""}>
      <span className="nav-icon">{icon}</span>
      {label}
    </a>
  );
}

function Loading() {
  return (
    <div className="loading-grid">
      <div />
      <div />
      <div />
      <div className="loading-wide" />
    </div>
  );
}

function Overview({ data }: { data: DashboardData }) {
  const [benchmark, setBenchmark] = useState("all");
  const [provider, setProvider] = useState("all");
  const [model, setModel] = useState("all");
  const models = useMemo(
    () => [...new Set(data.history.runs.map((run) => run.model))].sort(),
    [data.history.runs],
  );
  const points = data.points.filter(
    (point) =>
      (benchmark === "all" || point.benchmark_id === benchmark) &&
      (provider === "all" || point.provider === provider) &&
      (model === "all" || point.model === model),
  );
  const successfulRuns = data.history.runs.filter((run) => run.status === "succeeded");
  const latestText = successfulRuns.find((run) => run.summary.ttft_ms_p50 !== null);
  const latestTps = successfulRuns.find((run) => run.summary.tokens_per_second_p50 !== null);

  return (
    <>
      <section className="host-strip">
        <span className="host-chip">{data.host.architecture}</span>
        <span>{data.host.cpu ?? "CPU details unavailable"}</span>
        <span>{data.host.logical_cpu_count ?? "-"} logical CPUs</span>
        <span>{bytes(data.host.total_memory_bytes)}</span>
      </section>

      <section className="filters" aria-label="Analysis filters">
        <Filter label="Benchmark" value={benchmark} onChange={setBenchmark}>
          <option value="all">All succeeded runs</option>
          {data.benchmarks.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </Filter>
        <Filter label="Provider" value={provider} onChange={setProvider}>
          <option value="all">All providers</option>
          <option value="ollama">Ollama</option>
          <option value="transformers">Transformers Serve</option>
          <option value="tei">HF TEI</option>
        </Filter>
        <Filter label="Model" value={model} onChange={setModel}>
          <option value="all">All models</option>
          {models.map((name) => <option key={name}>{name}</option>)}
        </Filter>
      </section>

      <section className="metric-grid">
        <MetricCard label="Successful runs" value={String(data.history.succeeded)} detail={`${data.history.failed} failed retained`} tone="green" />
        <MetricCard label="Latest median TTFT" value={metric(latestText?.summary.ttft_ms_p50 ?? null, " ms")} detail={latestText?.model ?? "Run a text benchmark"} tone="violet" />
        <MetricCard label="Latest median TPS" value={metric(latestTps?.summary.tokens_per_second_p50 ?? null, "")} detail={latestTps?.model ?? "Exact token count required"} tone="orange" />
        <MetricCard label="Installed benchmarks" value={String(data.benchmarks.length)} detail={`${new Set(data.benchmarks.map((item) => item.task)).size} task families`} tone="blue" />
      </section>

      {data.history.runs.length === 0 ? <EmptyState /> : (
        <section className="chart-grid">
          <div className="panel pareto-panel">
            <PanelTitle title="Latency and memory frontier" detail="Lower left is better. Frontiers compare the same pack version and host." />
            <ParetoChart points={points} />
          </div>
          <div className="panel">
            <PanelTitle title="Decode throughput" detail="Median exact completion tokens per second." />
            <ThroughputBars points={points} />
          </div>
        </section>
      )}

      <section className="panel recent-panel">
        <PanelTitle title="Recent runs" detail="Every attempt is retained; analysis uses succeeded runs only." action={<a href="#/runs">View all</a>} />
        <RunTable runs={data.history.runs.slice(0, 8)} compact />
      </section>
    </>
  );
}

function Filter({ label, value, onChange, children }: { label: string; value: string; onChange: (value: string) => void; children: React.ReactNode }) {
  return (
    <label>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>{children}</select>
    </label>
  );
}

function MetricCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone: string }) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-accent" />
      <p>{label}</p>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function PanelTitle({ title, detail, action }: { title: string; detail: string; action?: React.ReactNode }) {
  return (
    <div className="panel-title">
      <div>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
      {action}
    </div>
  );
}

function ParetoChart({ points }: { points: AnalysisPoint[] }) {
  const usable = points.filter((point) => point.latency_ms !== null && point.memory_bytes !== null);
  if (usable.length === 0) return <ChartEmpty message="Runs need both latency and Pi memory telemetry." />;
  const width = 680;
  const height = 280;
  const pad = { left: 58, right: 24, top: 18, bottom: 44 };
  const maxX = Math.max(...usable.map((point) => point.latency_ms!), 1) * 1.08;
  const minMemory = Math.min(...usable.map((point) => point.memory_bytes!));
  const maxMemory = Math.max(...usable.map((point) => point.memory_bytes!));
  const memoryRange = Math.max(maxMemory - minMemory, maxMemory * 0.1, 1);
  const x = (value: number) => pad.left + (value / maxX) * (width - pad.left - pad.right);
  const y = (value: number) => pad.top + ((maxMemory + memoryRange * 0.08 - value) / (memoryRange * 1.16)) * (height - pad.top - pad.bottom);
  const colors: Record<string, string> = { ollama: "#8b5cf6", transformers: "#f59e0b", tei: "#22c55e" };

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Latency and memory Pareto scatterplot">
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
          <g key={ratio}>
            <line className="grid-line" x1={pad.left} x2={width - pad.right} y1={pad.top + ratio * (height - pad.top - pad.bottom)} y2={pad.top + ratio * (height - pad.top - pad.bottom)} />
          </g>
        ))}
        <line className="axis-line" x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} />
        <line className="axis-line" x1={pad.left} x2={pad.left} y1={pad.top} y2={height - pad.bottom} />
        {usable.map((point) => (
          <g key={point.run_id}>
            {point.pareto && <circle className="pareto-ring" cx={x(point.latency_ms!)} cy={y(point.memory_bytes!)} r="10" />}
            <circle cx={x(point.latency_ms!)} cy={y(point.memory_bytes!)} r="5.5" fill={colors[point.provider]}>
              <title>{`${point.model}: ${point.latency_ms!.toFixed(1)} ms, ${bytes(point.memory_bytes)}`}</title>
            </circle>
          </g>
        ))}
        <text className="axis-label" x={(pad.left + width - pad.right) / 2} y={height - 8}>Median latency (ms)</text>
        <text className="axis-label" transform={`translate(14 ${(pad.top + height - pad.bottom) / 2}) rotate(-90)`}>Peak memory</text>
        <text className="tick-label" x={pad.left} y={height - pad.bottom + 18}>0</text>
        <text className="tick-label" textAnchor="end" x={width - pad.right} y={height - pad.bottom + 18}>{maxX.toFixed(0)}</text>
        <text className="tick-label" x={pad.left - 8} textAnchor="end" y={pad.top + 4}>{bytes(maxMemory)}</text>
        <text className="tick-label" x={pad.left - 8} textAnchor="end" y={height - pad.bottom}>{bytes(minMemory)}</text>
      </svg>
      <div className="legend">
        {Object.entries(colors).map(([key, color]) => <span key={key}><i style={{ background: color }} />{providerLabel(key as AnalysisPoint["provider"])}</span>)}
      </div>
    </div>
  );
}

function ThroughputBars({ points }: { points: AnalysisPoint[] }) {
  const usable = points.filter((point) => point.tokens_per_second !== null).slice(0, 8);
  if (usable.length === 0) return <ChartEmpty message="No exact token throughput is available for this selection." />;
  const max = Math.max(...usable.map((point) => point.tokens_per_second!), 1);
  return (
    <div className="bars">
      {usable.map((point) => (
        <div className="bar-row" key={point.run_id}>
          <span title={point.model}>{point.model}</span>
          <div><i style={{ width: `${(point.tokens_per_second! / max) * 100}%` }} /></div>
          <strong>{point.tokens_per_second!.toFixed(1)}</strong>
        </div>
      ))}
    </div>
  );
}

function ChartEmpty({ message }: { message: string }) {
  return <div className="chart-empty"><span>◇</span><p>{message}</p></div>;
}

function EmptyState() {
  return (
    <section className="empty-state">
      <span className="empty-icon">⌁</span>
      <div>
        <p className="eyebrow">No benchmark data yet</p>
        <h2>Start with the text generation smoke benchmark</h2>
        <p>Run this over SSH on the Pi, then refresh the dashboard.</p>
        <code>qit run core/text-generation-smoke --provider ollama --model gemma3:1b</code>
      </div>
    </section>
  );
}

function RunHistory({ runs }: { runs: Run[] }) {
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const visible = runs.filter((run) =>
    (status === "all" || run.status === status) &&
    `${run.model} ${run.benchmark_id} ${run.provider}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <section className="panel history-panel">
      <div className="history-tools">
        <div className="segmented">
          {(["all", "succeeded", "failed"] as const).map((value) => <button key={value} className={status === value ? "active" : ""} onClick={() => setStatus(value)}>{value}</button>)}
        </div>
        <input aria-label="Search runs" placeholder="Search model or benchmark" value={search} onChange={(event) => setSearch(event.target.value)} />
      </div>
      <RunTable runs={visible} />
    </section>
  );
}

function RunTable({ runs, compact = false }: { runs: Run[]; compact?: boolean }) {
  if (runs.length === 0) return <div className="table-empty">No matching runs.</div>;
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Outcome</th><th>Model</th><th>Benchmark</th><th>Provider</th><th>Median latency</th><th>TTFT</th><th>TPS</th>{!compact && <th>Peak RAM</th>}<th>Started</th></tr></thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id} title={run.error ?? undefined}>
              <td><span className={`run-status ${run.status}`}><i />{run.status}</span></td>
              <td className="model-cell">{run.model}</td>
              <td><span>{run.benchmark_name}</span><small>{taskLabel(run.task)}</small></td>
              <td>{providerLabel(run.provider)}</td>
              <td>{metric(run.summary.latency_ms_p50, " ms")}</td>
              <td>{metric(run.summary.ttft_ms_p50, " ms")}</td>
              <td>{metric(run.summary.tokens_per_second_p50, "")}</td>
              {!compact && <td>{bytes(run.summary.process_rss_bytes_peak ?? run.summary.host_memory_used_bytes_peak)}</td>}
              <td>{new Date(run.started_at_ms).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Benchmarks({ benchmarks }: { benchmarks: Benchmark[] }) {
  const [task, setTask] = useState("all");
  const tasks = [...new Set(benchmarks.map((benchmark) => benchmark.task))];
  const visible = benchmarks.filter((benchmark) => task === "all" || benchmark.task === task);
  return (
    <>
      <section className="bench-intro">
        <div>
          <p className="eyebrow">Declarative and offline after install</p>
          <h2>{benchmarks.length} installed benchmarks</h2>
          <p>Benchmark packs contain only definitions, cases, and media. Provider code stays in the trusted q.it binary.</p>
        </div>
        <label className="task-filter">Task<select value={task} onChange={(event) => setTask(event.target.value)}><option value="all">All tasks</option>{tasks.map((value) => <option key={value} value={value}>{taskLabel(value)}</option>)}</select></label>
      </section>
      <section className="benchmark-grid">
        {visible.map((benchmark) => (
          <article className="benchmark-card" key={benchmark.id}>
            <div className="benchmark-head"><span>{taskGlyph(benchmark.task)}</span><small>{benchmark.built_in ? "Built in" : benchmark.pack_name}</small></div>
            <h3>{benchmark.name}</h3>
            <p>{benchmark.description}</p>
            <div className="tags">{benchmark.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
            <dl><div><dt>Cases</dt><dd>{benchmark.case_count}</dd></div><div><dt>Pack</dt><dd>{benchmark.pack_version}</dd></div></dl>
            <div className="provider-row">{benchmark.supported_providers.length > 0 ? benchmark.supported_providers.map((provider) => <span key={provider}>{providerLabel(provider)}</span>) : <span>Provider adapter pending</span>}</div>
            <code>qit run {benchmark.id} --model &lt;model&gt;</code>
          </article>
        ))}
      </section>
    </>
  );
}

function taskGlyph(task: Benchmark["task"]): string {
  const glyphs: Record<Benchmark["task"], string> = {
    text_generation: "Aa",
    embedding: "⌘",
    image_to_text: "◩",
    speech_to_text: "∿",
    reranking: "⇅",
    text_to_image: "◇",
  };
  return glyphs[task];
}
