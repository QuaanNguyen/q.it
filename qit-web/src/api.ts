export type Task =
  | "text_generation"
  | "embedding"
  | "image_to_text"
  | "speech_to_text"
  | "reranking"
  | "text_to_image";

export type Provider = "ollama" | "transformers" | "hf_serve" | "tei";

export type Benchmark = {
  id: string;
  pack_id: string;
  pack_name: string;
  pack_version: string;
  name: string;
  description: string;
  task: Task;
  case_count: number;
  tags: string[];
  default_max_output_tokens: number;
  built_in: boolean;
  supported_providers: Provider[];
};

export type RunSummary = {
  sample_count: number;
  successful_sample_count: number;
  success_rate: number;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  ttft_ms_p50: number | null;
  ttft_ms_p95: number | null;
  tokens_per_second_p50: number | null;
  host_cpu_percent_mean: number | null;
  host_cpu_percent_peak: number | null;
  host_memory_used_bytes_peak: number | null;
  process_rss_bytes_peak: number | null;
  quality_score_mean: number | null;
};

export type Run = {
  id: string;
  benchmark_id: string;
  benchmark_name: string;
  pack_version: string;
  task: Task;
  provider: Provider;
  base_url: string;
  model: string;
  host_name: string;
  host: Host;
  status: "running" | "succeeded" | "failed";
  started_at_ms: number;
  finished_at_ms: number | null;
  iterations: number;
  warmups: number;
  max_output_tokens: number;
  target_pid: number | null;
  error: string | null;
  summary: RunSummary;
};

export type RunsResponse = {
  runs: Run[];
  succeeded: number;
  failed: number;
  running: number;
};

export type AnalysisPoint = {
  run_id: string;
  benchmark_id: string;
  pack_version: string;
  model: string;
  provider: Provider;
  host_name: string;
  latency_ms: number | null;
  ttft_ms: number | null;
  tokens_per_second: number | null;
  memory_bytes: number | null;
  cpu_percent: number | null;
  quality_score: number | null;
  pareto: boolean;
};

export type Host = {
  host_name: string;
  operating_system: string;
  architecture: string;
  cpu: string | null;
  logical_cpu_count: number | null;
  total_memory_bytes: number | null;
  raspberry_pi: boolean;
  kernel: string | null;
  device_model: string | null;
  qit_version: string;
};

export type ProviderInfo = {
  id: Provider;
  name: string;
  default_base_url: string;
  tasks: Task[];
};

export type StartRun = {
  benchmark_id: string;
  provider: Provider;
  model: string;
  base_url: string;
  api_key: string | null;
  iterations: number;
  warmups: number;
  max_output_tokens: number | null;
  timeout_seconds: number;
  target_pid: number | null;
};

export type Sample = {
  id: string;
  case_id: string;
  iteration: number;
  succeeded: boolean;
  error: string | null;
  latency_ms: number | null;
  ttft_ms: number | null;
  tokens_per_second: number | null;
  quality_score: number | null;
  output_excerpt: string | null;
};

export type RunDetail = { run: Run; samples: Sample[] };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    throw new Error(body.error ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  host: () => request<Host>("/api/host"),
  providers: () => request<ProviderInfo[]>("/api/providers"),
  benchmarks: () => request<Benchmark[]>("/api/benchmarks"),
  runs: () => request<RunsResponse>("/api/runs?limit=10000"),
  detail: (id: string) =>
    request<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  start: (body: StartRun) =>
    request<Run>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export function providerLabel(provider: Provider): string {
  if (provider === "transformers") return "Transformers Serve";
  if (provider === "tei") return "HF TEI";
  if (provider === "hf_serve") return "Hugging Face Serve";
  return "Ollama";
}

export function taskLabel(task: Task): string {
  return task
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export function bytes(value: number | null): string {
  if (value === null) return "Not measured";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  return `${(value / 1024 ** 2).toFixed(0)} MiB`;
}

export function metric(value: number | null, suffix: string): string {
  return value === null ? "-" : `${value.toFixed(1)}${suffix}`;
}
