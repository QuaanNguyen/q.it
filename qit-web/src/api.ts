export type Artifact = {
  id: string;
  org: string;
  filename: string;
  bytes: number;
  architecture: string | null;
  context_length: number | null;
  block_count: number | null;
  head_count: number | null;
  confidence: string;
  estimate_bytes: number;
  fit: string;
  throughput_tps: number | null;
  peak_rss_bytes: number | null;
  kind: string;
  generate_supported: boolean;
};

export type Hardware = {
  device_class: string;
  chip: string;
  unified_memory_bytes: number;
  metal_recommended_working_set_bytes: number | null;
  os_reserve_bytes: number;
  budget_bytes: number;
  headroom_bytes: number;
  memory_pressure: string | null;
  free_ram_bytes: number | null;
  loaded_rss_bytes: number;
  worker_path: string | null;
};

export type Reservation = {
  id: string;
  artifact_id: string;
  n_ctx: number;
  n_gpu_layers: number;
  n_parallel: number;
  estimate_bytes: number;
};

export type SessionStatus =
  | "not_loaded"
  | "starting"
  | "loaded"
  | "stopping"
  | "failed";

export type Session = {
  id: string;
  artifact_id: string;
  n_ctx: number;
  n_gpu_layers: number;
  n_parallel: number;
  status: SessionStatus;
  last_error?: string;
  log_path?: string;
};

export type Capacity = {
  hardware: Hardware;
  pins: Reservation[];
  what_ifs: Reservation[];
  sessions: Session[];
};

export type Settings = {
  os_reserve_bytes: number | null;
  os_reserve_source: "env" | "setting" | "default";
  effective_os_reserve_bytes: number;
};

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type GenerateDone = {
  prompt_tokens: number;
  completion_tokens: number;
  n_ctx: number;
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function errorMessage(text: string): string {
  try {
    const j = JSON.parse(text) as { error?: string };
    return j.error ?? text;
  } catch {
    return text;
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (res.status === 204) {
    return undefined as T;
  }
  const text = await res.text();
  if (!res.ok) {
    throw new ApiError(res.status, errorMessage(text));
  }
  return text ? (JSON.parse(text) as T) : (undefined as T);
}

export function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${n} B`;
}

export function fmtTokens(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(k >= 10 ? 0 : 1)}k`;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => fetch("/api/health").then((r) => parse<{ ok: boolean }>(r)),
  hardware: () => fetch("/api/hardware").then((r) => parse<Hardware>(r)),
  scan: () =>
    fetch("/api/scan", { method: "POST" }).then((r) =>
      parse<{ artifacts: Artifact[] }>(r)
    ),
  catalog: (n_ctx: number) =>
    fetch(`/api/catalog?n_ctx=${n_ctx}`).then((r) =>
      parse<{ artifacts: Artifact[] }>(r)
    ),
  capacity: () => fetch("/api/capacity").then((r) => parse<Capacity>(r)),
  sessions: () => fetch("/api/sessions").then((r) => parse<Session[]>(r)),
  settings: () => fetch("/api/settings").then((r) => parse<Settings>(r)),
  updateSettings: (os_reserve_bytes: number | null) =>
    fetch("/api/settings", { ...json({ os_reserve_bytes }), method: "PUT" }).then(
      (r) => parse<Settings>(r)
    ),
  pin: (artifact_id: string, n_ctx: number) =>
    fetch("/api/pins", json({ artifact_id, n_ctx })).then((r) =>
      parse<Reservation>(r)
    ),
  whatIf: (artifact_id: string, n_ctx: number) =>
    fetch("/api/what-ifs", json({ artifact_id, n_ctx })).then((r) =>
      parse<Reservation>(r)
    ),
  deletePin: (id: string) => fetch(`/api/pins/${id}`, { method: "DELETE" }),
  deleteWhatIf: (id: string) => fetch(`/api/what-ifs/${id}`, { method: "DELETE" }),
  clearWhatIfs: () => fetch("/api/what-ifs", { method: "DELETE" }),
  start: (artifact_id: string, n_ctx: number) =>
    fetch("/api/sessions", json({ artifact_id, n_ctx })).then((r) =>
      parse<Session>(r)
    ),
  stop: (id: string) =>
    fetch(`/api/sessions/${id}/stop`, { method: "POST" }).then((r) =>
      parse<Session>(r)
    ),
};

export type GenerateHandlers = {
  onToken: (token: string) => void;
  onDone: (done: GenerateDone) => void;
  onError: (message: string) => void;
};

export async function generate(
  artifact_id: string,
  n_ctx: number,
  messages: ChatMessage[],
  signal: AbortSignal,
  handlers: GenerateHandlers
): Promise<void> {
  const res = await fetch("/api/generate", {
    ...json({ artifact_id, n_ctx, messages }),
    signal,
  });
  if (!res.ok) {
    throw new ApiError(res.status, errorMessage(await res.text()));
  }
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      const event = frame.match(/^event: (.*)$/m)?.[1];
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (event === "token") handlers.onToken(data);
      else if (event === "done") handlers.onDone(JSON.parse(data) as GenerateDone);
      else if (event === "error") handlers.onError(data);
    }
  }
}
