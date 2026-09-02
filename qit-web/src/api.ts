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

export type Session = {
  id: string;
  artifact_id: string;
  n_ctx: number;
  n_gpu_layers: number;
  n_parallel: number;
  status: string;
};

export type Capacity = {
  hardware: Hardware;
  pins: Reservation[];
  what_ifs: Reservation[];
  sessions: Session[];
};

async function parse<T>(res: Response): Promise<T> {
  if (res.status === 204) {
    return undefined as T;
  }
  const text = await res.text();
  if (!res.ok) {
    try {
      const j = JSON.parse(text) as { error?: string };
      throw new Error(j.error ?? text);
    } catch (e) {
      if (e instanceof Error && !e.message.startsWith("{")) {
        throw e;
      }
      throw new Error(text);
    }
  }
  return text ? (JSON.parse(text) as T) : (undefined as T);
}

export function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${n} B`;
}

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
  pin: (artifact_id: string, n_ctx: number) =>
    fetch("/api/pins", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ artifact_id, n_ctx }),
    }).then((r) => parse<Reservation>(r)),
  whatIf: (artifact_id: string, n_ctx: number) =>
    fetch("/api/what-ifs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ artifact_id, n_ctx }),
    }).then((r) => parse<Reservation>(r)),
  deletePin: (id: string) => fetch(`/api/pins/${id}`, { method: "DELETE" }),
  deleteWhatIf: (id: string) => fetch(`/api/what-ifs/${id}`, { method: "DELETE" }),
  clearWhatIfs: () => fetch("/api/what-ifs", { method: "DELETE" }),
  start: (artifact_id: string, n_ctx: number) =>
    fetch("/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ artifact_id, n_ctx }),
    }).then((r) => parse<Session>(r)),
  stop: (id: string) =>
    fetch(`/api/sessions/${id}/stop`, { method: "POST" }).then((r) =>
      parse<Session>(r)
    ),
};
