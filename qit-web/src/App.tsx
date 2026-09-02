import { useCallback, useEffect, useRef, useState } from "react";
import { api, fmtBytes, type Artifact, type Capacity } from "./api";

const PRESETS = [4096, 8192, 16384, 32768];

function routeFromHash(): "catalog" | "capacity" | "settings" {
  const raw = location.hash.replace("#/", "");
  if (raw === "capacity" || raw === "settings") return raw;
  return "catalog";
}

export default function App() {
  const [route, setRoute] = useState(routeFromHash);
  const [nCtx, setNCtx] = useState(4096);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [capacity, setCapacity] = useState<Capacity | null>(null);
  const [prompt, setPrompt] = useState("Say hello in one word.");
  const [stream, setStream] = useState("");
  const [generating, setGenerating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    if (!location.hash) location.hash = "#/catalog";
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshCatalog = useCallback(async () => {
    await api.scan();
    const cat = await api.catalog(nCtx);
    setArtifacts(cat.artifacts);
  }, [nCtx]);

  const refreshCapacity = useCallback(async () => {
    setCapacity(await api.capacity());
  }, []);

  useEffect(() => {
    setError(null);
    const run = async () => {
      try {
        if (route === "settings") return;
        if (route === "capacity") await refreshCapacity();
        else await refreshCatalog();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void run();
  }, [route, refreshCatalog, refreshCapacity]);

  async function generate(id: string) {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setGenerating(true);
    setStream("");
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ artifact_id: id, n_ctx: nCtx, prompt }),
        signal: ac.signal,
      });
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buf = "";
      let text = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const ev = part.match(/^event: (.*)$/m)?.[1];
          const data = part.match(/^data: (.*)$/m)?.[1] ?? "";
          if (ev === "token") text += data;
          if (ev === "error") text += `\n${data}`;
        }
        setStream(text);
      }
      await refreshCatalog();
    } catch (e) {
      if ((e as { name?: string }).name !== "AbortError") {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setGenerating(false);
      abortRef.current = null;
    }
  }

  return (
    <>
      <aside>
        <strong>q.it</strong>
        <nav>
          <a href="#/catalog" className={route === "catalog" ? "active" : ""}>
            Catalog
          </a>
          <a href="#/capacity" className={route === "capacity" ? "active" : ""}>
            Capacity
          </a>
          <a href="#/settings" className={route === "settings" ? "active" : ""}>
            Settings
          </a>
        </nav>
      </aside>
      <main>
        {error && (
          <>
            <h1>Error</h1>
            <pre>{error}</pre>
          </>
        )}
        {!error && route === "settings" && (
          <>
            <h1>Settings</h1>
            <p className="muted">
              Hugging Face token sign-in is not in this milestone.
            </p>
          </>
        )}
        {!error && route === "capacity" && capacity && (
          <>
            <h1>Capacity</h1>
            <div className="panel">
              <div>
                {capacity.hardware.chip} · {capacity.hardware.device_class}
              </div>
              <div>
                Unified {fmtBytes(capacity.hardware.unified_memory_bytes)} · OS
                reserve {fmtBytes(capacity.hardware.os_reserve_bytes)}
              </div>
              <div>
                Metal cap{" "}
                {capacity.hardware.metal_recommended_working_set_bytes != null
                  ? fmtBytes(capacity.hardware.metal_recommended_working_set_bytes)
                  : "n/a"}
              </div>
              <div>
                Budget {fmtBytes(capacity.hardware.budget_bytes)} · Headroom{" "}
                {fmtBytes(capacity.hardware.headroom_bytes)}
              </div>
              <div className="muted">
                Free RAM (not used for fit){" "}
                {capacity.hardware.free_ram_bytes != null
                  ? fmtBytes(capacity.hardware.free_ram_bytes)
                  : "n/a"}{" "}
                · worker RSS {fmtBytes(capacity.hardware.loaded_rss_bytes)}
              </div>
            </div>
            <h2>Pins</h2>
            <ReservationTable
              rows={capacity.pins}
              onRemove={async (id) => {
                await api.deletePin(id);
                await refreshCapacity();
              }}
            />
            <h2>What-if</h2>
            <ReservationTable
              rows={capacity.what_ifs}
              onRemove={async (id) => {
                await api.deleteWhatIf(id);
                await refreshCapacity();
              }}
            />
            <div className="row">
              <button
                onClick={async () => {
                  await api.clearWhatIfs();
                  await refreshCapacity();
                }}
              >
                Clear what-ifs
              </button>
            </div>
            <h2>Sessions</h2>
            <SessionTable
              rows={capacity.sessions}
              onStop={async (id) => {
                await api.stop(id);
                await refreshCapacity();
              }}
            />
          </>
        )}
        {!error && route === "catalog" && (
          <>
            <h1>Catalog</h1>
            <div className="row">
              <label>
                Context{" "}
                <select
                  value={nCtx}
                  onChange={(e) => setNCtx(Number(e.target.value))}
                >
                  {PRESETS.map((n) => (
                    <option key={n} value={n}>
                      {n / 1024}k
                    </option>
                  ))}
                </select>
              </label>
              <button onClick={() => void refreshCatalog()}>Scan library</button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Artifact</th>
                  <th>Size</th>
                  <th>Fit</th>
                  <th>Confidence</th>
                  <th>tok/s</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {artifacts.map((a) => (
                  <tr key={a.id}>
                    <td>{a.id}</td>
                    <td>{fmtBytes(a.bytes)}</td>
                    <td className={a.fit}>{a.fit}</td>
                    <td>{a.confidence}</td>
                    <td>
                      {a.throughput_tps != null
                        ? a.throughput_tps.toFixed(1)
                        : "—"}
                    </td>
                    <td>
                      <button
                        onClick={async () => {
                          await api.pin(a.id, nCtx);
                          location.hash = "#/capacity";
                        }}
                      >
                        Pin
                      </button>
                      <button
                        onClick={async () => {
                          await api.whatIf(a.id, nCtx);
                          location.hash = "#/capacity";
                        }}
                      >
                        What-if
                      </button>
                      <button
                        onClick={async () => {
                          await api.start(a.id, nCtx);
                          location.hash = "#/capacity";
                        }}
                      >
                        Start
                      </button>
                      <button onClick={() => void generate(a.id)}>Generate</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="panel">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              {generating && (
                <button
                  onClick={() => abortRef.current?.abort()}
                >
                  Cancel
                </button>
              )}
              <div className="stream">{stream}</div>
            </div>
          </>
        )}
      </main>
    </>
  );
}

function ReservationTable({
  rows,
  onRemove,
}: {
  rows: Capacity["pins"];
  onRemove: (id: string) => void;
}) {
  if (!rows.length) return <p className="muted">None</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Artifact</th>
          <th>n_ctx</th>
          <th>Estimate</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td>{r.artifact_id}</td>
            <td>{r.n_ctx}</td>
            <td>{fmtBytes(r.estimate_bytes)}</td>
            <td>
              <button onClick={() => onRemove(r.id)}>Remove</button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SessionTable({
  rows,
  onStop,
}: {
  rows: Capacity["sessions"];
  onStop: (id: string) => void;
}) {
  if (!rows.length) return <p className="muted">None</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Id</th>
          <th>Artifact</th>
          <th>Status</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s.id}>
            <td>{s.id.slice(0, 8)}</td>
            <td>{s.artifact_id}</td>
            <td>{s.status}</td>
            <td>
              <button onClick={() => onStop(s.id)}>Stop</button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
