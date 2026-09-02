import { useCallback, useEffect, useState } from "react";
import { api, fmtBytes, type Capacity } from "./api";

export function CapacityPage() {
  const [capacity, setCapacity] = useState<Capacity | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setCapacity(await api.capacity());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  if (error) return <div className="error">{error}</div>;
  if (!capacity) return <p className="muted">Loading…</p>;
  const hw = capacity.hardware;

  return (
    <>
      <h1>Capacity</h1>
      <p className="lede">
        {hw.chip} · {hw.device_class}
      </p>
      <div className="panel">
        <div className="stat-grid">
          <Stat label="Unified memory" value={fmtBytes(hw.unified_memory_bytes)} />
          <Stat label="OS reserve" value={fmtBytes(hw.os_reserve_bytes)} />
          <Stat
            label="Metal cap"
            value={
              hw.metal_recommended_working_set_bytes != null
                ? fmtBytes(hw.metal_recommended_working_set_bytes)
                : "n/a"
            }
          />
          <Stat label="Budget" value={fmtBytes(hw.budget_bytes)} />
          <Stat label="Headroom" value={fmtBytes(hw.headroom_bytes)} />
        </div>
        <p className="note" style={{ marginBottom: 0 }}>
          Right now (not used for fit): free RAM{" "}
          {hw.free_ram_bytes != null ? fmtBytes(hw.free_ram_bytes) : "n/a"} · worker RSS{" "}
          {fmtBytes(hw.loaded_rss_bytes)}
          {hw.memory_pressure ? ` · pressure ${hw.memory_pressure}` : ""}
        </p>
      </div>

      <h2>Sessions</h2>
      {capacity.sessions.length === 0 ? (
        <p className="muted">None</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Artifact</th>
              <th>Context</th>
              <th>Status</th>
              <th>Detail</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {capacity.sessions.map((s) => (
              <tr key={s.id}>
                <td>{s.artifact_id}</td>
                <td>{s.n_ctx / 1024}k</td>
                <td>
                  <span className={`pill ${statusClass(s.status)}`}>{s.status}</span>
                </td>
                <td className="muted" style={{ maxWidth: 360, wordBreak: "break-word" }}>
                  {s.last_error ?? (s.log_path ? <span className="mono">{s.log_path}</span> : "")}
                </td>
                <td>
                  <div className="actions">
                    {(s.status === "loaded" || s.status === "starting") && (
                      <button
                        onClick={async () => {
                          await api.stop(s.id);
                          await refresh();
                        }}
                      >
                        Stop
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Pins</h2>
      <ReservationTable
        rows={capacity.pins}
        onRemove={async (id) => {
          await api.deletePin(id);
          await refresh();
        }}
      />

      <h2>What-if</h2>
      <ReservationTable
        rows={capacity.what_ifs}
        onRemove={async (id) => {
          await api.deleteWhatIf(id);
          await refresh();
        }}
      />
      {capacity.what_ifs.length > 0 && (
        <div className="row">
          <button
            className="quiet"
            onClick={async () => {
              await api.clearWhatIfs();
              await refresh();
            }}
          >
            Clear what-ifs
          </button>
        </div>
      )}
    </>
  );
}

function statusClass(status: string): string {
  if (status === "loaded") return "Fits";
  if (status === "failed") return "No";
  return "Tight";
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
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
          <th>Context</th>
          <th>Estimate</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id}>
            <td>{r.artifact_id}</td>
            <td>{r.n_ctx / 1024}k</td>
            <td>{fmtBytes(r.estimate_bytes)}</td>
            <td>
              <div className="actions">
                <button className="quiet" onClick={() => onRemove(r.id)}>
                  Remove
                </button>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
