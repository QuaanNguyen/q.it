import { fmtBytes } from "../api";
import { RowActions } from "./RowActions";
import { TryPanel } from "./TryPanel";
import { useCatalog, type CatalogModel, type RowModel } from "./useCatalog";

export function CatalogPage() {
  const model = useCatalog();
  return (
    <>
      <h1>Catalog</h1>
      <p className="lede">One card per artifact. Try opens inside the card.</p>
      {model.error && <div className="error">{model.error}</div>}
      <WorkerWarning model={model} />
      <Toolbar model={model} />
      {model.rows.length === 0 && <Empty model={model} />}
      {model.rows.map((row) => (
        <ArtifactCard key={row.artifact.id} row={row} model={model} />
      ))}
    </>
  );
}

function ArtifactCard({ row, model }: { row: RowModel; model: CatalogModel }) {
  const a = row.artifact;
  return (
    <div className="card">
      <div className="head">
        <span className="name">{a.filename}</span>
        <span className={`pill ${a.fit}`}>{a.fit}</span>
      </div>
      <div className="meta">
        <span>{a.org}</span>
        <span>{fmtBytes(a.bytes)}</span>
        <span>
          est. {fmtBytes(a.estimate_bytes)} at {model.nCtx / 1024}k
        </span>
        <span>{a.confidence}</span>
        <span>
          {a.throughput_tps != null ? `${a.throughput_tps.toFixed(1)} tok/s` : "no run yet"}
        </span>
      </div>
      <RowActions row={row} model={model} />
      {model.tryFor === a.id && (
        <TryPanel
          artifact={a}
          nCtx={model.nCtx}
          status={row.status}
          startError={row.error}
          onClose={model.closeTry}
        />
      )}
    </div>
  );
}

function Toolbar({ model }: { model: CatalogModel }) {
  return (
    <div className="toolbar">
      <label className="field">
        Context
        <select value={model.nCtx} onChange={(e) => model.setNCtx(Number(e.target.value))}>
          {model.presets.map((n) => (
            <option key={n} value={n}>
              {n / 1024}k
            </option>
          ))}
        </select>
      </label>
      <button onClick={() => void model.rescan()}>Scan library</button>
    </div>
  );
}

function Empty({ model }: { model: CatalogModel }) {
  return (
    <p className="muted">
      No GGUF files found under the library root.
      {model.workerPath === null && " No worker binary either."}
    </p>
  );
}

function WorkerWarning({ model }: { model: CatalogModel }) {
  if (model.workerPath !== null) return null;
  return (
    <p className="note">
      No worker binary found. Install llama.cpp (<code>brew install llama.cpp</code>) or set{" "}
      <code>QIT_WORKER_PATH</code>.
    </p>
  );
}
