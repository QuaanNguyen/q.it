import { fmtBytes, type ModelPackage } from "../api";
import { RowActions } from "./RowActions";
import { StartControl } from "./StartControl";
import { TryPanel } from "./TryPanel";
import { useCatalog, type CatalogModel, type RowModel } from "./useCatalog";

export function CatalogPage() {
  const model = useCatalog();
  return (
    <>
      <h1>Catalog</h1>
      <p className="lede">Owned model packages and local GGUF artifacts.</p>
      {model.error && <div className="error">{model.error}</div>}
      <WorkerWarning model={model} />
      <Toolbar model={model} />
      <h2>Owned packages</h2>
      <PackageRecommendation packages={model.packages} />
      <PackageFamilies packages={model.packages} model={model} />
      <h2>Local GGUF artifacts</h2>
      {model.rows.length === 0 && <Empty model={model} />}
      {model.rows.map((row) => (
        <ArtifactCard key={row.artifact.id} row={row} model={model} />
      ))}
    </>
  );
}

function PackageRecommendation({ packages }: { packages: ModelPackage[] }) {
  const recommended = packages
    .filter((modelPackage) => modelPackage.ready && modelPackage.fits)
    .sort((left, right) => left.estimate_bytes - right.estimate_bytes)[0];
  if (!recommended) return null;
  return (
    <p className="note">
      Recommended for this device: <strong>{recommended.name}</strong> at{" "}
      {fmtBytes(recommended.estimate_bytes)}.
    </p>
  );
}

function PackageFamilies({ packages, model }: { packages: ModelPackage[]; model: CatalogModel }) {
  const families = new Map<string, ModelPackage[]>();
  for (const modelPackage of packages) {
    const alternatives = families.get(modelPackage.family) ?? [];
    alternatives.push(modelPackage);
    families.set(modelPackage.family, alternatives);
  }
  return (
    <>
      {[...families.entries()].map(([family, alternatives]) => (
        <section className="package-family" key={family}>
          <h3>{family}</h3>
          {alternatives.map((modelPackage) => (
            <PackageCard key={modelPackage.id} modelPackage={modelPackage} model={model} />
          ))}
        </section>
      ))}
    </>
  );
}

function PackageCard({
  modelPackage,
  model,
}: {
  modelPackage: ModelPackage;
  model: CatalogModel;
}) {
  const state = model.packageState(modelPackage.id);
  return (
    <div className="card">
      <div className="head">
        <span className="name">{modelPackage.name}</span>
        <span className="package-statuses">
          <span className={`pill ${modelPackage.fits ? "Fits" : "No"}`}>
            {modelPackage.fits ? "Fits" : "Doesn't fit"}
          </span>
          <span className={`pill ${modelPackage.ready ? "Fits" : "No"}`}>
            {modelPackage.ready ? "Ready" : "Not ready"}
          </span>
        </span>
      </div>
      <div className="meta">
        <span>{modelPackage.format.toUpperCase()}</span>
        <span>est. {fmtBytes(modelPackage.estimate_bytes)}</span>
        <span>{modelPackage.capabilities.inputs.join(", ")} in</span>
        <span>{modelPackage.capabilities.outputs.join(", ")} out</span>
        <span>{modelPackage.capabilities.tasks.join(", ")}</span>
      </div>
      {modelPackage.readiness_reason && (
        <p className="readiness-reason">
          {readinessReason(modelPackage.readiness_reason)}
        </p>
      )}
      {modelPackage.ready && (
        <div className="actions">
          <StartControl
            status={state.status}
            error={state.error}
            onStart={() => void model.startPackage(modelPackage.id)}
            onStop={() => void model.stop(modelPackage.id)}
            onInspect={model.inspect}
          />
        </div>
      )}
    </div>
  );
}

function readinessReason(reason: NonNullable<ModelPackage["readiness_reason"]>): string {
  const labels = {
    missing_required_files: "Missing required files",
    insufficient_memory: "Insufficient memory",
    runtime_missing: "Runtime missing",
  };
  return labels[reason];
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
