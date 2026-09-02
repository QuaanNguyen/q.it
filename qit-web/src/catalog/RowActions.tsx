import { StartControl } from "./StartControl";
import type { CatalogModel, RowModel } from "./useCatalog";

export function RowActions({ row, model }: { row: RowModel; model: CatalogModel }) {
  const { artifact } = row;
  const trying = model.tryFor === artifact.id;
  return (
    <div className="actions">
      <button className="quiet" onClick={() => void model.pin(artifact.id)}>
        Pin
      </button>
      <button className="quiet" onClick={() => void model.whatIf(artifact.id)}>
        What-if
      </button>
      <StartControl
        status={row.status}
        error={row.error}
        onStart={() => void model.start(artifact.id)}
        onStop={() => void model.stop(artifact.id)}
        onInspect={model.inspect}
      />
      <button
        className="primary"
        onClick={() => (trying ? model.closeTry() : model.openTry(artifact.id))}
        disabled={row.status === "stopping"}
      >
        {trying ? "Close" : "Try"}
      </button>
    </div>
  );
}
