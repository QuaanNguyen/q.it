import { useState } from "react";

export type RowStatus = "idle" | "starting" | "loaded" | "stopping" | "failed";

type Props = {
  status: RowStatus;
  error?: string;
  onStart: () => void;
  onStop: () => void;
  onInspect: () => void;
};

const TITLES: Record<RowStatus, string> = {
  idle: "",
  starting: "Loading…",
  loaded: "Loaded",
  stopping: "Stopping…",
  failed: "Failed to load",
};

function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z" />
    </svg>
  );
}

export function StartControl({ status, error, onStart, onStop, onInspect }: Props) {
  const [hover, setHover] = useState(false);
  const busy = status === "starting" || status === "stopping";
  const showIcon = status !== "idle";
  const iconClass =
    status === "loaded" ? "ok" : status === "failed" ? "fail" : "busy";

  return (
    <>
      {showIcon && (
        <span
          className={`status-icon ${iconClass}`}
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
        >
          {busy && <span className="spinner" />}
          {status === "loaded" && (
            <span className="pop-in" style={{ display: "inline-flex" }}>
              <CheckIcon />
            </span>
          )}
          {status === "failed" && (
            <span className="pop-in" style={{ display: "inline-flex" }}>
              <CrossIcon />
            </span>
          )}
          {hover && (
            <span className="bubble" onMouseDown={(e) => e.preventDefault()}>
              <div className="title">{TITLES[status]}</div>
              {error && <div className="detail">{error}</div>}
              <button onClick={onInspect}>Inspect</button>
            </span>
          )}
        </span>
      )}
      {status === "loaded" || status === "stopping" ? (
        <button onClick={onStop} disabled={busy}>
          Stop
        </button>
      ) : (
        <button onClick={onStart} disabled={busy}>
          Start
        </button>
      )}
    </>
  );
}
