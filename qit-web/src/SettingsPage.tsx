import { useEffect, useState } from "react";
import { api, fmtBytes, type Settings } from "./api";

const SOURCE_TEXT: Record<Settings["os_reserve_source"], string> = {
  env: "QIT_OS_RESERVE_BYTES is set and overrides this value.",
  setting: "Saved setting in use.",
  default: "Default: 25% of unified memory.",
};

export function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api
      .settings()
      .then((s) => {
        setSettings(s);
        setDraft(s.os_reserve_bytes != null ? String(s.os_reserve_bytes) : "");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  async function save(value: number | null) {
    setError(null);
    setSaved(false);
    try {
      const next = await api.updateSettings(value);
      setSettings(next);
      setDraft(next.os_reserve_bytes != null ? String(next.os_reserve_bytes) : "");
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const parsed = draft.trim() === "" ? null : Number(draft);
  const valid = parsed === null || (Number.isInteger(parsed) && parsed >= 0);

  return (
    <>
      <h1>Settings</h1>
      <p className="lede">Planner inputs that persist across restarts.</p>
      {error && <div className="error">{error}</div>}
      {settings && (
        <div className="panel settings-form">
          <label className="field" style={{ display: "grid", gap: "0.3rem" }}>
            <span>OS reserve (bytes)</span>
            <input
              value={draft}
              inputMode="numeric"
              placeholder="default"
              onChange={(e) => {
                setDraft(e.target.value);
                setSaved(false);
              }}
            />
          </label>
          <div className="note">
            Effective: <strong>{fmtBytes(settings.effective_os_reserve_bytes)}</strong>.{" "}
            {SOURCE_TEXT[settings.os_reserve_source]}
          </div>
          <div className="row" style={{ margin: 0 }}>
            <button className="primary" disabled={!valid} onClick={() => void save(parsed)}>
              Save
            </button>
            <button className="quiet" onClick={() => void save(null)}>
              Reset to default
            </button>
            {saved && <span className="muted">Saved. Budget and fit updated.</span>}
          </div>
        </div>
      )}
      <p className="note">Hugging Face token sign-in is not in this milestone.</p>
    </>
  );
}
