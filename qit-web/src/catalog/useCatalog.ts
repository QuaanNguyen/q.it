import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Artifact, type Session } from "../api";
import type { RowStatus } from "./StartControl";

const PRESETS = [4096, 8192, 16384, 32768];

export function presetsForArtifacts(artifacts: Artifact[]): number[] {
  const caps = artifacts
    .map((a) => a.context_length)
    .filter((n): n is number => n != null);
  const max = caps.length ? Math.min(...caps) : 32768;
  const presets = PRESETS.filter((n) => n <= max);
  if (max < 32768 && !presets.includes(max)) {
    presets.push(max);
    presets.sort((a, b) => a - b);
  }
  return presets;
}

type Pending = "starting" | "stopping";

export type RowModel = {
  artifact: Artifact;
  session: Session | undefined;
  status: RowStatus;
  error?: string;
};

export type CatalogModel = {
  rows: RowModel[];
  presets: number[];
  nCtx: number;
  setNCtx: (n: number) => void;
  workerPath: string | null | undefined;
  error: string | null;
  tryFor: string | null;
  rescan: () => Promise<void>;
  start: (id: string) => Promise<void>;
  stop: (id: string) => Promise<void>;
  pin: (id: string) => Promise<void>;
  whatIf: (id: string) => Promise<void>;
  openTry: (id: string) => void;
  closeTry: () => void;
  inspect: () => void;
};

export function useCatalog(): CatalogModel {
  const [nCtx, setNCtx] = useState(4096);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [pending, setPending] = useState<Record<string, Pending>>({});
  const [localErrors, setLocalErrors] = useState<Record<string, string>>({});
  const [workerPath, setWorkerPath] = useState<string | null | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [tryFor, setTryFor] = useState<string | null>(null);

  const refreshSessions = useCallback(async () => {
    setSessions(await api.sessions());
  }, []);

  const rescan = useCallback(async () => {
    try {
      const [hw] = await Promise.all([api.hardware(), api.scan()]);
      setWorkerPath(hw.worker_path);
      const [cat, live] = await Promise.all([api.catalog(nCtx), api.sessions()]);
      setArtifacts(cat.artifacts);
      setSessions(live);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [nCtx]);

  useEffect(() => {
    void rescan();
  }, [rescan]);

  useEffect(() => {
    const timer = setInterval(() => void refreshSessions().catch(() => {}), 4000);
    return () => clearInterval(timer);
  }, [refreshSessions]);

  const sessionFor = useCallback(
    (id: string) => sessions.find((s) => s.artifact_id === id && s.n_ctx === nCtx),
    [sessions, nCtx]
  );

  const start = useCallback(
    async (id: string) => {
      setPending((p) => ({ ...p, [id]: "starting" }));
      setLocalErrors((e) => {
        const next = { ...e };
        delete next[id];
        return next;
      });
      try {
        await api.start(id, nCtx);
      } catch (e) {
        setLocalErrors((errs) => ({
          ...errs,
          [id]: e instanceof Error ? e.message : String(e),
        }));
      } finally {
        setPending((p) => {
          const next = { ...p };
          delete next[id];
          return next;
        });
        await refreshSessions().catch(() => {});
      }
    },
    [nCtx, refreshSessions]
  );

  const stop = useCallback(
    async (id: string) => {
      const session = sessionFor(id);
      if (!session) return;
      setPending((p) => ({ ...p, [id]: "stopping" }));
      try {
        await api.stop(session.id);
      } catch (e) {
        setLocalErrors((errs) => ({
          ...errs,
          [id]: e instanceof Error ? e.message : String(e),
        }));
      } finally {
        setPending((p) => {
          const next = { ...p };
          delete next[id];
          return next;
        });
        await refreshSessions().catch(() => {});
      }
      if (tryFor === id) setTryFor(null);
    },
    [sessionFor, refreshSessions, tryFor]
  );

  const rows = useMemo<RowModel[]>(
    () =>
      artifacts.map((artifact) => {
        const session = sessionFor(artifact.id);
        const busy = pending[artifact.id];
        const localError = localErrors[artifact.id];
        let status: RowStatus = "idle";
        if (busy) status = busy;
        else if (session?.status === "loaded") status = "loaded";
        else if (session?.status === "failed" || localError) status = "failed";
        return {
          artifact,
          session,
          status,
          error: localError ?? session?.last_error,
        };
      }),
    [artifacts, sessionFor, pending, localErrors]
  );

  const openTry = useCallback(
    (id: string) => {
      setTryFor(id);
      const session = sessionFor(id);
      if (session?.status !== "loaded" && !pending[id]) {
        void start(id);
      }
    },
    [sessionFor, pending, start]
  );

  const pin = useCallback(
    async (id: string) => {
      await api.pin(id, nCtx);
      location.hash = "#/capacity";
    },
    [nCtx]
  );

  const whatIf = useCallback(
    async (id: string) => {
      await api.whatIf(id, nCtx);
      location.hash = "#/capacity";
    },
    [nCtx]
  );

  return {
    rows,
    presets: presetsForArtifacts(artifacts),
    nCtx,
    setNCtx,
    workerPath,
    error,
    tryFor,
    rescan,
    start,
    stop,
    pin,
    whatIf,
    openTry,
    closeTry: () => setTryFor(null),
    inspect: () => {
      location.hash = "#/capacity";
    },
  };
}
