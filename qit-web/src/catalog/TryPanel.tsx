import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  generate,
  type Artifact,
  type ChatMessage,
  type GenerateDone,
} from "../api";
import { ContextSquare } from "./ContextSquare";
import type { RowStatus } from "./StartControl";

type Turn =
  | { kind: "user"; content: string }
  | { kind: "assistant"; content: string; streaming: boolean }
  | { kind: "system"; content: string };

type Props = {
  artifact: Artifact;
  nCtx: number;
  status: RowStatus;
  startError?: string;
  onClose: () => void;
};

function toMessages(turns: Turn[]): ChatMessage[] {
  return turns.flatMap((t) =>
    t.kind === "user" || t.kind === "assistant"
      ? [{ role: t.kind, content: t.content }]
      : []
  );
}

export function TryPanel({
  artifact,
  nCtx,
  status,
  startError,
  onClose,
}: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [usage, setUsage] = useState<GenerateDone | null>(null);
  const [liveTokens, setLiveTokens] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const ready = status === "loaded";
  const settled = usage ? usage.prompt_tokens + usage.completion_tokens : 0;
  const used = streaming ? settled + liveTokens : settled;

  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  useEffect(() => {
    if (ready) inputRef.current?.focus();
  }, [ready]);

  useEffect(() => () => abortRef.current?.abort(), []);

  function appendToAssistant(token: string) {
    setTurns((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && last.kind === "assistant") {
        next[next.length - 1] = { ...last, content: last.content + token };
      }
      return next;
    });
  }

  function finishAssistant() {
    setTurns((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && last.kind === "assistant") {
        next[next.length - 1] = { ...last, streaming: false };
      }
      return next;
    });
  }

  function pushSystem(content: string) {
    setTurns((prev) => [...prev, { kind: "system", content }]);
  }

  async function send() {
    const content = draft.trim();
    if (!content || streaming || !ready) return;
    const history = toMessages(turns);
    const outgoing: ChatMessage[] = [...history, { role: "user", content }];
    setDraft("");
    setTurns((prev) => [
      ...prev,
      { kind: "user", content },
      { kind: "assistant", content: "", streaming: true },
    ]);
    setStreaming(true);
    setLiveTokens(0);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await generate(artifact.id, nCtx, outgoing, ac.signal, {
        onToken: (token) => {
          appendToAssistant(token);
          setLiveTokens((n) => n + 1);
        },
        onDone: (done) => setUsage(done),
        onError: (message) => pushSystem(message),
      });
    } catch (e) {
      if ((e as { name?: string }).name === "AbortError") {
        pushSystem("Stopped.");
      } else if (e instanceof ApiError && e.status === 409) {
        pushSystem("Another Try is running. Wait for it to finish.");
      } else {
        pushSystem(e instanceof Error ? e.message : String(e));
      }
    } finally {
      finishAssistant();
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  const headline =
    status === "starting"
      ? "Loading model…"
      : status === "failed"
        ? "Failed to load"
        : status === "loaded"
          ? `${nCtx / 1024}k context`
          : "Not loaded";

  return (
    <div className="try">
      <div className="chat">
        <div className="chat-head">
          <span>
            <span className="mono">{artifact.filename}</span> · {headline}
          </span>
          <button className="quiet" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="transcript" ref={transcriptRef}>
          {turns.length === 0 && (
            <div className="msg system">
              {status === "starting" && (
                <span>
                  <span className="spinner" style={{ display: "inline-block", verticalAlign: "-2px", marginRight: 6 }} />
                  Starting a worker for this artifact.
                </span>
              )}
              {status === "failed" && (startError ?? "The worker did not start.")}
              {status === "loaded" && "Ask anything. The transcript stays until you close this window."}
            </div>
          )}
          {turns.map((t, i) => (
            <div
              key={i}
              className={`msg ${t.kind}${
                t.kind === "assistant" && t.streaming ? " streaming" : ""
              }`}
            >
              {t.content}
            </div>
          ))}
        </div>
        <div className="composer">
          <span className="prompt-mark">›</span>
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={ready ? "Message the model" : "Waiting for the worker…"}
            disabled={!ready || streaming}
          />
          {streaming ? (
            <button onClick={() => abortRef.current?.abort()}>Stop</button>
          ) : (
            <button
              className="primary"
              onClick={() => void send()}
              disabled={!ready || !draft.trim()}
            >
              Send
            </button>
          )}
          <div className="hint">
            <kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline
          </div>
        </div>
      </div>
      <ContextSquare used={used} total={nCtx} />
    </div>
  );
}
