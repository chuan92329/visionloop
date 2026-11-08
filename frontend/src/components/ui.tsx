import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { STATUS_LABEL, type ProjectStats, type Status } from "../api";

// ------------------------------------------------------------------ toasts
type Toast = { id: number; text: string; kind: "info" | "error" };
const ToastCtx = createContext<(text: string, kind?: Toast["kind"]) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((text: string, kind: Toast["kind"] = "info") => {
    const id = next.current++;
    setToasts((t) => [...t, { id, text, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 7000 : 3500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === "error" ? "error" : ""}`}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);

/** Wrap an async action: shows errors as toasts and tracks a busy flag. */
export function useAction() {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const run = useCallback(
    async <T,>(fn: () => Promise<T>, success?: string): Promise<T | undefined> => {
      setBusy(true);
      try {
        const r = await fn();
        if (success) toast(success);
        return r;
      } catch (e) {
        toast(e instanceof Error ? e.message : String(e), "error");
        return undefined;
      } finally {
        setBusy(false);
      }
    },
    [toast],
  );
  return { busy, run };
}

// ------------------------------------------------------------------ modal
export function Modal({
  title,
  onClose,
  children,
  footer,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
        {footer && <div className="row" style={{ justifyContent: "flex-end" }}>{footer}</div>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ status
export function StatusPill({ status }: { status: Status }) {
  return <span className={`pill st-${status}`}>{STATUS_LABEL[status]}</span>;
}

const ORDER: Status[] = ["unlabeled", "auto", "in_progress", "reviewed"];

export function PipelineBar({
  stats,
  compact = false,
  onPick,
}: {
  stats: ProjectStats;
  compact?: boolean;
  onPick?: (s: Status) => void;
}) {
  const total = Math.max(stats.total, 1);
  return (
    <div className="pipeline">
      <div
        className={`pipeline-bar ${compact ? "compact" : ""}`}
        role="img"
        aria-label={ORDER.map((s) => `${STATUS_LABEL[s]} ${stats[s]}`).join("，")}
      >
        {ORDER.map((s) => (
          <span key={s} className={`bg-${s}`} style={{ width: `${(stats[s] / total) * 100}%` }} />
        ))}
      </div>
      {!compact && (
        <div className="pipeline-legend">
          {ORDER.map((s) => (
            <button key={s} type="button" onClick={() => onPick?.(s)} disabled={!onPick}>
              <span className={`swatch bg-${s}`} />
              {STATUS_LABEL[s]} <b>{stats[s]}</b>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function Progress({ value, kind }: { value: number; kind?: "auto" }) {
  return (
    <div className={`progress ${kind ?? ""}`} role="progressbar" aria-valuenow={Math.round(value * 100)}>
      <span style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "剛剛";
  if (s < 3600) return `${Math.floor(s / 60)} 分鐘前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小時前`;
  return d.toLocaleDateString("zh-TW");
}
