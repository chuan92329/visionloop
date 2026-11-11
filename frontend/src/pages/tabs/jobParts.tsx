// Pieces shared by the 輔助標註 and 訓練與權重 tabs: the job list, a few form controls, plugin ordering.

import { useCallback, useEffect, useState } from "react";
import { api, STATUS_LABEL, type Job, type PluginInfo, type Project, type Status, type Weights } from "../../api";
import { Progress, timeAgo } from "../../components/ui";

export const JOB_STATUS: Record<Job["status"], string> = {
  queued: "排隊中",
  running: "執行中",
  succeeded: "完成",
  failed: "失敗",
  cancelled: "已取消",
};

export function StatusChecks({
  project,
  choices,
  value,
  onChange,
}: {
  project: Project;
  choices: Status[];
  value: Status[];
  onChange: (next: Status[]) => void;
}) {
  return (
    <div className="field-group">
      <span className="small muted">標註對象</span>
      <div className="row" style={{ flexWrap: "wrap", gap: 12 }}>
        {choices.map((s) => (
          <label key={s} className="check">
            <input
              type="checkbox"
              checked={value.includes(s)}
              onChange={(e) => onChange(e.target.checked ? [...value, s] : value.filter((x) => x !== s))}
            />
            {STATUS_LABEL[s]}（{project.stats[s]}）
          </label>
        ))}
      </div>
    </div>
  );
}

// trains on the confirmed images and labels the rest in one step, keeps no weights
export function isInitialLabeling(p: PluginInfo): boolean {
  return p.uses_reviewed_images && !p.trainable;
}

export const PLUGIN_ORDER = ["probe", "rfdetr", "yolov9", "grounding-dino", "rtdetrv2", "sam2", "rfdetr-seg", "yolov9-seg"];
export function sortPlugins(list: PluginInfo[]): PluginInfo[] {
  const rank = (p: PluginInfo) => (PLUGIN_ORDER.indexOf(p.id) === -1 ? 500 : PLUGIN_ORDER.indexOf(p.id));
  return [...list].sort((a, b) => rank(a) - rank(b));
}

export function DeviceSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <label className="field">
      運算裝置
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="auto">自動</option>
        <option value="0">GPU 0</option>
        <option value="1">GPU 1</option>
        <option value="cpu">CPU</option>
      </select>
    </label>
  );
}

// ------------------------------------------------------------------- jobs
export function JobRow({
  job,
  plugins,
  onLog,
  onCancel,
  onDelete,
}: {
  job: Job;
  plugins: PluginInfo[];
  onLog: () => void;
  onCancel: () => void;
  onDelete: () => void;
}) {
  const name = plugins.find((p) => p.id === job.plugin)?.name ?? job.plugin;
  const live = job.status === "queued" || job.status === "running";
  const color = { succeeded: "st-reviewed", failed: "", cancelled: "st-unlabeled", running: "st-in_progress", queued: "st-unlabeled" }[job.status];
  return (
    <div className="job-row">
      <div style={{ minWidth: 0 }}>
        <div className="row">
          <b>
            #{job.id} {jobLabel(job, plugins)}
          </b>
          <span className="muted small">{name}</span>
          <span className={`pill ${color}`} style={job.status === "failed" ? { color: "var(--danger)" } : undefined}>
            {JOB_STATUS[job.status]}
          </span>
        </div>
        <p className="small muted" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={job.message}>
          {job.message} <span className="faint">，{timeAgo(job.created_at)}</span>
        </p>
      </div>
      <div>{live ? <Progress value={job.progress} kind={job.kind === "autolabel" ? "auto" : undefined} /> : null}</div>
      <div className="row">
        <button className="ghost" onClick={onLog}>
          紀錄
        </button>
        {live ? (
          <button className="ghost danger" onClick={onCancel}>
            取消
          </button>
        ) : (
          <button className="ghost danger" onClick={onDelete}>
            刪除
          </button>
        )}
      </div>
    </div>
  );
}

export function JobLog({ jobId, onClose }: { jobId: number; onClose: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const j = await api.job(jobId);
      if (stop) return;
      setJob(j);
      if (j.status === "queued" || j.status === "running") setTimeout(tick, 1500);
    };
    tick();
    return () => {
      stop = true;
    };
  }, [jobId]);
  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: "min(900px, 100%)" }} role="dialog" aria-label="任務紀錄">
        <div className="row">
          <h2>任務 #{jobId} 紀錄</h2>
          <span className="spacer" />
          <button className="ghost" onClick={onClose}>
            關閉
          </button>
        </div>
        {job && (
          <>
            <p className="small muted">
              {JOB_STATUS[job.status]}：{job.message}
            </p>
            {Object.keys(job.result ?? {}).length > 0 && (
              <pre className="log" style={{ maxHeight: 120 }}>
                {JSON.stringify(job.result, null, 2)}
              </pre>
            )}
            <pre className="log">{job.log || "（尚無輸出）"}</pre>
          </>
        )}
      </div>
    </div>
  );
}

export function isLive(j: Job): boolean {
  return j.status === "queued" || j.status === "running";
}

/** What a job does, in the words the forms use. */
export function jobLabel(job: Job, plugins: PluginInfo[]): string {
  if (job.kind === "train") return "訓練";
  const p = plugins.find((x) => x.id === job.plugin);
  if (p && isInitialLabeling(p)) return "初步標註";
  if (p?.prompt_mode === "boxes") return "框轉多邊形";
  return "自動標註";
}

/** The project's jobs and weights, refreshed while anything is queued or running. */
export function useProjectJobs(projectId: number, onChanged: () => void) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [weights, setWeights] = useState<Weights[]>([]);

  const refresh = useCallback(async () => {
    const [j, w] = await Promise.all([api.jobs(projectId), api.weights(projectId)]);
    setJobs(j);
    setWeights(w);
  }, [projectId]);

  const active = jobs.some(isLive);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(async () => {
      await refresh();
      onChanged(); // image counts move while a job labels
    }, 1500);
    return () => clearInterval(t);
  }, [active, refresh, onChanged]);

  return { jobs, weights, refresh, active };
}

export type PolygonDetail = "normal" | "fine";

/** Size, outline detail and overwrite: the same choices wherever boxes are turned into polygons. */
export function OutlineOptions({
  plugin,
  variant,
  onVariant,
  detail,
  onDetail,
  overwrite,
  onOverwrite,
}: {
  plugin: PluginInfo;
  variant: string;
  onVariant: (v: string) => void;
  detail: PolygonDetail;
  onDetail: (d: PolygonDetail) => void;
  overwrite: boolean;
  onOverwrite: (o: boolean) => void;
}) {
  return (
    <>
      {plugin.variants.length > 1 && (
        <label className="field">
          尺寸
          <select value={variant} onChange={(e) => onVariant(e.target.value)}>
            {plugin.variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
                {v.description ? `，${v.description}` : ""}
              </option>
            ))}
          </select>
        </label>
      )}
      <label className="field">
        輪廓點數
        <select value={detail} onChange={(e) => onDetail(e.target.value as PolygonDetail)}>
          <option value="normal">標準（12MP 照片約 80 點）</option>
          <option value="fine">密集（約 250 點，手動調整較不方便）</option>
        </select>
      </label>
      <label className="check">
        <input type="checkbox" checked={overwrite} onChange={(e) => onOverwrite(e.target.checked)} />
        重新描繪已有多邊形的物件
      </label>
      {overwrite && (
        <p className="warning small">現有的多邊形會被取代，包含你手動調整過的。舊版本保留在版本紀錄裡，可以逐張還原。</p>
      )}
    </>
  );
}
