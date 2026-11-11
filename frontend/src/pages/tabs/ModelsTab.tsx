import { useCallback, useEffect, useMemo, useState } from "react";
import { api, STATUS_LABEL, type Job, type PluginInfo, type PluginList, type Project, type SplitStats, type Status, type SystemInfo, type Variant, type Weights } from "../../api";
import { Modal, Progress, timeAgo, useAction, useToast } from "../../components/ui";

const JOB_STATUS: Record<Job["status"], string> = {
  queued: "排隊中",
  running: "執行中",
  succeeded: "完成",
  failed: "失敗",
  cancelled: "已取消",
};

type JobFilter = "all" | Job["kind"] | "problem";
const JOBS_SHOWN = 5;

function isProblemJob(j: Job): boolean {
  return j.status === "failed" || j.status === "cancelled";
}

export default function ModelsTab({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [loadErrors, setLoadErrors] = useState<{ file: string; error: string }[]>([]);
  const [pluginDir, setPluginDir] = useState("");
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [weights, setWeights] = useState<Weights[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [logJob, setLogJob] = useState<number | null>(null);
  const [importing, setImporting] = useState(false);
  const [jobFilter, setJobFilter] = useState<JobFilter>("all");
  const [showAllJobs, setShowAllJobs] = useState(false);
  const { run } = useAction();

  const refresh = useCallback(async () => {
    const [j, w] = await Promise.all([api.jobs(project.id), api.weights(project.id)]);
    setJobs(j);
    setWeights(w);
  }, [project.id]);

  const applyPlugins = (p: PluginList) => {
    setPlugins(p.plugins);
    setLoadErrors(p.load_errors);
    setPluginDir(p.plugin_dir);
  };

  useEffect(() => {
    run(async () => {
      const [p, s] = await Promise.all([api.plugins(), api.system()]);
      applyPlugins(p);
      setSystem(s);
      await refresh();
    });
  }, [run, refresh]);

  const reloadPlugins = async () => {
    const p = await run(() => api.reloadPlugins(), "已重新載入外掛");
    if (p) applyPlugins(p);
  };

  // poll while something is running
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(async () => {
      const before = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
      await refresh();
      if (before) onChanged();
    }, 1500);
    return () => clearInterval(t);
  }, [active, refresh, onChanged, jobs]);

  const maxVram = system?.gpus.length ? Math.max(...system.gpus.map((g) => g.memory_gb)) : 0;

  const clearable = jobs.filter(isProblemJob);
  const shownJobs = jobs.filter((j) => (jobFilter === "all" ? true : jobFilter === "problem" ? isProblemJob(j) : j.kind === jobFilter));

  const deleteJob = async (j: Job) => {
    if (!confirm(`刪除任務 #${j.id} 的紀錄？權重和標註不受影響。`)) return;
    await run(() => api.deleteJob(j.id), "已刪除任務紀錄");
    if (logJob === j.id) setLogJob(null);
    refresh();
  };

  const clearFinished = async () => {
    if (!confirm(`刪除 ${clearable.length} 筆失敗與已取消的任務紀錄？`)) return;
    await run(() => api.clearJobs(project.id, ["failed", "cancelled"]), "已清除任務紀錄");
    setLogJob(null);
    refresh();
  };

  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="three-col">
        <TrainPanel project={project} plugins={plugins} weights={weights} maxVram={maxVram} onStarted={refresh} />
        <AutoLabelForm project={project} plugins={plugins} weights={weights} onStarted={refresh} onChanged={onChanged} />
        <OutlineForm project={project} plugins={plugins} onStarted={refresh} />
      </div>

      <div className="panel stack">
        <div className="row" style={{ flexWrap: "wrap" }}>
          <h2>任務</h2>
          {active && <span className="small muted">每 1.5 秒自動更新</span>}
          <span className="spacer" />
          {clearable.length > 0 && (
            <button className="ghost danger" onClick={clearFinished}>
              清除失敗與已取消（{clearable.length}）
            </button>
          )}
          <select aria-label="篩選任務" value={jobFilter} onChange={(e) => setJobFilter(e.target.value as JobFilter)}>
            <option value="all">全部</option>
            <option value="train">訓練</option>
            <option value="autolabel">自動標註</option>
            <option value="problem">失敗與已取消</option>
          </select>
        </div>
        {jobs.length === 0 && <p className="muted">還沒有執行過任務。</p>}
        {jobs.length > 0 && shownJobs.length === 0 && <p className="muted">沒有符合的任務。</p>}
        <div>
          {(showAllJobs ? shownJobs : shownJobs.slice(0, JOBS_SHOWN)).map((j) => (
            <JobRow
              key={j.id}
              job={j}
              plugins={plugins}
              onLog={() => setLogJob(j.id)}
              onCancel={() => run(() => api.cancelJob(j.id)).then(refresh)}
              onDelete={() => deleteJob(j)}
            />
          ))}
        </div>
        {shownJobs.length > JOBS_SHOWN && (
          <button className="ghost" onClick={() => setShowAllJobs(!showAllJobs)}>
            {showAllJobs ? "只顯示最近幾筆" : `顯示全部（${shownJobs.length}）`}
          </button>
        )}
      </div>

      <div className="two-col">
        <div className="panel stack">
          <div className="row">
            <h2>權重</h2>
            <span className="spacer" />
            <button onClick={() => setImporting(true)}>匯入現成權重</button>
          </div>
          {weights.length === 0 && (
            <p className="muted small">訓練完成的模型會出現在這裡。也可以匯入官方的 COCO 預訓練權重，直接拿來做第一輪自動標註。</p>
          )}
          {weights.map((w) => (
            <WeightsRow
              key={w.id}
              w={w}
              plugin={plugins.find((p) => p.id === w.plugin)}
              onDelete={async () => {
                if (!confirm(`刪除權重「${w.name}」？`)) return;
                await run(() => api.deleteWeights(w.id), "已刪除權重");
                refresh();
              }}
            />
          ))}
        </div>

        <div className="panel stack">
          <h2>這台電腦</h2>
          {system && system.gpus.length > 0 ? (
            system.gpus.map((g) => (
              <p key={g.index}>
                GPU {g.index}：{g.name}，{g.memory_gb} GB
              </p>
            ))
          ) : (
            <p className="muted small">偵測不到 NVIDIA 顯示卡（找不到 nvidia-smi）。訓練仍可在 CPU 上執行，但會非常慢。</p>
          )}
          <div className="row" style={{ marginTop: 6 }}>
            <h3>模型外掛</h3>
            <span className="spacer" />
            <button className="ghost small" onClick={reloadPlugins}>
              重新載入外掛
            </button>
          </div>
          <p className="small muted">
            要匯入其他模型系列：把外掛 .py 檔放進 <code>{pluginDir}</code>，再按「重新載入外掛」，它就會出現在訓練選單。寫法見該資料夾的 README.md。外掛是會在這台電腦執行的程式碼，只放你信任的檔案。
          </p>
          {plugins.map((p) => (
            <div key={p.id} className="stack" style={{ gap: 2, paddingBottom: 8, borderBottom: "1px solid var(--line)" }}>
              <div className="row">
                <b>{p.name}</b>
                <span className="spacer" />
                <span className={`pill ${p.available ? "st-reviewed" : "st-unlabeled"}`}>{p.available ? "可用" : "未設定"}</span>
              </div>
              <span className="small muted">
                授權：{p.license}
                {p.homepage && (
                  <>
                    ，
                    <a href={p.homepage} target="_blank" rel="noreferrer">
                      原始專案
                    </a>
                  </>
                )}
              </span>
              {!p.available && <span className="small st-in_progress">{p.status_message}</span>}
            </div>
          ))}
          {loadErrors.map((e) => (
            <p key={e.file} className="warning error small">
              外掛 {e.file} 載入失敗：{e.error}
            </p>
          ))}
        </div>
      </div>

      {logJob !== null && <JobLog jobId={logJob} onClose={() => setLogJob(null)} />}
      {importing && (
        <ImportWeights
          project={project}
          plugins={plugins}
          onClose={() => setImporting(false)}
          onDone={() => {
            setImporting(false);
            refresh();
          }}
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------- auto-label
function AutoLabelForm({
  project,
  plugins,
  weights,
  onStarted,
  onChanged,
}: {
  project: Project;
  plugins: PluginInfo[];
  weights: Weights[];
  onStarted: () => void;
  onChanged: () => void;
}) {
  const usable = sortPlugins(plugins.filter((p) => p.available && !isInitialLabeling(p) && p.prompt_mode !== "boxes"));
  const [pluginId, setPluginId] = useState("");
  // until the user picks one: the model trained last, because that is the next step after training
  const latest = [...weights]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .find((w) => usable.some((p) => p.id === w.plugin));
  const plugin = usable.find((p) => p.id === pluginId) ?? usable.find((p) => p.id === latest?.plugin) ?? usable[0];
  const [weightsId, setWeightsId] = useState<number | "" | "none">("");
  const [variant, setVariant] = useState("");
  const [confByPlugin, setConfByPlugin] = useState<Record<string, number>>({});
  const conf = (plugin && confByPlugin[plugin.id]) ?? plugin?.default_confidence ?? 0.25;
  const [targets, setTargets] = useState<Status[]>(["unlabeled"]);
  const [device, setDevice] = useState("auto");
  const [prompts,setPrompts] = useState<Record<number, string>>(() => loadPrompts(project.id));
  const [exemplars, setExemplars] = useState<Record<string, number> | null>(null);
  const { busy, run } = useAction();
  const toast = useToast();

  useEffect(() => {
    api.exemplars(project.id).then(setExemplars).catch(() => setExemplars({}));
  }, [project.id, project.stats.reviewed]);

  const pluginWeights = weights.filter((w) => w.plugin === plugin?.id);
  const selectedWeights =
    weightsId === "none" ? undefined : pluginWeights.find((w) => w.id === weightsId) ?? pluginWeights[0];
  const variantId = plugin?.variants.find((v) => v.id === variant)?.id ?? plugin?.variants[0]?.id ?? "";
  const exemplarTotal = exemplars ? Object.values(exemplars).reduce((a, b) => a + b, 0) : 0;
  const mode = plugin?.prompt_mode ?? "none";
  const count = targets.reduce((n, s) => n + project.stats[s], 0);

  const start = async () => {
    if (!plugin) return;
    if (mode === "text") savePrompts(project.id, prompts);
    const job = await run(
      () =>
        api.autolabel(project.id, {
          plugin: plugin.id,
          weights_id: selectedWeights?.id ?? null,
          variant: selectedWeights ? undefined : variantId,
          confidence: conf,
          statuses: targets,
          device,
          prompts: mode === "text" ? prompts : undefined,
        }),
      "自動標註已開始",
    );
    if (job) onStarted();
  };

  const clearAuto = async () => {
    if (!confirm(`清除 ${project.stats.auto} 張「自動標註」圖片上的所有框，讓它們回到「未標註」？人工編輯過或已確認的圖片不受影響。`)) return;
    const r = await run(() => api.clearAuto(project.id));
    if (r) {
      toast(`已清除 ${r.cleared} 張圖片的自動標註`);
      onChanged();
    }
  };

  const needsWeights = plugin?.needs_weights && !selectedWeights;
  const missingExamples = mode === "exemplars" && exemplarTotal === 0;

  return (
    <div className="panel" data-tour="autolabel">
      <h2>自動標註</h2>
      <p className="small muted">用訓練好的權重，或不用訓練的模型。</p>
      <label className="field">
        模型
        <select value={plugin?.id ?? ""} onChange={(e) => setPluginId(e.target.value)}>
          {usable.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>

      {(pluginWeights.length > 0 || plugin?.needs_weights) && (
        <label className="field">
          權重
          <select
            value={selectedWeights?.id ?? (plugin?.needs_weights ? "" : "none")}
            onChange={(e) => setWeightsId(e.target.value === "none" ? "none" : Number(e.target.value) || "")}
            disabled={!pluginWeights.length}
          >
            {pluginWeights.length === 0 && <option value="">沒有可用的權重</option>}
            {!plugin?.needs_weights && <option value="none">{plugin?.no_weights_label}</option>}
            {pluginWeights.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {!selectedWeights && plugin && plugin.variants.length > 1 && (
        <label className="field">
          尺寸
          <select value={variantId} onChange={(e) => setVariant(e.target.value)}>
            {plugin.variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
      )}

      {mode === "exemplars" && exemplars && (
        <div className={missingExamples ? "warning small" : "small"}>
          {missingExamples ? (
            "先確認幾個物件，每類 3～5 個。"
          ) : (
            <>
              已確認：
              {project.classes.map((c) => (
                <span key={c.id} style={{ marginLeft: 8 }} className={exemplars[String(c.id)] ? "" : "faint"}>
                  {c.name} {exemplars[String(c.id)] ?? 0} 個
                </span>
              ))}
            </>
          )}
        </div>
      )}

      {mode === "text" && (
        <div className="stack" style={{ gap: 6 }}>
          <span className="small muted">用英文描述每個類別，留空就用類別名稱。</span>
          {project.classes.map((c) => (
            <label key={c.id} className="row small">
              <span className="swatch" style={{ background: c.color }} />
              <span style={{ width: 90, overflow: "hidden", textOverflow: "ellipsis" }}>{c.name}</span>
              <input
                style={{ flex: 1 }}
                value={prompts[c.id] ?? ""}
                placeholder={c.name}
                onChange={(e) => setPrompts({ ...prompts, [c.id]: e.target.value })}
              />
            </label>
          ))}
        </div>
      )}

      <label className="field">
        信心門檻：{conf.toFixed(2)}
        <input
          type="range"
          min={0.05}
          max={0.95}
          step={0.05}
          value={conf}
          onChange={(e) => plugin && setConfByPlugin({ ...confByPlugin, [plugin.id]: Number(e.target.value) })}
        />
      </label>
      <StatusChecks project={project} choices={["unlabeled", "auto"]} value={targets} onChange={setTargets} />
      <DeviceSelect value={device} onChange={setDevice} />
      {needsWeights && <p className="warning small">需要權重：先訓練，或匯入權重。</p>}
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || !plugin || needsWeights || missingExamples || count === 0}>
          自動標註 {count} 張圖片
        </button>
        <span className="spacer" />
        {project.stats.auto > 0 && (
          <button className="ghost danger" onClick={clearAuto} disabled={busy}>
            清除自動標註（{project.stats.auto}）
          </button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------- box to polygon
function OutlineForm({
  project,
  plugins,
  onStarted,
}: {
  project: Project;
  plugins: PluginInfo[];
  onStarted: () => void;
}) {
  const usable = sortPlugins(plugins.filter((p) => p.available && p.prompt_mode === "boxes"));
  const [pluginId, setPluginId] = useState("");
  const plugin = usable.find((p) => p.id === pluginId) ?? usable[0];
  const [variant, setVariant] = useState("");
  const variantId = plugin?.variants.find((v) => v.id === variant)?.id ?? plugin?.variants[0]?.id ?? "";
  const [targets, setTargets] = useState<Status[]>(["auto"]);
  const [detail, setDetail] = useState<"normal" | "fine">("normal");
  const [overwrite, setOverwrite] = useState(false);
  const [device, setDevice] = useState("auto");
  const { busy, run } = useAction();

  if (!plugin) return null;
  const count = targets.reduce((n, s) => n + project.stats[s], 0);

  const start = async () => {
    const job = await run(
      () =>
        api.autolabel(project.id, {
          plugin: plugin.id,
          weights_id: null,
          variant: variantId,
          confidence: plugin.default_confidence,
          statuses: targets,
          device,
          polygon_detail: detail,
          overwrite_polygons: overwrite,
        }),
      "框轉多邊形已開始",
    );
    if (job) onStarted();
  };

  return (
    <div className="panel">
      <h2>框轉多邊形</h2>
      <p className="small muted">把已有的框描成多邊形，不會找新物件。</p>
      {usable.length > 1 && (
        <label className="field">
          模型
          <select value={plugin.id} onChange={(e) => setPluginId(e.target.value)}>
            {usable.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {plugin.variants.length > 1 && (
        <label className="field">
          尺寸
          <select value={variantId} onChange={(e) => setVariant(e.target.value)}>
            {plugin.variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
      )}
      <label className="field">
        輪廓點數
        <select value={detail} onChange={(e) => setDetail(e.target.value as "normal" | "fine")}>
          <option value="normal">標準（約 80 點）</option>
          <option value="fine">密集（約 250 點）</option>
        </select>
      </label>
      <StatusChecks project={project} choices={["auto", "in_progress", "reviewed"]} value={targets} onChange={setTargets} />
      <label className="check">
        <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
        重新描繪已有多邊形的物件
      </label>
      <DeviceSelect value={device} onChange={setDevice} />
      {overwrite && <p className="warning small">現有多邊形會被取代，可從版本紀錄還原。</p>}
      {targets.includes("reviewed") && <p className="warning small">已確認的圖片轉換後要重新確認。</p>}
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || count === 0}>
          轉換 {count} 張圖片的框
        </button>
      </div>
    </div>
  );
}

function StatusChecks({
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
function isInitialLabeling(p: PluginInfo): boolean {
  return p.uses_reviewed_images && !p.trainable;
}

const PLUGIN_ORDER = ["probe", "rfdetr", "yolov9", "grounding-dino", "rtdetrv2", "sam2", "rfdetr-seg", "yolov9-seg"];
function sortPlugins(list: PluginInfo[]): PluginInfo[] {
  const rank = (p: PluginInfo) => (PLUGIN_ORDER.indexOf(p.id) === -1 ? 500 : PLUGIN_ORDER.indexOf(p.id));
  return [...list].sort((a, b) => rank(a) - rank(b));
}

function loadPrompts(pid: number): Record<number, string> {
  try {
    return JSON.parse(localStorage.getItem(`visionloop.prompts.${pid}`) || "{}");
  } catch {
    return {};
  }
}

function savePrompts(pid: number, prompts: Record<number, string>) {
  try {
    localStorage.setItem(`visionloop.prompts.${pid}`, JSON.stringify(prompts));
  } catch {
    /* storage unavailable */
  }
}

// ------------------------------------------------------------------ train
function TrainPanel({
  project,
  plugins,
  weights,
  maxVram,
  onStarted,
}: {
  project: Project;
  plugins: PluginInfo[];
  weights: Weights[];
  maxVram: number;
  onStarted: () => void;
}) {
  const initial = sortPlugins(plugins.filter((p) => p.available && isInitialLabeling(p)));
  const [purpose, setPurpose] = useState<"initial" | "final">("initial");
  const active = initial.length ? purpose : "final";
  // both forms stay mounted, so switching back does not lose what was filled in
  const shown = (p: typeof active) => ({ display: active === p ? undefined : "none" });
  return (
    <div className="panel" data-tour="train">
      <h2>訓練</h2>
      <label className="field">
        訓練目的
        <select value={active} onChange={(e) => setPurpose(e.target.value as typeof active)}>
          <option value="initial" disabled={!initial.length}>
            初步標註（Linear Probe）{initial.length ? "" : "，未設定"}
          </option>
          <option value="final">正式模型（YOLO、RF-DETR 等）</option>
        </select>
      </label>
      {initial.length > 0 && (
        <div className="form-body" style={shown("initial")}>
          <InitialLabelForm project={project} plugins={initial} onStarted={onStarted} />
        </div>
      )}
      <div className="form-body" style={shown("final")}>
        <TrainForm project={project} plugins={plugins} weights={weights} maxVram={maxVram} onStarted={onStarted} />
      </div>
    </div>
  );
}

function InitialLabelForm({
  project,
  plugins,
  onStarted,
}: {
  project: Project;
  plugins: PluginInfo[];
  onStarted: () => void;
}) {
  const [pluginId, setPluginId] = useState("");
  const plugin = plugins.find((p) => p.id === pluginId) ?? plugins[0];
  const [variant, setVariant] = useState("");
  const variantId = plugin.variants.find((v) => v.id === variant)?.id ?? plugin.variants[0]?.id ?? "";
  const [head, setHead] = useState<"linear" | "mlp">("linear");
  const [confInput, setConfInput] = useState<number | null>(null);
  const conf = confInput ?? plugin.default_confidence;
  const [targets, setTargets] = useState<Status[]>(["unlabeled"]);
  const [device, setDevice] = useState("auto");
  const [exemplars, setExemplars] = useState<Record<string, number> | null>(null);
  const { busy, run } = useAction();

  useEffect(() => {
    api.exemplars(project.id).then(setExemplars).catch(() => setExemplars({}));
  }, [project.id, project.stats.reviewed]);

  const noExamples = exemplars !== null && Object.values(exemplars).every((n) => !n);
  const count = targets.reduce((n, s) => n + project.stats[s], 0);
  // the first size is the recommended default; only another choice has something to point out
  const variantNote = variantId !== plugin.variants[0]?.id ? plugin.variants.find((v) => v.id === variantId)?.description : "";

  const start = async () => {
    const job = await run(
      () =>
        api.autolabel(project.id, {
          plugin: plugin.id,
          weights_id: null,
          variant: variantId,
          confidence: conf,
          statuses: targets,
          device,
          probe_head: head,
        }),
      "初步標註已開始",
    );
    if (job) onStarted();
  };

  return (
    <>
      <p className="small muted">先用已確認的圖片訓練，再標註其餘圖片。</p>
      {plugins.length > 1 && (
        <label className="field">
          模型
          <select value={plugin.id} onChange={(e) => setPluginId(e.target.value)}>
            {plugins.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {exemplars && (
        <div className={noExamples ? "warning small" : "small"}>
          {noExamples ? (
            "先確認幾個物件，每類 3～5 個。"
          ) : (
            <>
              已確認：
              {project.classes.map((c) => (
                <span key={c.id} style={{ marginLeft: 8 }} className={exemplars[String(c.id)] ? "" : "faint"}>
                  {c.name} {exemplars[String(c.id)] ?? 0} 個
                </span>
              ))}
            </>
          )}
        </div>
      )}
      <label className="field">
        骨幹
        <select value={variantId} onChange={(e) => setVariant(e.target.value)}>
          {plugin.variants.map((v) => (
            <option key={v.id} value={v.id}>
              {v.name}
            </option>
          ))}
        </select>
      </label>
      {variantNote && <p className="small muted">{variantNote}</p>}
      <label className="field">
        探針
        <select value={head} onChange={(e) => setHead(e.target.value as "linear" | "mlp")}>
          <option value="linear">線性層（建議）</option>
          <option value="mlp">小型 MLP</option>
        </select>
      </label>
      <label className="field">
        信心門檻：{conf.toFixed(2)}
        <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConfInput(Number(e.target.value))} />
      </label>
      <StatusChecks project={project} choices={["unlabeled", "auto"]} value={targets} onChange={setTargets} />
      <DeviceSelect value={device} onChange={setDevice} />
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || noExamples || count === 0}>
          開始初步標註（{count} 張）
        </button>
      </div>
    </>
  );
}

function TrainForm({
  project,
  plugins,
  weights,
  maxVram,
  onStarted,
}: {
  project: Project;
  plugins: PluginInfo[];
  weights: Weights[];
  maxVram: number;
  onStarted: () => void;
}) {
  const usable = sortPlugins(plugins.filter((p) => p.available && p.trainable));
  const [task, setTask] = useState<PluginInfo["task"]>("detect");
  const inTask = usable.filter((p) => p.task === task);
  const families = [...new Set(inTask.map((p) => p.family))];
  const [familyChoice, setFamilyChoice] = useState("");
  const family = families.includes(familyChoice) ? familyChoice : families[0];
  // a family may consist of several plugins: the third level lists all their sizes
  const choices = useMemo(
    () => inTask.filter((p) => p.family === family).flatMap((p) => p.variants.map((variant) => ({ plugin: p, variant }))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [plugins, task, family],
  );
  const key = (c: { plugin: PluginInfo; variant: Variant }) => `${c.plugin.id}|${c.variant.id}`;
  const recommended = useMemo(() => {
    if (!choices.length) return "";
    // without a detected GPU, start with the smallest size
    if (!maxVram) return key(choices[0]);
    const fit = choices.filter((c) => c.variant.min_vram_gb <= maxVram);
    return key(fit.length ? fit[fit.length - 1] : choices[0]);
  }, [choices, maxVram]);
  const [choiceKey, setChoiceKey] = useState("");
  const choice = choices.find((c) => key(c) === choiceKey) ?? choices.find((c) => key(c) === recommended);
  const plugin = choice?.plugin;
  const variantInfo = choice?.variant;
  const [epochs, setEpochs] = useState(50);
  const [batch, setBatch] = useState(8);
  const [imgszInput, setImgszInput] = useState<number | null>(null); // null: the size's own default
  const [device, setDevice] = useState("auto");
  const [initId, setInitId] = useState<number | "">("");
  const [stats, setStats] = useState<SplitStats | null>(null);
  const { busy, run } = useAction();

  useEffect(() => {
    api.splitStats(project.id).then(setStats).catch(() => setStats(null));
  }, [project.id, project.stats.reviewed, project.stats.train]);

  // anything that changes the model starts again from that model's defaults
  const resetFor = () => {
    setImgszInput(null);
    setInitId("");
  };

  const baseSize = variantInfo?.image_size ?? 640;
  const imgsz = imgszInput ?? baseSize;
  const step = variantInfo?.image_step ?? 32;
  const sizeOk = imgsz >= step && imgsz % step === 0;
  const nearest = Math.max(step, Math.round(imgsz / step) * step);
  // min_vram_gb is for batch 8 at the default size; memory grows with the pixel count
  const estimatedVram = variantInfo?.min_vram_gb ? variantInfo.min_vram_gb * (imgsz / baseSize) ** 2 * (batch / 8) : 0;
  const tooBig = maxVram > 0 && estimatedVram > maxVram;
  const canTrain = project.stats.train > 0 && project.stats.valid > 0;
  const noPolygons = task === "segment" && stats !== null && stats.polygons === 0;
  const initOptions = weights.filter((w) => w.plugin === plugin?.id);

  const start = async () => {
    if (!plugin || !variantInfo) return;
    const job = await run(
      () =>
        api.train(project.id, {
          plugin: plugin.id,
          variant: variantInfo.id,
          epochs,
          batch_size: batch,
          image_size: imgsz,
          device,
          init_weights_id: initId || null,
        }),
      "訓練已開始",
    );
    if (job) onStarted();
  };

  return (
    <>
      <p className="small muted">
        train {project.stats.train} 張、valid {project.stats.valid} 張。
      </p>
      <label className="field">
        任務類型
        <select
          value={task}
          onChange={(e) => {
            setTask(e.target.value as PluginInfo["task"]);
            setFamilyChoice("");
            setChoiceKey("");
            resetFor();
          }}
        >
          <option value="detect">物件偵測（框）</option>
          <option value="segment">實例分割（多邊形）</option>
        </select>
      </label>
      {task === "segment" && stats && (
        <p className={`small ${stats.polygons === 0 ? "warning" : "muted"}`}>
          {stats.polygons === 0
            ? "還沒有多邊形：先畫多邊形，或用框轉多邊形並確認。"
            : `${stats.polygons} 個有多邊形${stats.boxes_without_polygon ? `，${stats.boxes_without_polygon} 個只有框（當成矩形）` : ""}。`}
        </p>
      )}
      <label className="field">
        模型系列
        <select
          value={family ?? ""}
          disabled={!families.length}
          onChange={(e) => {
            setFamilyChoice(e.target.value);
            setChoiceKey("");
            resetFor();
          }}
        >
          {!families.length && <option value="">沒有可用的模型</option>}
          {families.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        尺寸
        <select
          value={choice ? key(choice) : ""}
          onChange={(e) => {
            setChoiceKey(e.target.value);
            resetFor();
          }}
        >
          {choices.map((c) => (
            <option key={key(c)} value={key(c)}>
              {c.variant.name}
              {c.variant.min_vram_gb ? `，約 ${c.variant.min_vram_gb} GB` : ""}
              {key(c) === recommended && maxVram ? "（建議）" : ""}
            </option>
          ))}
        </select>
      </label>
      <div className="split-sliders">
        <label className="field">
          Epochs
          <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
        </label>
        <label className="field">
          Batch size
          <input type="number" min={1} value={batch} onChange={(e) => setBatch(Number(e.target.value))} />
        </label>
        <label className="field">
          影像尺寸
          <input type="number" min={step} step={step} value={imgsz} onChange={(e) => setImgszInput(Number(e.target.value))} />
        </label>
      </div>
      {!sizeOk && (
        <p className="warning small">
          影像尺寸必須是 {step} 的倍數，最接近的是 {nearest}。
        </p>
      )}
      {sizeOk && tooBig && (
        <p className="warning small">
          粗估需要 {Math.round(estimatedVram)} GB 顯存，超過你的 {maxVram} GB，可能很慢或失敗。
        </p>
      )}
      <label className="field">
        接續訓練自（選填）
        <select value={initId} onChange={(e) => setInitId(Number(e.target.value) || "")}>
          <option value="">不接續</option>
          {initOptions.map((w) => (
            <option key={w.id} value={w.id}>
              {w.name}
            </option>
          ))}
        </select>
      </label>
      <DeviceSelect value={device} onChange={setDevice} />
      {!canTrain && <p className="warning small">先到「資料切分與匯出」完成切分。</p>}
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || !plugin || !canTrain || !sizeOk || noPolygons}>
          開始訓練
        </button>
      </div>
    </>
  );
}

function DeviceSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
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
function JobRow({
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
            #{job.id} {job.kind === "train" ? "訓練" : "自動標註"}
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

function JobLog({ jobId, onClose }: { jobId: number; onClose: () => void }) {
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

// ---------------------------------------------------------------- weights
function WeightsRow({ w, plugin, onDelete }: { w: Weights; plugin?: PluginInfo; onDelete: () => void }) {
  const metrics = Object.entries(w.metrics).filter(([k]) => k.startsWith("mAP") || k === "precision" || k === "recall");
  const { run, busy } = useAction();
  const toast = useToast();
  // the training charts (results.png, confusion matrix, PR curves...) sit next to the weights
  const openFolder = async () => {
    const r = await run(() => api.openRunFolder(w.id));
    if (r) toast(r.opened ? "已開啟訓練資料夾" : `訓練圖表在：${r.path}`);
  };
  return (
    <div className="stack" style={{ gap: 2, paddingBottom: 10, borderBottom: "1px solid var(--line)" }}>
      <div className="row">
        <b>{w.name}</b>
        <span className="spacer" />
        {w.job_id != null && (
          <button className="ghost" onClick={openFolder} disabled={busy} title="訓練曲線、混淆矩陣、PR 曲線和批次預覽圖都在這個資料夾">
            開啟訓練資料夾
          </button>
        )}
        <button className="ghost danger" onClick={onDelete}>
          刪除
        </button>
      </div>
      <span className="small muted">
        {plugin?.name ?? w.plugin}，{w.class_names.length} 個類別，{timeAgo(w.created_at)}
        {w.metrics.imported ? "，匯入" : ""}
      </span>
      {metrics.length > 0 && (
        <span className="small">
          {metrics.map(([k, v]) => (
            <span key={k} style={{ marginRight: 14 }}>
              {k} <b>{String(v)}</b>
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

const COCO80 =
  "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";

function ImportWeights({
  project,
  plugins,
  onClose,
  onDone,
}: {
  project: Project;
  plugins: PluginInfo[];
  onClose: () => void;
  onDone: () => void;
}) {
  const real = plugins.filter((p) => p.trainable);
  const [pluginId, setPluginId] = useState(real[0]?.id ?? "");
  const plugin = real.find((p) => p.id === pluginId);
  const [variant, setVariant] = useState(plugin?.variants[0]?.id ?? "");
  const [name, setName] = useState("");
  const [classes, setClasses] = useState(COCO80);
  const [file, setFile] = useState<File | null>(null);
  const { busy, run } = useAction();

  const matched = classes
    .split(/[\n,]/)
    .map((s) => s.trim().toLowerCase())
    .filter((s) => project.classes.some((c) => c.name.trim().toLowerCase() === s));

  const submit = async () => {
    if (!file) return;
    const fd = new FormData();
    fd.append("plugin", pluginId);
    fd.append("variant", variant || plugin?.variants[0]?.id || "");
    fd.append("name", name);
    fd.append("class_names", classes);
    fd.append("file", file);
    if (await run(() => api.importWeights(project.id, fd), "已匯入權重")) onDone();
  };

  return (
    <Modal
      title="匯入現成權重"
      onClose={onClose}
      footer={
        <>
          <button className="ghost" onClick={onClose}>
            取消
          </button>
          <button className="primary" disabled={busy || !file || !pluginId} onClick={submit}>
            匯入
          </button>
        </>
      }
    >
      <p className="small muted">
        例如原作者發布的 COCO 預訓練權重（yolov9-c-converted.pt、rtdetrv2_r18vd_*.pth）。權重的類別會用名稱對應到專案類別。
      </p>
      <label className="field">
        模型
        <select
          value={pluginId}
          onChange={(e) => {
            setPluginId(e.target.value);
            setVariant(real.find((p) => p.id === e.target.value)?.variants[0]?.id ?? "");
          }}
        >
          {real.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        尺寸
        <select value={variant} onChange={(e) => setVariant(e.target.value)}>
          {plugin?.variants.map((v) => (
            <option key={v.id} value={v.id}>
              {v.name}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        權重檔
        <input type="file" accept=".pt,.pth" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <label className="field">
        顯示名稱（選填）
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder={file?.name ?? ""} />
      </label>
      <label className="field">
        權重的類別名稱（依權重的類別順序，預設為 COCO 80 類）
        <textarea rows={4} value={classes} onChange={(e) => setClasses(e.target.value)} />
      </label>
      <p className={`small ${matched.length ? "" : "st-in_progress"}`}>
        {matched.length
          ? `可對應到專案類別：${matched.join("、")}`
          : "沒有任何類別名稱對得上專案類別，自動標註不會產生任何框"}
      </p>
    </Modal>
  );
}
