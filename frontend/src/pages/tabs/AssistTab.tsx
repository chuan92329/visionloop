// 輔助標註: everything that labels images for you. A first pass from a few confirmed examples
// (Linear Probe), labeling with trained weights or a model that needs no training, and turning
// boxes into polygons. Training the final model lives in the 訓練與權重 tab.

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Job, type PluginInfo, type Project, type Status, type Weights } from "../../api";
import { useAction, useToast } from "../../components/ui";
import {
  DeviceSelect, isInitialLabeling, isLive, JobLog, JobRow, OutlineOptions, sortPlugins, StatusChecks, useProjectJobs,
  type PolygonDetail,
} from "./jobParts";

const RECENT_JOBS = 3;
const ENOUGH_EXAMPLES = 3; // per class, below this the first pass is a guess

export default function AssistTab({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [plugins, setPlugins] = useState<PluginInfo[] | null>(null);
  const [logJob, setLogJob] = useState<number | null>(null);
  const { jobs, weights, refresh, active } = useProjectJobs(project.id, onChanged);
  const { run } = useAction();
  const nav = useNavigate();

  useEffect(() => {
    run(async () => {
      setPlugins((await api.plugins()).plugins);
      await refresh();
    });
  }, [run, refresh]);

  if (!plugins) return null;

  const initial = sortPlugins(plugins.filter((p) => p.available && isInitialLabeling(p)));
  // what is running, then the last few that finished
  const mine = jobs.filter((j) => j.kind === "autolabel");
  const running = mine.filter(isLive);
  const shown = [...running, ...mine.filter((j) => !isLive(j))].slice(0, Math.max(RECENT_JOBS, running.length));

  const deleteJob = async (j: Job) => {
    if (!confirm(`刪除任務 #${j.id} 的紀錄？標註不受影響。`)) return;
    await run(() => api.deleteJob(j.id), "已刪除任務紀錄");
    if (logJob === j.id) setLogJob(null);
    refresh();
  };

  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="three-col">
        <div className="panel" data-tour="probe">
          <h2>初步標註</h2>
          {initial.length > 0 ? (
            <InitialLabelForm project={project} plugins={initial} onStarted={refresh} />
          ) : (
            <p className="warning small">初步標註（Linear Probe）還不能用，原因列在「訓練與權重」頁的「模型外掛」。</p>
          )}
        </div>
        <AutoLabelForm project={project} plugins={plugins} weights={weights} onStarted={refresh} onChanged={onChanged} />
        <OutlineForm project={project} plugins={plugins} onStarted={refresh} />
      </div>

      <div className="panel stack">
        <div className="row" style={{ flexWrap: "wrap" }}>
          <h2>進度</h2>
          {active && <span className="small muted">每 1.5 秒自動更新</span>}
          <span className="spacer" />
          {project.stats.auto > 0 && (
            <button className={active ? "" : "primary"} onClick={() => nav(`/projects/${project.id}/images?status=auto`)}>
              審核自動標註的圖片（{project.stats.auto}）
            </button>
          )}
        </div>
        {mine.length === 0 && <p className="muted">還沒有執行過。標註完成後，圖片會變成「自動標註」，等你審核。</p>}
        <div>
          {shown.map((j) => (
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
        {mine.length > shown.length && <p className="small muted">更早的紀錄在「訓練與權重」頁的「任務」。</p>}
      </div>

      {logJob !== null && <JobLog jobId={logJob} onClose={() => setLogJob(null)} />}
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
      <h2>用模型自動標註</h2>
      <p className="small muted">用你訓練好的權重，或不用訓練的模型（例如用文字找物）。</p>
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
      {needsWeights && <p className="warning small">需要權重：先到「訓練與權重」頁訓練，或匯入權重。</p>}
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
  const [detail, setDetail] = useState<PolygonDetail>("normal");
  const [overwrite, setOverwrite] = useState(false);
  const [device, setDevice] = useState("auto");
  const { busy, run } = useAction();

  if (!plugin)
    return (
      <div className="panel">
        <h2>框轉多邊形</h2>
        <p className="warning small">SAM 2 目前無法使用，原因列在「訓練與權重」頁的「模型外掛」。</p>
      </div>
    );
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
      <p className="small muted">由 SAM 2 把已有的框描成多邊形，不會找新物件。只轉某幾張：在「圖片」頁選取後按「框轉多邊形」。</p>
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
      <OutlineOptions
        plugin={plugin}
        variant={variantId}
        onVariant={setVariant}
        detail={detail}
        onDetail={setDetail}
        overwrite={overwrite}
        onOverwrite={setOverwrite}
      />
      <StatusChecks project={project} choices={["auto", "in_progress", "reviewed"]} value={targets} onChange={setTargets} />
      <DeviceSelect value={device} onChange={setDevice} />
      {targets.includes("reviewed") && <p className="warning small">已確認的圖片轉換後要重新確認。</p>}
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || count === 0}>
          轉換 {count} 張圖片的框
        </button>
      </div>
    </div>
  );
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
      <p className="small muted">用你已確認的少量範例訓練幾秒鐘，幫其餘圖片先標一輪（Linear Probe）。每多確認一批，再跑一次會更準。</p>
      {exemplars &&
        (noExamples ? (
          <p className="warning small">先確認幾個物件，每類 3～5 個。</p>
        ) : (
          <div className="field-group">
            <span className="small muted">已確認的範例</span>
            {project.classes.map((c) => {
              const n = exemplars[String(c.id)] ?? 0;
              return (
                <div key={c.id} className="row small">
                  <span className="swatch" style={{ background: c.color }} />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{c.name}</span>
                  <span className="spacer" />
                  <span className={n >= ENOUGH_EXAMPLES ? "" : "st-in_progress"}>
                    {n} 個{n >= ENOUGH_EXAMPLES ? "" : n ? "，建議再標幾個" : "，這一類不會被標到"}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
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
        信心門檻：{conf.toFixed(2)}
        <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConfInput(Number(e.target.value))} />
      </label>
      <StatusChecks project={project} choices={["unlabeled", "auto"]} value={targets} onChange={setTargets} />
      <details className="advanced">
        <summary className="small muted">進階設定（探針、運算裝置）</summary>
        <div className="form-body">
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
          <label className="field">
            探針
            <select value={head} onChange={(e) => setHead(e.target.value as "linear" | "mlp")}>
              <option value="linear">線性層（建議）</option>
              <option value="mlp">小型 MLP</option>
            </select>
          </label>
          <DeviceSelect value={device} onChange={setDevice} />
        </div>
      </details>
      <div className="row actions">
        <button className="primary" onClick={start} disabled={busy || noExamples || count === 0}>
          開始初步標註（{count} 張）
        </button>
      </div>
    </>
  );
}
