import { useEffect, useMemo, useState } from "react";
import { api, type Job, type PluginInfo, type PluginList, type Project, type SplitStats, type SystemInfo, type Variant, type Weights } from "../../api";
import { Modal, timeAgo, useAction, useToast } from "../../components/ui";
import { DeviceSelect, JobLog, JobRow, sortPlugins, useProjectJobs } from "./jobParts";

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
  const [logJob, setLogJob] = useState<number | null>(null);
  const [importing, setImporting] = useState(false);
  const [jobFilter, setJobFilter] = useState<JobFilter>("all");
  const [showAllJobs, setShowAllJobs] = useState(false);
  const { run } = useAction();
  const { jobs, weights, refresh, active } = useProjectJobs(project.id, onChanged);

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
      <div className="two-col">
        <TrainPanel project={project} plugins={plugins} weights={weights} maxVram={maxVram} onStarted={refresh} />
        <div className="panel stack">
          <div className="row">
            <h2>權重</h2>
            <span className="spacer" />
            <button onClick={() => setImporting(true)}>匯入現成權重</button>
          </div>
          {weights.length === 0 && (
            <p className="muted small">訓練完成的模型會出現在這裡，可以到「輔助標註」頁用它標下一批圖片。也可以匯入官方的 COCO 預訓練權重，直接拿來做第一輪自動標註。</p>
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
            <option value="autolabel">輔助標註</option>
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

// ------------------------------------------------------------------ train
function TrainPanel(props: {
  project: Project;
  plugins: PluginInfo[];
  weights: Weights[];
  maxVram: number;
  onStarted: () => void;
}) {
  return (
    <div className="panel form-body" data-tour="train">
      <h2>訓練</h2>
      <TrainForm {...props} />
    </div>
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
            ? "還沒有多邊形：先畫多邊形，或到「輔助標註」頁用框轉多邊形並確認。"
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
