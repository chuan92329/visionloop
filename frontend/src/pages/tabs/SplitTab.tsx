import { useEffect, useState } from "react";
import { api, type Project, type Split, type SplitStats } from "../../api";
import { Modal, useAction, useToast } from "../../components/ui";

const SPLITS: Split[] = ["train", "valid", "test"];

export default function SplitTab({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const cfg = project.split_config;
  const [pct, setPct] = useState({
    train: Math.round(cfg.train * 100),
    valid: Math.round(cfg.valid * 100),
    test: Math.round(cfg.test * 100),
  });
  const [seed, setSeed] = useState(cfg.seed);
  const [mode, setMode] = useState<"incremental" | "full">("incremental");
  const [stratify, setStratify] = useState(true);
  const [lockAfter, setLockAfter] = useState(false);
  const [stats, setStats] = useState<SplitStats | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [grouping, setGrouping] = useState(false);
  const { busy, run } = useAction();

  useEffect(() => {
    run(() => api.splitStats(project.id)).then((s) => s && setStats(s));
  }, [project.id, run]);

  const sum = pct.train + pct.valid + pct.test;
  const locked = stats?.test_locked ?? project.test_locked;

  const doSplit = async () => {
    const s = await run(() =>
      api.runSplit(project.id, {
        config: { train: pct.train / 100, valid: pct.valid / 100, test: pct.test / 100, seed },
        mode,
        stratify,
        lock_test: lockAfter && !locked ? true : undefined,
      }),
    );
    if (s) {
      setStats(s);
      setWarnings(s.warnings);
      onChanged();
    }
  };

  const toggleLock = async () => {
    const s = await run(() => api.lockTest(project.id, !locked), locked ? "已解除 test 鎖定" : "已鎖定 test 集合");
    if (s) {
      setStats(s);
      onChanged();
    }
  };

  const classNames = project.classes.map((c) => c.name);

  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="two-col">
        <div className="panel stack" data-tour="split">
          <h2>切分設定</h2>
          <p className="small muted">只有「已確認」的圖片會被切分。同一個群組的圖片一定會分在同一個集合。</p>

          <div className="split-sliders">
            {SPLITS.map((s) => (
              <label key={s} className="field">
                {s}（%）
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={pct[s]}
                  disabled={s === "test" && locked}
                  onChange={(e) => setPct({ ...pct, [s]: Math.max(0, Math.min(100, Number(e.target.value))) })}
                />
              </label>
            ))}
          </div>
          <div className="ratio-bar" aria-hidden>
            {SPLITS.map((s) =>
              pct[s] > 0 ? (
                <span key={s} className={`split-${s}`} style={{ width: `${(pct[s] / Math.max(sum, 1)) * 100}%` }}>
                  {s} {pct[s]}%
                </span>
              ) : null,
            )}
          </div>
          {sum !== 100 && <p className="warning error">三個比例加起來要等於 100%（目前 {sum}%）</p>}

          <fieldset style={{ border: "none", padding: 0, margin: 0 }} className="stack">
            <legend className="small muted" style={{ marginBottom: 6 }}>
              分配方式
            </legend>
            <label className="check">
              <input type="radio" checked={mode === "incremental"} onChange={() => setMode("incremental")} />
              <span>
                只分配新確認的圖片 <span className="muted small">（已經分好的圖片不會移動，建議平常使用）</span>
              </span>
            </label>
            <label className="check">
              <input type="radio" checked={mode === "full"} onChange={() => setMode("full")} />
              <span>
                全部重新分配 <span className="muted small">{locked ? "（鎖定的 test 不受影響）" : ""}</span>
              </span>
            </label>
          </fieldset>
          <label className="check">
            <input type="checkbox" checked={stratify} onChange={(e) => setStratify(e.target.checked)} />
            <span>
              依類別分層 <span className="muted small">（讓稀有類別也出現在每個集合）</span>
            </span>
          </label>
          {!locked && (
            <label className="check">
              <input type="checkbox" checked={lockAfter} onChange={(e) => setLockAfter(e.target.checked)} />
              <span>
                切分後鎖定 test <span className="muted small">（之後的模型都用同一批 test 圖片比較）</span>
              </span>
            </label>
          )}
          <label className="field" style={{ maxWidth: 160 }}>
            隨機種子
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <div className="row">
            <button className="primary" onClick={doSplit} disabled={busy || sum !== 100}>
              {mode === "full" ? "重新切分" : "切分新資料"}
            </button>
            <button className="ghost" onClick={() => setGrouping(true)}>
              依檔名自動分群組
            </button>
          </div>
        </div>

        <div className="panel stack">
          <div className="row">
            <h2>目前結果</h2>
            <span className="spacer" />
            {locked ? <span className="pill st-reviewed">test 已鎖定</span> : null}
          </div>
          {stats && (
            <>
              <table className="data">
                <thead>
                  <tr>
                    <th>類別</th>
                    {SPLITS.map((s) => (
                      <th key={s}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <b>圖片數</b>
                    </td>
                    {SPLITS.map((s) => (
                      <td key={s}>
                        <b>{stats.images[s]}</b>
                      </td>
                    ))}
                  </tr>
                  {classNames.map((n) => (
                    <tr key={n}>
                      <td>{n}</td>
                      {SPLITS.map((s) => {
                        const v = stats.boxes[s]?.[n] ?? 0;
                        const someData = stats.images[s] > 0;
                        return (
                          <td key={s} className={someData && v === 0 ? "zero" : ""}>
                            {v}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="small muted">
                已確認但尚未切分：{stats.unassigned_reviewed} 張；尚未確認（不會進入資料集）：{stats.not_reviewed} 張
              </p>
              {warnings.map((w) => (
                <p key={w} className="warning">
                  {w}
                </p>
              ))}
              <div className="row">
                <button onClick={toggleLock} disabled={busy || (!locked && stats.images.test === 0)}>
                  {locked ? "解除 test 鎖定" : "鎖定 test 集合"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="panel stack">
        <h2>匯出資料集</h2>
        <p className="small muted">
          匯出所有已確認且已切分的圖片。YOLO 格式包含 data.yaml 與 images/、labels/ 資料夾；COCO 格式每個集合一個 instances_*.json，有多邊形的物件會附上 segmentation。
        </p>
        {stats && stats.polygons > 0 && stats.boxes_without_polygon > 0 && (
          <p className="small muted">
            {stats.polygons} 個物件有多邊形，{stats.boxes_without_polygon} 個只有框；YOLO 分割格式會把只有框的物件當成矩形。
          </p>
        )}
        <div className="row">
          <a className="btn" href={api.exportUrl(project.id, "yolo")} download>
            下載 YOLO 格式
          </a>
          <a className="btn" href={api.exportUrl(project.id, "yolo-seg")} download>
            下載 YOLO 分割格式
          </a>
          <a className="btn" href={api.exportUrl(project.id, "coco")} download>
            下載 COCO 格式
          </a>
        </div>
      </div>

      {grouping && <AutoGroupModal projectId={project.id} onClose={() => setGrouping(false)} />}
    </div>
  );
}

function AutoGroupModal({ projectId, onClose }: { projectId: number; onClose: () => void }) {
  const [pattern, setPattern] = useState("^(.*?)[_\\-\\s]*\\d+$");
  const [overwrite, setOverwrite] = useState(false);
  const [sample, setSample] = useState("cam01_000123");
  const { busy, run } = useAction();
  const toast = useToast();

  let preview = "";
  try {
    const m = new RegExp(pattern).exec(sample);
    preview = m?.[1] ? `群組：${m[1]}` : "不符合，這張圖不會被分群組";
  } catch {
    preview = "規則運算式有錯誤";
  }

  const apply = async () => {
    const r = await run(() => api.autoGroup(projectId, pattern, overwrite));
    if (r) {
      toast(`已設定 ${r.updated} 張圖片的群組`);
      onClose();
    }
  };

  return (
    <Modal
      title="依檔名自動分群組"
      onClose={onClose}
      footer={
        <>
          <button className="ghost" onClick={onClose}>
            取消
          </button>
          <button className="primary" disabled={busy} onClick={apply}>
            套用
          </button>
        </>
      }
    >
      <p className="small muted">
        從影片擷取的連續影格幾乎一模一樣，如果分散到 train 和 test，test 分數會虛高。預設規則會去掉檔名結尾的編號，把
        cam01_000123、cam01_000124 放進同一個群組「cam01」。
      </p>
      <label className="field">
        規則（第一個括號抓到的文字就是群組名稱）
        <input value={pattern} onChange={(e) => setPattern(e.target.value)} />
      </label>
      <label className="field">
        試試看一個檔名（不含副檔名）
        <input value={sample} onChange={(e) => setSample(e.target.value)} />
      </label>
      <p className="small">{preview}</p>
      <label className="check">
        <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
        覆蓋已經手動設定過的群組
      </label>
    </Modal>
  );
}
