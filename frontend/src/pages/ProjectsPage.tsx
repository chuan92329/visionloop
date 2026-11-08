import { useEffect, useState } from "react";
import { useTour } from "../components/Tour";
import { Link, useNavigate } from "react-router-dom";
import { api, type Project } from "../api";
import { Modal, PipelineBar, timeAgo, useAction } from "../components/ui";

export function TopBar({ children }: { children?: React.ReactNode }) {
  const tour = useTour();
  return (
    <header className="topbar">
      <Link to="/" className="brand">
        <img src="/favicon.svg" alt="" />
        VisionLoop
      </Link>
      {children}
      {!children && <span className="spacer" />}
      <button className="ghost" onClick={tour.start} title="一步一步介紹怎麼使用">
        導覽
      </button>
    </header>
  );
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [creating, setCreating] = useState(false);
  const { run } = useAction();

  useEffect(() => {
    run(() => api.projects()).then((p) => p && setProjects(p));
  }, [run]);

  return (
    <>
      <TopBar>
        <span className="spacer" />
        <button className="primary" data-tour="new-project" onClick={() => setCreating(true)}>
          新增專案
        </button>
      </TopBar>
      <main className="page">
        <div className="row" style={{ marginBottom: 18, alignItems: "baseline" }}>
          <h1>專案</h1>
          {projects && <span className="muted">{projects.length} 個</span>}
        </div>

        {projects && projects.length === 0 && (
          <div className="empty-state">
            <h2>還沒有任何專案</h2>
            <p>一個專案對應一個偵測任務：一組類別、一批圖片，以及在上面訓練出來的模型。</p>
            <button className="primary" onClick={() => setCreating(true)}>
              建立第一個專案
            </button>
          </div>
        )}

        {projects && projects.length > 0 && (
          <div className="project-list">
            {projects.map((p) => (
              <Link key={p.id} to={`/projects/${p.id}`} className="project-row">
                {p.cover_image_id ? (
                  <img className="cover" src={api.thumbUrl(p.cover_image_id)} alt="" loading="lazy" />
                ) : (
                  <div className="cover empty">沒有圖片</div>
                )}
                <div style={{ minWidth: 0 }}>
                  <h2>{p.name}</h2>
                  <p className="muted small">
                    {p.classes.length} 個類別，{p.stats.total} 張圖片，{p.stats.boxes} 個框，建立於
                    {timeAgo(p.created_at)}
                  </p>
                </div>
                <div className="stack" style={{ gap: 6 }}>
                  <PipelineBar stats={p.stats} compact />
                  <span className="small muted">
                    已確認 {p.stats.reviewed} / {p.stats.total}
                    {p.stats.auto > 0 && <span className="st-auto">，{p.stats.auto} 張待審核</span>}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </main>
      {creating && <CreateProject onClose={() => setCreating(false)} />}
    </>
  );
}

function CreateProject({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [classes, setClasses] = useState("");
  const { busy, run } = useAction();
  const nav = useNavigate();
  const list = classes
    .split(/[\n,，]/)
    .map((s) => s.trim())
    .filter(Boolean);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const p = await run(() => api.createProject(name, desc, list));
    if (p) nav(`/projects/${p.id}`);
  };

  return (
    <Modal title="新增專案" onClose={onClose}>
      <form className="stack" onSubmit={submit}>
        <label className="field">
          專案名稱
          <input autoFocus required value={name} onChange={(e) => setName(e.target.value)} placeholder="例如：產線瑕疵檢測" />
        </label>
        <label className="field">
          說明（選填）
          <input value={desc} onChange={(e) => setDesc(e.target.value)} />
        </label>
        <label className="field">
          類別（每行一個，或用逗號分隔，之後也可以再改）
          <textarea rows={4} value={classes} onChange={(e) => setClasses(e.target.value)} placeholder={"scratch\ndent\nstain"} />
        </label>
        {list.length > 0 && <p className="small muted">將建立 {list.length} 個類別：{list.join("、")}</p>}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button type="button" className="ghost" onClick={onClose}>
            取消
          </button>
          <button className="primary" disabled={busy || !name.trim()}>
            建立專案
          </button>
        </div>
      </form>
    </Modal>
  );
}
