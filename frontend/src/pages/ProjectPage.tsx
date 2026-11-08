import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, type Project, type Status } from "../api";
import { Modal, PipelineBar, useAction } from "../components/ui";
import { TopBar } from "./ProjectsPage";
import ImagesTab from "./tabs/ImagesTab";
import ClassesTab from "./tabs/ClassesTab";
import SplitTab from "./tabs/SplitTab";
import ModelsTab from "./tabs/ModelsTab";

const TABS = [
  { id: "images", label: "圖片" },
  { id: "classes", label: "類別" },
  { id: "split", label: "資料切分與匯出" },
  { id: "models", label: "自動標註與訓練" },
] as const;

export default function ProjectPage() {
  const { projectId, tab = "images" } = useParams();
  const pid = Number(projectId);
  const [project, setProject] = useState<Project | null>(null);
  const [editing, setEditing] = useState(false);
  const [, setSearch] = useSearchParams();
  const nav = useNavigate();
  const { run } = useAction();

  const reload = useCallback(async () => {
    const p = await run(() => api.project(pid));
    if (p) setProject(p);
  }, [pid, run]);

  useEffect(() => {
    reload();
  }, [reload]);

  if (!project) {
    return <TopBar />;
  }

  const pickStatus = (s: Status) => {
    if (tab !== "images") nav(`/projects/${pid}/images?status=${s}`);
    else setSearch({ status: s });
  };

  return (
    <>
      <TopBar>
        <span className="faint">/</span>
        <span>{project.name}</span>
        <span className="spacer" />
        <button className="ghost" onClick={() => setEditing(true)}>
          專案設定
        </button>
      </TopBar>
      <main className="page">
        <div className="stack" style={{ gap: 18 }}>
          <div>
            <h1>{project.name}</h1>
            {project.description && <p className="muted">{project.description}</p>}
          </div>
          <PipelineBar stats={project.stats} onPick={pickStatus} />
        </div>

        <nav className="tabs">
          {TABS.map((t) => (
            <Link key={t.id} to={`/projects/${pid}/${t.id}`} className={tab === t.id ? "active" : ""}>
              {t.label}
            </Link>
          ))}
        </nav>

        {tab === "images" && <ImagesTab project={project} onChanged={reload} />}
        {tab === "classes" && <ClassesTab project={project} onChanged={reload} />}
        {tab === "split" && <SplitTab project={project} onChanged={reload} />}
        {tab === "models" && <ModelsTab project={project} onChanged={reload} />}
      </main>
      {editing && <ProjectSettings project={project} onClose={() => setEditing(false)} onSaved={reload} />}
    </>
  );
}

function ProjectSettings({
  project,
  onClose,
  onSaved,
}: {
  project: Project;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(project.name);
  const [desc, setDesc] = useState(project.description);
  const [confirm, setConfirm] = useState("");
  const { busy, run } = useAction();
  const nav = useNavigate();

  const save = async () => {
    if (await run(() => api.updateProject(project.id, { name, description: desc }), "已儲存專案設定")) {
      onSaved();
      onClose();
    }
  };
  const remove = async () => {
    await run(() => api.deleteProject(project.id), "專案已刪除");
    nav("/");
  };

  return (
    <Modal title="專案設定" onClose={onClose}>
      <label className="field">
        專案名稱
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label className="field">
        說明
        <input value={desc} onChange={(e) => setDesc(e.target.value)} />
      </label>
      <div className="row" style={{ justifyContent: "flex-end" }}>
        <button className="ghost" onClick={onClose}>
          取消
        </button>
        <button className="primary" disabled={busy || !name.trim()} onClick={save}>
          儲存
        </button>
      </div>
      <hr style={{ border: "none", borderTop: "1px solid var(--line)", width: "100%" }} />
      <div className="stack" style={{ gap: 8 }}>
        <h3>刪除專案</h3>
        <p className="small muted">
          會永久刪除 {project.stats.total} 張圖片、所有標註、版本紀錄與訓練出的權重，無法復原。請輸入專案名稱確認。
        </p>
        <div className="row">
          <input value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder={project.name} style={{ flex: 1 }} />
          <button className="danger" disabled={confirm !== project.name || busy} onClick={remove}>
            永久刪除
          </button>
        </div>
      </div>
    </Modal>
  );
}
