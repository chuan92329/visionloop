import { useState } from "react";
import { api, type LabelClass, type Project } from "../../api";
import { Modal, useAction } from "../../components/ui";

export default function ClassesTab({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [name, setName] = useState("");
  const [deleting, setDeleting] = useState<{ cls: LabelClass; images: number; boxes: number } | null>(null);
  const { busy, run } = useAction();

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    const names = name
      .split(/[,，]/)
      .map((s) => s.trim())
      .filter(Boolean);
    let ok = true;
    for (const n of names) ok = !!(await run(() => api.createClass(project.id, n))) && ok;
    if (ok) setName("");
    onChanged();
  };

  const update = async (cls: LabelClass, patch: Partial<LabelClass>) => {
    if (await run(() => api.updateClass(cls.id, patch))) onChanged();
  };

  const move = async (idx: number, dir: -1 | 1) => {
    const list = [...project.classes];
    const j = idx + dir;
    if (j < 0 || j >= list.length) return;
    [list[idx], list[j]] = [list[j], list[idx]];
    await run(async () => {
      for (let i = 0; i < list.length; i++) {
        if (list[i].order_index !== i) await api.updateClass(list[i].id, { order_index: i });
      }
    });
    onChanged();
  };

  const askDelete = async (cls: LabelClass) => {
    const usage = await run(() => api.classUsage(cls.id));
    if (usage) setDeleting({ cls, ...usage });
  };

  const doDelete = async () => {
    if (!deleting) return;
    if (await run(() => api.deleteClass(deleting.cls.id).then(() => true), `已刪除類別「${deleting.cls.name}」`)) {
      setDeleting(null);
      onChanged();
    }
  };

  return (
    <div className="two-col">
      <div className="panel">
        <h2 style={{ marginBottom: 6 }}>類別</h2>
        <p className="small muted" style={{ marginBottom: 12 }}>
          順序就是匯出時的類別編號（第一個是 0）。前 9 個類別在標註畫面可以用數字鍵 1–9 切換。
        </p>
        {project.classes.length === 0 && <p className="muted">還沒有類別，先在下面新增。</p>}
        {project.classes.map((c, i) => (
          <div key={c.id} className="class-row">
            <span className="faint small" title="匯出時的類別編號">
              {i}
            </span>
            <input
              type="color"
              className="color-input"
              defaultValue={c.color}
              aria-label={`${c.name} 的顏色`}
              onBlur={(e) => e.target.value !== c.color && update(c, { color: e.target.value })}
            />
            <input
              defaultValue={c.name}
              aria-label="類別名稱"
              onBlur={(e) => e.target.value.trim() && e.target.value !== c.name && update(c, { name: e.target.value })}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
            />
            <div className="row" style={{ gap: 2 }}>
              <button className="ghost icon" aria-label="上移" disabled={i === 0} onClick={() => move(i, -1)}>
                ↑
              </button>
              <button className="ghost icon" aria-label="下移" disabled={i === project.classes.length - 1} onClick={() => move(i, 1)}>
                ↓
              </button>
            </div>
            <button className="ghost danger" onClick={() => askDelete(c)}>
              刪除
            </button>
          </div>
        ))}
        <form className="row" style={{ marginTop: 14 }} onSubmit={add}>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="新類別名稱（可用逗號一次新增多個）" style={{ flex: 1 }} />
          <button disabled={busy || !name.trim()}>新增類別</button>
        </form>
      </div>

      <div className="panel stack small muted">
        <h3 style={{ color: "var(--text)" }}>關於類別名稱</h3>
        <p>
          自動標註時，模型輸出的類別是用「名稱」對應到這裡的類別（不分大小寫）。例如用 COCO 預訓練權重做自動標註，專案裡叫做
          <b> person</b> 的類別就會接收模型的 person 偵測結果，其他名稱對不上的偵測會被忽略。
        </p>
        <p>改名不會影響已經存在的標註；刪除類別會移除它所有的框，受影響的圖片需要重新確認。</p>
      </div>

      {deleting && (
        <Modal
          title={`刪除類別「${deleting.cls.name}」`}
          onClose={() => setDeleting(null)}
          footer={
            <>
              <button className="ghost" onClick={() => setDeleting(null)}>
                取消
              </button>
              <button className="danger" onClick={doDelete} disabled={busy}>
                刪除類別
              </button>
            </>
          }
        >
          {deleting.boxes > 0 ? (
            <p>
              這個類別目前有 <b>{deleting.boxes}</b> 個框，分布在 <b>{deleting.images}</b> 張圖片上。刪除後這些框會一併移除，
              受影響的已確認圖片會改回「編輯中」，需要重新確認。舊的標註仍保留在每張圖的版本紀錄裡。
            </p>
          ) : (
            <p>這個類別目前沒有任何標註。</p>
          )}
        </Modal>
      )}
    </div>
  );
}
