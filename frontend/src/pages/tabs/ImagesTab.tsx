import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, STATUS_LABEL, type ImageInfo, type ImageQuery, type PluginInfo, type Project, type Status } from "../../api";
import { Modal, Progress, useAction, useToast } from "../../components/ui";
import { queryFromSearch, searchFromQuery } from "../../query";
import { OutlineOptions, type PolygonDetail } from "./jobParts";

const PAGE = 120;
const UPLOAD_CHUNK = 16;
const STATUS_FILTERS: (Status | "all")[] = ["all", "unlabeled", "auto", "in_progress", "reviewed"];

export default function ImagesTab({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [sp, setSp] = useSearchParams();
  const query = queryFromSearch(sp);
  const queryKey = sp.toString();
  const [items, setItems] = useState<ImageInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [upload, setUpload] = useState<{ done: number; total: number } | null>(null);
  const [groupModal, setGroupModal] = useState(false);
  const [outlineModal, setOutlineModal] = useState(false);
  const { busy, run } = useAction();
  const toast = useToast();
  const nav = useNavigate();

  const load = useCallback(
    async (offset = 0) => {
      const page = await run(() => api.images(project.id, { ...queryFromSearch(new URLSearchParams(queryKey)), offset, limit: PAGE }));
      if (!page) return;
      setTotal(page.total);
      setItems((prev) => (offset === 0 ? page.items : [...prev, ...page.items]));
    },
    [project.id, queryKey, run],
  );

  useEffect(() => {
    setSelected(new Set());
    load(0);
  }, [load]);

  const setQuery = (patch: Partial<ImageQuery>) => {
    const next = { ...query, ...patch };
    setSp(new URLSearchParams(searchFromQuery(next).slice(1)), { replace: true });
  };

  // ---------------------------------------------------------------- upload
  const doUpload = async (files: File[]) => {
    const images = files.filter((f) => /\.(jpe?g|png|bmp|webp|tiff?|heic|heif|dng)$/i.test(f.name));
    if (!images.length) {
      toast("沒有可上傳的圖片（支援 jpg、png、bmp、webp、tif、heic、dng）", "error");
      return;
    }
    setUpload({ done: 0, total: images.length });
    let created = 0;
    const skipped: string[] = [];
    try {
      for (let i = 0; i < images.length; i += UPLOAD_CHUNK) {
        const r = await api.upload(project.id, images.slice(i, i + UPLOAD_CHUNK));
        created += r.created.length;
        r.skipped.forEach((s) => skipped.push(`${s.filename}：${s.reason}`));
        setUpload({ done: Math.min(i + UPLOAD_CHUNK, images.length), total: images.length });
      }
      toast(`已上傳 ${created} 張圖片${skipped.length ? `，略過 ${skipped.length} 張` : ""}`);
      if (skipped.length) toast(skipped.slice(0, 5).join("\n") + (skipped.length > 5 ? "…" : ""), "error");
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), "error");
    } finally {
      setUpload(null);
      load(0);
      onChanged();
    }
  };

  // ------------------------------------------------------------- selection
  const toggle = (id: number) =>
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const removeSelected = async () => {
    if (!confirm(`確定刪除 ${selected.size} 張圖片和它們的標註？此動作無法復原。`)) return;
    if (await run(() => api.deleteImages(project.id, [...selected]), `已刪除 ${selected.size} 張圖片`)) {
      setSelected(new Set());
      load(0);
      onChanged();
    }
  };

  const startAnnotating = async () => {
    const ids = await run(() => api.imageIds(project.id, query));
    if (!ids?.length) return toast("目前的篩選條件下沒有圖片", "error");
    nav(`/projects/${project.id}/annotate/${ids[0]}${searchFromQuery(query)}`);
  };

  const statusValue = query.status?.length === 1 ? query.status[0] : "all";
  const href = (id: number) => `/projects/${project.id}/annotate/${id}${searchFromQuery(query)}`;

  return (
    <div className="stack" style={{ gap: 16 }}>
      <Dropzone onFiles={doUpload} uploading={upload} />

      <div className="filters">
        <div className="seg" role="group" aria-label="依狀態篩選" data-tour="status-filter">
          {STATUS_FILTERS.map((s) => (
            <button
              key={s}
              className={statusValue === s ? "on" : ""}
              onClick={() => setQuery({ status: s === "all" ? undefined : [s] })}
            >
              {s === "all" ? "全部" : STATUS_LABEL[s]}
            </button>
          ))}
        </div>
        <select
          aria-label="依資料集篩選"
          value={query.split ?? ""}
          onChange={(e) => setQuery({ split: (e.target.value || undefined) as ImageQuery["split"] })}
        >
          <option value="">所有集合</option>
          <option value="train">train</option>
          <option value="valid">valid</option>
          <option value="test">test</option>
          <option value="none">尚未切分</option>
        </select>
        <select
          aria-label="排序"
          value={`${query.sort ?? "id"}:${query.order ?? "asc"}`}
          onChange={(e) => {
            const [sort, order] = e.target.value.split(":");
            setQuery({ sort: sort as ImageQuery["sort"], order: order as "asc" | "desc" });
          }}
        >
          <option value="id:asc">上傳順序</option>
          <option value="filename:asc">檔名</option>
          <option value="min_score:asc">最低信心分數優先</option>
          <option value="updated_at:desc">最近修改</option>
          <option value="box_count:desc">框數最多</option>
        </select>
        <input
          type="search"
          placeholder="搜尋檔名"
          defaultValue={query.q ?? ""}
          onKeyDown={(e) => e.key === "Enter" && setQuery({ q: (e.target as HTMLInputElement).value || undefined })}
          style={{ width: 160 }}
        />
        <span className="spacer" />
        <span className="muted small">{total} 張</span>
        <button className="primary" data-tour="start-label" onClick={startAnnotating} disabled={!total || busy}>
          {query.status?.[0] === "auto" ? "開始審核" : "開始標註"}
        </button>
      </div>

      {selected.size > 0 && (
        <div className="row panel" style={{ padding: "8px 12px" }}>
          <span>已選 {selected.size} 張</span>
          <button className="ghost" onClick={() => setSelected(new Set(items.map((i) => i.id)))}>
            全選這頁
          </button>
          <button className="ghost" onClick={() => setSelected(new Set())}>
            取消選取
          </button>
          <span className="spacer" />
          <button onClick={() => setOutlineModal(true)}>框轉多邊形</button>
          <button onClick={() => setGroupModal(true)}>設定群組</button>
          <button className="danger" onClick={removeSelected}>
            刪除
          </button>
        </div>
      )}

      {items.length === 0 && !busy && (
        <div className="empty-state">
          {project.stats.total === 0 ? (
            <>
              <h2>先上傳一些圖片</h2>
              <p>把圖片或整個資料夾拖到上面的區塊。重複的圖片會自動略過。</p>
            </>
          ) : (
            <>
              <h2>沒有符合條件的圖片</h2>
              <button onClick={() => setSp(new URLSearchParams(), { replace: true })}>清除篩選</button>
            </>
          )}
        </div>
      )}

      <div className="grid">
        {items.map((img) => (
          <Tile key={img.id} img={img} href={href(img.id)} selected={selected.has(img.id)} onToggle={() => toggle(img.id)} />
        ))}
      </div>

      {items.length < total && (
        <div className="row" style={{ justifyContent: "center" }}>
          <button onClick={() => load(items.length)} disabled={busy}>
            載入更多（還有 {total - items.length} 張）
          </button>
        </div>
      )}

      {outlineModal && (
        <OutlineModal
          projectId={project.id}
          images={items.filter((i) => selected.has(i.id))}
          onClose={() => setOutlineModal(false)}
          onDone={() => {
            setOutlineModal(false);
            setSelected(new Set());
          }}
        />
      )}

      {groupModal && (
        <GroupModal
          projectId={project.id}
          ids={[...selected]}
          onClose={() => setGroupModal(false)}
          onDone={() => {
            setGroupModal(false);
            load(0);
          }}
        />
      )}
    </div>
  );
}

function Tile({
  img,
  href,
  selected,
  onToggle,
}: {
  img: ImageInfo;
  href: string;
  selected: boolean;
  onToggle: () => void;
}) {
  return (
    <Link to={href} className={`tile ${selected ? "selected" : ""}`} title={`${img.filename}（${STATUS_LABEL[img.status]}）`}>
      <img src={api.thumbUrl(img.id)} alt={img.filename} loading="lazy" />
      <input
        type="checkbox"
        className="sel"
        checked={selected}
        aria-label={`選取 ${img.filename}`}
        onClick={(e) => e.stopPropagation()}
        onChange={onToggle}
      />
      <div className="meta">
        <span className="name">{img.filename}</span>
        {img.min_score !== null && img.status === "auto" && (
          <span className="st-auto" title="最低信心分數">
            {img.min_score.toFixed(2)}
          </span>
        )}
        <span className="faint">{img.box_count}</span>
        {img.split && <span className={`split-tag ${img.split}`}>{img.split}</span>}
      </div>
      <span className={`edge bg-${img.status}`} />
    </Link>
  );
}

function Dropzone({
  onFiles,
  uploading,
}: {
  onFiles: (files: File[]) => void;
  uploading: { done: number; total: number } | null;
}) {
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const dirRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    dirRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const files = await filesFromDrop(e.dataTransfer);
    onFiles(files);
  };

  if (uploading) {
    return (
      <div className="dropzone stack" style={{ gap: 8 }}>
        <span>
          上傳中 {uploading.done} / {uploading.total}
        </span>
        <Progress value={uploading.done / uploading.total} />
      </div>
    );
  }
  return (
    <div
      data-tour="upload"
      className={`dropzone ${over ? "over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <div className="row" style={{ justifyContent: "center", flexWrap: "wrap" }}>
        <span>把圖片或資料夾拖到這裡，或</span>
        <button onClick={() => fileRef.current?.click()}>選擇圖片</button>
        <button onClick={() => dirRef.current?.click()}>選擇資料夾</button>
      </div>
      <input ref={fileRef} type="file" multiple accept="image/*,.tif,.tiff,.heic,.heif,.dng" hidden onChange={(e) => onFiles([...(e.target.files ?? [])])} />
      <input ref={dirRef} type="file" multiple hidden onChange={(e) => onFiles([...(e.target.files ?? [])])} />
    </div>
  );
}

async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries = [...dt.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  if (!entries.length) return [...dt.files];
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      out.push(await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej)));
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        for (const e of batch) await walk(e);
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

function OutlineModal({
  projectId,
  images,
  onClose,
  onDone,
}: {
  projectId: number;
  images: ImageInfo[];
  onClose: () => void;
  onDone: () => void;
}) {
  const [plugin, setPlugin] = useState<PluginInfo | null | undefined>(undefined);
  const [variant, setVariant] = useState("");
  const [detail, setDetail] = useState<PolygonDetail>("normal");
  const [overwrite, setOverwrite] = useState(false);
  const { busy, run } = useAction();

  useEffect(() => {
    api
      .plugins()
      .then((r) => {
        const found = r.plugins.find((p) => p.prompt_mode === "boxes" && p.available) ?? null;
        setPlugin(found);
        setVariant(found?.variants[1]?.id ?? found?.variants[0]?.id ?? "");
      })
      .catch(() => setPlugin(null));
  }, []);

  const withBoxes = images.filter((i) => i.box_count > 0);
  const reviewed = withBoxes.filter((i) => i.status === "reviewed").length;

  const start = async () => {
    if (!plugin) return;
    const job = await run(
      () =>
        api.autolabel(projectId, {
          plugin: plugin.id,
          variant,
          confidence: 0,
          statuses: [],
          image_ids: withBoxes.map((i) => i.id),
          polygon_detail: detail,
          overwrite_polygons: overwrite,
          device: "auto",
        }),
      "已開始轉換，可到「輔助標註」頁查看進度",
    );
    if (job) onDone();
  };

  return (
    <Modal
      title="把框轉成多邊形"
      onClose={onClose}
      footer={
        <>
          <button className="ghost" onClick={onClose}>
            取消
          </button>
          <button className="primary" onClick={start} disabled={busy || !plugin || withBoxes.length === 0}>
            轉換 {withBoxes.length} 張圖片
          </button>
        </>
      }
    >
      {plugin === null && <p className="warning small">SAM 2 目前無法使用，請確認模型環境已安裝。</p>}
      <p className="muted small">由 SAM 2 依照既有的框描出物件輪廓。轉換失敗的框會保留原狀。</p>
      {plugin && (
        <OutlineOptions
          plugin={plugin}
          variant={variant}
          onVariant={setVariant}
          detail={detail}
          onDetail={setDetail}
          overwrite={overwrite}
          onOverwrite={setOverwrite}
        />
      )}
      {withBoxes.length < images.length && <p className="small muted">選取的圖片中有 {images.length - withBoxes.length} 張沒有框，會略過。</p>}
      {reviewed > 0 && (
        <p className="warning small">其中 {reviewed} 張已確認的圖片轉換後會變回「自動標註」，要重新確認才會進入訓練資料。原本的框保留在版本紀錄裡，可以還原。</p>
      )}
    </Modal>
  );
}

function GroupModal({
  projectId,
  ids,
  onClose,
  onDone,
}: {
  projectId: number;
  ids: number[];
  onClose: () => void;
  onDone: () => void;
}) {
  const [group, setGroup] = useState("");
  const { busy, run } = useAction();
  const save = async () => {
    const ok = await run(() => api.setGroup(projectId, ids, group.trim() || null), group.trim() ? `已把 ${ids.length} 張圖片設為群組「${group.trim()}」` : `已清除 ${ids.length} 張圖片的群組`);
    if (ok) onDone();
  };
  return (
    <Modal
      title="設定群組"
      onClose={onClose}
      footer={
        <>
          <button className="ghost" onClick={onClose}>
            取消
          </button>
          <button className="primary" onClick={save} disabled={busy}>
            套用到 {ids.length} 張圖片
          </button>
        </>
      }
    >
      <p className="muted small">
        同一個群組的圖片在切分時一定會在同一個集合，避免同一段影片或同一個場景同時出現在 train 和 test。留空代表清除群組。
      </p>
      <input autoFocus value={group} onChange={(e) => setGroup(e.target.value)} placeholder="例如：camera_03_morning" />
    </Modal>
  );
}
