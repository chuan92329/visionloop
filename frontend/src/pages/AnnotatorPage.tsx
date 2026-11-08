import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, ApiError, type ImageInfo, type LabelClass, type Project, type Revision } from "../api";
import AnnotCanvas, { type CanvasHandle, type SamPoint } from "../annotator/AnnotCanvas";
import { keepInside, newUid, polygonBBox, sameBoxes, translatePolygon, type EBox, type Pt, type Rect } from "../annotator/geometry";
import { History } from "../annotator/history";
import { StatusPill, timeAgo, useToast } from "../components/ui";
import LabelingTips from "../components/LabelingTips";
import { queryFromSearch, searchFromQuery } from "../query";

const ACTION_LABEL: Record<Revision["action"], string> = {
  save: "儲存",
  review: "確認",
  auto_label: "自動標註",
  revert: "還原",
  clear: "清除自動標註",
};

type Tool = "select" | "draw" | "polygon" | "sam";
type SamState = { phase: "loading" | "ready" | "busy" | "empty" | "error"; message?: string };

export default function AnnotatorPage() {
  const { projectId, imageId } = useParams();
  const pid = Number(projectId);
  const iid = Number(imageId);
  const [sp] = useSearchParams();
  const queryString = searchFromQuery(queryFromSearch(sp));
  const nav = useNavigate();
  const toast = useToast();
  const canvas = useRef<CanvasHandle>(null);

  const [project, setProject] = useState<Project | null>(null);
  const [queue, setQueue] = useState<number[]>([]);
  const [image, setImage] = useState<ImageInfo | null>(null);
  const [boxes, setBoxes] = useState<EBox[]>([]);
  const [saved, setSaved] = useState<EBox[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("draw");
  const [activeClass, setActiveClass] = useState<number | null>(null);
  const [lowThreshold, setLowThreshold] = useState(0.5);
  const [showLabels, setShowLabels] = useState(true);
  const [sortByScore, setSortByScore] = useState(false);
  const [revisions, setRevisions] = useState<Revision[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const history = useRef(new History<EBox[]>());
  const [, force] = useState(0);
  // click-to-segment
  const [samPoints, setSamPoints] = useState<SamPoint[]>([]);
  const [samBox, setSamBox] = useState<Rect | null>(null);
  const [samPreview, setSamPreview] = useState<Pt[] | null>(null);
  const [samState, setSamState] = useState<SamState>({ phase: "loading" });
  // cut objects already outlined out of the result: a partly covered object gets its visible part
  const [samExclude, setSamExclude] = useState(true);
  const [tipsOpen, setTipsOpen] = useState(false);
  const samSeq = useRef(0); // only the answer to the latest prompt is shown

  const classes = useMemo(() => new Map((project?.classes ?? []).map((c) => [c.id, c])), [project]);
  const dirty = !sameBoxes(boxes, saved);
  const pos = queue.indexOf(iid);

  // ------------------------------------------------------------------ load
  useEffect(() => {
    api.project(pid).then((p) => {
      setProject(p);
      setActiveClass((c) => c ?? p.classes[0]?.id ?? null);
    });
  }, [pid]);

  useEffect(() => {
    api.imageIds(pid, queryFromSearch(new URLSearchParams(queryString))).then(setQueue);
  }, [pid, queryString]);

  const loadImage = useCallback(async () => {
    try {
      const s = await api.annotations(iid);
      const eb = s.boxes.map((b) => ({ ...b, uid: newUid() }));
      setImage(s.image);
      setBoxes(eb);
      setSaved(eb);
      setSelected(null);
      setRevisions(null);
      history.current.clear();
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), "error");
    }
  }, [iid, toast]);

  useEffect(() => {
    loadImage();
  }, [loadImage]);

  // warn before closing the tab with unsaved work
  useEffect(() => {
    const h = (e: BeforeUnloadEvent) => {
      if (dirty) e.preventDefault();
    };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [dirty]);

  // ---------------------------------------------------------------- edits
  const commit = useCallback(
    (next: EBox[]) => {
      history.current.push(boxes);
      setBoxes(next);
      force((n) => n + 1);
    },
    [boxes],
  );

  const undo = () => {
    const prev = history.current.undo(boxes);
    if (prev) {
      setBoxes(prev);
      force((n) => n + 1);
    }
  };
  const redo = () => {
    const next = history.current.redo(boxes);
    if (next) {
      setBoxes(next);
      force((n) => n + 1);
    }
  };

  const createBox = (r: Rect) => {
    if (activeClass === null) {
      toast("請先在「類別」頁新增至少一個類別", "error");
      return;
    }
    const b: EBox = { uid: newUid(), class_id: activeClass, ...r, source: "manual", score: null };
    commit([...boxes, b]);
    setSelected(b.uid);
  };

  const createPolygon = (points: Pt[], select = true) => {
    if (activeClass === null) {
      toast("請先在「類別」頁新增至少一個類別", "error");
      return false;
    }
    const b: EBox = { uid: newUid(), class_id: activeClass, ...polygonBBox(points), polygon: points, source: "manual", score: null };
    commit([...boxes, b]);
    setSelected(select ? b.uid : null);
    return true;
  };

  // ------------------------------------------------------- click-to-segment
  const clearSam = () => {
    samSeq.current++;
    setSamPoints([]);
    setSamBox(null);
    setSamPreview(null);
    setSamState((s) => (s.phase === "loading" || s.phase === "error" ? s : { phase: "ready" }));
  };

  // load the model and encode the image as soon as the tool is picked, so the first click is quick
  const loadedId = image?.id;
  useEffect(() => {
    samSeq.current++;
    setSamPoints([]);
    setSamBox(null);
    setSamPreview(null);
    if (tool !== "sam" || loadedId === undefined) return;
    let current = true;
    setSamState({ phase: "loading" });
    // the first use downloads the model: say so, or the wait looks like a hang
    api
      .samInfo()
      .then((i) => {
        if (current && !i.cached)
          setSamState((s) =>
            s.phase === "loading"
              ? { phase: "loading", message: `第一次使用，正在下載 ${i.model.split("/").pop()}${i.size_mb ? `（約 ${i.size_mb} MB）` : ""}…` }
              : s,
          );
      })
      .catch(() => {});
    api
      .samPrepare(loadedId)
      .then(() => current && setSamState({ phase: "ready" }))
      .catch((e) => current && setSamState({ phase: "error", message: e instanceof Error ? e.message : String(e) }));
    return () => {
      current = false;
    };
  }, [tool, loadedId]);

  const runSam = (points: SamPoint[], box: Rect | null, exclude = samExclude) => {
    const seq = ++samSeq.current;
    if (!image || (!points.length && !box)) {
      setSamPreview(null);
      setSamState({ phase: "ready" });
      return;
    }
    setSamState({ phase: "busy" });
    const outlines = exclude ? boxes.filter((b) => b.polygon).map((b) => b.polygon as Pt[]) : [];
    api
      .samSegment(image.id, { points, box: box && [box.x, box.y, box.w, box.h], exclude: outlines })
      .then((r) => {
        if (seq !== samSeq.current) return;
        setSamPreview(r.polygon);
        // everything SAM found is already annotated: usually an earlier outline also covers this object
        const covered = !r.polygon && r.removed > 0.9;
        setSamState(
          r.polygon ? { phase: "ready" } : { phase: "empty", message: covered ? "這裡已被其他物件的輪廓蓋住，請先檢查那個輪廓" : undefined },
        );
      })
      .catch((e) => seq === samSeq.current && setSamState({ phase: "error", message: e instanceof Error ? e.message : String(e) }));
  };

  const addSamPoint = (p: SamPoint) => {
    const next = [...samPoints, p];
    setSamPoints(next);
    runSam(next, samBox);
  };

  const setSamPrompt = (r: Rect) => {
    setSamBox(r);
    runSam(samPoints, r);
  };

  const acceptSam = () => {
    // not selected: the next object usually follows, and a number key would otherwise re-class this one
    if (samPreview && samState.phase === "ready" && createPolygon(samPreview, false)) clearSam();
  };

  // any human touch turns a model box into a human box
  const changeBox = (uid: string, patch: Partial<EBox>, human = true) =>
    commit(boxes.map((b) => (b.uid === uid ? { ...b, ...patch, ...(human ? { source: "manual" as const, score: null } : {}) } : b)));

  const deleteBox = (uid: string) => {
    commit(boxes.filter((b) => b.uid !== uid));
    if (selected === uid) setSelected(null);
  };

  const setFlag = (uid: string, flag: "hidden" | "locked") =>
    setBoxes((bs) => bs.map((b) => (b.uid === uid ? { ...b, [flag]: !b[flag] } : b)));

  // ----------------------------------------------------------------- save
  const save = useCallback(
    async (markReviewed: boolean): Promise<boolean> => {
      if (!image) return false;
      if (!markReviewed && !dirty) return true;
      setSaving(true);
      try {
        const s = await api.saveAnnotations(image.id, boxes, image.version, markReviewed);
        // keep the client ids/flags so the selection survives a save
        const eb = s.boxes.map((b, i) => ({ ...b, uid: boxes[i]?.uid ?? newUid(), hidden: boxes[i]?.hidden, locked: boxes[i]?.locked }));
        setImage(s.image);
        setBoxes(eb);
        setSaved(eb);
        setRevisions(null);
        return true;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        toast(msg, "error");
        if (e instanceof ApiError && e.status === 409) loadImage();
        return false;
      } finally {
        setSaving(false);
      }
    },
    [image, boxes, dirty, toast, loadImage],
  );

  const go = useCallback(
    async (delta: number) => {
      if (!(await save(false))) return;
      const target = queue[pos + delta];
      if (target === undefined) {
        toast(delta > 0 ? "已經是最後一張" : "已經是第一張");
        return;
      }
      nav(`/projects/${pid}/annotate/${target}${queryString}`, { replace: true });
    },
    [save, queue, pos, nav, pid, queryString, toast],
  );

  const confirmAndNext = useCallback(async () => {
    if (!(await save(true))) return;
    const target = queue[pos + 1];
    if (target === undefined) {
      toast("已確認。這是佇列中的最後一張圖片");
      return;
    }
    nav(`/projects/${pid}/annotate/${target}${queryString}`, { replace: true });
  }, [save, queue, pos, nav, pid, queryString, toast]);

  // ------------------------------------------------------------- hotkeys
  const visibleOrder = useMemo(() => orderedBoxes(boxes, sortByScore), [boxes, sortByScore]);
  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement;
    if (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT") return;
    if (tipsOpen) return; // the tips dialog handles its own keys (Esc closes it)
    if (e.key === "?") {
      setTipsOpen(true);
      return;
    }
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    const sel = boxes.find((b) => b.uid === selected);

    // while SAM has prompts these keys act on them, not on the image or the selected object
    if (tool === "sam" && (samPoints.length > 0 || samBox)) {
      if (e.key === "Enter") {
        e.preventDefault();
        acceptSam();
        return;
      }
      if (e.key === "Escape") {
        clearSam();
        return;
      }
      if (e.key === "Backspace" || e.key === "Delete" || (mod && k === "z")) {
        e.preventDefault();
        if (samPoints.length) {
          const next = samPoints.slice(0, -1);
          setSamPoints(next);
          runSam(next, samBox);
        } else {
          setSamBox(null);
          runSam([], null);
        }
        return;
      }
    }

    // while a polygon is being drawn these keys act on it, not on the image or the selected object
    if (canvas.current?.hasDraft()) {
      if (e.key === "Enter") {
        e.preventDefault();
        canvas.current.closeDraft();
        return;
      }
      if (e.key === "Escape") {
        canvas.current.cancelDraft();
        return;
      }
      if (e.key === "Backspace" || e.key === "Delete" || (mod && k === "z")) {
        e.preventDefault();
        canvas.current.popDraftVertex();
        return;
      }
    }

    if (mod && k === "z") {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    } else if (mod && k === "y") {
      e.preventDefault();
      redo();
    } else if (mod && k === "s") {
      e.preventDefault();
      save(false).then((ok) => ok && toast("已儲存"));
    } else if (mod) {
      return;
    } else if (e.key === "Enter") {
      e.preventDefault();
      confirmAndNext();
    } else if (k === "v") setTool("select");
    else if (k === "b" || k === "n") setTool("draw");
    else if (k === "p") setTool("polygon");
    else if (k === "s") setTool("sam");
    else if (k === "f") canvas.current?.fit();
    else if (k === "l") setShowLabels((x) => !x);
    else if (k === "h" && sel) setFlag(sel.uid, "hidden");
    else if (/^[1-9]$/.test(e.key)) {
      const c = project?.classes[Number(e.key) - 1];
      if (!c) return;
      setActiveClass(c.id);
      if (sel && sel.class_id !== c.id) changeBox(sel.uid, { class_id: c.id });
    } else if ((e.key === "Delete" || e.key === "Backspace") && sel) {
      e.preventDefault();
      deleteBox(sel.uid);
    } else if (e.key === "Escape") setSelected(null);
    else if (e.key === "Tab") {
      e.preventDefault();
      const list = visibleOrder.filter((b) => !b.hidden);
      if (!list.length) return;
      const i = list.findIndex((b) => b.uid === selected);
      const next = e.shiftKey ? (i <= 0 ? list.length - 1 : i - 1) : (i + 1) % list.length;
      setSelected(list[next].uid);
      setTool("select");
    } else if (e.key.startsWith("Arrow") && sel && !sel.locked && image) {
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
      const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
      const r = keepInside({ x: sel.x + dx, y: sel.y + dy, w: sel.w, h: sel.h }, image.width, image.height);
      changeBox(sel.uid, sel.polygon ? { ...r, polygon: translatePolygon(sel.polygon, r.x - sel.x, r.y - sel.y) } : r);
    } else if (k === "d" || e.key === "ArrowRight" || e.key === "PageDown") go(1);
    else if (k === "a" || e.key === "ArrowLeft" || e.key === "PageUp") go(-1);
    else if (e.key === "+" || e.key === "=") canvas.current?.zoom(1.25);
    else if (e.key === "-") canvas.current?.zoom(0.8);
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  // ------------------------------------------------------------- render
  if (!project || !image) {
    return <div className="annotator" />;
  }
  const back = `/projects/${pid}/images${queryString}`;
  const activeColor = (activeClass !== null && classes.get(activeClass)?.color) || "#ffffff";
  const counts = new Map<number, number>();
  boxes.forEach((b) => counts.set(b.class_id, (counts.get(b.class_id) ?? 0) + 1));
  const autoLeft = boxes.filter((b) => b.source === "auto").length;

  return (
    <div className="annotator">
      <header className="topbar">
        <Link to={back} className="crumb" onClick={(e) => dirty && !confirm("有未儲存的變更，確定離開？") && e.preventDefault()}>
          ← {project.name}
        </Link>
        <span className="faint">/</span>
        <b style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 280 }} title={image.filename}>
          {image.filename}
        </b>
        <StatusPill status={image.status} />
        {image.split && <span className={`split-tag ${image.split}`}>{image.split}</span>}
        <span className="spacer" />
        <button className="ghost icon" onClick={() => go(-1)} disabled={pos <= 0} aria-label="上一張（A）" title="上一張（A）">
          ‹
        </button>
        <span className="small muted">{pos >= 0 ? `${pos + 1} / ${queue.length}` : "不在目前的篩選中"}</span>
        <button className="ghost icon" onClick={() => go(1)} disabled={pos < 0 || pos >= queue.length - 1} aria-label="下一張（D）" title="下一張（D）">
          ›
        </button>
        <span className={`save-state ${dirty ? "dirty" : ""}`}>{saving ? "儲存中…" : dirty ? "有未儲存的變更" : "已儲存"}</span>
        <button className="ghost icon" onClick={() => setTipsOpen(true)} aria-label="怎麼標會越來越快（?）" title="怎麼標會越來越快（?）">
          ?
        </button>
        <button onClick={() => save(false).then((ok) => ok && toast("已儲存"))} disabled={!dirty || saving} title="Ctrl+S">
          儲存
        </button>
        <button className="primary" data-tour="confirm" onClick={confirmAndNext} disabled={saving} title="Enter">
          確認並下一張 <kbd style={{ background: "transparent", color: "inherit", borderColor: "rgba(0,0,0,.25)" }}>Enter</kbd>
        </button>
      </header>

      <nav className="tool-rail" aria-label="工具" data-tour="tools">
        <button className={`icon ${tool === "select" ? "on" : ""}`} onClick={() => setTool("select")} title="選取與調整（V）" aria-label="選取工具">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><path d="M3 2l9 5-4 1.2L6 13z" fill="currentColor" /></svg>
        </button>
        <button className={`icon ${tool === "draw" ? "on" : ""}`} onClick={() => setTool("draw")} title="畫框（B）" aria-label="畫框工具">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><rect x="2.5" y="3.5" width="11" height="9" fill="none" stroke="currentColor" strokeWidth="1.6" /><path d="M2.5 1v4M1 3.5h3" stroke="currentColor" strokeWidth="1.4" /></svg>
        </button>
        <button className={`icon ${tool === "polygon" ? "on" : ""}`} onClick={() => setTool("polygon")} title="畫多邊形（P）：點擊加頂點，點第一個頂點、雙擊或按 Enter 完成" aria-label="多邊形工具">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><path d="M8 2l5.5 4-2 7.5h-7l-2-7.5z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>
        </button>
        <button
          className={`icon ${tool === "sam" ? "on" : ""}`}
          onClick={() => setTool("sam")}
          title="點選分割（S）：左鍵點物件，右鍵排除，拖曳框選，Enter 完成"
          aria-label="點選分割工具"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
            <path d="M3 9.5c0-3 2.2-6 5.5-6S14 6 13 9s-3.5 4.5-6 4.5S3 12 3 9.5z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2.2 1.6" />
            <circle cx="8" cy="8.5" r="1.8" fill="currentColor" />
          </svg>
        </button>
        <hr />
        <button className="icon" onClick={undo} disabled={!history.current.canUndo} title="復原（Ctrl+Z）" aria-label="復原">
          ↶
        </button>
        <button className="icon" onClick={redo} disabled={!history.current.canRedo} title="重做（Ctrl+Shift+Z）" aria-label="重做">
          ↷
        </button>
        <hr />
        <button className="icon" onClick={() => canvas.current?.zoom(1.25)} title="放大（+）" aria-label="放大">
          +
        </button>
        <button className="icon" onClick={() => canvas.current?.zoom(0.8)} title="縮小（-）" aria-label="縮小">
          −
        </button>
        <button className="icon" onClick={() => canvas.current?.fit()} title="符合視窗（F）" aria-label="符合視窗">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
        </button>
        <button className={`icon ${showLabels ? "on" : ""}`} onClick={() => setShowLabels((x) => !x)} title="顯示標籤（L）" aria-label="顯示標籤">
          Aa
        </button>
      </nav>

      <div style={{ position: "relative", minWidth: 0, minHeight: 0, display: "grid" }}>
        <AnnotCanvas
          ref={canvas}
          imageUrl={api.imageUrl(image.id)}
          imgW={image.width}
          imgH={image.height}
          boxes={boxes}
          classes={classes}
          selected={selected}
          tool={tool}
          drawColor={activeColor}
          lowThreshold={lowThreshold}
          showLabels={showLabels}
          onSelect={setSelected}
          onChange={(uid, r) => changeBox(uid, r)}
          onCreate={createBox}
          onChangePolygon={(uid, points) => changeBox(uid, { ...polygonBBox(points), polygon: points })}
          onCreatePolygon={createPolygon}
          onCursor={setCursor}
          onZoom={setZoom}
          sam={{ points: samPoints, box: samBox, preview: samPreview }}
          onSamPoint={addSamPoint}
          onSamBox={setSamPrompt}
        />
        <span className="zoom-label">{Math.round(zoom * 100)}%</span>
        {tool === "sam" && (
          <div className={`sam-hint ${samState.phase === "error" || samState.phase === "empty" ? "warn" : ""}`} role="status">
            <span>
              {samState.phase === "loading" && (samState.message ?? "SAM 2 載入中…")}
              {samState.phase === "busy" && "計算中…"}
              {samState.phase === "empty" && (samState.message ?? "找不到物件，換個位置點點看")}
              {samState.phase === "error" && `SAM 2 無法使用：${samState.message}`}
              {samState.phase === "ready" && (samPreview ? "Enter 完成・Esc 取消・右鍵排除" : "左鍵點物件・右鍵排除・拖曳框選")}
            </span>
            {samState.phase !== "error" && (
              <label className="check" title="把已標好的多邊形從新輪廓中扣掉：被壓住的物件只會得到露出來的部分">
                <input
                  type="checkbox"
                  checked={samExclude}
                  onChange={(e) => {
                    setSamExclude(e.target.checked);
                    runSam(samPoints, samBox, e.target.checked);
                  }}
                />
                扣除已標註
              </label>
            )}
            {samState.phase === "ready" && samPreview && (
              <button className="primary" onClick={acceptSam}>
                完成
              </button>
            )}
          </div>
        )}
      </div>

      <aside className="side">
        <section>
          <h3>
            類別 <span className="count">按數字鍵切換</span>
          </h3>
          {project.classes.length === 0 && (
            <p className="small muted">
              沒有類別。<Link to={`/projects/${pid}/classes`}>新增類別</Link>
            </p>
          )}
          <div className="class-pick" data-tour="classes">
            {project.classes.map((c, i) => (
              <button
                key={c.id}
                className={activeClass === c.id ? "on" : ""}
                onClick={() => {
                  setActiveClass(c.id);
                  const sel = boxes.find((b) => b.uid === selected);
                  if (sel && sel.class_id !== c.id) changeBox(sel.uid, { class_id: c.id });
                }}
              >
                <span className="swatch" style={{ background: c.color }} />
                {c.name}
                <span className="faint small">{counts.get(c.id) ?? ""}</span>
                {i < 9 && <kbd>{i + 1}</kbd>}
              </button>
            ))}
          </div>
        </section>

        <section>
          <h3>
            物件 <span className="count">{boxes.length}</span>
            <span className="spacer" />
            <label className="check small muted" style={{ fontWeight: 400 }} data-tour="low-first">
              <input type="checkbox" checked={sortByScore} onChange={(e) => setSortByScore(e.target.checked)} />
              低分優先
            </label>
          </h3>
          {autoLeft > 0 && (
            <p className="small st-auto" style={{ marginBottom: 8 }}>
              {autoLeft} 個框來自模型（虛線），修改後就會變成人工標註
            </p>
          )}
          {boxes.length === 0 && <p className="small muted">按 B 後在圖片上拖曳畫框，按 P 逐點畫多邊形，或按 S 點物件自動描輪廓。沒有物件的圖片也可以直接確認，作為負樣本。</p>}
          <div className="stack" style={{ gap: 2 }}>
            {visibleOrder.map((b) => (
              <ObjectRow
                key={b.uid}
                b={b}
                classes={project.classes}
                selected={b.uid === selected}
                low={b.source === "auto" && b.score !== null && b.score < lowThreshold}
                onSelect={() => {
                  setSelected(b.uid);
                  setTool("select");
                }}
                onClass={(cid) => changeBox(b.uid, { class_id: cid })}
                onHide={() => setFlag(b.uid, "hidden")}
                onLock={() => setFlag(b.uid, "locked")}
                onDelete={() => deleteBox(b.uid)}
              />
            ))}
          </div>
        </section>

        <section>
          <h3>需要注意的分數</h3>
          <label className="field">
            模型信心低於 {lowThreshold.toFixed(2)} 的框以黃色標示
            <input type="range" min={0.05} max={0.95} step={0.05} value={lowThreshold} onChange={(e) => setLowThreshold(Number(e.target.value))} />
          </label>
        </section>

        <section>
          <h3>
            版本紀錄 <span className="count">目前 v{image.version}</span>
          </h3>
          {revisions === null ? (
            <button className="ghost small" onClick={() => api.revisions(image.id).then(setRevisions)}>
              顯示歷史版本
            </button>
          ) : revisions.length === 0 ? (
            <p className="small muted">還沒有任何儲存紀錄。</p>
          ) : (
            revisions.map((r) => (
              <div key={r.id} className="rev">
                <span className={`pill st-${r.status}`}>v{r.version}</span>
                <span className="muted">
                  {ACTION_LABEL[r.action]}，{r.boxes.length} 個框，{timeAgo(r.created_at)}
                </span>
                {r.version !== image.version && (
                  <button
                    className="ghost small"
                    onClick={async () => {
                      if (dirty && !confirm("還原會捨棄目前未儲存的變更，繼續？")) return;
                      try {
                        await api.restoreRevision(image.id, r.version);
                        await loadImage();
                        toast(`已還原到 v${r.version}`);
                      } catch (e) {
                        toast(e instanceof Error ? e.message : String(e), "error");
                      }
                    }}
                  >
                    還原
                  </button>
                )}
              </div>
            ))
          )}
        </section>
      </aside>

      <footer className="statusbar">
        <span>
          {image.width}×{image.height}
          {cursor && `，游標 ${cursor.x}, ${cursor.y}`}
        </span>
        {image.group_key && <span>群組 {image.group_key}</span>}
        <span className="spacer" />
        <span>
          <kbd>B</kbd> 畫框 <kbd>P</kbd> 多邊形 <kbd>S</kbd> 點選分割 <kbd>V</kbd> 選取 <kbd>1-9</kbd> 類別 <kbd>Tab</kbd> 下一個框 <kbd>Del</kbd> 刪除 <kbd>A</kbd>/<kbd>D</kbd> 上/下一張{" "}
          <kbd>空白鍵</kbd>+拖曳 平移 <kbd>滾輪</kbd> 縮放 <kbd>?</kbd> 技巧
        </span>
      </footer>
      {tipsOpen && <LabelingTips onClose={() => setTipsOpen(false)} />}
    </div>
  );
}

function orderedBoxes(boxes: EBox[], byScore: boolean): EBox[] {
  if (!byScore) return boxes;
  return [...boxes].sort((a, b) => (a.score ?? 2) - (b.score ?? 2));
}

function ObjectRow({
  b,
  classes,
  selected,
  low,
  onSelect,
  onClass,
  onHide,
  onLock,
  onDelete,
}: {
  b: EBox;
  classes: LabelClass[];
  selected: boolean;
  low: boolean;
  onSelect: () => void;
  onClass: (id: number) => void;
  onHide: () => void;
  onLock: () => void;
  onDelete: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  const color = classes.find((c) => c.id === b.class_id)?.color ?? "#fff";
  return (
    <div ref={ref} className={`obj ${selected ? "on" : ""} ${b.hidden ? "hidden-box" : ""}`} onClick={onSelect}>
      <span
        className="swatch"
        style={b.source === "auto" ? { border: `2px dashed ${color}`, background: "transparent" } : { background: color }}
        title={b.source === "auto" ? "模型產生" : "人工標註"}
      />
      <select value={b.class_id} onClick={(e) => e.stopPropagation()} onChange={(e) => onClass(Number(e.target.value))} aria-label="類別">
        {classes.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
          </option>
        ))}
      </select>
      <span className={`score ${low ? "low" : ""}`}>{b.source === "auto" && b.score !== null ? b.score.toFixed(2) : ""}</span>
      <button
        className="ghost icon"
        onClick={(e) => {
          e.stopPropagation();
          onHide();
        }}
        title={b.hidden ? "顯示（H）" : "隱藏（H）"}
        aria-label={b.hidden ? "顯示" : "隱藏"}
      >
        <Icon name={b.hidden ? "eye-off" : "eye"} />
      </button>
      <div className="row" style={{ gap: 0 }}>
        <button
          className={`ghost icon ${b.locked ? "on" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onLock();
          }}
          title={b.locked ? "解除鎖定" : "鎖定（避免誤動）"}
          aria-label={b.locked ? "解除鎖定" : "鎖定"}
          style={b.locked ? { color: "var(--st-progress)" } : undefined}
        >
          <Icon name={b.locked ? "lock" : "unlock"} />
        </button>
        <button
          className="ghost icon danger"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          title="刪除（Del）"
          aria-label="刪除"
        >
          <Icon name="trash" />
        </button>
      </div>
    </div>
  );
}

function Icon({ name }: { name: "eye" | "eye-off" | "lock" | "unlock" | "trash" }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden>
      {name === "eye" && (
        <>
          <path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" {...common} />
          <circle cx="8" cy="8" r="2" {...common} />
        </>
      )}
      {name === "eye-off" && (
        <>
          <path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z" {...common} opacity={0.5} />
          <path d="M2.5 13.5l11-11" {...common} />
        </>
      )}
      {(name === "lock" || name === "unlock") && (
        <>
          <rect x="3.5" y="7" width="9" height="6.5" rx="1" {...common} />
          <path d={name === "lock" ? "M5.5 7V5a2.5 2.5 0 015 0v2" : "M5.5 7V5a2.5 2.5 0 014.8-1"} {...common} />
        </>
      )}
      {name === "trash" && <path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.5h5.6l.7-8.5" {...common} />}
    </svg>
  );
}
