// Guided tour: dims the page, lights up one part of it and explains it, step by step across pages.
// Shown once on the first visit; replayed from the "導覽" button in the top bar or the ? dialog.

import {
  createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type CSSProperties, type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api";

type Page = "projects" | "images" | "annotate" | "assist" | "models" | "split";

interface TourStep {
  page?: Page; // where the step lives; none = stay where the tour is
  target?: string; // data-tour id to light up; none = a card in the middle
  title: string;
  body: string;
}

export const TOUR_STEPS: TourStep[] = [
  {
    title: "歡迎使用 VisionLoop",
    body: "標一點，學一點，模型越準，標得越快。接下來用一分鐘帶你走一次：先手動標幾張，讓模型幫你框剩下的，你只要審核修正，一圈一圈越來越省力。",
  },
  {
    page: "projects",
    target: "new-project",
    title: "1. 建立專案",
    body: "每個專案有自己的圖片和類別。建立時就可以先輸入類別，例如 bolt、nut。",
  },
  {
    page: "images",
    target: "upload",
    title: "2. 上傳圖片",
    body: "把圖片或整個資料夾拖到這裡。iPhone 的 HEIC 和相機的 DNG 也可以。",
  },
  {
    page: "images",
    target: "status-filter",
    title: "圖片的狀態",
    body: "未標註 → 自動標註（模型框的）→ 編輯中 → 已確認。只有「已確認」的圖片會拿來訓練。",
  },
  {
    page: "images",
    target: "start-label",
    title: "3. 開始標註",
    body: "依目前的篩選，一張接一張標。",
  },
  {
    page: "annotate",
    target: "tools",
    title: "標註工具",
    body: "畫框（B）、多邊形（P）、點選分割（S）：按 S 後點一下物件，SAM 2 就會描出輪廓。",
  },
  {
    page: "annotate",
    target: "classes",
    title: "切換類別",
    body: "用數字鍵 1–9 切換；選取一個物件再按數字，就能直接改它的類別。",
  },
  {
    page: "annotate",
    target: "confirm",
    title: "每個類別先標 3～5 個就好",
    body: "標完按 Enter 確認並跳下一張。不用一開始就標很多，下一步讓模型幫你。",
  },
  {
    page: "assist",
    target: "probe",
    title: "4. 初步標註",
    body: "在「輔助標註」頁按「開始初步標註」，用你確認過的幾張圖訓練幾秒鐘，就能幫你框剩下的圖片。每多確認一批，再跑一次會更準。",
  },
  {
    page: "annotate",
    target: "low-first",
    title: "5. 審核：只改錯的",
    body: "模型框的是虛線。勾「低分優先」先看模型最沒把握的框，改錯的、補漏的，按 Enter 下一張。修正這些對模型幫助最大。",
  },
  {
    page: "split",
    target: "split",
    title: "6. 資料夠了就訓練正式模型",
    body: "累積約 50 張已確認的圖片後，在這裡分好 train / valid / test，再到「訓練與權重」頁訓練 YOLO 或 RF-DETR。",
  },
  {
    page: "assist",
    target: "autolabel",
    title: "7. 用你的模型標下一批",
    body: "訓練好的權重會出現在這裡的「權重」選單，用它自動標註新圖片，再回去審核。一圈一圈下去，模型越準，要改的越少。",
  },
  {
    title: "就是這樣！",
    body: "在標註畫面按 ? 可以看快捷鍵和省時間的技巧；想再看一次導覽，按右上角的「導覽」。",
  },
];

const SEEN_KEY = "vl-tour-seen";
const PAD = 8;

interface TourApi {
  start: () => void;
}
const TourContext = createContext<TourApi>({ start: () => {} });
export const useTour = () => useContext(TourContext);

function markSeen() {
  try {
    localStorage.setItem(SEEN_KEY, "1");
  } catch {
    /* private window */
  }
}
function seen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === "1";
  } catch {
    return true; // storage blocked: do not nag on every visit
  }
}

interface Resolved extends TourStep {
  path?: string;
}

/** Turns page names into URLs for an existing project and image; steps that need one we do not have are dropped. */
async function resolveSteps(currentPid: number | null): Promise<Resolved[]> {
  let pid = currentPid;
  if (pid == null) {
    const projects = await api.projects().catch(() => []);
    // a project with images can show every step, including the annotator
    pid = (projects.find((p) => p.stats.total > 0) ?? projects[0])?.id ?? null;
  }
  let imageId: number | null = null;
  if (pid != null) {
    const ids = await api.imageIds(pid, {}).catch(() => []);
    imageId = ids.length ? ids[0] : null;
  }
  const out: Resolved[] = [];
  for (const s of TOUR_STEPS) {
    if (!s.page) out.push(s);
    else if (s.page === "projects") out.push({ ...s, path: "/" });
    else if (pid == null) continue;
    else if (s.page === "annotate") {
      if (imageId != null) out.push({ ...s, path: `/projects/${pid}/annotate/${imageId}` });
    } else out.push({ ...s, path: `/projects/${pid}/${s.page}` });
  }
  if (pid == null)
    out.splice(out.length - 1, 0, {
      title: "先建立第一個專案",
      body: "建立專案並上傳圖片後，按右上角的「導覽」，就能繼續看標註、初步標註和訓練的介紹。",
    });
  return out;
}

export function TourProvider({ children }: { children: ReactNode }) {
  const [steps, setSteps] = useState<Resolved[] | null>(null);
  const [index, setIndex] = useState(0);
  const loc = useLocation();
  const nav = useNavigate();

  const start = useCallback(async () => {
    const m = /^\/projects\/(\d+)/.exec(window.location.pathname);
    const resolved = await resolveSteps(m ? Number(m[1]) : null);
    setIndex(0);
    setSteps(resolved);
  }, []);

  // first visit: offer the tour once
  useEffect(() => {
    if (!seen()) start();
  }, [start]);

  const close = useCallback(() => {
    markSeen();
    setSteps(null);
  }, []);

  const step = steps?.[index];
  useEffect(() => {
    if (step?.path && loc.pathname !== step.path) nav(step.path);
  }, [step, loc.pathname, nav]);

  const value = useMemo(() => ({ start }), [start]);
  return (
    <TourContext.Provider value={value}>
      {children}
      {steps && step && (
        <TourOverlay
          step={step}
          index={index}
          total={steps.length}
          onPrev={() => setIndex((i) => Math.max(0, i - 1))}
          onNext={() => (index + 1 < steps.length ? setIndex(index + 1) : close())}
          onClose={close}
          ready={!step.path || loc.pathname === step.path}
        />
      )}
    </TourContext.Provider>
  );
}

function TourOverlay({
  step,
  index,
  total,
  onPrev,
  onNext,
  onClose,
  ready,
}: {
  step: Resolved;
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
  ready: boolean;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [missing, setMissing] = useState(false);
  const card = useRef<HTMLDivElement>(null);
  const [cardSize, setCardSize] = useState({ w: 360, h: 180 });

  // find the element (pages load their data first), keep following it while the page moves
  useEffect(() => {
    setRect(null);
    setMissing(false);
    if (!step.target || !ready) return;
    let scrolled = false;
    const began = Date.now();
    const timer = window.setInterval(() => {
      const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
      if (!el || el.offsetParent === null) {
        if (Date.now() - began > 4000) setMissing(true);
        return;
      }
      if (!scrolled) {
        el.scrollIntoView({ block: "center", behavior: "smooth" });
        scrolled = true;
      }
      const r = el.getBoundingClientRect();
      setRect((old) =>
        old && old.x === r.x && old.y === r.y && old.width === r.width && old.height === r.height ? old : r,
      );
    }, 120);
    return () => window.clearInterval(timer);
  }, [step, ready]);

  // the tour owns the keyboard: nothing reaches the page underneath (Enter would confirm an image)
  const keys = useRef({ onPrev, onNext, onClose });
  keys.current = { onPrev, onNext, onClose };
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      e.stopImmediatePropagation();
      if (e.key === "Escape") keys.current.onClose();
      else if (e.key === "Enter" || e.key === "ArrowRight") {
        e.preventDefault();
        keys.current.onNext();
      } else if (e.key === "ArrowLeft") keys.current.onPrev();
      else if (e.key !== "Tab") e.preventDefault();
    };
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, []);

  useLayoutEffect(() => {
    if (card.current) setCardSize({ w: card.current.offsetWidth, h: card.current.offsetHeight });
  }, [step, rect]);

  const waiting = !!step.target && !rect && !missing;
  const pos = placeCard(rect, cardSize);

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label={step.title}>
      {rect ? (
        <div
          className="tour-spot"
          style={{ left: rect.left - PAD, top: rect.top - PAD, width: rect.width + PAD * 2, height: rect.height + PAD * 2 }}
        />
      ) : (
        <div className="tour-dim" />
      )}
      <div ref={card} className={`tour-card ${waiting ? "waiting" : ""}`} style={pos}>
        <div className="tour-count">
          {index + 1} / {total}
        </div>
        <h3>{step.title}</h3>
        <p>{step.body}</p>
        <div className="tour-actions">
          <button className="ghost" onClick={onClose}>
            {index + 1 === total ? "關閉" : "略過導覽"}
          </button>
          <span className="spacer" />
          {index > 0 && <button onClick={onPrev}>上一步</button>}
          <button className="primary" onClick={onNext} autoFocus>
            {index === 0 ? "開始" : index + 1 === total ? "完成" : "下一步"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Below the element if it fits, else above, else beside it; a centred card when nothing is lit. */
function placeCard(rect: DOMRect | null, size: { w: number; h: number }): CSSProperties {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (!rect) return { left: Math.max(16, (vw - size.w) / 2), top: Math.max(16, (vh - size.h) / 2) };
  const gap = 14;
  const clampX = (x: number) => Math.min(Math.max(16, x), vw - size.w - 16);
  const clampY = (y: number) => Math.min(Math.max(16, y), vh - size.h - 16);
  if (rect.bottom + PAD + gap + size.h <= vh - 16) return { left: clampX(rect.left), top: rect.bottom + PAD + gap };
  if (rect.top - PAD - gap - size.h >= 16) return { left: clampX(rect.left), top: rect.top - PAD - gap - size.h };
  if (rect.right + PAD + gap + size.w <= vw - 16) return { left: rect.right + PAD + gap, top: clampY(rect.top) };
  return { left: clampX(rect.left - PAD - gap - size.w), top: clampY(rect.top) };
}
