import type Konva from "konva";
import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, forwardRef } from "react";
import { Circle, Group, Image as KImage, Label, Layer, Line, Rect, Stage, Tag, Text, Transformer } from "react-konva";
import type { LabelClass } from "../api";
import {
  clampPolygon,
  clampRect,
  fitView,
  insertVertex,
  keepInside,
  polygonBBox,
  rectFromPoints,
  removeVertex,
  translatePolygon,
  zoomAt,
  type EBox,
  type Pt,
  type Rect as R,
} from "./geometry";

/** x, y in image pixels; 1 = part of the object, 0 = not part of it */
export type SamPoint = [number, number, 0 | 1];

const AUTO_COLOR = "#a78bfa";
const SAM_YES = "#3ecf8e";
const SAM_NO = "#ef5b5b";
const LOW_COLOR = "#f2b84b";
const CLOSE_RADIUS = 9; // screen pixels around the first vertex that close the polygon

export interface CanvasHandle {
  fit: () => void;
  zoom: (factor: number) => void;
  /** A polygon is being drawn: Enter, Esc and Backspace act on it instead of the page. */
  hasDraft: () => boolean;
  closeDraft: () => void;
  cancelDraft: () => void;
  popDraftVertex: () => void;
}

interface Props {
  imageUrl: string;
  imgW: number;
  imgH: number;
  boxes: EBox[];
  classes: Map<number, LabelClass>;
  selected: string | null;
  tool: "select" | "draw" | "polygon" | "sam";
  /** click-to-segment: the prompts so far and the outline SAM 2 made from them */
  sam?: { points: SamPoint[]; box: R | null; preview: Pt[] | null };
  onSamPoint?: (p: SamPoint) => void;
  onSamBox?: (r: R) => void;
  drawColor: string;
  lowThreshold: number;
  showLabels: boolean;
  onSelect: (uid: string | null) => void;
  onChange: (uid: string, rect: R) => void;
  onCreate: (rect: R) => void;
  onChangePolygon: (uid: string, points: Pt[]) => void;
  onCreatePolygon: (points: Pt[]) => void;
  onCursor: (p: { x: number; y: number } | null) => void;
  onZoom: (scale: number) => void;
}

const AnnotCanvas = forwardRef<CanvasHandle, Props>(function AnnotCanvas(props, ref) {
  const { imageUrl, imgW, imgH, boxes, classes, selected, tool, drawColor, lowThreshold, showLabels } = props;
  const wrap = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [drawing, setDrawing] = useState<{ a: { x: number; y: number }; b: { x: number; y: number } } | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [draft, setDraft] = useState<Pt[]>([]);
  const repeatClick = useRef(false); // the last click landed on the last draft vertex
  const samNegative = useRef(false); // shift was held when the click started
  // a vertex being dragged, shown live and committed on release
  const [liveVertex, setLiveVertex] = useState<{ uid: string; index: number; p: Pt } | null>(null);
  const [movingPolygon, setMovingPolygon] = useState<string | null>(null);
  const panning = useRef<{ sx: number; sy: number; vx: number; vy: number } | null>(null);
  const [space, setSpace] = useState(false);

  // ---------------------------------------------------------- layout/image
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setImg(null);
    const im = new window.Image();
    im.src = imageUrl;
    im.onload = () => setImg(im);
    return () => {
      im.onload = null;
    };
  }, [imageUrl]);

  const fit = useCallback(() => {
    if (size.w > 0 && size.h > 0) setView(fitView(imgW, imgH, size.w, size.h));
  }, [imgW, imgH, size.w, size.h]);

  // refit when a new image arrives or the panel is resized
  useEffect(() => {
    fit();
  }, [fit, imageUrl]);

  useEffect(() => props.onZoom(view.scale), [view.scale]); // eslint-disable-line react-hooks/exhaustive-deps

  const closeDraft = () => {
    const bb = draft.length >= 3 ? polygonBBox(draft) : null;
    if (!bb) return; // keep drawing until there are three vertices
    setDraft([]);
    if (bb.w >= 1 && bb.h >= 1) props.onCreatePolygon(draft);
  };

  // an unfinished polygon does not survive a change of image or tool
  useEffect(() => setDraft([]), [imageUrl, tool]);

  useImperativeHandle(ref, () => ({
    fit,
    zoom: (f: number) => setView((v) => zoomAt(v, { x: size.w / 2, y: size.h / 2 }, f)),
    hasDraft: () => draft.length > 0,
    closeDraft,
    cancelDraft: () => setDraft([]),
    popDraftVertex: () => setDraft((d) => d.slice(0, -1)),
  }));

  // space bar = temporary hand tool
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault();
        setSpace(true);
      }
    };
    const up = (e: KeyboardEvent) => e.code === "Space" && setSpace(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // ------------------------------------------------------------ transformer
  useEffect(() => {
    const tr = trRef.current;
    const stage = stageRef.current;
    if (!tr || !stage) return;
    const box = boxes.find((b) => b.uid === selected);
    // polygons are edited through their vertices, not the resize handles
    const node = selected && box && !box.polygon && !box.hidden && !box.locked && tool === "select" ? stage.findOne(`#${selected}`) : null;
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, boxes, tool, view]);

  // --------------------------------------------------------------- helpers
  const toImage = (p: { x: number; y: number }) => ({
    x: (p.x - viewRef.current.x) / viewRef.current.scale,
    y: (p.y - viewRef.current.y) / viewRef.current.scale,
  });

  const stagePointer = () => stageRef.current?.getPointerPosition() ?? null;

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const p = stagePointer();
    if (!p) return;
    const factor = Math.pow(1.0015, -e.evt.deltaY);
    setView((v) => zoomAt(v, p, factor));
  };

  const onMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const p = stagePointer();
    if (!p) return;
    const btn = e.evt.button;
    const onBackground = e.target === stageRef.current || e.target.name() === "image";
    const onTransformer = e.target.getParent()?.className === "Transformer";
    if (tool === "sam" && !space && (btn === 0 || btn === 2)) {
      const ip = toImage(p);
      if (ip.x < 0 || ip.y < 0 || ip.x > imgW || ip.y > imgH) return;
      if (btn === 2) {
        props.onSamPoint?.([ip.x, ip.y, 0]); // right click: not part of the object
        return;
      }
      // a left press is a click (point) or the start of a drag (box), decided on release
      setDrawing({ a: ip, b: ip });
      samNegative.current = e.evt.shiftKey;
      return;
    }
    if (btn === 1 || btn === 2 || space || (tool === "select" && btn === 0 && onBackground)) {
      if (tool === "select" && btn === 0 && onBackground && !space) props.onSelect(null);
      panning.current = { sx: p.x, sy: p.y, vx: viewRef.current.x, vy: viewRef.current.y };
      return;
    }
    if (tool === "draw" && btn === 0 && !onTransformer) {
      const ip = toImage(p);
      const clamped = { x: Math.max(0, Math.min(imgW, ip.x)), y: Math.max(0, Math.min(imgH, ip.y)) };
      setDrawing({ a: clamped, b: clamped });
    }
    if (tool === "polygon" && btn === 0) {
      const ip = toImage(p);
      const vertex: Pt = clampPolygon([[ip.x, ip.y]], imgW, imgH)[0];
      const screenDist = (q: Pt) => Math.hypot(q[0] - vertex[0], q[1] - vertex[1]) * viewRef.current.scale;
      repeatClick.current = draft.length > 0 && screenDist(draft[draft.length - 1]) <= 4;
      if (draft.length >= 3 && screenDist(draft[0]) <= CLOSE_RADIUS) {
        closeDraft();
      } else if (!repeatClick.current) {
        setDraft([...draft, vertex]);
      }
    }
  };

  const onMouseMove = () => {
    const p = stagePointer();
    if (!p) return;
    if (panning.current) {
      const pc = panning.current;
      setView((v) => ({ ...v, x: pc.vx + p.x - pc.sx, y: pc.vy + p.y - pc.sy }));
      return;
    }
    const ip = toImage(p);
    const inside = ip.x >= 0 && ip.y >= 0 && ip.x <= imgW && ip.y <= imgH;
    setPointer(inside ? ip : null);
    props.onCursor(inside ? { x: Math.round(ip.x), y: Math.round(ip.y) } : null);
    if (drawing) {
      setDrawing({ ...drawing, b: { x: Math.max(0, Math.min(imgW, ip.x)), y: Math.max(0, Math.min(imgH, ip.y)) } });
    }
  };

  const onMouseUp = () => {
    panning.current = null;
    if (drawing && tool === "sam") {
      const r = clampRect(rectFromPoints(drawing.a, drawing.b), imgW, imgH);
      setDrawing(null);
      const sc = viewRef.current.scale;
      if (r && r.w * sc >= 4 && r.h * sc >= 4) props.onSamBox?.(r);
      else props.onSamPoint?.([drawing.a.x, drawing.a.y, samNegative.current ? 0 : 1]);
      return;
    }
    if (drawing) {
      const r = clampRect(rectFromPoints(drawing.a, drawing.b), imgW, imgH);
      setDrawing(null);
      // ignore accidental clicks: at least 4 screen pixels in both directions
      if (r && r.w * viewRef.current.scale >= 4 && r.h * viewRef.current.scale >= 4) props.onCreate(r);
    }
  };

  // ---------------------------------------------------------------- render
  const s = view.scale;
  const cursor = space || panning.current ? "grab" : tool === "draw" || tool === "polygon" || tool === "sam" ? "crosshair" : "default";
  const sam = tool === "sam" ? props.sam : undefined;

  return (
    <div ref={wrap} className="stage-wrap" style={{ cursor }} onContextMenu={(e) => e.preventDefault()}>
      <Stage
        ref={stageRef}
        width={size.w}
        height={size.h}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onDblClick={() => {
          // Konva also reports two quick clicks at different places: only a click that
          // landed on the last vertex (and so added nothing) is a real double click
          if (tool === "polygon" && repeatClick.current) closeDraft();
        }}
        onMouseLeave={() => {
          setPointer(null);
          props.onCursor(null);
        }}
      >
        <Layer>
          <Group x={view.x} y={view.y} scaleX={s} scaleY={s}>
            {img && <KImage image={img} width={imgW} height={imgH} name="image" />}
            {boxes.map((b) => {
              if (b.hidden) return null;
              const cls = classes.get(b.class_id);
              const color = cls?.color ?? "#ffffff";
              const isAuto = b.source === "auto";
              const isLow = isAuto && b.score !== null && b.score < lowThreshold;
              const isSel = b.uid === selected;
              if (b.polygon) {
                const points = b.polygon;
                const shown = liveVertex?.uid === b.uid ? points.map((q, i) => (i === liveVertex.index ? liveVertex.p : q)) : points;
                const editable = isSel && tool === "select" && !b.locked && !space;
                return (
                  <Group key={b.uid}>
                    <Line
                      id={b.uid}
                      points={shown.flat()}
                      closed
                      stroke={color}
                      strokeWidth={isSel ? 3 : 2}
                      strokeScaleEnabled={false}
                      hitStrokeWidth={12}
                      dash={isAuto ? [7 / s, 4 / s] : undefined}
                      fill={isLow ? "rgba(242,184,75,0.18)" : isSel ? hexA(color, 0.22) : hexA(color, 0.1)}
                      draggable={tool === "select" && !b.locked && !space}
                      onMouseDown={(e) => {
                        if (tool === "select" && e.evt.button === 0 && !space) {
                          e.cancelBubble = true;
                          props.onSelect(b.uid);
                        }
                      }}
                      onDblClick={() => {
                        const p = stagePointer();
                        if (!p || !editable) return;
                        const ip = toImage(p);
                        props.onChangePolygon(b.uid, insertVertex(points, clampPolygon([[ip.x, ip.y]], imgW, imgH)[0]));
                      }}
                      onDragStart={() => setMovingPolygon(b.uid)}
                      dragBoundFunc={(pos) => {
                        const v = viewRef.current;
                        const dx = Math.max(-b.x, Math.min(imgW - b.x - b.w, (pos.x - v.x) / v.scale));
                        const dy = Math.max(-b.y, Math.min(imgH - b.y - b.h, (pos.y - v.y) / v.scale));
                        return { x: dx * v.scale + v.x, y: dy * v.scale + v.y };
                      }}
                      onDragEnd={(e) => {
                        const n = e.target;
                        const [dx, dy] = [n.x(), n.y()];
                        n.position({ x: 0, y: 0 }); // the offset moves into the vertices
                        setMovingPolygon(null);
                        props.onChangePolygon(b.uid, translatePolygon(points, dx, dy));
                      }}
                    />
                    {editable &&
                      movingPolygon !== b.uid &&
                      shown.map(([vx, vy], i) => (
                        <Circle
                          key={i}
                          x={vx}
                          y={vy}
                          radius={5 / s}
                          fill="#1e2430"
                          stroke="#e7eaf0"
                          strokeWidth={1.5}
                          strokeScaleEnabled={false}
                          draggable
                          onMouseDown={(e) => {
                            e.cancelBubble = true;
                          }}
                          onContextMenu={(e) => {
                            e.evt.preventDefault();
                            const next = removeVertex(points, i);
                            if (next) props.onChangePolygon(b.uid, next);
                          }}
                          onMouseEnter={(e) => {
                            e.target.getStage()!.container().style.cursor = "pointer";
                          }}
                          onMouseLeave={(e) => {
                            e.target.getStage()!.container().style.cursor = "";
                          }}
                          dragBoundFunc={(pos) => {
                            const v = viewRef.current;
                            const [ix, iy] = clampPolygon([[(pos.x - v.x) / v.scale, (pos.y - v.y) / v.scale]], imgW, imgH)[0];
                            return { x: ix * v.scale + v.x, y: iy * v.scale + v.y };
                          }}
                          onDragMove={(e) => setLiveVertex({ uid: b.uid, index: i, p: [e.target.x(), e.target.y()] })}
                          onDragEnd={(e) => {
                            setLiveVertex(null);
                            props.onChangePolygon(b.uid, points.map((q, k) => (k === i ? [e.target.x(), e.target.y()] : q)));
                          }}
                        />
                      ))}
                  </Group>
                );
              }
              return (
                <Rect
                  key={b.uid}
                  id={b.uid}
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  stroke={color}
                  strokeWidth={isSel ? 3 : 2}
                  strokeScaleEnabled={false}
                  dash={isAuto ? [7 / s, 4 / s] : undefined}
                  fill={isLow ? "rgba(242,184,75,0.18)" : isSel ? hexA(color, 0.18) : hexA(color, 0.06)}
                  draggable={tool === "select" && !b.locked && !space}
                  onMouseDown={(e) => {
                    if (tool === "select" && e.evt.button === 0 && !space) {
                      e.cancelBubble = true;
                      props.onSelect(b.uid);
                    }
                  }}
                  onMouseEnter={(e) => {
                    if (tool === "select" && !b.locked) e.target.getStage()!.container().style.cursor = "move";
                  }}
                  onMouseLeave={(e) => {
                    e.target.getStage()!.container().style.cursor = "";
                  }}
                  dragBoundFunc={(pos) => {
                    const v = viewRef.current;
                    const r = keepInside({ x: (pos.x - v.x) / v.scale, y: (pos.y - v.y) / v.scale, w: b.w, h: b.h }, imgW, imgH);
                    return { x: r.x * v.scale + v.x, y: r.y * v.scale + v.y };
                  }}
                  onDragEnd={(e) => {
                    const n = e.target;
                    props.onChange(b.uid, keepInside({ x: n.x(), y: n.y(), w: b.w, h: b.h }, imgW, imgH));
                  }}
                  onTransformEnd={(e) => {
                    const n = e.target;
                    const r = clampRect(
                      { x: n.x(), y: n.y(), w: n.width() * n.scaleX(), h: n.height() * n.scaleY() },
                      imgW,
                      imgH,
                    );
                    n.scaleX(1);
                    n.scaleY(1);
                    if (r) props.onChange(b.uid, r);
                  }}
                />
              );
            })}

            {showLabels &&
              boxes.map((b) => {
                if (b.hidden) return null;
                const cls = classes.get(b.class_id);
                const isAuto = b.source === "auto";
                const isLow = isAuto && b.score !== null && b.score < lowThreshold;
                const text = `${cls?.name ?? "?"}${isAuto && b.score !== null ? ` ${b.score.toFixed(2)}` : ""}`;
                const labelH = 18 / s;
                const above = b.y - labelH >= 0;
                return (
                  <Label key={`l-${b.uid}`} x={b.x} y={above ? b.y - labelH : b.y} scaleX={1 / s} scaleY={1 / s} listening={false}>
                    <Tag fill={isLow ? LOW_COLOR : isAuto ? AUTO_COLOR : cls?.color ?? "#fff"} cornerRadius={2} />
                    <Text text={text} fontSize={12} padding={3} fill="#10141b" fontStyle="600" fontFamily="Times New Roman, 標楷體, DFKai-SB, BiauKai, serif" />
                  </Label>
                );
              })}

            {drawing && (
              <Rect
                {...(() => {
                  const r = rectFromPoints(drawing.a, drawing.b);
                  return { x: r.x, y: r.y, width: r.w, height: r.h };
                })()}
                stroke={drawColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                fill={hexA(drawColor, 0.12)}
                listening={false}
              />
            )}

            {draft.length > 0 && (
              <>
                <Line
                  points={[...draft.flat(), ...(pointer ? [pointer.x, pointer.y] : [])]}
                  stroke={drawColor}
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  fill={hexA(drawColor, 0.12)}
                  closed={draft.length >= 2}
                  listening={false}
                />
                {draft.map(([vx, vy], i) => (
                  <Circle
                    key={i}
                    x={vx}
                    y={vy}
                    radius={(i === 0 && draft.length >= 3 ? CLOSE_RADIUS : 4) / s}
                    fill={i === 0 ? drawColor : "#1e2430"}
                    stroke="#e7eaf0"
                    strokeWidth={1.5}
                    strokeScaleEnabled={false}
                    opacity={i === 0 && draft.length >= 3 ? 0.75 : 1}
                    listening={false}
                  />
                ))}
              </>
            )}

            {sam?.preview && (
              <Line
                points={sam.preview.flat()}
                closed
                stroke={drawColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                dash={[6 / s, 4 / s]}
                fill={hexA(drawColor, 0.25)}
                listening={false}
              />
            )}
            {sam?.box && (
              <Rect
                x={sam.box.x}
                y={sam.box.y}
                width={sam.box.w}
                height={sam.box.h}
                stroke="#e7eaf0"
                strokeWidth={1.5}
                strokeScaleEnabled={false}
                dash={[4 / s, 4 / s]}
                listening={false}
              />
            )}
            {sam?.points.map(([px, py, label], i) => (
              <Circle
                key={`sam-${i}`}
                x={px}
                y={py}
                radius={5 / s}
                fill={label ? SAM_YES : SAM_NO}
                stroke="#10141b"
                strokeWidth={1.5}
                strokeScaleEnabled={false}
                listening={false}
              />
            ))}

            {(tool === "draw" || tool === "polygon" || tool === "sam") && pointer && !space && (
              <>
                <Line points={[0, pointer.y, imgW, pointer.y]} stroke="#ffffff" opacity={0.55} strokeWidth={1} strokeScaleEnabled={false} dash={[4 / s, 4 / s]} listening={false} />
                <Line points={[pointer.x, 0, pointer.x, imgH]} stroke="#ffffff" opacity={0.55} strokeWidth={1} strokeScaleEnabled={false} dash={[4 / s, 4 / s]} listening={false} />
              </>
            )}
          </Group>
          <Transformer
            ref={trRef}
            rotateEnabled={false}
            keepRatio={false}
            ignoreStroke
            anchorSize={9}
            anchorCornerRadius={2}
            anchorStroke="#e7eaf0"
            anchorFill="#1e2430"
            borderStroke="#e7eaf0"
            borderDash={[3, 3]}
            boundBoxFunc={(oldBox, newBox) => {
              const v = viewRef.current;
              const minPx = 3;
              const left = v.x;
              const top = v.y;
              const right = v.x + imgW * v.scale;
              const bottom = v.y + imgH * v.scale;
              if (newBox.width < minPx || newBox.height < minPx) return oldBox;
              if (newBox.x < left - 0.5 || newBox.y < top - 0.5) return oldBox;
              if (newBox.x + newBox.width > right + 0.5 || newBox.y + newBox.height > bottom + 0.5) return oldBox;
              return newBox;
            }}
          />
        </Layer>
      </Stage>
      {!img && <div className="loading">載入圖片中…</div>}
    </div>
  );
});

export default AnnotCanvas;

function hexA(hex: string, a: number): string {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) return `rgba(255,255,255,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}
