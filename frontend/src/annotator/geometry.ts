import type { Box } from "../api";

/** Box as edited in the browser: a stable client id plus view-only flags. */
export interface EBox extends Box {
  uid: string;
  hidden?: boolean;
  locked?: boolean;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

let counter = 0;
export const newUid = () => `b${Date.now().toString(36)}${(counter++).toString(36)}`;

/** Rectangle spanned by two corner points, in any drag direction. */
export function rectFromPoints(a: { x: number; y: number }, b: { x: number; y: number }): Rect {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
}

/** Clip a rectangle to the image. Returns null if nothing meaningful is left. */
export function clampRect(r: Rect, width: number, height: number, minSize = 1): Rect | null {
  const x1 = Math.max(0, Math.min(width, r.x));
  const y1 = Math.max(0, Math.min(height, r.y));
  const x2 = Math.max(0, Math.min(width, r.x + r.w));
  const y2 = Math.max(0, Math.min(height, r.y + r.h));
  if (x2 - x1 < minSize || y2 - y1 < minSize) return null;
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/** Move a rectangle so it lies fully inside the image (keeps its size when possible). */
export function keepInside(r: Rect, width: number, height: number): Rect {
  const w = Math.min(r.w, width);
  const h = Math.min(r.h, height);
  return { x: Math.max(0, Math.min(width - w, r.x)), y: Math.max(0, Math.min(height - h, r.y)), w, h };
}

/** Scale and offset that fit an image into a viewport with a margin. */
export function fitView(imgW: number, imgH: number, viewW: number, viewH: number, margin = 0.94) {
  const scale = Math.min(viewW / imgW, viewH / imgH) * margin;
  return { scale, x: (viewW - imgW * scale) / 2, y: (viewH - imgH * scale) / 2 };
}

/** Zoom around a fixed screen point (the mouse cursor). */
export function zoomAt(
  view: { scale: number; x: number; y: number },
  point: { x: number; y: number },
  factor: number,
  min = 0.02,
  max = 40,
) {
  const scale = Math.max(min, Math.min(max, view.scale * factor));
  const ix = (point.x - view.x) / view.scale;
  const iy = (point.y - view.y) / view.scale;
  return { scale, x: point.x - ix * scale, y: point.y - iy * scale };
}

export const round2 = (v: number) => Math.round(v * 100) / 100;

// ------------------------------------------------------------------ polygons
export type Pt = [number, number];

export function polygonBBox(points: Pt[]): Rect {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

export function clampPolygon(points: Pt[], width: number, height: number): Pt[] {
  return points.map(([x, y]) => [Math.max(0, Math.min(width, x)), Math.max(0, Math.min(height, y))]);
}

export function translatePolygon(points: Pt[], dx: number, dy: number): Pt[] {
  return points.map(([x, y]) => [x + dx, y + dy]);
}

/** Insert a vertex on the edge closest to ``p`` (edge i runs from vertex i to the next one). */
export function insertVertex(points: Pt[], p: Pt): Pt[] {
  let best = 0;
  let bestDist = Infinity;
  points.forEach((a, i) => {
    const d = distToSegment(p, a, points[(i + 1) % points.length]);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return [...points.slice(0, best + 1), p, ...points.slice(best + 1)];
}

/** Remove a vertex; null when that would leave fewer than 3. */
export function removeVertex(points: Pt[], index: number): Pt[] | null {
  return points.length <= 3 ? null : points.filter((_, i) => i !== index);
}

function distToSegment(p: Pt, a: Pt, b: Pt): number {
  const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

export function sameBoxes(a: EBox[], b: EBox[]): boolean {
  if (a.length !== b.length) return false;
  const outline = (x: EBox) => (x.polygon ?? []).map(([px, py]) => `${round2(px)},${round2(py)}`).join(" ");
  const key = (x: EBox) => `${x.class_id}|${round2(x.x)}|${round2(x.y)}|${round2(x.w)}|${round2(x.h)}|${x.source}|${outline(x)}`;
  const ka = a.map(key).sort();
  const kb = b.map(key).sort();
  return ka.every((k, i) => k === kb[i]);
}
