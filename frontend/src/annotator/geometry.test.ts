import { describe, expect, it } from "vitest";
import {
  clampPolygon,
  clampRect,
  fitView,
  insertVertex,
  keepInside,
  polygonBBox,
  rectFromPoints,
  removeVertex,
  sameBoxes,
  translatePolygon,
  zoomAt,
  type EBox,
  type Pt,
} from "./geometry";
import { History } from "./history";

describe("polygons", () => {
  const square: Pt[] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
  ];

  it("derives the bounding box", () => {
    expect(polygonBBox([[5, 8], [20, 2], [9, 30]])).toEqual({ x: 5, y: 2, w: 15, h: 28 });
  });

  it("moves vertices inside the image", () => {
    expect(clampPolygon([[-5, 3], [120, 50], [10, 90]], 100, 80)).toEqual([[0, 3], [100, 50], [10, 80]]);
  });

  it("translates every vertex", () => {
    expect(translatePolygon([[1, 2], [3, 4], [5, 6]], 10, -1)).toEqual([[11, 1], [13, 3], [15, 5]]);
  });

  it("inserts a vertex on the closest edge, including the closing one", () => {
    expect(insertVertex(square, [5, 1])).toEqual([[0, 0], [5, 1], [10, 0], [10, 10], [0, 10]]);
    expect(insertVertex(square, [1, 5])).toEqual([[0, 0], [10, 0], [10, 10], [0, 10], [1, 5]]);
  });

  it("keeps at least three vertices", () => {
    expect(removeVertex(square, 1)).toEqual([[0, 0], [10, 10], [0, 10]]);
    expect(removeVertex(square.slice(0, 3), 0)).toBeNull();
  });

  it("notices a changed outline even when the bounding box is the same", () => {
    const base = { uid: "a", class_id: 1, x: 0, y: 0, w: 10, h: 10, source: "manual", score: null } as const;
    const a: EBox = { ...base, polygon: [[0, 0], [10, 0], [5, 10]] };
    const b: EBox = { ...base, polygon: [[0, 0], [10, 0], [6, 10]] };
    expect(sameBoxes([a], [b])).toBe(false);
    expect(sameBoxes([a], [{ ...a, uid: "c" }])).toBe(true);
    expect(sameBoxes([a], [{ ...base }])).toBe(false);
  });
});

describe("geometry", () => {
  it("builds a rect from points dragged in any direction", () => {
    expect(rectFromPoints({ x: 50, y: 40 }, { x: 10, y: 60 })).toEqual({ x: 10, y: 40, w: 40, h: 20 });
  });

  it("clamps to the image and drops degenerate boxes", () => {
    expect(clampRect({ x: -10, y: 5, w: 30, h: 10 }, 100, 100)).toEqual({ x: 0, y: 5, w: 20, h: 10 });
    expect(clampRect({ x: 120, y: 5, w: 30, h: 10 }, 100, 100)).toBeNull();
  });

  it("keeps a moved box inside without resizing it", () => {
    expect(keepInside({ x: 90, y: -5, w: 20, h: 10 }, 100, 100)).toEqual({ x: 80, y: 0, w: 20, h: 10 });
  });

  it("fits and centres an image", () => {
    const v = fitView(200, 100, 400, 400, 1);
    expect(v.scale).toBe(2);
    expect(v).toMatchObject({ x: 0, y: 100 });
  });

  it("zooms around the cursor keeping the point under it fixed", () => {
    const v = { scale: 1, x: 0, y: 0 };
    const z = zoomAt(v, { x: 50, y: 50 }, 2);
    expect(z).toEqual({ scale: 2, x: -50, y: -50 });
  });

  it("compares box sets ignoring order and ui flags", () => {
    const a: EBox = { uid: "1", class_id: 1, x: 1, y: 2, w: 3, h: 4, source: "manual", score: null };
    const b: EBox = { ...a, uid: "2", x: 10 };
    expect(sameBoxes([a, b], [{ ...b, hidden: true }, a])).toBe(true);
    expect(sameBoxes([a], [b])).toBe(false);
  });
});

describe("history", () => {
  it("undoes and redoes", () => {
    const h = new History<number>();
    h.push(1);
    h.push(2);
    expect(h.undo(3)).toBe(2);
    expect(h.undo(2)).toBe(1);
    expect(h.redo(1)).toBe(2);
    h.push(5); // new change clears the redo stack
    expect(h.canRedo).toBe(false);
  });
});
