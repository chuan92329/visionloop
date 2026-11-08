"""Fake model used by the tests – NOT a real model.

Exercises the whole loop (auto-label -> review -> split -> train ->
auto-label again) without PyTorch or a GPU.  It is not part of the platform.

* "Prediction" finds bright blobs on a darker background with a simple
  threshold + connected components.
* "Training" learns the average box size / aspect ratio of every class;
  prediction then gives each blob the class whose shape is closest.  Before
  any training, classes are guessed at random.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageFilter, ImageStat

from app.plugins.base import ModelPlugin, PredictContext, Prediction, TrainContext, TrainResult, Variant

WORK_WIDTH = 320  # blobs are found on a downscaled copy


class DemoPlugin(ModelPlugin):
    id = "demo"
    name = "Demo（簡易亮點偵測，測試流程用）"
    description = "不是真的模型：用亮度門檻找出物件、用框的形狀猜類別，只用來測試自動標註、審核與訓練流程，不需要 GPU 或 PyTorch。"
    license = "MIT"
    homepage = ""
    dataset_format = "yolo"
    needs_weights = False
    variants = [
        Variant("fast", "Fast", "模擬訓練，每個 epoch 0.05 秒", 0.0),
        Variant("slow", "Slow", "模擬訓練，每個 epoch 0.5 秒（方便觀察進度條）", 0.0),
    ]

    # ------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> TrainResult:
        shapes = _class_shapes(ctx.dataset_dir / "labels" / "train")
        delay = 0.5 if ctx.variant == "slow" else 0.05
        for epoch in range(ctx.epochs):
            ctx.hooks.check_cancel()
            time.sleep(delay)
            ctx.hooks.progress((epoch + 1) / ctx.epochs, f"epoch {epoch + 1}/{ctx.epochs}")
        ctx.hooks.log(f"learned box shapes for {len(shapes)} classes")

        # "validation": how often the nearest-shape rule picks the right class on valid boxes
        correct = total = 0
        for c, items in _class_items(ctx.dataset_dir / "labels" / "valid").items():
            for w, h in items:
                total += 1
                correct += _nearest(shapes, w, h) == c
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        weights = ctx.output_dir / "demo_weights.json"
        weights.write_text(json.dumps({"shapes": shapes, "classes": ctx.class_names}))
        metrics = {"note": "demo plugin"}
        if total:
            metrics["class_accuracy"] = round(correct / total, 3)
        return TrainResult(weights, metrics)

    # ---------------------------------------------------------- predict
    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        shapes: dict[str, list[float]] = {}
        if ctx.weights_path and ctx.weights_path.exists():
            shapes = json.loads(ctx.weights_path.read_text()).get("shapes", {})
        n_cls = max(len(ctx.class_names), 1)
        for path in images:
            ctx.hooks.check_cancel()
            rng = random.Random(int(hashlib.md5(path.name.encode()).hexdigest()[:8], 16))
            preds = []
            for x, y, w, h, contrast in _blobs(path):
                with PILImage.open(path) as im:
                    W, H = im.size
                if shapes:
                    cls = _nearest(shapes, w / W, h / H)
                    score = 0.55 + 0.4 * contrast + rng.uniform(-0.12, 0.05)
                else:
                    cls = rng.randrange(n_cls)
                    score = 0.25 + 0.35 * contrast + rng.uniform(-0.05, 0.1)
                score = round(max(0.05, min(0.99, score)), 3)
                if score >= ctx.confidence:
                    # a little localisation noise, like a real detector
                    jitter = lambda v, s: v + rng.uniform(-0.04, 0.04) * s  # noqa: E731
                    preds.append(Prediction(cls, jitter(x, w), jitter(y, h), w * rng.uniform(0.96, 1.06), h * rng.uniform(0.96, 1.06), score))
            yield path, preds


# ------------------------------------------------------------------ helpers
def _class_items(label_dir: Path) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                out.setdefault(int(p[0]), []).append((float(p[3]), float(p[4])))
    return out


def _class_shapes(label_dir: Path) -> dict[str, list[float]]:
    """Mean of (log w, log h) per class, in normalised image units."""
    shapes = {}
    for c, items in _class_items(label_dir).items():
        lw = sum(math.log(max(w, 1e-4)) for w, _ in items) / len(items)
        lh = sum(math.log(max(h, 1e-4)) for _, h in items) / len(items)
        shapes[str(c)] = [lw, lh]
    return shapes


def _nearest(shapes: dict[str, list[float]], w: float, h: float) -> int:
    lw, lh = math.log(max(w, 1e-4)), math.log(max(h, 1e-4))
    best = min(shapes.items(), key=lambda kv: (kv[1][0] - lw) ** 2 + (kv[1][1] - lh) ** 2)
    return int(best[0])


def _blobs(path: Path) -> list[tuple[float, float, float, float, float]]:
    """Bright connected regions -> (x, y, w, h, contrast 0..1) in original pixels."""
    with PILImage.open(path) as im:
        W, H = im.size
        scale = WORK_WIDTH / W if W > WORK_WIDTH else 1.0
        g = im.convert("L").resize((max(1, int(W * scale)), max(1, int(H * scale))))
    g = g.filter(ImageFilter.MedianFilter(3))
    stat = ImageStat.Stat(g)
    mean, std = stat.mean[0], stat.stddev[0]
    thr = mean + 1.1 * std
    w, h = g.size
    px = g.load()
    seen = bytearray(w * h)
    found = []
    for y0 in range(h):
        for x0 in range(w):
            i0 = y0 * w + x0
            if seen[i0] or px[x0, y0] <= thr:
                continue
            q = deque([(x0, y0)])
            seen[i0] = 1
            minx = maxx = x0
            miny = maxy = y0
            area, total = 0, 0.0
            while q:
                x, y = q.popleft()
                area += 1
                total += px[x, y]
                minx, maxx, miny, maxy = min(minx, x), max(maxx, x), min(miny, y), max(maxy, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        j = ny * w + nx
                        if not seen[j] and px[nx, ny] > thr:
                            seen[j] = 1
                            q.append((nx, ny))
            if area < 25:
                continue
            contrast = min(1.0, (total / area - mean) / (255 - mean + 1e-6) * 1.6)
            found.append(
                (minx / scale, miny / scale, (maxx - minx + 1) / scale, (maxy - miny + 1) / scale, contrast)
            )
    return found
