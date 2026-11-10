"""Frozen-backbone patch probe, run in the model environment.

1. A frozen backbone (DINOv2; C-RADIOv4 only when enabled in local_settings.json) turns every image
   into a grid of patch features.  Features of confirmed images are cached on
   disk, so later runs only compute features for new images.
2. Every patch of a confirmed image gets a soft label from the confirmed
   annotations: the fraction of the patch each class covers, the rest
   background.  An object with a polygon covers only its outline, so the
   background inside its bounding box is not taught as object.  Confirmed
   images without boxes are pure background (negative examples).
3. A linear layer (or a small MLP) is trained on those patch labels; with
   the backbone frozen this takes seconds.
4. On a new image the probe gives a probability map per class.  The
   objectness map (1 - background) is upsampled to pixel resolution and
   thresholded; every connected region becomes one box, labelled with the
   class that has the most probability mass inside it.  Score = mean
   objectness x class purity of the region.

Input: JSON spec (see ``ProbePlugin`` in modern.py).  Output: one JSON line
per image ``{"path": ..., "boxes": [[class_index, x, y, w, h, score], ...]}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

try:  # run as a script: the module next to it
    import hf_local
except ImportError:  # imported as part of the platform (tests)
    from . import hf_local

# ------------------------------------------------------------------ backbones
# Each backbone turns a PIL image into a (h, w, D) feature grid.  Adding another
# backbone only means adding a class with the same three members.


class DinoV2Backbone:
    def __init__(self, model_id: str, long_side: int, device: str, torch):
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.model = hf_local.from_pretrained(AutoModel, model_id).to(device).eval()
        if device.startswith("cuda"):
            self.model = self.model.half()
        proc = hf_local.from_pretrained(AutoImageProcessor, model_id)
        self.mean = torch.tensor(proc.image_mean).view(3, 1, 1)
        self.std = torch.tensor(proc.image_std).view(3, 1, 1)
        self.patch = int(self.model.config.patch_size)
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.long_side = long_side
        self.key = f"{model_id}@{long_side}"

    def grid_size(self, w: int, h: int) -> tuple[int, int]:
        s = self.long_side / max(w, h)
        return max(1, round(h * s / self.patch)), max(1, round(w * s / self.patch))

    def features(self, img):
        torch = self.torch
        gh, gw = self.grid_size(*img.size)
        x = img.resize((gw * self.patch, gh * self.patch), resample=3)  # bicubic
        t = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1)
        t = ((t - self.mean) / self.std)[None].to(self.device, dtype=next(self.model.parameters()).dtype)
        with torch.no_grad():
            tokens = self.model(pixel_values=t).last_hidden_state[0, self.n_prefix :]
        return tokens.reshape(gh, gw, -1).float()


class RadioBackbone:
    """NVIDIA C-RADIO (e.g. nvidia/C-RADIOv4-SO400M). Runs the model's own code from
    Hugging Face (trust_remote_code). NVIDIA Open Model License – commercial use allowed."""

    def __init__(self, model_id: str, long_side: int, device: str, torch):
        from transformers import AutoModel, CLIPImageProcessor

        self.torch = torch
        self.device = device
        self.proc = hf_local.from_pretrained(CLIPImageProcessor, model_id)
        self.model = hf_local.from_pretrained(AutoModel, model_id, trust_remote_code=True).to(device).eval()
        self.patch = 16
        self.long_side = long_side
        self.key = f"{model_id}@{long_side}"

    def grid_size(self, w: int, h: int) -> tuple[int, int]:
        s = self.long_side / max(w, h)
        return max(1, round(h * s / self.patch)), max(1, round(w * s / self.patch))

    def features(self, img):
        torch = self.torch
        gh, gw = self.grid_size(*img.size)
        x = img.resize((gw * self.patch, gh * self.patch), resample=3)
        pv = self.proc(images=x, return_tensors="pt", do_resize=False, do_center_crop=False).pixel_values
        with torch.no_grad(), torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16,
                                             enabled=self.device.startswith("cuda")):
            _summary, spatial = self.model(pv.to(self.device))
        return spatial[0].reshape(gh, gw, -1).float()


BACKBONES = {"dinov2": DinoV2Backbone, "radio": RadioBackbone}


# ------------------------------------------------------------------ helpers
def load_rgb(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


class FeatureCache:
    """fp16 feature grids on disk, keyed by backbone + image path + file size/mtime."""

    def __init__(self, root: str | None, backbone):
        self.root = Path(root) if root else None
        self.backbone = backbone
        self.tag = hashlib.sha1(backbone.key.encode()).hexdigest()[:12]
        self.hits = self.misses = 0

    def _file(self, path: str) -> Path | None:
        if not self.root:
            return None
        st = os.stat(path)
        h = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:20]
        return self.root / self.tag / f"{h}.npy"

    def get(self, path: str, torch, device):
        f = self._file(path)
        if f is not None and f.exists():
            self.hits += 1
            return torch.from_numpy(np.load(f).astype(np.float32)).to(device)
        self.misses += 1
        img = load_rgb(path)
        feats = self.backbone.features(img)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, feats.half().cpu().numpy())
        return feats


def polygon_coverage(points, img_w, img_h, gh, gw, supersample=8):
    """Fraction (gh, gw) of every patch inside a polygon given in image pixels.

    Tests the centre of every sub-pixel, scanline by scanline.  A drawing library would also
    paint the pixels the edge touches, which inflates thin objects by a large fraction."""
    W, H = gw * supersample, gh * supersample
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0] * W / img_w, pts[:, 1] * H / img_h
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    mask = np.zeros((H, W), dtype=np.float32)
    for row in range(max(0, math.floor(y.min())), min(H, math.ceil(y.max()))):
        yc = row + 0.5
        crossing = (y <= yc) != (y2 <= yc)
        xs = np.sort(x[crossing] + (yc - y[crossing]) / (y2[crossing] - y[crossing]) * (x2[crossing] - x[crossing]))
        for a, b in zip(xs[::2], xs[1::2]):  # even-odd rule
            mask[row, max(0, math.ceil(a - 0.5)) : max(0, math.ceil(b - 0.5))] = 1
    return mask.reshape(gh, supersample, gw, supersample).mean(axis=(1, 3))


def patch_targets(boxes, img_w, img_h, gh, gw, n_cls, polygons=None):
    """Soft per-patch targets (gh, gw, n_cls+1): the fraction of the patch covered by each
    class, the rest is background. Training on coverage makes P(object) ~ coverage, so the
    0.5 contour of the upsampled map lands on the object's edge instead of a patch outside it.

    ``polygons`` has one entry per box: the object's outline, or None to use the box."""
    cover = np.zeros((gh, gw, n_cls), dtype=np.float32)
    pw, ph = img_w / gw, img_h / gh
    xs = np.arange(gw) * pw
    ys = np.arange(gh) * ph
    for i, (c, x, y, w, h) in enumerate(boxes):
        outline = polygons[i] if polygons else None
        if outline:
            covered = polygon_coverage(outline, img_w, img_h, gh, gw)
        else:
            ox = np.clip(np.minimum(xs + pw, x + w) - np.maximum(xs, x), 0, None) / pw  # (gw,)
            oy = np.clip(np.minimum(ys + ph, y + h) - np.maximum(ys, y), 0, None) / ph  # (gh,)
            covered = oy[:, None] * ox[None, :]
        cover[..., int(c)] = np.maximum(cover[..., int(c)], covered)
    total = cover.sum(-1, keepdims=True)
    cover = np.where(total > 1, cover / np.maximum(total, 1e-6), cover)  # overlapping boxes
    bg = 1.0 - cover.sum(-1, keepdims=True)
    return np.concatenate([cover, bg], -1)


def build_training_set(samples, cache, torch, device, n_cls, log):
    feats, labels = [], []
    for i, s in enumerate(samples, 1):
        f = cache.get(s["image"], torch, device)
        gh, gw, _ = f.shape
        tgt = patch_targets(s["boxes"], s["width"], s["height"], gh, gw, n_cls, s.get("polygons"))
        feats.append(f.reshape(-1, f.shape[-1]))
        labels.append(torch.from_numpy(tgt.reshape(-1, n_cls + 1)).to(device))
        if i % 10 == 0 or i == len(samples):
            log(f"features {i}/{len(samples)}")
    return torch.cat(feats), torch.cat(labels)


def train_probe(X, T, n_cls, torch, head: str, epochs: int, log):
    """X: (N, D) patch features, T: (N, n_cls+1) soft targets (rows sum to 1)."""
    mass = T.sum(0)
    log("patch mass per class: " + ", ".join(f"{v:.0f}" for v in mass.tolist()) + " (last = background)")
    if mass[:n_cls].sum() < 1:
        raise SystemExit("no object patches: the confirmed boxes are smaller than one patch; "
                         "try the high-resolution backbone")
    mean, std = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mean) / std
    D = X.shape[1]
    if head == "mlp":
        model = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                    torch.nn.Linear(256, n_cls + 1))
    else:
        model = torch.nn.Linear(D, n_cls + 1)
    model = model.to(X.device)
    # balanced class weights: background patches outnumber objects by far
    present = mass > 0
    cw = torch.where(present, mass.sum() / (mass.clamp(min=1e-6) * present.sum()), torch.zeros_like(mass))
    sample_w = (T * cw).sum(1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2 if head == "linear" else 2e-3, weight_decay=1e-2)
    n = Xn.shape[0]
    bs = min(n, 8192)
    # a few hundred steps at least: with few images there are only a handful of batches per epoch
    steps = max(400, epochs * math.ceil(n / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(0)
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g).to(X.device)
        logp = torch.log_softmax(model(Xn[idx]), -1)
        loss = (-(T[idx] * logp).sum(1) * sample_w[idx]).sum() / sample_w[idx].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"probe step {step + 1}/{steps} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        pred = model(Xn).argmax(1)
        hard = T.argmax(1)
        sure = T.max(1).values >= 0.5
        fg = sure & (hard < n_cls)
        bg = sure & (hard == n_cls)
        acc_fg = (pred[fg] == hard[fg]).float().mean().item() if fg.any() else float("nan")
        acc_bg = (pred[bg] == hard[bg]).float().mean().item() if bg.any() else float("nan")
        log(f"train accuracy: objects {acc_fg:.3f}, background {acc_bg:.3f}")
    return model, mean, std


def boxes_from_probs(probs, img_w, img_h, n_cls, pixel_thr, min_area, torch):
    """probs: (n_cls+1, gh, gw) softmax -> list of [cls, x, y, w, h, score] in pixels.

    Regions are found on the objectness map (1 - P(background)) first and only then
    assigned a class, so an object whose patches disagree about the class still
    becomes one box instead of several class fragments."""
    from scipy import ndimage

    # work at most ~1024 px on the long side, then scale back
    scale = min(1.0, 1024 / max(img_w, img_h))
    W, H = max(1, round(img_w * scale)), max(1, round(img_h * scale))
    up = torch.nn.functional.interpolate(probs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    up = up.cpu().numpy()
    obj = 1.0 - up[n_cls]
    mask = obj >= pixel_thr
    if not mask.any():
        return []
    lab, _n = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        region = lab[sl] == k
        area = region.sum() / (scale * scale)
        if area < min_area:
            continue
        cls_mass = up[:n_cls][:, sl[0], sl[1]][:, region].sum(1)  # (n_cls,)
        c = int(cls_mass.argmax())
        objectness = float(obj[sl][region].mean())
        purity = float(cls_mass[c] / max(cls_mass.sum(), 1e-6))  # how sure the class is
        score = objectness * purity
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append([c, round(x0 / scale, 2), round(y0 / scale, 2), round((x1 - x0) / scale, 2),
                    round((y1 - y0) / scale, 2), round(score, 4)])
    return out


def object_areas(samples):
    """Area of every confirmed object: its outline where it has one, otherwise its box."""
    areas = []
    for s in samples:
        polygons = s.get("polygons") or [None] * len(s["boxes"])
        for (_c, _x, _y, w, h), outline in zip(s["boxes"], polygons):
            if outline:
                xs, ys = np.array(outline, dtype=np.float64).T
                areas.append(float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2))
            else:
                areas.append(w * h)
    return areas


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred, truth, thr=0.5):
    """Greedy matching at IoU >= thr, same class. Returns (tp, fp, fn)."""
    tp = fp = 0
    used = set()
    for c, *box, _s in sorted(pred, key=lambda r: -r[-1]):
        best, best_j = 0.0, -1
        for j, (tc, *tbox) in enumerate(truth):
            if j in used or tc != c:
                continue
            v = iou(box, tbox)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            tp += 1
            used.add(best_j)
        else:
            fp += 1
    return tp, fp, len(truth) - len(used)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    import torch

    def log(msg):
        print(msg, flush=True)

    req = spec.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if req in ("auto", "", None) else (
        "cpu" if req == "cpu" else f"cuda:{str(req).split(',')[0]}")
    random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()
    log(f"torch {torch.__version__}, device {device}")
    log(f"loading {spec['model_id']} (first run downloads it from Hugging Face)")
    backbone = BACKBONES[spec["backbone"]](spec["model_id"], int(spec["long_side"]), device, torch)
    cache = FeatureCache(spec.get("cache_dir"), backbone)
    n_cls = len(spec["class_names"])
    samples = spec["training"]
    outlined = sum(1 for s in samples for p in (s.get("polygons") or []) if p)
    log(f"{len(samples)} confirmed images ({sum(len(s['boxes']) for s in samples)} objects, "
        f"{outlined} trained on their outline), grid {backbone.patch}px patches at {backbone.long_side}px")

    head, epochs = spec.get("head", "linear"), int(spec.get("epochs", 30))
    pixel_thr = float(spec.get("pixel_threshold", 0.5))

    def min_area_from(train):
        # a thin diagonal object fills far less than its box, and would be dropped by a box-based limit
        areas = object_areas(train)
        return 0.15 * min(areas) if areas else 16.0

    # ---- quick self-check on held-out confirmed images (only when there are enough)
    if spec.get("self_check", True) and len(samples) >= 6:
        rng = random.Random(0)
        order = list(range(len(samples)))
        rng.shuffle(order)
        k = max(1, len(samples) // 5)
        hold, train = [samples[i] for i in order[:k]], [samples[i] for i in order[k:]]
        if any(s["boxes"] for s in train):
            X, y = build_training_set(train, cache, torch, device, n_cls, lambda m: None)
            model, mean, std = train_probe(X, y, n_cls, torch, head, epochs, lambda m: None)
            tp = fp = fn = 0
            for s in hold:
                f = cache.get(s["image"], torch, device)
                with torch.no_grad():
                    pr = torch.softmax(model(((f - mean) / std)), -1).permute(2, 0, 1)
                pred = [b for b in boxes_from_probs(pr, s["width"], s["height"], n_cls, pixel_thr,
                                                    min_area_from(train), torch)
                        if b[-1] >= float(spec["confidence"])]
                a, b, c = evaluate(pred, [list(map(float, bb)) for bb in s["boxes"]])
                tp, fp, fn = tp + a, fp + b, fn + c
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            log(f"SELF-CHECK on {len(hold)} held-out confirmed images (IoU>=0.5): recall {rec:.2f}, "
                f"precision {prec:.2f}  (tp {tp}, false positives {fp}, missed {fn})")

    # ---- train on everything
    X, y = build_training_set(samples, cache, torch, device, n_cls, log)
    model, mean, std = train_probe(X, y, n_cls, torch, head, epochs, log)
    log(f"feature cache: {cache.hits} reused, {cache.misses} computed; training took {time.time() - t0:.1f}s")
    min_area = min_area_from(samples)

    # ---- predict (features of unconfirmed images are not cached: they change status soon)
    images = spec["images"]
    no_cache = FeatureCache(None, backbone)
    conf = float(spec["confidence"])
    with open(spec["out"], "w", encoding="utf-8") as out:
        for i, path in enumerate(images, 1):
            img = load_rgb(path)
            f = no_cache.get(path, torch, device)
            with torch.no_grad():
                pr = torch.softmax(model((f - mean) / std), -1).permute(2, 0, 1)
            boxes = [b for b in boxes_from_probs(pr, img.size[0], img.size[1], n_cls, pixel_thr, min_area, torch)
                     if b[-1] >= conf]
            out.write(json.dumps({"path": path, "boxes": boxes}) + "\n")
            out.flush()
            print(f"[{i}/{len(images)}]", flush=True)
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
