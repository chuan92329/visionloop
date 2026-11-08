"""Generate a small synthetic dataset and load it into a running VisionLoop.

The images show bolts, nuts and washers on a conveyor belt.  They are
drawn procedurally, so the dataset has no licence restrictions and the
ground-truth boxes are known exactly.

    python scripts/demo_data.py                       # server on http://127.0.0.1:8000
    python scripts/demo_data.py --url http://host:8000 --images 60

What it does:
  1. creates the project "Conveyor parts (demo)" with classes bolt / nut / washer
  2. uploads the images; consecutive frames share a group key (cam1_..., cam2_...)
  3. confirms the first 40% with the true boxes (as if a person labelled them)
  4. splits them; the rest stay unlabeled, ready for auto-labelling

Only depends on Pillow and the standard library.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFilter

W, H = 960, 600
CLASSES = ["bolt", "nut", "washer"]


# --------------------------------------------------------------- drawing
def belt(rng: random.Random) -> Image.Image:
    base = rng.randint(70, 95)
    img = Image.new("RGB", (W, H), (base, base + 3, base + 8))
    d = ImageDraw.Draw(img)
    for y in range(0, H, 6):  # belt ribs
        shade = base - 8 + rng.randint(-3, 3)
        d.line([(0, y), (W, y)], fill=(shade, shade + 2, shade + 6), width=2)
    for _ in range(2500):  # grit
        x, y = rng.randrange(W), rng.randrange(H)
        v = base + rng.randint(-25, 25)
        d.point((x, y), fill=(v, v, v + 4))
    d.rectangle([0, 0, W, 26], fill=(38, 40, 44))  # belt rails
    d.rectangle([0, H - 26, W, H], fill=(38, 40, 44))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def metal(rng: random.Random) -> tuple[int, int, int]:
    v = rng.randint(150, 205)
    tint = rng.choice([(0, 0, 0), (12, 8, -6), (-6, 0, 10)])  # steel / brass-ish / zinc
    return (v + tint[0], v + tint[1], v + tint[2])


def hexagon(cx, cy, r, rot):
    return [(cx + r * math.cos(rot + k * math.pi / 3), cy + r * math.sin(rot + k * math.pi / 3)) for k in range(6)]


def draw_part(d: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, s: float, rng: random.Random):
    col = metal(rng)
    dark = tuple(max(0, c - 70) for c in col)
    shadow = (40, 42, 46)
    rot = rng.uniform(0, math.pi)
    if kind == "nut":
        d.polygon(hexagon(cx + 4, cy + 4, s, rot), fill=shadow)
        d.polygon(hexagon(cx, cy, s, rot), fill=col, outline=dark)
        r = s * 0.45
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 62, 66), outline=dark)
        return [cx - s, cy - s, cx + s, cy + s]
    if kind == "washer":
        d.ellipse([cx - s + 4, cy - s + 4, cx + s + 4, cy + s + 4], fill=shadow)
        d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=col, outline=dark, width=2)
        r = s * 0.42
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(78, 80, 86), outline=dark)
        return [cx - s, cy - s, cx + s, cy + s]
    # bolt: hex head + threaded shaft, horizontal or vertical
    length = s * rng.uniform(2.6, 3.6)
    thick = s * 0.42
    horizontal = rng.random() < 0.5
    head = s * 0.75
    if horizontal:
        x0 = cx - length / 2
        d.rectangle([x0 + head, cy - thick / 2 + 3, x0 + length + 3, cy + thick / 2 + 3], fill=shadow)
        d.rectangle([x0 + head, cy - thick / 2, x0 + length, cy + thick / 2], fill=col, outline=dark)
        for tx in range(int(x0 + head + 6), int(x0 + length), 6):
            d.line([(tx, cy - thick / 2), (tx - 3, cy + thick / 2)], fill=dark)
        d.polygon(hexagon(x0 + head / 2, cy, head, math.pi / 6), fill=col, outline=dark)
        return [x0 - head / 2, cy - head, x0 + length, cy + head]
    y0 = cy - length / 2
    d.rectangle([cx - thick / 2 + 3, y0 + head, cx + thick / 2 + 3, y0 + length + 3], fill=shadow)
    d.rectangle([cx - thick / 2, y0 + head, cx + thick / 2, y0 + length], fill=col, outline=dark)
    for ty in range(int(y0 + head + 6), int(y0 + length), 6):
        d.line([(cx - thick / 2, ty), (cx + thick / 2, ty - 3)], fill=dark)
    d.polygon(hexagon(cx, y0 + head / 2, head, 0), fill=col, outline=dark)
    return [cx - head, y0 - head / 2, cx + head, y0 + length]


def make_image(seed: int) -> tuple[bytes, list[dict]]:
    rng = random.Random(seed)
    img = belt(rng)
    d = ImageDraw.Draw(img)
    boxes, placed = [], []
    for _ in range(rng.randint(3, 7)):
        kind = rng.choices(CLASSES, weights=[4, 3, 2])[0]
        s = rng.uniform(26, 48)
        for _try in range(30):
            cx, cy = rng.uniform(90, W - 90), rng.uniform(110, H - 110)
            if all(math.hypot(cx - px, cy - py) > (s + ps) * 2.2 for px, py, ps in placed):
                break
        placed.append((cx, cy, s))
        x1, y1, x2, y2 = draw_part(d, kind, cx, cy, s, rng)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        boxes.append({"class": kind, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), boxes


# ------------------------------------------------------------------- API
def call(base: str, method: str, path: str, body=None, files=None):
    headers = {}
    data = None
    if files is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for name, (fname, content, ctype) in files:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n"
            )
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"null")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--images", type=int, default=48)
    ap.add_argument("--reviewed", type=float, default=0.4, help="fraction confirmed with ground truth")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    project = call(base, "POST", "/api/projects", {
        "name": "Conveyor parts (demo)",
        "description": "合成的輸送帶零件影像：螺栓、螺帽、墊圈",
        "classes": CLASSES,
    })
    pid = project["id"]
    class_id = {c["name"]: c["id"] for c in project["classes"]}
    print(f"project #{pid} created")

    truth: dict[int, list[dict]] = {}
    for start in range(0, args.images, 12):
        files, metas = [], []
        for i in range(start, min(start + 12, args.images)):
            cam = 1 + (i // 4) % 3  # 4 consecutive frames per camera clip
            name = f"cam{cam}_clip{i // 4:02d}_{i % 4:04d}.jpg"
            content, boxes = make_image(1000 + i)
            files.append(("files", (name, content, "image/jpeg")))
            metas.append(boxes)
        r = call(base, "POST", f"/api/projects/{pid}/images", files=files)
        for img, boxes in zip(r["created"], metas):
            truth[img["id"]] = boxes
    print(f"uploaded {len(truth)} images")

    call(base, "POST", f"/api/projects/{pid}/auto-group", {"pattern": r"^(.*?)[_\-\s]*\d+$"})

    ids = sorted(truth)
    n_review = int(len(ids) * args.reviewed)
    for img_id in ids[:n_review]:
        call(base, "PUT", f"/api/images/{img_id}/annotations", {
            "boxes": [{"class_id": class_id[b["class"]], "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
                      for b in truth[img_id]],
            "mark_reviewed": True,
        })
    print(f"confirmed {n_review} images with ground truth")

    stats = call(base, "POST", f"/api/projects/{pid}/split", {"lock_test": True})
    print(f"split: {stats['images']}")
    print(f"open {base}/projects/{pid}")


if __name__ == "__main__":
    main()
