"""What a model predicted, kept the way YOLO's own predict leaves it.

A labeling job with a model writes ``data/runs/job_NNNNN/predict/``:

* ``images/``   every image with its boxes, class names and scores drawn on it
* ``labels/``   one YOLO txt per image: ``class cx cy w h`` (normalised, no score)
* ``classes.txt`` the class names, line number = class index in the txt files

It is the result at the time the job ran; later edits in the annotator do not change it.
The folder goes when the job's record is deleted.
"""

from __future__ import annotations


import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps

from ..config import settings

PREDICT_DIR = "predict"

# fonts that can draw Chinese class names; the first one found is used
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


@dataclass
class PredictedBox:
    class_index: int  # index into the class list written to classes.txt
    x: float
    y: float
    w: float
    h: float
    score: float
    polygon: list[list[float]] | None = None


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow before 10.1
        return ImageFont.load_default()


def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return 59, 130, 246


def _text_on(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def yolo_lines(boxes: Iterable[PredictedBox], width: int, height: int) -> list[str]:
    lines = []
    for b in boxes:
        cx = min(max((b.x + b.w / 2) / width, 0.0), 1.0)
        cy = min(max((b.y + b.h / 2) / height, 0.0), 1.0)
        w = min(max(b.w / width, 0.0), 1.0)
        h = min(max(b.h / height, 0.0), 1.0)
        lines.append(f"{b.class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


class PredictionWriter:
    """Collects one job's predictions into ``<work_dir>/predict``."""

    def __init__(self, work_dir: Path, class_names: list[str], colors: list[str]) -> None:
        self.folder = work_dir / PREDICT_DIR
        (self.folder / "images").mkdir(parents=True, exist_ok=True)
        (self.folder / "labels").mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.colors = [_rgb(c) for c in colors]
        self.count = 0
        (self.folder / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    def add(self, source: Path, stem: str, boxes: list[PredictedBox]) -> None:
        with PILImage.open(source) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        lines = yolo_lines(boxes, img.width, img.height)
        (self.folder / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        self._draw(img, boxes)
        img.save(self.folder / "images" / f"{stem}.jpg", quality=90)
        self.count += 1

    def _draw(self, img: PILImage.Image, boxes: list[PredictedBox]) -> None:
        if not boxes:
            return
        side = max(img.width, img.height)
        line = max(2, round(side / 400))
        font = _font(max(12, round(side / 60)))
        draw = ImageDraw.Draw(img)
        for b in boxes:
            color = self.colors[b.class_index % len(self.colors)] if self.colors else (59, 130, 246)
            if b.polygon and len(b.polygon) >= 3:
                draw.polygon([tuple(p) for p in b.polygon], outline=color, width=line)
            draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=line)
            name = self.class_names[b.class_index] if b.class_index < len(self.class_names) else str(b.class_index)
            label = f"{name} {b.score:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw, th = right - left, bottom - top
            pad = max(2, line)
            ty = b.y - th - 2 * pad if b.y - th - 2 * pad >= 0 else b.y
            draw.rectangle([b.x, ty, b.x + tw + 2 * pad, ty + th + 2 * pad], fill=color)
            draw.text((b.x + pad - left, ty + pad - top), label, fill=_text_on(color), font=font)


def predict_folder(result: dict | None) -> Path | None:
    """The job's prediction folder, if it wrote one and it is still there."""
    name = (result or {}).get("predict_dir")
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    folder = (settings.runs_dir / name / PREDICT_DIR).resolve()
    if settings.runs_dir.resolve() not in folder.parents or not folder.is_dir():
        return None
    return folder


def zip_folder(folder: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                arc = path.relative_to(folder).as_posix()
                # the drawn images are already compressed
                kind = zipfile.ZIP_STORED if path.suffix.lower() == ".jpg" else zipfile.ZIP_DEFLATED
                zf.write(path, arc, compress_type=kind)
    return dest
