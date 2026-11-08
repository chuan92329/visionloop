"""Dataset level operations: split, statistics and export (YOLO / COCO)."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import SPLITS, STATUS_REVIEWED, Annotation, Image, LabelClass, Project, utcnow
from . import storage
from .annotations import polygon_area
from .split import SplitItem, plan_split


# ------------------------------------------------------------------ split
def run_split(
    db: Session,
    project: Project,
    *,
    ratios: dict[str, float],
    seed: int,
    mode: str,
    stratify: bool,
    lock_test: bool | None,
) -> list[str]:
    images = db.scalars(select(Image).where(Image.project_id == project.id)).all()
    counts: dict[int, Counter] = {img.id: Counter() for img in images}
    for img_id, cls_id in db.execute(
        select(Annotation.image_id, Annotation.class_id)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project.id)
    ):
        counts[img_id][cls_id] += 1

    items = [
        SplitItem(
            image_id=img.id,
            group_key=img.group_key,
            current_split=img.split,
            reviewed=img.status == STATUS_REVIEWED,
            class_counts=counts[img.id],
        )
        for img in images
    ]
    class_names = {c.id: c.name for c in project.classes}
    plan = plan_split(
        items,
        ratios,
        seed=seed,
        mode=mode,
        stratify=stratify,
        test_locked=project.test_locked,
        class_names=class_names,
    )
    by_id = {img.id: img for img in images}
    for img_id, split in plan.assignments.items():
        by_id[img_id].split = split

    project.split_config = {**ratios, "seed": seed}
    if lock_test is True and not project.test_locked:
        has_test = any(img.split == "test" for img in images)
        if has_test:
            project.test_locked = True
            project.test_locked_at = utcnow()
        else:
            plan.warnings.append("test 集合是空的，暫不鎖定")
    elif lock_test is False:
        project.test_locked = False
        project.test_locked_at = None
    return plan.warnings


def split_stats(db: Session, project: Project) -> dict:
    class_names = {c.id: c.name for c in project.classes}
    images = Counter()
    unassigned_reviewed = 0
    not_reviewed = 0
    for split, status in db.execute(
        select(Image.split, Image.status).where(Image.project_id == project.id)
    ):
        if status != STATUS_REVIEWED:
            not_reviewed += 1
        elif split is None:
            unassigned_reviewed += 1
        else:
            images[split] += 1

    boxes: dict[str, dict[str, int]] = {s: {n: 0 for n in class_names.values()} for s in SPLITS}
    polygons = boxes_without_polygon = 0
    for split, cls_id, polygon in db.execute(
        select(Image.split, Annotation.class_id, Annotation.polygon)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project.id, Image.status == STATUS_REVIEWED, Image.split.is_not(None))
    ):
        boxes[split][class_names.get(cls_id, "?")] += 1
        if polygon:
            polygons += 1
        else:
            boxes_without_polygon += 1

    return {
        "images": {s: images[s] for s in SPLITS},
        "boxes": boxes,
        "polygons": polygons,
        "boxes_without_polygon": boxes_without_polygon,
        "unassigned_reviewed": unassigned_reviewed,
        "not_reviewed": not_reviewed,
        "test_locked": project.test_locked,
    }


def auto_group(db: Session, project: Project, pattern: str, overwrite: bool) -> int:
    rx = re.compile(pattern)
    if rx.groups < 1:
        raise ValueError("規則運算式必須包含一個括號群組 ( )")
    n = 0
    for img in db.scalars(select(Image).where(Image.project_id == project.id)):
        if img.group_key and not overwrite:
            continue
        m = rx.match(Path(img.filename).stem)
        if m and m.group(1):
            img.group_key = m.group(1)
            n += 1
    return n


# ----------------------------------------------------------------- export
@dataclass
class ExportEntry:
    arcname: str
    source: Path | None = None
    data: bytes | None = None


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-.]+", "_", Path(name).stem, flags=re.UNICODE).strip("._")
    return stem[:80] or "image"


def export_filename(img: Image) -> str:
    return f"{img.id:06d}_{_safe_stem(img.filename)}{Path(img.stored_name).suffix}"


def _dataset_images(db: Session, project: Project, include_splits: tuple[str, ...]) -> list[Image]:
    return list(
        db.scalars(
            select(Image)
            .options(selectinload(Image.annotations))
            .where(
                Image.project_id == project.id,
                Image.status == STATUS_REVIEWED,
                Image.split.in_(include_splits),
            )
            .order_by(Image.id)
        )
    )


def _outline(a: Annotation) -> list[list[float]]:
    """The object's polygon; a plain box counts as a rectangle."""
    return a.polygon or [[a.x, a.y], [a.x + a.w, a.y], [a.x + a.w, a.y + a.h], [a.x, a.y + a.h]]


def iter_yolo(
    db: Session, project: Project, *, dataset_root: str = ".", segmentation: bool = False
) -> Iterator[ExportEntry]:
    """``segmentation``: label lines are ``class x1 y1 x2 y2 ...`` instead of ``class cx cy w h``."""
    classes: list[LabelClass] = list(project.classes)
    index = {c.id: i for i, c in enumerate(classes)}
    images = _dataset_images(db, project, SPLITS)
    if not images:
        raise ValueError("沒有已確認且已切分的圖片可以匯出，請先確認標註並執行資料切分")
    present = {img.split for img in images}

    data_yaml = {
        "path": dataset_root,
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test" if "test" in present else None,
        "nc": len(classes),
        "names": [c.name for c in classes],
    }
    if data_yaml["test"] is None:
        del data_yaml["test"]
    header = (
        "# Exported by VisionLoop\n"
        "# 若訓練框架找不到資料，請把 path 改成這個資料夾的絕對路徑\n"
    )
    yield ExportEntry(
        "data.yaml",
        data=(header + yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False)).encode(),
    )

    for img in images:
        fname = export_filename(img)
        yield ExportEntry(f"images/{img.split}/{fname}", source=storage.image_path(project.id, img.stored_name))
        lines = []
        for a in img.annotations:
            if segmentation:
                coords = " ".join(f"{x / img.width:.6f} {y / img.height:.6f}" for x, y in _outline(a))
                lines.append(f"{index[a.class_id]} {coords}")
                continue
            cx = (a.x + a.w / 2) / img.width
            cy = (a.y + a.h / 2) / img.height
            lines.append(
                f"{index[a.class_id]} {cx:.6f} {cy:.6f} {a.w / img.width:.6f} {a.h / img.height:.6f}"
            )
        yield ExportEntry(
            f"labels/{img.split}/{Path(fname).stem}.txt",
            data=("\n".join(lines) + ("\n" if lines else "")).encode(),
        )


def coco_dict(project: Project, images: list[Image], *, category_id_offset: int = 1) -> dict:
    classes: list[LabelClass] = list(project.classes)
    cat_id = {c.id: i + category_id_offset for i, c in enumerate(classes)}
    out_images, out_anns = [], []
    ann_id = 1
    for img in images:
        out_images.append(
            {"id": img.id, "file_name": export_filename(img), "width": img.width, "height": img.height}
        )
        for a in img.annotations:
            ann = {
                "id": ann_id,
                "image_id": img.id,
                "category_id": cat_id[a.class_id],
                "bbox": [round(a.x, 2), round(a.y, 2), round(a.w, 2), round(a.h, 2)],
                "area": round(polygon_area(a.polygon) if a.polygon else a.w * a.h, 2),
                "iscrowd": 0,
            }
            if a.polygon:
                ann["segmentation"] = [[round(v, 2) for p in a.polygon for v in p]]
            out_anns.append(ann)
            ann_id += 1
    return {
        "info": {"description": f"{project.name} (VisionLoop export)", "date_created": utcnow().isoformat()},
        "licenses": [],
        "images": out_images,
        "annotations": out_anns,
        "categories": [
            {"id": cat_id[c.id], "name": c.name, "supercategory": "object"} for c in classes
        ],
    }


def iter_coco(db: Session, project: Project, *, category_id_offset: int = 1) -> Iterator[ExportEntry]:
    images = _dataset_images(db, project, SPLITS)
    if not images:
        raise ValueError("沒有已確認且已切分的圖片可以匯出，請先確認標註並執行資料切分")
    for split in SPLITS:
        subset = [img for img in images if img.split == split]
        if not subset:
            continue
        doc = coco_dict(project, subset, category_id_offset=category_id_offset)
        yield ExportEntry(
            f"annotations/instances_{split}.json",
            data=json.dumps(doc, ensure_ascii=False, indent=1).encode(),
        )
        for img in subset:
            yield ExportEntry(
                f"images/{split}/{export_filename(img)}",
                source=storage.image_path(project.id, img.stored_name),
            )


def write_zip(entries: Iterator[ExportEntry], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for e in entries:
            if e.source is not None:
                # images are already compressed -> store them
                zf.write(e.source, e.arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.writestr(e.arcname, e.data or b"")
    return dest


def write_dir(entries: Iterator[ExportEntry], dest: Path, *, link: bool = True) -> Path:
    """Materialise a dataset on disk (used to feed training plugins).

    Images are hard-linked when possible so large datasets cost no extra space.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for e in entries:
        target = dest / e.arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        if e.source is not None:
            if link:
                try:
                    os.link(e.source, target)
                    continue
                except OSError:
                    pass
            shutil.copy2(e.source, target)
        else:
            target.write_bytes(e.data or b"")
    return dest
