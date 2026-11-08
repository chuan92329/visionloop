"""Single place where an image's boxes are replaced.

Every change (manual save, review, auto-label, revert) goes through
``apply_boxes`` so that status rules, box clamping and the revision history
stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    STATUS_AUTO,
    STATUS_IN_PROGRESS,
    STATUS_REVIEWED,
    STATUS_UNLABELED,
    Annotation,
    Image,
    LabelClass,
    Revision,
    utcnow,
)

Action = Literal["save", "review", "auto_label", "revert", "clear"]

MIN_BOX_SIZE = 1.0  # pixels


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    source: str = "manual"
    score: float | None = None
    polygon: list[list[float]] | None = None  # [[x, y], ...]; x/y/w/h are then its bounding box


class AnnotationError(ValueError):
    pass


class VersionConflict(AnnotationError):
    pass


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula."""
    return abs(
        sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, points[1:] + points[:1]))
    ) / 2


def clamp_polygon(points: list[list[float]], width: int, height: int) -> list[list[float]] | None:
    """Move vertices inside the image and drop repeated ones; None if no real area is left."""
    out: list[list[float]] = []
    for x, y in points:
        p = [round(max(0.0, min(float(width), x)), 2), round(max(0.0, min(float(height), y)), 2)]
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3 or polygon_area(out) < MIN_BOX_SIZE:
        return None
    return out


def clamp_box(b: Box, width: int, height: int) -> Box | None:
    """Clip a box to the image; drop it if it collapses."""
    if b.polygon is not None:
        polygon = clamp_polygon(b.polygon, width, height)
        if polygon is None:
            return None
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            return None
        return Box(b.class_id, x, y, round(w, 2), round(h, 2), b.source, b.score, polygon)
    x1 = max(0.0, min(float(width), b.x))
    y1 = max(0.0, min(float(height), b.y))
    x2 = max(0.0, min(float(width), b.x + b.w))
    y2 = max(0.0, min(float(height), b.y + b.h))
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None
    return Box(b.class_id, round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2), b.source, b.score)


def _key(b: Box | Annotation) -> tuple:
    outline = tuple((round(x, 1), round(y, 1)) for x, y in b.polygon) if b.polygon else ()
    return (b.class_id, round(b.x, 1), round(b.y, 1), round(b.w, 1), round(b.h, 1), b.source, outline)


def snapshot(boxes: list[Box], class_names: dict[int, str]) -> list[dict]:
    return [
        {
            "class_id": b.class_id,
            "class_name": class_names.get(b.class_id, "?"),
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "source": b.source,
            "score": b.score,
            "polygon": b.polygon,
        }
        for b in boxes
    ]


def apply_boxes(
    db: Session,
    image: Image,
    boxes: list[Box],
    action: Action,
    *,
    base_version: int | None = None,
    note: str = "",
) -> Image:
    if base_version is not None and base_version != image.version:
        raise VersionConflict(
            f"這張圖已經被更新過（目前版本 {image.version}，你編輯的是版本 {base_version}），請重新載入"
        )

    class_names = dict(
        db.execute(
            select(LabelClass.id, LabelClass.name).where(LabelClass.project_id == image.project_id)
        ).all()
    )
    cleaned: list[Box] = []
    for b in boxes:
        if b.class_id not in class_names:
            raise AnnotationError(f"類別 id {b.class_id} 不屬於這個專案")
        c = clamp_box(b, image.width, image.height)
        if c is not None:
            cleaned.append(c)

    changed = sorted(map(_key, cleaned)) != sorted(map(_key, image.annotations))

    # ---- status transition
    if action == "review":
        new_status = STATUS_REVIEWED
    elif action == "auto_label":
        new_status = STATUS_AUTO
    elif action == "revert":
        new_status = STATUS_IN_PROGRESS
    elif action == "clear":
        new_status = STATUS_UNLABELED
    else:  # save
        if not changed:
            new_status = image.status
        elif image.status == STATUS_UNLABELED and not cleaned:
            new_status = STATUS_UNLABELED
        else:
            new_status = STATUS_IN_PROGRESS

    if not changed and new_status == image.status and action == "save":
        return image  # nothing to record

    image.annotations.clear()
    db.flush()
    for b in cleaned:
        image.annotations.append(
            Annotation(
                class_id=b.class_id, x=b.x, y=b.y, w=b.w, h=b.h, polygon=b.polygon, source=b.source, score=b.score
            )
        )

    now = utcnow()
    image.version += 1
    image.status = new_status
    image.box_count = len(cleaned)
    auto_scores = [b.score for b in cleaned if b.source == "auto" and b.score is not None]
    image.min_score = min(auto_scores) if auto_scores else None
    image.updated_at = now
    if new_status == STATUS_REVIEWED:
        image.reviewed_at = now

    db.add(
        Revision(
            image_id=image.id,
            version=image.version,
            action=action,
            status=new_status,
            boxes=snapshot(cleaned, class_names),
            note=note,
            created_at=now,
        )
    )
    return image


def boxes_from_revision(rev: Revision, valid_class_ids: set[int]) -> list[Box]:
    out = []
    for d in rev.boxes:
        if d["class_id"] in valid_class_ids:
            out.append(
                Box(
                    d["class_id"], d["x"], d["y"], d["w"], d["h"],
                    d.get("source", "manual"), d.get("score"), d.get("polygon"),
                )
            )
    return out


def reset_status_for_review(image: Image) -> None:
    """Used when a class is deleted: a reviewed image whose boxes changed must be re-checked."""
    if image.status == STATUS_REVIEWED:
        image.status = STATUS_IN_PROGRESS
