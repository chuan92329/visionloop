from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import LabelClass, Revision
from ..schemas import AnnotationState, BoxOut, ImageOut, RevisionOut, SaveAnnotations
from ..services.annotations import AnnotationError, Box, VersionConflict, apply_boxes, boxes_from_revision
from .common import get_image

router = APIRouter(prefix="/api", tags=["annotations"])


def _state(img) -> AnnotationState:
    return AnnotationState(
        image=ImageOut.model_validate(img),
        boxes=[BoxOut.model_validate(a) for a in img.annotations],
    )


@router.get("/images/{image_id}/annotations", response_model=AnnotationState)
def read_annotations(image_id: int, db: Session = Depends(get_db)):
    return _state(get_image(db, image_id))


@router.put("/images/{image_id}/annotations", response_model=AnnotationState)
def save_annotations(image_id: int, body: SaveAnnotations, db: Session = Depends(get_db)):
    img = get_image(db, image_id)
    boxes = [
        Box(b.class_id, b.x, b.y, b.w, b.h, b.source, b.score, [list(p) for p in b.polygon] if b.polygon else None)
        for b in body.boxes
    ]
    try:
        apply_boxes(
            db,
            img,
            boxes,
            "review" if body.mark_reviewed else "save",
            base_version=body.base_version,
            note=body.note,
        )
    except VersionConflict as exc:
        db.rollback()
        raise HTTPException(409, str(exc))
    except AnnotationError as exc:
        db.rollback()
        raise HTTPException(422, str(exc))
    db.commit()
    db.refresh(img)
    return _state(img)


@router.get("/images/{image_id}/revisions", response_model=list[RevisionOut])
def list_revisions(image_id: int, db: Session = Depends(get_db)):
    get_image(db, image_id)
    return db.scalars(
        select(Revision).where(Revision.image_id == image_id).order_by(Revision.version.desc())
    ).all()


@router.post("/images/{image_id}/revisions/{version}/restore", response_model=AnnotationState)
def restore_revision(image_id: int, version: int, db: Session = Depends(get_db)):
    img = get_image(db, image_id)
    rev = db.scalar(select(Revision).where(Revision.image_id == image_id, Revision.version == version))
    if rev is None:
        raise HTTPException(404, "找不到這個版本")
    valid = set(db.scalars(select(LabelClass.id).where(LabelClass.project_id == img.project_id)))
    boxes = boxes_from_revision(rev, valid)
    apply_boxes(db, img, boxes, "revert", note=f"還原到版本 {version}")
    db.commit()
    db.refresh(img)
    return _state(img)
