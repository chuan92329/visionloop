from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import IMAGE_STATUSES, Image
from ..schemas import ImageOut, ImagePage, ImageUpdate, UploadResult
from ..services import storage
from .common import get_image, get_project

router = APIRouter(prefix="/api", tags=["images"])

SortKey = Literal["id", "filename", "min_score", "updated_at", "box_count"]


def _filtered(
    project_id: int,
    status: list[str] | None,
    split: str | None,
    q: str | None,
    max_score: float | None,
    group: str | None,
) -> Select:
    stmt = select(Image).where(Image.project_id == project_id)
    if status:
        bad = set(status) - set(IMAGE_STATUSES)
        if bad:
            raise HTTPException(422, f"未知的狀態：{', '.join(bad)}")
        stmt = stmt.where(Image.status.in_(status))
    if split == "none":
        stmt = stmt.where(Image.split.is_(None))
    elif split:
        stmt = stmt.where(Image.split == split)
    if q:
        stmt = stmt.where(Image.filename.ilike(f"%{q}%"))
    if max_score is not None:
        stmt = stmt.where(Image.min_score.is_not(None), Image.min_score <= max_score)
    if group:
        stmt = stmt.where(Image.group_key == group)
    return stmt


def _ordered(stmt: Select, sort: SortKey, order: str) -> Select:
    col = getattr(Image, sort)
    if sort == "min_score":
        # images without auto boxes go last
        stmt = stmt.order_by(Image.min_score.is_(None))
    stmt = stmt.order_by(col.desc() if order == "desc" else col.asc(), Image.id.asc())
    return stmt


@router.post("/projects/{project_id}/images", response_model=UploadResult, status_code=201)
async def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
    group_key: str | None = Form(None),
    # (missing multipart booleans parse as False, so the opt-in flag is the safe default)
    allow_duplicates: bool = Form(False),
    db: Session = Depends(get_db),
):
    project = get_project(db, project_id)
    known = set(db.scalars(select(Image.sha1).where(Image.project_id == project.id)))
    created, skipped = [], []
    for f in files:
        data = await f.read()
        name = f.filename or "image"
        try:
            stored = storage.store_image(project.id, name, data)
        except storage.InvalidImage as exc:
            skipped.append({"filename": name, "reason": str(exc)})
            continue
        if not allow_duplicates and stored.sha1 in known:
            storage.delete_image_files(project.id, stored.stored_name)
            skipped.append({"filename": name, "reason": "重複的圖片（內容相同）"})
            continue
        known.add(stored.sha1)
        img = Image(
            project_id=project.id,
            filename=name,
            stored_name=stored.stored_name,
            sha1=stored.sha1,
            width=stored.width,
            height=stored.height,
            group_key=(group_key or None),
        )
        db.add(img)
        db.flush()
        created.append(img)
    db.commit()
    return UploadResult(created=[ImageOut.model_validate(i) for i in created], skipped=skipped)


@router.get("/projects/{project_id}/images", response_model=ImagePage)
def list_images(
    project_id: int,
    status: list[str] | None = Query(None),
    split: str | None = None,
    q: str | None = None,
    max_score: float | None = None,
    group: str | None = None,
    sort: SortKey = "id",
    order: Literal["asc", "desc"] = "asc",
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=500),
    db: Session = Depends(get_db),
):
    get_project(db, project_id)
    stmt = _filtered(project_id, status, split, q, max_score, group)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    items = db.scalars(_ordered(stmt, sort, order).offset(offset).limit(limit)).all()
    return ImagePage(items=items, total=total, offset=offset, limit=limit)


@router.get("/projects/{project_id}/image-ids", response_model=list[int])
def list_image_ids(
    project_id: int,
    status: list[str] | None = Query(None),
    split: str | None = None,
    q: str | None = None,
    max_score: float | None = None,
    group: str | None = None,
    sort: SortKey = "id",
    order: Literal["asc", "desc"] = "asc",
    db: Session = Depends(get_db),
):
    """Ordered ids for the current filter – the annotator uses it as its work queue."""
    get_project(db, project_id)
    stmt = _ordered(_filtered(project_id, status, split, q, max_score, group), sort, order)
    return list(db.scalars(stmt.with_only_columns(Image.id)))


@router.get("/images/{image_id}", response_model=ImageOut)
def read_image(image_id: int, db: Session = Depends(get_db)):
    return get_image(db, image_id)


@router.patch("/images/{image_id}", response_model=ImageOut)
def update_image(image_id: int, body: ImageUpdate, db: Session = Depends(get_db)):
    img = get_image(db, image_id)
    if "group_key" in body.model_fields_set:
        img.group_key = body.group_key or None
    if body.split is not None or body.clear_split:
        if img.project.test_locked and (img.split == "test" or body.split == "test"):
            raise HTTPException(409, "test 集合已鎖定，不能手動移入或移出")
        img.split = None if body.clear_split else body.split
    db.commit()
    return img


class BulkDelete(BaseModel):
    ids: list[int]


class BulkGroup(BaseModel):
    ids: list[int]
    group_key: str | None = None


@router.post("/projects/{project_id}/images/group")
def set_group(project_id: int, body: BulkGroup, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    imgs = db.scalars(select(Image).where(Image.project_id == project.id, Image.id.in_(body.ids))).all()
    for i in imgs:
        i.group_key = (body.group_key or "").strip() or None
    db.commit()
    return {"updated": len(imgs)}


@router.post("/projects/{project_id}/images/delete")
def delete_images(project_id: int, body: BulkDelete, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    imgs = db.scalars(select(Image).where(Image.project_id == project.id, Image.id.in_(body.ids))).all()
    locked = [i.id for i in imgs if project.test_locked and i.split == "test"]
    if locked:
        raise HTTPException(409, f"有 {len(locked)} 張圖片在已鎖定的 test 集合中，請先解除鎖定")
    names = [i.stored_name for i in imgs]
    for i in imgs:
        db.delete(i)
    db.commit()
    for n in names:
        storage.delete_image_files(project.id, n)
    return {"deleted": len(names)}


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db)):
    img = get_image(db, image_id)
    path = storage.image_path(img.project_id, img.stored_name)
    if not path.exists():
        raise HTTPException(404, "圖片檔案不存在")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


@router.get("/images/{image_id}/thumb")
def image_thumb(image_id: int, db: Session = Depends(get_db)):
    img = get_image(db, image_id)
    path = storage.thumb_path(img.project_id, img.stored_name)
    if not path.exists():
        from PIL import Image as PILImage

        with PILImage.open(storage.image_path(img.project_id, img.stored_name)) as src:
            storage.make_thumbnail(src, path)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
