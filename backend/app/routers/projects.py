from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Annotation, Image, LabelClass, Project
from ..schemas import LabelClassIn, LabelClassOut, LabelClassUpdate, ProjectIn, ProjectOut, ProjectUpdate
from ..services import storage
from ..services.annotations import Box, apply_boxes
from .common import get_class, get_project, next_color, project_out

router = APIRouter(prefix="/api", tags=["projects"])


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    projects = db.scalars(select(Project).order_by(Project.created_at.desc())).all()
    return [project_out(db, p) for p in projects]


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectIn, db: Session = Depends(get_db)):
    project = Project(name=body.name.strip(), description=body.description)
    db.add(project)
    db.flush()
    seen = set()
    for name in body.classes:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            project.classes.append(
                LabelClass(name=name, color=next_color(project), order_index=len(project.classes))
            )
            db.flush()
    db.commit()
    db.refresh(project)
    return project_out(db, project)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def read_project(project_id: int, db: Session = Depends(get_db)):
    return project_out(db, get_project(db, project_id))


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, body: ProjectUpdate, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    if body.name is not None:
        project.name = body.name.strip()
    if body.description is not None:
        project.description = body.description
    db.commit()
    return project_out(db, project)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    db.delete(project)
    db.commit()
    storage.delete_project_files(project_id)


# ------------------------------------------------------------------ classes
@router.post("/projects/{project_id}/classes", response_model=LabelClassOut, status_code=201)
def create_class(project_id: int, body: LabelClassIn, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    cls = LabelClass(
        project_id=project.id,
        name=body.name.strip(),
        color=body.color or next_color(project),
        order_index=max((c.order_index for c in project.classes), default=-1) + 1,
    )
    db.add(cls)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"類別「{body.name}」已存在")
    return cls


@router.patch("/classes/{class_id}", response_model=LabelClassOut)
def update_class(class_id: int, body: LabelClassUpdate, db: Session = Depends(get_db)):
    cls = get_class(db, class_id)
    if body.name is not None:
        cls.name = body.name.strip()
    if body.color is not None:
        cls.color = body.color
    if body.order_index is not None:
        cls.order_index = body.order_index
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"類別「{body.name}」已存在")
    return cls


@router.get("/classes/{class_id}/usage")
def class_usage(class_id: int, db: Session = Depends(get_db)):
    get_class(db, class_id)
    image_ids = set(db.scalars(select(Annotation.image_id).where(Annotation.class_id == class_id)))
    boxes = db.query(Annotation).filter(Annotation.class_id == class_id).count()
    return {"images": len(image_ids), "boxes": boxes}


@router.delete("/classes/{class_id}", status_code=204)
def delete_class(class_id: int, db: Session = Depends(get_db)):
    """Delete a class. Affected images get a new revision without those boxes
    and must be confirmed again (status -> in_progress)."""
    cls = get_class(db, class_id)
    image_ids = set(db.scalars(select(Annotation.image_id).where(Annotation.class_id == class_id)))
    for img in db.scalars(select(Image).where(Image.id.in_(image_ids))):
        keep = [
            Box(a.class_id, a.x, a.y, a.w, a.h, a.source, a.score, a.polygon)
            for a in img.annotations
            if a.class_id != class_id
        ]
        apply_boxes(db, img, keep, "save", note=f"刪除類別「{cls.name}」")
    db.flush()
    db.delete(cls)
    db.commit()
