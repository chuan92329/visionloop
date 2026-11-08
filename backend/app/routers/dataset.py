from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import utcnow
from ..schemas import AutoGroupRequest, SplitConfig, SplitRequest, SplitStats
from ..services import dataset
from .common import get_project

router = APIRouter(prefix="/api/projects/{project_id}", tags=["dataset"])


@router.get("/split", response_model=SplitStats)
def get_split(project_id: int, db: Session = Depends(get_db)):
    return SplitStats(**dataset.split_stats(db, get_project(db, project_id)))


@router.post("/split", response_model=SplitStats)
def do_split(project_id: int, body: SplitRequest, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    cfg = body.config or SplitConfig(**project.split_config)
    warnings = dataset.run_split(
        db,
        project,
        ratios={"train": cfg.train, "valid": cfg.valid, "test": cfg.test},
        seed=cfg.seed,
        mode=body.mode,
        stratify=body.stratify,
        lock_test=body.lock_test,
    )
    db.commit()
    return SplitStats(**dataset.split_stats(db, project), warnings=warnings)


class LockBody(BaseModel):
    locked: bool


@router.post("/split/lock", response_model=SplitStats)
def lock_test(project_id: int, body: LockBody, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    stats = dataset.split_stats(db, project)
    if body.locked and stats["images"]["test"] == 0:
        raise HTTPException(409, "test 集合是空的，無法鎖定")
    project.test_locked = body.locked
    project.test_locked_at = utcnow() if body.locked else None
    db.commit()
    return SplitStats(**dataset.split_stats(db, project))


@router.post("/auto-group")
def auto_group(project_id: int, body: AutoGroupRequest, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    try:
        n = dataset.auto_group(db, project, body.pattern, body.overwrite)
    except (re.error, ValueError) as exc:
        raise HTTPException(422, f"規則運算式錯誤：{exc}")
    db.commit()
    return {"updated": n}


@router.get("/export")
def export(
    project_id: int,
    background: BackgroundTasks,
    format: Literal["yolo", "yolo-seg", "coco"] = "yolo",
    db: Session = Depends(get_db),
):
    project = get_project(db, project_id)
    try:
        if format == "coco":
            entries = dataset.iter_coco(db, project)
        else:
            entries = dataset.iter_yolo(db, project, segmentation=format == "yolo-seg")
        tmp = Path(tempfile.mkdtemp(prefix="visionloop-export-")) / "export.zip"
        dataset.write_zip(entries, tmp)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    background.add_task(lambda: (tmp.unlink(missing_ok=True), tmp.parent.rmdir()))
    safe = re.sub(r"[^\w\-]+", "_", project.name, flags=re.UNICODE).strip("_") or "dataset"
    filename = f"{safe}_{format}_{utcnow():%Y%m%d_%H%M}.zip"
    return FileResponse(tmp, media_type="application/zip", filename=filename)
