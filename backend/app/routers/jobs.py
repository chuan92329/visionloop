from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import IMAGE_STATUSES, STATUS_AUTO, STATUS_REVIEWED, Annotation, Image, Job, ModelWeights
from ..plugins import registry
from ..plugins.base import NO_WINDOW
from ..schemas import AutoLabelRequest, JobDetail, JobOut, ModelWeightsOut, TrainRequest
from ..services.annotations import apply_boxes
from ..services.dataset import split_stats
from ..services.jobs import AUTOLABEL_ALLOWED, delete_job_record, next_job_id, runner
from .common import get_project, is_local_request

router = APIRouter(prefix="/api", tags=["models"])

FINISHED_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


class ClearAuto(BaseModel):
    ids: list[int] | None = None


class ClearJobs(BaseModel):
    statuses: list[str]


# ----------------------------------------------------------- plugins/system
@router.get("/plugins")
def list_plugins():
    return {
        "plugins": [p.describe() for p in registry.all_plugins()],
        "load_errors": registry.load_errors(),
        "plugin_dir": str(settings.plugin_dir),
    }


@router.post("/plugins/reload")
def reload_plugins():
    """Pick up model families added to the plugin folder without restarting the server."""
    registry.load_plugins(settings.plugin_dir)
    return list_plugins()


@router.get("/system")
def system_info():
    gpus = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=NO_WINDOW,
            ).stdout
            for line in out.strip().splitlines():
                idx, name, mem = [x.strip() for x in line.split(",")]
                gpus.append({"index": int(idx), "name": name, "memory_gb": round(int(mem) / 1024, 1)})
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    return {"gpus": gpus, "torch_installed": importlib.util.find_spec("torch") is not None}


def _plugin_or_422(plugin_id: str, *, need_available: bool = True):
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(422, f"找不到模型外掛「{plugin_id}」")
    if need_available:
        st = plugin.status()
        if not st.available:
            raise HTTPException(422, f"{plugin.name} 目前無法使用：{st.message}")
    return plugin


# --------------------------------------------------------------------- jobs
@router.post("/projects/{project_id}/autolabel", response_model=JobOut, status_code=201)
def start_autolabel(project_id: int, body: AutoLabelRequest, db: Session = Depends(get_db)):
    get_project(db, project_id)
    plugin = _plugin_or_422(body.plugin)
    # outlining keeps every object, so it may also run on images people worked on
    bad = set(body.statuses) - (set(IMAGE_STATUSES) if plugin.prompt_mode == "boxes" else AUTOLABEL_ALLOWED)
    if bad:
        raise HTTPException(422, "自動標註只能處理「未標註」或「自動標註」的圖片，不會覆蓋人工結果")
    if body.weights_id is not None:
        w = db.get(ModelWeights, body.weights_id)
        if w is None or w.project_id != project_id:
            raise HTTPException(422, "找不到這個權重")
        if w.plugin != plugin.id:
            raise HTTPException(422, f"這個權重屬於 {w.plugin}，不能用 {plugin.id} 執行")
    elif plugin.needs_weights:
        raise HTTPException(422, "這個模型需要先選擇權重")
    if plugin.prompt_mode == "exemplars" and not exemplar_counts(db, project_id):
        raise HTTPException(422, "這個模型需要範例：請先在幾張圖片上標出物件並按「確認」")
    job = Job(id=next_job_id(db), project_id=project_id, kind="autolabel", plugin=plugin.id,
              params=body.model_dump(), message="排隊中")
    db.add(job)
    db.commit()
    runner.submit(job.id)
    return job


@router.post("/projects/{project_id}/train", response_model=JobOut, status_code=201)
def start_train(project_id: int, body: TrainRequest, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    plugin = _plugin_or_422(body.plugin)
    if not plugin.trainable:
        raise HTTPException(422, f"{plugin.name} 不需要訓練，請直接用它做自動標註")
    stats = split_stats(db, project)
    if stats["images"]["train"] == 0 or stats["images"]["valid"] == 0:
        raise HTTPException(409, "train 和 valid 都至少要有一張已確認的圖片，請先完成審核並執行資料切分")
    if plugin.task == "segment" and not stats["polygons"]:
        raise HTTPException(409, "分割模型需要多邊形標註：請先畫多邊形，或用 SAM 2 把框轉成多邊形並確認")
    variant = next((v for v in plugin.variants if v.id == body.variant), None)
    if variant is None:
        raise HTTPException(422, f"{plugin.name} 沒有「{body.variant}」這個尺寸")
    image_size = body.image_size or variant.image_size or 640
    if image_size % variant.image_step:
        raise HTTPException(422, f"{variant.name} 的影像尺寸必須是 {variant.image_step} 的倍數，{image_size} 不符合")
    if body.init_weights_id is not None:
        w = db.get(ModelWeights, body.init_weights_id)
        if w is None or w.project_id != project_id or w.plugin != plugin.id:
            raise HTTPException(422, "初始權重不存在或不屬於這個模型")
    job = Job(id=next_job_id(db), project_id=project_id, kind="train", plugin=plugin.id,
              params={**body.model_dump(), "image_size": image_size}, message="排隊中")
    db.add(job)
    db.commit()
    runner.submit(job.id)
    return job


def exemplar_counts(db: Session, project_id: int) -> dict[int, int]:
    rows = db.execute(
        select(Annotation.class_id, func.count())
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Image.status == STATUS_REVIEWED)
        .group_by(Annotation.class_id)
    )
    return {cid: n for cid, n in rows}


@router.get("/projects/{project_id}/exemplars")
def exemplar_stats(project_id: int, db: Session = Depends(get_db)):
    """Confirmed boxes per class – the examples few-shot models can use."""
    get_project(db, project_id)
    return {str(k): v for k, v in exemplar_counts(db, project_id).items()}


@router.post("/projects/{project_id}/clear-auto")
def clear_auto_labels(project_id: int, body: ClearAuto, db: Session = Depends(get_db)):
    """Remove model-generated boxes from images that are still in the "auto" state and put them
    back to "unlabeled". Images a person has touched are never affected."""
    get_project(db, project_id)
    stmt = select(Image).where(Image.project_id == project_id, Image.status == STATUS_AUTO)
    if body.ids:
        stmt = stmt.where(Image.id.in_(body.ids))
    n = 0
    for img in db.scalars(stmt):
        apply_boxes(db, img, [], "clear", note="清除自動標註")
        n += 1
    db.commit()
    return {"cleared": n}


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: int, db: Session = Depends(get_db)):
    get_project(db, project_id)
    return db.scalars(
        select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(100)
    ).all()


@router.get("/jobs/{job_id}", response_model=JobDetail)
def read_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "找不到任務")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "找不到任務")
    if job.status in ("queued", "running"):
        runner.cancel(job_id)
        if job.status == "queued":
            job.message = "取消中"
            db.commit()
    return job


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Annotations the job made and weights it trained stay."""
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "找不到任務")
    if job.status not in FINISHED_JOB_STATUSES:
        raise HTTPException(409, "任務還在執行，請先取消")
    delete_job_record(db, job)
    db.commit()


@router.post("/projects/{project_id}/clear-jobs")
def clear_jobs(project_id: int, body: ClearJobs, db: Session = Depends(get_db)):
    get_project(db, project_id)
    if set(body.statuses) - FINISHED_JOB_STATUSES:
        raise HTTPException(422, "只能清除已結束的任務")
    jobs = db.scalars(select(Job).where(Job.project_id == project_id, Job.status.in_(body.statuses))).all()
    for job in jobs:
        delete_job_record(db, job)
    db.commit()
    return {"deleted": len(jobs)}


# ------------------------------------------------------------------ weights
@router.get("/projects/{project_id}/weights", response_model=list[ModelWeightsOut])
def list_weights(project_id: int, db: Session = Depends(get_db)):
    get_project(db, project_id)
    return db.scalars(
        select(ModelWeights).where(ModelWeights.project_id == project_id).order_by(ModelWeights.id.desc())
    ).all()


@router.post("/projects/{project_id}/weights/import", response_model=ModelWeightsOut, status_code=201)
async def import_weights(
    project_id: int,
    plugin: str = Form(...),
    variant: str = Form(...),
    name: str = Form(""),
    class_names: str = Form(..., description="類別名稱，以換行或逗號分隔，順序需與權重一致"),
    image_size: int = Form(640),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an existing checkpoint (e.g. COCO-pretrained) for auto-labelling.
    Its classes are matched to project classes by name."""
    get_project(db, project_id)
    p = _plugin_or_422(plugin, need_available=False)
    if variant not in {v.id for v in p.variants}:
        raise HTTPException(422, f"{p.name} 沒有「{variant}」這個尺寸")
    names = [n.strip() for n in re.split(r"[\n,]", class_names) if n.strip()]
    if not names:
        raise HTTPException(422, "請提供權重的類別名稱")
    folder = settings.data_dir / "weights" / str(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    # keep the original file name: plugins use it, e.g. YOLOv9 picks detect.py for "*-converted.pt"
    original = Path(file.filename or "weights.pt").name
    safe = re.sub(r"[^\w\-.]+", "_", original).strip("._") or "weights.pt"
    if not Path(safe).suffix:
        safe += ".pt"
    dest = folder / f"{uuid.uuid4().hex[:8]}_{safe}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    w = ModelWeights(
        project_id=project_id,
        plugin=p.id,
        variant=variant,
        name=name.strip() or (file.filename or "匯入的權重"),
        path=str(dest.resolve()),
        class_names=names,
        metrics={"imported": True, "train_image_size": image_size},
    )
    db.add(w)
    db.commit()
    return w


def run_folder(w: ModelWeights) -> Path | None:
    """The folder a training run wrote its charts and logs to: the one above "weights/"."""
    if w.job_id is None:
        return None
    folder = Path(w.path).resolve().parent
    if folder.name == "weights":
        folder = folder.parent
    if settings.data_dir.resolve() not in folder.parents or not folder.is_dir():
        return None
    return folder


def _reveal(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(folder)])
    else:
        raise OSError("no file manager")


@router.post("/weights/{weights_id}/open-folder")
def open_run_folder(weights_id: int, request: Request, db: Session = Depends(get_db)):
    """Show the training charts (results.png, confusion matrix, PR curves...) in the file manager.
    Only for the computer the platform runs on: a browser elsewhere gets the path instead."""
    w = db.get(ModelWeights, weights_id)
    if w is None:
        raise HTTPException(404, "找不到權重")
    folder = run_folder(w)
    if folder is None:
        raise HTTPException(404, "找不到這個權重的訓練資料夾")
    opened = False
    if is_local_request(request):
        try:
            _reveal(folder)
            opened = True
        except OSError:
            pass
    return {"path": str(folder), "opened": opened}


@router.delete("/weights/{weights_id}", status_code=204)
def delete_weights(weights_id: int, db: Session = Depends(get_db)):
    w = db.get(ModelWeights, weights_id)
    if w is None:
        raise HTTPException(404, "找不到權重")
    path = Path(w.path)
    db.delete(w)
    db.commit()
    # only remove files VisionLoop owns
    if settings.data_dir.resolve() in path.resolve().parents:
        path.unlink(missing_ok=True)
