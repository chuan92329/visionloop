"""Background jobs (auto-labelling and training).

Jobs run one at a time on a single worker thread – a desktop machine
usually has one GPU and two trainings at once would only fight over it.
State lives in the ``jobs`` table so the UI can poll it and the history
survives restarts.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select

from ..config import settings
from ..db import SessionLocal
from ..models import (
    STATUS_AUTO,
    STATUS_REVIEWED,
    STATUS_UNLABELED,
    Annotation,
    Image,
    Job,
    ModelWeights,
    Project,
    utcnow,
)
from ..plugins import registry
from ..plugins.base import Cancelled, Exemplar, JobHooks, PredictContext, ReviewedImage, TrainContext
from . import dataset, storage
from .annotations import Box, apply_boxes

log = logging.getLogger(__name__)

MAX_LOG_CHARS = 200_000
# auto-label never overwrites human work
AUTOLABEL_ALLOWED = {STATUS_UNLABELED, STATUS_AUTO}


class JobRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[int] = queue.Queue()
        self._cancel: set[int] = set()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ control
    def start(self) -> None:
        self._recover()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="visionloop-jobs", daemon=True)
            self._thread.start()

    def submit(self, job_id: int) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: int) -> None:
        with self._lock:
            self._cancel.add(job_id)

    def _recover(self) -> None:
        """Jobs left 'running' by a previous process can never finish – mark them failed,
        and re-queue jobs that never started."""
        with SessionLocal() as db:
            for job in db.scalars(select(Job).where(Job.status.in_(("running", "queued")))):
                if job.status == "running":
                    job.status = "failed"
                    job.message = "伺服器重新啟動，任務中斷"
                    job.finished_at = utcnow()
                else:
                    self._queue.put(job.id)
            db.commit()

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            except Exception:  # noqa: BLE001
                log.exception("job %s crashed", job_id)

    # ---------------------------------------------------------- execution
    def _run(self, job_id: int) -> None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None or job.status != "queued":
                return
            if job_id in self._cancel:
                job.status, job.message, job.finished_at = "cancelled", "已取消", utcnow()
                db.commit()
                return
            job.status, job.started_at, job.message = "running", utcnow(), "開始執行"
            db.commit()

        hooks = _Hooks(job_id, lambda: job_id in self._cancel)
        try:
            with SessionLocal() as db:
                job = db.get(Job, job_id)
                kind, plugin_id, params, project_id = job.kind, job.plugin, dict(job.params), job.project_id
            plugin = registry.get_plugin(plugin_id)
            if plugin is None:
                raise RuntimeError(f"找不到模型外掛「{plugin_id}」")
            work_dir = new_work_dir(job_id)
            if kind == "autolabel":
                result = _autolabel(project_id, plugin, params, work_dir, hooks)
            elif kind == "train":
                result = _train(project_id, plugin, params, work_dir, hooks, job_id)
            else:
                raise RuntimeError(f"未知的任務類型 {kind}")
            hooks.flush()
            with SessionLocal() as db:
                job = db.get(Job, job_id)
                job.status, job.progress, job.result = "succeeded", 1.0, result
                job.message = result.get("summary", "完成")
                job.finished_at = utcnow()
                db.commit()
        except Cancelled:
            hooks.log("任務已取消")
            hooks.flush()
            self._finish(job_id, "cancelled", "已取消")
        except Exception as exc:  # noqa: BLE001
            hooks.log(traceback.format_exc())
            hooks.flush()
            self._finish(job_id, "failed", str(exc) or type(exc).__name__)
        finally:
            with self._lock:
                self._cancel.discard(job_id)

    @staticmethod
    def _finish(job_id: int, status: str, message: str) -> None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            job.status, job.message, job.finished_at = status, message[:2000], utcnow()
            db.commit()


class _Hooks(JobHooks):
    """Buffers log lines / progress and writes them to the DB at most once per second."""

    def __init__(self, job_id: int, should_stop: Callable[[], bool]) -> None:
        self.job_id = job_id
        self._buf: list[str] = []
        self._progress: tuple[float, str] | None = None
        self._last_flush = 0.0
        super().__init__(log=self._log, progress=self._set_progress, should_stop=should_stop)

    def _log(self, line: str) -> None:
        self._buf.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        self._maybe_flush()

    def _set_progress(self, frac: float, msg: str) -> None:
        self._progress = (max(0.0, min(1.0, frac)), msg)
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush > 1.0:
            self.flush()

    def flush(self) -> None:
        self._last_flush = time.monotonic()
        if not self._buf and self._progress is None:
            return
        with SessionLocal() as db:
            job = db.get(Job, self.job_id)
            if job is None:
                return
            if self._buf:
                text = (job.log or "") + "\n".join(self._buf) + "\n"
                if len(text) > MAX_LOG_CHARS:
                    text = "...(較早的紀錄已截斷)...\n" + text[-MAX_LOG_CHARS:]
                job.log = text
                self._buf.clear()
            if self._progress is not None:
                job.progress, job.message = self._progress
                self._progress = None
            db.commit()


# ------------------------------------------------------------- auto-label
def _outline_boxes(project_id: int, plugin, params: dict, work_dir: Path, hooks: _Hooks) -> dict:
    """prompt_mode == "boxes": add outlines to existing boxes, keeping every object.

    Unlike auto-labelling this may touch confirmed images (the user picks them): the result is
    a model suggestion, so they go back to "auto", and the revision history keeps the old boxes.
    """
    with SessionLocal() as db:
        stmt = select(Image).where(Image.project_id == project_id)
        if params.get("image_ids"):
            stmt = stmt.where(Image.id.in_(params["image_ids"]))
        else:
            stmt = stmt.where(Image.status.in_(params.get("statuses", [])))
        targets: dict[str, tuple[int, int, list[int]]] = {}
        prompts: dict[str, list[tuple[float, float, float, float]]] = {}
        for img in db.scalars(stmt.order_by(Image.id)):
            pending = [a for a in img.annotations if params.get("overwrite_polygons") or not a.polygon]
            if pending:
                path = str(storage.image_path(project_id, img.stored_name))
                targets[path] = (img.id, img.version, [a.id for a in pending])
                prompts[path] = [(a.x, a.y, a.w, a.h) for a in pending]

    if not targets:
        return {"summary": "沒有需要轉換的框（選擇的圖片沒有框，或都已經有多邊形）", "labelled": 0}

    hooks.log(f"共 {len(targets)} 張圖片，{sum(map(len, prompts.values()))} 個框")
    ctx = PredictContext(
        weights_path=None,
        class_names=[],
        variant=params.get("variant"),
        confidence=0.0,
        device=params.get("device", "auto"),
        work_dir=work_dir,
        hooks=hooks,
        box_prompts=prompts,
        options={"polygon_detail": params.get("polygon_detail", "normal")},
    )
    done = outlined = failed = skipped = 0
    paths = [Path(p) for p in targets]
    for path, preds in plugin.predict(ctx, paths):
        hooks.check_cancel()
        img_id, version, ann_ids = targets[str(path)]
        found = {a: p for a, p in zip(ann_ids, preds) if p.polygon}
        with SessionLocal() as db:
            img = db.get(Image, img_id)
            if img is None or img.version != version:  # edited while the job was running
                skipped += 1
            elif found:
                boxes = []
                for a in img.annotations:
                    p = found.get(a.id)
                    if p:
                        boxes.append(Box(a.class_id, p.x, p.y, p.w, p.h, "auto", round(float(p.score), 4), p.polygon))
                    else:
                        boxes.append(Box(a.class_id, a.x, a.y, a.w, a.h, a.source, a.score, a.polygon))
                apply_boxes(db, img, boxes, "auto_label", note=plugin.name)
                db.commit()
                outlined += len(found)
                failed += len(ann_ids) - len(found)
            else:
                failed += len(ann_ids)
        done += 1
        hooks.progress(done / len(paths), f"轉換 {done}/{len(paths)}")

    shutil.rmtree(work_dir, ignore_errors=True)
    summary = f"完成 {done - skipped} 張，{outlined} 個框轉成多邊形，請到標註畫面確認"
    if failed:
        summary += f"；{failed} 個框無法轉換，保留原狀"
    if skipped:
        summary += f"；{skipped} 張在執行期間被人工修改，已略過"
    return {"summary": summary, "labelled": done - skipped, "boxes": outlined, "skipped": skipped}


def _autolabel(project_id: int, plugin, params: dict, work_dir: Path, hooks: _Hooks) -> dict:
    if plugin.prompt_mode == "boxes":
        return _outline_boxes(project_id, plugin, params, work_dir, hooks)
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        classes = list(project.classes)
        by_name = {c.name.strip().lower(): c.id for c in classes}

        weights = None
        if params.get("weights_id"):
            weights = db.get(ModelWeights, params["weights_id"])
            if weights is None:
                raise RuntimeError("找不到選擇的權重")
        elif plugin.needs_weights:
            raise RuntimeError("這個模型需要先選擇權重（先訓練一次，或匯入現成的權重）")

        weight_classes = list(weights.class_names) if weights else [c.name for c in classes]
        # map the model's class index -> project class id by (case-insensitive) name
        index_map = {
            i: by_name[n.strip().lower()] for i, n in enumerate(weight_classes) if n.strip().lower() in by_name
        }
        unmapped = [n for n in weight_classes if n.strip().lower() not in by_name]
        if not index_map:
            raise RuntimeError("權重的類別名稱與專案類別完全對不上，請確認類別名稱一致")
        if unmapped:
            hooks.log(f"以下模型類別在專案中不存在，會被忽略：{', '.join(unmapped[:20])}")

        reviewed = _collect_reviewed(db, project, int(params.get("max_reviewed_images", 1000))) \
            if plugin.uses_reviewed_images else []
        exemplars = _collect_exemplars(db, project, int(params.get("max_exemplars_per_class", 20))) \
            if plugin.prompt_mode == "exemplars" and not plugin.uses_reviewed_images else []
        if plugin.uses_reviewed_images:
            exemplars = [Exemplar(r.image, c, x, y, w, h) for r in reviewed for (c, x, y, w, h) in r.boxes]
        if plugin.prompt_mode == "exemplars":
            if not exemplars:
                raise RuntimeError("沒有範例：請先在幾張圖片上標出物件並按「確認」，再執行")
            per_class = {}
            for e in exemplars:
                per_class[weight_classes[e.class_index]] = per_class.get(weight_classes[e.class_index], 0) + 1
            hooks.log("範例：" + "、".join(f"{k} {v} 個" for k, v in per_class.items())
                      + (f"（{len(reviewed)} 張已確認圖片，其中 {sum(1 for r in reviewed if not r.boxes)} 張沒有物件）"
                         if plugin.uses_reviewed_images else ""))
            missing = [c.name for c in classes if c.name not in per_class]
            if missing:
                hooks.log(f"這些類別沒有範例，不會被標註：{', '.join(missing)}")

        prompts: list[str] = []
        if plugin.prompt_mode == "text":
            given = {str(k): v for k, v in (params.get("prompts") or {}).items()}
            prompts = [(given.get(str(c.id)) or c.name).strip() for c in classes]
            hooks.log("類別描述：" + "；".join(f"{c.name} → {p}" for c, p in zip(classes, prompts)))

        stmt = select(Image).where(Image.project_id == project_id, Image.status.in_(AUTOLABEL_ALLOWED))
        if params.get("image_ids"):
            stmt = stmt.where(Image.id.in_(params["image_ids"]))
        else:
            statuses = [s for s in params.get("statuses", [STATUS_UNLABELED]) if s in AUTOLABEL_ALLOWED]
            stmt = stmt.where(Image.status.in_(statuses or [STATUS_UNLABELED]))
        targets = {
            str(storage.image_path(project_id, img.stored_name)): (img.id, img.version)
            for img in db.scalars(stmt.order_by(Image.id))
        }
        weights_path = Path(weights.path) if weights else None
        variant = weights.variant if weights else params.get("variant")
        image_size = int((weights.metrics or {}).get("train_image_size", 640)) if weights else 640

    if not targets:
        return {"summary": "沒有符合條件的圖片（只會處理「未標註」或「自動標註」的圖片）", "labelled": 0}

    hooks.log(f"共 {len(targets)} 張圖片，信心門檻 {params.get('confidence', 0.25)}")
    ctx = PredictContext(
        weights_path=weights_path,
        class_names=weight_classes,
        variant=variant,
        confidence=float(params.get("confidence", 0.25)),
        device=params.get("device", "auto"),
        work_dir=work_dir,
        hooks=hooks,
        image_size=image_size,
        exemplars=exemplars,
        prompts=prompts,
        reviewed=reviewed,
        cache_dir=settings.data_dir / "feature_cache" / str(project_id),
        options={k: params[k] for k in ("probe_head",) if k in params},
    )
    unknown_names: dict[str, int] = {}

    def to_class_id(p) -> int | None:
        if p.class_name:
            cid = by_name.get(p.class_name.strip().lower())
            if cid is None:
                unknown_names[p.class_name] = unknown_names.get(p.class_name, 0) + 1
            return cid
        return index_map.get(p.class_index)

    done = boxes_total = skipped = 0
    paths = [Path(p) for p in targets]
    for path, preds in plugin.predict(ctx, paths):
        hooks.check_cancel()
        img_id, version = targets[str(path)]
        boxes = [
            Box(cid, p.x, p.y, p.w, p.h, "auto", round(float(p.score), 4), p.polygon)
            for p in preds
            if (cid := to_class_id(p)) is not None
        ]
        with SessionLocal() as db:
            img = db.get(Image, img_id)
            # someone may have started editing while the job was running – never clobber that
            if img is None or img.status not in AUTOLABEL_ALLOWED or img.version != version:
                skipped += 1
            else:
                apply_boxes(db, img, boxes, "auto_label", note=f"{plugin.name}")
                db.commit()
                boxes_total += len(boxes)
        done += 1
        hooks.progress(done / len(paths), f"自動標註 {done}/{len(paths)}")

    shutil.rmtree(work_dir, ignore_errors=True)
    summary = f"完成 {done - skipped} 張，共 {boxes_total} 個框"
    if skipped:
        summary += f"；{skipped} 張在執行期間被人工修改，已略過"
    if unknown_names:
        top = sorted(unknown_names.items(), key=lambda kv: -kv[1])[:8]
        hooks.log("模型偵測到、但專案中沒有的類別：" + "、".join(f"{k}（{v}）" for k, v in top))
        if boxes_total == 0:
            summary += "；模型找到的類別名稱都不在專案中（例如 " + "、".join(k for k, _ in top[:3]) + "）"
    return {"summary": summary, "labelled": done - skipped, "boxes": boxes_total, "skipped": skipped}


# ----------------------------------------------------------------- train
def _collect_reviewed(db, project: Project, limit: int) -> list[ReviewedImage]:
    """Most recently confirmed images with all their boxes."""
    index = {c.id: i for i, c in enumerate(project.classes)}
    imgs = db.scalars(
        select(Image)
        .where(Image.project_id == project.id, Image.status == STATUS_REVIEWED)
        .order_by(Image.reviewed_at.desc(), Image.id)
        .limit(limit)
    ).all()
    out = []
    for img in imgs:
        known = [a for a in img.annotations if a.class_id in index]
        boxes = [(index[a.class_id], a.x, a.y, a.w, a.h) for a in known]
        polygons = [a.polygon for a in known]
        out.append(ReviewedImage(storage.image_path(project.id, img.stored_name), img.width, img.height, boxes, polygons))
    return out


def _collect_exemplars(db, project: Project, per_class: int) -> list[Exemplar]:
    """Confirmed boxes used as visual examples, at most ``per_class`` per class (newest first)."""
    index = {c.id: i for i, c in enumerate(project.classes)}
    counts: dict[int, int] = {}
    out: list[Exemplar] = []
    rows = db.execute(
        select(Annotation, Image)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project.id, Image.status == STATUS_REVIEWED)
        .order_by(Image.reviewed_at.desc(), Annotation.id)
    )
    for ann, img in rows:
        if counts.get(ann.class_id, 0) >= per_class or ann.class_id not in index:
            continue
        counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
        out.append(
            Exemplar(storage.image_path(project.id, img.stored_name), index[ann.class_id], ann.x, ann.y, ann.w, ann.h)
        )
    return out


def _train(project_id: int, plugin, params: dict, work_dir: Path, hooks: _Hooks, job_id: int) -> dict:
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        stats = dataset.split_stats(db, project)
        if stats["images"]["train"] == 0 or stats["images"]["valid"] == 0:
            raise RuntimeError("train 和 valid 都至少要有一張已確認的圖片，請先完成審核並執行資料切分")
        class_names = [c.name for c in project.classes]
        ds_dir = work_dir / "dataset"
        hooks.log(f"匯出資料集（{plugin.dataset_format}）：train {stats['images']['train']}、"
                  f"valid {stats['images']['valid']}、test {stats['images']['test']}")
        if plugin.dataset_format == "coco":
            entries = dataset.iter_coco(db, project, category_id_offset=plugin.coco_category_offset)
        else:
            segmentation = plugin.dataset_format == "yolo-seg"
            if segmentation and stats["boxes_without_polygon"]:
                hooks.log(f"注意：{stats['boxes_without_polygon']} 個物件只有框，訓練時會當成矩形，可能拉低分割品質")
            entries = dataset.iter_yolo(db, project, dataset_root=str(ds_dir.resolve()), segmentation=segmentation)
        dataset.write_dir(entries, ds_dir)
        init_weights = None
        if params.get("init_weights_id"):
            w = db.get(ModelWeights, params["init_weights_id"])
            init_weights = Path(w.path) if w else None

    ctx = TrainContext(
        dataset_dir=ds_dir,
        class_names=class_names,
        variant=params["variant"],
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        image_size=int(params["image_size"]),
        device=params.get("device", "auto"),
        output_dir=work_dir / "output",
        init_weights=init_weights,
        extra=params.get("extra", {}),
        hooks=hooks,
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    result = plugin.train(ctx)

    metrics = {**result.metrics, "train_image_size": ctx.image_size,
               "train_images": stats["images"]["train"], "valid_images": stats["images"]["valid"]}
    variant_name = next((v.name for v in plugin.variants if v.id == ctx.variant), ctx.variant)
    with SessionLocal() as db:
        w = ModelWeights(
            project_id=project_id,
            plugin=plugin.id,
            variant=ctx.variant,
            name=f"{variant_name}（任務 #{job_id}）",
            path=str(Path(result.weights_path).resolve()),
            class_names=class_names,
            metrics=metrics,
            job_id=job_id,
        )
        db.add(w)
        db.commit()
        weights_id = w.id
    # the dataset copy is no longer needed (images are hard links anyway)
    shutil.rmtree(ds_dir, ignore_errors=True)
    m = ", ".join(f"{k} {v}" for k, v in result.metrics.items() if k.startswith("mAP"))
    return {"summary": f"訓練完成{('：' + m) if m else ''}", "weights_id": weights_id, "metrics": metrics}


runner = JobRunner()


# --------------------------------------------------------------- job records
def next_job_id(db) -> int:
    """SQLite hands out a deleted row's id again.  Weights outlive their job and are named after
    it, so an id they still refer to must not come back."""
    last_job = db.scalar(select(func.max(Job.id))) or 0
    last_weights = db.scalar(select(func.max(ModelWeights.job_id))) or 0
    return max(last_job, last_weights) + 1


def new_work_dir(job_id: int) -> Path:
    """A folder of its own: one left by an older job may hold files that must not be overwritten."""
    work_dir = settings.runs_dir / f"job_{job_id:05d}"
    if work_dir.exists():
        work_dir = settings.runs_dir / f"job_{job_id:05d}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def delete_job_record(db, job: Job) -> None:
    """Remove a finished job and its leftover files.  Annotations it made stay, and so do the
    weights it trained: they live inside its run folder, which is then kept."""
    job_id = job.id
    db.delete(job)
    db.flush()
    weights = [Path(p).resolve() for p in db.scalars(select(ModelWeights.path))]
    base = f"job_{job_id:05d}"
    if not settings.runs_dir.exists():
        return
    for folder in settings.runs_dir.iterdir():
        if folder.is_dir() and (folder.name == base or folder.name.startswith(f"{base}_")):
            resolved = folder.resolve()
            if not any(resolved in w.parents for w in weights):
                shutil.rmtree(folder, ignore_errors=True)

