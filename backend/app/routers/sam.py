"""Click-to-segment for the annotator: point and box prompts -> one outline."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..plugins import registry
from ..services import sam_interactive, storage
from .common import get_image

router = APIRouter(prefix="/api", tags=["sam"])


class SamPrompt(BaseModel):
    #: [x, y, 1] marks the object, [x, y, 0] marks something that is not part of it (image pixels)
    points: list[tuple[float, float, Literal[0, 1]]] = Field(default_factory=list, max_length=50)
    box: tuple[float, float, float, float] | None = None  # x, y, w, h
    detail: Literal["normal", "fine"] = "normal"
    #: outlines already annotated; their area is cut from the result (only the visible part of
    #: an object covered by another one is kept)
    exclude: list[list[tuple[float, float]]] = Field(default_factory=list, max_length=2000)


def _payload(db: Session, image_id: int, op: str) -> dict:
    sam = registry.get_plugin("sam2")
    status = sam.status() if sam else None
    if status is None or not status.available:
        raise HTTPException(503, f"SAM 2 目前無法使用：{status.message if status else '找不到 SAM 2'}")
    img = get_image(db, image_id)
    path = storage.image_path(img.project_id, img.stored_name)
    # a new key whenever the file changes, so the helper never reuses another image's encoding
    key = f"{path}|{path.stat().st_mtime_ns}"
    return {"op": op, "model": sam_interactive.DEFAULT_MODEL, "device": "auto", "image": str(path), "key": key}


@router.get("/sam/info")
def info():
    """Whether the click-to-segment model still has to be downloaded (the first use waits for it)."""
    from ..plugins.builtin.modern import HF_SIZES_MB, hf_cached

    model = sam_interactive.DEFAULT_MODEL
    return {"model": model, "cached": hf_cached(model), "size_mb": HF_SIZES_MB.get(model)}


def _ask(payload: dict) -> dict:
    try:
        return sam_interactive.session.request(payload)
    except sam_interactive.SamError as exc:
        from ..plugins.builtin.modern import explain_failure

        raise HTTPException(502, explain_failure(str(exc)) or str(exc)) from exc


@router.post("/images/{image_id}/sam/prepare")
def prepare(image_id: int, db: Session = Depends(get_db)):
    """Load the model and encode the image ahead of the first click."""
    _ask(_payload(db, image_id, "prepare"))
    return {"ready": True}


@router.post("/images/{image_id}/sam")
def segment(image_id: int, body: SamPrompt, db: Session = Depends(get_db)):
    if not body.points and body.box is None:
        raise HTTPException(422, "請點選物件或框選範圍")
    if body.box is not None and (body.box[2] <= 0 or body.box[3] <= 0):
        raise HTTPException(422, "框的寬高必須大於 0")
    answer = _ask({
        **_payload(db, image_id, "segment"),
        "points": [list(p) for p in body.points],
        "box": list(body.box) if body.box else None,
        "detail": body.detail,
        "exclude": [[list(p) for p in outline] for outline in body.exclude if len(outline) >= 3],
    })
    return {"polygon": answer.get("polygon"), "score": answer.get("score"), "removed": answer.get("removed", 0)}
