from __future__ import annotations

import ipaddress

from fastapi import HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Image, LabelClass, Project
from ..schemas import ProjectOut, ProjectStats

# Distinct, colour-blind friendly-ish palette for new classes
PALETTE = [
    "#3b82f6", "#f97316", "#22c55e", "#e11d48", "#a855f7", "#eab308", "#06b6d4", "#f43f5e",
    "#84cc16", "#6366f1", "#14b8a6", "#d946ef", "#f59e0b", "#0ea5e9", "#10b981", "#ef4444",
]


def get_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "找不到專案")
    return project


def get_image(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "找不到圖片")
    return image


def get_class(db: Session, class_id: int) -> LabelClass:
    cls = db.get(LabelClass, class_id)
    if cls is None:
        raise HTTPException(404, "找不到類別")
    return cls


def next_color(project: Project) -> str:
    used = {c.color.lower() for c in project.classes}
    for color in PALETTE:
        if color not in used:
            return color
    return PALETTE[len(project.classes) % len(PALETTE)]


def project_stats(db: Session, project_id: int) -> ProjectStats:
    stats = ProjectStats()
    for status, n in db.execute(
        select(Image.status, func.count()).where(Image.project_id == project_id).group_by(Image.status)
    ):
        setattr(stats, status, n)
        stats.total += n
    for split, n in db.execute(
        select(Image.split, func.count())
        .where(Image.project_id == project_id, Image.split.is_not(None))
        .group_by(Image.split)
    ):
        setattr(stats, split, n)
    stats.boxes = db.scalar(
        select(func.coalesce(func.sum(Image.box_count), 0)).where(Image.project_id == project_id)
    )
    return stats


def project_out(db: Session, project: Project) -> ProjectOut:
    out = ProjectOut.model_validate(project)
    out.stats = project_stats(db, project.id)
    out.cover_image_id = db.scalar(
        select(Image.id).where(Image.project_id == project.id).order_by(Image.id).limit(1)
    )
    return out


def is_local_request(request: Request) -> bool:
    """From the computer VisionLoop runs on (the desktop app), not a browser elsewhere."""
    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "testclient")
