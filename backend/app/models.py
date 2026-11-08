"""ORM models.

Box coordinates are stored in *image pixel space* (top-left x/y, width,
height).  Conversion to YOLO / COCO happens only at export time, so the
annotation data never depends on a training framework.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Image review status
STATUS_UNLABELED = "unlabeled"  # nothing yet
STATUS_AUTO = "auto"  # pre-labelled by a model, waiting for a human
STATUS_IN_PROGRESS = "in_progress"  # a human edited it but has not confirmed
STATUS_REVIEWED = "reviewed"  # confirmed by a human -> eligible for the dataset
IMAGE_STATUSES = (STATUS_UNLABELED, STATUS_AUTO, STATUS_IN_PROGRESS, STATUS_REVIEWED)

SPLITS = ("train", "valid", "test")

DEFAULT_SPLIT_CONFIG = {"train": 0.7, "valid": 0.2, "test": 0.1, "seed": 42}


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    split_config: Mapped[dict] = mapped_column(JSON, default=lambda: dict(DEFAULT_SPLIT_CONFIG))
    test_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    test_locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    classes: Mapped[list[LabelClass]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="LabelClass.order_index"
    )
    images: Mapped[list[Image]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class LabelClass(Base):
    __tablename__ = "label_classes"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(16), default="#3b82f6")
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    project: Mapped[Project] = relationship(back_populates="classes")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(500))  # original file name
    stored_name: Mapped[str] = mapped_column(String(200))  # name on disk (unique)
    sha1: Mapped[str] = mapped_column(String(40), index=True)  # duplicate detection
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default=STATUS_UNLABELED, index=True)
    split: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    # Images sharing a group key always land in the same split (e.g. frames of one video)
    group_key: Mapped[str | None] = mapped_column(String(300), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    box_count: Mapped[int] = mapped_column(Integer, default=0)
    # Lowest confidence among model-generated boxes still on the image (for "review low score first")
    min_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="images")
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="image", cascade="all, delete-orphan", order_by="Annotation.id"
    )
    revisions: Mapped[list[Revision]] = relationship(
        back_populates="image", cascade="all, delete-orphan", order_by="Revision.version"
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("label_classes.id", ondelete="CASCADE"), index=True)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    # optional outline [[x, y], ...] in image pixels; x/y/w/h are then its bounding box,
    # so everything that only needs boxes keeps working unchanged
    polygon: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    # "auto" boxes carry the model confidence; once a human touches a box it becomes "manual"
    source: Mapped[str] = mapped_column(String(10), default="manual")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    image: Mapped[Image] = relationship(back_populates="annotations")


class Revision(Base):
    """Immutable snapshot of an image's boxes after every change (audit trail + undo across sessions)."""

    __tablename__ = "revisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(20))  # save | auto_label | review | revert
    status: Mapped[str] = mapped_column(String(20))
    boxes: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    image: Mapped[Image] = relationship(back_populates="revisions")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # autolabel | train
    plugin: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    log: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelWeights(Base):
    """A trained (or imported) checkpoint that can be used for auto-labelling."""

    __tablename__ = "model_weights"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    plugin: Mapped[str] = mapped_column(String(100))
    variant: Mapped[str] = mapped_column(String(50), default="")
    name: Mapped[str] = mapped_column(String(200))
    path: Mapped[str] = mapped_column(Text)
    class_names: Mapped[list] = mapped_column(JSON, default=list)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
