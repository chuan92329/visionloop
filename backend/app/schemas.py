from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- classes
class LabelClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: str | None = None


class LabelClassUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    color: str | None = None
    order_index: int | None = None


class LabelClassOut(ORM):
    id: int
    name: str
    color: str
    order_index: int


# --------------------------------------------------------------- projects
class SplitConfig(BaseModel):
    train: float = Field(0.7, ge=0, le=1)
    valid: float = Field(0.2, ge=0, le=1)
    test: float = Field(0.1, ge=0, le=1)
    seed: int = 42

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train + valid + test 必須等於 1（目前為 {total:.3f}）")
        if self.train <= 0:
            raise ValueError("train 比例必須大於 0")
        return self


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    classes: list[str] = []


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectStats(BaseModel):
    total: int = 0
    unlabeled: int = 0
    auto: int = 0
    in_progress: int = 0
    reviewed: int = 0
    train: int = 0
    valid: int = 0
    test: int = 0
    boxes: int = 0


class ProjectOut(ORM):
    id: int
    name: str
    description: str
    created_at: datetime
    split_config: dict
    test_locked: bool
    test_locked_at: datetime | None
    classes: list[LabelClassOut]
    stats: ProjectStats = ProjectStats()
    cover_image_id: int | None = None


# ----------------------------------------------------------------- images
class ImageOut(ORM):
    id: int
    project_id: int
    filename: str
    width: int
    height: int
    status: str
    split: str | None
    group_key: str | None
    version: int
    box_count: int
    min_score: float | None
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None


class ImagePage(BaseModel):
    items: list[ImageOut]
    total: int
    offset: int
    limit: int


class ImageUpdate(BaseModel):
    group_key: str | None = None
    split: Literal["train", "valid", "test"] | None = None
    clear_split: bool = False


class UploadResult(BaseModel):
    created: list[ImageOut]
    skipped: list[dict]


# ------------------------------------------------------------ annotations
class BoxIn(BaseModel):
    class_id: int
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)
    source: Literal["manual", "auto"] = "manual"
    score: float | None = Field(default=None, ge=0, le=1)
    polygon: list[tuple[float, float]] | None = Field(default=None, min_length=3)


class BoxOut(ORM):
    id: int
    class_id: int
    x: float
    y: float
    w: float
    h: float
    source: str
    score: float | None
    polygon: list[list[float]] | None = None


class SaveAnnotations(BaseModel):
    boxes: list[BoxIn]
    # optimistic locking: the version the client started editing from
    base_version: int | None = None
    # "save" keeps the status (or moves auto/unlabeled -> in_progress), "review" confirms the image
    mark_reviewed: bool = False
    note: str = ""

    @field_validator("note")
    @classmethod
    def _short_note(cls, v: str) -> str:
        return v[:500]


class AnnotationState(BaseModel):
    image: ImageOut
    boxes: list[BoxOut]


class RevisionOut(ORM):
    id: int
    version: int
    action: str
    status: str
    boxes: list
    note: str
    created_at: datetime


# -------------------------------------------------------------- dataset
class SplitRequest(BaseModel):
    config: SplitConfig | None = None
    # incremental: only assign reviewed images that have no split yet (stable across runs)
    # full: reshuffle everything except a locked test set
    mode: Literal["incremental", "full"] = "incremental"
    stratify: bool = True
    lock_test: bool | None = None


class SplitStats(BaseModel):
    images: dict[str, int]
    boxes: dict[str, dict[str, int]]  # split -> class name -> count
    # exported objects with / without an outline (the latter become rectangles in segmentation exports)
    polygons: int = 0
    boxes_without_polygon: int = 0
    unassigned_reviewed: int
    not_reviewed: int
    test_locked: bool
    warnings: list[str] = []


class AutoGroupRequest(BaseModel):
    # regex with ONE capture group applied to the file name (without extension);
    # default strips a trailing frame number: "cam1_000123" -> "cam1"
    pattern: str = r"^(.*?)[_\-\s]*\d+$"
    overwrite: bool = False


# ------------------------------------------------------------ jobs/plugins
class AutoLabelRequest(BaseModel):
    plugin: str
    weights_id: int | None = None
    variant: str | None = None
    confidence: float = Field(0.25, ge=0, le=1)
    # which images: by status (default only unlabeled) or explicit ids
    statuses: list[str] = ["unlabeled"]
    image_ids: list[int] | None = None
    device: str = "auto"
    # text-prompted models: class id -> description (defaults to the class name)
    prompts: dict[int, str] = {}
    # few-shot models: how many confirmed boxes per class are used as examples
    max_exemplars_per_class: int = Field(20, ge=1, le=200)
    # probe models: how many confirmed images (newest first) are used for training
    max_reviewed_images: int = Field(1000, ge=1, le=20000)
    # probe models: linear (default) or mlp head
    probe_head: Literal["linear", "mlp"] = "linear"
    # box -> outline models: how many vertices, and whether to redo objects that already have an outline
    polygon_detail: Literal["normal", "fine"] = "normal"
    overwrite_polygons: bool = False


class TrainRequest(BaseModel):
    plugin: str
    variant: str
    epochs: int = Field(50, ge=1, le=10000)
    batch_size: int = Field(8, ge=1, le=1024)
    # None: the size's own default
    image_size: int | None = Field(None, ge=32, le=4096)
    device: str = "auto"
    init_weights_id: int | None = None
    extra: dict = {}


class JobOut(ORM):
    id: int
    project_id: int
    kind: str
    plugin: str
    status: str
    progress: float
    message: str
    params: dict
    result: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobDetail(JobOut):
    log: str


class ModelWeightsOut(ORM):
    id: int
    plugin: str
    variant: str
    name: str
    class_names: list
    metrics: dict
    job_id: int | None
    created_at: datetime
