"""Model plugin interface.

A plugin wraps one detection framework.  The platform never imports a
framework directly; it only talks to plugins through this interface, so
frameworks under different licences (GPL, Apache, ...) stay out of the
platform's own MIT-licensed code.  The bundled YOLOv9 / RT-DETR plugins go
one step further and run the original authors' repositories as separate
processes.

Writing a plugin
----------------
Subclass :class:`ModelPlugin`, implement :meth:`train` and :meth:`predict`
and drop the file into ``backend/plugins/``.  It is discovered on start-up.
See ``backend/plugins/README.md`` for a minimal example.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Child processes must not open a console window: the desktop app has no console, so each
# console program it starts (python, nvidia-smi, taskkill) would flash a black window.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Variant:
    """A model size the user can pick, e.g. YOLOv9-S or RT-DETR-R18."""

    id: str
    name: str
    description: str = ""
    # rough VRAM needed to train at batch 8 / 640px; used to recommend sizes
    min_vram_gb: float = 0.0
    params_m: float | None = None
    #: the size this variant trains at unless told otherwise (None: 640)
    image_size: int | None = None
    #: training image sizes must be a multiple of this
    image_step: int = 32


@dataclass
class PluginStatus:
    available: bool
    message: str = ""


@dataclass
class Prediction:
    """One detected box in *pixel* coordinates of the original image."""

    class_index: int  # index into the weights' class list (ignored when class_name is set)
    x: float
    y: float
    w: float
    h: float
    score: float
    #: set by models that know their own label names (e.g. COCO-pretrained RF-DETR);
    #: the platform then maps the name to a project class instead of the index
    class_name: str | None = None
    #: outline [[x, y], ...] in pixels, from models that segment
    polygon: list[list[float]] | None = None


@dataclass
class Exemplar:
    """A human-confirmed box used as a visual example by few-shot plugins."""

    image: Path
    class_index: int  # index into PredictContext.class_names
    x: float
    y: float
    w: float
    h: float


@dataclass
class ReviewedImage:
    """A human-confirmed image with all its boxes (an image without boxes is a negative example)."""

    image: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]]  # (class_index, x, y, w, h)
    #: one entry per box: its outline [[x, y], ...] in pixels, or None if it is only a box
    polygons: list[list[list[float]] | None] = field(default_factory=list)


@dataclass
class TrainResult:
    weights_path: Path
    metrics: dict = field(default_factory=dict)


class Cancelled(Exception):
    pass


@dataclass
class JobHooks:
    """Callbacks a plugin uses to report back to the platform."""

    log: Callable[[str], None]
    progress: Callable[[float, str], None]  # fraction 0..1, short message
    should_stop: Callable[[], bool]

    def check_cancel(self) -> None:
        if self.should_stop():
            raise Cancelled()


@dataclass
class TrainContext:
    dataset_dir: Path  # exported dataset in the plugin's ``dataset_format``
    class_names: list[str]
    variant: str
    epochs: int
    batch_size: int
    image_size: int
    device: str  # "auto", "cpu", "0", "0,1", ...
    output_dir: Path
    init_weights: Path | None
    extra: dict
    hooks: JobHooks


@dataclass
class PredictContext:
    weights_path: Path | None
    class_names: list[str]  # classes the weights were trained on
    variant: str | None
    confidence: float
    device: str
    work_dir: Path
    hooks: JobHooks
    image_size: int = 640
    #: few-shot plugins (prompt_mode == "exemplars"): confirmed boxes to imitate
    exemplars: list[Exemplar] = field(default_factory=list)
    #: text plugins (prompt_mode == "text"): one description per entry of class_names
    prompts: list[str] = field(default_factory=list)
    #: plugins with uses_reviewed_images: every confirmed image with its boxes
    reviewed: list[ReviewedImage] = field(default_factory=list)
    #: a persistent folder a plugin may use to cache per-image results (e.g. backbone features)
    cache_dir: Path | None = None
    #: plugin-specific options from the auto-label request (e.g. probe_head)
    options: dict = field(default_factory=dict)
    #: prompt_mode == "boxes": image path -> boxes (x, y, w, h) to outline
    box_prompts: dict[str, list[tuple[float, float, float, float]]] = field(default_factory=dict)


class ModelPlugin(ABC):
    #: unique id used in the API, e.g. "yolov9"
    id: str = ""
    name: str = ""
    description: str = ""
    #: model family shown in the training form, e.g. "YOLOv9" (default: the plugin's name)
    family: str = ""
    #: "detect" trains on boxes, "segment" on polygons
    task: Literal["detect", "segment"] = "detect"
    #: licence of the *model code* the plugin drives
    license: str = ""
    homepage: str = ""
    #: dataset layout handed to ``train``
    dataset_format: Literal["yolo", "yolo-seg", "coco"] = "yolo"
    #: COCO category ids start at this number (only for dataset_format == "coco")
    coco_category_offset: int = 1
    #: can predict without weights trained in this platform (e.g. COCO-pretrained, zero-shot)
    needs_weights: bool = True
    #: label for the "no weights" choice in the UI when needs_weights is False
    no_weights_label: str = "不需要權重"
    #: False for zero-/few-shot models that only predict
    trainable: bool = True
    #: "exemplars": uses confirmed boxes as examples; "text": uses a description per class;
    #: "boxes": outlines the existing boxes of each image, one Prediction per box in input order
    #: (polygon=None where it could not), instead of finding new objects
    prompt_mode: Literal["none", "exemplars", "text", "boxes"] = "none"
    #: wants every confirmed image (with boxes, and without boxes as negatives) in PredictContext.reviewed
    uses_reviewed_images: bool = False
    #: suggested confidence threshold for auto-labelling
    default_confidence: float = 0.25
    variants: list[Variant] = []

    def status(self) -> PluginStatus:
        return PluginStatus(True)

    @abstractmethod
    def train(self, ctx: TrainContext) -> TrainResult: ...

    @abstractmethod
    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        """Yield ``(image_path, predictions)`` for every input image."""

    def describe(self) -> dict:
        st = self.status()
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "family": self.family or self.name,
            "task": self.task,
            "license": self.license,
            "homepage": self.homepage,
            "dataset_format": self.dataset_format,
            "needs_weights": self.needs_weights,
            "no_weights_label": self.no_weights_label,
            "trainable": self.trainable,
            "prompt_mode": self.prompt_mode,
            "default_confidence": self.default_confidence,
            "uses_reviewed_images": self.uses_reviewed_images,
            "variants": [asdict(v) for v in self.variants],
            "available": st.available,
            "status_message": st.message,
        }


# --------------------------------------------------------------- helpers
def run_process(
    cmd: list[str],
    *,
    cwd: Path,
    hooks: JobHooks,
    env: dict | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run an external command, stream its output into the job log and
    terminate it if the job is cancelled.  ``\\r``-separated progress bars
    (tqdm) are split into separate lines."""
    hooks.log("$ " + " ".join(map(str, cmd)))
    # UTF-8 so Chinese paths/messages survive on Windows (default console code page is cp950)
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(env or {})}
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover - windows
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    assert proc.stdout is not None
    buf = b""
    last_logged = ""
    try:
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0), default=-1)
                if idx < 0:
                    break
                line = buf[:idx].decode("utf-8", "replace").rstrip()
                buf = buf[idx + 1 :]
                if not line:
                    continue
                if on_line:
                    on_line(line)
                # progress bars redraw many times per second; only log finished bars
                is_bar = "%|" in line
                if not is_bar or "100%|" in line:
                    if line != last_logged:
                        hooks.log(line)
                    last_logged = line
            if hooks.should_stop():
                _terminate(proc)
                raise Cancelled()
        if buf.strip():
            line = buf.decode("utf-8", "replace").rstrip()
            hooks.log(line)
            if on_line:
                on_line(line)
        return proc.wait()
    finally:
        if proc.poll() is None:
            _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:  # pragma: no cover
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        kill_tree(proc)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a child and everything it started (e.g. data-loader workers of a training script)."""
    if os.name == "nt":  # pragma: no cover - windows
        # a hidden child has a console of its own, so Ctrl+Break from here may not reach it
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, creationflags=NO_WINDOW)
    if proc.poll() is None:
        proc.kill()


def python_executable(env_var: str) -> str:
    """Interpreter for a plugin; each framework may live in its own venv.

    Falls back to the shared model environment (VISIONLOOP_MODELS_PYTHON),
    then to the interpreter running the platform."""
    return os.environ.get(env_var) or os.environ.get("VISIONLOOP_MODELS_PYTHON") or sys.executable


_module_cache: dict[tuple[str, str], tuple[float, bool, str]] = {}


def check_python_module(python: str, module: str, ttl: float = 30.0) -> tuple[bool, str]:
    """Is ``module`` importable by ``python``?  Uses find_spec (fast, does not import torch).
    Cached for ``ttl`` seconds because the UI polls plugin status."""
    import time

    key = (python, module)
    hit = _module_cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1], hit[2]
    if not python or (os.path.sep in python and not os.path.exists(python)):
        result = (False, f"找不到 Python：{python}")
    else:
        try:
            r = subprocess.run(
                [python, "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({module!r}) else 3)"],
                capture_output=True,
                timeout=20,
                creationflags=NO_WINDOW,
            )
            result = (True, python) if r.returncode == 0 else (False, f"{python} 裡沒有安裝 {module}")
        except (OSError, subprocess.SubprocessError) as exc:
            result = (False, f"無法執行 {python}：{exc}")
    _module_cache[key] = (time.monotonic(), *result)
    return result


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
