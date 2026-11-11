"""YOLOv9 – drives the *original authors'* repository (WongKinYiu/yolov9).

Licence note: that repository is GPL-3.0.  VisionLoop does not bundle or
import any of its code; it runs the repo's own scripts as a separate process
in a folder you clone yourself: ``train_dual.py`` / ``detect_dual.py`` for
YOLOv9, which trains with an auxiliary branch, and ``train.py`` / ``detect.py``
for GELAN, which has none.  (Ultralytics' YOLOv9 re-implementation is
intentionally not used.)

Setup
-----
    git clone https://github.com/WongKinYiu/yolov9
    pip install -r yolov9/requirements.txt          # ideally in its own venv
    export VISIONLOOP_YOLOV9_REPO=/path/to/yolov9
    export VISIONLOOP_YOLOV9_PYTHON=/path/to/venv/bin/python   # optional

Pretrained weights for fine-tuning (e.g. ``yolov9-c.pt``, ``gelan-s.pt``) are taken from
``$VISIONLOOP_YOLOV9_REPO/weights/`` and downloaded there from the authors' release on first
use of a size.  YOLOv9-T has none published and trains from scratch.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import yaml
from PIL import Image as PILImage

from ..base import (
    ModelPlugin,
    PluginStatus,
    PredictContext,
    Prediction,
    TrainContext,
    TrainResult,
    Variant,
    chunked,
    python_executable,
    run_process,
)

EPOCH_RX = re.compile(r"^\s*(\d+)/(\d+)\s")


def _repo() -> Path | None:
    p = os.environ.get("VISIONLOOP_YOLOV9_REPO")
    if p:
        return Path(p).expanduser()
    default = Path(__file__).resolve().parents[4] / "models" / "yolov9"  # cloned by setup_windows.ps1
    return default if default.exists() else None


# The repo predates PyTorch 2.6, whose torch.load() defaults to weights_only=True and refuses
# YOLOv9 checkpoints (pickled model objects). This env var restores the old default for the
# repo's own trusted checkpoints without modifying its (GPL) code.
REPO_ENV = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
# The repo also uses NumPy functions removed in NumPy 2.x; its scripts are started through
# this launcher, which puts them back (see yolov9_compat.py).
COMPAT = str(Path(__file__).with_name("yolov9_compat.py"))


def _device(device: str) -> str:
    return "" if device in ("", "auto") else device


class YoloV9Plugin(ModelPlugin):
    id = "yolov9"
    name = "YOLOv9（原作者版本）"
    family = "YOLOv9"
    description = "WongKinYiu/yolov9 官方實作，以獨立程序執行。YOLOv9 使用 train_dual.py / detect_dual.py，GELAN 使用 train.py / detect.py。"
    license = "GPL-3.0（模型程式碼，外部執行，不隨本專案散布）"
    homepage = "https://github.com/WongKinYiu/yolov9"
    dataset_format = "yolo"
    # Smallest first. The form recommends the last size that fits the GPU, so at each size YOLOv9
    # comes after GELAN. VRAM of GELAN is an upper bound: without the auxiliary branch it needs less.
    variants = [
        Variant("t", "YOLOv9-T", "最輕量，沒有官方預訓練權重（從頭訓練，需要大量資料）", 2, 2.0),
        Variant("gelan-s", "GELAN-S", "小型，沒有輔助分支", 4),
        Variant("s", "YOLOv9-S", "小型，速度與準確度平衡", 4, 7.1),
        Variant("gelan-m", "GELAN-M", "中型，沒有輔助分支", 6),
        Variant("m", "YOLOv9-M", "中型", 6, 20.0),
        Variant("gelan-c", "GELAN-C", "沒有輔助分支，訓練較省資源", 8, 25.3),
        Variant("c", "YOLOv9-C", "官方主力尺寸", 8, 25.3),
        Variant("gelan-e", "GELAN-E", "大型，沒有輔助分支", 16),
        Variant("e", "YOLOv9-E", "最大、最準，需要大顯存", 16, 57.3),
    ]

    def status(self) -> PluginStatus:
        repo = _repo()
        if repo is None:
            return PluginStatus(False, "尚未安裝 YOLOv9（GPL-3.0）：執行 scripts\\setup_windows.ps1 -Yolov9 yes 安裝，或設定 VISIONLOOP_YOLOV9_REPO")
        if not (repo / "train_dual.py").exists():
            return PluginStatus(False, f"{repo} 裡找不到 train_dual.py，請確認路徑")
        return PluginStatus(True, str(repo))

    # ------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> TrainResult:
        repo = _require_repo(self)
        hooks = ctx.hooks
        v = ctx.variant if ctx.variant in {x.id for x in self.variants} else "c"

        data_yaml = ctx.dataset_dir / "data.yaml"
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        data["path"] = str(ctx.dataset_dir.resolve())  # the repo resolves relative paths against itself
        data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        init = ctx.init_weights or _pretrained(repo, f"{_model_name(v)}.pt", hooks)
        if init:
            hooks.log(f"初始權重：{init}")
        else:
            hooks.log(f"{_model_name(v)} 沒有官方預訓練權重，將從頭開始訓練（資料少時效果會差很多，建議改用 S 以上的尺寸）")

        close_mosaic = min(15, ctx.epochs // 2)
        cmd = [
            python_executable("VISIONLOOP_YOLOV9_PYTHON"),
            COMPAT,
            "train.py" if _is_gelan(v) else "train_dual.py",
            "--workers", str(ctx.extra.get("workers", 4)),
            "--device", _device(ctx.device),
            "--batch", str(ctx.batch_size),
            "--data", str(data_yaml),
            "--img", str(ctx.image_size),
            "--cfg", f"models/detect/{_model_name(v)}.yaml",
            "--weights", str(init) if init else "",
            "--hyp", "data/hyps/hyp.scratch-high.yaml",
            "--min-items", "0",
            "--epochs", str(ctx.epochs),
            "--close-mosaic", str(close_mosaic),
            "--project", str(ctx.output_dir),
            "--name", "train",
            "--exist-ok",
        ]
        if ctx.extra.get("noplots", False):  # the charts (results.png, confusion matrix, PR curves) are on by default
            cmd.append("--noplots")

        total = ctx.epochs

        def on_line(line: str) -> None:
            m = EPOCH_RX.match(line)
            if m:
                epoch, last = int(m.group(1)), int(m.group(2))
                hooks.progress(min(epoch / (last + 1), 0.99), f"epoch {epoch + 1}/{last + 1}")

        hooks.progress(0.0, f"epoch 1/{total}")
        code = run_process(cmd, cwd=repo, hooks=hooks, env=REPO_ENV, on_line=on_line)
        run_dir = ctx.output_dir / "train"
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            best = run_dir / "weights" / "last.pt"
        if code != 0 or not best.exists():
            raise RuntimeError(f"YOLOv9 訓練失敗（exit code {code}），請查看任務紀錄")
        return TrainResult(best, _read_metrics(run_dir / "results.csv"))

    # ---------------------------------------------------------- predict
    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        repo = _require_repo(self)
        if ctx.weights_path is None:
            raise RuntimeError("YOLOv9 自動標註需要先選擇一個權重")
        # dual-branch checkpoints use detect_dual.py, re-parameterised ("converted") / GELAN use detect.py.
        # Weights trained here are all called best.pt, so the size decides; the name covers imported files.
        wname = ctx.weights_path.name.lower()
        single = _is_gelan(ctx.variant) or "converted" in wname or "gelan" in wname
        script = "detect.py" if single else "detect_dual.py"
        imgsz = ctx.image_size

        for batch_no, batch in enumerate(chunked(images, 500)):
            ctx.hooks.check_cancel()
            src = ctx.work_dir / f"src_{batch_no}"
            out = ctx.work_dir / f"pred_{batch_no}"
            shutil.rmtree(src, ignore_errors=True)
            src.mkdir(parents=True)
            for p in batch:
                _link(p, src / p.name)
            cmd = [
                python_executable("VISIONLOOP_YOLOV9_PYTHON"),
                COMPAT,
                script,
                "--weights", str(ctx.weights_path),
                "--source", str(src),
                "--img", str(imgsz),
                "--conf-thres", str(ctx.confidence),
                "--device", _device(ctx.device),
                "--save-txt",
                "--save-conf",
                "--nosave",
                "--project", str(ctx.work_dir),
                "--name", out.name,
                "--exist-ok",
            ]
            code = run_process(cmd, cwd=repo, hooks=ctx.hooks, env=REPO_ENV)
            if code != 0:
                raise RuntimeError(f"YOLOv9 推論失敗（exit code {code}），請查看任務紀錄")
            for p in batch:
                yield p, _read_yolo_txt(out / "labels" / f"{p.stem}.txt", p)


class YoloV9SegPlugin(ModelPlugin):
    """Instance segmentation from the same repository (``segment/train.py``, ``segment/predict.py``).

    GELAN-C-SEG is the only size with published pretrained weights (``weights/gelan-c-seg.pt``).
    """

    id = "yolov9-seg"
    name = "YOLOv9 分割（GELAN-C-SEG，原作者版本）"
    family = "YOLOv9"
    task = "segment"
    description = "WongKinYiu/yolov9 官方的分割模型，以獨立程序執行 segment/train.py / segment/predict.py。只有框的物件會被當成矩形。"
    license = "GPL-3.0（模型程式碼，外部執行，不隨本專案散布）"
    homepage = "https://github.com/WongKinYiu/yolov9"
    dataset_format = "yolo-seg"
    variants = [Variant("gelan-c", "GELAN-C-SEG", "官方提供預訓練權重的分割尺寸", 8, 27.4)]

    def status(self) -> PluginStatus:
        repo = _repo()
        if repo is None:
            return PluginStatus(False, "尚未安裝 YOLOv9（GPL-3.0）：執行 scripts\\setup_windows.ps1 -Yolov9 yes 安裝，或設定 VISIONLOOP_YOLOV9_REPO")
        if not (repo / "segment" / "train.py").exists():
            return PluginStatus(False, f"{repo} 裡找不到 segment/train.py，請確認路徑")
        return PluginStatus(True, str(repo))

    def train(self, ctx: TrainContext) -> TrainResult:
        repo = _require_repo(self)
        hooks = ctx.hooks
        data_yaml = ctx.dataset_dir / "data.yaml"
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        data["path"] = str(ctx.dataset_dir.resolve())  # the repo resolves relative paths against itself
        data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

        init = ctx.init_weights or _pretrained(repo, "gelan-c-seg.pt", hooks)
        hooks.log(f"初始權重：{init}")

        cmd = [
            python_executable("VISIONLOOP_YOLOV9_PYTHON"),
            COMPAT,
            "segment/train.py",
            "--workers", str(ctx.extra.get("workers", 4)),
            "--device", _device(ctx.device),
            "--batch", str(ctx.batch_size),
            "--data", str(data_yaml),
            "--img", str(ctx.image_size),
            "--cfg", "models/segment/gelan-c-seg.yaml",
            "--weights", str(init) if init else "",
            "--hyp", "data/hyps/hyp.scratch-high.yaml",
            "--no-overlap",
            "--epochs", str(ctx.epochs),
            "--close-mosaic", str(min(15, ctx.epochs // 2)),
            "--project", str(ctx.output_dir),
            "--name", "train",
            "--exist-ok",
        ]
        if ctx.extra.get("noplots", False):
            cmd.append("--noplots")

        def on_line(line: str) -> None:
            m = EPOCH_RX.match(line)
            if m:
                epoch, last = int(m.group(1)), int(m.group(2))
                hooks.progress(min(epoch / (last + 1), 0.99), f"epoch {epoch + 1}/{last + 1}")

        hooks.progress(0.0, f"epoch 1/{ctx.epochs}")
        code = run_process(cmd, cwd=repo, hooks=hooks, env=REPO_ENV, on_line=on_line)
        run_dir = ctx.output_dir / "train"
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            best = run_dir / "weights" / "last.pt"
        if code != 0 or not best.exists():
            raise RuntimeError(f"YOLOv9 分割訓練失敗（exit code {code}），請查看任務紀錄")
        return TrainResult(best, _read_metrics(run_dir / "results.csv", suffix="(M)"))

    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        repo = _require_repo(self)
        if ctx.weights_path is None:
            raise RuntimeError("YOLOv9 分割的自動標註需要先選擇一個權重")
        for batch_no, batch in enumerate(chunked(images, 500)):
            ctx.hooks.check_cancel()
            src = ctx.work_dir / f"src_{batch_no}"
            out = ctx.work_dir / f"pred_{batch_no}"
            shutil.rmtree(src, ignore_errors=True)
            src.mkdir(parents=True)
            for p in batch:
                _link(p, src / p.name)
            cmd = [
                python_executable("VISIONLOOP_YOLOV9_PYTHON"),
                COMPAT,
                "segment/predict.py",
                "--weights", str(ctx.weights_path),
                "--source", str(src),
                "--img", str(ctx.image_size),
                "--conf-thres", str(ctx.confidence),
                "--device", _device(ctx.device),
                "--save-txt",
                "--save-conf",
                "--nosave",
                "--project", str(ctx.work_dir),
                "--name", out.name,
                "--exist-ok",
            ]
            code = run_process(cmd, cwd=repo, hooks=ctx.hooks, env=REPO_ENV)
            if code != 0:
                raise RuntimeError(f"YOLOv9 分割推論失敗（exit code {code}），請查看任務紀錄")
            for p in batch:
                yield p, _read_seg_txt(out / "labels" / f"{p.stem}.txt", p)


# ---------------------------------------------------------------- helpers
def _read_seg_txt(path: Path, image: Path) -> list[Prediction]:
    """``class x1 y1 x2 y2 ... conf`` lines, normalised."""
    if not path.exists():
        return []
    with PILImage.open(image) as im:
        W, H = im.size
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        values = list(map(float, parts[1:-1]))
        if len(values) < 6 or len(values) % 2:  # an empty mask has no usable outline
            continue
        polygon = [[x * W, y * H] for x, y in zip(values[0::2], values[1::2])]
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        out.append(Prediction(
            int(parts[0]), min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), float(parts[-1]), polygon=polygon,
        ))
    return out


def _require_repo(plugin: ModelPlugin) -> Path:
    st = plugin.status()
    if not st.available:
        raise RuntimeError(st.message)
    return _repo()  # type: ignore[return-value]


def _is_gelan(variant: str | None) -> bool:
    return bool(variant) and variant.startswith("gelan-")


def _model_name(variant: str) -> str:
    """The file name the repo uses for a size, without extension: gelan-c, yolov9-c."""
    return variant if _is_gelan(variant) else f"yolov9-{variant}"


# Training checkpoints the authors publish (with the auxiliary branch, so train_dual.py can fine-tune
# them), in MB. YOLOv9-T only has a converted (inference) checkpoint, so it trains from scratch.
RELEASE_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/"
PUBLISHED_MB = {
    "yolov9-s.pt": 19, "yolov9-m.pt": 63, "yolov9-c.pt": 98, "yolov9-e.pt": 133,
    "gelan-s.pt": 14, "gelan-m.pt": 38, "gelan-c.pt": 49, "gelan-e.pt": 111,
    "gelan-c-seg.pt": 53,
}


def _download(url: str, dest: Path, hooks) -> None:
    """Fetches ``url`` to ``dest`` through a temporary file, so an interrupted download leaves nothing."""
    import urllib.request

    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as r, part.open("wb") as f:
            while chunk := r.read(1 << 20):
                hooks.check_cancel()
                f.write(chunk)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def _pretrained(repo: Path, name: str, hooks) -> Path | None:
    """The authors' pretrained checkpoint ``name``, downloaded into ``repo/weights`` on first use.

    Without one, a few hundred labelled images train to a mAP of about 0, so a failed download
    stops the job instead of silently training from scratch.
    """
    for p in (repo / "weights" / name, repo / name):
        if p.exists():
            return p.resolve()
    if name not in PUBLISHED_MB:
        return None
    dest = repo / "weights" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    hooks.progress(0.0, f"第一次使用，正在下載 {name}（約 {PUBLISHED_MB[name]} MB）…")
    hooks.log(f"下載預訓練權重 {RELEASE_URL}{name}")
    try:
        _download(RELEASE_URL + name, dest, hooks)
    except OSError as exc:
        raise RuntimeError(
            f"無法下載預訓練權重 {name}（{exc}）。沒有它從頭訓練，資料少時 mAP 幾乎是 0。"
            f"請確認網路可以連到 github.com 後重試，或手動下載 {RELEASE_URL}{name} 放到 {dest.parent}"
        ) from exc
    return dest.resolve()


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_yolo_txt(path: Path, image: Path) -> list[Prediction]:
    if not path.exists():
        return []
    with PILImage.open(image) as im:
        W, H = im.size
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        c, cx, cy, w, h, conf = int(parts[0]), *map(float, parts[1:6])
        out.append(Prediction(c, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, conf))
    return out


def _read_metrics(results_csv: Path, suffix: str = "") -> dict:
    """``suffix``: the segmentation scripts log box "(B)" and mask "(M)" columns."""
    if not results_csv.exists():
        return {}
    with results_csv.open(encoding="utf-8") as f:
        # the segmentation scripts end the file with a summary row that does not match the header
        rows = [
            {k.strip(): v.strip() for k, v in r.items()}
            for r in csv.DictReader(f)
            if None not in r and None not in r.values()
        ]
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get(f"metrics/mAP_0.5:0.95{suffix}", 0) or 0))
    pick = {
        "mAP50": f"metrics/mAP_0.5{suffix}",
        "mAP50-95": f"metrics/mAP_0.5:0.95{suffix}",
        "precision": f"metrics/precision{suffix}",
        "recall": f"metrics/recall{suffix}",
    }
    out = {k: round(float(best[v]), 4) for k, v in pick.items() if best.get(v)}
    out["best_epoch"] = int(float(best.get("epoch", 0)))
    out["epochs"] = len(rows)
    return out
