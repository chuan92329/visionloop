"""RT-DETRv2 – drives the *original authors'* repository (lyuwenyu/RT-DETR,
folder ``rtdetrv2_pytorch``), Apache-2.0.  Ultralytics' RT-DETR
re-implementation is intentionally not used.

Setup
-----
    git clone https://github.com/lyuwenyu/RT-DETR
    pip install -r RT-DETR/rtdetrv2_pytorch/requirements.txt
    export VISIONLOOP_RTDETR_REPO=/path/to/RT-DETR/rtdetrv2_pytorch
    export VISIONLOOP_RTDETR_PYTHON=/path/to/venv/bin/python   # optional

Optional COCO-pretrained checkpoints (recommended) go into
``$VISIONLOOP_RTDETR_REPO/weights/``; a file whose name starts with
``rtdetrv2_<size>`` (e.g. ``rtdetrv2_r18vd_120e_coco_rerun_48.1.pth``) is
used for fine-tuning automatically.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

from ..base import (
    ModelPlugin,
    PluginStatus,
    PredictContext,
    Prediction,
    TrainContext,
    TrainResult,
    Variant,
    python_executable,
    run_process,
)

EPOCH_RX = re.compile(r"Epoch: \[(\d+)\]")
HELPER = Path(__file__).with_name("rtdetr_predict_helper.py")

CONFIGS = {
    "r18vd": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
    "r34vd": "configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",
    "r50vd_m": "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
    "r50vd": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
    "r101vd": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
}


def _repo() -> Path | None:
    p = os.environ.get("VISIONLOOP_RTDETR_REPO")
    return Path(p).expanduser() if p else None


def _device(device: str) -> str | None:
    if device in ("", "auto"):
        return None
    if device == "cpu":
        return "cpu"
    return f"cuda:{device.split(',')[0]}"


class RTDETRv2Plugin(ModelPlugin):
    id = "rtdetrv2"
    name = "RT-DETRv2（原作者版本）"
    family = "RT-DETRv2"
    description = "lyuwenyu/RT-DETR 官方實作（rtdetrv2_pytorch），以獨立程序執行。"
    license = "Apache-2.0"
    homepage = "https://github.com/lyuwenyu/RT-DETR"
    dataset_format = "coco"
    coco_category_offset = 0  # with remap_mscoco_category=False, category_id is used as the label index
    variants = [
        Variant("r18vd", "RT-DETRv2-S (R18)", "最輕量的 RT-DETR", 6, 20),
        Variant("r34vd", "RT-DETRv2-M* (R34)", "小型", 8, 31),
        Variant("r50vd_m", "RT-DETRv2-M (R50-M)", "中型", 10, 36),
        Variant("r50vd", "RT-DETRv2-L (R50)", "大型", 12, 42),
        Variant("r101vd", "RT-DETRv2-X (R101)", "最大，需要大顯存", 20, 76),
    ]

    def status(self) -> PluginStatus:
        repo = _repo()
        if repo is None:
            return PluginStatus(
                False, "尚未設定環境變數 VISIONLOOP_RTDETR_REPO（指向 RT-DETR/rtdetrv2_pytorch 資料夾）"
            )
        if not (repo / "tools" / "train.py").exists():
            return PluginStatus(False, f"{repo} 裡找不到 tools/train.py，請確認指向 rtdetrv2_pytorch 資料夾")
        return PluginStatus(True, str(repo))

    # ------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> TrainResult:
        repo = _require_repo(self)
        hooks = ctx.hooks
        v = ctx.variant if ctx.variant in CONFIGS else "r18vd"
        ds = ctx.dataset_dir.resolve()
        if ctx.image_size != 640:
            hooks.log("注意：RT-DETRv2 設定檔固定使用 640×640，image_size 參數不會生效")

        stop = max(1, round(ctx.epochs * 0.95))
        workers = int(ctx.extra.get("workers", 4))
        updates = [
            f"num_classes={len(ctx.class_names)}",
            "remap_mscoco_category=False",
            f"train_dataloader.dataset.img_folder={ds / 'images' / 'train'}",
            f"train_dataloader.dataset.ann_file={ds / 'annotations' / 'instances_train.json'}",
            f"val_dataloader.dataset.img_folder={ds / 'images' / 'valid'}",
            f"val_dataloader.dataset.ann_file={ds / 'annotations' / 'instances_valid.json'}",
            f"epoches={ctx.epochs}",
            f"train_dataloader.total_batch_size={ctx.batch_size}",
            f"val_dataloader.total_batch_size={ctx.batch_size}",
            f"train_dataloader.num_workers={workers}",
            f"val_dataloader.num_workers={workers}",
            f"train_dataloader.dataset.transforms.policy.epoch={stop}",
            f"train_dataloader.collate_fn.stop_epoch={stop}",
            "checkpoint_freq=100000",  # keep only last.pth / best.pth
        ]
        cmd = [
            python_executable("VISIONLOOP_RTDETR_PYTHON"),
            "tools/train.py",
            "-c", CONFIGS[v],
            "--use-amp",
            "--seed", "0",
            "--output-dir", str(ctx.output_dir),
        ]
        dev = _device(ctx.device)
        if dev:
            cmd += ["-d", dev]
        init = ctx.init_weights or _find_pretrained(repo, v)
        if init:
            cmd += ["-t", str(init)]
            hooks.log(f"微調自：{init}")
        else:
            hooks.log("找不到 COCO 預訓練權重，將只用 ImageNet backbone 開始訓練")
        cmd += ["-u", *updates]

        def on_line(line: str) -> None:
            m = EPOCH_RX.search(line)
            if m:
                e = int(m.group(1))
                hooks.progress(min(e / ctx.epochs, 0.99), f"epoch {e + 1}/{ctx.epochs}")

        hooks.progress(0.0, f"epoch 1/{ctx.epochs}")
        code = run_process(cmd, cwd=repo, hooks=hooks, on_line=on_line)
        best = ctx.output_dir / "best.pth"
        if not best.exists():
            best = ctx.output_dir / "last.pth"
        if code != 0 or not best.exists():
            raise RuntimeError(f"RT-DETRv2 訓練失敗（exit code {code}），請查看任務紀錄")
        metrics = _read_metrics(ctx.output_dir / "log.txt")
        metrics["config"] = CONFIGS[v]
        return TrainResult(best, metrics)

    # ---------------------------------------------------------- predict
    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        repo = _require_repo(self)
        if ctx.weights_path is None:
            raise RuntimeError("RT-DETRv2 自動標註需要先選擇一個權重")
        v = ctx.variant if ctx.variant in CONFIGS else "r18vd"
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        list_file = ctx.work_dir / "images.txt"
        list_file.write_text("\n".join(str(p) for p in images))
        out_file = ctx.work_dir / "predictions.jsonl"
        cmd = [
            python_executable("VISIONLOOP_RTDETR_PYTHON"),
            str(HELPER),
            "--repo", str(repo),
            "--config", CONFIGS[v],
            "--checkpoint", str(ctx.weights_path),
            "--num-classes", str(len(ctx.class_names)),
            "--conf", str(ctx.confidence),
            "--list", str(list_file),
            "--out", str(out_file),
            "--device", _device(ctx.device) or "auto",
        ]
        code = run_process(cmd, cwd=repo, hooks=ctx.hooks, on_line=_progress_parser(ctx, len(images)))
        if code != 0 or not out_file.exists():
            raise RuntimeError(f"RT-DETRv2 推論失敗（exit code {code}），請查看任務紀錄")
        results: dict[str, list[Prediction]] = {}
        for line in out_file.read_text().splitlines():
            rec = json.loads(line)
            results[rec["path"]] = [
                Prediction(int(c), x1, y1, x2 - x1, y2 - y1, s) for c, x1, y1, x2, y2, s in rec["boxes"]
            ]
        for p in images:
            yield p, results.get(str(p), [])


def _progress_parser(ctx: PredictContext, total: int):
    rx = re.compile(r"^\[(\d+)/(\d+)\]")

    def on_line(line: str) -> None:
        m = rx.match(line)
        if m:
            ctx.hooks.progress(int(m.group(1)) / max(total, 1) * 0.95, f"推論 {m.group(1)}/{total}")

    return on_line


def _require_repo(plugin: ModelPlugin) -> Path:
    st = plugin.status()
    if not st.available:
        raise RuntimeError(st.message)
    return _repo()  # type: ignore[return-value]


def _find_pretrained(repo: Path, variant: str) -> Path | None:
    folder = repo / "weights"
    if not folder.is_dir():
        return None
    prefix = f"rtdetrv2_{variant}_"
    for p in sorted(folder.glob(prefix + "*.pth")):
        if variant == "r50vd" and p.name[len(prefix):].startswith("m_"):
            continue  # that is the R50-M checkpoint
        return p.resolve()
    return None


def _read_metrics(log_txt: Path) -> dict:
    if not log_txt.exists():
        return {}
    best: dict | None = None
    epochs = 0
    for line in log_txt.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats = rec.get("test_coco_eval_bbox")
        if not stats:
            continue
        epochs += 1
        if best is None or stats[0] > best["test_coco_eval_bbox"][0]:
            best = rec
    if best is None:
        return {}
    s = best["test_coco_eval_bbox"]
    return {"mAP50-95": round(s[0], 4), "mAP50": round(s[1], 4), "best_epoch": best.get("epoch"), "epochs": epochs}
