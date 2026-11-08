"""RF-DETR, SAM 2 and Grounding DINO.

All three run in the shared model environment (``VISIONLOOP_MODELS_PYTHON``,
created by ``scripts/setup_windows.ps1``) through small helper scripts, so the
platform itself never imports PyTorch.

* RF-DETR (Roboflow, Apache-2.0 for Nano/Small/Medium/Large): trainable
  detector with a DINOv2 backbone; works with few images thanks to strong
  pre-training.  Without trained weights it uses the COCO-pretrained model.
* SAM 2 (Meta, Apache-2.0): turns existing boxes into outlines.
* Grounding DINO (IDEA Research, Apache-2.0): zero-shot, uses a text
  description per class.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
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
    check_python_module,
    python_executable,
    run_process,
)

HERE = Path(__file__).parent
PROGRESS_RX = re.compile(r"^\[(\d+)/(\d+)\]")
# RF-DETR's COCO-pretrained weights, downloaded once and shared by every job (models/ is git-ignored)
RFDETR_WEIGHTS_DIR = HERE.parents[3] / "models" / "rfdetr"
DOWNLOAD_RX = re.compile(r"^@@download (\S+) (\d+)$")  # printed by a helper before it downloads

# approximate download size (MB) of the Hugging Face models, for the "first use" notice
HF_SIZES_MB = {
    "facebook/dinov2-with-registers-small": 90,
    "facebook/dinov2-with-registers-base": 350,
    "facebook/dinov2-with-registers-large": 1200,
    "facebook/sam2.1-hiera-tiny": 160,
    "facebook/sam2.1-hiera-small": 190,
    "facebook/sam2.1-hiera-base-plus": 330,
    "facebook/sam2.1-hiera-large": 900,
    "IDEA-Research/grounding-dino-tiny": 690,
    "IDEA-Research/grounding-dino-base": 940,
}


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cached(model_id: str) -> bool:
    """Has this Hugging Face model been downloaded already (by setup or an earlier run)?"""
    snapshots = hf_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(f.is_file() for f in snapshots.rglob("*"))


def download_notice(model_id: str) -> str | None:
    """Job message for a model that has to be downloaded first, so a long wait is explained."""
    if hf_cached(model_id):
        return None
    size = HF_SIZES_MB.get(model_id)
    return f"第一次使用，正在下載 {model_id.split('/')[-1]}" + (f"（約 {size} MB）" if size else "") + "…"


def announce_download(hooks, model_id: str) -> None:
    msg = download_notice(model_id)
    if msg:
        hooks.progress(0.0, msg)


def with_download_notice(hooks, on_line=None):
    """Wraps a helper's line handler: "@@download NAME MB" becomes the job message."""

    def handle(line: str) -> None:
        m = DOWNLOAD_RX.match(line)
        if m:
            hooks.progress(0.0, f"第一次使用，正在下載 {m.group(1)}（約 {m.group(2)} MB）…")
        elif on_line:
            on_line(line)

    return handle


def models_python() -> str:
    return python_executable("VISIONLOOP_MODELS_PYTHON")


def env_status(module: str) -> PluginStatus:
    py = os.environ.get("VISIONLOOP_MODELS_PYTHON")
    if not py:
        return PluginStatus(False, "尚未設定模型環境：請執行 scripts\\setup_windows.ps1，或設定 VISIONLOOP_MODELS_PYTHON")
    ok, msg = check_python_module(py, module)
    return PluginStatus(ok, msg if not ok else f"模型環境：{py}")


def _progress_from_helper(ctx_hooks, total: int, start: float = 0.0, span: float = 1.0):
    def on_line(line: str) -> None:
        m = PROGRESS_RX.match(line)
        if m:
            done = int(m.group(1))
            ctx_hooks.progress(start + span * done / max(total, 1) * 0.99, f"推論 {done}/{total}")

    return on_line


# what Hugging Face, requests and urllib say when a model cannot be downloaded
DOWNLOAD_FAILURE_RX = re.compile(
    r"couldn't connect to|LocalEntryNotFoundError|OfflineModeIsEnabled|ConnectionError|ConnectTimeout|"
    r"ReadTimeout|Max retries exceeded|Failed to establish a new connection|getaddrinfo failed|"
    r"NameResolutionError|SSLError|urlopen error|IncompleteRead|ChunkedEncodingError",
    re.IGNORECASE,
)
DOWNLOAD_FAILED = ("模型下載失敗：連不上下載網站。請確認網路連線（公司網路可能需要設定 Proxy）後再試一次；"
                   "只有第一次使用需要下載，詳細訊息見任務紀錄。")


def explain_failure(text: str) -> str | None:
    """A message a person can act on, for the failures we recognise (so far: no download)."""
    return DOWNLOAD_FAILED if DOWNLOAD_FAILURE_RX.search(text) else None


def _run_helper(script: str, args: list[str], spec: dict, work_dir: Path, hooks, on_line=None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / f"{Path(script).stem}_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    tail: deque[str] = deque(maxlen=200)  # the helper's last words, to explain a failure

    def keep(line: str) -> None:
        tail.append(line)
        if on_line:
            on_line(line)

    code = run_process(
        [models_python(), str(HERE / script), *args, "--spec", str(spec_path)],
        cwd=work_dir,
        hooks=hooks,
        on_line=keep,
    )
    if code != 0:
        raise RuntimeError(explain_failure("\n".join(tail)) or f"{script} 執行失敗（exit code {code}），請查看任務紀錄")


def _read_jsonl(path: Path) -> dict[str, list]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["path"]] = rec["boxes"]
    return out


# ------------------------------------------------------------------ RF-DETR
class RFDETRPlugin(ModelPlugin):
    id = "rfdetr"
    name = "RF-DETR"
    family = "RF-DETR"
    description = "Roboflow 的即時偵測模型，DINOv2 骨幹、預訓練強，少量資料也能訓練。只提供 Apache-2.0 授權的尺寸。"
    license = "Apache-2.0（Nano / Small / Medium / Large）"
    homepage = "https://github.com/roboflow/rf-detr"
    dataset_format = "yolo"
    needs_weights = False
    no_weights_label = "COCO 預訓練（類別名稱需對應 COCO 80 類）"
    variants = [
        Variant("nano", "RF-DETR Nano", "預設 384px，最快", 4, image_size=384),
        Variant("small", "RF-DETR Small", "預設 512px", 6, image_size=512),
        Variant("medium", "RF-DETR Medium", "預設 576px，速度與準確度平衡", 8, image_size=576),
        Variant("large", "RF-DETR Large", "預設 704px，最準", 12, image_size=704),
    ]

    def status(self) -> PluginStatus:
        return env_status("rfdetr")

    def train(self, ctx: TrainContext) -> TrainResult:
        hooks = ctx.hooks
        epoch_rx = re.compile(r"Epoch (\d+)")
        # RF-DETR is tuned for an effective batch of 16: keep it via gradient accumulation
        accum = max(1, round(16 / max(1, ctx.batch_size)))
        spec = {
            "dataset_dir": str(ctx.dataset_dir.resolve()),
            "variant": ctx.variant,
            "epochs": ctx.epochs,
            "batch_size": ctx.batch_size,
            "grad_accum_steps": int(ctx.extra.get("grad_accum_steps", accum)),
            "output_dir": str(ctx.output_dir.resolve()),
            "device": ctx.device,
            "num_workers": int(ctx.extra.get("workers", 2)),
            "init_weights": str(ctx.init_weights) if ctx.init_weights else None,
            "resolution": ctx.extra.get("resolution") or ctx.image_size,
            "pretrain_dir": str(RFDETR_WEIGHTS_DIR),
        }
        if ctx.extra.get("lr"):
            spec["lr"] = ctx.extra["lr"]

        def on_line(line: str) -> None:
            m = epoch_rx.search(line)
            if m:
                e = int(m.group(1))
                hooks.progress(min(e / ctx.epochs, 0.99), f"epoch {e + 1}/{ctx.epochs}")

        hooks.progress(0.0, f"epoch 1/{ctx.epochs}")
        _run_helper("rfdetr_helper.py", ["train"], spec, ctx.output_dir, hooks, with_download_notice(hooks, on_line))
        result = json.loads((ctx.output_dir / "visionloop_result.json").read_text(encoding="utf-8"))
        return TrainResult(Path(result["weights"]), result.get("metrics", {}))

    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        out = ctx.work_dir / "rfdetr_predictions.jsonl"
        spec = {
            "weights": str(ctx.weights_path) if ctx.weights_path else None,
            "variant": ctx.variant or "medium",
            "confidence": ctx.confidence,
            "device": ctx.device,
            "images": [str(p) for p in images],
            "out": str(out),
            "pretrain_dir": str(RFDETR_WEIGHTS_DIR),
        }
        _run_helper("rfdetr_helper.py", ["predict"], spec, ctx.work_dir, ctx.hooks,
                    with_download_notice(ctx.hooks, _progress_from_helper(ctx.hooks, len(images))))
        results = _read_jsonl(out)
        for p in images:
            preds = []
            for cls, name, x, y, w, h, s, *polygon in results.get(str(p), []):
                # RF-DETR reports the label name (from our data.yaml, or COCO for the pretrained
                # model); mapping by name is robust to class re-ordering
                preds.append(Prediction(int(cls), x, y, w, h, s, class_name=name or None,
                                        polygon=polygon[0] if polygon else None))
            yield p, preds


class RFDETRSegPlugin(RFDETRPlugin):
    id = "rfdetr-seg"
    name = "RF-DETR Seg（分割）"
    task = "segment"
    description = (
        "RF-DETR 的實例分割版本，用多邊形標註訓練，自動標註時會產生多邊形。只有框的物件會被當成矩形，建議先把框轉成多邊形。"
        "訓練時的 mask 分數在低解析度計算、很容易滿分，請以實際自動標註的結果判斷品質。"
    )
    dataset_format = "yolo-seg"
    needs_weights = True
    variants = [
        Variant("seg-nano", "RF-DETR Seg Nano", "預設 312px，最快", 4, image_size=312, image_step=12),
        Variant("seg-small", "RF-DETR Seg Small", "預設 384px", 6, image_size=384, image_step=24),
        Variant("seg-medium", "RF-DETR Seg Medium", "預設 432px，速度與準確度平衡", 8, image_size=432, image_step=24),
        Variant("seg-large", "RF-DETR Seg Large", "預設 504px，最準", 12, image_size=504, image_step=24),
    ]


# -------------------------------------------------------------------- SAM 2
class SAM2Plugin(ModelPlugin):
    id = "sam2"
    name = "SAM 2（把框轉成多邊形）"
    description = "用既有的框當提示，由 SAM 2 描出物件輪廓。轉換失敗的框會保留原狀。"
    license = "Apache-2.0"
    homepage = "https://github.com/facebookresearch/sam2"
    needs_weights = False
    trainable = False
    prompt_mode = "boxes"
    variants = [
        # largest first = the default: it only runs on boxes, so its speed matters little
        Variant("facebook/sam2.1-hiera-large", "SAM 2.1 Large", "最準（建議）", 4),
        Variant("facebook/sam2.1-hiera-base-plus", "SAM 2.1 Base+", "", 3),
        Variant("facebook/sam2.1-hiera-small", "SAM 2.1 Small", "速度與準確度平衡", 2),
        Variant("facebook/sam2.1-hiera-tiny", "SAM 2.1 Tiny", "最快", 2),
    ]

    def status(self) -> PluginStatus:
        return env_status("transformers")

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - not trainable
        raise RuntimeError("SAM 2 不需要訓練")

    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        out = ctx.work_dir / "sam_predictions.jsonl"
        model = ctx.variant or self.variants[0].id
        announce_download(ctx.hooks, model)
        spec = {
            "model": model,
            "device": ctx.device,
            "detail": ctx.options.get("polygon_detail", "normal"),
            "images": {str(p): ctx.box_prompts[str(p)] for p in images},
            "out": str(out),
        }
        _run_helper("sam_helper.py", [], spec, ctx.work_dir, ctx.hooks, _progress_from_helper(ctx.hooks, len(images)))
        results = _read_jsonl(out)
        for p in images:
            preds = []
            for (x, y, w, h), (score, polygon) in zip(ctx.box_prompts[str(p)], results.get(str(p), [])):
                preds.append(Prediction(0, x, y, w, h, score, polygon=polygon))
            yield p, preds


# ----------------------------------------------------------- Grounding DINO
class GroundingDINOPlugin(ModelPlugin):
    id = "grounding-dino"
    name = "Grounding DINO 文字找物（零樣本）"
    description = "用英文描述每個類別（例如 a hex nut），不需要任何標註就能自動找出物件。"
    license = "Apache-2.0"
    homepage = "https://huggingface.co/IDEA-Research/grounding-dino-base"
    needs_weights = False
    no_weights_label = "不需要權重（使用類別的文字描述）"
    trainable = False
    prompt_mode = "text"
    default_confidence = 0.35
    variants = [
        Variant("base", "Grounding DINO Base", "IDEA-Research/grounding-dino-base，較準（建議）", 5),
        Variant("tiny", "Grounding DINO Tiny", "IDEA-Research/grounding-dino-tiny，較快", 3),
    ]
    model_ids = {"tiny": "IDEA-Research/grounding-dino-tiny", "base": "IDEA-Research/grounding-dino-base"}

    def status(self) -> PluginStatus:
        return env_status("transformers")

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - not trainable
        raise RuntimeError("Grounding DINO 不需要訓練，請直接在「自動標註」使用")

    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        if not any(p.strip() for p in ctx.prompts):
            raise RuntimeError("請為至少一個類別填寫英文描述")
        out = ctx.work_dir / "gdino_predictions.jsonl"
        model_id = self.model_ids.get(ctx.variant or "base", self.model_ids["base"])
        announce_download(ctx.hooks, model_id)
        spec = {
            "mode": "grounding_dino",
            "model_id": model_id,
            "confidence": ctx.confidence,
            "device": ctx.device,
            "prompts": ctx.prompts,
            "images": [str(p) for p in images],
            "out": str(out),
        }
        _run_helper("hf_detect_helper.py", [], spec, ctx.work_dir, ctx.hooks,
                    _progress_from_helper(ctx.hooks, len(images)))
        results = _read_jsonl(out)
        for p in images:
            yield p, [Prediction(int(c), x, y, w, h, s) for c, x, y, w, h, s in results.get(str(p), [])]


# ------------------------------------------------------------- patch probe
class ProbePlugin(ModelPlugin):
    """Frozen backbone + linear probe on patch features, retrained on every run from all
    confirmed images. See probe_helper.py for the method."""

    id = "probe"
    name = "DINOv2 Probe（凍結骨幹 + 線性探針）"
    description = (
        "凍結 DINOv2，只用你已確認的圖片訓練一層線性探針（幾秒鐘），再把每個類別的熱力圖轉成框。"
        "每次執行都會用最新的確認結果重新訓練；有多邊形的物件會用輪廓訓練，輸出仍然是框；"
        "沒有物件的已確認圖片會當作負樣本。"
    )
    license = "Apache-2.0（DINOv2）"
    homepage = "https://huggingface.co/facebook/dinov2-with-registers-base"
    needs_weights = False
    no_weights_label = "不需要權重（每次用已確認的圖片重新訓練探針）"
    trainable = False
    prompt_mode = "exemplars"
    uses_reviewed_images = True
    default_confidence = 0.5
    # id -> (backbone kind, Hugging Face model, long side in pixels)
    BACKBONES = {
        "dinov2-s": ("dinov2", "facebook/dinov2-with-registers-small", 896),
        "dinov2-b": ("dinov2", "facebook/dinov2-with-registers-base", 896),
        "dinov2-l": ("dinov2", "facebook/dinov2-with-registers-large", 896),
        "dinov2-b-hr": ("dinov2", "facebook/dinov2-with-registers-base", 1344),
    }
    variants = [
        # first = default: on a real project (12 MP photos) the higher resolution fitted boxes more
        # tightly than Base (mean IoU 0.960 vs 0.941) and than Large (0.944), at the same recall
        Variant("dinov2-b-hr", "DINOv2 Base 高解析度（1344px）", "建議預設：框最貼合物件，約 2 GB 顯存", 2),
        Variant("dinov2-b", "DINOv2 Base", "896px，較快", 2),
        Variant("dinov2-l", "DINOv2 Large", "較慢，多數資料與 Base 差不多", 4),
        Variant("dinov2-s", "DINOv2 Small", "最快，特徵較弱", 1),
    ]
    # NVIDIA C-RADIOv4 (NVIDIA Open Model License, runs the model's own code from Hugging Face).
    # Off unless "experimental_backbones": true is in local_settings.json.
    if os.environ.get("VISIONLOOP_EXPERIMENTAL_BACKBONES", "").lower() in {"1", "true"}:
        BACKBONES = {
            **BACKBONES,
            "cradio-v4-so400m": ("radio", "nvidia/C-RADIOv4-SO400M", 1024),
            "cradio-v4-h": ("radio", "nvidia/C-RADIOv4-H", 1024),
        }
        variants = [
            *variants,
            Variant("cradio-v4-so400m", "C-RADIOv4 SO400M（實驗性）",
                    "NVIDIA Open Model License；首次使用會執行 Hugging Face 上的模型程式碼", 4),
            Variant("cradio-v4-h", "C-RADIOv4 H（實驗性）", "同上，較大", 6),
        ]

    def status(self) -> PluginStatus:
        return env_status("transformers")

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - not trainable
        raise RuntimeError("Probe 會在每次自動標註時自動重新訓練，不需要另外訓練")

    def predict(self, ctx: PredictContext, images: list[Path]) -> Iterator[tuple[Path, list[Prediction]]]:
        if not any(r.boxes for r in ctx.reviewed):
            raise RuntimeError("沒有範例：請先在幾張圖片上標出物件並按「確認」")
        kind, model_id, long_side = self.BACKBONES.get(ctx.variant or "dinov2-b-hr", self.BACKBONES["dinov2-b-hr"])
        announce_download(ctx.hooks, model_id)
        out = ctx.work_dir / "probe_predictions.jsonl"
        spec = {
            "backbone": kind,
            "model_id": model_id,
            "long_side": long_side,
            "class_names": ctx.class_names,
            "confidence": ctx.confidence,
            "device": ctx.device,
            "head": ctx.options.get("probe_head", "linear"),
            "cache_dir": str(ctx.cache_dir) if ctx.cache_dir else None,
            "training": [
                {"image": str(r.image), "width": r.width, "height": r.height, "boxes": [list(b) for b in r.boxes],
                 "polygons": r.polygons}
                for r in ctx.reviewed
            ],
            "images": [str(p) for p in images],
            "out": str(out),
        }
        _run_helper("probe_helper.py", [], spec, ctx.work_dir, ctx.hooks, _progress_from_helper(ctx.hooks, len(images)))
        results = _read_jsonl(out)
        for p in images:
            yield p, [Prediction(int(c), x, y, w, h, s) for c, x, y, w, h, s in results.get(str(p), [])]
