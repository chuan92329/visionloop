"""Download the default model weights ahead of time, so the first click does not wait.

Run in the model environment (setup_windows.ps1 does this):

    python scripts/fetch_weights.py [--vram GB]

Downloads the size each menu selects by default; every other size is downloaded the first time it
is used. All of these are Apache-2.0. YOLOv9 / GELAN (GPL-3.0) are handled by setup_windows.ps1,
which asks first.

    DINOv2 Base       initial labelling (Linear Probe)        ~350 MB
    SAM 2.1 Large     click-to-segment, boxes to outlines     ~900 MB
    Grounding DINO    find objects from a text description    ~940 MB
    RF-DETR           COCO-pretrained starting point for training: the largest size that this
                      GPU can train (Nano 4 GB, Small 6 GB, Medium 8 GB, Large 12 GB of memory)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_MODELS = [
    ("DINOv2 Base（初步標註）", "facebook/dinov2-with-registers-base"),
    ("SAM 2.1 Large（點選分割、框轉多邊形）", "facebook/sam2.1-hiera-large"),
    ("Grounding DINO Base（文字找物件）", "IDEA-Research/grounding-dino-base"),
]
RFDETR_BY_VRAM = [(12, "large", "RFDETRLarge"), (8, "medium", "RFDETRMedium"), (6, "small", "RFDETRSmall"),
                  (0, "nano", "RFDETRNano")]


def gpu_memory_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def fetch_hf(label: str, repo: str) -> bool:
    from huggingface_hub import snapshot_download

    print(f"==> {label}：{repo}", flush=True)
    # transformers loads the safetensors copy; skip duplicate formats some repos also carry
    path = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    if not any(Path(path).glob("*.safetensors")):
        snapshot_download(repo)  # only an older format is published: take everything
    return True


def fetch_rfdetr(vram: float) -> bool:
    sys.path.insert(0, str(ROOT / "backend" / "app" / "plugins" / "builtin"))
    import rfdetr
    from rfdetr.assets.model_weights import download_pretrain_weights
    from rfdetr_helper import shared_pretrain

    _, variant, cls_name = next(v for v in RFDETR_BY_VRAM if vram >= v[0])
    target = shared_pretrain(getattr(rfdetr, cls_name), variant, {"pretrain_dir": str(ROOT / "models" / "rfdetr")})
    path = target.get("pretrain_weights")
    print(f"==> RF-DETR {variant.capitalize()}（訓練的起點，依 {vram:.0f} GB 顯存選擇）：{path}", flush=True)
    if path and not Path(path).exists():
        download_pretrain_weights(path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram", type=float, default=None, help="GPU memory in GB (default: ask nvidia-smi)")
    args = ap.parse_args()
    vram = gpu_memory_gb() if args.vram is None else args.vram

    failed = []
    for label, repo in HF_MODELS:
        try:
            fetch_hf(label, repo)
        except Exception as exc:  # noqa: BLE001 - keep going; it is downloaded on first use instead
            print(f"    下載失敗：{exc}（第一次使用時會再試一次）", flush=True)
            failed.append(label)
    try:
        fetch_rfdetr(vram)
    except Exception as exc:  # noqa: BLE001
        print(f"    下載失敗：{exc}（第一次訓練時會再試一次）", flush=True)
        failed.append("RF-DETR")
    print("模型權重已就緒" if not failed else f"未完成：{'、'.join(failed)}", flush=True)
    return 0  # a failed download is not fatal: the app downloads it on first use


if __name__ == "__main__":
    sys.exit(main())
