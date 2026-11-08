"""Check the model environment: PyTorch + CUDA, RF-DETR, transformers, YOLOv9.

    <vl-models python> scripts/check_models_env.py [--yolov9 PATH]

Prints one line per check and exits non-zero if something essential is missing.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolov9", default=str(Path(__file__).resolve().parents[1] / "models" / "yolov9"))
    args = ap.parse_args()
    failed = False

    print(f"{OK} Python {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch
    except ImportError:
        print(f"{FAIL} PyTorch 沒有安裝")
        return 1
    print(f"{OK} PyTorch {torch.__version__}，CUDA build {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{OK} GPU：{props.name}，{props.total_memory / 1024**3:.1f} GB，compute capability {props.major}.{props.minor}")
        try:
            x = torch.randn(256, 256, device="cuda")
            (x @ x).sum().item()
            print(f"{OK} GPU 運算測試通過")
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} GPU 運算失敗：{exc}")
            failed = True
    else:
        print(f"{FAIL} PyTorch 看不到 GPU：可能裝到 CPU 版，或驅動太舊（需要支援 CUDA {torch.version.cuda}）")
        failed = True

    if not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "autocast"):
        print(f"{WARN} torch.cuda.amp.autocast 已移除，YOLOv9 原始碼可能無法訓練")

    for mod, label in (("torchvision", "torchvision"), ("transformers", "transformers（Grounding DINO、SAM 2）"),
                       ("rfdetr", "RF-DETR"), ("pytorch_lightning", "RF-DETR 訓練套件"),
                       ("scipy", "scipy（DINOv2 Probe）")):
        try:
            m = importlib.import_module(mod)
            print(f"{OK} {label} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} {label}：{exc}")
            failed = True

    try:
        import pkg_resources  # noqa: F401  (YOLOv9 imports it)
    except ImportError:
        print(f"{FAIL} 缺少 pkg_resources（YOLOv9 需要）：請執行 pip install \"setuptools<81\"")
        failed = True

    repo = Path(args.yolov9)
    if (repo / "train_dual.py").exists():
        sys.path.insert(0, str(repo))
        cwd = os.getcwd()
        try:
            os.chdir(repo)
            importlib.import_module("models.yolo")
            print(f"{OK} YOLOv9 原始碼可以載入（{repo}）")
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} YOLOv9 載入失敗：{type(exc).__name__}: {exc}")
            failed = True
        finally:
            os.chdir(cwd)
        weights = sorted(p.name for p in (repo / "weights").glob("*.pt")) if (repo / "weights").is_dir() else []
        print(f"{OK if weights else WARN} YOLOv9 預訓練權重：{', '.join(weights) or '沒有（會從頭訓練）'}")
    else:
        print(f"{WARN} 找不到 YOLOv9（{repo}），YOLOv9 外掛將無法使用")

    print("全部通過" if not failed else "有項目失敗，請依照上面標示的項目檢查安裝，或附上這段輸出到 GitHub Issues 回報")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
