"""RF-DETR training / inference, run in the model environment (``pip install "rfdetr[train]"``).

    python rfdetr_helper.py train   --spec train.json
    python rfdetr_helper.py predict --spec predict.json

Only the Apache-2.0 sizes (Nano, Small, Medium, Large) are offered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
}


def model_class(variant: str):
    import rfdetr

    return getattr(rfdetr, VARIANTS[variant])


# approximate download size (MB) of RF-DETR's COCO-pretrained weights
PRETRAIN_MB = {"nano": 120, "small": 130, "medium": 140, "large": 520,
               "seg-nano": 130, "seg-small": 140, "seg-medium": 150, "seg-large": 160}


def shared_pretrain(cls, variant: str, spec: dict) -> dict:
    """Point RF-DETR at one shared copy of its pretrained weights instead of the job folder
    (it would download them again for every job), and say so before a download starts."""
    folder = spec.get("pretrain_dir")
    if not folder:
        return {}
    # the config class's default file name; RFDETRLarge picks its config in get_model_config()
    import rfdetr.config as cfg

    config = getattr(cfg, cls.__name__ + "Config", None) or cls._model_config_class
    field = config.model_fields.get("pretrain_weights")
    name = Path(field.default or "").name if field else ""
    if not name:
        return {}
    path = Path(folder) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"@@download {name} {PRETRAIN_MB.get(variant, 150)}", flush=True)
    return {"pretrain_weights": str(path)}


def torch_device(requested: str) -> str | None:
    if requested in ("", "auto", None):
        return None  # let RF-DETR / Lightning decide
    if requested == "cpu":
        return "cpu"
    return f"cuda:{str(requested).split(',')[0]}"


# ------------------------------------------------------------------ train
def train(spec: dict) -> None:
    cls = model_class(spec["variant"])
    segmentation = spec["variant"].startswith("seg-")
    kwargs = {}
    if spec.get("device") == "cpu":
        kwargs["device"] = "cpu"
    if spec.get("init_weights"):
        kwargs["pretrain_weights"] = spec["init_weights"]
        print(f"fine-tuning from {spec['init_weights']}", flush=True)
    elif spec.get("no_pretrain"):
        kwargs["pretrain_weights"] = None  # offline / from scratch (much worse with little data)
    else:
        kwargs.update(shared_pretrain(cls, spec["variant"], spec))
    model = cls(**kwargs)

    train_kwargs = dict(
        dataset_dir=spec["dataset_dir"],
        epochs=int(spec["epochs"]),
        batch_size=int(spec["batch_size"]),
        grad_accum_steps=int(spec.get("grad_accum_steps", 1)),
        output_dir=spec["output_dir"],
        num_workers=int(spec.get("num_workers", 2)),
        progress_bar="tqdm",
        tensorboard=False,
    )
    if segmentation:
        # upsampling every mask to full image size made validating 12 MP photos take minutes per epoch
        train_kwargs["eval_masks_head_resolution"] = True
    if spec.get("lr"):
        train_kwargs["lr"] = float(spec["lr"])
    device = torch_device(spec.get("device", "auto"))
    if device:
        train_kwargs["device"] = device
    res = spec.get("resolution")
    if res:
        block = model.model_config.patch_size * model.model_config.num_windows
        if int(res) % block == 0:
            train_kwargs["resolution"] = int(res)
        else:
            print(f"note: resolution {res} is not a multiple of {block}; using the default "
                  f"{model.model_config.resolution}", flush=True)
    print(f"training {VARIANTS[spec['variant']]} at {train_kwargs.get('resolution', model.model_config.resolution)}px",
          flush=True)
    model.train(**train_kwargs)

    out = Path(spec["output_dir"])
    names = ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]
    if segmentation:
        # mask mAP easily saturates at 1.0 from the first epoch on small datasets, which freezes the
        # "best" checkpoint at a barely trained model (real case: scores 0.08 vs 0.92 for the final one)
        names.insert(0, "last_ema.pth")
    for name in names:
        if (out / name).exists():
            best = out / name
            break
    else:
        sys.exit("training finished but no checkpoint was written")
    result = {"weights": str(best), "metrics": read_metrics(out / "metrics.csv", final=best.name == "last_ema.pth")}
    (out / "visionloop_result.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"best checkpoint: {best}", flush=True)


def read_metrics(path: Path, final: bool = False) -> dict:
    """Pick mAP columns from Lightning's metrics.csv without depending on exact names.

    Reports the best epoch, or with ``final`` the last one (when those are the weights being kept).
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    cols = rows[0].keys() if rows else []
    # segmentation models also log mask scores: those are what they are selected on
    segm = any("segm" in c.lower() for c in cols)

    def find(pred):
        for c in cols:
            lc = c.lower()
            if "val" in lc and "map" in lc and "class" not in lc and ("segm" in lc) == segm and pred(lc):
                return c
        return None

    c5095 = find(lambda c: "50_95" in c or "50:95" in c or c.endswith("/map") or c.endswith("_map"))
    c50 = find(lambda c: ("map_50" in c or "map50" in c) and "95" not in c)
    best: dict = {}
    if c5095:
        vals = [(float(r[c5095]), r) for r in rows if r.get(c5095) not in (None, "")]
        if vals:
            score, row = vals[-1] if final else max(vals, key=lambda t: t[0])
            best["mAP50-95"] = round(score, 4)
            if c50 and row.get(c50):
                best["mAP50"] = round(float(row[c50]), 4)
            if row.get("epoch") not in (None, ""):
                best["best_epoch"] = int(float(row["epoch"]))
    return best


# ---------------------------------------------------------------- predict
def predict(spec: dict) -> None:
    import rfdetr
    from PIL import Image

    kwargs = {}
    if spec.get("device") == "cpu":
        kwargs["device"] = "cpu"
    if spec.get("weights"):
        model = rfdetr.from_checkpoint(spec["weights"], **kwargs)
    else:
        variant = spec.get("variant") or "medium"
        cls = model_class(variant)
        model = cls(**kwargs, **shared_pretrain(cls, variant, spec))  # COCO-pretrained
        print("no trained weights given: using the COCO-pretrained model", flush=True)
    print(f"device {model.model_config.device}", flush=True)
    images = spec["images"]
    with open(spec["out"], "w", encoding="utf-8") as out:
        for i, path in enumerate(images, 1):
            with Image.open(path) as im:
                img = im.convert("RGB")
            det = model.predict(img, threshold=float(spec["confidence"]), include_source_image=False)
            names = det.data.get("class_name") if getattr(det, "data", None) is not None else None
            masks = getattr(det, "mask", None)
            rec = []
            for k in range(len(det)):
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
                name = str(names[k]) if names is not None and len(names) > k else ""
                rec.append([int(det.class_id[k]), name, round(x1, 2), round(y1, 2), round(x2 - x1, 2),
                            round(y2 - y1, 2), round(float(det.confidence[k]), 4)])
                if masks is not None:
                    from mask_polygon import mask_to_polygon

                    rec[-1].append(mask_to_polygon(masks[k]))
            out.write(json.dumps({"path": path, "boxes": rec}) + "\n")
            out.flush()
            print(f"[{i}/{len(images)}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("device") not in (None, "", "auto", "cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(spec["device"]).split(",")[0])
        spec["device"] = "0"  # after masking, the chosen GPU is cuda:0
    (train if args.command == "train" else predict)(spec)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    main()
