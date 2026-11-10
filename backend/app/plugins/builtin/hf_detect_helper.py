"""Zero-shot detection with Hugging Face transformers, run in the model environment.

Used by the ``grounding-dino`` plugin (text-conditioned): one forward pass per
class description, e.g. "a hex nut."; overlapping detections of different
classes are de-duplicated afterwards.

Input is a JSON spec (see the plugins); output is one JSON line per image:
``{"path": ..., "boxes": [[class_index, x, y, w, h, score], ...]}`` in pixels.
Progress lines look like ``[12/240]`` so the platform can show a progress bar.
"""

from __future__ import annotations

import argparse
import json
import sys

try:  # run as a script: the module next to it
    import hf_local
except ImportError:  # imported as part of the platform (tests)
    from . import hf_local


def pick_device(torch, requested: str) -> str:
    if requested in ("", "auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return f"cuda:{str(requested).split(',')[0]}"


def load_rgb(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


# --------------------------------------------------------- Grounding DINO
def run_grounding_dino(spec: dict, torch, device: str) -> None:
    from torchvision.ops import batched_nms, nms
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = hf_local.from_pretrained(AutoProcessor, spec["model_id"])
    model = hf_local.from_pretrained(AutoModelForZeroShotObjectDetection, spec["model_id"]).to(device).eval()

    prompts = [(i, p.strip()) for i, p in enumerate(spec["prompts"]) if p and p.strip()]
    if not prompts:
        raise SystemExit("no class descriptions were given")
    conf = float(spec["confidence"])
    text_threshold = float(spec.get("text_threshold", min(conf, 0.25)))
    images = spec["images"]
    with open(spec["out"], "w", encoding="utf-8") as out:
        for i, path in enumerate(images, 1):
            img = load_rgb(path)
            W, H = img.size
            all_boxes, all_scores, all_labels = [], [], []
            for cls, prompt in prompts:
                text = prompt.lower()
                if not text.endswith("."):
                    text += "."
                inputs = processor(images=img, text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                res = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=conf,
                    text_threshold=text_threshold,
                    target_sizes=[(H, W)],
                )[0]
                if len(res["scores"]):
                    all_boxes.append(res["boxes"])
                    all_scores.append(res["scores"])
                    all_labels.append(torch.full_like(res["scores"], cls, dtype=torch.long))
            rec = []
            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                labels = torch.cat(all_labels)
                k = batched_nms(boxes, scores, labels, 0.5)  # duplicates within a class
                boxes, scores, labels = boxes[k], scores[k], labels[k]
                k = nms(boxes, scores, 0.8)  # the same object matched by two descriptions
                boxes, scores, labels = boxes[k], scores[k], labels[k]
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
                rec = [
                    [int(c), round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2] - b[0]), 2),
                     round(float(b[3] - b[1]), 2), round(float(s), 4)]
                    for b, s, c in zip(boxes, scores, labels)
                    if float(b[2] - b[0]) >= 2 and float(b[3] - b[1]) >= 2
                ]
            out.write(json.dumps({"path": path, "boxes": rec}) + "\n")
            out.flush()
            print(f"[{i}/{len(images)}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    import torch

    device = pick_device(torch, spec.get("device", "auto"))
    print(f"torch {torch.__version__}, device {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.startswith("cuda") else ""), flush=True)
    print(f"loading {spec['model_id']} (first run downloads it from Hugging Face)", flush=True)
    if spec["mode"] == "grounding_dino":
        run_grounding_dino(spec, torch, device)
    else:
        sys.exit(f"unknown mode {spec['mode']}")


if __name__ == "__main__":
    main()
