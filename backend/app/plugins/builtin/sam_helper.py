"""SAM 2 box -> outline, run in the model environment (``transformers``).

    python sam_helper.py --spec spec.json

spec: {"model": ..., "device": ..., "detail": "normal" or "fine", "images": {path: [[x, y, w, h], ...]}, "out": ...}
Writes one JSON line per image: {"path": ..., "boxes": [[score, polygon or null], ...]},
in the order of the input boxes.
"""

from __future__ import annotations

import argparse
import json

try:  # run as a script: the module next to it
    import hf_local
except ImportError:  # imported as part of the platform (tests)
    from . import hf_local

BATCH = 32  # boxes per forward pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    import torch
    from mask_polygon import mask_to_polygon
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    requested = spec.get("device", "auto")
    if requested in ("", "auto", None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu" if requested == "cpu" else f"cuda:{str(requested).split(',')[0]}"
    print(f"torch {torch.__version__}, device {device}", flush=True)
    print(f"loading {spec['model']} (first run downloads it from Hugging Face)", flush=True)
    processor = hf_local.from_pretrained(Sam2Processor, spec["model"])
    model = hf_local.from_pretrained(Sam2Model, spec["model"]).to(device).eval()

    images = spec["images"]
    with open(spec["out"], "w", encoding="utf-8") as out:
        for i, (path, boxes) in enumerate(images.items(), 1):
            with Image.open(path) as im:
                image = im.convert("RGB")
            rec = []
            for start in range(0, len(boxes), BATCH):
                xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes[start:start + BATCH]]
                inputs = processor(images=image, input_boxes=[xyxy], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
                scores = outputs.iou_scores[0].cpu().numpy().reshape(len(xyxy), -1)
                for k in range(len(xyxy)):
                    mask = masks[k].numpy().reshape(-1, image.height, image.width)[0]
                    polygon = mask_to_polygon(mask, spec.get("detail", "normal"))
                    rec.append([round(float(scores[k][0]), 4), polygon])
            out.write(json.dumps({"path": path, "boxes": rec}) + "\n")
            out.flush()
            print(f"[{i}/{len(images)}]", flush=True)


if __name__ == "__main__":
    main()
