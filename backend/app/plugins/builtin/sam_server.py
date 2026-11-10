"""Interactive SAM 2 for the annotator, run in the model environment (``transformers``).

The platform starts this once and keeps it running: the model stays loaded and the
embedding of the image being annotated is kept, so a click only runs the small
mask decoder.  One JSON request per stdin line, one reply per stdout line that
starts with ``@@`` (other output is logging).  It exits when stdin closes, which
also happens when the platform process dies.

Requests::

    {"op": "prepare", "model": ..., "device": ..., "image": path, "key": "..."}
    {"op": "segment", ... same ..., "points": [[x, y, 1 or 0], ...], "box": [x, y, w, h] or null,
     "detail": "normal" or "fine", "exclude": [[[x, y], ...], ...]}

``exclude``: outlines of objects already annotated.  Their area is removed from the mask
before it becomes a polygon, so an object partly covered by another one gets only its
visible part.

Replies: {"ok": true, "polygon": [[x, y], ...] or null, "score": float} or {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import sys

try:  # run as a script: the module next to it
    import hf_local
except ImportError:  # imported as part of the platform (tests)
    from . import hf_local

MARK = "@@"


def reply(obj: dict) -> None:
    sys.stdout.write(MARK + json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    import cv2
    import numpy as np
    import torch
    from mask_polygon import mask_to_polygon
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    loaded: dict = {"model_id": None, "device": None, "model": None, "processor": None}
    image: dict = {"key": None, "embeddings": None, "size": None}

    def ensure_model(model_id: str, requested: str) -> None:
        if requested in ("", "auto", None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu" if requested == "cpu" else f"cuda:{str(requested).split(',')[0]}"
        if loaded["model_id"] == model_id and loaded["device"] == device:
            return
        print(f"loading {model_id} on {device}", flush=True)
        loaded.update(model=None, processor=None)
        image.update(key=None, embeddings=None)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        loaded["processor"] = hf_local.from_pretrained(Sam2Processor, model_id)
        loaded["model"] = hf_local.from_pretrained(Sam2Model, model_id).to(device).eval()
        loaded.update(model_id=model_id, device=device)

    def ensure_image(path: str, key: str) -> None:
        if image["key"] == key:
            return
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        inputs = loaded["processor"](images=rgb, return_tensors="pt").to(loaded["device"])
        with torch.no_grad():
            image["embeddings"] = loaded["model"].get_image_embeddings(inputs["pixel_values"])
        image.update(key=key, size=[rgb.height, rgb.width])

    def segment(req: dict) -> dict:
        processor, model = loaded["processor"], loaded["model"]
        points = req.get("points") or []
        box = req.get("box")
        kwargs = {}
        if points:
            kwargs["input_points"] = [[[[float(x), float(y)] for x, y, _ in points]]]
            kwargs["input_labels"] = [[[int(label) for _, _, label in points]]]
        if box:
            x, y, w, h = box
            kwargs["input_boxes"] = [[[float(x), float(y), float(x + w), float(y + h)]]]
        enc = processor(original_sizes=[image["size"]], return_tensors="pt", **kwargs).to(loaded["device"])
        prompts = {k: enc[k] for k in ("input_points", "input_labels", "input_boxes") if k in enc}
        # a single click is ambiguous (part or whole?): ask for three masks and keep the best one
        multi = len(points) == 1 and not box
        with torch.no_grad():
            out = model(image_embeddings=image["embeddings"], multimask_output=multi, **prompts)
        masks = processor.post_process_masks(out.pred_masks.cpu(), [image["size"]])[0]
        scores = out.iou_scores.cpu().reshape(-1)
        best = int(scores.argmax()) if multi else 0
        mask = masks.reshape(-1, *image["size"])[best].numpy()
        removed = 0.0
        if req.get("exclude"):
            taken = np.zeros(mask.shape, np.uint8)
            outlines = [np.round(np.asarray(p, np.float64)).astype(np.int32).reshape(-1, 1, 2)
                        for p in req["exclude"] if len(p) >= 3]
            cv2.fillPoly(taken, outlines, 1)
            before = int(mask.sum())
            mask = mask & (taken == 0)
            removed = 1 - int(mask.sum()) / before if before else 0.0
        # only the largest region is kept, so slivers left along a removed outline disappear
        polygon = mask_to_polygon(mask, req.get("detail", "normal"))
        return {"ok": True, "polygon": polygon, "score": round(float(scores[best]), 4), "removed": round(removed, 3)}

    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            ensure_model(req["model"], req.get("device", "auto"))
            ensure_image(req["image"], req["key"])
            reply(segment(req) if req.get("op") == "segment" else {"ok": True})
        except Exception as exc:  # noqa: BLE001 - every request gets an answer
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
