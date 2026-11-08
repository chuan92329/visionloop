"""Batch inference for RT-DETRv2, run in the RT-DETR environment.

Executed by the rtdetrv2 plugin as a separate process with the official
repository on ``sys.path``; it follows the repo's own
``references/deploy/rtdetrv2_torch.py`` and writes one JSON line per image:
``{"path": ..., "boxes": [[class, x1, y1, x2, y2, score], ...]}``.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from src.core import YAMLConfig  # type: ignore[import-not-found]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = YAMLConfig(args.config, num_classes=args.num_classes, remap_mscoco_category=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = Model().to(device).eval()
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])

    paths = [p for p in open(args.list, encoding="utf-8").read().splitlines() if p]
    with open(args.out, "w", encoding="utf-8") as out, torch.no_grad():
        for i, path in enumerate(paths, 1):
            im = Image.open(path).convert("RGB")
            w, h = im.size
            labels, boxes, scores = model(
                tf(im)[None].to(device), torch.tensor([[w, h]], device=device)
            )
            keep = scores[0] >= args.conf
            rec = [
                [int(c), *[round(float(v), 2) for v in b], round(float(s), 4)]
                for c, b, s in zip(labels[0][keep], boxes[0][keep], scores[0][keep])
            ]
            out.write(json.dumps({"path": path, "boxes": rec}) + "\n")
            if i % 10 == 0 or i == len(paths):
                print(f"[{i}/{len(paths)}]", flush=True)


if __name__ == "__main__":
    main()
