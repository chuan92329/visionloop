"""Tests for the YOLOv9 / RT-DETR adapters against *stub* repositories.

The stubs mimic the command-line interface and output files of the official
repositories (WongKinYiu/yolov9, lyuwenyu/RT-DETR) so the wiring – command
construction, progress parsing, output/metrics parsing and coordinate
conversion – is tested without PyTorch or a GPU.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from app.plugins.base import JobHooks, PredictContext, TrainContext
from app.plugins.builtin import rtdetr_official, yolov9_official


class Recorder:
    def __init__(self):
        self.lines: list[str] = []
        self.progress: list[float] = []

    def hooks(self) -> JobHooks:
        return JobHooks(log=self.lines.append, progress=lambda f, m: self.progress.append(f), should_stop=lambda: False)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture()
def images(tmp_path):
    out = []
    for i in range(2):
        p = tmp_path / "imgs" / f"img{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (200, 100)).save(p)
        out.append(p)
    return out


# ------------------------------------------------------------------ YOLOv9
@pytest.fixture()
def yolo_repo(tmp_path, monkeypatch):
    repo = tmp_path / "yolov9"
    write(
        repo / "train_dual.py",
        """
        import argparse, pathlib, sys, yaml
        ap = argparse.ArgumentParser()
        for a in ["--workers","--device","--batch","--data","--img","--cfg","--weights","--hyp",
                  "--min-items","--epochs","--close-mosaic","--project","--name"]:
            ap.add_argument(a)
        ap.add_argument("--exist-ok", action="store_true"); ap.add_argument("--noplots", action="store_true")
        a = ap.parse_args()
        data = yaml.safe_load(open(a.data))
        assert pathlib.Path(data["path"]).is_absolute(), "data path must be absolute"
        assert a.cfg == "models/detect/yolov9-s.yaml"
        n = int(a.epochs)
        for e in range(n):
            sys.stdout.write(f"      {e}/{n-1}      1.2G      3.1      2.2      1.9        12       640: 50%|#####\\r")
            print(f"      {e}/{n-1}      1.2G      3.1      2.2      1.9        12       640: 100%|##########")
        out = pathlib.Path(a.project) / a.name
        (out / "weights").mkdir(parents=True, exist_ok=True)
        (out / "weights" / "best.pt").write_text("w")
        (out / "results.csv").write_text(
            "               epoch,      train/box_loss,   metrics/precision,      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95\\n"
            "                   0,               3.1,                0.5,                 0.4,                 0.30,                0.10\\n"
            "                   1,               2.1,                0.7,                 0.6,                 0.62,                0.41\\n")
        """,
    )
    write(
        repo / "detect_dual.py",
        """
        import argparse, pathlib
        ap = argparse.ArgumentParser()
        for a in ["--weights","--source","--img","--conf-thres","--device","--project","--name"]:
            ap.add_argument(a)
        for f in ["--save-txt","--save-conf","--nosave","--exist-ok"]:
            ap.add_argument(f, action="store_true")
        a = ap.parse_args()
        out = pathlib.Path(a.project) / a.name / "labels"
        out.mkdir(parents=True, exist_ok=True)
        for img in sorted(pathlib.Path(a.source).iterdir()):
            if img.stem.endswith("0"):   # img1 -> no detections (no file, like the real script)
                (out / f"{img.stem}.txt").write_text("1 0.5 0.5 0.2 0.4 0.87\\n0 0.25 0.25 0.1 0.1 0.33\\n")
        """,
    )
    monkeypatch.setenv("VISIONLOOP_YOLOV9_REPO", str(repo))
    return repo


def test_yolov9_status(monkeypatch, tmp_path):
    # a missing folder, not an unset variable: the default models/yolov9 may exist on this machine
    monkeypatch.setenv("VISIONLOOP_YOLOV9_REPO", str(tmp_path / "missing"))
    assert not yolov9_official.YoloV9Plugin().status().available
    monkeypatch.setenv("VISIONLOOP_YOLOV9_REPO", str(tmp_path))
    assert "train_dual.py" in yolov9_official.YoloV9Plugin().status().message


def test_yolov9_train_and_predict(yolo_repo, tmp_path, images):
    rec = Recorder()
    ds = tmp_path / "ds"
    write(ds / "data.yaml", "path: .\ntrain: images/train\nval: images/valid\nnc: 2\nnames: [a, b]\n")
    ctx = TrainContext(
        dataset_dir=ds, class_names=["a", "b"], variant="s", epochs=3, batch_size=4, image_size=640,
        device="auto", output_dir=tmp_path / "out", init_weights=None, extra={}, hooks=rec.hooks(),
    )
    result = yolov9_official.YoloV9Plugin().train(ctx)
    assert result.weights_path.name == "best.pt"
    assert result.metrics["mAP50-95"] == 0.41 and result.metrics["best_epoch"] == 1
    assert rec.progress and max(rec.progress) < 1.0
    # unfinished progress bars are not logged, finished ones are
    assert not any("50%|" in line for line in rec.lines)
    assert any("100%|" in line for line in rec.lines)

    pctx = PredictContext(
        weights_path=result.weights_path, class_names=["a", "b"], variant="s", confidence=0.25,
        device="cpu", work_dir=tmp_path / "pred", hooks=rec.hooks(),
    )
    preds = dict(yolov9_official.YoloV9Plugin().predict(pctx, images))
    p0 = sorted(preds[images[0]], key=lambda p: -p.score)
    assert len(p0) == 2 and preds[images[1]] == []
    # cx .5 cy .5 w .2 h .4 on 200x100 -> x 80, y 30, w 40, h 40
    assert (p0[0].class_index, round(p0[0].x), round(p0[0].y), round(p0[0].w), round(p0[0].h)) == (1, 80, 30, 40, 40)
    assert p0[0].score == pytest.approx(0.87)


def test_gelan_uses_the_single_branch_scripts_and_its_own_weights(yolo_repo, tmp_path, images):
    # the same stubs under the names GELAN must call; the dual-branch scripts now fail if they are used
    train = (yolo_repo / "train_dual.py").read_text().replace("yolov9-s.yaml", "gelan-c.yaml")
    (yolo_repo / "train.py").write_text(train)
    (yolo_repo / "detect.py").write_text((yolo_repo / "detect_dual.py").read_text())
    (yolo_repo / "train_dual.py").write_text("import sys; sys.exit('train_dual.py must not be used')")
    (yolo_repo / "detect_dual.py").write_text("import sys; sys.exit('detect_dual.py must not be used')")
    (yolo_repo / "weights").mkdir()
    (yolo_repo / "weights" / "gelan-c.pt").write_text("w")
    (yolo_repo / "weights" / "yolov9-c.pt").write_text("w")

    rec = Recorder()
    ds = tmp_path / "ds"
    write(ds / "data.yaml", "path: .\ntrain: images/train\nval: images/valid\nnc: 2\nnames: [a, b]\n")
    ctx = TrainContext(
        dataset_dir=ds, class_names=["a", "b"], variant="gelan-c", epochs=2, batch_size=4, image_size=640,
        device="auto", output_dir=tmp_path / "out", init_weights=None, extra={}, hooks=rec.hooks(),
    )
    plugin = yolov9_official.YoloV9Plugin()
    result = plugin.train(ctx)
    assert any(line.startswith("初始權重") and "gelan-c.pt" in line for line in rec.lines)

    # trained weights are called best.pt: only the size says they are GELAN
    assert result.weights_path.name == "best.pt"
    pctx = PredictContext(
        weights_path=result.weights_path, class_names=["a", "b"], variant="gelan-c", confidence=0.25,
        device="cpu", work_dir=tmp_path / "pred", hooks=rec.hooks(),
    )
    assert len(dict(plugin.predict(pctx, images))[images[0]]) == 2


def test_yolov9_sizes_keep_their_defaults():
    variants = yolov9_official.YoloV9Plugin.variants
    ids = [v.id for v in variants]
    assert {"gelan-s", "gelan-m", "gelan-c", "gelan-e"} <= set(ids)
    assert "gelan-t" not in ids  # no published pretrained weights
    assert ids[0] == "t"  # without a GPU the form starts with the smallest
    # the form recommends the last size that fits: 16 GB still gets YOLOv9-E, not its GELAN twin
    assert [v.id for v in variants if v.min_vram_gb <= 16][-1] == "e"
    assert yolov9_official._model_name("c") == "yolov9-c"
    assert yolov9_official._model_name("gelan-c") == "gelan-c"


# ------------------------------------------------------------------ RT-DETR
@pytest.fixture()
def rtdetr_repo(tmp_path, monkeypatch):
    repo = tmp_path / "rtdetrv2_pytorch"
    write(
        repo / "tools" / "train.py",
        """
        import argparse, json, pathlib
        ap = argparse.ArgumentParser()
        ap.add_argument("-c"); ap.add_argument("-t"); ap.add_argument("-d"); ap.add_argument("--seed")
        ap.add_argument("--use-amp", action="store_true"); ap.add_argument("--output-dir")
        ap.add_argument("-u", nargs="+")
        a = ap.parse_args()
        upd = dict(kv.split("=", 1) for kv in a.u)
        assert upd["num_classes"] == "2" and upd["remap_mscoco_category"] == "False"
        assert upd["train_dataloader.dataset.ann_file"].endswith("instances_train.json")
        assert a.c.endswith("rtdetrv2_r18vd_120e_coco.yml")
        out = pathlib.Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / "log.txt", "w") as f:
            for e in range(int(upd["epoches"])):
                print(f"Epoch: [{e}]  [ 0/10]  eta: 0:00:01  lr: 0.0001  loss: 20.1")
                f.write(json.dumps({"epoch": e, "test_coco_eval_bbox": [0.2 + e / 10, 0.4 + e / 10] + [0] * 10}) + "\\n")
        (out / "best.pth").write_text("w")
        """,
    )
    monkeypatch.setenv("VISIONLOOP_RTDETR_REPO", str(repo))
    return repo


def test_rtdetr_train_and_predict(rtdetr_repo, tmp_path, images, monkeypatch):
    rec = Recorder()
    ctx = TrainContext(
        dataset_dir=tmp_path / "ds", class_names=["a", "b"], variant="r18vd", epochs=3, batch_size=4,
        image_size=640, device="0", output_dir=tmp_path / "out", init_weights=None, extra={}, hooks=rec.hooks(),
    )
    result = rtdetr_official.RTDETRv2Plugin().train(ctx)
    assert result.weights_path.name == "best.pth"
    assert result.metrics["mAP50-95"] == pytest.approx(0.4) and result.metrics["best_epoch"] == 2

    helper = tmp_path / "helper.py"
    write(
        helper,
        """
        import argparse, json
        ap = argparse.ArgumentParser()
        for a in ["--repo","--config","--checkpoint","--num-classes","--conf","--list","--out","--device"]:
            ap.add_argument(a)
        a = ap.parse_args()
        paths = open(a.list).read().splitlines()
        with open(a.out, "w") as f:
            for p in paths:
                f.write(json.dumps({"path": p, "boxes": [[1, 10, 20, 50, 60, 0.9]]}) + "\\n")
        print(f"[{len(paths)}/{len(paths)}]")
        """,
    )
    monkeypatch.setattr(rtdetr_official, "HELPER", helper)
    pctx = PredictContext(
        weights_path=result.weights_path, class_names=["a", "b"], variant="r18vd", confidence=0.25,
        device="auto", work_dir=tmp_path / "pred", hooks=rec.hooks(),
    )
    preds = dict(rtdetr_official.RTDETRv2Plugin().predict(pctx, images))
    p = preds[images[1]][0]
    assert (p.class_index, p.x, p.y, p.w, p.h, p.score) == (1, 10, 20, 40, 40, 0.9)


def test_rtdetr_pretrained_lookup(tmp_path):
    w = tmp_path / "weights"
    w.mkdir()
    (w / "rtdetrv2_r50vd_m_7x_coco_ema.pth").write_text("")
    assert rtdetr_official._find_pretrained(tmp_path, "r50vd") is None
    assert rtdetr_official._find_pretrained(tmp_path, "r50vd_m").name.startswith("rtdetrv2_r50vd_m_")
    (w / "rtdetrv2_r50vd_6x_coco_ema.pth").write_text("")
    assert rtdetr_official._find_pretrained(tmp_path, "r50vd").name == "rtdetrv2_r50vd_6x_coco_ema.pth"


def test_device_mapping():
    assert yolov9_official._device("auto") == ""
    assert yolov9_official._device("0") == "0"
    assert rtdetr_official._device("auto") is None
    assert rtdetr_official._device("1") == "cuda:1"
    assert rtdetr_official._device("cpu") == "cpu"


def test_metrics_files_missing_are_tolerated(tmp_path):
    assert yolov9_official._read_metrics(tmp_path / "nope.csv") == {}
    assert rtdetr_official._read_metrics(tmp_path / "nope.txt") == {}
    (tmp_path / "log.txt").write_text("not json\n" + json.dumps({"epoch": 0}) + "\n")
    assert rtdetr_official._read_metrics(tmp_path / "log.txt") == {}
