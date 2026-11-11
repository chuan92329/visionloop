"""YOLOv9 segmentation adapter against a stub repository, plus the training-form metadata."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from PIL import Image

from app.config import settings
from app.plugins import registry
from app.plugins.base import JobHooks, ModelPlugin, PredictContext, TrainContext, TrainResult, Variant
from app.plugins.builtin import yolov9_official

from .conftest import upload


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def hooks() -> JobHooks:
    return JobHooks(log=lambda s: None, progress=lambda f, m: None, should_stop=lambda: False)


@pytest.fixture()
def seg_repo(tmp_path, monkeypatch):
    repo = tmp_path / "yolov9"
    write(
        repo / "segment" / "train.py",
        """
        import argparse, pathlib, yaml
        ap = argparse.ArgumentParser()
        for a in ["--workers","--device","--batch","--data","--img","--cfg","--weights","--hyp","--epochs",
                  "--close-mosaic","--project","--name"]:
            ap.add_argument(a)
        for f in ["--no-overlap","--exist-ok","--noplots"]:
            ap.add_argument(f, action="store_true")
        a = ap.parse_args()
        assert pathlib.Path(yaml.safe_load(open(a.data))["path"]).is_absolute()
        assert a.cfg == "models/segment/gelan-c-seg.yaml"
        assert not a.noplots
        assert a.img == "768"
        assert a.weights.endswith("gelan-c-seg.pt")
        out = pathlib.Path(a.project) / a.name
        (out / "weights").mkdir(parents=True, exist_ok=True)
        (out / "weights" / "best.pt").write_text("w")
        (out / "results.csv").write_text(
            "epoch,metrics/mAP_0.5(B),metrics/mAP_0.5:0.95(B),metrics/precision(M),metrics/recall(M),metrics/mAP_0.5(M),metrics/mAP_0.5:0.95(M)\\n"
            "0,0.9,0.9,0.5,0.4,0.3,0.1\\n"
            "1,0.1,0.1,0.7,0.6,0.62,0.41\\n"
            "2,0.99,0.99\\n")  # the real script's summary row does not match the header
        """,
    )
    write(
        repo / "segment" / "predict.py",
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
        for img in pathlib.Path(a.source).iterdir():
            if img.stem.endswith("0"):   # img1 -> no detections (no file, like the real script)
                (out / f"{img.stem}.txt").write_text("1 0.1 0.2 0.5 0.2 0.5 0.8 0.87\\n0 0.75\\n")
        """,
    )
    (repo / "weights").mkdir()
    (repo / "weights" / "gelan-c-seg.pt").write_text("pretrained")
    monkeypatch.setenv("VISIONLOOP_YOLOV9_REPO", str(repo))
    return repo


def test_train_reports_mask_scores_and_predict_returns_polygons(seg_repo, tmp_path):
    plugin = yolov9_official.YoloV9SegPlugin()
    assert plugin.status().available
    ds = tmp_path / "ds"
    write(ds / "data.yaml", "path: .\ntrain: images/train\nval: images/valid\nnc: 2\nnames: [a, b]\n")
    result = plugin.train(TrainContext(
        dataset_dir=ds, class_names=["a", "b"], variant="gelan-c", epochs=2, batch_size=4, image_size=768,
        device="auto", output_dir=tmp_path / "out", init_weights=None, extra={}, hooks=hooks(),
    ))
    assert result.weights_path.name == "best.pt"
    # picked on the mask score, not the box score of epoch 0
    assert result.metrics["mAP50-95"] == 0.41 and result.metrics["mAP50"] == 0.62
    assert result.metrics["precision"] == 0.7 and result.metrics["best_epoch"] == 1

    images = []
    for i in range(2):
        p = tmp_path / "imgs" / f"img{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (200, 100)).save(p)
        images.append(p)
    preds = dict(plugin.predict(PredictContext(
        weights_path=result.weights_path, class_names=["a", "b"], variant="gelan-c", confidence=0.25,
        device="cpu", work_dir=tmp_path / "pred", hooks=hooks(),
    ), images))
    assert preds[images[1]] == []
    # the line of an empty mask is skipped
    only = preds[images[0]]
    assert len(only) == 1
    p = only[0]
    assert p.class_index == 1 and p.score == pytest.approx(0.87)
    assert [v for point in p.polygon for v in point] == pytest.approx([20, 20, 100, 20, 100, 80])
    assert (p.x, p.y, p.w, p.h) == pytest.approx((20, 20, 80, 60))


def test_launcher_corrects_the_prototype_bug_in_memory_only(tmp_path):
    import subprocess
    import sys

    source = textwrap.dedent(
        """
        import sys
        model = lambda *a, **k: ("pred", ("x", "mc", "prototypes"))
        im = augment = visualize = None
        for _ in range(1):
            if True:
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            print(proto if isinstance(proto, str) else "tuple", sys.argv[1:], __file__.endswith("predict.py"))
        """
    )
    script = tmp_path / "segment" / "predict.py"
    script.parent.mkdir()
    script.write_text(source)
    launcher = Path(yolov9_official.COMPAT)
    out = subprocess.run([sys.executable, str(launcher), str(script), "--a"], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "prototypes ['--a'] True"
    assert script.read_text() == source

    # any other script runs unchanged
    other = tmp_path / "predict.py"
    other.write_text(source)
    out = subprocess.run([sys.executable, str(launcher), str(other)], capture_output=True, text=True, check=True).stdout
    assert out.startswith("tuple")


def test_form_metadata(client):
    plugins = {p["id"]: p for p in client.get("/api/plugins").json()["plugins"]}
    assert (plugins["yolov9"]["family"], plugins["yolov9"]["task"]) == ("YOLOv9", "detect")
    assert (plugins["yolov9-seg"]["family"], plugins["yolov9-seg"]["task"]) == ("YOLOv9", "segment")
    assert (plugins["rfdetr"]["family"], plugins["rfdetr"]["task"]) == ("RF-DETR", "detect")
    assert (plugins["rfdetr-seg"]["family"], plugins["rfdetr-seg"]["task"]) == ("RF-DETR", "segment")
    assert {v["image_step"] for v in plugins["rfdetr-seg"]["variants"]} == {12, 24}


def test_plugin_dropped_in_the_folder_appears_after_reload(client):
    body = client.get("/api/plugins").json()
    assert body["plugin_dir"] == str(settings.plugin_dir)
    assert "mine" not in {p["id"] for p in body["plugins"]}
    (settings.plugin_dir / "mine.py").write_text(textwrap.dedent(
        """
        from app.plugins.base import ModelPlugin, Variant

        class Mine(ModelPlugin):
            id = "mine"
            name = "Mine"
            family = "My Models"
            task = "segment"
            dataset_format = "yolo-seg"
            variants = [Variant("a", "A")]

            def train(self, ctx):
                raise NotImplementedError

            def predict(self, ctx, images):
                return iter(())
        """
    ))
    body = client.post("/api/plugins/reload").json()
    mine = next(p for p in body["plugins"] if p["id"] == "mine")
    assert (mine["family"], mine["task"]) == ("My Models", "segment")
    assert "yolov9" in {p["id"] for p in body["plugins"]}  # the built-in ones stay


class Sized(ModelPlugin):
    id = "sized"
    name = "Sized"
    variants = [Variant("a", "A", image_size=312, image_step=12), Variant("b", "B")]
    seen: list[int] = []

    def train(self, ctx):
        Sized.seen.append(ctx.image_size)
        weights = ctx.output_dir / "w.pt"
        weights.write_text("w")
        return TrainResult(weights, {})

    def predict(self, ctx, images):
        return iter(())


def test_image_size_defaults_and_validation(client, project):
    from .conftest import wait_job

    registry._register(Sized(), "test")
    pid = project["id"]
    cid = project["classes"][0]["id"]
    for img in upload(client, pid, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])["created"]:
        box = {"class_id": cid, "x": 10, "y": 10, "w": 40, "h": 30}
        client.put(f"/api/images/{img['id']}/annotations", json={"boxes": [box], "mark_reviewed": True})
    client.post(f"/api/projects/{pid}/split", json={"config": {"train": 0.5, "valid": 0.5, "test": 0, "seed": 1}, "mode": "full"})

    r = client.post(f"/api/projects/{pid}/train", json={"plugin": "sized", "variant": "a", "image_size": 1000})
    assert r.status_code == 422 and "12 的倍數" in r.json()["detail"]

    # 640 is honoured, no size means the size's own default, and 640 for a variant without one
    for body, expected in (({"variant": "a", "image_size": 624}, 624), ({"variant": "a"}, 312), ({"variant": "b"}, 640)):
        r = client.post(f"/api/projects/{pid}/train", json={"plugin": "sized", **body})
        assert r.status_code == 201, r.text
        assert wait_job(client, r.json()["id"])["status"] == "succeeded"
        assert Sized.seen[-1] == expected
