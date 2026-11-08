import io
import json
import zipfile

import pytest
import yaml

from .conftest import make_jpeg, upload, wait_job


def cls_ids(project):
    return {c["name"]: c["id"] for c in project["classes"]}


def box(class_id, x=10, y=10, w=40, h=30, **kw):
    return {"class_id": class_id, "x": x, "y": y, "w": w, "h": h, **kw}


def review_all(client, ids, class_id):
    for i in ids:
        r = client.put(f"/api/images/{i}/annotations", json={"boxes": [box(class_id)], "mark_reviewed": True})
        assert r.status_code == 200


# ------------------------------------------------------------------ images
def test_upload_rejects_bad_files_and_duplicates(client, project):
    pid = project["id"]
    jpg = make_jpeg()
    files = [
        ("files", ("a.jpg", jpg, "image/jpeg")),
        ("files", ("a_copy.jpg", jpg, "image/jpeg")),
        ("files", ("notes.txt", b"hello", "text/plain")),
        ("files", ("broken.png", b"not a png", "image/png")),
    ]
    r = client.post(f"/api/projects/{pid}/images", files=files)
    body = r.json()
    assert len(body["created"]) == 1
    assert {s["filename"] for s in body["skipped"]} == {"a_copy.jpg", "notes.txt", "broken.png"}
    img = body["created"][0]
    assert (img["width"], img["height"]) == (200, 100)
    assert client.get(f"/api/images/{img['id']}/file").status_code == 200
    assert client.get(f"/api/images/{img['id']}/thumb").headers["content-type"] == "image/jpeg"


def test_upload_converts_heic_and_rejects_broken_dng(client, project):
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image as PILImage

    pillow_heif.register_heif_opener()
    src = PILImage.open(io.BytesIO(make_jpeg()))
    exif = PILImage.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display -> 100 x 200 upright
    heic = io.BytesIO()
    src.save(heic, "HEIF", exif=exif.tobytes())

    files = [
        ("files", ("phone.heic", heic.getvalue(), "image/heic")),
        ("files", ("broken.dng", b"not a raw file", "image/x-adobe-dng")),
    ]
    body = client.post(f"/api/projects/{project['id']}/images", files=files).json()
    assert [s["filename"] for s in body["skipped"]] == ["broken.dng"]
    img = body["created"][0]
    assert img["filename"] == "phone.heic"
    assert (img["width"], img["height"]) == (100, 200)
    r = client.get(f"/api/images/{img['id']}/file")
    assert r.status_code == 200
    with PILImage.open(io.BytesIO(r.content)) as served:
        assert served.format == "JPEG"
        assert served.size == (100, 200)


def test_list_filters_and_queue(client, project):
    pid = project["id"]
    created = upload(client, pid, [f"img{i}.jpg" for i in range(5)])["created"]
    ids = [c["id"] for c in created]
    c = cls_ids(project)["cat"]
    review_all(client, ids[:2], c)
    page = client.get(f"/api/projects/{pid}/images", params={"status": "reviewed"}).json()
    assert page["total"] == 2
    queue = client.get(f"/api/projects/{pid}/image-ids", params={"status": ["unlabeled"]}).json()
    assert queue == ids[2:]


# ------------------------------------------------------------- annotations
def test_status_flow_revisions_and_restore(client, project):
    pid = project["id"]
    img = upload(client, pid, ["x.jpg"])["created"][0]
    cat = cls_ids(project)["cat"]
    url = f"/api/images/{img['id']}/annotations"

    # boxes are clipped to the image
    s = client.put(url, json={"boxes": [box(cat, x=-20, w=60)], "base_version": 0}).json()
    assert s["image"]["status"] == "in_progress"
    assert s["boxes"][0]["x"] == 0 and s["boxes"][0]["w"] == 40

    # stale edit is rejected
    assert client.put(url, json={"boxes": [], "base_version": 0}).status_code == 409

    s = client.put(url, json={"boxes": [box(cat), box(cat, x=100)], "mark_reviewed": True}).json()
    assert s["image"]["status"] == "reviewed" and s["image"]["version"] == 2

    # saving unchanged boxes of a reviewed image keeps it reviewed and creates no revision
    s = client.put(url, json={"boxes": [box(cat), box(cat, x=100)]}).json()
    assert s["image"]["status"] == "reviewed" and s["image"]["version"] == 2

    revs = client.get(f"/api/images/{img['id']}/revisions").json()
    assert [r["version"] for r in revs] == [2, 1]
    assert revs[1]["boxes"][0]["class_name"] == "cat"

    s = client.post(f"/api/images/{img['id']}/revisions/1/restore").json()
    assert len(s["boxes"]) == 1 and s["image"]["status"] == "in_progress"


def test_unknown_class_rejected(client, project):
    img = upload(client, project["id"], ["x.jpg"])["created"][0]
    r = client.put(f"/api/images/{img['id']}/annotations", json={"boxes": [box(9999)]})
    assert r.status_code == 422


def test_deleting_class_requires_rereview(client, project):
    pid = project["id"]
    ids = cls_ids(project)
    img = upload(client, pid, ["x.jpg"])["created"][0]
    client.put(
        f"/api/images/{img['id']}/annotations",
        json={"boxes": [box(ids["cat"]), box(ids["dog"], x=80)], "mark_reviewed": True},
    )
    assert client.get(f"/api/classes/{ids['dog']}/usage").json() == {"images": 1, "boxes": 1}
    assert client.delete(f"/api/classes/{ids['dog']}").status_code == 204
    s = client.get(f"/api/images/{img['id']}/annotations").json()
    assert len(s["boxes"]) == 1 and s["image"]["status"] == "in_progress"


# ----------------------------------------------------------- split/export
def test_split_lock_and_export_yolo_and_coco(client, project):
    pid = project["id"]
    ids = cls_ids(project)
    created = upload(client, pid, [f"img{i}.jpg" for i in range(30)])["created"]
    for k, img in enumerate(created):
        client.put(
            f"/api/images/{img['id']}/annotations",
            json={"boxes": [box([ids["cat"], ids["dog"]][k % 2], x=20, y=30, w=40, h=20)], "mark_reviewed": True},
        )

    stats = client.post(f"/api/projects/{pid}/split", json={"lock_test": True}).json()
    assert sum(stats["images"].values()) == 30
    assert stats["test_locked"] is True
    test_before = set(
        client.get(f"/api/projects/{pid}/image-ids", params={"split": "test"}).json()
    )

    # locked test set cannot be edited by hand ...
    tid = next(iter(test_before))
    assert client.patch(f"/api/images/{tid}", json={"split": "train"}).status_code == 409
    # ... and a full reshuffle keeps it
    client.post(f"/api/projects/{pid}/split", json={"mode": "full"})
    test_after = set(client.get(f"/api/projects/{pid}/image-ids", params={"split": "test"}).json())
    assert test_before == test_after

    # YOLO
    r = client.get(f"/api/projects/{pid}/export", params={"format": "yolo"})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    data = yaml.safe_load(z.read("data.yaml"))
    assert data["names"] == ["cat", "dog", "bird"] and data["nc"] == 3
    labels = [n for n in z.namelist() if n.startswith("labels/") and n.endswith(".txt")]
    assert len(labels) == 30
    line = z.read(labels[0]).decode().split()
    # box (20,30,40,20) on 200x100 -> cx .2 cy .4 w .2 h .2
    assert [float(v) for v in line[1:]] == [0.2, 0.4, 0.2, 0.2]

    # COCO
    r = client.get(f"/api/projects/{pid}/export", params={"format": "coco"})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    train = json.loads(z.read("annotations/instances_train.json"))
    assert [c["id"] for c in train["categories"]] == [1, 2, 3]
    assert train["annotations"][0]["bbox"] == [20.0, 30.0, 40.0, 20.0]


def test_export_without_reviewed_data_is_409(client, project):
    upload(client, project["id"], ["a.jpg"])
    r = client.get(f"/api/projects/{project['id']}/export")
    assert r.status_code == 409


def test_auto_group(client, project):
    pid = project["id"]
    upload(client, pid, ["cam1_0001.jpg", "cam1_0002.jpg", "cam2-15.jpg", "single.jpg"])
    assert client.post(f"/api/projects/{pid}/auto-group", json={}).json()["updated"] == 3
    groups = {i["filename"]: i["group_key"] for i in client.get(f"/api/projects/{pid}/images").json()["items"]}
    assert groups == {"cam1_0001.jpg": "cam1", "cam1_0002.jpg": "cam1", "cam2-15.jpg": "cam2", "single.jpg": None}


# ------------------------------------------------------- the full loop
def test_closed_loop_with_demo_plugin(client, project):
    """auto-label -> human review -> split -> train -> auto-label with trained weights"""
    pid = project["id"]
    ids = cls_ids(project)
    created = upload(client, pid, [f"img{i}.jpg" for i in range(20)])["created"]

    job = client.post(f"/api/projects/{pid}/autolabel", json={"plugin": "demo", "confidence": 0.0}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded", job
    auto = client.get(f"/api/projects/{pid}/images", params={"status": "auto"}).json()
    assert auto["total"] == 20
    assert all(i["min_score"] is not None for i in auto["items"])

    # a human confirms 15 of them (the rest stay "auto")
    for img in created[:15]:
        s = client.get(f"/api/images/{img['id']}/annotations").json()
        client.put(
            f"/api/images/{img['id']}/annotations",
            json={"boxes": [{k: b[k] for k in ("class_id", "x", "y", "w", "h", "source", "score")} for b in s["boxes"]],
                  "mark_reviewed": True},
        )

    # training is refused before splitting
    assert client.post(f"/api/projects/{pid}/train", json={"plugin": "demo", "variant": "fast"}).status_code == 409
    client.post(f"/api/projects/{pid}/split", json={})

    job = client.post(f"/api/projects/{pid}/train", json={"plugin": "demo", "variant": "fast", "epochs": 3}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}").json()["log"]
    weights = client.get(f"/api/projects/{pid}/weights").json()
    assert len(weights) == 1 and weights[0]["class_names"] == ["cat", "dog", "bird"]

    # auto-label never touches reviewed images
    reviewed_before = client.get(f"/api/images/{created[0]['id']}").json()["version"]
    job = client.post(
        f"/api/projects/{pid}/autolabel",
        json={"plugin": "demo", "weights_id": weights[0]["id"], "statuses": ["auto"], "confidence": 0.0},
    ).json()
    job = wait_job(client, job["id"])
    assert job["result"]["labelled"] == 5
    assert client.get(f"/api/images/{created[0]['id']}").json()["version"] == reviewed_before

    # asking to auto-label reviewed images is refused outright
    r = client.post(f"/api/projects/{pid}/autolabel", json={"plugin": "demo", "statuses": ["reviewed"]})
    assert r.status_code == 422
    assert ids  # keep linter quiet


def test_unavailable_plugin_is_reported(client, project):
    plugins = {p["id"]: p for p in client.get("/api/plugins").json()["plugins"]}
    assert plugins["demo"]["available"]
    assert {"yolov9", "rtdetrv2"} <= set(plugins)
    if not plugins["yolov9"]["available"]:
        r = client.post(f"/api/projects/{project['id']}/autolabel", json={"plugin": "yolov9", "weights_id": 1})
        assert r.status_code == 422


def test_user_plugin_folder(tmp_path, client):
    from app.config import settings
    from app.plugins import registry

    (settings.plugin_dir / "my_model.py").write_text(
        "from app.plugins.base import ModelPlugin, Variant, TrainResult\n"
        "class Mine(ModelPlugin):\n"
        "    id = 'mine'\n    name = 'Mine'\n    variants = [Variant('a', 'A')]\n"
        "    def train(self, ctx): return TrainResult(ctx.output_dir / 'w')\n"
        "    def predict(self, ctx, images):\n        for p in images: yield p, []\n"
    )
    (settings.plugin_dir / "broken.py").write_text("raise RuntimeError('boom')\n")
    registry.load_plugins(settings.plugin_dir)
    body = client.get("/api/plugins").json()
    assert "mine" in {p["id"] for p in body["plugins"]}
    assert body["load_errors"][0]["file"] == "broken.py"


def test_imported_weights_keep_original_name(client, project):
    from pathlib import Path

    from app.db import SessionLocal
    from app.models import ModelWeights

    r = client.post(
        f"/api/projects/{project['id']}/weights/import",
        data={"plugin": "yolov9", "variant": "c", "class_names": "person, car"},
        files={"file": ("yolov9-c-converted.pt", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["class_names"] == ["person", "car"]
    with SessionLocal() as db:
        path = Path(db.get(ModelWeights, r.json()["id"]).path)
    # the YOLOv9 plugin relies on "converted" being in the name to pick detect.py
    assert path.name.endswith("_yolov9-c-converted.pt") and path.read_bytes() == b"fake"
