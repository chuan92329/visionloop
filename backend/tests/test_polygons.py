import io
import json
import sqlite3
import zipfile

from sqlalchemy import inspect

from app.db import get_engine, init_engine

from .conftest import upload

TRIANGLE = [[10, 10], [60, 10], [35, 50]]


def setup_image(client, project):
    img = upload(client, project["id"], ["x.jpg"])["created"][0]
    return img, project["classes"][0]["id"]


def save(client, img_id, class_id, polygon=None, **kw):
    box = {"class_id": class_id, "x": 1, "y": 1, "w": 5, "h": 5, "polygon": polygon}
    return client.put(f"/api/images/{img_id}/annotations", json={"boxes": [box], **kw})


def split_all(client, pid):
    r = client.post(
        f"/api/projects/{pid}/split",
        json={"config": {"train": 1.0, "valid": 0.0, "test": 0.0, "seed": 1}, "mode": "full"},
    )
    assert r.status_code == 200, r.text


def export_zip(client, pid, fmt):
    r = client.get(f"/api/projects/{pid}/export", params={"format": fmt})
    assert r.status_code == 200, r.text
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_polygon_round_trip_derives_the_box(client, project):
    img, cid = setup_image(client, project)
    box = save(client, img["id"], cid, TRIANGLE).json()["boxes"][0]
    assert box["polygon"] == TRIANGLE
    assert (box["x"], box["y"], box["w"], box["h"]) == (10, 10, 50, 40)  # the client's x/y/w/h are ignored
    again = client.get(f"/api/images/{img['id']}/annotations").json()["boxes"][0]
    assert again["polygon"] == TRIANGLE


def test_plain_box_has_no_polygon(client, project):
    img, cid = setup_image(client, project)
    box = save(client, img["id"], cid).json()["boxes"][0]
    assert box["polygon"] is None
    assert (box["x"], box["w"]) == (1, 5)


def test_vertices_are_clipped_to_the_image(client, project):
    img, cid = setup_image(client, project)  # 200 x 100
    box = save(client, img["id"], cid, [[-20, 10], [500, 10], [100, 300]]).json()["boxes"][0]
    assert box["polygon"] == [[0, 10], [200, 10], [100, 100]]
    assert (box["x"], box["y"], box["w"], box["h"]) == (0, 10, 200, 90)


def test_degenerate_polygons_are_dropped_or_rejected(client, project):
    img, cid = setup_image(client, project)
    assert save(client, img["id"], cid, [[10, 10], [20, 10]]).status_code == 422  # fewer than 3 points
    body = save(client, img["id"], cid, [[10, 10], [20, 10], [30, 10]]).json()  # no area
    assert body["boxes"] == []


def test_moving_only_a_vertex_creates_a_new_version(client, project):
    img, cid = setup_image(client, project)
    v1 = save(client, img["id"], cid, TRIANGLE).json()["image"]["version"]
    # same bounding box, different outline
    v2 = save(client, img["id"], cid, [[10, 10], [60, 10], [30, 50]]).json()["image"]["version"]
    assert v2 == v1 + 1
    assert save(client, img["id"], cid, [[10, 10], [60, 10], [30, 50]]).json()["image"]["version"] == v2


def test_restoring_a_revision_keeps_the_polygon(client, project):
    img, cid = setup_image(client, project)
    v1 = save(client, img["id"], cid, TRIANGLE).json()["image"]["version"]
    save(client, img["id"], cid)
    r = client.post(f"/api/images/{img['id']}/revisions/{v1}/restore")
    assert r.json()["boxes"][0]["polygon"] == TRIANGLE


def test_exports(client, project):
    pid = project["id"]
    created = upload(client, pid, ["a.jpg", "b.jpg"])["created"]
    cid = project["classes"][0]["id"]
    assert save(client, created[0]["id"], cid, TRIANGLE, mark_reviewed=True).status_code == 200
    assert save(client, created[1]["id"], cid, mark_reviewed=True).status_code == 200
    split_all(client, pid)

    stats = client.get(f"/api/projects/{pid}/split").json()
    assert (stats["polygons"], stats["boxes_without_polygon"]) == (1, 1)

    with export_zip(client, pid, "yolo") as z:
        labels = sorted(n for n in z.namelist() if n.startswith("labels/"))
        # detection export is unchanged: class cx cy w h
        assert [len(z.read(n).decode().split()) for n in labels] == [5, 5]

    with export_zip(client, pid, "yolo-seg") as z:
        labels = sorted(n for n in z.namelist() if n.startswith("labels/"))
        first, second = (z.read(n).decode().split() for n in labels)
        assert first == ["0", "0.050000", "0.100000", "0.300000", "0.100000", "0.175000", "0.500000"]
        assert len(second) == 9  # a plain box becomes a rectangle

    with export_zip(client, pid, "coco") as z:
        doc = json.loads(z.read("annotations/instances_train.json"))
        with_outline, plain = sorted(doc["annotations"], key=lambda a: a["image_id"])
        assert with_outline["segmentation"] == [[10, 10, 60, 10, 35, 50]]
        assert with_outline["area"] == 1000  # the polygon's area, not the box's 2000
        assert "segmentation" not in plain


def test_old_database_gets_the_column_and_keeps_its_boxes(tmp_path):
    path = tmp_path / "old.sqlite3"
    init_engine(path)
    get_engine().dispose()
    with sqlite3.connect(path) as c:
        c.execute("PRAGMA foreign_keys=OFF")
        c.execute("ALTER TABLE annotations DROP COLUMN polygon")
        c.execute("INSERT INTO annotations (image_id, class_id, x, y, w, h, source) VALUES (1, 1, 1, 2, 3, 4, 'manual')")
    c.close()

    init_engine(path)
    assert "polygon" in {col["name"] for col in inspect(get_engine()).get_columns("annotations")}
    get_engine().dispose()
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT x, h, polygon FROM annotations").fetchall() == [(1.0, 4.0, None)]
    c.close()
