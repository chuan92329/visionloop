from app.plugins import registry
from app.plugins.base import ModelPlugin, Prediction, Variant

from .conftest import upload, wait_job


class FakeOutliner(ModelPlugin):
    """Outlines every box as a triangle, except that it fails on boxes wider than 100 px."""

    id = "fake-outliner"
    name = "Fake outliner"
    needs_weights = False
    trainable = False
    prompt_mode = "boxes"
    variants = [Variant("v", "v")]
    seen: dict = {}

    def train(self, ctx):
        raise NotImplementedError

    def predict(self, ctx, images):
        FakeOutliner.options = ctx.options
        for p in images:
            FakeOutliner.seen[p.name] = ctx.box_prompts[str(p)]
            yield p, [
                Prediction(0, x, y, w, h, 0.9, polygon=None if w > 100 else [[x, y], [x + w, y], [x, y + h]])
                for x, y, w, h in ctx.box_prompts[str(p)]
            ]


class FakeSeg(ModelPlugin):
    id = "fake-seg"
    name = "Fake seg"
    task = "segment"
    dataset_format = "yolo-seg"
    variants = [Variant("v", "v")]

    def train(self, ctx):
        raise NotImplementedError

    def predict(self, ctx, images):
        return iter(())


def start(client, pid, **kw):
    return client.post(f"/api/projects/{pid}/autolabel", json={"plugin": FakeOutliner.id, **kw})


def test_outlines_boxes_of_reviewed_images_and_keeps_everything_else(client, project):
    registry._register(FakeOutliner(), "test")
    pid = project["id"]
    cid = project["classes"][0]["id"]
    img = upload(client, pid, ["a.jpg"])["created"][0]
    polygon = [[150, 10], [190, 10], [170, 50]]
    boxes = [
        {"class_id": cid, "x": 10, "y": 10, "w": 40, "h": 30},  # gets outlined
        {"class_id": cid, "x": 0, "y": 50, "w": 120, "h": 30},  # the model fails on this one
        {"class_id": cid, "x": 1, "y": 1, "w": 2, "h": 2, "polygon": polygon},  # already has an outline
    ]
    r = client.put(f"/api/images/{img['id']}/annotations", json={"boxes": boxes, "mark_reviewed": True})
    assert r.status_code == 200, r.text
    before = r.json()["image"]["version"]

    r = start(client, pid, statuses=["reviewed"])
    assert r.status_code == 201, r.text
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "succeeded", job
    assert "1 個框轉成多邊形" in job["message"] and "1 個框無法轉換" in job["message"]

    # only boxes without an outline were sent to the model
    assert [len(v) for v in FakeOutliner.seen.values()] == [2]

    state = client.get(f"/api/images/{img['id']}/annotations").json()
    assert state["image"]["status"] == "auto"
    outlined, failed, kept = state["boxes"]
    assert outlined["polygon"] == [[10, 10], [50, 10], [10, 40]]
    assert (outlined["source"], outlined["score"]) == ("auto", 0.9)
    assert failed["polygon"] is None
    assert (failed["source"], failed["w"]) == ("manual", 120)
    assert kept["polygon"] == polygon
    assert kept["source"] == "manual"

    # the old boxes can be brought back
    r = client.post(f"/api/images/{img['id']}/revisions/{before}/restore")
    assert [b["polygon"] for b in r.json()["boxes"]] == [None, None, polygon]


def test_only_the_selected_images_are_outlined(client, project):
    registry._register(FakeOutliner(), "test")
    pid = project["id"]
    cid = project["classes"][0]["id"]
    chosen, other = upload(client, pid, ["a.jpg", "b.jpg"])["created"]
    box = {"class_id": cid, "x": 10, "y": 10, "w": 40, "h": 30}
    for img in (chosen, other):
        client.put(f"/api/images/{img['id']}/annotations", json={"boxes": [box], "mark_reviewed": True})

    r = start(client, pid, statuses=[], image_ids=[chosen["id"]])
    assert r.status_code == 201, r.text
    assert wait_job(client, r.json()["id"])["status"] == "succeeded"

    state = client.get(f"/api/images/{chosen['id']}/annotations").json()
    assert state["image"]["status"] == "auto"
    assert state["boxes"][0]["polygon"] is not None
    state = client.get(f"/api/images/{other['id']}/annotations").json()
    assert state["image"]["status"] == "reviewed"
    assert state["boxes"][0]["polygon"] is None


def test_existing_outlines_are_redone_only_on_request(client, project):
    registry._register(FakeOutliner(), "test")
    pid = project["id"]
    cid = project["classes"][0]["id"]
    img = upload(client, pid, ["a.jpg"])["created"][0]
    old = [[10, 10], [50, 10], [50, 40], [10, 40]]
    box = {"class_id": cid, "x": 10, "y": 10, "w": 40, "h": 30, "polygon": old}
    client.put(f"/api/images/{img['id']}/annotations", json={"boxes": [box], "mark_reviewed": True})

    job = wait_job(client, start(client, pid, statuses=[], image_ids=[img["id"]]).json()["id"])
    assert job["result"]["labelled"] == 0
    assert client.get(f"/api/images/{img['id']}/annotations").json()["boxes"][0]["polygon"] == old

    r = start(client, pid, statuses=[], image_ids=[img["id"]], overwrite_polygons=True, polygon_detail="fine")
    assert wait_job(client, r.json()["id"])["status"] == "succeeded"
    assert FakeOutliner.options["polygon_detail"] == "fine"
    state = client.get(f"/api/images/{img['id']}/annotations").json()
    assert state["boxes"][0]["polygon"] == [[10, 10], [50, 10], [10, 40]]
    assert state["image"]["status"] == "auto"


def test_nothing_to_outline(client, project):
    registry._register(FakeOutliner(), "test")
    upload(client, project["id"], ["a.jpg"])
    r = start(client, project["id"], statuses=["unlabeled"])
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "succeeded"
    assert job["result"]["labelled"] == 0


def test_normal_models_still_cannot_touch_reviewed_images(client, project):
    r = client.post(f"/api/projects/{project['id']}/autolabel", json={"plugin": "demo", "statuses": ["reviewed"]})
    assert r.status_code == 422


def test_segmentation_training_needs_polygons(client, project):
    registry._register(FakeSeg(), "test")
    pid = project["id"]
    cid = project["classes"][0]["id"]
    for img in upload(client, pid, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])["created"]:
        box = {"class_id": cid, "x": 10, "y": 10, "w": 40, "h": 30}
        client.put(f"/api/images/{img['id']}/annotations", json={"boxes": [box], "mark_reviewed": True})
    client.post(f"/api/projects/{pid}/split", json={"config": {"train": 0.5, "valid": 0.5, "test": 0, "seed": 1}, "mode": "full"})
    r = client.post(f"/api/projects/{pid}/train", json={"plugin": "fake-seg", "variant": "v"})
    assert r.status_code == 409
    assert "多邊形" in r.json()["detail"]
