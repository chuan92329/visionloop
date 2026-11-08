"""RF-DETR / Grounding DINO / probe plugins against a stub "model environment".

VISIONLOOP_MODELS_PYTHON points at a tiny fake interpreter that understands the
helper-script protocol (spec in, JSON lines out), so the whole platform side –
exemplar collection, text prompts, name mapping, job handling – is tested
without PyTorch.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap

import pytest

from app.plugins import base

from .conftest import upload, wait_job

STUB = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import json, sys, pathlib
    if sys.argv[1] == "-c":          # module availability check
        sys.exit(0)
    helper = pathlib.Path(sys.argv[1]).name
    spec = json.load(open(sys.argv[sys.argv.index("--spec") + 1], encoding="utf-8"))
    pathlib.Path(spec.get("out") or spec.get("output_dir") or ".").parent.mkdir(parents=True, exist_ok=True)
    if helper == "hf_detect_helper.py":
        # blank descriptions fall back to the class name
        assert spec["prompts"] == ["cat", "a small dog", "bird"], spec["prompts"]
        cls = spec["prompts"].index("a small dog")
        with open(spec["out"], "w") as f:
            for i, p in enumerate(spec["images"], 1):
                f.write(json.dumps({{"path": p, "boxes": [[cls, 10, 10, 30, 20, 0.81]]}}) + "\\n")
                print(f"[{{i}}/{{len(spec['images'])}}]")
    elif helper == "probe_helper.py":
        assert spec["backbone"] == "dinov2" and spec["model_id"].startswith("facebook/dinov2")
        assert spec["cache_dir"] and spec["head"] == "mlp"
        n_neg = sum(1 for t in spec["training"] if not t["boxes"])
        print(f"training on {{len(spec['training'])}} images, {{n_neg}} negatives")
        cls = next(t["boxes"][0][0] for t in spec["training"] if t["boxes"])
        with open(spec["out"], "w") as f:
            for p in spec["images"]:
                f.write(json.dumps({{"path": p, "boxes": [[cls, 12, 12, 40, 30, 0.77]]}}) + "\\n")
    elif helper == "rfdetr_helper.py" and sys.argv[2] == "train":
        out = pathlib.Path(spec["output_dir"]); out.mkdir(parents=True, exist_ok=True)
        for e in range(spec["epochs"]):
            print(f"Epoch {{e}}: 100%|##########| 3/3")
        (out / "checkpoint_best_total.pth").write_text("w")
        (out / "visionloop_result.json").write_text(json.dumps(
            {{"weights": str(out / "checkpoint_best_total.pth"), "metrics": {{"mAP50-95": 0.5}}}}))
    elif helper == "rfdetr_helper.py":
        names = ["dog", "person"] if spec["weights"] is None else ["cat"]
        with open(spec["out"], "w") as f:
            for p in spec["images"]:
                f.write(json.dumps({{"path": p, "boxes": [[7, n, 5, 5, 20, 20, 0.9] for n in names]}}) + "\\n")
    else:
        sys.exit("unexpected call " + " ".join(sys.argv))
    """
)


@pytest.fixture()
def models_env(tmp_path, monkeypatch):
    stub = tmp_path / "fake-python"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("VISIONLOOP_MODELS_PYTHON", str(stub))
    base._module_cache.clear()
    yield stub
    base._module_cache.clear()


def ids_of(project):
    return {c["name"]: c["id"] for c in project["classes"]}


@pytest.mark.skipif(os.name != "posix", reason="stub interpreter uses a shebang")
def test_plugins_listed_and_available(client, models_env):
    plugins = {p["id"]: p for p in client.get("/api/plugins").json()["plugins"]}
    for pid in ("rfdetr", "grounding-dino"):
        assert plugins[pid]["available"], plugins[pid]["status_message"]
    assert plugins["grounding-dino"]["prompt_mode"] == "text"
    assert plugins["rfdetr"]["trainable"] and not plugins["rfdetr"]["needs_weights"]


def test_plugins_unavailable_without_env(client, monkeypatch):
    monkeypatch.delenv("VISIONLOOP_MODELS_PYTHON", raising=False)
    base._module_cache.clear()
    plugins = {p["id"]: p for p in client.get("/api/plugins").json()["plugins"]}
    assert not plugins["grounding-dino"]["available"]
    assert "setup_windows.ps1" in plugins["grounding-dino"]["status_message"]


@pytest.mark.skipif(os.name != "posix", reason="stub interpreter uses a shebang")
def test_grounding_dino_uses_class_descriptions(client, project, models_env):
    pid = project["id"]
    ids = ids_of(project)
    created = upload(client, pid, ["a.jpg", "b.jpg"])["created"]
    job = client.post(
        f"/api/projects/{pid}/autolabel",
        json={"plugin": "grounding-dino", "prompts": {ids["cat"]: "", ids["dog"]: "a small dog", ids["bird"]: ""},
              "variant": "tiny"},
    ).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}").json()["log"]
    s = client.get(f"/api/images/{created[0]['id']}/annotations").json()
    assert [b["class_id"] for b in s["boxes"]] == [ids["dog"]]


@pytest.mark.skipif(os.name != "posix", reason="stub interpreter uses a shebang")
def test_rfdetr_coco_names_train_and_trained_weights(client, project, models_env):
    pid = project["id"]
    ids = ids_of(project)
    created = upload(client, pid, [f"img{i}.jpg" for i in range(8)])["created"]

    # COCO-pretrained (no weights): labels come back by name; "person" is not a project class
    job = wait_job(client, client.post(f"/api/projects/{pid}/autolabel",
                                       json={"plugin": "rfdetr", "variant": "nano"}).json()["id"])
    assert job["status"] == "succeeded"
    s = client.get(f"/api/images/{created[0]['id']}/annotations").json()
    assert [b["class_id"] for b in s["boxes"]] == [ids["dog"]]
    assert "person" in client.get(f"/api/jobs/{job['id']}").json()["log"]

    # clear the model's boxes again
    r = client.post(f"/api/projects/{pid}/clear-auto", json={}).json()
    assert r["cleared"] == 8
    assert client.get(f"/api/projects/{pid}").json()["stats"]["unlabeled"] == 8
    revs = client.get(f"/api/images/{created[0]['id']}/revisions").json()
    assert revs[0]["action"] == "clear" and revs[0]["status"] == "unlabeled"

    # train, then auto-label with the trained checkpoint (class index -> name "cat")
    for img in created[:6]:
        client.put(f"/api/images/{img['id']}/annotations",
                   json={"boxes": [{"class_id": ids["cat"], "x": 5, "y": 5, "w": 30, "h": 30}], "mark_reviewed": True})
    client.post(f"/api/projects/{pid}/split", json={"config": {"train": 0.5, "valid": 0.5, "test": 0, "seed": 1}})
    job = wait_job(client, client.post(f"/api/projects/{pid}/train",
                                       json={"plugin": "rfdetr", "variant": "small", "epochs": 3}).json()["id"])
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}").json()["log"]
    w = client.get(f"/api/projects/{pid}/weights").json()[0]
    assert w["plugin"] == "rfdetr" and w["metrics"]["mAP50-95"] == 0.5
    job = wait_job(client, client.post(f"/api/projects/{pid}/autolabel",
                                       json={"plugin": "rfdetr", "weights_id": w["id"]}).json()["id"])
    assert job["result"]["labelled"] == 2
    s = client.get(f"/api/images/{created[7]['id']}/annotations").json()
    assert [b["class_id"] for b in s["boxes"]] == [ids["cat"]]


def test_local_settings_file(tmp_path, monkeypatch):
    from app import config

    monkeypatch.delenv("VISIONLOOP_MODELS_PYTHON", raising=False)
    monkeypatch.setenv("VISIONLOOP_YOLOV9_REPO", "/from/env")
    f = tmp_path / "local_settings.json"
    # PowerShell 5 writes UTF-8 with a BOM
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps({"models_python": "C:/x/python.exe", "yolov9_repo": "/from/file"}).encode())
    try:
        config.apply_local_settings(f)
        assert os.environ["VISIONLOOP_MODELS_PYTHON"] == "C:/x/python.exe"
        assert os.environ["VISIONLOOP_YOLOV9_REPO"] == "/from/env"  # real env vars win
    finally:
        os.environ.pop("VISIONLOOP_MODELS_PYTHON", None)


@pytest.mark.skipif(os.name != "posix", reason="stub interpreter uses a shebang")
def test_probe_trains_on_all_confirmed_images(client, project, models_env):
    pid = project["id"]
    dog = ids_of(project)["dog"]
    created = upload(client, pid, [f"img{i}.jpg" for i in range(6)])["created"]
    r = client.post(f"/api/projects/{pid}/autolabel", json={"plugin": "probe"})
    assert r.status_code == 422  # nothing confirmed yet

    client.put(f"/api/images/{created[0]['id']}/annotations",
               json={"boxes": [{"class_id": dog, "x": 40, "y": 25, "w": 60, "h": 50}], "mark_reviewed": True})
    # a confirmed image without objects is a negative example
    client.put(f"/api/images/{created[1]['id']}/annotations", json={"boxes": [], "mark_reviewed": True})

    plugins = {p["id"]: p for p in client.get("/api/plugins").json()["plugins"]}
    assert plugins["probe"]["uses_reviewed_images"] and plugins["probe"]["available"]

    job = client.post(f"/api/projects/{pid}/autolabel",
                      json={"plugin": "probe", "variant": "dinov2-b", "probe_head": "mlp"}).json()
    job = wait_job(client, job["id"])
    detail = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "succeeded", detail["log"]
    assert "training on 2 images, 1 negatives" in detail["log"]
    assert job["result"]["labelled"] == 4
    s = client.get(f"/api/images/{created[5]['id']}/annotations").json()
    assert [(b["class_id"], b["score"]) for b in s["boxes"]] == [(dog, 0.77)]
