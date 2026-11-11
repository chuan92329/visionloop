"""The "open training folder" button: it shows the run folder holding the training charts."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import settings
from app.db import SessionLocal
from app.models import ModelWeights
from app.routers import jobs as jobs_router
from app.routers.common import is_local_request

from .test_job_records import add_job


def add_weights(project_id: int, path, job_id: int | None) -> int:
    with SessionLocal() as db:
        w = ModelWeights(project_id=project_id, plugin="demo", variant="", name="w", path=str(path.resolve()),
                         class_names=["a"], metrics={}, job_id=job_id)
        db.add(w)
        db.commit()
        return w.id


def yolo_run(job_id: int):
    """The layout YOLOv9 writes: charts in output/train, weights in output/train/weights."""
    train = settings.runs_dir / f"job_{job_id:05d}" / "output" / "train"
    (train / "weights").mkdir(parents=True)
    (train / "results.png").write_bytes(b"png")
    (train / "weights" / "best.pt").write_text("weights")
    return train


def test_opens_the_folder_with_the_charts(client, project, monkeypatch):
    opened = []
    monkeypatch.setattr(jobs_router, "_reveal", opened.append)
    job_id = add_job(project["id"], folder=False)
    train = yolo_run(job_id)
    wid = add_weights(project["id"], train / "weights" / "best.pt", job_id)

    r = client.post(f"/api/weights/{wid}/open-folder")
    assert r.status_code == 200, r.text
    assert r.json() == {"path": str(train.resolve()), "opened": True}
    assert opened == [train.resolve()]


def test_weights_beside_the_logs_open_their_own_folder(client, project, monkeypatch):
    opened = []
    monkeypatch.setattr(jobs_router, "_reveal", opened.append)
    job_id = add_job(project["id"])
    out = settings.runs_dir / f"job_{job_id:05d}" / "output"
    (out / "best.pt").write_text("weights")
    wid = add_weights(project["id"], out / "best.pt", job_id)

    assert client.post(f"/api/weights/{wid}/open-folder").json()["path"] == str(out.resolve())
    assert opened == [out.resolve()]


def test_imported_weights_have_no_training_folder(client, project, monkeypatch):
    monkeypatch.setattr(jobs_router, "_reveal", lambda f: (_ for _ in ()).throw(AssertionError("opened")))
    folder = settings.data_dir / "weights" / str(project["id"])
    folder.mkdir(parents=True)
    (folder / "x.pt").write_text("weights")
    wid = add_weights(project["id"], folder / "x.pt", None)
    assert client.post(f"/api/weights/{wid}/open-folder").status_code == 404
    assert client.post("/api/weights/9999/open-folder").status_code == 404


def test_never_opens_a_folder_outside_the_data_folder(client, project, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_router, "_reveal", lambda f: (_ for _ in ()).throw(AssertionError("opened")))
    elsewhere = tmp_path / "elsewhere" / "weights"
    elsewhere.mkdir(parents=True)
    (elsewhere / "best.pt").write_text("weights")
    wid = add_weights(project["id"], elsewhere / "best.pt", add_job(project["id"]))
    assert client.post(f"/api/weights/{wid}/open-folder").status_code == 404


def test_a_browser_on_another_computer_only_gets_the_path(client, project, monkeypatch):
    monkeypatch.setattr(jobs_router, "_reveal", lambda f: (_ for _ in ()).throw(AssertionError("opened")))
    monkeypatch.setattr(jobs_router, "is_local_request", lambda request: False)
    job_id = add_job(project["id"], folder=False)
    train = yolo_run(job_id)
    wid = add_weights(project["id"], train / "weights" / "best.pt", job_id)
    assert client.post(f"/api/weights/{wid}/open-folder").json() == {"path": str(train.resolve()), "opened": False}


def test_a_missing_file_manager_still_returns_the_path(client, project, monkeypatch):
    def fail(folder):
        raise OSError("no file manager")

    monkeypatch.setattr(jobs_router, "_reveal", fail)
    job_id = add_job(project["id"], folder=False)
    train = yolo_run(job_id)
    wid = add_weights(project["id"], train / "weights" / "best.pt", job_id)
    assert client.post(f"/api/weights/{wid}/open-folder").json()["opened"] is False


def test_only_loopback_counts_as_this_computer():
    def req(host):
        return SimpleNamespace(client=SimpleNamespace(host=host))

    assert is_local_request(req("127.0.0.1"))
    assert is_local_request(req("::1"))
    assert not is_local_request(req("192.168.1.20"))
    assert not is_local_request(req("172.17.0.1"))
    assert not is_local_request(SimpleNamespace(client=None))
