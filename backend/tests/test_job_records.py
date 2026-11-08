"""Deleting a job's record must not take its weights or annotations with it."""

from __future__ import annotations

from app.config import settings
from app.db import SessionLocal
from app.models import Job, ModelWeights
from app.services.jobs import new_work_dir, next_job_id


def add_job(project_id: int, status: str = "succeeded", kind: str = "train", folder: bool = True) -> int:
    with SessionLocal() as db:
        job = Job(project_id=project_id, kind=kind, plugin="demo", status=status)
        db.add(job)
        db.commit()
        job_id = job.id
    if folder:
        run = settings.runs_dir / f"job_{job_id:05d}"
        (run / "output").mkdir(parents=True)
        (run / "output" / "log.txt").write_text("x")
    return job_id


def add_weights(project_id: int, job_id: int):
    path = settings.runs_dir / f"job_{job_id:05d}" / "output" / "best.pt"
    path.write_text("weights")
    with SessionLocal() as db:
        w = ModelWeights(project_id=project_id, plugin="demo", variant="", name="w", path=str(path.resolve()),
                         class_names=["a"], metrics={}, job_id=job_id)
        db.add(w)
        db.commit()
        return w.id, path


def test_deleting_a_job_keeps_the_weights_it_trained(client, project):
    pid = project["id"]
    job_id = add_job(pid)
    weights_id, path = add_weights(pid, job_id)

    assert client.delete(f"/api/jobs/{job_id}").status_code == 204

    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert path.read_text() == "weights"
    assert [w["id"] for w in client.get(f"/api/projects/{pid}/weights").json()] == [weights_id]


def test_deleting_a_job_without_weights_removes_its_folder(client, project):
    job_id = add_job(project["id"], status="failed")
    run = settings.runs_dir / f"job_{job_id:05d}"
    assert run.exists()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert not run.exists()


def test_only_the_jobs_own_folder_is_removed(client, project):
    pid = project["id"]
    first, second = add_job(pid), add_job(pid)
    assert client.delete(f"/api/jobs/{first}").status_code == 204
    assert (settings.runs_dir / f"job_{second:05d}").exists()


def test_a_job_that_is_still_going_cannot_be_deleted(client, project):
    for status in ("queued", "running"):
        job_id = add_job(project["id"], status=status, folder=False)
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409
        assert client.get(f"/api/jobs/{job_id}").status_code == 200
        with SessionLocal() as db:  # do not leave anything for the runner
            db.get(Job, job_id).status = "cancelled"
            db.commit()


def test_clear_removes_only_the_chosen_statuses_of_this_project(client, project):
    pid = project["id"]
    other = client.post("/api/projects", json={"name": "other", "classes": ["a"]}).json()["id"]
    failed, cancelled, done = add_job(pid, "failed"), add_job(pid, "cancelled"), add_job(pid, "succeeded")
    elsewhere = add_job(other, "failed")

    r = client.post(f"/api/projects/{pid}/clear-jobs", json={"statuses": ["failed", "cancelled"]})
    assert r.json() == {"deleted": 2}

    assert client.get(f"/api/jobs/{failed}").status_code == 404
    assert client.get(f"/api/jobs/{cancelled}").status_code == 404
    assert client.get(f"/api/jobs/{done}").status_code == 200
    assert client.get(f"/api/jobs/{elsewhere}").status_code == 200


def test_clear_refuses_jobs_that_are_not_finished(client, project):
    r = client.post(f"/api/projects/{project['id']}/clear-jobs", json={"statuses": ["running"]})
    assert r.status_code == 422


def test_an_id_that_weights_refer_to_is_not_given_out_again(client, project):
    pid = project["id"]
    job_id = add_job(pid)
    add_weights(pid, job_id)
    client.delete(f"/api/jobs/{job_id}")
    with SessionLocal() as db:
        assert next_job_id(db) > job_id


def test_a_new_job_does_not_write_into_an_old_folder(client, project):
    job_id = add_job(project["id"])
    old = settings.runs_dir / f"job_{job_id:05d}"
    fresh = new_work_dir(job_id)
    assert fresh != old
    assert fresh.is_dir()
    assert (old / "output" / "log.txt").exists()
