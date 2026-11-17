"""A model's labeling job keeps its predictions like YOLO's own predict: drawn images and
plain YOLO txt files (no scores), until the job's record is deleted."""

import io
import zipfile

from PIL import Image as PILImage

from app.config import settings
from app.services.predictions import PredictedBox, PredictionWriter, predict_folder, yolo_lines

from .conftest import upload, wait_job


def test_yolo_lines_have_five_columns_and_no_score():
    lines = yolo_lines([PredictedBox(1, 10, 20, 40, 30, 0.87)], 200, 100)
    assert lines == ["1 0.150000 0.350000 0.200000 0.300000"]


def test_writer_draws_boxes_and_writes_labels(tmp_path):
    src = tmp_path / "a.jpg"
    PILImage.new("RGB", (200, 100), "white").save(src)
    w = PredictionWriter(tmp_path / "job", ["螺絲", "nut"], ["#ff0000", "#00ff00"])
    w.add(src, "000001_a", [PredictedBox(0, 10, 20, 40, 30, 0.5), PredictedBox(1, 100, 50, 50, 40, 0.9)])
    w.add(src, "000002_b", [])

    folder = tmp_path / "job" / "predict"
    assert (folder / "classes.txt").read_text(encoding="utf-8") == "螺絲\nnut\n"
    assert (folder / "labels" / "000001_a.txt").read_text().splitlines()[1].split()[0] == "1"
    assert all(len(l.split()) == 5 for l in (folder / "labels" / "000001_a.txt").read_text().splitlines())
    assert (folder / "labels" / "000002_b.txt").read_text() == ""
    drawn = PILImage.open(folder / "images" / "000001_a.jpg")
    assert drawn.size == (200, 100)
    assert drawn.getpixel((10, 40))[0] > 200 and drawn.getpixel((10, 40))[1] < 80  # the red box edge
    assert PILImage.open(folder / "images" / "000002_b.jpg").getpixel((100, 50)) == (255, 255, 255)
    assert w.count == 2


def test_labeling_job_keeps_predictions_until_its_record_goes(client, project):
    pid = project["id"]
    upload(client, pid, [f"img{i}.jpg" for i in range(4)])
    job = wait_job(client, client.post(f"/api/projects/{pid}/autolabel", json={"plugin": "demo", "confidence": 0.0}).json()["id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["predicted_images"] == 4

    folder = predict_folder(job["result"])
    assert folder is not None
    assert sorted(p.name for p in folder.iterdir()) == ["classes.txt", "images", "labels"]
    assert len(list((folder / "images").glob("*.jpg"))) == 4
    # nothing else of the job's scratch files is left
    assert [p.name for p in folder.parent.iterdir()] == ["predict"]
    class_count = len(project["classes"])
    for txt in (folder / "labels").glob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5 and 0 <= int(parts[0]) < class_count

    r = client.get(f"/api/jobs/{job['id']}/predictions.zip")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "classes.txt" in names and sum(n.startswith("images/") for n in names) == 4
    assert sum(n.startswith("labels/") for n in names) == 4

    opened = client.post(f"/api/jobs/{job['id']}/open-predictions")
    assert opened.status_code == 200 and opened.json()["path"] == str(folder)

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert not folder.parent.exists()


def test_job_without_predictions_answers_404(client, project):
    pid = project["id"]
    upload(client, pid, ["a.jpg"])
    job = wait_job(client, client.post(f"/api/projects/{pid}/autolabel",
                                       json={"plugin": "demo", "statuses": ["auto"]}).json()["id"])
    assert "predict_dir" not in job["result"]  # nothing matched, nothing ran
    assert client.get(f"/api/jobs/{job['id']}/predictions.zip").status_code == 404
    assert client.get("/api/jobs/99999/predictions.zip").status_code == 404


def test_predict_folder_rejects_paths_outside_runs(client):
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    for bad in ("..", "../x", "a/b", "a\\b", ""):
        assert predict_folder({"predict_dir": bad}) is None
    assert predict_folder(None) is None
