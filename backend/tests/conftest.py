from __future__ import annotations

import io
import logging
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import settings
from app.plugins import registry

from .demo_plugin import DemoPlugin

logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture()
def client(tmp_path):
    settings.data_dir = tmp_path / "data"
    settings.plugin_dir = tmp_path / "plugins"
    settings.plugin_dir.mkdir()
    from app.main import app

    with TestClient(app) as c:
        registry._register(DemoPlugin(), "tests")  # a fake model, so jobs run without a GPU
        yield c


def make_jpeg(w: int = 200, h: int = 100, color=(120, 30, 30)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), color)
    # one bright object so the demo plugin's blob detector finds something
    ImageDraw.Draw(img).rectangle([w // 5, h // 4, w // 2, h * 3 // 4], fill=(240, 240, 240))
    img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def project(client):
    r = client.post("/api/projects", json={"name": "demo", "classes": ["cat", "dog", "bird"]})
    assert r.status_code == 201
    return r.json()


def upload(client, project_id: int, names: list[str], **form):
    files = [
        ("files", (n, make_jpeg(color=(i * 7 % 255, i * 13 % 255, 40)), "image/jpeg"))
        for i, n in enumerate(names)
    ]
    r = client.post(f"/api/projects/{project_id}/images", files=files, data=form)
    assert r.status_code == 201, r.text
    return r.json()


def wait_job(client, job_id: int, timeout: float = 20) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish: {job}")
