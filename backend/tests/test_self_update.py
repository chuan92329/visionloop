"""Updating from inside the app: download checks, swapping program files, and going back."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from app import __version__
from app.config import settings
from app.routers import updates
from app.services import self_update as su


# ------------------------------------------------------------- fixtures
def write_install(root: Path, files: dict[str, str]) -> None:
    """A program folder as unpacked from a release zip."""
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    (root / "frontend" / "dist").mkdir(parents=True, exist_ok=True)
    (root / "frontend" / "dist" / ".release").write_text("v0.0.1")
    listed = sorted({*files, "frontend/dist/.release", su.MANIFEST})
    (root / su.MANIFEST).write_text("\n".join(listed) + "\n", encoding="utf-8")


def make_release(path: Path, version: str, files: dict[str, str], manifest: bool = True) -> Path:
    top = f"visionloop-{version}"
    files = {**files, "backend/app/__init__.py": f'__version__ = "{version.lstrip("v")}"\n',
             "frontend/dist/.release": version}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/", "")
        for rel, text in files.items():
            zf.writestr(f"{top}/{rel}", text)
        if manifest:
            zf.writestr(f"{top}/{su.MANIFEST}", "\n".join(sorted({*files, su.MANIFEST})) + "\n")
    return path


OLD = {
    "backend/app/__init__.py": '__version__ = "0.0.1"\n',
    "backend/app/main.py": "old main\n",
    "backend/requirements.txt": "fastapi\n",
    "scripts/gone.py": "shipped before, not any more\n",
}


@pytest.fixture()
def install(tmp_path, monkeypatch):
    root, data = tmp_path / "vl", tmp_path / "vl" / "data"
    write_install(root, OLD)
    # what belongs to the user, never shipped by a release
    (data / "projects").mkdir(parents=True)
    (data / "projects" / "p.txt").write_text("my project")
    (root / "local_settings.json").write_text("{}")
    (root / "models" / "yolov9").mkdir(parents=True)
    (root / "models" / "yolov9" / "train.py").write_text("their repo")
    (root / "backend" / "plugins").mkdir(parents=True)
    (root / "backend" / "plugins" / "mine.py").write_text("my plugin")
    monkeypatch.setattr(su, "check_can_start", lambda *a: True)
    pip = []
    monkeypatch.setattr(su, "_run", lambda cmd, *a, **k: pip.append(cmd) or True)
    return root, data, pip


NEW = {
    "backend/app/main.py": "new main\n",
    "backend/requirements.txt": "fastapi\n",
    "scripts/added.py": "new file\n",
}


def user_files_untouched(root: Path, data: Path) -> None:
    assert (data / "projects" / "p.txt").read_text() == "my project"
    assert (root / "local_settings.json").read_text() == "{}"
    assert (root / "models" / "yolov9" / "train.py").read_text() == "their repo"
    assert (root / "backend" / "plugins" / "mine.py").read_text() == "my plugin"


def pending(data: Path, zip_path: Path, version: str) -> None:
    su._write_json(su.pending_path(data), {"version": version, "zip": str(zip_path), "sha256": "x"})


# -------------------------------------------------------------- install
def test_installs_the_new_files_and_keeps_the_users(install, tmp_path):
    root, data, pip = install
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", NEW), "v0.0.2")

    assert su.install_pending(root, data, data / "log.txt") is True

    assert (root / "backend" / "app" / "main.py").read_text() == "new main\n"
    assert '"0.0.2"' in (root / "backend" / "app" / "__init__.py").read_text()
    assert (root / "scripts" / "added.py").exists()
    assert not (root / "scripts" / "gone.py").exists()  # the old release shipped it, the new one not
    user_files_untouched(root, data)
    assert pip == []  # requirements unchanged: nothing to install
    assert not su.pending_path(data).exists() and not (tmp_path / "u.zip").exists()
    assert su.trial_path(data).exists()


def test_changed_requirements_are_installed(install, tmp_path):
    root, data, pip = install
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", {**NEW, "backend/requirements.txt": "fastapi\nnew\n"}),
            "v0.0.2")
    assert su.install_pending(root, data, data / "log.txt") is True
    assert len(pip) == 1 and pip[0][1:4] == ["-m", "pip", "install"]


def test_failed_check_puts_everything_back(install, tmp_path, monkeypatch):
    root, data, _ = install
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and "data" not in p.parts}
    monkeypatch.setattr(su, "check_can_start", lambda *a: False)
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", NEW), "v0.0.2")

    assert su.install_pending(root, data, data / "log.txt") is False

    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and "data" not in p.parts}
    assert after == before
    result = su.last_result(data)
    assert result["ok"] is False and "無法啟動" in result["message"]
    assert not su.pending_path(data).exists()  # not tried again on every start


def test_failed_pip_puts_everything_back(install, tmp_path, monkeypatch):
    root, data, _ = install
    monkeypatch.setattr(su, "_run", lambda *a, **k: False)
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", {**NEW, "backend/requirements.txt": "x\n"}), "v0.0.2")
    assert su.install_pending(root, data, data / "log.txt") is False
    assert (root / "backend" / "app" / "main.py").read_text() == "old main\n"
    assert (root / "scripts" / "gone.py").exists() and not (root / "scripts" / "added.py").exists()


def test_failed_first_start_goes_back(install, tmp_path):
    root, data, _ = install
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", NEW), "v0.0.2")
    assert su.install_pending(root, data, data / "log.txt")

    assert su.undo_failed_start(root, data, "伺服器沒有啟動，記錄在 <code>log</code>。") is True

    assert (root / "backend" / "app" / "main.py").read_text() == "old main\n"
    assert '"0.0.1"' in (root / "backend" / "app" / "__init__.py").read_text()
    assert (root / "scripts" / "gone.py").exists() and not (root / "scripts" / "added.py").exists()
    user_files_untouched(root, data)
    result = su.last_result(data)
    assert result["ok"] is False and "<code>" not in result["message"] and "記錄在 log" in result["message"]
    assert su.undo_failed_start(root, data, "again") is False  # only once


def test_successful_start_finishes_the_update(install, tmp_path):
    root, data, _ = install
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", NEW), "v0.0.2")
    su.install_pending(root, data, data / "log.txt")
    su.confirm_started(data)
    assert su.last_result(data) == {"ok": True, "version": "v0.0.2"}
    assert su.undo_failed_start(root, data, "later") is False  # a later failure is not the update's
    su.acknowledge(data)
    assert su.last_result(data) is None


def test_release_cannot_write_over_user_files(install, tmp_path):
    root, data, _ = install
    evil = {**NEW, "local_settings.json": "{\"x\": 1}", "data/projects/p.txt": "gone", "models/yolov9/train.py": "x"}
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", evil), "v0.0.2")
    assert su.install_pending(root, data, data / "log.txt") is True
    user_files_untouched(root, data)


def test_git_checkout_is_never_updated(install, tmp_path):
    root, data, _ = install
    (root / ".git").mkdir()
    pending(data, make_release(tmp_path / "u.zip", "v0.0.2", NEW), "v0.0.2")
    assert su.install_pending(root, data, data / "log.txt") is False
    assert (root / "backend" / "app" / "main.py").read_text() == "old main\n"
    assert "zip" in su.last_result(data)["message"]


# ------------------------------------------------------------ zip checks
@pytest.mark.parametrize("name", ["../evil.py", "/abs.py", "C:/x.py", "visionloop-v0.0.2/../../x.py"])
def test_unsafe_paths_are_refused(tmp_path, name):
    z = make_release(tmp_path / "u.zip", "v0.0.2", NEW)
    with zipfile.ZipFile(z, "a") as zf:
        zf.writestr(name, "x")
    with pytest.raises(su.UpdateError):
        su.inspect_zip(z, "v0.0.2")


def test_zip_must_hold_the_version_it_claims(tmp_path):
    z = make_release(tmp_path / "u.zip", "v0.0.2", NEW)
    with pytest.raises(su.UpdateError, match="版本"):
        su.inspect_zip(z, "v0.0.3")
    with pytest.raises(su.UpdateError):
        su.inspect_zip(make_release(tmp_path / "n.zip", "v0.0.2", NEW, manifest=False), "v0.0.2")
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    with pytest.raises(su.UpdateError):
        su.inspect_zip(tmp_path / "bad.zip", "v0.0.2")


# --------------------------------------------------------------- download
@pytest.fixture()
def server(tmp_path):
    """Serves files the way GitHub serves release assets."""
    files: dict[str, bytes] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = files.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield files, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def release_for(files, base, tmp_path, version="v0.0.2", checksum=None):
    data = make_release(tmp_path / "r.zip", version, NEW).read_bytes()
    files[f"/{version}.zip"] = data
    files[f"/{version}.zip.sha256"] = f"{checksum or hashlib.sha256(data).hexdigest()}  x.zip\n".encode()
    return {"latest": version, "assets": [
        {"name": f"visionloop-{version}.zip", "url": f"{base}/{version}.zip"},
        {"name": f"visionloop-{version}.zip.sha256", "url": f"{base}/{version}.zip.sha256"}]}


def wait_done(d, data):
    end = time.time() + 20
    while d.status(data)["state"] == "downloading" and time.time() < end:
        time.sleep(0.05)
    return d.status(data)


def test_download_checks_the_checksum_and_leaves_a_pending_update(server, tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONLOOP_UPDATE_API", "test")
    files, base = server
    d, data = su.Downloader(), tmp_path / "data"
    d.start(release_for(files, base, tmp_path), data)
    st = wait_done(d, data)
    assert st["state"] == "ready", st
    p = json.loads(su.pending_path(data).read_text(encoding="utf-8"))
    assert p["version"] == "v0.0.2" and Path(p["zip"]).is_file()
    assert su.Downloader().status(data)["state"] == "ready"  # also after the server restarts


def test_download_with_wrong_checksum_is_thrown_away(server, tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONLOOP_UPDATE_API", "test")
    files, base = server
    d, data = su.Downloader(), tmp_path / "data"
    d.start(release_for(files, base, tmp_path, checksum="0" * 64), data)
    st = wait_done(d, data)
    assert st["state"] == "error" and "校驗碼" in st["message"]
    assert not su.pending_path(data).exists() and not list(su.updates_dir(data).glob("*.zip*"))


def test_release_without_update_files_says_so(tmp_path):
    d, data = su.Downloader(), tmp_path / "data"
    d.start({"latest": "v0.0.2", "assets": []}, data)
    st = wait_done(d, data)
    assert st["state"] == "error" and "手動下載" in st["message"]


def test_plain_http_is_refused_outside_tests(server, tmp_path, monkeypatch):
    monkeypatch.delenv("VISIONLOOP_UPDATE_API", raising=False)
    files, base = server
    d, data = su.Downloader(), tmp_path / "data"
    d.start(release_for(files, base, tmp_path), data)
    st = wait_done(d, data)
    assert st["state"] == "error" and "不安全" in st["message"]


# -------------------------------------------------------------------- API
def test_only_the_desktop_app_from_a_release_can_install(client, monkeypatch):
    monkeypatch.setenv("VISIONLOOP_CHECK_UPDATES", "false")
    monkeypatch.setattr(su, "restart_hook", None)
    r = client.get("/api/update-check").json()
    assert r["can_install"] is False and r["install"]["state"] == "idle"
    assert client.post("/api/update/download").status_code == 409
    assert client.post("/api/update/restart").status_code == 409


def test_restart_waits_for_running_jobs_and_a_finished_download(client, project, monkeypatch):
    from app.db import SessionLocal
    from app.models import Job

    calls = []
    monkeypatch.setattr(su, "restart_hook", lambda: calls.append(1))
    monkeypatch.setattr(su, "installed_from_release", lambda root=None: True)
    assert client.post("/api/update/restart").status_code == 409  # nothing downloaded yet

    z = make_release(settings.data_dir / "u.zip", "v99.0.0", NEW)
    pending(settings.data_dir, z, "v99.0.0")
    with SessionLocal() as db:
        db.add(Job(project_id=project["id"], kind="train", plugin="demo", status="running"))
        db.commit()
    r = client.post("/api/update/restart")
    assert r.status_code == 409 and "任務" in r.json()["detail"]
    with SessionLocal() as db:
        for j in db.query(Job).all():
            j.status = "succeeded"
        db.commit()
    assert client.post("/api/update/restart").json() == {"restarting": True}
    time.sleep(1)
    assert calls == [1]


def test_only_this_computer_may_install(client, monkeypatch):
    monkeypatch.setattr(updates, "is_local_request", lambda request: False)
    assert client.post("/api/update/download").status_code == 403
    assert client.post("/api/update/restart").status_code == 403


def test_last_result_is_reported_once(client, monkeypatch):
    monkeypatch.setenv("VISIONLOOP_CHECK_UPDATES", "false")
    su._write_json(su.result_path(settings.data_dir), {"ok": True, "version": "v" + __version__})
    assert client.get("/api/update-check").json()["last"]["ok"] is True
    client.post("/api/update/ack")
    assert client.get("/api/update-check").json()["last"] is None
