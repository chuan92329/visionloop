"""Updating VisionLoop from inside the app.

Two halves, both standard library only (the desktop launcher uses the second half before the
server and its dependencies are loaded):

1. Download (in the running server): fetch the release zip and its .sha256 from GitHub, check the
   checksum and the zip's contents, and leave it in data/updates/ as a pending update.
2. Install (in scripts/desktop.py, while the window shows the splash and nothing is loaded yet):
   back up every program file the update touches, swap in the new files, check the new version
   can start (byte-compiles and imports), and only then restart into it.  If that check, the
   dependency install or the first start of the new version fails, the backup is put back.

What is never touched: data/ (projects, images, weights, runs), models/, tools/, envs/,
local_settings.json, frontend/node_modules and any file the release does not ship, such as
plugins people added themselves.  A program file is only removed when the previous release
shipped it (release_manifest.txt) and the new one no longer does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

PROGRAM_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "release_manifest.txt"
PROTECTED_DIRS = ("data", "models", "tools", "envs", ".git", "frontend/node_modules")
PROTECTED_FILES = ("local_settings.json",)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHUNK = 1 << 20

# set by scripts/desktop.py: closes the window, stops the server and starts VisionLoop again
restart_hook: Callable[[], None] | None = None


class UpdateError(Exception):
    pass


# ------------------------------------------------------------------ paths
def updates_dir(data_dir: Path) -> Path:
    return data_dir / "updates"


def pending_path(data_dir: Path) -> Path:
    return updates_dir(data_dir) / "pending.json"


def trial_path(data_dir: Path) -> Path:
    return updates_dir(data_dir) / "trial.json"


def result_path(data_dir: Path) -> Path:
    return updates_dir(data_dir) / "result.json"


def backup_dir(data_dir: Path) -> Path:
    return updates_dir(data_dir) / "backup"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def protected(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    first = rel.split("/", 1)[0]
    if first.startswith("_backup") or ("/" not in rel and rel.lower().endswith(".zip")):
        return True
    return rel in PROTECTED_FILES or any(rel == d or rel.startswith(d + "/") for d in PROTECTED_DIRS)


def installed_from_release(root: Path = PROGRAM_ROOT) -> bool:
    """A git checkout updates with git pull; only an unpacked release zip updates itself."""
    return (root / MANIFEST).is_file() and (root / "frontend" / "dist" / ".release").is_file() \
        and not (root / ".git").exists()


# ------------------------------------------------------------- zip checks
def _member_path(name: str, top: str) -> str | None:
    """The member's path inside the program folder, or None for the top folder itself.
    Anything that could land outside the program folder is refused."""
    p = PurePosixPath(name)
    if p.is_absolute() or ":" in name or "\\" in name or ".." in p.parts:
        raise UpdateError(f"更新檔裡有不安全的路徑：{name}")
    if not p.parts or p.parts[0] != top:
        raise UpdateError("更新檔的結構不對：應該只有一個最上層資料夾")
    rest = PurePosixPath(*p.parts[1:]) if len(p.parts) > 1 else None
    return str(rest) if rest else None


def inspect_zip(zip_path: Path, version: str) -> list[str]:
    """Check a release zip and return the files it ships (paths inside the program folder)."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdateError("更新檔不是有效的 zip") from exc
    with zf:
        names = zf.namelist()
        if not names:
            raise UpdateError("更新檔是空的")
        top = PurePosixPath(names[0]).parts[0]
        files = []
        for info in zf.infolist():
            rel = _member_path(info.filename, top)
            if rel and not info.is_dir():
                files.append(rel)
        if MANIFEST not in files or "backend/app/__init__.py" not in files:
            raise UpdateError("更新檔裡缺少必要的檔案")
        init = zf.read(f"{top}/backend/app/__init__.py").decode("utf-8", "replace")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
        if not m or m.group(1) != version.lstrip("v"):
            raise UpdateError(f"更新檔的版本和 {version} 不符")
        if zf.testzip() is not None:
            raise UpdateError("更新檔已損壞")
    return files


# --------------------------------------------------------------- download
def _assets(release: dict) -> tuple[str, str]:
    tag = release["latest"]
    by_name = {a["name"]: a["url"] for a in release.get("assets", [])}
    zip_url, sum_url = by_name.get(f"visionloop-{tag}.zip"), by_name.get(f"visionloop-{tag}.zip.sha256")
    if not zip_url or not sum_url:
        raise UpdateError("這個版本沒有提供自動更新用的檔案，請到 GitHub 手動下載")
    for url in (zip_url, sum_url):
        if not url.startswith("https://") and not os.environ.get("VISIONLOOP_UPDATE_API"):
            raise UpdateError("更新檔的網址不安全")
    return zip_url, sum_url


def _open(url: str, timeout: float = 30):
    from .. import __version__

    req = urllib.request.Request(url, headers={"User-Agent": f"VisionLoop/{__version__}"})
    return urllib.request.urlopen(req, timeout=timeout)


class Downloader:
    """One download at a time, in a background thread; the UI polls status()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {"state": "idle"}

    def status(self, data_dir: Path) -> dict:
        with self._lock:
            state = dict(self._state)
        if state["state"] in ("idle", "error"):
            pending = _read_json(pending_path(data_dir))
            if pending and Path(pending.get("zip", "")).is_file():
                return {"state": "ready", "version": pending["version"], "progress": 1.0}
        return state

    def start(self, release: dict, data_dir: Path) -> dict:
        with self._lock:
            if self._state["state"] == "downloading":
                return dict(self._state)
            self._state = {"state": "downloading", "version": release["latest"], "progress": 0.0}
        threading.Thread(target=self._run, args=(release, data_dir), daemon=True).start()
        return self.status(data_dir)

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def _run(self, release: dict, data_dir: Path) -> None:
        tag = release["latest"]
        folder = updates_dir(data_dir)
        part = folder / f"visionloop-{tag}.zip.part"
        try:
            zip_url, sum_url = _assets(release)
            folder.mkdir(parents=True, exist_ok=True)
            with _open(sum_url) as r:
                expected = r.read().decode("ascii", "replace").split()[0].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise UpdateError("更新檔的校驗碼格式不對")
            digest = hashlib.sha256()
            with _open(zip_url, timeout=60) as r, part.open("wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if total:
                        self._set(progress=min(done / total, 0.99))
            if digest.hexdigest() != expected:
                raise UpdateError("下載的檔案和校驗碼不符，可能下載不完整，請再試一次")
            inspect_zip(part, tag)
            final = folder / f"visionloop-{tag}.zip"
            os.replace(part, final)
            for old in folder.glob("visionloop-*.zip"):
                if old != final:
                    old.unlink(missing_ok=True)
            _write_json(pending_path(data_dir), {"version": tag, "zip": str(final), "sha256": expected})
            self._set(state="ready", progress=1.0)
        except Exception as exc:  # noqa: BLE001 - shown to the user, who can try again
            log.warning("update download failed: %s", exc)
            part.unlink(missing_ok=True)
            message = str(exc) if isinstance(exc, UpdateError) else "下載失敗，請確認網路連線後再試一次"
            self._set(state="error", message=message)


downloader = Downloader()


# ---------------------------------------------------------------- install
def _manifest(path: Path) -> set[str] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return {ln.strip().replace("\\", "/") for ln in lines if ln.strip()}


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _python_for_tools() -> str:
    """python.exe next to pythonw.exe: pip and the checks run hidden either way."""
    exe = Path(sys.executable)
    console = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
    return str(console if console.exists() else exe)


def _run(cmd: list[str], cwd: Path, logfile: Path, timeout: float, env: dict | None = None) -> bool:
    with logfile.open("a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(cmd)}\n")
        out.flush()
        try:
            r = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout,
                               creationflags=NO_WINDOW, env={**os.environ, **(env or {}), "PYTHONUTF8": "1"})
        except (OSError, subprocess.SubprocessError) as exc:
            out.write(f"{exc}\n")
            return False
    return r.returncode == 0


def check_can_start(root: Path, version: str, scratch: Path, logfile: Path) -> bool:
    """The new files byte-compile and the server imports, as the same Python will run them."""
    py = _python_for_tools()
    if not _run([py, "-X", "utf8", "-m", "compileall", "-q", "backend/app", "scripts"], root, logfile, 300):
        return False
    probe = ("import sys; sys.path.insert(0, 'backend'); import app, app.main, app.services.self_update; "
             f"sys.exit(0 if app.__version__ == {version.lstrip('v')!r} else 3)")
    scratch.mkdir(parents=True, exist_ok=True)
    return _run([py, "-X", "utf8", "-c", probe], root, logfile, 300,
                env={"VISIONLOOP_DATA_DIR": str(scratch), "VISIONLOOP_CHECK_UPDATES": "false"})


def install(root: Path, zip_path: Path, version: str, data_dir: Path, logfile: Path) -> None:
    """Swap in the new program files.  Raises UpdateError after putting everything back."""
    files = inspect_zip(zip_path, version)
    work = updates_dir(data_dir)
    staging, backup = work / "staging", backup_dir(data_dir)
    for d in (staging, backup):
        shutil.rmtree(d, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        top = PurePosixPath(zf.namelist()[0]).parts[0]
        zf.extractall(staging)
    src = staging / top
    new = {f for f in (_manifest(src / MANIFEST) or set(files)) if not protected(f)}
    missing = [f for f in new if not (src / f).is_file()]
    if missing:
        raise UpdateError(f"更新檔缺少 {missing[0]}")
    old = _manifest(root / MANIFEST) or set()
    remove = {f for f in old - new if not protected(f)}

    from .. import __version__ as current

    record = {"from": current, "to": version, "replaced": [], "created": []}
    for rel in sorted(new | remove):
        target = root / rel
        if target.is_file():
            dest = backup / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, dest)
            record["replaced"].append(rel)
        elif rel in new:
            record["created"].append(rel)
    _write_json(backup / "record.json", record)  # written before anything changes

    try:
        for rel in sorted(new):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, target)
        for rel in sorted(remove):
            (root / rel).unlink(missing_ok=True)
        old_req = backup / "files" / "backend" / "requirements.txt"
        new_req = root / "backend" / "requirements.txt"
        if not old_req.is_file() or _lines(old_req) != _lines(new_req):
            if not _run([_python_for_tools(), "-m", "pip", "install", "-r", str(new_req)], root, logfile, 1800):
                raise UpdateError("安裝新版需要的套件失敗")
        if not check_can_start(root, version, work / "check", logfile):
            raise UpdateError("新版本無法啟動")
    except Exception as exc:
        rollback(root, data_dir)
        raise exc if isinstance(exc, UpdateError) else UpdateError(f"更新失敗：{exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(work / "check", ignore_errors=True)


def rollback(root: Path, data_dir: Path) -> bool:
    """Put back the program files the last update replaced; remove the ones it added."""
    backup = backup_dir(data_dir)
    record = _read_json(backup / "record.json")
    if record is None:
        return False
    for rel in record["replaced"]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup / "files" / rel, target)
    for rel in record["created"]:
        (root / rel).unlink(missing_ok=True)
    return True


def _clear_pending(data_dir: Path) -> None:
    pending = _read_json(pending_path(data_dir))
    if pending:
        Path(pending.get("zip", "")).unlink(missing_ok=True)
    pending_path(data_dir).unlink(missing_ok=True)


def install_pending(root: Path, data_dir: Path, logfile: Path) -> bool:
    """Run by the launcher before anything is loaded.  True: the new version is in place and
    VisionLoop must start again to run it.  False: nothing to do, or it failed and the current
    version stays (the reason is kept for the UI)."""
    pending = _read_json(pending_path(data_dir))
    if not pending:
        return False
    version = pending.get("version", "")
    try:
        if not installed_from_release(root):
            raise UpdateError("這個資料夾不是從發佈的 zip 安裝的，無法自動更新")
        install(root, Path(pending["zip"]), version, data_dir, logfile)
    except Exception as exc:  # noqa: BLE001
        log.warning("update to %s failed: %s", version, exc)
        message = str(exc) if isinstance(exc, UpdateError) else f"更新失敗：{exc}"
        _write_json(result_path(data_dir), {"ok": False, "version": version, "message": message})
        _clear_pending(data_dir)
        return False
    _clear_pending(data_dir)
    _write_json(trial_path(data_dir), {"version": version, "at": time.time()})
    return True


def confirm_started(data_dir: Path) -> None:
    """The new version started: the update is done."""
    trial = _read_json(trial_path(data_dir))
    if trial is None:
        return
    _write_json(result_path(data_dir), {"ok": True, "version": trial["version"]})
    trial_path(data_dir).unlink(missing_ok=True)


def undo_failed_start(root: Path, data_dir: Path, reason: str) -> bool:
    """The first start after an update failed: go back.  True when VisionLoop must start again."""
    trial = _read_json(trial_path(data_dir))
    if trial is None:
        return False
    trial_path(data_dir).unlink(missing_ok=True)
    restored = rollback(root, data_dir)
    _write_json(result_path(data_dir), {"ok": False, "version": trial["version"],
                                        "message": f"新版本啟動失敗，已還原：{re.sub(r'<[^>]+>', '', reason)}"})
    return restored


def last_result(data_dir: Path) -> dict | None:
    return _read_json(result_path(data_dir))


def acknowledge(data_dir: Path) -> None:
    result_path(data_dir).unlink(missing_ok=True)
