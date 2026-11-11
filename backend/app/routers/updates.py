"""Is a newer VisionLoop released on GitHub?

Asks GitHub for the latest release at most once a day. Only the version number is requested;
nothing about the user's projects is sent. Without a connection the answer is simply "unknown"
after a few seconds; the check never delays anything else. Turned off with
"check_updates": false in local_settings.json.

The desktop app can also install the update itself: see app/services/self_update.py.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import settings
from ..db import get_db
from ..models import Job
from ..services import self_update
from .common import is_local_request

router = APIRouter(prefix="/api", tags=["updates"])

REPO = "chuan92329/visionloop"
LATEST_URL = os.environ.get("VISIONLOOP_UPDATE_API") or f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases"
CHECK_EVERY = 24 * 3600  # seconds
TIMEOUT = 4  # seconds; an unreachable network must not keep the answer waiting
_lock = threading.Lock()


def version_tuple(text: str) -> tuple[int, ...] | None:
    m = re.match(r"^v?(\d+(?:\.\d+)*)$", text.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def is_newer(latest: str, current: str) -> bool:
    a, b = version_tuple(latest), version_tuple(current)
    return bool(a and b and a > b)


def _cache_file():
    return settings.data_dir / "update_check.json"


def _fetch() -> dict:
    req = urllib.request.Request(LATEST_URL, headers={"Accept": "application/vnd.github+json",
                                                      "User-Agent": f"VisionLoop/{__version__}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.loads(r.read())
    return {"latest": data.get("tag_name", ""), "url": data.get("html_url") or RELEASES_URL,
            "notes": (data.get("body") or "")[:2000], "checked_at": time.time(),
            "assets": [{"name": a.get("name", ""), "url": a.get("browser_download_url", "")}
                       for a in data.get("assets") or []]}


def latest_release(force: bool = False) -> dict | None:
    """The cached answer if it is less than a day old, otherwise a fresh one (None if unreachable)."""
    path = _cache_file()
    with _lock:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if cached and not force and time.time() - cached.get("checked_at", 0) < CHECK_EVERY:
            return cached
        try:
            fresh = _fetch()
        except Exception:  # noqa: BLE001 - offline, blocked or rate limited: try again next time
            return cached
        try:
            settings.ensure_dirs()
            path.write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return fresh


@router.get("/update-check")
def update_check():
    enabled = os.environ.get("VISIONLOOP_CHECK_UPDATES", "").lower() not in {"0", "false", "no", "off"}
    answer = {"current": __version__, "enabled": enabled, "latest": None, "newer": False,
              "url": RELEASES_URL, "notes": "", "can_install": can_install(),
              "install": self_update.downloader.status(settings.data_dir),
              "last": self_update.last_result(settings.data_dir)}
    if not enabled:
        return answer
    release = latest_release()
    if release:
        answer.update(latest=release["latest"], url=release["url"], notes=release.get("notes", ""),
                      newer=is_newer(release["latest"], __version__))
    return answer


# ------------------------------------------------------------- self-update
def can_install() -> bool:
    """Only the desktop app installed from a release zip can replace its own files."""
    return self_update.restart_hook is not None and self_update.installed_from_release()


def _only_here(request: Request) -> None:
    if not is_local_request(request):
        raise HTTPException(403, "只能在執行 VisionLoop 的這台電腦上更新")
    if not can_install():
        raise HTTPException(409, "這個安裝方式無法自動更新，請到 GitHub 下載新版")


@router.post("/update/download")
def update_download(request: Request):
    _only_here(request)
    release = latest_release(force=True)
    if not release or not is_newer(release["latest"], __version__):
        raise HTTPException(409, "目前沒有可以更新的新版本")
    return self_update.downloader.start(release, settings.data_dir)


@router.get("/update/status")
def update_status():
    return self_update.downloader.status(settings.data_dir)


@router.post("/update/restart")
def update_restart(request: Request, db: Session = Depends(get_db)):
    """Close the window and start VisionLoop again; the launcher installs the update first."""
    _only_here(request)
    busy = db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(("queued", "running"))))
    if busy:
        raise HTTPException(409, "還有任務在執行或排隊，請等它完成（或取消）後再更新")
    if self_update.downloader.status(settings.data_dir)["state"] != "ready":
        raise HTTPException(409, "更新檔還沒下載完成")
    hook = self_update.restart_hook
    # answer first: the server stops a moment later
    threading.Timer(0.5, hook).start()
    return {"restarting": True}


@router.post("/update/ack")
def update_ack():
    self_update.acknowledge(settings.data_dir)
    return {"ok": True}
