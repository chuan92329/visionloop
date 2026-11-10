"""Is a newer VisionLoop released on GitHub?

Asks GitHub for the latest release at most once a day. Only the version number is requested;
nothing about the user's projects is sent. Without a connection the answer is simply "unknown"
after a few seconds; the check never delays anything else. Turned off with
"check_updates": false in local_settings.json.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request

from fastapi import APIRouter

from .. import __version__
from ..config import settings

router = APIRouter(prefix="/api", tags=["updates"])

REPO = "chuan92329/visionloop"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
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
            "notes": (data.get("body") or "")[:2000], "checked_at": time.time()}


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
              "url": RELEASES_URL, "notes": ""}
    if not enabled:
        return answer
    release = latest_release()
    if release:
        answer.update(latest=release["latest"], url=release["url"], notes=release.get("notes", ""),
                      newer=is_newer(release["latest"], __version__))
    return answer
