"""Runtime configuration, read from environment variables.

All data (SQLite database, uploaded images, thumbnails, training runs) lives
under one data directory so the whole workspace can be backed up or moved by
copying a single folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SETTINGS = REPO_ROOT / "local_settings.json"

# keys of local_settings.json (written by scripts/setup_windows.ps1) -> environment variables
_LOCAL_KEYS = {
    "models_python": "VISIONLOOP_MODELS_PYTHON",
    "yolov9_repo": "VISIONLOOP_YOLOV9_REPO",
    "yolov9_python": "VISIONLOOP_YOLOV9_PYTHON",
    "rtdetr_repo": "VISIONLOOP_RTDETR_REPO",
    "rtdetr_python": "VISIONLOOP_RTDETR_PYTHON",
    "data_dir": "VISIONLOOP_DATA_DIR",
    "hf_home": "HF_HOME",
    # personal opt-in, not part of the published defaults: shows the C-RADIOv4 Probe backbones
    "experimental_backbones": "VISIONLOOP_EXPERIMENTAL_BACKBONES",
    # false: never ask GitHub whether a newer version exists
    "check_updates": "VISIONLOOP_CHECK_UPDATES",
}


def apply_local_settings(path: Path = LOCAL_SETTINGS) -> dict:
    """Machine-specific paths live in local_settings.json (git-ignored) so nobody has to
    set environment variables before every start. Real environment variables still win."""
    import json

    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    for key, env in _LOCAL_KEYS.items():
        value = data.get(key)
        if value is not None and value != "" and not os.environ.get(env):
            # booleans as "true" / "false" (a false switch must be passed on, not dropped)
            os.environ[env] = str(value).lower() if isinstance(value, bool) else str(value)
    return data


apply_local_settings()


def _default_data_dir() -> Path:
    # <repo>/data when running from source
    return Path(__file__).resolve().parents[2] / "data"


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("VISIONLOOP_DATA_DIR", _default_data_dir()))
    )
    # Extra folder scanned for user-written model plugins (*.py)
    plugin_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("VISIONLOOP_PLUGIN_DIR", Path(__file__).resolve().parents[1] / "plugins")
        )
    )
    # Built frontend served by FastAPI in production mode
    frontend_dist: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "VISIONLOOP_FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"
            )
        )
    )
    thumbnail_size: int = 320

    @property
    def db_path(self) -> Path:
        return self.data_dir / "visionloop.sqlite3"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.projects_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
