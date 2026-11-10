"""Load a Hugging Face model from the local cache without asking the Hub first.

By default transformers checks huggingface.co for a newer version on every load, even when the model
is already downloaded. Without a connection that check only gives up after a network timeout
(more than a minute; with retries several minutes), so click-to-segment and the other tools seemed
to do nothing. ``local_files_only`` alone is not enough: some loads (e.g. the optional adapter
config of AutoModelForZeroShotObjectDetection) still ask the Hub. So while a downloaded model is
loaded the whole Hub client is put in offline mode; only a model that is not downloaded yet goes
to the Hub.

Imported by the helper scripts, which run in the model environment.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def _cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def downloaded(model_id: str) -> bool:
    snapshots = _cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.glob("*/config.json"))


@contextmanager
def offline():
    """The Hub client in offline mode for the duration (it reads the flag at call time)."""
    try:
        import huggingface_hub.constants as hub
    except ImportError:  # not installed (only the environment variable matters then)
        hub = None
    before_env = os.environ.get("HF_HUB_OFFLINE")
    before_flag = hub.HF_HUB_OFFLINE if hub else None
    os.environ["HF_HUB_OFFLINE"] = "1"
    if hub:
        hub.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        if hub:
            hub.HF_HUB_OFFLINE = before_flag
        if before_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = before_env


def from_pretrained(cls, model_id: str, **kwargs):
    if downloaded(model_id):
        try:
            with offline():
                return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
        except OSError:  # a file it needs is missing from the cache: fetch it
            pass
    return cls.from_pretrained(model_id, **kwargs)
