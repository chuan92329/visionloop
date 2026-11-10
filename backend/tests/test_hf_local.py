"""Downloaded models load without asking huggingface.co first (no waiting on a network timeout)."""

from __future__ import annotations

import pytest

from app.plugins.builtin import hf_local


class Model:
    calls: list[dict] = []
    cached = True

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append(kwargs)
        if kwargs.get("local_files_only") and not cls.cached:
            raise OSError("not in the cache")
        return model_id


REAL_DOWNLOADED = hf_local.downloaded


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    Model.calls = []
    monkeypatch.setattr(hf_local, "downloaded", lambda model_id: Model.cached)


def test_a_downloaded_model_is_loaded_from_the_cache_only():
    Model.cached = True
    assert hf_local.from_pretrained(Model, "m", trust_remote_code=True) == "m"
    assert Model.calls == [{"local_files_only": True, "trust_remote_code": True}]


def test_a_missing_model_is_downloaded():
    Model.cached = False
    assert hf_local.from_pretrained(Model, "m") == "m"
    assert Model.calls == [{}]


def test_the_hub_client_is_offline_while_a_downloaded_model_loads(monkeypatch):
    import os

    hub = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen = []

    class Probe:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen.append((hub.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")))

    monkeypatch.setattr(hf_local, "downloaded", lambda model_id: True)
    hf_local.from_pretrained(Probe, "m")
    assert seen == [(True, "1")]
    assert os.environ.get("HF_HUB_OFFLINE") is None  # restored afterwards


def test_downloaded_needs_a_snapshot_with_config(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_local, "downloaded", REAL_DOWNLOADED)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert not hf_local.downloaded("org/m")
    (snap / "config.json").write_text("{}")
    assert hf_local.downloaded("org/m")


def test_every_helper_loads_through_it():
    from pathlib import Path

    here = Path(hf_local.__file__).parent
    for name in ("sam_server.py", "sam_helper.py", "hf_detect_helper.py", "probe_helper.py"):
        source = (here / name).read_text(encoding="utf-8")
        assert ".from_pretrained(" not in source.replace("hf_local.from_pretrained(", ""), name
