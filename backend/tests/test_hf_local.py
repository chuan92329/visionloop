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


@pytest.fixture(autouse=True)
def reset():
    Model.calls = []


def test_a_downloaded_model_is_loaded_from_the_cache_only():
    Model.cached = True
    assert hf_local.from_pretrained(Model, "m", trust_remote_code=True) == "m"
    assert Model.calls == [{"local_files_only": True, "trust_remote_code": True}]


def test_a_missing_model_is_downloaded():
    Model.cached = False
    assert hf_local.from_pretrained(Model, "m") == "m"
    assert Model.calls == [{"local_files_only": True}, {}]


def test_every_helper_loads_through_it():
    from pathlib import Path

    here = Path(hf_local.__file__).parent
    for name in ("sam_server.py", "sam_helper.py", "hf_detect_helper.py", "probe_helper.py"):
        source = (here / name).read_text(encoding="utf-8")
        assert ".from_pretrained(" not in source.replace("hf_local.from_pretrained(", ""), name
