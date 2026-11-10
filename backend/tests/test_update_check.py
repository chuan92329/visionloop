"""The "newer version available" check: version comparison, daily cache, offline, switched off."""

from __future__ import annotations

import json
import time

import pytest

from app.routers import updates


def test_version_comparison():
    assert updates.is_newer("v0.1.3", "0.1.2")
    assert updates.is_newer("v0.2.0", "0.1.10")
    assert updates.is_newer("v0.1.10", "0.1.9")  # numbers, not text
    assert not updates.is_newer("v0.1.2", "0.1.2")
    assert not updates.is_newer("v0.1.1", "0.1.2")
    assert not updates.is_newer("nightly", "0.1.2")


@pytest.fixture()
def github(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.delenv("VISIONLOOP_CHECK_UPDATES", raising=False)
    calls = []

    def fetch(latest="v9.0.0"):
        calls.append(1)
        return {"latest": latest, "url": "https://github.com/x/releases/tag/" + latest, "notes": "n",
                "checked_at": time.time()}

    monkeypatch.setattr(updates, "_fetch", fetch)
    return calls


def test_newer_release_is_reported_and_cached_for_a_day(client, github):
    r = client.get("/api/update-check").json()
    assert r["newer"] is True and r["latest"] == "v9.0.0" and r["url"].endswith("v9.0.0")
    client.get("/api/update-check")
    assert len(github) == 1  # second answer came from the cache


def test_offline_is_not_an_error(client, github, monkeypatch):
    def offline():
        raise OSError("timed out")

    monkeypatch.setattr(updates, "_fetch", offline)
    r = client.get("/api/update-check")
    assert r.status_code == 200 and r.json()["newer"] is False and r.json()["latest"] is None


def test_switched_off_never_asks_github(client, github, monkeypatch):
    monkeypatch.setenv("VISIONLOOP_CHECK_UPDATES", "false")
    r = client.get("/api/update-check").json()
    assert r["enabled"] is False and r["newer"] is False and github == []


def test_false_in_local_settings_reaches_the_environment(tmp_path, monkeypatch):
    from app import config

    monkeypatch.delenv("VISIONLOOP_CHECK_UPDATES", raising=False)
    settings_file = tmp_path / "local_settings.json"
    settings_file.write_text(json.dumps({"check_updates": False}), encoding="utf-8")
    config.apply_local_settings(settings_file)
    import os

    assert os.environ["VISIONLOOP_CHECK_UPDATES"] == "false"
