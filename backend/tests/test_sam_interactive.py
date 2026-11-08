"""Click-to-segment: the long-lived helper protocol and the API, against a fake helper."""

from __future__ import annotations

import sys
import textwrap

import pytest

from app.plugins import registry
from app.plugins.base import PluginStatus
from app.services import sam_interactive
from app.services.sam_interactive import SamError, SamSession

from .conftest import upload

FAKE = textwrap.dedent(
    """
    import json, sys
    print("loading fake model", flush=True)  # logging must not be taken for an answer
    seen = set()
    for line in sys.stdin:
        req = json.loads(line)
        if "crash" in req["image"]:
            sys.exit(3)
        if "bad" in req["image"]:
            print("@@" + json.dumps({"ok": False, "error": "ValueError: bad image"}), flush=True)
            continue
        first = req["key"] not in seen
        seen.add(req["key"])
        if req["op"] == "prepare":
            print("@@" + json.dumps({"ok": True}), flush=True)
            continue
        if req["box"]:
            x, y, w, h = req["box"]
        else:
            px, py, _ = req["points"][0]
            x, y, w, h = px - 5, py - 5, 10, 10
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        if req.get("exclude"):  # echo what it was asked to cut out
            poly = req["exclude"][0]
        # the score tells the test how many points arrived and whether the image was encoded again
        print("@@" + json.dumps({"ok": True, "polygon": poly, "score": len(req["points"]) + (0.5 if first else 0)}), flush=True)
    """
)


@pytest.fixture()
def fake_session(tmp_path):
    helper = tmp_path / "fake_sam.py"
    helper.write_text(FAKE)
    s = SamSession(helper=helper, python=sys.executable, idle_seconds=60)
    yield s
    s.stop()


def ask(session, image="a.jpg", key="k1", **kw):
    return session.request({"op": "segment", "model": "m", "image": image, "key": key,
                            "points": kw.get("points", [[50, 40, 1]]), "box": kw.get("box")})


def test_one_process_answers_every_click(fake_session):
    first = ask(fake_session)
    assert first["polygon"] == [[45, 35], [55, 35], [55, 45], [45, 45]]
    assert first["score"] == 1.5  # first request for this image
    pid = fake_session._proc.pid
    again = ask(fake_session, points=[[50, 40, 1], [70, 40, 0]])
    assert again["score"] == 2  # same image, not encoded again
    assert fake_session._proc.pid == pid


def test_errors_are_reported_and_a_crashed_helper_is_restarted(fake_session):
    with pytest.raises(SamError, match="bad image"):
        ask(fake_session, image="bad.jpg")
    with pytest.raises(SamError, match="意外結束"):
        ask(fake_session, image="crash.jpg")
    assert not fake_session.running
    assert ask(fake_session)["polygon"]  # started again


def test_stop_ends_the_helper(fake_session):
    ask(fake_session)
    proc = fake_session._proc
    fake_session.stop()
    assert proc.poll() is not None
    assert not fake_session.running


@pytest.fixture()
def sam_api(client, monkeypatch, fake_session):
    monkeypatch.setattr(sam_interactive, "session", fake_session)
    monkeypatch.setattr(type(registry.get_plugin("sam2")), "status", lambda self: PluginStatus(True))
    return client


def test_segment_endpoint(sam_api, project):
    img = upload(sam_api, project["id"], ["a.jpg"])["created"][0]
    assert sam_api.post(f"/api/images/{img['id']}/sam/prepare").json() == {"ready": True}
    r = sam_api.post(f"/api/images/{img['id']}/sam", json={"points": [], "box": [10, 20, 30, 40]})
    assert r.status_code == 200, r.text
    assert r.json()["polygon"] == [[10, 20], [40, 20], [40, 60], [10, 60]]


def test_outlines_to_cut_out_reach_the_helper(sam_api, project):
    img = upload(sam_api, project["id"], ["a.jpg"])["created"][0]
    outline = [[1, 1], [9, 1], [9, 9]]
    r = sam_api.post(f"/api/images/{img['id']}/sam",
                     json={"points": [[5, 5, 1]], "exclude": [[[0, 0], [1, 1]], outline]})
    assert r.status_code == 200, r.text
    assert r.json()["polygon"] == outline  # the two-point outline was dropped before sending


def test_segment_needs_a_prompt(sam_api, project):
    img = upload(sam_api, project["id"], ["a.jpg"])["created"][0]
    assert sam_api.post(f"/api/images/{img['id']}/sam", json={"points": []}).status_code == 422
    assert sam_api.post(f"/api/images/{img['id']}/sam", json={"points": [[1, 2, 3]]}).status_code == 422
    assert sam_api.post(f"/api/images/{img['id']}/sam", json={"box": [1, 2, 0, 5]}).status_code == 422


def test_segment_without_the_model_environment(client, project, monkeypatch):
    monkeypatch.setattr(type(registry.get_plugin("sam2")), "status", lambda self: PluginStatus(False, "沒有模型環境"))
    img = upload(client, project["id"], ["a.jpg"])["created"][0]
    r = client.post(f"/api/images/{img['id']}/sam", json={"points": [[5, 5, 1]]})
    assert r.status_code == 503 and "沒有模型環境" in r.json()["detail"]
