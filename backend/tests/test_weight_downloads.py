"""Default model sizes and the "first use downloads the model" notices."""

from __future__ import annotations

from app.plugins.builtin import modern


class Hooks:
    def __init__(self):
        self.messages: list[str] = []

    def progress(self, fraction: float, message: str = "") -> None:
        self.messages.append(message)


def cache_model(root, model_id: str) -> None:
    snap = root / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")


def test_notice_only_for_models_not_downloaded_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    model = "facebook/sam2.1-hiera-large"
    hooks = Hooks()
    modern.announce_download(hooks, model)
    assert hooks.messages == ["第一次使用，正在下載 sam2.1-hiera-large（約 900 MB）…"]

    cache_model(tmp_path, model)
    assert modern.hf_cached(model)
    hooks = Hooks()
    modern.announce_download(hooks, model)
    assert hooks.messages == []


def test_helper_download_line_becomes_the_job_message():
    hooks, seen = Hooks(), []
    handle = modern.with_download_notice(hooks, seen.append)
    handle("@@download rf-detr-large-2026.pth 520")
    handle("Epoch 1")
    assert hooks.messages == ["第一次使用，正在下載 rf-detr-large-2026.pth（約 520 MB）…"]
    assert seen == ["Epoch 1"]  # every other line still reaches the original handler


def test_menus_default_to_the_best_model():
    assert modern.SAM2Plugin.variants[0].id == "facebook/sam2.1-hiera-large"
    assert modern.GroundingDINOPlugin.variants[0].id == "base"
    assert modern.ProbePlugin.variants[0].id == "dinov2-b-hr"


def test_rfdetr_weights_are_shared_between_jobs():
    assert modern.RFDETR_WEIGHTS_DIR.parts[-2:] == ("models", "rfdetr")


def test_sam_info_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    r = client.get("/api/sam/info").json()
    assert r == {"model": "facebook/sam2.1-hiera-large", "cached": False, "size_mb": 900}
    cache_model(tmp_path, "facebook/sam2.1-hiera-large")
    assert client.get("/api/sam/info").json()["cached"] is True


def test_a_failed_download_is_explained():
    log = ("Traceback (most recent call last):\n  ...\n"
           "OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them "
           "in the cached files.")
    assert modern.explain_failure(log) == modern.DOWNLOAD_FAILED
    assert modern.explain_failure("requests.exceptions.ConnectionError: Max retries exceeded") == modern.DOWNLOAD_FAILED
    assert modern.explain_failure("RuntimeError: CUDA out of memory") is None  # other failures keep their own message


def test_click_to_segment_explains_a_failed_download(client, project, monkeypatch):
    from app.plugins import registry
    from app.plugins.base import PluginStatus
    from app.services import sam_interactive

    from .conftest import upload

    class Offline:
        def request(self, payload):
            raise sam_interactive.SamError("OSError: We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(sam_interactive, "session", Offline())
    monkeypatch.setattr(type(registry.get_plugin("sam2")), "status", lambda self: PluginStatus(True))
    img = upload(client, project["id"], ["a.jpg"])["created"][0]
    r = client.post(f"/api/images/{img['id']}/sam/prepare")
    assert r.status_code == 502 and r.json()["detail"] == modern.DOWNLOAD_FAILED
