"""Discovers model plugins: the bundled ones plus every ``*.py`` file in the
user plugin folder (``backend/plugins`` by default)."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from .base import ModelPlugin

log = logging.getLogger(__name__)

_plugins: dict[str, ModelPlugin] = {}
_load_errors: list[dict] = []


def _register(plugin: ModelPlugin, origin: str) -> None:
    if not plugin.id:
        raise ValueError(f"{origin}: plugin has no id")
    if plugin.id in _plugins:
        log.warning("plugin id %s from %s overrides an existing plugin", plugin.id, origin)
    _plugins[plugin.id] = plugin


def _plugins_in_module(module) -> list[ModelPlugin]:
    found = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ModelPlugin) and obj is not ModelPlugin and obj.__module__ == module.__name__:
            if not inspect.isabstract(obj):
                found.append(obj())
    return found


def load_plugins(user_dir: Path | None) -> None:
    _plugins.clear()
    _load_errors.clear()

    from .builtin import modern, rtdetr_official, yolov9_official

    for mod in (modern, yolov9_official, rtdetr_official):
        for p in _plugins_in_module(mod):
            _register(p, mod.__name__)

    if user_dir and user_dir.is_dir():
        for path in sorted(user_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            mod_name = f"visionloop_user_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                assert spec and spec.loader
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                for p in _plugins_in_module(module):
                    _register(p, str(path))
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not kill the server
                log.exception("failed to load plugin %s", path)
                _load_errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})


def all_plugins() -> list[ModelPlugin]:
    return list(_plugins.values())


def get_plugin(plugin_id: str) -> ModelPlugin | None:
    return _plugins.get(plugin_id)


def load_errors() -> list[dict]:
    return list(_load_errors)
