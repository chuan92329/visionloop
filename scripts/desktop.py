"""VisionLoop in its own window instead of a browser tab.

    pythonw scripts/desktop.py        (what the VisionLoop shortcut runs; no console window)

The API server runs in a background thread of this process, bound to 127.0.0.1 only, and the
window shows the same web UI through the system web view (Edge WebView2 on Windows). Closing the
window stops the server, the job runner and the click-to-segment helper.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

PREFERRED_PORT = int(os.environ.get("VISIONLOOP_PORT", "8000"))
ICON = ROOT / "scripts" / "visionloop.ico"
SPLASH = ROOT / "scripts" / "splash.html"
SPLASH_LIMIT = 8.0  # never wait longer than this for the opening to report it has finished
FRONTEND = ROOT / "frontend"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOW = dict(width=1440, height=900, min_size=(1000, 650), background_color="#1e2430")


def windows_app_id() -> None:
    """Group the window under its own taskbar entry (with its own icon), not under pythonw.exe."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VisionLoop.Desktop")


def windows_icon() -> None:
    """pywebview only takes an icon on Linux; on Windows set it on our own top-level windows."""
    if sys.platform != "win32" or not ICON.exists():
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON = 1, 0x10, 0x80
    big = user32.LoadImageW(None, str(ICON), IMAGE_ICON, 256, 256, LR_LOADFROMFILE)
    small = user32.LoadImageW(None, str(ICON), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
    pid = os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            user32.SendMessageW(hwnd, WM_SETICON, 1, big)
            user32.SendMessageW(hwnd, WM_SETICON, 0, small)
        return True

    user32.EnumWindows(each, 0)


def running_visionloop(port: int) -> bool:
    """Is a VisionLoop server (e.g. a second window, or uvicorn started by hand) already answering here?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001 - nothing there, or something that is not VisionLoop
        return False


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def error_html(message: str) -> str:
    return (
        "<body style='font-family:\"Times New Roman\",標楷體,serif;background:#1e2430;color:#e6e8ee;"
        f"padding:48px;font-size:18px;line-height:1.7'><h2>VisionLoop 無法啟動</h2><p>{message}</p></body>"
    )


def ui_needs_build() -> bool:
    """The built UI is missing or older than its sources (e.g. after a git pull). A release zip
    ships it already built (marked by dist/.release) and has nothing to build it with."""
    dist = FRONTEND / "dist"
    if (dist / ".release").exists():
        return False
    index = dist / "index.html"
    if not index.exists():
        return True
    built = index.stat().st_mtime
    sources = [FRONTEND / "index.html", FRONTEND / "package.json"]
    sources += [f for d in ("src", "public") for f in (FRONTEND / d).rglob("*") if f.is_file()]
    return any(f.exists() and f.stat().st_mtime > built for f in sources)


def build_ui(log_path: Path) -> None:
    """npm install (first time) + npm run build, hidden. Node.js may live in the conda env or in
    tools\\node (downloaded by setup_windows.ps1 when Python comes from uv)."""
    env_dir = Path(sys.executable).parent
    path = os.pathsep.join([str(ROOT / "tools" / "node"), str(env_dir), str(env_dir / "Library" / "bin"),
                            str(env_dir / "Scripts"), os.environ.get("PATH", "")])
    npm = shutil.which("npm", path=path)
    if not npm:
        raise RuntimeError("網頁介面需要重新建置，但找不到 Node.js。請重新執行 "
                           "<code>scripts\\setup_windows.ps1</code>。")
    steps = ([] if (FRONTEND / "node_modules").exists() else [[npm, "install"]]) + [[npm, "run", "build"]]
    with open(log_path, "a", encoding="utf-8") as log:
        for cmd in steps:
            r = subprocess.run(cmd, cwd=FRONTEND, env={**os.environ, "PATH": path}, stdout=log,
                               stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            if r.returncode != 0:
                raise RuntimeError(f"建置網頁介面失敗，錯誤訊息記錄在 <code>{log_path}</code>。")


class Boot:
    """Runs behind the splash: starts the server, then swaps the splash for the app."""

    def __init__(self) -> None:
        self.server = None
        self.thread: threading.Thread | None = None

    def __call__(self, window) -> None:
        shown = time.time()
        try:
            url = self.start_server()
        except RuntimeError as exc:
            window.load_html(error_html(str(exc)))
            return
        # the server is ready; now wait for the opening animation to finish as well
        while time.time() - shown < SPLASH_LIMIT:
            try:
                if window.evaluate_js("window.vlDone === true"):
                    break
            except Exception:  # noqa: BLE001 - page not ready to answer yet
                pass
            time.sleep(0.1)
        try:
            window.evaluate_js("window.vlLeave && window.vlLeave()")
            time.sleep(0.35)
        except Exception:  # noqa: BLE001 - the fade is only cosmetic
            pass
        window.load_url(url)

    def start_server(self) -> str:
        # Another VisionLoop is already serving this data folder: show it, never start a second
        # server on the same database.
        if running_visionloop(PREFERRED_PORT):
            return f"http://127.0.0.1:{PREFERRED_PORT}"

        from app.config import settings

        settings.ensure_dirs()
        log = settings.data_dir / "visionloop.log"  # there is no console to print to
        if ui_needs_build():  # behind the splash, so the window is already up
            build_ui(log)
        if not (settings.frontend_dist / "index.html").exists():
            raise RuntimeError("網頁介面還沒建置。請重新執行 <code>scripts\\setup_windows.ps1</code>。")
        logging.basicConfig(filename=log, level=logging.INFO, encoding="utf-8",
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

        import uvicorn
        from app.main import app

        port = PREFERRED_PORT if port_free(PREFERRED_PORT) else free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                                    log_config=None))
        self.server.install_signal_handlers = lambda: None  # not the main thread
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        deadline = time.time() + 60
        while not self.server.started:
            if not self.thread.is_alive() or time.time() > deadline:
                raise RuntimeError(f"伺服器沒有啟動，錯誤訊息記錄在 <code>{log}</code>。")
            time.sleep(0.05)
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True  # runs the app's shutdown: stops the SAM helper
            self.thread.join(timeout=10)


def ensure_utf8_mode() -> None:
    """Read and write files as UTF-8, not the Windows code page (cp950 on Traditional Chinese
    Windows), which cannot read the Chinese class names in data.yaml. The shortcut starts us with
    -X utf8; started any other way, start again in UTF-8 mode."""
    if sys.flags.utf8_mode:
        return
    os.environ["PYTHONUTF8"] = "1"
    subprocess.Popen([sys.executable, "-X", "utf8", *sys.argv])
    sys.exit(0)


def main() -> None:
    ensure_utf8_mode()
    try:
        import webview
    except ImportError:  # started from the shortcut there is no console to print this to
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, "缺少 pywebview。請重新執行 scripts\\setup_windows.ps1。",
                                         "VisionLoop 無法啟動", 0x10)
        return

    webview.settings["ALLOW_DOWNLOADS"] = True  # dataset export, weights download
    windows_app_id()
    window = webview.create_window("VisionLoop", html=SPLASH.read_text(encoding="utf-8"), **WINDOW)
    window.events.shown += windows_icon
    boot = Boot()
    webview.start(boot, window)  # boot runs in its own thread; start() returns when the window closes
    boot.stop()


if __name__ == "__main__":
    main()
