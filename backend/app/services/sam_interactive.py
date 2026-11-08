"""Click-to-segment in the annotator.

SAM 2 runs in the model environment as a long-lived helper (``sam_server.py``), so the
model is loaded once and the image being annotated is encoded once; every further
click only costs the mask decoder.  The helper is started on first use and stopped
after a while without requests, which frees the GPU memory it holds.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path

from ..plugins.base import NO_WINDOW, python_executable

HELPER = Path(__file__).resolve().parents[1] / "plugins" / "builtin" / "sam_server.py"
MARK = "@@"
DEFAULT_MODEL = "facebook/sam2.1-hiera-large"
IDLE_SECONDS = 600
FIRST_REPLY_SECONDS = 300  # the first request loads the model (and may download it)
REPLY_SECONDS = 60


class SamError(RuntimeError):
    pass


class SamSession:
    def __init__(self, helper: Path = HELPER, python: str | None = None, idle_seconds: float = IDLE_SECONDS) -> None:
        self.helper = helper
        self.python = python
        self.idle_seconds = idle_seconds
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._log: deque[str] = deque(maxlen=30)
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._answered = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        self._proc = subprocess.Popen(
            [self.python or python_executable("VISIONLOOP_MODELS_PYTHON"), str(self.helper)],
            cwd=str(self.helper.parent),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=NO_WINDOW,
        )
        self._lines = queue.Queue()
        self._answered = False
        threading.Thread(target=self._read, args=(self._proc, self._lines), daemon=True).start()

    @staticmethod
    def _read(proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)  # the helper ended

    def request(self, payload: dict) -> dict:
        with self._lock:
            if not self.running:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                self._stop_locked()
                raise SamError(f"SAM 2 已經結束：{self._tail()}") from exc
            wait = REPLY_SECONDS if self._answered else FIRST_REPLY_SECONDS
            while True:
                try:
                    line = self._lines.get(timeout=wait)
                except queue.Empty:
                    self._stop_locked()
                    raise SamError(f"SAM 2 超過 {wait} 秒沒有回應，已重新啟動") from None
                if line is None:
                    self._stop_locked()
                    raise SamError(f"SAM 2 意外結束：{self._tail()}")
                if line.startswith(MARK):
                    break
                self._log.append(line)
            self._answered = True
            self._arm_idle_timer()
        answer = json.loads(line[len(MARK):])
        if not answer.get("ok"):
            raise SamError(answer.get("error") or "SAM 2 失敗")
        return answer

    def _tail(self) -> str:
        return " / ".join(list(self._log)[-3:]) or "沒有輸出"

    def _arm_idle_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle_seconds, self.stop)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # the helper exits when its input ends
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


session = SamSession()
