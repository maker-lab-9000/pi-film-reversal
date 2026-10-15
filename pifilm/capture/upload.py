"""The settings menu's UPLOAD row: a Nextcloud sync run as a child process.

Why a child process: the sync (``pifilm.nextcloud``) reads the ``.env`` that holds
the Nextcloud password and drives rclone, and neither belongs in the capture
process. Run as ``python -m pifilm.nextcloud --apply --env PATH``, the password
lives only in the child, the capture service never opens that file, and rclone's
memory and file handles go when the child ends. A child also cannot block a
capture: a network that hangs, an rclone that stalls, a sync that takes ten
minutes all happen in another process, while this one only reads its output on a
daemon thread, ``pifilm-upload``. The child starts in a session (and process
group) of its own, and when the service stops ``close`` sends that whole group
SIGTERM and then SIGKILL after the timeout, so rclone goes with it even when the
service is stopped outside systemd, which would otherwise kill the whole cgroup.
rclone uploads each file whole, so the next run carries on from what was copied.

Why the output is read with ``errors="replace"``: rclone prints file names, and a
name that is not UTF-8 would otherwise raise in the reader thread, which would
stop draining the pipe; the child then blocks on a full pipe, the row stays on
UPLOADING and the child keeps the sync's lock. If reading fails anyway, the line
``nextcloud: output unreadable (...)`` is logged and the exit is still recorded.
``PYTHONUNBUFFERED=1`` makes the child's lines arrive as they are printed, not
when its buffer fills, outside systemd too.

Why the token is removed from the child's environment: ``PIFILM_REMOTE_TOKEN``
is the bearer token of the capture API, read from the systemd unit's environment
file. The sync has no use for it, and a child (and rclone, which it starts in
turn) should not hold a secret it does not need, where a crash report or a
process inspector could expose it. Every other variable is kept, so ``HOME``,
``PATH`` and ``XDG_STATE_HOME`` lead the child to the same lock file and rclone
as the timer's run (``PYTHONUNBUFFERED`` is the one variable added).

Why the status is read from the last line: the sync's guards (no network, no
Nextcloud, another sync running) all exit 0, like a successful copy, because a
timer firing every few minutes must stay quiet off the home network. The exit
code alone cannot tell a copy from a skip, but each outcome ends with its own
line (``rclone copied ...``, ``no network on ...``, ``Nextcloud ... not reachable
...``, ``another Nextcloud sync is running ...``), so the row matches the start
of the last line. A blank line is neither logged nor kept, so a trailing newline
cannot hide the outcome. An exit other than 0 is a failure whatever was printed;
a child that cannot be started at all is recorded as exit 127, the shell's
"command not found".

All state is behind one lock: the viewfinder reads ``status`` every frame from
the loop thread while the reader thread records lines and the exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: Removed from the child's environment: the capture API's token.
TOKEN_ENV = "PIFILM_REMOTE_TOKEN"

#: The exit recorded when the child cannot be started.
EXIT_NOT_STARTED = 127

_JOIN_SECONDS = 2.0


class Upload:
    """Runs one Nextcloud sync at a time and describes it for the UPLOAD row."""

    def __init__(
        self,
        env_path: Path,
        *,
        python: str = sys.executable,
        popen: Callable[..., Any] = subprocess.Popen,
        clock: Any = time,
        log: Callable[[str], None] = print,
        environ: Mapping[str, str] = os.environ,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self._env_path = Path(env_path)
        self._python = python
        self._popen = popen
        self._clock = clock
        self._log = log
        self._environ = environ
        self._killpg = killpg
        self._lock = threading.Lock()
        self._process: Any = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._running = False
        self._start_mono = 0.0
        self._last_line = ""
        self._exit: int | None = None
        self._finish = 0.0

    def start(self) -> bool:
        """Start a sync; False, starting nothing, while one is running."""
        argv = [self._python, "-m", "pifilm.nextcloud", "--apply", "--env", str(self._env_path)]
        env = {k: v for k, v in self._environ.items() if k != TOKEN_ENV}
        env["PYTHONUNBUFFERED"] = "1"
        with self._lock:
            if self._running:
                return False
            self._started = True
            self._last_line = ""
            self._exit = None
            self._start_mono = self._clock.monotonic()
            try:
                process = self._popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                    env=env,
                    start_new_session=True,
                )
            except OSError as exc:
                self._process = None
                self._exit = EXIT_NOT_STARTED
                self._finish = self._clock.time()
                failed = exc
            else:
                failed = None
                self._process = process
                self._running = True
                self._thread = threading.Thread(
                    target=self._read, args=(process,), name="pifilm-upload", daemon=True
                )
                self._thread.start()
        if failed is not None:
            self._log(f"nextcloud: cannot start ({failed})")
        return True

    def _read(self, process: Any) -> None:
        try:
            for raw in process.stdout:
                line = raw.strip()
                if not line:
                    continue
                self._log(f"nextcloud: {line}")
                with self._lock:
                    self._last_line = line
        except Exception as exc:  # noqa: BLE001 - the exit must still be recorded
            self._log(f"nextcloud: output unreadable ({exc})")
        finally:
            code = process.wait()
            with self._lock:
                self._exit = code
                self._finish = self._clock.time()
                self._running = False

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def status(self) -> str:
        """The UPLOAD row's value, upper case."""
        with self._lock:
            if not self._started:
                return "TAP TO UPLOAD"
            if self._running:
                elapsed = max(0, int(self._clock.monotonic() - self._start_mono))
                minutes, seconds = divmod(elapsed, 60)
                return f"UPLOADING {minutes}:{seconds:02d}"
            code, line, finish = self._exit, self._last_line, self._finish
        if code != 0:
            return f"FAILED (EXIT {code})"
        if line.startswith("rclone copied"):
            return f"DONE {time.strftime('%H:%M', time.localtime(finish))}"
        if line.startswith("no network on"):
            return "SKIPPED: NO NETWORK"
        if line.startswith("Nextcloud ") and "not reachable" in line:
            return "SKIPPED: NO NEXTCLOUD"
        if line.startswith("another Nextcloud sync is running"):
            return "SKIPPED: BUSY"
        return "SKIPPED"

    def _signal(self, process: Any, sig: int) -> None:
        """Send ``sig`` to the child's process group, or to the child alone if that fails."""
        try:
            self._killpg(process.pid, sig)
        except OSError:  # ProcessLookupError included: the group is already gone
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()

    def close(self, timeout: float = 5.0) -> None:
        """Stop a running child's group (SIGTERM, then SIGKILL after ``timeout``) and join."""
        with self._lock:
            process = self._process if self._running else None
            thread = self._thread
        if process is not None:
            self._signal(process, signal.SIGTERM)
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
                process.wait()
        if thread is not None:
            thread.join(_JOIN_SECONDS)
