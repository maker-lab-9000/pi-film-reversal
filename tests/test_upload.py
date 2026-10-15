"""The settings menu's upload: a child process, its log lines and the UPLOAD row's text."""

from __future__ import annotations

import io
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pifilm.capture.upload import Upload


class FakeClock:
    def __init__(self, wall: float = 1_760_000_000.0, mono: float = 100.0) -> None:
        self.wall = wall
        self.mono = mono

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


class FakeProcess:
    def __init__(self, lines, code=0, *, hold=False, wait_timeouts=0):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.code = code
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.wait_timeouts = wait_timeouts
        self.terminated = False
        self.killed = False
        self.waits: list[float | None] = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is not None and self.wait_timeouts:
            self.wait_timeouts -= 1
            raise subprocess.TimeoutExpired("python", timeout)
        self.release.wait(10)
        return self.code

    def poll(self):
        return self.code if self.release.is_set() else None

    def terminate(self):
        self.terminated = True
        if not self.wait_timeouts:
            self.release.set()

    def kill(self):
        self.killed = True
        self.release.set()


class FakePopen:
    def __init__(self, process: FakeProcess | None = None, error: OSError | None = None):
        self.process = process
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        return self.process


def make(process=None, *, error=None, environ=None, clock=None):
    popen = FakePopen(process, error)
    logs: list[str] = []
    upload = Upload(
        Path("/etc/pifilm/nextcloud.env"),
        python="/usr/bin/python3",
        popen=popen,
        clock=clock or FakeClock(),
        log=logs.append,
        environ=environ if environ is not None else {"HOME": "/home/pi"},
    )
    return upload, popen, logs


def finish(upload: Upload) -> None:
    deadline = time.monotonic() + 5
    while upload.running and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not upload.running


def test_start_runs_the_sync_module_with_apply_and_the_env_path():
    upload, popen, _ = make(FakeProcess(["rclone copied a -> b (exit 0)"]))
    assert upload.start()
    finish(upload)
    argv, kwargs = popen.calls[0]
    assert argv == [
        "/usr/bin/python3", "-m", "pifilm.nextcloud", "--apply",
        "--env", "/etc/pifilm/nextcloud.env",
    ]
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["bufsize"] == 1


def test_child_environment_lacks_the_remote_token_and_keeps_the_rest():
    environ = {"HOME": "/home/pi", "PATH": "/usr/bin", "PIFILM_REMOTE_TOKEN": "secret"}
    upload, popen, _ = make(FakeProcess([]), environ=environ)
    upload.start()
    finish(upload)
    env = popen.calls[0][1]["env"]
    assert env == {"HOME": "/home/pi", "PATH": "/usr/bin"}
    assert environ["PIFILM_REMOTE_TOKEN"] == "secret"


def test_each_line_is_logged_with_the_nextcloud_prefix():
    lines = ["Transferred: 3 files", "  rclone copied /a -> nextcloud:B (exit 0)  "]
    upload, _, logs = make(FakeProcess(lines))
    upload.start()
    finish(upload)
    assert logs == [
        "nextcloud: Transferred: 3 files",
        "nextcloud: rclone copied /a -> nextcloud:B (exit 0)",
    ]


def test_status_before_any_start_is_tap_to_upload():
    upload, _, _ = make(FakeProcess([]))
    assert upload.status() == "TAP TO UPLOAD"
    assert not upload.running


def test_status_while_running_shows_elapsed_minutes_and_seconds():
    clock = FakeClock(mono=100.0)
    process = FakeProcess([], hold=True)
    upload, _, _ = make(process, clock=clock)
    upload.start()
    assert upload.running
    assert upload.status() == "UPLOADING 0:00"
    clock.mono = 142.9
    assert upload.status() == "UPLOADING 0:42"
    clock.mono = 100.0 + 605
    assert upload.status() == "UPLOADING 10:05"
    process.release.set()
    finish(upload)


def test_status_done_shows_the_local_finish_time():
    clock = FakeClock(wall=1_760_000_000.0)
    upload, _, _ = make(FakeProcess(["rclone copied /a -> nextcloud:B (exit 0)"]), clock=clock)
    upload.start()
    finish(upload)
    expected = time.strftime("%H:%M", time.localtime(1_760_000_000.0))
    assert upload.status() == f"DONE {expected}"


@pytest.mark.parametrize(
    ("line", "text"),
    [
        ("no network on eth0 or wlan0; skipping Nextcloud sync", "SKIPPED: NO NETWORK"),
        (
            "Nextcloud cloud.example:443 not reachable over wlan0 (10.42.0.1); skipping",
            "SKIPPED: NO NEXTCLOUD",
        ),
        ("another Nextcloud sync is running; skipping", "SKIPPED: BUSY"),
        ("something else entirely", "SKIPPED"),
    ],
)
def test_status_exit_zero_is_chosen_from_the_last_line(line, text):
    upload, _, _ = make(FakeProcess(["earlier line", line]))
    upload.start()
    finish(upload)
    assert upload.status() == text


def test_status_exit_zero_with_no_output_is_skipped():
    upload, _, _ = make(FakeProcess([]))
    upload.start()
    finish(upload)
    assert upload.status() == "SKIPPED"


def test_status_nonzero_exit_is_failed_with_the_code():
    upload, _, _ = make(FakeProcess(["rclone copied /a -> nextcloud:B (exit 4)"], code=4))
    upload.start()
    finish(upload)
    assert upload.status() == "FAILED (EXIT 4)"


def test_start_returns_false_while_one_is_running():
    process = FakeProcess([], hold=True)
    upload, popen, _ = make(process)
    assert upload.start()
    assert not upload.start()
    assert len(popen.calls) == 1
    process.release.set()
    finish(upload)


def test_start_again_after_a_finish_runs_a_new_child():
    upload, popen, _ = make(FakeProcess(["no network on eth0; skipping Nextcloud sync"]))
    upload.start()
    finish(upload)
    assert upload.status() == "SKIPPED: NO NETWORK"
    popen.process = FakeProcess(["rclone copied /a -> nextcloud:B (exit 0)"])
    assert upload.start()
    finish(upload)
    assert len(popen.calls) == 2
    assert upload.status().startswith("DONE ")


def test_close_terminates_a_running_child_and_joins():
    process = FakeProcess([], code=-15, hold=True)
    upload, _, _ = make(process)
    upload.start()
    upload.close(timeout=0.5)
    assert process.terminated
    assert not process.killed
    assert 0.5 in process.waits
    assert not upload.running
    assert upload.status() == "FAILED (EXIT -15)"


def test_close_kills_a_child_that_outlasts_the_timeout():
    process = FakeProcess([], code=-9, hold=True, wait_timeouts=1)
    upload, _, _ = make(process)
    upload.start()
    upload.close(timeout=0.5)
    assert process.terminated
    assert process.killed
    assert not upload.running


def test_close_with_nothing_running_does_nothing_and_is_safe_twice():
    upload, _, _ = make(FakeProcess([]))
    upload.close()
    upload.close()
    process = FakeProcess([])
    upload, _, _ = make(process)
    upload.start()
    finish(upload)
    upload.close()
    upload.close()
    assert not process.terminated
    assert not process.killed


def test_popen_oserror_is_logged_and_failed_with_exit_127():
    upload, _, logs = make(error=FileNotFoundError(2, "No such file", "/usr/bin/python3"))
    upload.start()
    finish(upload)
    assert upload.status() == "FAILED (EXIT 127)"
    assert len(logs) == 1
    assert logs[0].startswith("nextcloud: cannot start")
    upload.close()
