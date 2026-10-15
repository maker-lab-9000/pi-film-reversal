"""The disk-space check behind the viewfinder's PHOTOS / SD warning."""

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pifilm.capture.disk import (
    DEFAULT_WARN_PERCENT,
    DiskMonitor,
    Usage,
    measure,
    watched_paths,
)

GIB = 1024 ** 3


def _vfs(*, blocks, bfree, bavail, frsize=4096):
    return SimpleNamespace(f_blocks=blocks, f_bfree=bfree, f_bavail=bavail, f_frsize=frsize)


def _vfs_for(percent_used_of_df, *, total_gib=30.0):
    """A filesystem whose df Use% is exactly ``percent_used_of_df`` (no reserved blocks)."""
    frsize = 4096
    blocks = int(total_gib * GIB / frsize)
    free = blocks - blocks * percent_used_of_df // 100
    return _vfs(blocks=blocks, bfree=free, bavail=free, frsize=frsize)


def test_measure_is_df_use_percent_with_reserved_blocks_left_out():
    # used = (1000 - 300) blocks, available = 200 blocks: 700 / 900 = 77.8 %, df says 78.
    usage = measure("/x", statvfs=lambda p: _vfs(blocks=1000, bfree=300, bavail=200))
    assert usage == Usage(percent=78, free_bytes=200 * 4096)


def test_measure_rounds_up_like_df():
    # 901 / 1000 = 90.1 %: df shows 91, so a filesystem a hair over 90 is "91% full".
    assert measure("/x", statvfs=lambda p: _vfs(blocks=1000, bfree=99, bavail=99)).percent == 91
    # Exactly 90 stays 90.
    assert measure("/x", statvfs=lambda p: _vfs(blocks=1000, bfree=100, bavail=100)).percent == 90


def test_measure_of_an_empty_filesystem_is_zero_percent():
    assert measure("/x", statvfs=lambda p: _vfs(blocks=0, bfree=0, bavail=0)).percent == 0


def test_watched_paths_are_the_photo_folder_first_then_the_root():
    assert watched_paths("/home/g/Pictures/pifilm") == (
        ("PHOTOS", Path("/home/g/Pictures/pifilm")), ("SD", Path("/")))


class FakeFs:
    """statvfs and stat over a table of paths: {path: (device, statvfs result or OSError)}."""

    def __init__(self, table):
        self.table = table
        self.statvfs_calls = []

    def statvfs(self, path):
        self.statvfs_calls.append(str(path))
        result = self.table[str(path)][1]
        if isinstance(result, OSError):
            raise result
        return result

    def stat(self, path):
        device, result = self.table[str(path)]
        if isinstance(result, OSError):
            raise result
        return SimpleNamespace(st_dev=device)


def _monitor(fs, lines, threshold=90, paths=None):
    return DiskMonitor(
        paths or (("PHOTOS", Path("/photos")), ("SD", Path("/"))), threshold,
        log=lines.append, statvfs=fs.statvfs, stat=fs.stat)


def test_nothing_low_is_an_empty_warning_and_no_log_line():
    fs = FakeFs({"/photos": (2, _vfs_for(50)), "/": (1, _vfs_for(70))})
    lines = []
    monitor = _monitor(fs, lines)
    assert monitor.warning == ()
    monitor.poll_once()
    assert monitor.warning == ()
    assert lines == []


def test_crossing_up_logs_once_and_staying_low_logs_nothing():
    fs = FakeFs({"/photos": (2, _vfs_for(92)), "/": (1, _vfs_for(70))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    assert monitor.warning == (("PHOTOS", 92),)
    assert lines == ["disk: photos 92% full (2.4 GB free)"]
    monitor.poll_once()
    monitor.poll_once()
    assert len(lines) == 1


def test_crossing_down_logs_the_back_to_line_and_clears_the_warning():
    fs = FakeFs({"/photos": (2, _vfs_for(92)), "/": (1, _vfs_for(70))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    fs.table["/photos"] = (2, _vfs_for(85))
    monitor.poll_once()
    assert monitor.warning == ()
    assert lines[-1] == "disk: photos back to 85% full (4.5 GB free)"
    monitor.poll_once()
    assert len(lines) == 2


def test_the_threshold_is_inclusive_and_both_filesystems_warn_photos_first():
    fs = FakeFs({"/photos": (2, _vfs_for(90)), "/": (1, _vfs_for(91, total_gib=59))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    assert monitor.warning == (("PHOTOS", 90), ("SD", 91))
    assert lines == ["disk: photos 90% full (3.0 GB free)", "disk: sd 91% full (5.3 GB free)"]


def test_a_threshold_of_100_warns_only_when_full():
    fs = FakeFs({"/photos": (2, _vfs_for(99)), "/": (1, _vfs_for(100))})
    lines = []
    monitor = _monitor(fs, lines, threshold=100)
    monitor.poll_once()
    assert monitor.warning == (("SD", 100),)


def test_one_filesystem_for_both_paths_is_measured_once_and_labelled_photos():
    fs = FakeFs({"/photos": (1, _vfs_for(95)), "/": (1, _vfs_for(95))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    assert monitor.warning == (("PHOTOS", 95),)
    assert fs.statvfs_calls == ["/photos"]
    assert lines == ["disk: photos 95% full (1.5 GB free)"]


def test_a_path_that_cannot_be_measured_is_skipped_after_one_log_line():
    fs = FakeFs({"/photos": (2, FileNotFoundError(2, "No such file or directory")),
                 "/": (1, _vfs_for(93))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    monitor.poll_once()
    assert monitor.warning == (("SD", 93),)
    cannot = [line for line in lines if "cannot" in line]
    assert len(cannot) == 1
    assert cannot[0].startswith("disk: cannot measure photos (/photos):")
    # Once it can be measured again it takes part as usual, and a later failure is
    # worth one more line.
    fs.table["/photos"] = (2, _vfs_for(96))
    monitor.poll_once()
    assert monitor.warning == (("PHOTOS", 96), ("SD", 93))
    fs.table["/photos"] = (2, OSError(5, "Input/output error"))
    monitor.poll_once()
    assert len([line for line in lines if "cannot" in line]) == 2
    assert monitor.warning == (("SD", 93),)


def test_a_statvfs_failure_after_stat_is_skipped_too():
    class Flaky(FakeFs):
        def statvfs(self, path):
            if str(path) == "/":
                raise PermissionError(13, "Permission denied")
            return super().statvfs(path)

    fs = Flaky({"/photos": (2, _vfs_for(50)), "/": (1, _vfs_for(50))})
    lines = []
    monitor = _monitor(fs, lines)
    monitor.poll_once()
    assert monitor.warning == ()
    assert len(lines) == 1 and lines[0].startswith("disk: cannot measure sd (/):")


def test_any_exception_from_a_measurement_never_leaves_the_poller():
    def broken(path):
        raise RuntimeError("bug")

    lines = []
    monitor = DiskMonitor((("PHOTOS", Path("/photos")),), 90, log=lines.append,
                          statvfs=broken, stat=lambda p: SimpleNamespace(st_dev=1))
    monitor.poll_once()  # does not raise
    assert monitor.warning == ()
    assert len(lines) == 1


def test_a_log_that_raises_does_not_stop_the_poll():
    fs = FakeFs({"/photos": (2, _vfs_for(92)), "/": (1, _vfs_for(70))})

    def log(line):
        raise OSError("stdout closed")

    monitor = DiskMonitor((("PHOTOS", Path("/photos")), ("SD", Path("/"))), 90, log=log,
                          statvfs=fs.statvfs, stat=fs.stat)
    monitor.poll_once()
    assert monitor.warning == (("PHOTOS", 92),)


@pytest.mark.parametrize("threshold", [0, 101])
def test_a_threshold_outside_1_to_100_is_refused(threshold):
    with pytest.raises(ValueError):
        DiskMonitor((), threshold)


def test_the_default_threshold_is_90():
    assert DEFAULT_WARN_PERCENT == 90


def test_the_poller_checks_at_once_on_its_own_thread_and_stops_cleanly():
    fs = FakeFs({"/photos": (2, _vfs_for(92)), "/": (1, _vfs_for(70))})
    polled = threading.Event()
    lines = []

    def log(line):
        lines.append(line)
        polled.set()

    waits = []
    monitor = DiskMonitor((("PHOTOS", Path("/photos")), ("SD", Path("/"))), 90, log=log,
                          statvfs=fs.statvfs, stat=fs.stat, interval_s=3600.0)
    original_wait = monitor._closed.wait

    def wait(seconds):
        waits.append((threading.current_thread().name, seconds))
        return original_wait(seconds)

    monitor._sleep = wait
    monitor.start()
    try:
        # The first check does not wait for the first interval.
        assert polled.wait(2.0)
        assert monitor.warning == (("PHOTOS", 92),)
    finally:
        monitor.close()
    assert monitor._thread is not None and not monitor._thread.is_alive()
    assert waits and waits[0] == ("pifilm-disk", 3600.0)
    assert monitor._thread.daemon


def test_the_poller_polls_again_after_each_interval():
    fs = FakeFs({"/photos": (2, _vfs_for(50)), "/": (1, _vfs_for(50))})
    three = threading.Event()
    count = [0]
    monitor = DiskMonitor((("PHOTOS", Path("/photos")), ("SD", Path("/"))), 90,
                          statvfs=fs.statvfs, stat=fs.stat, interval_s=0.0,
                          sleep=lambda s: None)
    real_poll = monitor.poll_once

    def poll():
        real_poll()
        count[0] += 1
        if count[0] >= 3:
            three.set()

    monitor.poll_once = poll
    monitor.start()
    try:
        assert three.wait(2.0)
    finally:
        monitor.close()


def test_close_without_start_and_twice_is_harmless():
    monitor = DiskMonitor((), 90)
    monitor.close()
    monitor.close()


def test_the_real_root_filesystem_can_be_measured():
    usage = measure(os.sep)
    assert 0 <= usage.percent <= 100 and usage.free_bytes >= 0
