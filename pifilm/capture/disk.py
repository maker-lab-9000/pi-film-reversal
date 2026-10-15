"""How full the photo volume and the system SD card are, for the viewfinder's warning.

The Pi has two filesystems that fill: the photo folder, normally its own fixed-size
volume (``docs/photo-volume.md``), and the system SD card at ``/``. Either one full
ends the service's usefulness quietly: a photo volume that is full cannot take the
next original, and a full root filesystem stops the journal, the pending grade
entries and the OS itself. So the capture app checks both once a minute and the
viewfinder shows an amber ``PHOTOS 92%`` / ``SD 91%`` tag while either is at or
above ``--disk-warn-percent`` (90 by default). A warning only: photographs are
still taken, because a shot that fits is worth more than a refusal.

The percentage is the one ``df`` prints as Use%, so the tag and a ``df -h`` over
SSH agree: ``used / (used + available)`` with ``used = (f_blocks - f_bfree)`` and
``available = f_bavail`` (blocks reserved for root count as neither), rounded up
to a whole percent as df does, so "91%" never hides a filesystem that is 90.1 %
full. The free space in the log lines is that same ``available``, in GB of 2**30
bytes, the unit ``df -h`` calls G.

When the photo folder is on the root filesystem (no separate volume, or a run on
a Mac), the two paths are one filesystem: it is measured once and called PHOTOS,
so one shortage does not show twice. A path that cannot be measured (a volume not
mounted yet, a folder not created yet) is left out after one log line naming it,
and is measured again at the next check; nothing a measurement raises leaves the
poller, whose thread must outlive any one bad minute.

The poller is a daemon thread, ``pifilm-disk``, shaped like the UPS poller in
``power.py``: it checks at once when started, then sleeps on an event that
``close`` sets, so stopping it never waits out the interval. The current warning
is read by display threads at every frame under a lock and never touches the disk.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WARN_PERCENT = 90
DEFAULT_INTERVAL_S = 60.0
PHOTOS_LABEL, SD_LABEL = "PHOTOS", "SD"
_GB = 1024 ** 3

DiskWarning = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Usage:
    percent: int      # df's Use%: whole, rounded up
    free_bytes: int   # available to an ordinary user (f_bavail)


def measure(path: str | os.PathLike, *, statvfs=os.statvfs) -> Usage:
    """The df Use% and the free bytes of the filesystem holding ``path``.

    Raises OSError when the path cannot be measured. Integer arithmetic, as df's own,
    so a filesystem that is exactly 90 % full is 90 and not 91 by a rounding error.
    """
    vfs = statvfs(path)
    used = (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize
    available = vfs.f_bavail * vfs.f_frsize
    total = used + available
    percent = 0 if total <= 0 else -(-used * 100 // total)
    return Usage(percent=percent, free_bytes=available)


def watched_paths(out_root: str | os.PathLike) -> tuple[tuple[str, Path], ...]:
    """The photo folder (PHOTOS) and the root filesystem (SD), in that order."""
    return ((PHOTOS_LABEL, Path(out_root)), (SD_LABEL, Path("/")))


class DiskMonitor:
    """Checks the filesystems on its own thread; ``warning`` never touches the disk."""

    def __init__(
        self,
        paths: Sequence[tuple[str, str | os.PathLike]],
        threshold: int = DEFAULT_WARN_PERCENT,
        *,
        log: Callable[[str], None] = print,
        statvfs=os.statvfs,
        stat=os.stat,
        interval_s: float = DEFAULT_INTERVAL_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) \
                or not 1 <= threshold <= 100:
            raise ValueError(f"threshold must be a whole percent from 1 to 100, got {threshold!r}")
        self._paths = tuple((label, Path(path)) for label, path in paths)
        self._threshold = threshold
        self._log = log
        self._statvfs = statvfs
        self._stat = stat
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._warning: DiskWarning = ()
        self._low: set[str] = set()        # labels last seen at or above the threshold
        self._failing: set[str] = set()    # labels whose last measurement failed
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._sleep = sleep if sleep is not None else self._closed.wait

    @property
    def warning(self) -> DiskWarning:
        """``(label, percent)`` for each low filesystem, photos first; empty when none."""
        with self._lock:
            return self._warning

    def _say(self, message: str) -> None:
        try:
            self._log(message)
        except Exception:
            pass  # a closed stdout must not end the poller

    def _measure_all(self) -> list[tuple[str, Usage]]:
        seen_devices: set[int] = set()
        measured = []
        for label, path in self._paths:
            try:
                device = self._stat(path).st_dev
                if device in seen_devices:
                    continue  # the same filesystem as an earlier path: shown once
                usage = measure(path, statvfs=self._statvfs)
            except Exception as exc:
                if label not in self._failing:
                    self._failing.add(label)
                    self._say(f"disk: cannot measure {label.lower()} ({path}): {exc}")
                continue
            self._failing.discard(label)
            seen_devices.add(device)
            measured.append((label, usage))
        return measured

    def poll_once(self) -> None:
        warning = []
        for label, usage in self._measure_all():
            name, free = label.lower(), usage.free_bytes / _GB
            if usage.percent >= self._threshold:
                warning.append((label, usage.percent))
                if label not in self._low:
                    self._low.add(label)
                    self._say(f"disk: {name} {usage.percent}% full ({free:.1f} GB free)")
            elif label in self._low:
                self._low.discard(label)
                self._say(f"disk: {name} back to {usage.percent}% full ({free:.1f} GB free)")
        with self._lock:
            self._warning = tuple(warning)

    def _run(self) -> None:
        while not self._closed.is_set():
            self.poll_once()
            self._sleep(self._interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pifilm-disk", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
