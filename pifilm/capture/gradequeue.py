"""A bounded queue of grades behind the shutter, with pending entries on disk.

The camera is free for the next photograph once the original is saved; the
full-resolution grade is handed to the one worker here. This module knows
nothing about cameras, pipelines or what a grade is: a job is an opaque payload
and ``run`` is whatever the session wants done with it.

Why the queue is bounded, and why ``submit`` waits instead of failing: every
waiting job holds a full frame (about 37 MB at 12 MP on a Pi with 4 GB), so the
number waiting has to have a ceiling. When the ceiling is reached the next
capture waits for a slot. It never fails and never drops a photograph. The job
being run does not count against the ceiling, so ``depth`` is the number of
frames waiting, not the number in memory.

Why ``submit`` can return ``False``: stopping the service while a capture waits
for a slot must not deadlock. ``close`` wakes every waiting ``submit``, which
then returns ``False``; the caller still owns the job and decides what to do
with it. A ``queue.Queue`` cannot wake a blocked ``put``, which is why this is a
``Condition`` over two deques.

Why there are entry files: a grade that has been accepted but not finished
exists only in memory, and a power cut would leave an original with no graded
file and no record. Before a job is queued the session writes a small JSON entry
naming what is needed to do the grade again; it removes the entry after the
record is written. At start-up the entries still on disk are the grades to redo,
passed in as ``backlog``. The backlog is not bounded by ``depth`` because those
jobs carry no frame yet (the original is read from disk when the job runs) and
because refusing them would lose photographs.

Why a failure is counted: a grade that raises (disk full, unreadable original)
must not kill the worker or block later shots, so the worker logs it and carries
on. The entry stays on disk with one more attempt against it, so the grade is
tried again at the next start. An entry that fails ``MAX_ATTEMPTS`` times is
renamed ``*.failed``: it is kept for a person to look at, and it can no longer
stop or slow a start-up. Entries that cannot be read at all are set aside the
same way.

Entry files are written to a temporary file and moved into place, so a reader
sees the old entry, the new one, or none, never half of one. Temporary files
start with a dot and end ``.tmp``, so ``read_entries`` never takes one for an
entry.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENTRY_VERSION = 1
MAX_ATTEMPTS = 2


def default_state_dir() -> Path:
    """$XDG_STATE_HOME/pifilm/pending, or ~/.local/state/pifilm/pending."""
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "pifilm" / "pending"


def write_entry(state_dir: Path, name: str, payload: dict) -> Path:
    """Write ``<state_dir>/<name>.json`` atomically and return its path.

    The payload gains ``"version": ENTRY_VERSION`` and ``"attempts": 0`` unless it
    has them. The directory is created with mode 0o700: entries name files in the
    user's photo folder and are nobody else's business.
    """
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError(f"entry name must be a plain file name, got {name!r}")
    state_dir = Path(state_dir)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = state_dir / f"{name}.json"
    _write_atomic(path, {"version": ENTRY_VERSION, "attempts": 0, **payload})
    # The rename is in the directory, not in the file: without this a power cut
    # just after the shot could lose the entry, which exists to survive one.
    _fsync_dir(state_dir)
    return path


def read_entries(state_dir: Path) -> list[tuple[Path, dict]]:
    """Every ``*.json`` entry, sorted by file name. A missing directory is an empty list.

    An entry that is not valid JSON, not an object, or of another version is renamed
    to ``*.failed`` and skipped. This runs at start-up, so it never raises: a bad
    entry must not stop the camera from starting.
    """
    try:
        paths = sorted(Path(state_dir).glob("*.json"), key=lambda p: p.name)
    except OSError:
        return []
    entries: list[tuple[Path, dict]] = []
    for path in paths:
        payload = _load(path)
        if payload is None or payload.get("version") != ENTRY_VERSION:
            _set_aside(path)
            continue
        entries.append((path, payload))
    return entries


def note_failure(path: Path) -> bool:
    """Count one failed attempt against the entry; ``True`` while it may be tried again.

    The count is rewritten atomically. At ``MAX_ATTEMPTS`` the file is renamed to
    ``<name>.failed`` and ``False`` is returned. A missing file returns ``False``,
    and so does an entry that can no longer be read, which is set aside. Never
    raises: it is called by the worker between jobs.
    """
    try:
        path = Path(path)
        if not path.is_file():
            return False
        payload = _load(path)
        if payload is None:
            _set_aside(path)
            return False
        try:
            attempts = int(payload.get("attempts", 0)) + 1
        except (TypeError, ValueError):
            attempts = MAX_ATTEMPTS
        payload["attempts"] = attempts
        if attempts >= MAX_ATTEMPTS:
            try:
                _write_atomic(path, payload)
            except Exception:
                pass  # the count is for a person reading the file; setting it aside matters
            _set_aside(path)
            return False
        try:
            _write_atomic(path, payload)
        except Exception:
            # A count that cannot be written (disk full) would never rise, and the
            # entry would be retried at every start for ever.
            _set_aside(path)
            return False
        return True
    except Exception:
        return False


def remove_entry(path: Path | None) -> None:
    """Delete the entry; a missing file or ``None`` is fine."""
    if path is None:
        return
    Path(path).unlink(missing_ok=True)


def _load(path: Path) -> dict | None:
    """The entry's object, or ``None`` when it cannot be read as one."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers bad JSON and bad UTF-8
        return None
    return payload if isinstance(payload, dict) else None


def _set_aside(path: Path) -> None:
    """Rename an entry to ``*.failed`` so it is kept but never read again."""
    try:
        os.replace(path, path.with_suffix(".failed"))
    except OSError:
        pass


def _fsync_dir(directory: Path) -> None:
    """Make a rename in ``directory`` durable, where the platform allows it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, payload: dict) -> None:
    """Write next to ``path`` and move into place, so no reader sees half a file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class GradeJob:
    """One grade to do. The queue reads nothing from ``payload``."""

    payload: Any  # opaque to the queue; the session's own job data
    entry: Path | None = None  # its pending-entry file, if any
    queued_at: float = 0.0  # time.perf_counter() when accepted, set by the queue
    started_at: float = 0.0  # time.perf_counter() when the worker took it, set by the queue


class GradeQueue:
    """Run grade jobs one at a time on a worker, at most ``depth`` waiting."""

    def __init__(
        self,
        run: Callable[[GradeJob], None],
        *,
        depth: int = 3,
        backlog: Iterable[GradeJob] = (),
        log: Callable[[str], None] = print,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._run_job = run
        self._depth = depth
        self._log = log
        # One condition guards everything below. It is notified when a slot
        # frees (for submit), when a job arrives (for the worker) and on close
        # (for both), so every waiter rechecks its own predicate.
        self._cond = threading.Condition()
        self._backlog: deque[GradeJob] = deque(backlog)
        self._waiting: deque[GradeJob] = deque()
        self._running = False
        self._closed = False
        now = time.perf_counter()
        for job in self._backlog:
            job.queued_at = now
        self._worker = threading.Thread(target=self._work, name="pifilm-grade", daemon=True)
        self._worker.start()

    def submit(self, job: GradeJob) -> bool:
        """Queue ``job``, waiting for a slot if ``depth`` jobs are already waiting.

        ``False`` means the queue is closed (before the call, or while it waited)
        and the job was not taken: the caller still owns it.
        """
        with self._cond:
            while not self._closed and len(self._waiting) >= self._depth:
                self._cond.wait()
            if self._closed:
                return False
            job.queued_at = time.perf_counter()
            self._waiting.append(job)
            self._cond.notify_all()
            return True

    @property
    def pending(self) -> int:
        """Jobs waiting, backlog not yet run, and the one running. 0 means nothing left."""
        with self._cond:
            return len(self._waiting) + len(self._backlog) + int(self._running)

    def close(self, timeout: float | None = None) -> bool:
        """Stop accepting, let the worker finish what was accepted, and join it.

        ``True`` when the worker finished, ``False`` when ``timeout`` ran out first.
        Calling it again is harmless, and is how to wait longer after a timeout.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if threading.current_thread() is self._worker:
            return False  # a job closing its own queue cannot wait for itself
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _work(self) -> None:
        try:
            while self._work_one():
                pass
        finally:
            # However the worker ends, nobody may be left waiting for a slot it
            # will never free.
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _work_one(self) -> bool:
        """Run the next job; ``False`` when the queue is closed and empty."""
        with self._cond:
            while not (self._backlog or self._waiting or self._closed):
                self._cond.wait()
            if self._backlog:
                job = self._backlog.popleft()
            elif self._waiting:
                job = self._waiting.popleft()
            else:
                return False
            # Running is set under the same lock as the pop, so ``pending``
            # never dips to 0 between a job leaving the queue and starting.
            self._running = True
            job.started_at = time.perf_counter()
            self._cond.notify_all()  # a slot may have freed
        try:
            self._run_job(job)
        except Exception as exc:
            self._failed(job, exc)
        finally:
            with self._cond:
                self._running = False
        return True

    def _failed(self, job: GradeJob, exc: Exception) -> None:
        """Log and count a failed grade. Nothing here may end the worker."""
        try:
            self._log(f"grade failed: {exc}")
        except Exception:
            pass  # a closed stdout must not cost the photographs still queued
        if job.entry is not None:
            note_failure(job.entry)
