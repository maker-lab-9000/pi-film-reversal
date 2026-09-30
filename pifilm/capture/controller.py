"""Serialized, idempotent capture jobs shared by local and remote triggers."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Literal, Protocol

from .camera import CameraError

JobState = Literal["queued", "processing", "complete", "failed"]


class CaptureCallable(Protocol):
    camera: object

    def capture(self) -> object: ...


@dataclass(frozen=True)
class JobSnapshot:
    """An immutable view of one accepted capture request."""

    request_id: str
    state: JobState
    error_code: str | None = None
    result: object | None = None
    error_message: str | None = None

    @property
    def id(self) -> str:
        """Short alias for protocol clients that call the request identifier an ID."""
        return self.request_id


@dataclass(frozen=True)
class ControllerSnapshot:
    """An immutable summary safe to render from another thread."""

    active_job: JobSnapshot | None
    last_completed_job: JobSnapshot | None
    closed: bool
    # Every job that ended, complete or failed. A display polls this counter to
    # notice a shot taken from any trigger (Stick, SPACE, LCD) without the
    # controller knowing that displays exist.
    finished_count: int = 0
    last_finished_job: JobSnapshot | None = None
    # Double exposure, as the session reported it when the last job or toggle
    # finished. Updated under the same lock as ``finished_count``, so a display that
    # sees a job finish also sees the count that job left behind.
    double_exposure: bool = False
    exposures_taken: int = 0


@dataclass(frozen=True)
class _DoubleToggle:
    enabled: bool


class CaptureController:
    """Own one camera session in one worker, accepting at most one job at a time."""

    def __init__(self, session: CaptureCallable) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._active_request_id: str | None = None
        self._last_completed_job: JobSnapshot | None = None
        self._finished_count = 0
        self._last_finished_job: JobSnapshot | None = None
        self._closed = False
        self._double = self._read_double()
        self._work: queue.Queue[str | _DoubleToggle | None] = queue.Queue()
        self._worker = threading.Thread(
            target=self._run, name="pifilm-capture-worker", daemon=True,
        )
        self._worker.start()

    def submit(self, request_id: str) -> JobSnapshot:
        """Accept ``request_id`` once, or return its current immutable snapshot.

        A second, distinct request is not accepted while a capture is active.  It
        receives a standalone failed snapshot with ``busy`` so a transport can
        translate it to its own conflict response without creating a fake job.
        """
        with self._lock:
            if request_id in self._jobs:
                return self._jobs[request_id]
            if self._closed:
                return JobSnapshot(request_id, "failed", "closed")
            if self._active_request_id is not None:
                return JobSnapshot(request_id, "failed", "busy")
            job = JobSnapshot(request_id, "queued")
            self._jobs[request_id] = job
            self._active_request_id = request_id
            self._work.put(request_id)
            return job

    def set_double_exposure(self, enabled: bool) -> None:
        """Queue a double-exposure toggle behind any capture already accepted.

        It runs on the worker thread, which is the only thread that touches the
        session, so it can never change the mode in the middle of a pair's capture.
        """
        with self._lock:
            if self._closed:
                return
            self._work.put(_DoubleToggle(bool(enabled)))

    def status(self, request_id: str) -> JobSnapshot | None:
        with self._lock:
            return self._jobs.get(request_id)

    def snapshot(self) -> ControllerSnapshot:
        with self._lock:
            active = (
                self._jobs[self._active_request_id]
                if self._active_request_id is not None
                else None
            )
            return ControllerSnapshot(
                active,
                self._last_completed_job,
                self._closed,
                self._finished_count,
                self._last_finished_job,
                self._double[0],
                self._double[1],
            )

    def close(self) -> None:
        """Finish an in-flight capture, then close the camera on its worker thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._work.put(None)
        self._worker.join()

    def _run(self) -> None:
        try:
            while True:
                item = self._work.get()
                if item is None:
                    return
                if isinstance(item, _DoubleToggle):
                    self._apply_toggle(item.enabled)
                    continue
                request_id = item
                self._set_processing(request_id)
                try:
                    result = self._session.capture()
                except CameraError as exc:
                    self._finish(
                        request_id, "failed", error_code="camera_error", error_message=str(exc),
                    )
                except OSError as exc:
                    self._finish(
                        request_id, "failed", error_code="camera_error", error_message=str(exc),
                    )
                except Exception as exc:
                    self._finish(
                        request_id, "failed", error_code="capture_error", error_message=str(exc),
                    )
                else:
                    self._finish(request_id, "complete", result=result)
        finally:
            close = getattr(getattr(self._session, "camera", None), "close", None)
            if callable(close):
                close()

    def _read_double(self) -> tuple[bool, int]:
        state = getattr(self._session, "double_state", None)
        if not isinstance(state, tuple) or len(state) != 2:
            return False, 0
        return bool(state[0]), int(state[1])

    def _apply_toggle(self, enabled: bool) -> None:
        setter = getattr(self._session, "set_double_exposure", None)
        if callable(setter):
            setter(enabled)
        double = self._read_double()
        with self._lock:
            self._double = double

    def _set_processing(self, request_id: str) -> None:
        with self._lock:
            self._jobs[request_id] = JobSnapshot(request_id, "processing")

    def _finish(
        self,
        request_id: str,
        state: Literal["complete", "failed"],
        *,
        error_code: str | None = None,
        result: object | None = None,
        error_message: str | None = None,
    ) -> None:
        double = self._read_double()
        with self._lock:
            job = JobSnapshot(request_id, state, error_code, result, error_message)
            self._jobs[request_id] = job
            self._active_request_id = None
            self._finished_count += 1
            self._last_finished_job = job
            if state == "complete":
                self._last_completed_job = job
            self._double = double
