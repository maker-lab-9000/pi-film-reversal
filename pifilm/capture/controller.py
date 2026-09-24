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
        self._work: queue.Queue[str | None] = queue.Queue()
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
                request_id = self._work.get()
                if request_id is None:
                    return
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
        with self._lock:
            job = JobSnapshot(request_id, state, error_code, result, error_message)
            self._jobs[request_id] = job
            self._active_request_id = None
            self._finished_count += 1
            self._last_finished_job = job
            if state == "complete":
                self._last_completed_job = job
