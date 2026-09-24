from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import Mock

from pifilm.capture.camera import CameraError
from pifilm.capture.controller import CaptureController


@dataclass
class Result:
    value: str


class BlockingSession:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.camera = Mock()

    def capture(self) -> Result:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return Result("saved")


def wait_for(controller, request_id: str, state: str):
    for _ in range(200):
        job = controller.status(request_id)
        if job is not None and job.state == state:
            return job
        threading.Event().wait(0.01)
    raise AssertionError(f"job {request_id!r} did not become {state!r}")


def test_controller_captures_once_for_an_accepted_request():
    """Removing the single worker's capture call would leave the job unfinished."""
    from pifilm.capture.controller import CaptureController

    session = BlockingSession()
    controller = CaptureController(session)
    try:
        accepted = controller.submit("first")
        assert accepted.request_id == "first"
        assert accepted.state in {"queued", "processing"}
        assert session.started.wait(timeout=1)
        assert session.calls == 1
        session.release.set()
        complete = wait_for(controller, "first", "complete")
        assert complete.result == Result("saved")
        assert session.calls == 1
    finally:
        session.release.set()
        controller.close()


def test_controller_allows_only_one_active_job_and_returns_busy_for_another_id():
    """Dropping the active-job guard would start a second camera capture."""
    from pifilm.capture.controller import CaptureController

    session = BlockingSession()
    controller = CaptureController(session)
    try:
        controller.submit("first")
        assert session.started.wait(timeout=1)
        busy = controller.submit("second")
        assert busy.request_id == "second"
        assert busy.state == "failed"
        assert busy.error_code == "busy"
        assert controller.status("second") is None
        assert session.calls == 1
    finally:
        session.release.set()
        controller.close()


def test_controller_duplicate_id_returns_the_original_job_without_recapturing():
    """Treating a duplicate as new would make retries expose the camera twice."""
    from pifilm.capture.controller import CaptureController

    session = BlockingSession()
    controller = CaptureController(session)
    try:
        original = controller.submit("same")
        assert session.started.wait(timeout=1)
        duplicate = controller.submit("same")
        assert duplicate == controller.status("same")
        assert duplicate.request_id == original.request_id
        assert duplicate.error_code is None
        assert session.calls == 1
    finally:
        session.release.set()
        controller.close()


def test_camera_error_finishes_job_and_allows_a_later_request():
    """Keeping a failed job active would make the camera permanently busy."""
    from pifilm.capture.controller import CaptureController

    class FlakySession:
        def __init__(self) -> None:
            self.calls = 0
            self.camera = Mock()

        def capture(self):
            self.calls += 1
            if self.calls == 1:
                raise CameraError("dropped frame")
            return Result("recovered")

    session = FlakySession()
    controller = CaptureController(session)
    try:
        failed = wait_for(controller, controller.submit("failed").request_id, "failed")
        assert failed.error_code == "camera_error"
        complete = wait_for(controller, controller.submit("retry").request_id, "complete")
        assert complete.result == Result("recovered")
        assert session.calls == 2
    finally:
        controller.close()


def test_snapshots_are_immutable_values():
    """Mutating a returned status must not change the controller's published state."""
    from dataclasses import FrozenInstanceError

    from pifilm.capture.controller import CaptureController

    session = BlockingSession()
    controller = CaptureController(session)
    try:
        job = controller.submit("first")
        try:
            job.state = "failed"
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("job snapshot was mutable")
    finally:
        session.release.set()
        controller.close()


class _Session:
    def __init__(self, fail=False):
        self.camera = None
        self.fail = fail

    def capture(self):
        if self.fail:
            raise CameraError("boom")
        return "result"


def _wait_finished(controller, request_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = controller.status(request_id)
        if job is not None and job.state in ("complete", "failed"):
            return job
        time.sleep(0.005)
    raise AssertionError("job did not finish")


def test_snapshot_counts_finished_jobs_of_both_states():
    controller = CaptureController(_Session())
    try:
        assert controller.snapshot().finished_count == 0
        assert controller.snapshot().last_finished_job is None
        controller.submit("a")
        _wait_finished(controller, "a")
        snap = controller.snapshot()
        assert snap.finished_count == 1
        assert snap.last_finished_job.request_id == "a"
        assert snap.last_finished_job.state == "complete"
    finally:
        controller.close()


def test_failed_job_counts_as_finished_but_not_completed():
    controller = CaptureController(_Session(fail=True))
    try:
        controller.submit("b")
        _wait_finished(controller, "b")
        snap = controller.snapshot()
        assert snap.finished_count == 1
        assert snap.last_finished_job.state == "failed"
        assert snap.last_completed_job is None
    finally:
        controller.close()
