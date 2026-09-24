import importlib
import sys
import threading
import time

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession
from pifilm.capture.camera import FakeCamera, synthetic_frame
from pifilm.capture.controller import CaptureController
from pifilm.capture.errors import CameraError
from pifilm.display import DisplayError, viewfinder
from pifilm.display.cst3530 import TouchPoint
from pifilm.display.ui import BAR_TOP, SHUTTER_CENTRE
from pifilm.display.viewfinder import EV_STEP, ViewfinderLoop
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

DEFAULT_METADATA = {"ExposureTime": 4000, "AnalogueGain": 1.0, "Lux": 500.0}


class FakeDisplay:
    def __init__(self, fail_after=None):
        self.images = []
        self.fail_after = fail_after
        self.closed = False

    def show(self, image):
        if self.fail_after is not None and len(self.images) >= self.fail_after:
            raise DisplayError("spi")
        self.images.append(image)

    def close(self):
        self.closed = True


class ScriptedTouch:
    def __init__(self):
        self.script = []  # list of point lists, consumed one per read

    def tap(self, x, y):
        self.script += [[TouchPoint(x, y, 10)], []]

    def read(self):
        return self.script.pop(0) if self.script else []

    def close(self):
        pass


class GlitchyTouch(ScriptedTouch):
    """A panel that NAKs on the reads listed in ``fail_on``, as a shared I2C bus does."""

    def __init__(self, fail_on=(0,)):
        super().__init__()
        self.reads = 0
        self._fail_on = set(fail_on)

    def read(self):
        n, self.reads = self.reads, self.reads + 1
        if n in self._fail_on:
            raise DisplayError("touch I2C read failed: [Errno 121] Remote I/O error")
        return super().read()


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _make_controller(tmp_path, metadata=DEFAULT_METADATA):
    art = tmp_path / "art"
    if not art.exists():
        write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    camera = FakeCamera([synthetic_frame(48, 64)], metadata=metadata)
    session = CaptureSession(camera, Pipeline(Artifacts.load(art)), tmp_path / "shots",
                             seed_rng=np.random.default_rng(0))
    return camera, CaptureController(session)


@pytest.fixture
def controller(tmp_path):
    camera, ctl = _make_controller(tmp_path)
    yield camera, ctl
    ctl.close()


def _loop(camera, ctl, touch=None, display=None, clock=None, **kw):
    touch = touch or ScriptedTouch()
    display = display or FakeDisplay()
    clock = clock or FakeClock()
    loop = ViewfinderLoop(camera, ctl, display, touch, clock=clock, log=lambda *a: None, **kw)
    return loop, touch, display, clock


def _record_renderers(monkeypatch):
    """Record which review renderer ran, still drawing the real thing."""
    calls = {"review": [], "message": []}
    real_review, real_message = viewfinder.render_review, viewfinder.render_message

    def review(rgb, caption):
        calls["review"].append(caption)
        return real_review(rgb, caption)

    def message(title, detail):
        calls["message"].append((title, detail))
        return real_message(title, detail)

    monkeypatch.setattr(viewfinder, "render_review", review)
    monkeypatch.setattr(viewfinder, "render_message", message)
    return calls


def _until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met")


def test_live_step_shows_a_frame(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert loop.state == "LIVE"
    assert len(display.images) == 1 and display.images[0].size == (320, 240)


def test_shutter_tap_submits_through_the_controller_and_reviews_the_result(
    controller, monkeypatch,
):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    touch.tap(*SHUTTER_CENTRE)
    loop.step()  # down
    loop.step()  # up -> tap -> submit
    _until(lambda: ctl.snapshot().finished_count == 1)
    before = len(display.images)
    loop.step()
    assert loop.state == "REVIEW"
    assert ctl.snapshot().last_finished_job.state == "complete"
    assert len(display.images) == before + 1
    assert calls["message"] == []
    assert len(calls["review"]) == 1 and "ISO" in calls["review"][0]


def test_review_ends_on_tap_or_timeout(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, review_timeout=30.0)
    ctl.submit("ext")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    clock.t += 31
    loop.step()
    assert loop.state == "LIVE"
    ctl.submit("ext2")
    _until(lambda: ctl.snapshot().finished_count == 2)
    loop.step()
    assert loop.state == "REVIEW"
    touch.tap(160, 120)
    loop.step()
    loop.step()
    assert loop.state == "LIVE"


def test_ev_taps_step_by_a_third_and_clamp(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    for _ in range(8):
        touch.tap(310, BAR_TOP + 10)
    for _ in range(16):
        loop.step()
    assert loop.ev_comp == pytest.approx(2.0)
    assert camera.ev == pytest.approx(2.0)
    touch.tap(10, BAR_TOP + 10)
    loop.step()
    loop.step()
    assert loop.ev_comp == pytest.approx(2.0 - EV_STEP)


def test_failed_job_shows_a_message_in_review(tmp_path, monkeypatch):
    class Boom:
        camera = None

        def capture(self):
            raise CameraError("lens cap")

    ctl = CaptureController(Boom())
    try:
        calls = _record_renderers(monkeypatch)
        camera = FakeCamera([synthetic_frame(48, 64)])
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("x")
        _until(lambda: ctl.snapshot().finished_count == 1)
        before = len(display.images)
        loop.step()
        assert loop.state == "REVIEW"
        assert len(display.images) == before + 1
        assert calls["review"] == []
        assert len(calls["message"]) == 1
        title, detail = calls["message"][0]
        assert title == "Capture failed" and "lens cap" in detail
    finally:
        ctl.close()


def test_zero_exposure_time_in_the_record_still_reviews(tmp_path, monkeypatch):
    camera, ctl = _make_controller(tmp_path, {"ExposureTime": 0, "AnalogueGain": "n/a"})
    try:
        calls = _record_renderers(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("zero")
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
        assert len(display.images) == 1
        assert calls["message"] == []
        assert calls["review"] == ["-  ISO -  EV 0"]
    finally:
        ctl.close()


def test_unreadable_graded_file_shows_the_review_fallback(controller, monkeypatch):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    ctl.submit("gone")
    _until(lambda: ctl.snapshot().finished_count == 1)
    ctl.snapshot().last_finished_job.result.pifilm.unlink()
    loop.step()
    assert loop.state == "REVIEW"
    assert len(display.images) == 1
    assert calls["review"] == []
    assert len(calls["message"]) == 1 and calls["message"][0][0] == "Review unavailable"


def test_camera_error_on_preview_keeps_the_loop_alive(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    original = camera.read
    camera.read = lambda *, full=True: (_ for _ in ()).throw(CameraError("busy"))
    loop.step()
    assert loop.state == "LIVE"
    camera.read = original
    loop.step()
    assert len(display.images) == 1


def test_touch_error_does_not_end_the_loop(controller):
    camera, ctl = controller
    touch = GlitchyTouch(fail_on=(0,))
    loop, _, display, clock = _loop(camera, ctl, touch=touch)
    loop.step()
    assert loop.state == "LIVE"
    assert len(display.images) == 1
    touch.tap(*SHUTTER_CENTRE)
    loop.step()
    loop.step()
    _until(lambda: ctl.snapshot().finished_count == 1)
    assert len(display.images) == 3


def test_touch_errors_do_not_count_toward_the_display_breaker(controller):
    camera, ctl = controller
    touch = GlitchyTouch(fail_on=range(20))
    loop, _, display, clock = _loop(camera, ctl, touch=touch, max_display_failures=5)
    for _ in range(10):
        loop.step()
    assert loop.state == "LIVE"
    assert len(display.images) == 10


def test_five_display_failures_end_the_loop(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=FakeDisplay(fail_after=0))
    with pytest.raises(DisplayError):
        for _ in range(5):
            loop.step()


def test_run_stops_on_event_and_paces_frames(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, frame_period=0.1)
    stop = threading.Event()
    calls = {"n": 0}
    real_step = loop.step

    def counting_step():
        calls["n"] += 1
        real_step()
        if calls["n"] == 3:
            stop.set()

    loop.step = counting_step
    loop.run(stop)
    assert calls["n"] == 3
    assert clock.t == pytest.approx(0.3, abs=0.01)


def test_viewfinder_imports_without_opencv(monkeypatch):
    """pifilm/display must stay importable where OpenCV is absent.

    OpenCV is deliberately not a base dependency, and ``pifilm.capture.camera``
    calls ``require_cv2()`` at module scope, so the viewfinder takes
    ``CameraError`` from ``pifilm.capture.errors`` instead.
    """
    import pifilm.display

    monkeypatch.setattr(pifilm.display, "viewfinder", viewfinder, raising=False)
    monkeypatch.setitem(sys.modules, "cv2", None)  # makes `import cv2` raise
    for name in ("pifilm.display.viewfinder", "pifilm.capture.errors", "pifilm.capture.camera"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    fresh = importlib.import_module("pifilm.display.viewfinder")

    assert fresh.ViewfinderLoop.__name__ == "ViewfinderLoop"
    assert "pifilm.capture.camera" not in sys.modules
