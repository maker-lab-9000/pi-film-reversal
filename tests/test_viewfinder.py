import threading
import time

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession
from pifilm.capture.camera import CameraError, FakeCamera, synthetic_frame
from pifilm.capture.controller import CaptureController
from pifilm.display import DisplayError
from pifilm.display.cst3530 import TouchPoint
from pifilm.display.ui import BAR_TOP, SHUTTER_CENTRE
from pifilm.display.viewfinder import EV_STEP, ViewfinderLoop
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline


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


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def controller(tmp_path):
    art = tmp_path / "art"
    write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    camera = FakeCamera([synthetic_frame(48, 64)], metadata={"ExposureTime": 4000,
                                                             "AnalogueGain": 1.0, "Lux": 500.0})
    session = CaptureSession(camera, Pipeline(Artifacts.load(art)), tmp_path / "shots",
                             seed_rng=np.random.default_rng(0))
    ctl = CaptureController(session)
    yield camera, ctl
    ctl.close()


def _loop(camera, ctl, touch=None, display=None, clock=None, **kw):
    touch = touch or ScriptedTouch()
    display = display or FakeDisplay()
    clock = clock or FakeClock()
    loop = ViewfinderLoop(camera, ctl, display, touch, clock=clock, log=lambda *a: None, **kw)
    return loop, touch, display, clock


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


def test_shutter_tap_submits_through_the_controller_and_reviews_the_result(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    touch.tap(*SHUTTER_CENTRE)
    loop.step()  # down
    loop.step()  # up -> tap -> submit
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    assert ctl.snapshot().last_finished_job.state == "complete"


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


def test_failed_job_shows_a_message_in_review(tmp_path):
    class Boom:
        camera = None

        def capture(self):
            raise CameraError("lens cap")

    ctl = CaptureController(Boom())
    try:
        camera = FakeCamera([synthetic_frame(48, 64)])
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("x")
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
    finally:
        ctl.close()


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
