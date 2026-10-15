import dataclasses
import importlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession
from pifilm.capture.camera import FakeCamera, synthetic_frame
from pifilm.capture.controller import CaptureController
from pifilm.capture.errors import CameraError, CameraStalled
from pifilm.display import DisplayError, viewfinder
from pifilm.display.cst3530 import TouchPoint
from pifilm.display.liveview import (
    BAR_TOP,
    CELL_HIT_X0,
    CELL_PITCH,
    DOUBLE_BOX,
    MINUS_CENTRE,
    PLUS_CENTRE,
    SHUTTER_CENTRE,
    Action,
    Cell,
)
from pifilm.display.viewfinder import (
    CAMERA_STALL_LIMIT,
    ISO_CELL,
    NOTICE_SECONDS,
    PRESS_FLASH_SECONDS,
    SHUTTER_CELL,
    TAP_MAX_HOLD,
    WB_CELL,
    ViewfinderLoop,
)
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

DEFAULT_METADATA = {"ExposureTime": 4000, "AnalogueGain": 1.0, "Lux": 500.0}
# A frame with all three metered values, none of them on a step: 1/250 s is a
# shutter step, ISO 760 is nearest 800 and 4249 K is nearest 4000 K.
METERED = {"ExposureTime": 4000, "AnalogueGain": 7.6, "ColourTemperature": 4249, "Lux": 500.0}


class FakeDisplay:
    def __init__(self, fail_after=None):
        self.images = []
        self.levels = []
        self.fail_after = fail_after
        self.closed = False

    def show(self, image):
        if self.fail_after is not None and len(self.images) >= self.fail_after:
            raise DisplayError("spi")
        self.images.append(image)

    def backlight(self, percent):
        self.levels.append(percent)

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


class GatedSession:
    """A session whose capture blocks until released, so a job stays active.

    Without it the fake capture finishes in milliseconds and "while the job is
    processing" is a race rather than a state the loop can be stepped through.
    """

    def __init__(self, session):
        self._session = session
        self.camera = session.camera
        self.started = threading.Event()
        self.release = threading.Event()

    def capture(self):
        self.started.set()
        assert self.release.wait(5.0), "capture was never released"
        return self._session.capture()

    def __getattr__(self, name):
        # Only called for attributes the wrapper lacks: double_state and
        # set_double_exposure fall through to the real session.
        return getattr(self._session, name)


def _gated_controller(tmp_path):
    art = tmp_path / "art"
    if not art.exists():
        write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    camera = FakeCamera([synthetic_frame(48, 64)], metadata=DEFAULT_METADATA)
    session = GatedSession(CaptureSession(camera, Pipeline(Artifacts.load(art)),
                                          tmp_path / "shots",
                                          seed_rng=np.random.default_rng(0)))
    return camera, CaptureController(session), session


def _count_reads(camera):
    """Wrap ``camera.read`` with a counter, returning the counter dict."""
    reads = {"n": 0}
    real = camera.read

    def counting(*, full=True):
        reads["n"] += 1
        return real(full=full)

    camera.read = counting
    return reads


def _spy_processing(monkeypatch):
    calls = {"n": 0}
    real = viewfinder.render_processing

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(viewfinder, "render_processing", spy)
    return calls


@pytest.fixture
def controller(tmp_path):
    camera, ctl = _make_controller(tmp_path)
    yield camera, ctl
    ctl.close()


def _loop(camera, ctl, touch=None, display=None, clock=None, **kw):
    touch = touch or ScriptedTouch()
    display = display or FakeDisplay()
    clock = clock or FakeClock()
    log = kw.pop("log", lambda *a: None)
    loop = ViewfinderLoop(camera, ctl, display, touch, clock=clock, log=log, **kw)
    return loop, touch, display, clock


def _record_renderers(monkeypatch):
    """Record which review renderer ran, still drawing the real thing."""
    calls = {"review": [], "message": []}
    real_review, real_message = viewfinder.render_review, viewfinder.render_message

    def review(rgb, caption, **kwargs):
        calls["review"].append(caption)
        return real_review(rgb, caption, **kwargs)

    def message(title, detail, **kwargs):
        calls["message"].append((title, detail))
        return real_message(title, detail, **kwargs)

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
    assert calls["review"] == ["1/250  ISO 100"]


def test_processing_shows_colour_bars_instead_of_the_live_view(tmp_path, monkeypatch):
    """A grade takes about 3 s; the LCD must say so the way the Stick does."""
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        bars = _spy_processing(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        touch.tap(*SHUTTER_CENTRE)
        loop.step()  # down
        loop.step()  # up -> tap -> submit
        _until(lambda: session.started.is_set())
        reads = _count_reads(camera)
        loop.step()
        assert bars["n"] == 1
        assert reads["n"] == 0  # no live frame while the job runs
        assert loop.state == "LIVE"
        session.release.set()
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
    finally:
        session.release.set()
        ctl.close()


def test_processing_screen_is_pushed_once_not_every_step(tmp_path, monkeypatch):
    """Re-blitting identical bars at 10 fps would waste the SPI bus for 3 s."""
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        bars = _spy_processing(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("gated")
        _until(lambda: session.started.is_set())
        before = len(display.images)
        for _ in range(3):
            loop.step()
        assert len(display.images) == before + 1
        assert bars["n"] == 1
    finally:
        session.release.set()
        ctl.close()


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
        assert calls["review"] == ["-  ISO -"]
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


def test_touch_errors_are_logged_at_most_once_per_interval(controller):
    """A panel that never answers must not write a line at 10 Hz for as long as
    the service runs."""
    camera, ctl = controller
    logs = []
    touch = GlitchyTouch(fail_on=range(20))
    loop, _, display, clock = _loop(camera, ctl, touch=touch, log=logs.append)
    for _ in range(10):
        loop.step()
    touch_lines = [line for line in logs if line.startswith("touch:")]
    assert len(touch_lines) <= 2
    assert "Remote I/O error" in touch_lines[0]


def test_frame_rate_excludes_the_review_pause(controller):
    """Acceptance item 1 reads the fps line to judge the live view; counting
    the up-to-30 s review screen as dropped frames makes it lie."""
    camera, ctl = controller
    logs = []
    # Idle dimming off: the clock jumps 5 minutes, which would darken the screen.
    loop, touch, display, clock = _loop(
        camera, ctl, log=logs.append, dim_after=0, off_after=0,
    )
    loop.step()                       # a live frame
    touch.tap(*SHUTTER_CENTRE)
    loop.step()                       # finger down
    loop.step()                       # release -> submit, plus a live frame
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    clock.t = 300.0
    loop.step()                       # review times out
    assert loop.state == "LIVE"
    clock.t = 310.0
    loop.step()                       # one frame, 10 s into a fresh window
    assert [line for line in logs if line.startswith("viewfinder:")] == ["viewfinder: 0.1 fps"]


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


def _spy_live(monkeypatch):
    """Record the ``MeterReading`` of every live frame, still drawing it."""
    readings = []
    real = viewfinder.render_live

    def spy(frame_rgb, reading, state, **kwargs):
        readings.append(reading)
        return real(frame_rgb, reading, state, **kwargs)

    monkeypatch.setattr(viewfinder, "render_live", spy)
    return readings


def _spy_states(monkeypatch):
    """Record the ``LiveState`` of every live frame, still drawing it."""
    states = []
    real = viewfinder.render_live

    def spy(frame_rgb, reading, state, **kwargs):
        states.append(state)
        return real(frame_rgb, reading, state, **kwargs)

    monkeypatch.setattr(viewfinder, "render_live", spy)
    return states


def _sharp_frame(shape=(96, 128), seed=0):
    """Rectangles of random size and grey: edges at every scale, like a subject."""
    rng = np.random.default_rng(seed)
    frame = np.full(shape, 128.0)
    for _ in range(60):
        h, w = rng.integers(3, shape[0] // 3), rng.integers(3, shape[1] // 3)
        y, x = rng.integers(0, shape[0] - h), rng.integers(0, shape[1] - w)
        frame[y:y + h, x:x + w] = rng.integers(20, 236)
    return np.repeat(frame.astype(np.uint8)[:, :, None], 3, axis=2)


def _blurred(rgb, k=15):
    """Near-Gaussian (three box passes): far out of focus at k=15."""
    out = rgb.astype(np.float32)
    pad = k // 2
    for _ in range(3):
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        acc = np.zeros_like(out)
        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + rgb.shape[0], dx:dx + rgb.shape[1]]
        out = acc / (k * k)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_focus_bar_peaks_on_a_sharp_frame_and_drops_when_it_blurs(controller, monkeypatch):
    """The IMX477 is focused by hand: racking past best focus must show on the
    bar, while the mark stays at the best level seen."""
    _, ctl = controller
    readings = _spy_live(monkeypatch)
    sharp = _sharp_frame()
    camera = FakeCamera([sharp, _blurred(sharp)])
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    clock.t += 0.1
    loop.step()
    assert readings[0].focus >= 0.8
    assert readings[1].focus <= 0.2
    assert readings[1].focus_peak == pytest.approx(readings[0].focus, rel=0.05)


def test_an_out_of_focus_lens_reads_low_from_start_up(controller, monkeypatch):
    """The defect found on the device: a lens left out of focus showed a full bar,
    because the bar was relative to its own recent peak and a blurred frame was
    the only thing it had seen. The bar must stay low however long it looks."""
    _, ctl = controller
    readings = _spy_live(monkeypatch)
    camera = FakeCamera([_blurred(_sharp_frame())] * 40)
    loop, touch, display, clock = _loop(camera, ctl)
    for _ in range(40):
        loop.step()
        clock.t += 0.1
    assert all(r.focus <= 0.2 for r in readings)
    assert readings[-1].focus_peak <= 0.2


def test_the_focus_peak_decays_by_wall_time_across_a_review(controller, monkeypatch):
    """Nothing is reset on review: the mark fades on the clock, so after a pause
    it no longer holds the old subject's level."""
    _, ctl = controller
    readings = _spy_live(monkeypatch)
    sharp = _sharp_frame()
    camera = FakeCamera([sharp, _blurred(sharp)])
    loop, touch, display, clock = _loop(camera, ctl, review_timeout=30.0)
    loop.step()                                  # sharp: sets the peak
    ctl.submit("ext")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    clock.t += 60.0                              # a minute on the review screen
    loop.step()                                  # review times out
    assert loop.state == "LIVE"
    loop.step()                                  # the blurred frame, a minute later
    assert readings[-1].focus <= 0.2
    # An absolute tolerance: the blurred frame now scores exactly 0.0, and a mark
    # that halves every few seconds only approaches zero.
    assert readings[-1].focus_peak == pytest.approx(readings[-1].focus, abs=1e-3)


# -- idle dimming ---------------------------------------------------------------


def _idle_loop(camera, ctl, **kw):
    kw.setdefault("dim_after", 60.0)
    kw.setdefault("off_after", 300.0)
    kw.setdefault("full_backlight", 80)
    return _loop(camera, ctl, **kw)


def _tap(loop, touch, clock, xy):
    touch.tap(*xy)
    loop.step()   # down
    clock.t += 0.1
    loop.step()   # up -> tap


def test_the_screen_dims_after_a_minute_and_turns_off_after_five(controller):
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    loop.step()
    assert display.levels[-1] == 80
    clock.t = 60.0
    loop.step()
    assert display.levels[-1] == 40
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0


def test_the_backlight_is_only_written_when_its_level_changes(controller):
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    for _ in range(5):
        loop.step()
        clock.t += 0.1
    assert display.levels == [80]


def test_no_camera_reads_or_frames_while_the_screen_is_off(controller):
    """Off is for the battery: the preview is neither read nor drawn."""
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    reads = _count_reads(camera)
    clock.t = 300.0
    loop.step()
    before_reads, before_frames = reads["n"], len(display.images)
    for _ in range(10):
        clock.t += 0.1
        loop.step()
    assert reads["n"] == before_reads and len(display.images) == before_frames


@pytest.mark.parametrize("key", [SHUTTER_CENTRE, PLUS_CENTRE, MINUS_CENTRE])
def test_a_tap_on_a_dark_screen_only_wakes_it_and_lights_nothing(controller, monkeypatch, key):
    """The user cannot see what they are touching: the shutter must not fire, no
    value may change, and the key under the finger is not lit when the screen
    comes back."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _idle_loop(camera, ctl)
    submitted = []
    monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0
    drawn = len(states)
    _tap(loop, touch, clock, key)
    loop.step()
    assert submitted == []
    assert camera.shutter_us is None
    assert display.levels[-1] == 80
    assert loop.state == "LIVE"
    woken = states[drawn:]
    assert woken and all(state.pressed is None for state in woken)


def test_a_tap_on_a_dimmed_screen_restores_it_and_acts(controller, monkeypatch):
    """Dimmed, the buttons are still visible, so the tap does what it hits."""
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    submitted = []
    monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
    clock.t = 60.0
    loop.step()
    assert display.levels[-1] == 40
    _tap(loop, touch, clock, SHUTTER_CENTRE)
    assert len(submitted) == 1
    assert display.levels[-1] == 80


def test_a_stick_capture_wakes_the_screen_into_review(controller):
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0
    ctl.submit("from-the-stick")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    assert display.levels[-1] == 80


def test_a_capture_in_progress_keeps_the_screen_awake(tmp_path):
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        loop, touch, display, clock = _idle_loop(camera, ctl)
        loop.step()
        ctl.submit("slow")
        _until(lambda: session.started.is_set())
        clock.t = 500.0
        loop.step()
        assert display.levels[-1] == 80
    finally:
        session.release.set()
        ctl.close()


def test_a_display_without_a_backlight_is_never_dimmed(controller):
    camera, ctl = controller

    class NoBacklight(FakeDisplay):
        backlight = None

    display = NoBacklight()
    loop, touch, _, clock = _idle_loop(camera, ctl, display=display)
    clock.t = 400.0
    loop.step()   # must not raise; the frame is still drawn
    assert display.images


# -- double exposure ------------------------------------------------------------


def _badge_centre():
    x0, y0, x1, y1 = DOUBLE_BOX
    return (x0 + x1) // 2, (y0 + y1) // 2


def _enable_double(loop, touch, clock, ctl):
    _tap(loop, touch, clock, _badge_centre())
    _until(lambda: ctl.snapshot().double_exposure)


def test_badge_tap_toggles_double_exposure(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    _tap(loop, touch, clock, _badge_centre())
    _until(lambda: not ctl.snapshot().double_exposure)


def test_badge_taps_during_processing_are_ignored(tmp_path):
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("busy")
        _until(lambda: session.started.is_set())
        _tap(loop, touch, clock, _badge_centre())
        session.release.set()
        _until(lambda: ctl.snapshot().finished_count == 1)
        # A toggle is queued behind the running job and applied after it finishes,
        # so drain the FIFO worker queue with a follow-up job before asserting.
        ctl.submit("drain")
        _until(lambda: ctl.status("drain").state == "complete")
        assert ctl.snapshot().double_exposure is False
    finally:
        session.release.set()
        ctl.close()


class _PendingController:
    """A controller whose snapshot reports photographs still being graded."""

    def __init__(self, ctl, pending):
        self._ctl, self.pending = ctl, pending

    def __getattr__(self, name):
        return getattr(self._ctl, name)

    def snapshot(self):
        return dataclasses.replace(self._ctl.snapshot(), pending_grades=self.pending)


def test_live_view_passes_the_developing_count_from_the_snapshot(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    pending = _PendingController(ctl, 2)
    loop, touch, display, clock = _loop(camera, pending)
    loop.step()
    assert states[-1].developing == 2
    # Read again at every frame: the count falls while no capture is running.
    pending.pending = 0
    loop.step()
    assert states[-1].developing == 0


def test_live_view_draws_no_count_for_a_controller_that_reports_none(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)

    class Old:
        def __getattr__(self, name):
            return getattr(ctl, name)

        def snapshot(self):
            snap = ctl.snapshot()
            return SimpleNamespace(**{k: v for k, v in vars(snap).items()
                                      if k != "pending_grades"})

    loop, touch, display, clock = _loop(camera, Old())
    loop.step()
    assert states[-1].developing == 0


class _TuningController:
    """A controller whose snapshot says the camera's tuning differs from the look's."""

    def __init__(self, ctl, mismatch):
        self._ctl, self.mismatch = ctl, mismatch

    def __getattr__(self, name):
        return getattr(self._ctl, name)

    def snapshot(self):
        return dataclasses.replace(self._ctl.snapshot(), tuning_mismatch=self.mismatch)


def test_live_view_passes_the_tuning_mismatch_from_the_snapshot(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    tuning = _TuningController(ctl, True)
    loop, touch, display, clock = _loop(camera, tuning)
    loop.step()
    assert states[-1].tuning_mismatch is True
    tuning.mismatch = False
    loop.step()
    assert states[-1].tuning_mismatch is False


def test_live_view_draws_no_tuning_tag_for_a_controller_that_reports_none(
        controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)

    class Old:
        def __getattr__(self, name):
            return getattr(ctl, name)

        def snapshot(self):
            snap = ctl.snapshot()
            return SimpleNamespace(**{k: v for k, v in vars(snap).items()
                                      if k != "tuning_mismatch"})

    loop, touch, display, clock = _loop(camera, Old())
    loop.step()
    assert states[-1].tuning_mismatch is False


def test_live_view_draws_the_badge_from_the_snapshot(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert states[-1].double == (False, 0)
    _enable_double(loop, touch, clock, ctl)
    loop.step()
    assert states[-1].double == (True, 0)


def test_exposure_one_shows_a_notice_not_a_review(controller, monkeypatch):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "NOTICE"
    assert calls["review"] == []
    assert calls["message"][-1][0] == "Exposure 1/2"
    clock.t += NOTICE_SECONDS - 0.1
    loop.step()
    assert loop.state == "NOTICE"
    clock.t += 0.2
    loop.step()
    assert loop.state == "LIVE"


def test_a_tap_ends_the_notice_early(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    _tap(loop, touch, clock, (160, 120))
    assert loop.state == "LIVE"
    assert ctl.snapshot().finished_count == 1  # the tap did not fire the shutter


def test_exposure_two_reviews_the_composite_with_a_2x_caption(controller, monkeypatch):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    ctl.submit("exp-2")
    _until(lambda: ctl.snapshot().finished_count == 2)
    loop.step()
    assert loop.state == "REVIEW"
    assert calls["review"][-1] == "2x  1/250  ISO 100"
    job = ctl.snapshot().last_finished_job
    assert job.result.pifilm.name.endswith("_double_graded.jpg")


def test_processing_labels_follow_the_pair(tmp_path, monkeypatch):
    camera, ctl, session = _gated_controller(tmp_path)
    labels = []
    real = viewfinder.render_processing

    def spy(label="Processing photo...", **kwargs):
        labels.append(label)
        return real(label, **kwargs)

    monkeypatch.setattr(viewfinder, "render_processing", spy)
    try:
        loop, touch, display, clock = _loop(camera, ctl)
        _enable_double(loop, touch, clock, ctl)
        for request_id, expected in (("a", "Exposure 1 of 2..."),
                                     ("b", "Developing double exposure...")):
            session.started.clear()
            session.release.clear()
            ctl.submit(request_id)
            _until(lambda: session.started.is_set())
            loop.step()
            assert labels[-1] == expected
            session.release.set()
            _until(lambda rid=request_id: ctl.status(rid).state == "complete")
            loop.step()   # NOTICE or REVIEW
            _tap(loop, touch, clock, (160, 120))  # back to LIVE
    finally:
        session.release.set()
        ctl.close()


# -- manual controls: select a cell, then + / - -------------------------------------

SETTERS = ("set_shutter", "set_iso", "set_colour_temperature")


def _cell(index):
    """A point in the middle of a strip cell's hit column."""
    return CELL_HIT_X0 + CELL_PITCH * index + CELL_PITCH // 2, BAR_TOP + 16


def _values(camera):
    return camera.shutter_us, camera.iso, camera.colour_temperature_k


@pytest.fixture
def metered(tmp_path):
    camera, ctl = _make_controller(tmp_path, METERED)
    yield camera, ctl
    ctl.close()


def test_the_shutter_cell_is_selected_at_start(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert loop.selected == SHUTTER_CELL
    assert states[-1].selected == SHUTTER_CELL


def test_plus_and_minus_change_only_the_selected_control(metered):
    camera, ctl = metered
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()  # one live frame, so the loop has metered values
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert _values(camera) == (4000, None, None)
    _tap(loop, touch, clock, _cell(ISO_CELL))
    assert loop.selected == ISO_CELL
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert _values(camera) == (4000, 800, None)
    _tap(loop, touch, clock, _cell(WB_CELL))
    assert loop.selected == WB_CELL
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert _values(camera) == (4000, 800, 4000)
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert _values(camera) == (4000, 800, 3200)
    _tap(loop, touch, clock, _cell(SHUTTER_CELL))
    assert loop.selected == SHUTTER_CELL
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert _values(camera) == (5000, 800, 3200)


@pytest.mark.parametrize("key", [PLUS_CENTRE, MINUS_CENTRE])
@pytest.mark.parametrize(("cell", "expected"), [
    (SHUTTER_CELL, (4000, None, None)),
    (ISO_CELL, (None, 800, None)),
    (WB_CELL, (None, None, 4000)),
])
def test_the_first_step_from_auto_lands_on_the_metered_step(metered, cell, expected, key):
    """Either direction: switching a control to manual must not jump the picture."""
    camera, ctl = metered
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _cell(cell))
    _tap(loop, touch, clock, key)
    assert _values(camera) == expected


def test_plus_is_a_faster_shutter_and_minus_a_slower_one(controller):
    camera, ctl = controller  # FakeCamera with ExposureTime 4000 in its metadata
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.shutter_us == 4000
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.shutter_us == 3125
    _tap(loop, touch, clock, MINUS_CENTRE)
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert camera.shutter_us == 5000


def test_plus_is_a_higher_iso_and_a_warmer_white_balance(metered):
    camera, ctl = metered
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _cell(ISO_CELL))
    _tap(loop, touch, clock, PLUS_CENTRE)
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.iso == 1600
    _tap(loop, touch, clock, _cell(WB_CELL))
    _tap(loop, touch, clock, PLUS_CENTRE)
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.colour_temperature_k == 5000


@pytest.mark.parametrize(("cell", "expected"), [
    (SHUTTER_CELL, (8000, None, None)),
    (ISO_CELL, (None, 400, None)),
    (WB_CELL, (None, None, 5600)),
])
def test_a_step_before_any_metered_frame_starts_at_the_default(controller, cell, expected):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    camera.read = lambda **k: (_ for _ in ()).throw(CameraError("no frame"))
    _tap(loop, touch, clock, _cell(cell))
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert _values(camera) == expected


@pytest.mark.parametrize("metadata", [
    None,
    {},
    {"ExposureTime": 0, "AnalogueGain": 0, "ColourTemperature": 0},
    {"ExposureTime": float("nan"), "AnalogueGain": float("nan"),
     "ColourTemperature": float("nan")},
    {"ExposureTime": "n/a", "AnalogueGain": "n/a", "ColourTemperature": "n/a"},
    {"ExposureTime": True, "AnalogueGain": True, "ColourTemperature": True},
    {"ExposureTime": float("inf"), "AnalogueGain": -1.0, "ColourTemperature": -5600},
])
def test_meaningless_metered_values_fall_back_to_the_defaults(tmp_path, monkeypatch, metadata):
    """The first frames can arrive with no metadata, or with zeros in it."""
    camera, ctl = _make_controller(tmp_path, metadata)
    try:
        states = _spy_states(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        loop.step()
        assert states[-1].cells[WB_CELL].value == "-"
        _tap(loop, touch, clock, PLUS_CENTRE)
        assert camera.shutter_us == 8000
        _tap(loop, touch, clock, _cell(WB_CELL))
        _tap(loop, touch, clock, PLUS_CENTRE)
        assert camera.colour_temperature_k == 5600
        _tap(loop, touch, clock, _cell(ISO_CELL))
        _tap(loop, touch, clock, PLUS_CENTRE)
        assert camera.iso == 400
        assert loop.state == "LIVE"
    finally:
        ctl.close()


@pytest.mark.parametrize(("cell", "setter", "bottom"), [
    (SHUTTER_CELL, "set_shutter", 1_000_000),   # the slowest step, 1 s
    (ISO_CELL, "set_iso", 100),
    (WB_CELL, "set_colour_temperature", 3200),
])
def test_minus_below_the_bottom_step_returns_the_control_to_auto(controller, cell, setter, bottom):
    camera, ctl = controller
    getattr(camera, setter)(bottom)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _cell(cell))
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert _values(camera) == (None, None, None)


@pytest.mark.parametrize(("cell", "setter", "top"), [
    (SHUTTER_CELL, "set_shutter", 500),         # the fastest step, 1/2000
    (ISO_CELL, "set_iso", 1600),
    (WB_CELL, "set_colour_temperature", 7500),
])
def test_plus_at_the_top_step_stays_there(controller, cell, setter, top):
    camera, ctl = controller
    getattr(camera, setter)(top)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _cell(cell))
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert _values(camera)[cell] == top


def _make_unavailable(monkeypatch, camera, how):
    if how == "wb_not_adjustable":
        camera.wb_adjustable = False
    else:
        monkeypatch.setattr(type(camera), how, None, raising=False)


@pytest.mark.parametrize(("how", "cell"), [
    ("set_shutter", SHUTTER_CELL),
    ("set_iso", ISO_CELL),
    ("set_colour_temperature", WB_CELL),
    ("wb_not_adjustable", WB_CELL),
])
def test_a_control_the_camera_cannot_change_is_grey_and_cannot_be_selected(
    metered, monkeypatch, how, cell,
):
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    _make_unavailable(monkeypatch, camera, how)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    # The first cell the camera can change is the one selected at start.
    start = ISO_CELL if cell == SHUTTER_CELL else SHUTTER_CELL
    assert loop.selected == start
    _tap(loop, touch, clock, _cell(cell))
    assert loop.selected == start
    _tap(loop, touch, clock, PLUS_CENTRE)
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert _values(camera)[cell] is None
    assert loop.state == "LIVE"
    enabled = [c.enabled for c in states[-1].cells]
    assert enabled == [index != cell for index in range(3)]
    assert states[-1].selected == start


def test_plus_and_minus_do_nothing_when_no_control_is_selectable(metered, monkeypatch):
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    for setter in SETTERS:
        _make_unavailable(monkeypatch, camera, setter)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert loop.selected is None
    for point in (_cell(SHUTTER_CELL), _cell(ISO_CELL), _cell(WB_CELL), PLUS_CENTRE,
                  MINUS_CENTRE):
        _tap(loop, touch, clock, point)
    assert loop.selected is None
    assert _values(camera) == (None, None, None)
    assert loop.state == "LIVE"
    assert states[-1].selected is None
    assert not any(c.enabled for c in states[-1].cells)


@pytest.mark.parametrize(("cell", "setter"), list(enumerate(SETTERS)))
def test_a_rejected_control_is_logged_not_fatal(metered, cell, setter):
    camera, ctl = metered
    lines = []

    def reject(value):
        raise CameraError("control rejected")

    setattr(camera, setter, reject)
    loop, touch, display, clock = _loop(camera, ctl, log=lines.append)
    loop.step()
    _tap(loop, touch, clock, _cell(cell))
    assert loop.selected == cell
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert "camera: control rejected" in lines
    assert loop.state == "LIVE"
    assert _values(camera) == (None, None, None)
    loop.step()
    assert len(display.images) >= 2


@pytest.mark.parametrize("cell", [SHUTTER_CELL, ISO_CELL])
def test_locked_auto_exposure_greys_the_shutter_and_iso_cells(metered, monkeypatch, cell):
    """``--ae-lock``: the camera reports ``exposure_adjustable = False``."""
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    camera.exposure_adjustable = False
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert loop.selected == WB_CELL
    _tap(loop, touch, clock, _cell(cell))
    assert loop.selected == WB_CELL
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert _values(camera) == (None, None, 4000)
    assert [c.enabled for c in states[-1].cells] == [False, False, True]


def test_a_cell_the_camera_cannot_change_shows_its_value_without_auto(metered, monkeypatch):
    """Under ``--awb-lock`` the white balance is fixed, not automatic, and under
    ``--ae-lock`` so is the exposure: the tag would be a lie. The value still reads."""
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    camera.exposure_adjustable = False
    camera.wb_adjustable = False
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert loop.selected is None
    assert states[-1].cells == (
        Cell("SHUTTER", "1/250", auto=False, enabled=False),
        Cell("ISO", "760", auto=False, enabled=False),
        Cell("WB", "4200K", auto=False, enabled=False),
    )


def test_iso_steps_above_the_sensors_gain_limit_are_not_offered(tmp_path):
    """Metered ISO 1250 is nearest 1600, which a sensor with gain 8 cannot reach:
    the first step from auto must land on a value ``set_iso`` accepts."""
    camera, ctl = _make_controller(
        tmp_path, {"ExposureTime": 4000, "AnalogueGain": 12.5, "Lux": 500.0})
    try:
        camera.max_gain = 8.0
        loop, touch, display, clock = _loop(camera, ctl)
        loop.step()
        _tap(loop, touch, clock, _cell(ISO_CELL))
        _tap(loop, touch, clock, PLUS_CENTRE)
        assert camera.iso == 800
        _tap(loop, touch, clock, PLUS_CENTRE)
        assert camera.iso == 800
        _tap(loop, touch, clock, MINUS_CENTRE)
        assert camera.iso == 400
    finally:
        ctl.close()


@pytest.mark.parametrize("bad", [None, 0, float("nan"), "n/a"])
def test_an_unknown_gain_limit_offers_every_iso_step(metered, bad):
    camera, ctl = metered
    camera.max_gain = bad
    camera.set_iso(800)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _cell(ISO_CELL))
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.iso == 1600


def test_automatic_cells_show_the_metered_values(metered, monkeypatch):
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert states[-1].cells == (
        Cell("SHUTTER", "1/250", auto=True, enabled=True),
        Cell("ISO", "760", auto=True, enabled=True),
        Cell("WB", "4200K", auto=True, enabled=True),
    )


def test_fixed_cells_show_the_fixed_values_without_auto(metered, monkeypatch):
    camera, ctl = metered
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    for cell in (SHUTTER_CELL, ISO_CELL, WB_CELL):
        _tap(loop, touch, clock, _cell(cell))
        _tap(loop, touch, clock, PLUS_CENTRE)   # onto the metered step
        _tap(loop, touch, clock, PLUS_CENTRE)   # and one step on
    loop.step()
    assert states[-1].cells == (
        Cell("SHUTTER", "1/320", auto=False, enabled=True),
        Cell("ISO", "1600", auto=False, enabled=True),
        Cell("WB", "5000K", auto=False, enabled=True),
    )
    assert states[-1].selected == WB_CELL


def test_the_iso_cell_reads_max_when_auto_gain_is_at_its_limit(metered, monkeypatch):
    camera, ctl = metered
    camera.max_gain = 7.6   # the metered analogue gain: nothing left to give
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert states[-1].cells[ISO_CELL] == Cell("ISO", "MAX", auto=True, enabled=True)
    camera.set_iso(800)   # a fixed ISO shows its own value, limit or not
    loop.step()
    assert states[-1].cells[ISO_CELL] == Cell("ISO", "800", auto=False, enabled=True)


def test_cells_show_a_dash_when_the_frame_has_no_metadata(tmp_path, monkeypatch):
    camera, ctl = _make_controller(tmp_path, None)
    try:
        states = _spy_states(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        loop.step()
        assert [c.value for c in states[-1].cells] == ["-", "-", "-"]
        assert all(c.auto for c in states[-1].cells)
    finally:
        ctl.close()


def test_the_meter_is_told_the_sensors_gain_limit(controller, monkeypatch):
    camera, ctl = controller
    seen = {}
    real_compute = viewfinder.compute_reading

    def compute(*a, **k):
        seen["max_gain"] = k.get("max_gain")
        return real_compute(*a, **k)

    monkeypatch.setattr(viewfinder, "compute_reading", compute)
    camera.max_gain = 16.0
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert seen == {"max_gain": 16.0}


# -- press feedback -----------------------------------------------------------------


def test_a_finger_held_on_plus_lights_it_every_frame_and_steps_once_on_release(
    controller, monkeypatch,
):
    """Actions fire on release, never while the finger is down: no key repeat."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    touch.script = [[TouchPoint(*PLUS_CENTRE, 10)]] * 5
    for _ in range(5):
        loop.step()
        assert states[-1].pressed is Action.PLUS
        assert camera.shutter_us is None
        clock.t += 0.1
    loop.step()   # the finger is up
    assert camera.shutter_us == 4000
    for _ in range(5):
        clock.t += 0.1
        loop.step()
    assert camera.shutter_us == 4000
    assert states[-1].pressed is None


def test_a_one_second_press_on_plus_still_steps_exactly_once(controller, monkeypatch):
    """A lit key invites a longer press than a blind tap; it must still count."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    touch.script = [[TouchPoint(*PLUS_CENTRE, 10)]] * 10
    for _ in range(10):
        loop.step()
        assert states[-1].pressed is Action.PLUS
        assert camera.shutter_us is None
        clock.t += 0.1
    assert clock.t == pytest.approx(1.0)
    loop.step()   # the finger is up, 1.0 s after it went down
    assert camera.shutter_us == 4000
    for _ in range(5):
        clock.t += 0.1
        loop.step()
    assert camera.shutter_us == 4000


def test_a_finger_resting_on_a_key_lights_it_but_never_steps(controller, monkeypatch):
    """Longer than ``TAP_MAX_HOLD`` is not a tap: a resting thumb changes nothing."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    polls = int(TAP_MAX_HOLD / 0.1) + 2
    touch.script = [[TouchPoint(*MINUS_CENTRE, 10)]] * polls
    for _ in range(polls):
        loop.step()
        assert states[-1].pressed is Action.MINUS
        clock.t += 0.1
    assert clock.t > TAP_MAX_HOLD
    loop.step()
    clock.t += PRESS_FLASH_SECONDS
    loop.step()
    assert camera.shutter_us is None
    assert states[-1].pressed is None


@pytest.mark.parametrize(("key", "action"), [
    (PLUS_CENTRE, Action.PLUS), (MINUS_CENTRE, Action.MINUS), (SHUTTER_CENTRE, Action.SHUTTER),
])
def test_a_tap_keeps_its_key_lit_for_the_flash_after_the_finger_is_up(
    controller, monkeypatch, key, action,
):
    """A tap can be over before a frame is drawn; the key must still be seen to react."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    monkeypatch.setattr(ctl, "submit", lambda rid: None)
    loop, touch, display, clock = _loop(camera, ctl)
    _tap(loop, touch, clock, key)             # the second step saw no finger
    assert states[-1].pressed is action
    clock.t += PRESS_FLASH_SECONDS / 2
    loop.step()
    assert states[-1].pressed is action
    clock.t += PRESS_FLASH_SECONDS / 2 + 0.01
    loop.step()
    assert states[-1].pressed is None


def test_only_the_keys_light_under_a_finger(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    for point in (_cell(ISO_CELL), _badge_centre(), (160, 100)):
        touch.script = [[TouchPoint(*point, 10)], []]
        loop.step()
        assert states[-1].pressed is None
        clock.t += 0.1
        loop.step()
        assert states[-1].pressed is None
        clock.t += 1.0


def test_a_touch_error_lights_nothing(controller, monkeypatch):
    """A failed read is "no finger": a key must not stay lit behind a dead panel."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    touch = GlitchyTouch(fail_on=(1,))
    touch.script = [[TouchPoint(*PLUS_CENTRE, 10)]]
    loop, _, display, clock = _loop(camera, ctl, touch=touch)
    loop.step()
    assert states[-1].pressed is Action.PLUS
    clock.t += TAP_MAX_HOLD + 0.5   # too long to be a tap, so no flash either
    loop.step()   # this read fails
    assert states[-1].pressed is None


class BrokenTouch(ScriptedTouch):
    """A touch device whose driver raises something that is not a DisplayError."""

    def __init__(self):
        super().__init__()
        self.broken = False

    def read(self):
        if self.broken:
            raise OSError("input device went away")
        return super().read()


def test_an_unexpected_touch_error_does_not_leave_a_key_lit(controller):
    """The touch thread logs such an error and carries on; the finger it last saw
    must not stay recorded, or the key under it is lit for as long as the fault lasts."""
    camera, ctl = controller
    touch = BrokenTouch()
    touch.script = [[TouchPoint(*PLUS_CENTRE, 10)]]
    loop, _, display, clock = _loop(camera, ctl, touch=touch)
    assert loop._poll_tap() is None
    assert loop._pressed(clock.monotonic()) is Action.PLUS
    touch.broken = True
    with pytest.raises(OSError):
        loop._poll_tap()
    assert loop._pressed(clock.monotonic()) is None


def test_the_touch_thread_clears_a_lit_key_when_the_device_fails(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    lines = []
    touch = BrokenTouch()
    touch.script = [[TouchPoint(*PLUS_CENTRE, 10)]] * 100_000
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=time, frame_period=0.02,
                          log=lines.append)
    stop = threading.Event()
    runner = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    runner.start()
    try:
        _until(lambda: any(state.pressed is Action.PLUS for state in list(states)))
        touch.broken = True
        _until(lambda: "touch: input device went away" in lines)
        drawn = len(states)
        _until(lambda: len(states) >= drawn + 3)
        assert all(state.pressed is None for state in list(states)[drawn + 1:])
    finally:
        stop.set()
        runner.join(5.0)
    assert not runner.is_alive()
    assert camera.shutter_us is None


# -- controls while a job is processing ----------------------------------------------


@pytest.mark.parametrize(("point", "kind"), [
    (None, "select"), (PLUS_CENTRE, "plus"), (MINUS_CENTRE, "minus"),
])
def test_cell_and_key_taps_during_processing_are_ignored(tmp_path, monkeypatch, point, kind):
    """The colour bars hide the cells and the keys, as they hide the 2x toggle:
    no value change, no selection change and no flash."""
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        states = _spy_states(monkeypatch)
        camera.set_shutter(4000)
        loop, touch, display, clock = _loop(camera, ctl)
        loop.step()
        ctl.submit("busy")
        _until(lambda: session.started.is_set())
        _tap(loop, touch, clock, point or _cell(ISO_CELL))
        assert loop.selected == SHUTTER_CELL
        assert _values(camera) == (4000, None, None)
        session.release.set()
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
        _tap(loop, touch, clock, (160, 100))   # back to LIVE, clock barely moved
        assert loop.state == "LIVE"
        loop.step()
        assert states[-1].pressed is None
        assert states[-1].selected == SHUTTER_CELL
        assert _values(camera) == (4000, None, None)
        # and the same tap acts again once the live view is back
        _tap(loop, touch, clock, point or _cell(ISO_CELL))
        assert (loop.selected, camera.shutter_us) == {
            "select": (ISO_CELL, 4000), "plus": (SHUTTER_CELL, 3125),
            "minus": (SHUTTER_CELL, 5000),
        }[kind]
    finally:
        session.release.set()
        ctl.close()


def test_a_shutter_tap_during_processing_still_reaches_the_controller(tmp_path, monkeypatch):
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("busy")
        _until(lambda: session.started.is_set())
        submitted = []
        monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
        _tap(loop, touch, clock, SHUTTER_CENTRE)
        assert len(submitted) == 1
    finally:
        session.release.set()
        ctl.close()


def test_the_shutter_key_is_not_lit_when_the_live_view_returns_after_the_shot(
    tmp_path, monkeypatch,
):
    """The tap's flash belongs to the frame it was tapped on. The fake clock does
    not move here, so only clearing the flash (not its timeout) can pass this."""
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        states = _spy_states(monkeypatch)
        loop, touch, display, clock = _loop(camera, ctl)
        touch.tap(*SHUTTER_CENTRE)
        loop.step()   # down
        loop.step()   # up -> tap -> submit; this frame shows the key lit
        assert states[-1].pressed is Action.SHUTTER
        _until(lambda: session.started.is_set())
        loop.step()   # colour bars
        session.release.set()
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
        touch.tap(160, 100)
        loop.step()
        loop.step()
        assert loop.state == "LIVE"
        drawn = len(states)
        loop.step()
        assert len(states) == drawn + 1
        assert states[-1].pressed is None
    finally:
        session.release.set()
        ctl.close()


# -- touch on its own thread (long exposures) ---------------------------------------


class TimedTouch:
    """A finger that is down for ``hold`` seconds of real time after ``press``."""

    def __init__(self):
        self._lock = threading.Lock()
        self._press = None

    def press(self, xy, hold=0.15):
        with self._lock:
            self._press = (xy, time.monotonic(), hold)

    def read(self):
        with self._lock:
            press = self._press
        if press is None:
            return []
        (x, y), t0, hold = press
        return [TouchPoint(x, y, 10)] if time.monotonic() - t0 < hold else []

    def close(self):
        pass


def _touch_thread_alive():
    return any(t.name == "pifilm-touch" and t.is_alive() for t in threading.enumerate())


def test_a_quick_tap_registers_while_a_long_exposure_blocks_the_preview(controller):
    """At 1 s the preview read blocks a whole second, longer than a tap lasts.

    Polled inline, the finger is down and up again between two polls and the
    tap is lost; the touch thread sees it and hands it to the next step.
    """
    camera, ctl = controller
    camera.set_shutter(1_000_000)  # one step slower is back to A
    real_read = camera.read
    reading = threading.Event()

    def slow_read(*, full=True):
        reading.set()
        time.sleep(1.0)
        return real_read(full=full)

    camera.read = slow_read
    touch = TimedTouch()
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=time)
    stop = threading.Event()
    runner = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    runner.start()
    try:
        assert reading.wait(2.0), "the loop never read a preview frame"
        touch.press(MINUS_CENTRE, hold=0.15)
        _until(lambda: camera.shutter_us is None, timeout=3.0)
    finally:
        stop.set()
        runner.join(5.0)
    assert not runner.is_alive()


def test_a_finger_held_on_plus_is_drawn_lit_while_the_touch_thread_polls(
    controller, monkeypatch,
):
    """Under run() the finger's position comes from the touch thread, not the loop."""
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    touch = TimedTouch()
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=time, frame_period=0.02)
    stop = threading.Event()
    runner = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    runner.start()
    try:
        _until(lambda: len(states) >= 2)
        assert all(state.pressed is None for state in list(states))
        touch.press(PLUS_CENTRE, hold=0.5)
        _until(lambda: any(state.pressed is Action.PLUS for state in list(states)))
        assert camera.shutter_us is None          # lit while down, nothing fired yet
        _until(lambda: camera.shutter_us == 4000)  # one step, on release
        _until(lambda: states[-1].pressed is None)
    finally:
        stop.set()
        runner.join(5.0)
    assert not runner.is_alive()
    assert camera.shutter_us == 4000


def test_the_touch_thread_stops_with_the_loop(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, clock=time, frame_period=0.02)
    stop = threading.Event()
    seen = {"alive": False}
    real_step = loop.step

    def watching_step():
        seen["alive"] = seen["alive"] or _touch_thread_alive()
        real_step()
        if len(display.images) >= 3:
            stop.set()

    loop.step = watching_step
    loop.run(stop)
    assert seen["alive"], "run() never started the touch thread"
    assert not _touch_thread_alive()


def test_the_touch_thread_stops_when_the_loop_raises(controller):
    """A display breaker trip ends run() by exception; the thread must not
    outlive it, because the caller closes the touch device next."""
    camera, ctl = controller
    loop, touch, display, clock = _loop(
        camera, ctl, display=FakeDisplay(fail_after=0), clock=time, frame_period=0.01,
    )
    stop = threading.Event()
    with pytest.raises(DisplayError):
        loop.run(stop)
    assert not stop.is_set()
    assert not _touch_thread_alive()


class QueuedAtStartTouch:
    """A device that was opened before the loop ran: the first read hands back a
    touch the kernel queued meanwhile, every later read finds no finger."""

    def __init__(self, xy):
        self._xy = xy
        self.reads = 0

    def read(self):
        n, self.reads = self.reads, self.reads + 1
        return [TouchPoint(*self._xy, 10)] if n == 0 else []

    def close(self):
        pass


def test_a_touch_queued_before_the_loop_started_takes_no_photo(controller):
    camera, ctl = controller
    submitted = []
    ctl.submit = submitted.append
    touch = QueuedAtStartTouch(SHUTTER_CENTRE)
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=time, frame_period=0.01)
    steps = {"n": 0}
    real_step = loop.step

    def counting_step():
        real_step()
        steps["n"] += 1

    loop.step = counting_step
    stop = threading.Event()
    runner = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    runner.start()
    try:
        # Long enough for the touch thread to see the point and its release, and
        # then for the loop to take whatever tap that made off the queue.
        _until(lambda: touch.reads >= 4)
        seen = steps["n"]
        _until(lambda: steps["n"] >= seen + 3)
    finally:
        stop.set()
        runner.join(5.0)
    assert not runner.is_alive()
    assert submitted == []


# -- the 640x480 DSI panel --------------------------------------------------------


class BigDisplay(FakeDisplay):
    width, height = 640, 480
    dimmable = False


def test_a_640_wide_display_is_drawn_at_640x480(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    loop.step()
    assert display.images[-1].size == (640, 480)


def test_taps_are_hit_tested_at_the_display_scale(controller, monkeypatch):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    submitted = []
    monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
    cx, cy = SHUTTER_CENTRE
    _tap(loop, touch, clock, (2 * cx, 2 * cy))
    assert len(submitted) == 1
    _tap(loop, touch, clock, (cx, cy))        # the 320x240 position is not the button here
    assert len(submitted) == 1


def test_cells_and_keys_are_hit_tested_at_the_display_scale(controller, monkeypatch):
    camera, ctl = controller
    states = _spy_states(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    x, y = _cell(ISO_CELL)
    _tap(loop, touch, clock, (2 * x, 2 * y))
    assert loop.selected == ISO_CELL
    touch.script = [[TouchPoint(2 * PLUS_CENTRE[0], 2 * PLUS_CENTRE[1], 10)], []]
    loop.step()
    assert states[-1].pressed is Action.PLUS
    clock.t += 0.1
    loop.step()
    assert camera.iso == 100 and camera.shutter_us is None


def test_review_and_processing_screens_use_the_display_scale(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    assert display.images[-1].size == (640, 480)


def test_a_non_dimmable_display_skips_dimming_but_still_turns_off(controller):
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl, display=BigDisplay())
    loop.step()
    clock.t = 60.0
    loop.step()
    assert display.levels == [80]
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0


def test_a_display_without_a_width_is_drawn_at_320x240(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert display.images[-1].size == (320, 240)


def test_a_camera_that_keeps_stalling_ends_the_loop(controller):
    """One late frame is survivable; a camera that has stopped delivering them
    is not, and only a restart of the process brings it back."""
    camera, ctl = controller
    logged = []
    loop, touch, display, clock = _loop(camera, ctl, log=logged.append)
    camera.read = lambda *, full=True: (_ for _ in ()).throw(CameraStalled("no frame"))
    for _ in range(CAMERA_STALL_LIMIT - 1):
        loop.step()
    assert loop.state == "LIVE"
    with pytest.raises(CameraStalled):
        loop.step()
    assert sum("camera: no frame" in line for line in logged) == CAMERA_STALL_LIMIT


def test_a_frame_between_stalls_resets_the_count(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    original = camera.read

    def stall(*, full=True):
        raise CameraStalled("no frame")

    for _ in range(3):
        camera.read = stall
        for _ in range(CAMERA_STALL_LIMIT - 1):
            loop.step()
        camera.read = original
        loop.step()
    assert len(display.images) == 3


# -- frames on their own thread (long exposures) ------------------------------------


class HeldClock(FakeClock):
    """Loop time moves only when the test moves it; a sleep just yields the CPU.

    Under ``run()`` the loop then turns over hundreds of times a second of real
    time while its own clock stands still, so what is drawn and counted depends
    on what the test releases, not on how fast the machine is.
    """

    def sleep(self, s):
        time.sleep(0.002)


class ReadGate:
    """Makes every preview read wait for a permit, as a long exposure does.

    A full read (the capture worker's still) passes straight through, uncounted.
    """

    def __init__(self, camera):
        self._real = camera.read
        self._permits = threading.Semaphore(0)
        self.reads = 0                      # preview reads begun
        self.returned = 0                   # ... and finished
        camera.read = self._read

    def _read(self, *, full=True):
        if full:
            return self._real(full=full)
        self.reads += 1
        if not self._permits.acquire(timeout=10.0):
            raise CameraError("the test never released this read")
        try:
            return self._real(full=False)
        finally:
            self.returned += 1

    def release(self, n=1):
        for _ in range(n):
            self._permits.release()


class ScriptedReads:
    """Preview reads that raise the scripted errors in order, then deliver frames."""

    def __init__(self, camera, script):
        self._real = camera.read
        self._script = list(script)
        self.reads = 0
        camera.read = self._read

    def _read(self, *, full=True):
        self.reads += 1
        outcome = self._script.pop(0) if self._script else None
        if outcome is not None:
            raise outcome
        return self._real(full=full)


class Running:
    """``loop.run`` on a thread, with whatever it raised and a count of its steps."""

    def __init__(self, loop):
        self.stop = threading.Event()
        self.raised = None
        self.steps = 0
        real_step = loop.step

        def counting_step():
            real_step()
            self.steps += 1

        loop.step = counting_step
        self._loop = loop
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self._loop.run(self.stop)
        except BaseException as exc:  # handed to the test, which asserts on it
            self.raised = exc

    def more_steps(self, n=5):
        """Wait until the loop has been round ``n`` more times."""
        target = self.steps + n
        _until(lambda: self.steps >= target or not self.thread.is_alive())
        assert self.steps >= target, f"the loop ended early: {self.raised!r}"

    def finish(self, timeout=5.0):
        self.stop.set()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "run() did not return"


def _frames_thread_alive():
    return any(t.name == "pifilm-frames" and t.is_alive() for t in threading.enumerate())


def _spy_frames(monkeypatch):
    """Record the picture and ``LiveState`` of every live redraw, still drawing it."""
    drawn = []
    real = viewfinder.render_live

    def spy(frame_rgb, reading, state, **kwargs):
        drawn.append((frame_rgb, state))
        return real(frame_rgb, reading, state, **kwargs)

    monkeypatch.setattr(viewfinder, "render_live", spy)
    return drawn


def _pictures(drawn):
    """How many different camera frames the redraws showed, in order."""
    count, last = 0, None
    for rgb, _ in list(drawn):
        if rgb is not last:
            count, last = count + 1, rgb
    return count


def test_a_step_without_run_still_reads_its_frame_inline(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    reads = _count_reads(camera)
    for _ in range(3):
        loop.step()
    assert reads["n"] == 3 and len(display.images) == 3
    assert not _frames_thread_alive()


def test_a_plus_tap_changes_the_value_and_lights_the_key_while_a_read_is_blocked(
    controller, monkeypatch,
):
    """At a 1 s shutter the camera read blocks for a second. The controls must
    not wait for it: the screen is redrawn, and a tap acted on, at the loop's
    own rate."""
    camera, ctl = controller
    gate = ReadGate(camera)
    drawn = _spy_frames(monkeypatch)
    touch = TimedTouch()
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=time, frame_period=0.02)
    running = Running(loop)
    try:
        gate.release()                            # the first frame, so there is a picture
        _until(lambda: len(drawn) >= 1 and gate.reads == 2)
        # From here to the end the second read never returns.
        before = len(drawn)
        _until(lambda: len(drawn) >= before + 3)  # still redrawn with the read blocked
        assert drawn[-1][1].cells[SHUTTER_CELL].auto is True
        touch.press(PLUS_CENTRE, hold=0.1)
        _until(lambda: camera.shutter_us == 4000)
        _until(lambda: any(
            state.pressed is Action.PLUS and state.cells[SHUTTER_CELL] == Cell(
                "SHUTTER", "1/250", auto=False)
            for _, state in list(drawn)[before:]
        ))
        assert gate.reads == 2 and _pictures(drawn) == 1
    finally:
        gate.release(10)
        running.finish()
    assert running.raised is None


def test_a_new_frame_is_drawn_when_it_arrives_and_the_rate_counts_camera_frames(
    tmp_path, monkeypatch,
):
    """The fps line is read on hardware as the camera's rate, so the redraws
    between two camera frames must not be counted in it."""
    art = tmp_path / "art"
    write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    camera = FakeCamera([synthetic_frame(48, 64) for _ in range(3)], metadata=DEFAULT_METADATA)
    ctl = CaptureController(CaptureSession(
        camera, Pipeline(Artifacts.load(art)), tmp_path / "shots",
        seed_rng=np.random.default_rng(0)))
    gate = ReadGate(camera)
    drawn = _spy_frames(monkeypatch)
    logs = []
    loop, _, _, clock = _loop(camera, ctl, clock=HeldClock(), log=logs.append)
    running = Running(loop)
    try:
        assert not drawn                          # nothing to show before the first frame
        for expected in (1, 2):
            gate.release()
            _until(lambda n=expected: _pictures(drawn) == n)
            shown = len(drawn)
            _until(lambda n=shown: len(drawn) >= n + 10)   # many redraws of one frame
            assert _pictures(drawn) == expected
        clock.t = 10.0
        gate.release()
        _until(lambda: _pictures(drawn) == 3)
        _until(lambda: any(line.startswith("viewfinder:") for line in logs))
    finally:
        gate.release(10)
        running.finish()
        ctl.close()
    assert running.raised is None
    assert [line for line in logs if line.startswith("viewfinder:")][0] == "viewfinder: 0.3 fps"


def test_the_frame_thread_does_not_read_while_the_screen_is_off(controller):
    camera, ctl = controller
    gate = ReadGate(camera)
    touch = TimedTouch()
    loop, _, display, clock = _idle_loop(camera, ctl, touch=touch, clock=HeldClock())
    running = Running(loop)
    try:
        gate.release()
        _until(lambda: gate.reads == 2)           # the second read is in flight
        clock.t = 300.0
        _until(lambda: display.levels[-1:] == [0])
        running.more_steps()
        gate.release(5)                           # that read returns; no other may start
        running.more_steps(20)
        time.sleep(0.05)
        assert gate.reads == 2
        touch.press((5, 5), hold=0.1)             # a tap wakes the screen
        _until(lambda: gate.reads >= 3)
    finally:
        gate.release(50)
        running.finish()
    assert running.raised is None


def test_the_frame_thread_does_not_read_during_processing_or_review(tmp_path):
    camera, ctl, session = _gated_controller(tmp_path)
    gate = ReadGate(camera)
    touch = TimedTouch()
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=HeldClock())
    running = Running(loop)
    try:
        gate.release()
        _until(lambda: gate.reads == 2)           # the second read is in flight
        ctl.submit("one")
        assert session.started.wait(2.0)
        running.more_steps()                      # the loop has seen the active job
        gate.release(5)                           # that read returns; no other may start
        running.more_steps(20)
        time.sleep(0.05)
        assert gate.reads == 2
        session.release.set()
        _until(lambda: loop.state == "REVIEW")
        running.more_steps(20)
        time.sleep(0.05)
        assert gate.reads == 2
        touch.press((5, 5), hold=0.1)             # a tap ends the review
        _until(lambda: loop.state == "LIVE" and gate.reads >= 3)
    finally:
        session.release.set()
        gate.release(50)
        running.finish()
        ctl.close()
    assert running.raised is None


def test_the_step_that_takes_the_photo_asks_for_no_preview_frame(tmp_path):
    """The still waits for the camera behind any preview read. A read asked for
    by the very step that pressed the shutter would be a second of shutter lag
    at a 1 s shutter, for a frame the colour bars are about to cover."""
    camera, ctl, session = _gated_controller(tmp_path)
    gate = ReadGate(camera)
    touch = TimedTouch()
    loop, _, _, _ = _loop(camera, ctl, touch=touch, clock=HeldClock())
    real_submit = ctl.submit

    def submit(request_id):
        # The read in flight returns while this step is still running, so the
        # frame thread is idle and would take a request made later in the step.
        gate.release()
        _until(lambda: gate.returned == 2)
        return real_submit(request_id)

    ctl.submit = submit
    running = Running(loop)
    try:
        gate.release()
        _until(lambda: gate.reads == 2)
        touch.press(SHUTTER_CENTRE, hold=0.1)
        assert session.started.wait(3.0)
        gate.release(5)
        running.more_steps(20)
        time.sleep(0.05)
        assert gate.reads == 2
    finally:
        session.release.set()
        gate.release(50)
        running.finish()
        ctl.close()
    assert running.raised is None


def test_a_frame_that_arrived_during_a_pause_is_shown_but_not_counted(tmp_path, monkeypatch):
    """A read in flight when the colour bars went up delivers its frame during
    the pause. Back in the live view it is better than a blank screen, but it
    is not a frame of this rate window."""
    camera, ctl, session = _gated_controller(tmp_path)
    gate = ReadGate(camera)
    drawn = _spy_frames(monkeypatch)
    touch = TimedTouch()
    logs = []
    loop, _, _, clock = _loop(camera, ctl, touch=touch, clock=HeldClock(), log=logs.append)
    running = Running(loop)
    try:
        gate.release()
        _until(lambda: _pictures(drawn) == 1 and gate.reads == 2)
        ctl.submit("one")
        assert session.started.wait(2.0)
        running.more_steps()
        gate.release()                            # read 2 delivers during processing
        session.release.set()
        _until(lambda: loop.state == "REVIEW")
        running.more_steps()
        touch.press((5, 5), hold=0.1)
        _until(lambda: _pictures(drawn) == 2)     # the paused frame is shown...
        _until(lambda: gate.reads == 3)           # ...and a fresh one is asked for
        clock.t = 10.0
        gate.release()
        _until(lambda: any(line.startswith("viewfinder:") for line in logs))
    finally:
        session.release.set()
        gate.release(50)
        running.finish()
        ctl.close()
    assert running.raised is None
    # One frame in the ten seconds since the live view came back, not two.
    assert [line for line in logs if line.startswith("viewfinder:")][0] == "viewfinder: 0.1 fps"


def test_a_camera_that_keeps_stalling_ends_run(controller):
    camera, ctl = controller
    logged = []
    reads = ScriptedReads(camera, [CameraStalled("no frame")] * 50)
    loop, _, display, _ = _loop(
        camera, ctl, clock=HeldClock(), log=logged.append, frame_period=0.01)
    running = Running(loop)
    running.thread.join(5.0)
    assert not running.thread.is_alive()
    assert isinstance(running.raised, CameraStalled)
    assert sum("camera: no frame" in line for line in logged) == CAMERA_STALL_LIMIT
    assert reads.reads >= CAMERA_STALL_LIMIT and display.images == []
    _until(lambda: not _frames_thread_alive())


def test_a_frame_between_stalls_resets_the_count_under_run(controller):
    camera, ctl = controller
    logged = []
    round_ = [CameraStalled("no frame")] * (CAMERA_STALL_LIMIT - 1) + [None]
    reads = ScriptedReads(camera, round_ * 3)
    loop, _, display, _ = _loop(
        camera, ctl, clock=HeldClock(), log=logged.append, frame_period=0.01)
    running = Running(loop)
    try:
        _until(lambda: reads.reads >= len(round_) * 3 + 2)
        running.more_steps()
    finally:
        running.finish()
    assert running.raised is None
    assert sum("camera: no frame" in line for line in logged) == 3 * (CAMERA_STALL_LIMIT - 1)
    assert display.images


def test_an_ordinary_camera_error_under_run_is_logged_and_survived(controller, monkeypatch):
    camera, ctl = controller
    logged = []
    reads = ScriptedReads(camera, [CameraError("busy"), CameraError("busy")])
    drawn = _spy_frames(monkeypatch)
    loop, _, _, _ = _loop(camera, ctl, clock=HeldClock(), log=logged.append, frame_period=0.01)
    running = Running(loop)
    try:
        _until(lambda: reads.reads >= 4 and _pictures(drawn) >= 2)
    finally:
        running.finish()
    assert running.raised is None
    assert logged.count("camera: busy") == 2


def test_an_unexpected_error_does_not_kill_the_frame_thread(controller, monkeypatch):
    """A thread that died would leave a frozen picture and no line in the log."""
    camera, ctl = controller
    logged = []
    reads = ScriptedReads(camera, [ValueError("boom")])
    drawn = _spy_frames(monkeypatch)
    loop, _, _, _ = _loop(camera, ctl, clock=HeldClock(), log=logged.append, frame_period=0.01)
    running = Running(loop)
    try:
        _until(lambda: reads.reads >= 3 and _pictures(drawn) >= 2)
        assert _frames_thread_alive()
    finally:
        running.finish()
    assert running.raised is None
    assert logged.count("frames: boom") == 1


def test_the_frame_thread_stops_with_the_loop(controller):
    camera, ctl = controller
    loop, _, display, _ = _loop(camera, ctl, clock=time, frame_period=0.02)
    running = Running(loop)
    try:
        _until(lambda: len(display.images) >= 3)
        assert _frames_thread_alive()
    finally:
        running.finish()
    assert not _frames_thread_alive()


def test_run_returns_when_stopped_with_a_read_still_blocked(controller, monkeypatch):
    """A read can outlast the wait for it (a stall takes five seconds to be
    called one). run() must come back anyway, and say what it left behind."""
    camera, ctl = controller
    monkeypatch.setattr(viewfinder, "FRAME_THREAD_JOIN_SECONDS", 0.1)
    gate = ReadGate(camera)
    logged = []
    loop, _, _, _ = _loop(camera, ctl, clock=HeldClock(), log=logged.append)
    running = Running(loop)
    try:
        _until(lambda: gate.reads == 1)           # blocked: no permit was ever given
        running.finish(timeout=3.0)
        assert any(line.startswith("frames:") and "did not stop" in line for line in logged)
    finally:
        gate.release(10)
    _until(lambda: not _frames_thread_alive())
    assert running.raised is None


# -- verifying fixed controls against the camera's metadata -------------------------

AGREEING = {"ExposureTime": 4000, "AnalogueGain": 8.0, "ColourTemperature": 7500, "Lux": 500.0}
RESEND_STEP = 0.64   # six frames this far apart: the sixth is 3.2 s after the first


def _picture(rgb):
    """The picture's colour as the journal lines give it: channel means, R/G and B/G."""
    r, g, b = (float(rgb[..., channel].mean()) for channel in range(3))
    return f"picture R/G {r / g:.2f} B/G {b / g:.2f}"


# Every frame of the fixture camera is this one, so every line ends the same way.
PICTURE = "; " + _picture(synthetic_frame(48, 64))


def _check_loop(camera, ctl, prefix="camera:", **kw):
    """A loop that never dims, and the ``camera:`` lines it logs (not the fps ones)."""
    logs = []

    def log(line):
        if line.startswith(prefix):
            logs.append(line)

    loop, touch, display, clock = _loop(camera, ctl, log=log, dim_after=0, off_after=0, **kw)
    return loop, touch, clock, logs


def _live_frames(loop, clock, n, dt=RESEND_STEP):
    """``n`` live steps, each a new camera frame, ``dt`` apart on the loop's clock."""
    for _ in range(n):
        loop.step()
        clock.t += dt


def _fix_wb(camera):
    camera.set_colour_temperature(7500)
    camera._metadata = {**AGREEING, "ColourTemperature": 4250}


def _fix_shutter(camera):
    camera.set_shutter(4000)
    camera._metadata = {**AGREEING, "ExposureTime": 33333}


def _fix_iso(camera):
    camera.set_iso(800)
    camera._metadata = {**AGREEING, "AnalogueGain": 1.0}


@pytest.mark.parametrize("fix, line", [
    (_fix_shutter, "camera: shutter 1/250 not in the live view after 3.2 s "
                   "(camera reports 1/30); resending" + PICTURE),
    (_fix_iso, "camera: ISO 800 not in the live view after 3.2 s "
               "(camera reports ISO 100); resending" + PICTURE),
])
def test_a_fixed_control_the_camera_does_not_report_is_resent(controller, fix, line):
    """A shutter or ISO the camera does not report is sent again first: on the
    Pi a live shutter change was seen applied even where a white balance was not."""
    camera, ctl = controller
    fix(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5)
    assert camera.reassert_count == 0 and logs == []
    _live_frames(loop, clock, 1)
    assert camera.reassert_count == 1
    assert logs == [line]


def test_auto_white_balance_that_was_never_fixed_is_not_checked(controller):
    camera, ctl = controller
    camera._metadata = dict(AGREEING)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 20)
    assert camera.reassert_count == 0 and logs == []


def test_controls_that_agree_with_the_camera_are_never_resent(controller):
    camera, ctl = controller
    camera.set_shutter(4000)
    camera.set_iso(800)
    camera.set_colour_temperature(7500)
    # Inside the tolerances: 5 % on time and gain, 100 K on white balance.
    camera._metadata = {"ExposureTime": 3990, "AnalogueGain": 8.2, "ColourTemperature": 7440}
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 20)
    assert camera.reassert_count == 0 and logs == []


def test_no_resend_before_three_seconds_however_many_frames(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 4, dt=0.5)       # frames at 0, 0.5, 1.0, 1.5
    _live_frames(loop, clock, 20, dt=0.05)     # ... and twenty more up to 2.95 s
    assert camera.reassert_count == 0
    clock.t = 3.0
    _live_frames(loop, clock, 1)
    assert camera.reassert_count == 1


def test_no_resend_before_six_frames_however_long_they_take(controller):
    """At a 1 s shutter three seconds is three frames, and the new value needs
    a frame or two to arrive in the first place."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5, dt=1.0)
    assert camera.reassert_count == 0
    _live_frames(loop, clock, 1, dt=1.0)
    assert camera.reassert_count == 1
    assert "after 5.0 s" in logs[0]


def test_a_frame_that_agrees_starts_the_wait_again(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    wrong = camera._metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5)
    camera._metadata = dict(AGREEING)
    _live_frames(loop, clock, 1)
    camera._metadata = wrong
    _live_frames(loop, clock, 5)
    assert camera.reassert_count == 0 and logs == []
    _live_frames(loop, clock, 1)
    assert camera.reassert_count == 1


def test_a_step_starts_the_wait_again(controller):
    """A new value has been sent: the camera gets the full wait to show it."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 4)
    _tap(loop, touch, clock, PLUS_CENTRE)       # two more frames; the second one steps
    assert camera.shutter_us != 4000
    # Without the step the eighth frame, 3.3 s in, would be resent to.
    _live_frames(loop, clock, 4)
    assert camera.reassert_count == 0 and logs == []
    _live_frames(loop, clock, 2)
    assert camera.reassert_count == 1


def test_leaving_the_live_view_starts_the_wait_again(controller):
    """A still re-applies the whole configuration, so what the frames before it
    showed says nothing about the ones after."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5)
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    _tap(loop, touch, clock, (5, 5))
    assert loop.state == "LIVE"
    _live_frames(loop, clock, 5)
    assert camera.reassert_count == 0 and logs == []
    _live_frames(loop, clock, 1)
    assert camera.reassert_count == 1


@pytest.mark.parametrize("metadata", [
    None, {}, {"ExposureTime": "4000", "AnalogueGain": None, "ColourTemperature": True},
    {"ExposureTime": float("nan"), "AnalogueGain": float("inf"), "ColourTemperature": [7500]},
])
def test_metadata_that_cannot_tell_is_never_a_mismatch(controller, metadata):
    camera, ctl = controller
    camera.set_shutter(4000)
    camera.set_iso(800)
    camera.set_colour_temperature(7500)
    camera._metadata = metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 20)
    camera.set_colour_temperature(None)
    _live_frames(loop, clock, 20)
    assert camera.reassert_count == 0 and logs == []


def test_a_control_the_camera_cannot_change_is_never_checked(controller, monkeypatch):
    camera, ctl = controller
    _fix_wb(camera)
    monkeypatch.setattr(camera, "wb_adjustable", False)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 20)
    assert (camera.reassert_count, camera.restart_count) == (0, 0) and logs == []


def test_the_first_agreeing_frame_after_a_resend_is_logged_once(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6)
    assert camera.reassert_count == 1
    camera._metadata = None                     # a frame that cannot tell proves nothing
    _live_frames(loop, clock, 1)
    assert len(logs) == 1
    camera._metadata = dict(AGREEING)
    _live_frames(loop, clock, 5)
    assert logs[1:] == ["camera: shutter 1/250 applied after resend"]


def test_resends_continue_but_the_line_is_logged_once_in_thirty_seconds(controller):
    """A camera that never takes the control is resent to every few seconds for
    as long as it lasts, without a journal line for each."""
    camera, ctl = controller
    _fix_shutter(camera)
    camera.restart_preview = None               # a camera that can only be resent to
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 60)               # resends at frames 6, 12, ... 60
    assert camera.reassert_count == 10
    assert logs == [
        "camera: shutter 1/250 not in the live view after 3.2 s "
        "(camera reports 1/30); resending" + PICTURE,
        "camera: shutter 1/250 not in the live view after 33.9 s "
        "(camera reports 1/30); resending (7 more)" + PICTURE,
    ]


def test_a_resend_the_camera_refuses_is_logged_and_survived(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    attempts = []

    def refuse():
        attempts.append(1)
        raise CameraError("Picamera2 failed to resend the controls: busy")

    camera.reassert_controls = refuse
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 12)
    assert len(attempts) == 2 and loop.state == "LIVE"
    assert logs.count("camera: Picamera2 failed to resend the controls: busy") == 2


def test_a_camera_that_cannot_resend_still_gets_the_journal_line(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    camera.reassert_controls = None
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6)
    assert len(logs) == 1 and "not in the live view" in logs[0]


def test_redraws_of_one_frame_do_not_count_as_frames_that_disagree(controller):
    """Under run() the loop redraws ten times a second whatever the camera
    does; only a new camera frame is new evidence."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    # As under run(), without the thread: the test publishes the frames itself.
    loop._want_frames = threading.Event()
    for seq in range(1, 6):
        loop._grab = viewfinder._Grab(seq, frame=camera.read(full=False))
        for _ in range(10):                     # one new frame, nine redraws of it
            loop.step()
            clock.t += 0.1
    assert camera.reassert_count == 0 and logs == []
    loop._grab = viewfinder._Grab(6, frame=camera.read(full=False))
    loop.step()
    assert camera.reassert_count == 1


# -- restarting the live view when resending does not help --------------------------

STILL_WRONG = ("camera: shutter 1/250 still not in the live view after 2 resends; "
               "restarting the live view" + PICTURE)


def test_two_resends_that_do_not_help_are_followed_by_a_restart_of_the_live_view(controller):
    """Seen on the Pi 4: a live control ignored for good, fixed value or return
    to auto alike, until a photo reconfigured the camera. The restart is that
    reconfiguration without the photo."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 17)
    assert (camera.reassert_count, camera.restart_count) == (2, 0)
    _live_frames(loop, clock, 1)                # the eighteenth frame, 10.9 s in
    assert (camera.reassert_count, camera.restart_count) == (2, 1)
    assert logs[-1] == STILL_WRONG
    camera._metadata = dict(AGREEING)
    _live_frames(loop, clock, 3)
    assert logs[-1] == "camera: shutter 1/250 applied after restarting the live view"
    assert logs.count(logs[-1]) == 1 and not any("after resend" in line for line in logs)
    _live_frames(loop, clock, 30)
    assert (camera.reassert_count, camera.restart_count) == (2, 1)


@pytest.mark.parametrize("frames", [6, 12])
def test_a_resend_that_cures_it_is_not_followed_by_a_restart(controller, frames):
    camera, ctl = controller
    _fix_shutter(camera)
    wrong = camera._metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, frames)           # one resend, or two
    assert camera.reassert_count == frames // 6
    camera._metadata = dict(AGREEING)
    _live_frames(loop, clock, 30)
    assert camera.restart_count == 0
    assert logs[-1] == "camera: shutter 1/250 applied after resend"
    # A new disagreement later starts from the beginning: two resends first.
    camera._metadata = wrong
    _live_frames(loop, clock, 17)
    assert camera.restart_count == 0
    _live_frames(loop, clock, 1)
    assert camera.restart_count == 1


def test_a_restart_the_camera_refuses_is_logged_and_survived(controller):
    camera, ctl = controller
    _fix_shutter(camera)

    def refuse():
        raise CameraError("Picamera2 failed to restart the live view: busy")

    camera.restart_preview = refuse
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 30)
    assert loop.state == "LIVE"
    assert logs.count("camera: Picamera2 failed to restart the live view: busy") == 1


def test_a_restart_that_stalls_counts_like_a_stalled_frame(controller):
    """A camera that cannot be restarted and then delivers no frames is the
    stalled camera the loop already gives up on."""
    camera, ctl = controller
    _fix_shutter(camera)

    def stall(*args, **kwargs):
        raise CameraStalled("no frame")

    camera.restart_preview = stall
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 18)               # the restart stalls: one
    assert logs[-1] == "camera: no frame" and loop.state == "LIVE"
    camera.read = stall
    for _ in range(CAMERA_STALL_LIMIT - 2):
        loop.step()
    with pytest.raises(CameraStalled):
        loop.step()


def test_a_camera_without_a_restart_keeps_being_resent_to(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    camera.restart_preview = None
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 30)
    assert camera.reassert_count == 5
    assert not any("restarting" in line for line in logs)


def test_a_mismatch_that_survives_its_restart_is_not_restarted_again(controller):
    """One restart for one selection: a restart blanks the picture, and one
    that did not help will not help the second time either."""
    camera, ctl = controller
    _fix_shutter(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 18)               # two resends, then the restart
    assert (camera.reassert_count, camera.restart_count) == (2, 1)
    _live_frames(loop, clock, 6)                # the next wait ends: said once, resent
    gave_up = "camera: shutter 1/250 not applied even after restarting the live view"
    assert logs[-1] == gave_up and camera.reassert_count == 3
    _live_frames(loop, clock, 120)              # well over a minute more
    assert camera.restart_count == 1 and camera.reassert_count == 23
    assert logs.count(gave_up) == 1 and logs.count(STILL_WRONG) == 1
    # A step is a new selection, and that may have its own restart.
    _tap(loop, touch, clock, PLUS_CENTRE)
    _live_frames(loop, clock, 18)
    assert camera.restart_count == 2


# -- white balance: restarted at once, never resent ---------------------------------

WB_STEP = 0.32   # six frames this far apart: the sixth is 1.6 s after the first
WB_RESTART = ("camera: white balance 7500K not in the live view after 1.6 s "
              "(camera reports 4250K); restarting the live view" + PICTURE)


def test_a_white_balance_the_camera_does_not_report_restarts_the_live_view(controller):
    """Measured on the Pi 4 on 2026-10-12: after a photo a live white-balance
    change was ignored 11 times out of 11, a resend cured 2 of about 7, and a
    restart of the live view cured it every time. So no resend is tried."""
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5, dt=WB_STEP)
    assert camera.restart_count == 0 and logs == []
    _live_frames(loop, clock, 1, dt=WB_STEP)
    assert (camera.reassert_count, camera.restart_count) == (0, 1)
    assert logs == [WB_RESTART]
    camera._metadata = dict(AGREEING)
    _live_frames(loop, clock, 3, dt=WB_STEP)
    assert logs[1:] == ["camera: white balance 7500K applied after restarting the live view"]
    _live_frames(loop, clock, 40, dt=WB_STEP)
    assert (camera.reassert_count, camera.restart_count) == (0, 1) and len(logs) == 2


def test_no_white_balance_restart_before_a_second_and_a_half(controller):
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 14, dt=0.1)       # frames up to 1.3 s
    assert camera.restart_count == 0
    clock.t = 1.5
    _live_frames(loop, clock, 1)
    assert camera.restart_count == 1


def test_no_white_balance_restart_before_six_frames(controller):
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5, dt=1.0)
    assert camera.restart_count == 0
    _live_frames(loop, clock, 1, dt=1.0)
    assert camera.restart_count == 1 and "after 5.0 s" in logs[0]


def test_an_auto_white_balance_that_has_not_moved_is_given_four_seconds(controller):
    """libcamera's own damping holds the old value for about two seconds before
    it drifts: a restart then would be a false alarm."""
    camera, ctl = controller
    camera.set_colour_temperature(7500)
    camera._metadata = dict(AGREEING)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 2, dt=0.5)
    camera.set_colour_temperature(None)
    _live_frames(loop, clock, 8, dt=0.5)        # 0 to 3.5 s on auto
    assert camera.restart_count == 0 and logs == []
    _live_frames(loop, clock, 1, dt=0.5)
    assert (camera.reassert_count, camera.restart_count) == (0, 1)
    assert logs == ["camera: auto white balance has not moved from 7500K after 4.0 s; "
                    "restarting the live view" + PICTURE]
    # That was the one restart of this return to auto; auto is not watched again.
    _live_frames(loop, clock, 40, dt=0.5)
    assert camera.restart_count == 1 and len(logs) == 1


def test_an_auto_white_balance_that_moves_in_time_is_left_alone(controller):
    camera, ctl = controller
    camera.set_colour_temperature(7500)
    camera._metadata = dict(AGREEING)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 2, dt=0.5)
    camera.set_colour_temperature(None)
    _live_frames(loop, clock, 6, dt=0.5)        # held for 2.5 s, as measured
    camera._metadata = {**AGREEING, "ColourTemperature": 7380}
    _live_frames(loop, clock, 40, dt=0.5)
    assert (camera.reassert_count, camera.restart_count) == (0, 0) and logs == []


def test_a_white_balance_that_survives_its_restart_is_left_alone(controller):
    """White balance does nothing further: no resend, no second restart, one line."""
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1
    _live_frames(loop, clock, 300, dt=WB_STEP)  # a minute and a half
    assert (camera.reassert_count, camera.restart_count) == (0, 1)
    assert logs == [
        WB_RESTART, "camera: white balance 7500K not applied even after restarting the live view"]


def _wb_selected_then_ignored(camera, ctl):
    """A loop with the WB cell selected at 5600 K, which the camera then stops reporting."""
    camera.set_colour_temperature(5600)
    camera._metadata = {**AGREEING, "ColourTemperature": 5600}
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _tap(loop, touch, clock, _cell(WB_CELL))
    camera._metadata = {**AGREEING, "ColourTemperature": 4250}
    return loop, touch, clock, logs


def test_a_new_selection_may_restart_the_live_view_again_at_once(controller):
    camera, ctl = controller
    loop, touch, clock, logs = _wb_selected_then_ignored(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1
    _live_frames(loop, clock, 30, dt=WB_STEP)   # nearly ten seconds; the same selection
    assert camera.restart_count == 1
    _tap(loop, touch, clock, PLUS_CENTRE)       # 6500 K: one frame of the new wait
    assert camera.colour_temperature_k == 6500
    _live_frames(loop, clock, 5, dt=WB_STEP)    # ... and five more, 1.28 s in
    assert camera.restart_count == 1
    _live_frames(loop, clock, 1, dt=WB_STEP)
    assert camera.restart_count == 2
    assert logs[-1].startswith("camera: white balance 6500K not in the live view after 1.6 s")


def test_two_restarts_are_never_closer_than_five_seconds(controller):
    """The backstop against a picture that blinks with every tap."""
    camera, ctl = controller
    loop, touch, clock, logs = _wb_selected_then_ignored(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1
    restarted_at = clock.t - WB_STEP
    _tap(loop, touch, clock, PLUS_CENTRE)
    _live_frames(loop, clock, 12, dt=WB_STEP)   # past both limits, 4 s after the restart
    assert clock.t - restarted_at < 5.0 and camera.restart_count == 1
    clock.t = restarted_at + 5.0
    _live_frames(loop, clock, 1)
    assert camera.restart_count == 2


def test_leaving_the_live_view_allows_a_white_balance_restart_again(controller):
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 40, dt=WB_STEP)
    assert camera.restart_count == 1
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    _tap(loop, touch, clock, (5, 5))
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 2


def test_a_white_balance_restart_the_camera_refuses_is_tried_once(controller):
    camera, ctl = controller
    _fix_wb(camera)
    attempts = []

    def refuse():
        attempts.append(1)
        raise CameraError("Picamera2 failed to restart the live view: busy")

    camera.restart_preview = refuse
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 60, dt=WB_STEP)
    assert len(attempts) == 1 and loop.state == "LIVE"
    assert logs[1:] == ["camera: Picamera2 failed to restart the live view: busy"]


def test_a_camera_that_cannot_restart_still_gets_the_white_balance_line(controller):
    camera, ctl = controller
    _fix_wb(camera)
    camera.restart_preview = None
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 60, dt=WB_STEP)
    assert camera.reassert_count == 0
    assert logs == ["camera: white balance 7500K not in the live view after 1.6 s "
                    "(camera reports 4250K); cannot restart the live view" + PICTURE]


# -- the control check: review follow-ups -------------------------------------------


@pytest.mark.parametrize("fix, key, value, mismatch", [
    (_fix_shutter, "ExposureTime", 4000 * 1.051, True),
    (_fix_shutter, "ExposureTime", 4000 * 0.949, True),
    (_fix_shutter, "ExposureTime", 4000 * 1.049, False),
    (_fix_shutter, "ExposureTime", 4000 * 0.951, False),
    (_fix_iso, "AnalogueGain", 8.0 * 1.051, True),
    (_fix_iso, "AnalogueGain", 8.0 * 0.949, True),
    (_fix_iso, "AnalogueGain", 8.0 * 1.049, False),
    (_fix_iso, "AnalogueGain", 8.0 * 0.951, False),
    (_fix_wb, "ColourTemperature", 7601, True),
    (_fix_wb, "ColourTemperature", 7399, True),
    (_fix_wb, "ColourTemperature", 7599, False),
    (_fix_wb, "ColourTemperature", 7401, False),
])
def test_the_tolerances_are_five_per_cent_and_a_hundred_kelvin(
    controller, fix, key, value, mismatch,
):
    camera, ctl = controller
    fix(camera)
    camera._metadata = {**AGREEING, key: value}
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6)
    assert (camera.reassert_count + camera.restart_count == 1) is mismatch
    assert bool(logs) is mismatch


def test_a_white_balance_that_flaps_restarts_the_live_view_once(controller):
    """A control that agrees for a frame and disagrees again must not buy a new
    restart each time: the picture would blink every five seconds for as long
    as it went on. The restart belongs to the selection."""
    camera, ctl = controller
    _fix_wb(camera)
    wrong = camera._metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1
    for _ in range(5):
        camera._metadata = dict(AGREEING)
        _live_frames(loop, clock, 1, dt=WB_STEP)
        camera._metadata = wrong
        _live_frames(loop, clock, 30, dt=WB_STEP)       # nearly ten seconds each time
    assert (camera.reassert_count, camera.restart_count) == (0, 1)
    assert sum("; restarting the live view;" in line for line in logs) == 1
    assert sum("not applied even after" in line for line in logs) == 1
    # A step that changes the white balance is a new selection: one more restart.
    _tap(loop, touch, clock, _cell(WB_CELL))
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert camera.colour_temperature_k == 6500
    _live_frames(loop, clock, 30, dt=WB_STEP)
    assert camera.restart_count == 2


def test_a_shutter_that_flaps_is_restarted_once_and_then_only_resent(controller):
    camera, ctl = controller
    _fix_shutter(camera)
    wrong = camera._metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 18)
    assert (camera.reassert_count, camera.restart_count) == (2, 1)
    for _ in range(3):
        camera._metadata = dict(AGREEING)
        _live_frames(loop, clock, 1)
        camera._metadata = wrong
        _live_frames(loop, clock, 30)                   # five waits: five resends
    assert (camera.reassert_count, camera.restart_count) == (17, 1)


def test_a_step_of_another_control_does_not_buy_a_new_restart(controller):
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1
    clock.t += 10.0
    _tap(loop, touch, clock, PLUS_CENTRE)               # the shutter cell is selected
    assert camera.shutter_us is not None
    camera._metadata = {**camera._metadata, "ExposureTime": camera.shutter_us}
    _live_frames(loop, clock, 30, dt=WB_STEP)
    assert camera.restart_count == 1


def test_two_controls_that_disagree_share_one_restart(controller):
    """The restart re-applies every control, so it is spent for each of them."""
    camera, ctl = controller
    camera.set_shutter(4000)
    camera.set_colour_temperature(7500)
    camera._metadata = {**AGREEING, "ExposureTime": 33333, "ColourTemperature": 4250}
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert camera.restart_count == 1                    # the white balance's
    _live_frames(loop, clock, 120)                      # the shutter: resends only
    assert camera.restart_count == 1 and camera.reassert_count >= 10
    assert sum("restarting the live view;" in line for line in logs) == 1


@pytest.mark.parametrize("fails", [True, False])
def test_the_five_seconds_between_restarts_count_from_the_end_of_the_first(controller, fails):
    """A restart can take seconds (it waits for the exposure in flight and for
    the camera); counted from its start, the gap to the next would shrink."""
    camera, ctl = controller
    loop, touch, clock, logs = _wb_selected_then_ignored(camera, ctl)
    attempts = []

    def slow_restart():
        attempts.append(clock.t)
        clock.t += 3.0
        if fails:
            raise CameraError("Picamera2 failed to restart the live view: busy")

    camera.restart_preview = slow_restart
    _live_frames(loop, clock, 6, dt=WB_STEP)
    assert len(attempts) == 1
    finished = attempts[0] + 3.0
    _tap(loop, touch, clock, PLUS_CENTRE)               # a new selection, also ignored
    _live_frames(loop, clock, 12, dt=WB_STEP)           # past both limits
    assert clock.t - finished < 5.0 and len(attempts) == 1
    clock.t = finished + 5.01                           # 5.0 is a rounding error either way
    _live_frames(loop, clock, 1)
    assert len(attempts) == 2


def test_a_still_auto_white_balance_is_restarted_once_and_never_again(controller):
    """With the lens capped or in the dark, auto reports the old value exactly
    and for good. One restart is the remedy; a second after every photo is not."""
    camera, ctl = controller
    camera.set_colour_temperature(7500)
    camera._metadata = dict(AGREEING)
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 2, dt=0.5)
    camera.set_colour_temperature(None)
    _live_frames(loop, clock, 40, dt=0.5)
    assert camera.restart_count == 1 and len(logs) == 1
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    _tap(loop, touch, clock, (5, 5))
    _live_frames(loop, clock, 40, dt=0.5)
    assert camera.restart_count == 1 and len(logs) == 1


def test_frames_that_cannot_tell_restart_the_wait_without_ending_it(controller):
    """Without this, a control whose metadata went missing for a minute would
    be acted on by the first frame that brought it back."""
    camera, ctl = controller
    _fix_shutter(camera)
    wrong = camera._metadata
    loop, touch, clock, logs = _check_loop(camera, ctl)
    _live_frames(loop, clock, 5)
    camera._metadata = None
    _live_frames(loop, clock, 1)
    camera._metadata = wrong
    _live_frames(loop, clock, 5)
    assert camera.reassert_count == 0 and logs == []
    _live_frames(loop, clock, 1)
    assert camera.reassert_count == 1
    assert "after 7.0 s" in logs[0]                     # counted from the first disagreement


def test_the_time_a_restart_takes_is_not_averaged_into_the_frame_rate(controller):
    camera, ctl = controller
    _fix_wb(camera)
    loop, touch, clock, logs = _check_loop(camera, ctl, prefix="viewfinder:")

    def slow_restart():
        clock.t += 8.0

    camera.restart_preview = slow_restart
    _live_frames(loop, clock, 60, dt=WB_STEP)
    assert float(logs[0].split()[1]) > 3.0              # 3.1 frames a second, not 0.8


# -- a trace line after every control step ------------------------------------------

TRACED = {"ExposureTime": 16667, "AnalogueGain": 4.0, "ColourTemperature": 7500,
          "ColourGains": [3.35, 1.36], "Lux": 500.0}
REPORTS = "camera reports 7500K gains (3.35, 1.36), exposure 1/60, ISO 400"


def _trace_loop(camera, ctl, metadata=TRACED):
    camera._metadata = metadata
    loop, touch, clock, logs = _check_loop(camera, ctl, prefix="control:")
    loop.step()                                 # one frame, so the first step has a metered value
    return loop, touch, clock, logs


@pytest.mark.parametrize("cell, selected", [
    (SHUTTER_CELL, "SHUTTER 1/60"), (ISO_CELL, "ISO 400"), (WB_CELL, "WB 7500K"),
])
def test_a_step_is_traced_once_two_seconds_later(controller, cell, selected):
    """The camera alone applies a live control correctly, so the fault seen on
    the Pi is in how the app runs: each step leaves what was selected, what the
    camera reported and what the picture looked like in the journal."""
    camera, ctl = controller
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, _cell(cell))
    _tap(loop, touch, clock, PLUS_CENTRE)
    _live_frames(loop, clock, 4, dt=0.5)        # frames 0 to 1.5 s after the step
    assert logs == []
    _live_frames(loop, clock, 1, dt=0.5)
    assert logs == [f"control: {selected} selected; {REPORTS}{PICTURE}"]
    _live_frames(loop, clock, 20, dt=0.5)
    assert len(logs) == 1


def test_the_trace_of_a_return_to_auto_says_auto(controller):
    camera, ctl = controller
    camera.set_colour_temperature(3200)
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, _cell(WB_CELL))
    _tap(loop, touch, clock, MINUS_CENTRE)
    assert camera.colour_temperature_k is None
    _live_frames(loop, clock, 5, dt=0.5)
    assert logs == [f"control: WB AUTO selected; {REPORTS}{PICTURE}"]


def test_the_trace_writes_a_dash_for_what_the_camera_does_not_report(controller):
    camera, ctl = controller
    camera._frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    loop, touch, clock, logs = _trace_loop(camera, ctl, metadata=None)
    _tap(loop, touch, clock, PLUS_CENTRE)
    assert camera.shutter_us is not None
    _live_frames(loop, clock, 5, dt=0.5)
    assert logs == [
        f"control: SHUTTER {viewfinder.shutter.label(camera.shutter_us)} selected; "
        "camera reports - gains -, exposure -, ISO -; picture R/G - B/G -"]


def test_a_second_step_replaces_the_pending_trace(controller):
    camera, ctl = controller
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, _cell(ISO_CELL))
    _tap(loop, touch, clock, PLUS_CENTRE)       # ISO 400
    _live_frames(loop, clock, 3, dt=0.5)
    _tap(loop, touch, clock, PLUS_CENTRE)       # ISO 800, 1.6 s after the first step
    _live_frames(loop, clock, 4, dt=0.5)        # up to 1.5 s after the second
    assert logs == []
    _live_frames(loop, clock, 2, dt=0.5)        # 2.0 s is a rounding error either way
    assert logs == [f"control: ISO 800 selected; {REPORTS}{PICTURE}"]


def test_leaving_the_live_view_cancels_the_trace(controller):
    """The photo reconfigures the camera: what follows says nothing of the step."""
    camera, ctl = controller
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, PLUS_CENTRE)
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    _tap(loop, touch, clock, (5, 5))
    assert loop.state == "LIVE"
    _live_frames(loop, clock, 20, dt=0.5)
    assert logs == []


def test_a_step_the_camera_rejects_is_not_traced(controller):
    camera, ctl = controller

    def refuse(us):
        raise CameraError("rejected")

    camera.set_shutter = refuse
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, PLUS_CENTRE)
    _live_frames(loop, clock, 20, dt=0.5)
    assert logs == []


def test_a_step_that_changes_nothing_is_not_traced(controller):
    camera, ctl = controller
    camera.set_colour_temperature(7500)         # the top step: + stays there
    loop, touch, clock, logs = _trace_loop(camera, ctl)
    _tap(loop, touch, clock, _cell(WB_CELL))
    _tap(loop, touch, clock, PLUS_CENTRE)
    _live_frames(loop, clock, 20, dt=0.5)
    assert logs == []
