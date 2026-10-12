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
    assert readings[-1].focus_peak == pytest.approx(readings[-1].focus)


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
