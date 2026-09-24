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

    def spy(frame_rgb, reading):
        readings.append(reading)
        return real(frame_rgb, reading)

    monkeypatch.setattr(viewfinder, "render_live", spy)
    return readings


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


def test_a_tap_on_a_dark_screen_only_wakes_it(controller, monkeypatch):
    """The user cannot see what they are touching: the shutter must not fire."""
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl)
    submitted = []
    monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0
    _tap(loop, touch, clock, SHUTTER_CENTRE)
    assert submitted == []
    assert display.levels[-1] == 80
    assert loop.state == "LIVE"


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
