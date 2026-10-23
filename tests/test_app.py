import argparse
import builtins
import dataclasses
import io
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import (
    CaptureSession,
    _announce,
    _colour_gains,
    main,
    run_headless_loop,
    run_preview_loop,
)
from pifilm.capture.camera import CameraError, FakeCamera, Frame, StreamInfo, synthetic_frame
from pifilm.capture.controller import CaptureController
from pifilm.capture.gradequeue import read_entries, write_entry
from pifilm.gpu.grader import GpuGrader
from pifilm.gpu.shader import BACKEND
from pifilm.grain import GrainParams
from pifilm.halation import HalationParams
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline
from pifilm.tuning import Resolved
from tests.gpu_fakes import FakeDevice, tile_artifacts


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def pipeline(tmp_path):
    d = tmp_path / "art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(d))


def _session(tmp_path, pipeline, camera=None, now=None):
    camera = camera or FakeCamera([synthetic_frame(90, 160)])
    return CaptureSession(camera, pipeline, tmp_path / "shots", now=now,
                          seed_rng=np.random.default_rng(0))


def test_remote_listener_refuses_to_start_without_a_token(monkeypatch, capsys):
    """A listener without the shared secret would be an unauthenticated shutter button."""
    monkeypatch.delenv("PIFILM_REMOTE_TOKEN", raising=False)

    assert main(["--remote-listen", "127.0.0.1:8765"]) == 2

    assert "PIFILM_REMOTE_TOKEN" in capsys.readouterr().err


def test_remote_listener_keeps_running_without_a_tty(monkeypatch, tmp_path):
    """The remote control path must not fall through to the terminal-only error."""
    from pifilm.capture import app

    listeners = []

    class Listener:
        def __init__(self, controller, token, listen, power=None):
            self.controller = controller
            self.token = token
            self.listen = listen
            self.power = power
            self.started = False
            self.closed = False
            listeners.append(self)

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "secret")
    monkeypatch.setattr(app, "RemoteCaptureServer", Listener)
    monkeypatch.setattr(app.Artifacts, "resolve", lambda _: object())
    monkeypatch.setattr(app, "Pipeline", lambda _: object())
    monkeypatch.setattr(app, "has_display", lambda: False)
    monkeypatch.setattr(app.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(app.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))

    assert app.main(["--fake", "--no-preview", "--remote-listen", "127.0.0.1:8765"]) == 0
    assert len(listeners) == 1
    assert listeners[0].started and listeners[0].closed


def test_ups_monitor_is_never_left_running_when_startup_fails(tmp_path, monkeypatch):
    from pifilm.capture import app

    class FakeMonitor:
        def __init__(self):
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

        def snapshot(self):
            return None

    monitor = FakeMonitor()
    monkeypatch.setattr(app, "build_ups", lambda name: monitor)

    code = app.main(
        ["--fake", "--no-preview", "--ups", "x728", "--artifacts", str(tmp_path / "missing")]
    )

    assert code == 2
    assert not monitor.started or monitor.closed


def test_raw_mode_writes_the_camera_bytes_verbatim(tmp_path, pipeline):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, cam, now=lambda: fixed).capture()
    assert result.original.name == "210507_original.jpg"
    assert result.original.read_bytes() == data, "the camera's own bytes must be saved unchanged"
    assert result.record["frame_source"] == "raw-mjpeg"


def test_decoded_mode_names_the_file_ungraded(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, now=lambda: fixed).capture()
    assert result.original.name == "210507_ungraded.jpg"
    assert result.record["frame_source"] == "decoded"


def test_both_outputs_come_from_one_acquisition(tmp_path, pipeline):
    """A camera whose frames differ every read would produce mismatched files."""

    class ChangingCamera:
        stream_info = StreamInfo(64, 48, 30.0, "MJPG", False)

        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return Frame(np.full((48, 64, 3), self.n * 20, np.uint8), None, "decoded")

        def close(self):
            pass

    cam = ChangingCamera()
    result = _session(tmp_path, pipeline, cam).capture()
    saved = np.asarray(Image.open(result.original).convert("RGB"))
    assert cam.n == 1, "capture must read exactly one frame"
    assert abs(int(saved.mean()) - 20) <= 2


def test_log_line_carries_full_provenance(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    session.capture()
    line = json.loads((tmp_path / "shots" / "2026-09-03" / "captures.jsonl").read_text().strip())
    assert set(line) >= {
        "timestamp", "original", "pifilm", "frame_source", "wb_gains", "exposure_gain",
        "clamped", "grain_seed", "lut_sha1", "normalize_sha1", "params_version",
        "package_version",
        "width", "height", "fourcc", "fps", "pipeline_ms", "shutter_to_saved_ms",
        "ev_comp",
    }
    assert line["shutter_to_saved_ms"] >= line["pipeline_ms"]
    assert isinstance(line["grain_seed"], int)
    # Exposure compensation is gone, but the field stays (always zero) so old and
    # new records read the same way.
    assert line["ev_comp"] == 0.0


def test_capture_grades_with_the_cameras_ev_compensation(tmp_path, pipeline):
    """The session grades with whatever ``camera.ev`` says, or normalisation undoes
    a darker exposure (found on the device with the viewfinder's old EV buttons).
    The cameras now hold it at 0.0; the attribute is set here by hand."""
    from pifilm.color import luminance, srgb_to_linear

    def mean_linear(path):
        rgb = load_rgb(path)[0].astype(np.float32) / 255.0
        return float(luminance(srgb_to_linear(rgb)).mean())

    from pifilm.color import linear_to_srgb

    frame = synthetic_frame(90, 160)
    # What AE delivers at EV -2: the same scene, a quarter of the light.
    dark = linear_to_srgb(srgb_to_linear(frame / 255.0) * 0.25)
    dark = np.clip(dark * 255.0 + 0.5, 0, 255).astype(np.uint8)
    zero = _session(tmp_path / "a", pipeline, FakeCamera([frame])).capture()
    cam = FakeCamera([dark])
    cam.ev = -2.0
    minus = _session(tmp_path / "b", pipeline, cam).capture()
    assert minus.record["ev_comp"] == -2.0 and zero.record["ev_comp"] == 0.0
    stops = np.log2(mean_linear(minus.pifilm) / mean_linear(zero.pifilm))
    assert stops == pytest.approx(-2.0, abs=0.3)


def test_recorded_seed_reproduces_the_graded_file(tmp_path, pipeline):
    """The seed is the only randomness, so it must pin the grade exactly.

    Two claims, checked separately, because conflating them hides which one
    broke. In memory the reproduction is bit-exact. Through the saved files
    it cannot be: both are quality-95 JPEGs, and grain is precisely the
    high-frequency content JPEG discards, so the bounds below are what the
    format allows (measured: mean 1.1, 99th percentile 4, worst pixel 8-12).
    """
    session = _session(tmp_path, pipeline)
    result = session.capture()
    seed = result.record["grain_seed"]
    assert isinstance(seed, int)

    frame = session.camera.read().rgb  # FakeCamera repeats the same frame
    first, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    second, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    assert np.array_equal(first, second), "the same seed must give the same pixels"

    original = np.asarray(Image.open(result.original).convert("RGB"))
    saved = np.asarray(Image.open(result.pifilm).convert("RGB"))
    again, _ = pipeline.process(original, rng=np.random.default_rng(seed))
    diff = np.abs(again.astype(int) - saved.astype(int))
    assert diff.mean() < 3.0
    assert np.percentile(diff, 99) <= 10


def test_same_second_captures_do_not_collide(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    a, b = session.capture(), session.capture()
    assert a.original.name == "210507_ungraded.jpg"
    assert b.original.name == "210507-2_ungraded.jpg"


def test_preview_frame_is_small_rgb(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    assert session.preview_frame(graded=True).shape == (360, 640, 3)
    assert session.preview_frame(graded=False).shape == (360, 640, 3)


def test_preview_frame_leaves_halation_and_grain_out(tmp_path, pipeline):
    """The live view is a viewfinder: the quick grade and the photograph have the
    halation, the live view does without it, as it does without the grain."""
    calls = []

    class Recording:
        def process(self, rgb, **kwargs):
            calls.append(kwargs)
            return pipeline.process(rgb, **kwargs)

    session = _session(tmp_path, pipeline)
    session.pipeline = Recording()
    session.preview_frame(graded=True)
    assert calls and calls[-1]["halation"] is False and calls[-1]["grain"] is False


def test_headless_loop_captures_on_space_and_quits_on_q(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    keys = iter([None, " ", "x", " ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 2
    assert len(list((tmp_path / "shots").rglob("*_graded.jpg"))) == 2
    assert any("Saved" in m for m in messages)


def test_headless_loop_reports_loading_before_saved_result(tmp_path, pipeline):
    """A synchronous direct capture would skip the shared job's loading state."""
    session = _session(tmp_path, pipeline)
    keys = iter([" ", "q"])
    messages = []

    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 1

    assert messages[1] == "Processing photo..."
    assert messages[2].startswith("Saved ")


def test_headless_loop_survives_a_camera_error(tmp_path, pipeline):
    class FlakyCamera(FakeCamera):
        def read(self):
            raise CameraError("boom")

    session = _session(tmp_path, pipeline, FlakyCamera())
    keys = iter([" ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 0
    assert any("boom" in m for m in messages)


def test_preview_loop_falls_back_when_gui_is_unavailable(tmp_path, pipeline, monkeypatch):
    import cv2

    def boom(*a, **k):
        raise cv2.error("no GUI support")

    monkeypatch.setattr(cv2, "namedWindow", boom)
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_capture_display_closes_owned_controller_when_window_initialization_fails(
    tmp_path, pipeline, monkeypatch,
):
    """Returning before the display loop must not leave a worker owning the camera."""
    import cv2

    from pifilm.capture import app
    from pifilm.capture.controller import CaptureController as RealCaptureController

    session = _session(tmp_path, pipeline)
    session.camera.close = Mock(wraps=session.camera.close)
    controllers = []

    class RecordingController:
        def __init__(self, session):
            self.real = RealCaptureController(session)
            controllers.append(self)

        def close(self):
            self.real.close()

    def no_gui(*args):
        raise cv2.error("no GUI")

    monkeypatch.setattr(app, "CaptureController", RecordingController)
    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    try:
        assert run_preview_loop(session, captures_only=True) is False
        assert len(controllers) == 1
        assert session.camera.close.call_count == 1
    finally:
        for controller in controllers:
            controller.close()


def test_preview_loop_falls_back_when_imshow_fails(tmp_path, pipeline, monkeypatch, capture_gui):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: (_ for _ in ()).throw(cv2.error("no GUI")))
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_survives_a_frame_error(tmp_path, pipeline, monkeypatch, capture_gui):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    keys = iter([ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1 if delay == 50 else next(keys))

    session = _session(tmp_path, pipeline)
    calls = {"n": 0}
    real = session.preview_frame

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CameraError("dropped frame")
        return real(*a, **k)

    monkeypatch.setattr(session, "preview_frame", flaky)
    assert run_preview_loop(session) is True  # a dropped frame must not end the session


@pytest.fixture
def capture_gui(monkeypatch):
    import cv2

    shown = []
    monkeypatch.setattr(cv2, "namedWindow", Mock())
    monkeypatch.setattr(cv2, "resizeWindow", Mock())
    monkeypatch.setattr(cv2, "setWindowProperty", Mock())
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: cv2.WINDOW_FREERATIO)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 640, 360))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(frame.copy()))
    return shown


@pytest.mark.parametrize("terminal_controls", [False, True])
def test_capture_display_changes_only_after_snapshot(
    tmp_path, pipeline, monkeypatch, capture_gui, terminal_controls,
):
    import cv2

    camera = FakeCamera([synthetic_frame(90, 160), 255 - synthetic_frame(90, 160)])
    camera.read = Mock(wraps=camera.read)
    pipeline.process = Mock(wraps=pipeline.process)
    session = _session(tmp_path, pipeline, camera)
    events = iter([" ", None, "p", " ", None, "Q"])

    def next_key():
        key = next(events)
        if key is None:
            # Give the worker a chance to finish; the following GUI iteration
            # owns the repaint after polling the immutable job snapshot.
            time.sleep(0.1)
            return None
        return key

    def wait_key(delay):
        if delay == 50:
            return -1
        return -1 if terminal_controls else (ord(key) if (key := next_key()) else -1)

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    if terminal_controls:

        def read_key():
            key = next_key()
            return key
    else:
        read_key = None
    assert run_preview_loop(session, captures_only=True, read_key=read_key)

    files = sorted((tmp_path / "shots").rglob("*_graded.jpg"), key=lambda p: p.stat().st_mtime_ns)
    assert len(files) == 2
    for displayed, saved in zip(capture_gui[2::2], files, strict=True):
        rgb, _ = load_rgb(saved)
        expected = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        assert np.array_equal(displayed, expected[:, :, ::-1])
    assert not np.array_equal(capture_gui[2], capture_gui[4])


def test_capture_display_polls_controller_and_shows_loading_before_result(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    """Calling ``session.capture`` on the GUI thread would prevent this repaint."""
    import cv2

    from pifilm.capture.controller import CaptureController

    session = _session(tmp_path, pipeline)
    capture = session.capture
    started = threading.Event()
    release = threading.Event()

    def delayed_capture(*, delay_s=0.0):
        started.set()
        assert release.wait(timeout=1)
        return capture()

    monkeypatch.setattr(session, "capture", delayed_capture)
    controller = CaptureController(session)
    waits = 0

    def wait_key(delay):
        nonlocal waits
        if delay in (1, 50):
            return -1
        waits += 1
        if waits == 1:
            return ord(" ")
        if waits == 2:
            assert started.wait(timeout=1)
            assert len(np.unique(capture_gui[-1][0], axis=0)) == 7
            release.set()
            time.sleep(0.05)
            return -1
        assert len(np.unique(capture_gui[-1][0], axis=0)) != 7
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    try:
        assert run_preview_loop(session, captures_only=True, controller=controller)
    finally:
        release.set()
        controller.close()


@pytest.mark.parametrize("finish_between_polls", [False, True])
def test_external_capture_updates_pi_display(
    tmp_path, pipeline, monkeypatch, capture_gui, finish_between_polls,
):
    import cv2

    from pifilm.capture.controller import CaptureController

    session = _session(tmp_path, pipeline)
    capture = session.capture
    release = threading.Event()
    finished = threading.Event()

    def delayed_capture(*, delay_s=0.0):
        assert release.wait(timeout=2)
        result = capture()
        finished.set()
        return result

    monkeypatch.setattr(session, "capture", delayed_capture)
    controller = CaptureController(session)
    waits = 0

    def finish_capture():
        release.set()
        assert finished.wait(timeout=2)
        for _ in range(200):
            if controller.status("remote-shot").state == "complete":
                return
            time.sleep(0.001)
        pytest.fail("capture did not complete")

    def wait_key(delay):
        nonlocal waits
        if delay in (1, 50):
            return -1
        waits += 1
        if waits == 1:
            controller.submit("remote-shot")
            if finish_between_polls:
                finish_capture()
            return -1
        if waits == 2 and not finish_between_polls:
            assert len(np.unique(capture_gui[-1][0], axis=0)) == 7
            finish_capture()
            return -1
        saved = controller.status("remote-shot").result.pifilm
        rgb, _ = load_rgb(saved)
        expected = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        assert np.array_equal(capture_gui[-1], expected)
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    try:
        assert run_preview_loop(session, captures_only=True, controller=controller)
    finally:
        release.set()
        controller.close()


def test_capture_display_waits_for_a_shutter_delay_before_the_loading_screen(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    """A GPIO press waiting out its delay is not capturing yet: no loading screen."""
    import cv2

    from pifilm.capture.controller import CaptureController

    session = _session(tmp_path, pipeline)
    controller = CaptureController(session)
    waits = 0
    shown = []

    def wait_key(delay):
        nonlocal waits
        if delay in (1, 50):
            return -1
        waits += 1
        if waits == 1:
            shown.append(len(capture_gui))
            assert controller.submit("button-shot", delay_s=30.0).state == "queued"
            return -1
        if waits < 4:
            assert controller.snapshot().delaying is True
            return -1
        assert len(capture_gui) == shown[0]  # nothing drawn over the last picture
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    try:
        assert run_preview_loop(session, captures_only=True, controller=controller)
    finally:
        controller.close()
    assert controller.status("button-shot").error_code == "closed"


@pytest.mark.parametrize("failed_attempt", [1, 2])
def test_capture_display_keeps_last_photo_after_failed_capture(
    tmp_path, pipeline, monkeypatch, capture_gui, capsys, failed_attempt,
):
    import cv2

    session = _session(tmp_path, pipeline)
    capture = session.capture
    attempts = 0

    def flaky_capture(*, delay_s=0.0):
        nonlocal attempts
        attempts += 1
        if attempts == failed_attempt:
            raise CameraError("dropped snapshot")
        return capture()

    monkeypatch.setattr(session, "capture", flaky_capture)
    events = iter([" ", None, " ", None, "q"])

    def next_key(delay):
        if delay in (1, 50):
            return -1
        key = next(events)
        if key is None:
            time.sleep(0.1)
            return -1
        return ord(key)

    monkeypatch.setattr(cv2, "waitKey", next_key)
    assert run_preview_loop(session, captures_only=True)
    assert "dropped snapshot" in capsys.readouterr().out
    if failed_attempt == 2:
        assert np.array_equal(capture_gui[-1], capture_gui[2])
    assert len(list((tmp_path / "shots").rglob("*_graded.jpg"))) == 1


def test_capture_display_preserves_portrait_aspect_ratio(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(160, 90)]))
    keys = iter([32, -1, ord("q")])

    def wait_key(delay):
        key = -1 if delay in (1, 50) else next(keys)
        if key == -1 and delay == 30:
            time.sleep(0.1)
        return key

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    assert run_preview_loop(session, captures_only=True)
    assert capture_gui[-1].shape == (360, 640, 3)
    assert not capture_gui[-1][:, :219].any()
    assert not capture_gui[-1][:, 421:].any()
    assert capture_gui[-1][:, 219:421].any()


@pytest.mark.parametrize(
    "size, image_rect",
    [
        ((1920, 1080), (240, 0, 1440, 1080)),
        ((1280, 1024), (0, 32, 1280, 960)),
        ((800, 480), (80, 0, 640, 480)),
        ((1080, 1920), (0, 555, 1080, 810)),
        ((3840, 2160), (480, 0, 2880, 2160)),
    ],
)
def test_fullscreen_capture_fits_display_resolution(
    tmp_path, pipeline, monkeypatch, capture_gui, size, image_rect,
):
    import cv2

    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *size))
    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(120, 160)]))
    keys = iter([32, -1, ord("q")])

    def wait_key(delay):
        key = -1 if delay in (1, 50) else next(keys)
        if key == -1 and delay == 30:
            time.sleep(0.1)
        return key

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    assert run_preview_loop(session, captures_only=True)
    cv2.namedWindow.assert_called_once_with(
        cv2.namedWindow.call_args.args[0], cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO,
    )
    cv2.setWindowProperty.assert_called_once_with(
        cv2.namedWindow.call_args.args[0], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
    )
    assert len(capture_gui) == 4  # initial drawable, fullscreen prompt, loading bars, photo
    for frame in capture_gui[1:]:
        assert frame.shape == (size[1], size[0], 3)
    assert len(np.unique(capture_gui[2][0], axis=0)) == 7

    left, top, width, height = image_rect
    saved, = (tmp_path / "shots").rglob("*_graded.jpg")
    rgb, _ = load_rgb(saved)
    expected = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    actual = capture_gui[-1]
    assert np.array_equal(actual[top:top + height, left:left + width], expected)
    borders = actual.copy()
    borders[top:top + height, left:left + width] = 0
    assert not borders.any()


def test_resolution_change_redraws_full_resolution_photo_without_recapture(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    from pifilm.capture.controller import CaptureController

    camera = FakeCamera([synthetic_frame(720, 1280)])
    camera.read = Mock(wraps=camera.read)
    pipeline.process = Mock(wraps=pipeline.process)
    session = _session(tmp_path, pipeline, camera)
    controller = CaptureController(session)
    keys = iter([32, -1, -1, ord("q")])
    pauses = 0

    def wait_key(delay):
        nonlocal pauses
        if delay in (1, 50):
            return -1
        key = next(keys)
        if key == -1:
            pauses += 1
            if pauses == 1:
                for _ in range(100):
                    if controller.snapshot().active_job is None:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("capture worker did not finish")
            else:
                monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 1280, 720))
        return key

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    try:
        assert run_preview_loop(session, captures_only=True, controller=controller)
    finally:
        controller.close()
    assert camera.read.call_count == pipeline.process.call_count == 1
    assert len(capture_gui) == 4  # prompt, loading, photo, resized photo
    saved, = (tmp_path / "shots").rglob("*_graded.jpg")
    rgb, _ = load_rgb(saved)
    assert np.array_equal(capture_gui[-1], rgb[:, :, ::-1])


def test_fullscreen_live_preview_preserves_camera_aspect_ratio(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(120, 160)]))
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 1920, 1080))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    assert run_preview_loop(session)
    assert capture_gui[-1].shape == (1080, 1920, 3)
    assert not capture_gui[-1][:, :240].any()
    assert not capture_gui[-1][:, 1680:].any()
    assert capture_gui[-1][:, 240:1680].any()


@pytest.mark.parametrize("geometry", [(0, 0, 0, 0), (0, 0, -1, -1), (0, 0, 1, 1), None])
def test_fullscreen_tolerates_unavailable_geometry(
    tmp_path, pipeline, monkeypatch, capture_gui, geometry,
):
    import cv2

    def get_rect(name):
        if geometry is None:
            raise cv2.error("geometry unavailable")
        return geometry

    monkeypatch.setattr(cv2, "getWindowImageRect", get_rect)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True)
    assert capture_gui[-1].shape == (360, 640, 3)


@pytest.mark.parametrize(
    "image_size, display_size",
    [((1280, 720), (1280, 1024)), ((1080, 608), (1080, 1920)), ((1440, 1080), (1920, 1080))],
)
def test_gtk_geometry_includes_borders_outside_the_image(
    monkeypatch, capture_gui, image_size, display_size,
):
    import cv2

    from pifilm.capture.app import _window_size

    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *image_size))
    monkeypatch.setattr(
        cv2, "getWindowProperty", lambda name, prop: display_size[0] / display_size[1],
    )
    assert _window_size("capture", (640, 360)) == display_size


def test_gtk_first_image_allocation_does_not_shrink_fullscreen_drawable(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    """GTK's first-image allocation overrides fullscreen if it hasn't been painted yet."""
    import cv2

    state = {"painted": False, "fullscreen": False, "stuck": False, "size": (1, 1)}

    def fullscreen(name, prop, value):
        if prop == cv2.WND_PROP_FULLSCREEN:
            state["fullscreen"] = True
            state["stuck"] = not state["painted"]

    keys = iter([32, ord("q")])

    def wait_key(delay):
        if capture_gui and not state["painted"]:
            state["painted"] = True
            state["size"] = capture_gui[0].shape[1::-1]
        if state["fullscreen"] and not state["stuck"]:
            state["size"] = (1920, 1080)
        return -1 if delay in (1, 50) else next(keys)

    monkeypatch.setattr(cv2, "setWindowProperty", fullscreen)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *state["size"]))
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True)
    assert not state["stuck"]
    assert capture_gui[-2].shape == (1080, 1920, 3)  # loading bars
    assert capture_gui[-1].shape == (1080, 1920, 3)  # saved photo
    cv2.resizeWindow.assert_any_call(cv2.namedWindow.call_args.args[0], 1920, 1080)


@pytest.mark.parametrize(
    "failure_at",
    ["namedWindow", "resizeWindow", "setWindowProperty", "imshow", "waitKey",
     "loading_image", "saved_image"],
)
def test_capture_display_gui_failure_cleans_up_and_allows_fallback(
    tmp_path, pipeline, monkeypatch, capture_gui, failure_at,
):
    import cv2

    cleanup = Mock()
    monkeypatch.setattr(cv2, "destroyAllWindows", cleanup)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 32)

    def boom(*args, **kwargs):
        raise cv2.error("no GUI")

    if failure_at in ("loading_image", "saved_image"):
        def show(name, frame):
            if len(capture_gui) == (1 if failure_at == "loading_image" else 2):
                boom()
            capture_gui.append(frame.copy())
        monkeypatch.setattr(cv2, "imshow", show)
    else:
        monkeypatch.setattr(cv2, failure_at, boom)
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True) is False
    assert cleanup.call_count == (0 if failure_at == "namedWindow" else 1)
    if failure_at == "saved_image":
        assert len(list((tmp_path / "shots").rglob("*_graded.jpg"))) == 1


@pytest.mark.parametrize("no_preview", [False, True])
@pytest.mark.parametrize("has_terminal", [False, True])
def test_main_show_captures_selects_snapshot_mode(
    tmp_path, monkeypatch, no_preview, has_terminal,
):
    from pifilm.capture import app

    monkeypatch.setattr(app, "has_display", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: has_terminal)
    terminal = Mock()
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    window = Mock(return_value=True)
    monkeypatch.setattr(app, "run_preview_loop", window)
    args = ["--fake", "--show-captures", "--out", str(tmp_path)]
    if no_preview:
        args.append("--no-preview")
    assert main(args) == 0
    assert window.call_args.kwargs["captures_only"] is True
    if has_terminal:
        window.call_args.kwargs["read_key"]()
        terminal.read.assert_called_once_with(timeout=0)
        terminal.__exit__.assert_called_once()


@pytest.mark.parametrize("display_available", [False, True])
def test_main_show_captures_falls_back_to_terminal(tmp_path, monkeypatch, display_available):
    from pifilm.capture import app

    monkeypatch.setattr(app, "has_display", lambda: display_available)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    terminal = Mock()
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr(app, "run_preview_loop", Mock(return_value=False))
    headless = Mock()
    monkeypatch.setattr(app, "run_headless_loop", headless)
    assert main(["--fake", "--no-preview", "--show-captures", "--out", str(tmp_path)]) == 0
    headless.assert_called_once()


def test_main_headless_without_tty_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(["--fake", "--no-preview", "--out", str(tmp_path)])
    assert code == 2
    assert "terminal" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    code = main(
        ["--fake", "--no-preview", "--out", str(tmp_path), "--artifacts", str(tmp_path / "nope")]
    )
    assert code == 2
    assert "params.json" in capsys.readouterr().err


class _CameraDouble:
    stream_info = StreamInfo(64, 48, 30.0, "TEST", False)

    def __init__(self):
        self.closed = False
        self.ev = 0.0

    def close(self):
        self.closed = True


@pytest.fixture
def camera_cli(monkeypatch):
    """Run ``main`` through its real terminal loop without camera hardware."""
    from pifilm.capture import app

    monkeypatch.setattr(app.Artifacts, "resolve", lambda _: object())
    monkeypatch.setattr(app, "Pipeline", lambda _: object())
    monkeypatch.setattr(app, "has_display", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    terminal = SimpleNamespace(
        __enter__=lambda self: self,
        __exit__=lambda self, *exc: None,
        read=lambda: "q",
    )

    class Terminal:
        def __enter__(self):
            return terminal

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(app, "TerminalKeys", Terminal)
    return app


def test_explicit_v4l2_never_imports_picamera2(camera_cli, monkeypatch):
    real_import = builtins.__import__
    opened = []

    def no_picamera_import(name, *args, **kwargs):
        if name == "picamera2":
            raise AssertionError("explicit V4L2 must not probe Picamera2")
        return real_import(name, *args, **kwargs)

    def open_v4l2(device):
        opened.append((device, _CameraDouble()))
        return opened[-1][1]

    monkeypatch.setattr(builtins, "__import__", no_picamera_import)
    monkeypatch.setattr(camera_cli, "V4L2Camera", open_v4l2)

    assert camera_cli.main(["--camera", "v4l2", "--no-preview"]) == 0
    assert opened[0][0] is None
    assert opened[0][1].closed


def test_explicit_picamera2_receives_tuning_file(camera_cli, monkeypatch):
    opened = []

    def open_picamera(tuning_file, **kwargs):
        opened.append((tuning_file, _CameraDouble()))
        return opened[-1][1]

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(
        camera_cli,
        "V4L2Camera",
        lambda device: (_ for _ in ()).throw(AssertionError("wrong backend")),
    )

    assert camera_cli.main(
        ["--camera", "picamera2", "--tuning-file", "delivered.json", "--no-preview"]
    ) == 0
    assert opened[0][0] == "delivered.json"
    assert opened[0][1].closed


def test_explicit_picamera2_reports_optional_library_load_failure(
    camera_cli, monkeypatch, capsys,
):
    real_import = builtins.__import__

    def broken_picamera_import(name, *args, **kwargs):
        if name == "picamera2":
            raise OSError("libcamera.so could not be loaded")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_picamera_import)

    assert camera_cli.main(["--camera", "picamera2", "--no-preview"]) == 2
    error = capsys.readouterr().err
    assert "Picamera2" in error
    assert "install" in error


def test_fake_bypasses_all_hardware_detection(camera_cli, monkeypatch):
    real_import = builtins.__import__

    def no_picamera_import(name, *args, **kwargs):
        if name == "picamera2":
            raise AssertionError("fake mode must not probe Picamera2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_picamera_import)
    monkeypatch.setattr(
        camera_cli,
        "V4L2Camera",
        lambda device: (_ for _ in ()).throw(AssertionError("fake opened V4L2")),
    )
    monkeypatch.setattr(
        camera_cli,
        "Picamera2Camera",
        lambda tuning, **kwargs: (_ for _ in ()).throw(AssertionError("fake opened Picamera2")),
        raising=False,
    )

    assert camera_cli.main(["--fake", "--no-preview"]) == 0


def test_fake_rejects_picamera2_only_flags(camera_cli, capsys):
    """``--fake`` selects no real camera, so Picamera2-only flags select nothing
    and would otherwise be silently ignored; they must error instead, same as
    on the V4L2 backend (Minor 6).
    """
    with pytest.raises(SystemExit) as exc_info:
        camera_cli.main(["--fake", "--no-preview", "--colour-gains", "1.8,2.1"])
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--colour-gains" in error
    assert "--fake" in error


def test_device_without_camera_choice_selects_v4l2(camera_cli, monkeypatch):
    opened = []

    def open_v4l2(device):
        opened.append((device, _CameraDouble()))
        return opened[-1][1]

    monkeypatch.setattr(camera_cli, "V4L2Camera", open_v4l2)

    assert camera_cli.main(["--device", "/dev/video9", "--no-preview"]) == 0
    assert opened[0][0] == "/dev/video9"
    assert opened[0][1].closed


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--camera", "picamera2", "--device", "/dev/video9"], "--device"),
        (["--camera", "v4l2", "--tuning-file", "sensor.json"], "--tuning-file"),
        (["--camera", "v4l2", "--autofocus", "continuous"], "--autofocus"),
        (["--camera", "v4l2", "--af-range", "normal"], "--af-range"),
        (["--camera", "v4l2", "--ae-lock"], "--ae-lock"),
        (["--camera", "v4l2", "--awb-lock"], "--awb-lock"),
        (["--camera", "v4l2", "--colour-gains", "1.0,1.0"], "--colour-gains"),
        (["--camera", "v4l2", "--ae-constraint", "highlight"], "--ae-constraint"),
        (["--camera", "v4l2", "--ae-metering", "matrix"], "--ae-metering"),
    ],
)
def test_camera_specific_options_reject_conflicting_backend(args, message, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert message in error
    assert "cannot be used" in error


def test_no_dng_is_accepted_with_explicit_v4l2(camera_cli, monkeypatch):
    """``--no-dng`` is wired independently into ``CaptureSession`` and is not
    Picamera2-only, unlike ``--autofocus``/``--af-range``/the locks/``--colour-gains``.
    """
    monkeypatch.setattr(camera_cli, "V4L2Camera", lambda device: _CameraDouble())

    assert camera_cli.main(["--camera", "v4l2", "--no-preview", "--no-dng"]) == 0


@pytest.mark.parametrize(
    "flag_args",
    [
        ["--ae-lock"],
        ["--awb-lock"],
        ["--colour-gains", "1.8,2.1"],
        ["--autofocus", "auto"],
        ["--af-range", "macro"],
        ["--tuning-file", "sensor.json"],
        ["--ae-constraint", "highlight"],
        ["--ae-metering", "spot"],
    ],
)
def test_implicit_v4l2_via_device_rejects_picamera2_only_flags(camera_cli, flag_args):
    """No ``--camera`` given, but ``--device`` forces V4L2 implicitly.

    The guard must still catch Picamera2-only flags here, not just when
    ``--camera v4l2`` is spelled out explicitly.
    """
    with pytest.raises(SystemExit) as exc_info:
        camera_cli.main(["--device", "/dev/video9", "--no-preview", *flag_args])
    assert exc_info.value.code == 2


def test_implicit_v4l2_via_missing_picamera2_rejects_picamera2_only_flags(
    camera_cli, monkeypatch,
):
    """No ``--camera``/``--device`` given; Picamera2 import failure forces V4L2."""
    real_import = builtins.__import__

    def missing_picamera(name, *args, **kwargs):
        if name == "picamera2":
            raise ModuleNotFoundError("No module named 'picamera2'", name="picamera2")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "picamera2", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_picamera)

    with pytest.raises(SystemExit) as exc_info:
        camera_cli.main(["--no-preview", "--ae-lock"])
    assert exc_info.value.code == 2


def test_default_prefers_picamera2_when_module_imports(camera_cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", SimpleNamespace())
    opened = []

    def open_picamera(tuning_file, **kwargs):
        opened.append((tuning_file, _CameraDouble()))
        return opened[-1][1]

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(
        camera_cli,
        "V4L2Camera",
        lambda device: (_ for _ in ()).throw(AssertionError("wrong backend")),
    )

    assert camera_cli.main(["--no-preview"]) == 0
    assert opened[0][0] is None
    assert opened[0][1].closed


def test_default_falls_back_to_v4l2_when_picamera2_is_unavailable(
    camera_cli, monkeypatch,
):
    real_import = builtins.__import__
    opened = []
    import_attempts = 0

    def missing_picamera(name, *args, **kwargs):
        nonlocal import_attempts
        if name == "picamera2":
            import_attempts += 1
            raise ModuleNotFoundError("No module named 'picamera2'", name="picamera2")
        return real_import(name, *args, **kwargs)

    def open_v4l2(device):
        opened.append((device, _CameraDouble()))
        return opened[-1][1]

    monkeypatch.delitem(sys.modules, "picamera2", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_picamera)
    monkeypatch.setattr(camera_cli, "V4L2Camera", open_v4l2)

    assert camera_cli.main(["--no-preview"]) == 0
    assert import_attempts == 1
    assert opened[0][0] is None
    assert opened[0][1].closed


def test_control_flags_reach_the_picamera2_backend(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main([
        "--camera", "picamera2", "--no-preview",
        "--autofocus", "manual", "--af-range", "macro",
        "--ae-lock", "--awb-lock", "--colour-gains", "1.8,2.1",
    ]) == 0

    assert backend_kwargs["autofocus"] == "manual"
    assert backend_kwargs["af_range"] == "macro"
    assert backend_kwargs["ae_lock"] is True
    assert backend_kwargs["awb_lock"] is True
    assert backend_kwargs["colour_gains"] == (1.8, 2.1)
    assert backend_kwargs["save_dng"] is True


def test_ae_shaping_flags_reach_the_picamera2_backend(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main([
        "--camera", "picamera2", "--no-preview",
        "--ae-constraint", "highlight", "--ae-metering", "matrix",
    ]) == 0

    assert backend_kwargs["ae_constraint"] == "highlight"
    assert backend_kwargs["ae_metering"] == "matrix"
    assert "ev" not in backend_kwargs


def test_default_flags_leave_ae_shaping_at_libcamera_defaults(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main(["--camera", "picamera2", "--no-preview"]) == 0

    assert backend_kwargs["ae_constraint"] == "normal"
    assert backend_kwargs["ae_metering"] == "centre"
    assert "ev" not in backend_kwargs


@pytest.mark.parametrize("argv", [
    ["--fake", "--no-preview", "--ev", "1"],
    ["--camera", "picamera2", "--no-preview", "--ev", "1"],
    ["--camera", "picamera2", "--no-preview", "--ev=-0.5"],
])
def test_the_ev_flag_is_gone(camera_cli, monkeypatch, capsys, argv):
    """Exposure compensation was removed with the viewfinder's EV buttons."""
    monkeypatch.setattr(
        camera_cli, "Picamera2Camera",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened a camera")),
        raising=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        camera_cli.main(argv)
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_default_flags_leave_ae_and_awb_auto_with_continuous_af(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main(["--camera", "picamera2", "--no-preview"]) == 0

    assert backend_kwargs["autofocus"] == "continuous"
    assert backend_kwargs["af_range"] == "normal"
    assert backend_kwargs["ae_lock"] is False
    assert backend_kwargs["awb_lock"] is False
    assert backend_kwargs["colour_gains"] is None
    assert backend_kwargs["save_dng"] is True


def test_no_dng_flag_disables_dng_on_backend_and_session(camera_cli, monkeypatch):
    backend_kwargs = {}
    session_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    class RecordingSession:
        def __init__(self, camera, pipeline, out, **kwargs):
            session_kwargs.update(kwargs)
            self.camera = camera

        def capture(self, *, delay_s=0.0):
            raise AssertionError("capture should not be called in this test")

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(camera_cli, "CaptureSession", RecordingSession)

    assert camera_cli.main(["--camera", "picamera2", "--no-preview", "--no-dng"]) == 0

    assert backend_kwargs["save_dng"] is False
    assert session_kwargs["save_dng"] is False


@pytest.mark.parametrize("value", ["1.8", "1.8,2.1,3.0", "a,b", "", "0,0", "-1,2"])
def test_malformed_colour_gains_is_an_argparse_error(value, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--colour-gains", value])
    assert exc_info.value.code == 2
    assert "--colour-gains" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0,0", "-1,2", "1.8,0", "0,1.8", "-0.5,-0.5"])
def test_colour_gains_type_function_rejects_non_positive_values(value):
    """Minor 5, isolated from CLI backend-selection: ``_colour_gains`` itself
    must reject non-positive gains, independent of which backend a run would
    select (a syntactically valid ``0,0``/negative pair must not silently
    parse to a gains tuple that would be rejected only by accident elsewhere).
    """
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _colour_gains(value)


def test_colour_gains_parses_to_a_float_tuple(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main(
        ["--camera", "picamera2", "--no-preview", "--colour-gains", "1.8,2.1"]
    ) == 0
    assert backend_kwargs["colour_gains"] == (1.8, 2.1)


def test_capture_writes_original_and_dng_and_records_metadata(tmp_path):
    import numpy as np

    from pifilm.artifacts import Artifacts
    from pifilm.capture.app import CaptureSession
    from pifilm.capture.camera import Frame, StreamInfo
    from pifilm.pipeline import Pipeline

    class MetaCamera:
        stream_info = StreamInfo(4608, 2592, 14.35, "RGB888", False,
                                 sensor_mode="4608x2592 SBGGR10_CSI2P", bit_depth=10,
                                 tuning_file="imx708_wide.json")
        def read(self):
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            metadata = {"ExposureTime": 9995, "AnalogueGain": 2.0, "Lux": 120.0}
            return Frame(rgb=rgb, jpeg=None, source="picamera2",
                         metadata=metadata,
                         dng=b"II*\x00fake-dng-bytes")
        def close(self): ...

    sess = CaptureSession(MetaCamera(), Pipeline(Artifacts.default()), tmp_path,
                          seed_rng=np.random.default_rng(0))
    result = sess.capture()
    day = result.pifilm.parent
    assert result.original.name.endswith("_original.jpg")
    dng = day / (result.original.name.replace("_original.jpg", ".dng"))
    assert dng.read_bytes() == b"II*\x00fake-dng-bytes"
    expected_metadata = {"ExposureTime": 9995, "AnalogueGain": 2.0, "Lux": 120.0}
    assert result.record["camera_metadata"] == expected_metadata
    assert result.record["dng"] == dng.name


def test_capture_skips_dng_when_disabled(tmp_path):
    import numpy as np

    from pifilm.artifacts import Artifacts
    from pifilm.capture.app import CaptureSession
    from pifilm.capture.camera import Frame, StreamInfo
    from pifilm.pipeline import Pipeline

    class C:
        stream_info = StreamInfo(8, 8, 0.0, "RGB888", False)
        def read(self):
            return Frame(np.zeros((8, 8, 3), np.uint8), None, "picamera2", dng=b"raw")
        def close(self): ...

    sess = CaptureSession(C(), Pipeline(Artifacts.default()), tmp_path,
                          seed_rng=np.random.default_rng(0), save_dng=False)
    result = sess.capture()
    assert not list(result.pifilm.parent.glob("*.dng"))
    assert "dng" not in result.record
    assert result.original.name.endswith("_original.jpg")


def test_v4l2_style_frame_without_metadata_is_unchanged(tmp_path):
    import numpy as np

    from pifilm.artifacts import Artifacts
    from pifilm.capture.app import CaptureSession
    from pifilm.capture.camera import Frame, StreamInfo
    from pifilm.pipeline import Pipeline

    class Usb:
        stream_info = StreamInfo(1920, 1080, 30.0, "MJPG", True)
        def read(self):
            return Frame(np.zeros((8, 8, 3), np.uint8), b"\xff\xd8jpg\xff\xd9",
                         "raw-mjpeg")
        def close(self): ...

    result = CaptureSession(Usb(), Pipeline(Artifacts.default()), tmp_path,
                            seed_rng=np.random.default_rng(0)).capture()
    assert result.original.name.endswith("_original.jpg")   # jpeg present -> original
    assert "camera_metadata" not in result.record
    assert "dng" not in result.record


def test_display_warns_and_continues_on_v4l2(camera_cli, monkeypatch, capsys):
    """The shipped service unit passes --display; on a V4L2 deployment a
    parser.error here would exit 2 and systemd would restart it forever."""
    monkeypatch.setattr(camera_cli, "V4L2Camera", lambda device: _CameraDouble())
    assert camera_cli.main(["--camera", "v4l2", "--display", "waveshare28", "--no-preview"]) == 0
    assert (
        "warning: display unavailable (the V4L2 backend has no preview mode); "
        "continuing without it"
    ) in capsys.readouterr().err


def test_display_on_v4l2_still_serves_the_stick(camera_cli, monkeypatch):
    """The service's own argument list: --display plus --remote-listen on a Pi
    whose camera is USB must keep the remote API, not die."""
    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "token")
    seen = {"viewfinder": 0}

    class ServerDouble:
        def __init__(self, controller, token, listen, power=None):
            pass

        def start(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(camera_cli, "V4L2Camera", lambda device: _CameraDouble())
    monkeypatch.setattr(camera_cli, "RemoteCaptureServer", ServerDouble)
    monkeypatch.setattr(
        camera_cli, "_run_viewfinder",
        lambda *a, **k: seen.__setitem__("viewfinder", seen["viewfinder"] + 1),
    )
    assert camera_cli.main([
        "--camera", "v4l2", "--display", "waveshare28", "--no-preview",
        "--remote-listen", "127.0.0.1:8765",
    ]) == 0
    assert seen["viewfinder"] == 0


def test_display_is_dropped_before_the_panel_is_opened_on_an_auto_v4l2_backend(
    camera_cli, monkeypatch, capsys,
):
    """The backend can resolve to V4L2 without --camera: with no Picamera2
    installed. The panel must not be opened only to be abandoned, and the
    viewfinder must not then run against a V4L2 camera."""
    opened = []
    monkeypatch.setattr(camera_cli, "_picamera2_available", lambda: False)
    monkeypatch.setattr(camera_cli, "V4L2Camera", lambda device: _CameraDouble())
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: opened.append(1))
    monkeypatch.setattr(camera_cli, "_run_viewfinder", lambda *a, **k: 1)
    assert camera_cli.main(["--display", "waveshare28", "--no-preview"]) == 0
    assert opened == []
    assert "the V4L2 backend has no preview mode" in capsys.readouterr().err


def test_display_requests_preview_mode_from_picamera2(camera_cli, monkeypatch):
    """The camera must be opened in preview mode only once the panel is open:
    preview mode is a continuous binned stream plus a mode switch per shot, and
    paying for it with no display to feed is pure cost."""
    opened = {}
    pair = (SimpleNamespace(close=lambda: None), SimpleNamespace(close=lambda: None))

    def open_picamera(tuning_file, **kwargs):
        opened.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: pair)
    monkeypatch.setattr(camera_cli, "_run_viewfinder", lambda *a, **k: 0)
    argv = ["--camera", "picamera2", "--display", "waveshare28", "--no-preview"]
    assert camera_cli.main(argv) == 0
    assert opened["preview"] is True


def test_display_fault_opens_the_camera_without_preview_mode(camera_cli, monkeypatch):
    from pifilm.display import DisplayError

    opened = {}

    def open_picamera(tuning_file, **kwargs):
        opened.update(kwargs)
        return _CameraDouble()

    def failing(args, out):
        raise DisplayError("no spi")

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(camera_cli, "_open_display", failing)
    argv = ["--camera", "picamera2", "--display", "waveshare28", "--no-preview"]
    assert camera_cli.main(argv) == 0
    assert opened["preview"] is None


def test_display_fault_falls_back_to_the_terminal(camera_cli, monkeypatch, capsys):
    from pifilm.display import DisplayError

    def failing(args, out):
        raise DisplayError("no spi")

    monkeypatch.setattr(camera_cli, "_open_display", failing)
    assert camera_cli.main(["--fake", "--display", "waveshare28", "--no-preview"]) == 0
    assert "no spi" in capsys.readouterr().err


def test_fake_display_writes_a_png_and_exits_on_stop(tmp_path, monkeypatch):
    from pifilm.capture import app

    monkeypatch.setattr(app.Artifacts, "resolve", lambda _: Artifacts.default())

    out = tmp_path / "shots"

    class StopSoon:
        """Stops once a frame has been written. The loop's first turn has no
        frame yet (the frame thread is still reading it), and the touch and
        frame threads ask this event too, so a fixed number of calls would not
        say how many turns the loop got. The count only bounds a failure."""

        def __init__(self):
            self.n = 0

        def is_set(self):
            self.n += 1
            return (out / "viewfinder-last.png").exists() or self.n > 500

    # Narrow the patch to app's own name: setattr on the real threading module
    # would also replace the Event that Thread.start() uses internally.
    monkeypatch.setattr(app, "threading", SimpleNamespace(Event=StopSoon))
    assert app.main(["--fake", "--display", "fake", "--no-preview", "--out", str(out)]) == 0
    assert (out / "viewfinder-last.png").exists()


def test_remote_listener_and_display_share_one_controller(camera_cli, monkeypatch, tmp_path):
    """The shipped service runs --remote-listen with --display: the viewfinder must
    take over the foreground (not the sleep loop) and the Stick must submit jobs
    through the very controller the panel is watching."""
    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "token")
    seen = {"viewfinder_calls": 0}

    class ServerDouble:
        def __init__(self, controller, token, listen, power=None):
            seen["server_controller"] = controller

        def start(self):
            return None

        def close(self):
            return None

    def viewfinder_double(camera, controller, display_pair, power, args, **kwargs):
        seen["viewfinder_calls"] += 1
        seen["viewfinder_controller"] = controller
        return 0

    def no_sleep(_seconds):
        raise KeyboardInterrupt  # the sleep loop must never be reached

    monkeypatch.setattr(camera_cli, "RemoteCaptureServer", ServerDouble)
    monkeypatch.setattr(camera_cli, "_run_viewfinder", viewfinder_double)
    monkeypatch.setattr(
        camera_cli, "time",
        SimpleNamespace(sleep=no_sleep, perf_counter=time.perf_counter),
    )

    assert camera_cli.main([
        "--fake", "--display", "fake", "--no-preview",
        "--remote-listen", "127.0.0.1:8765", "--out", str(tmp_path / "shots"),
    ]) == 0
    assert seen["viewfinder_calls"] == 1
    assert seen["viewfinder_controller"] is seen["server_controller"]


def test_display_breaker_keeps_the_remote_api_alive(camera_cli, monkeypatch, tmp_path, capsys):
    """Five consecutive SPI failures must cost the screen, not the Stick: the
    process used to return 1, systemd restarted it, and the Stick got a few
    seconds of service per cycle."""
    from pifilm.display import DisplayError

    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "token")
    slept = []

    class ServerDouble:
        def __init__(self, controller, token, listen, power=None):
            pass

        def start(self):
            return None

        def close(self):
            return None

    class DeadDisplay:
        closed = False

        def show(self, image):
            raise DisplayError("spi write failed")

        def close(self):
            self.closed = True

    class Touch:
        closed = False

        def read(self):
            return []

        def close(self):
            self.closed = True

    display, touch = DeadDisplay(), Touch()

    def no_sleep(seconds):
        slept.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(camera_cli, "RemoteCaptureServer", ServerDouble)
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: (display, touch))
    monkeypatch.setattr(
        camera_cli, "time",
        SimpleNamespace(sleep=no_sleep, perf_counter=time.perf_counter),
    )

    assert camera_cli.main([
        "--fake", "--display", "waveshare28", "--no-preview",
        "--remote-listen", "127.0.0.1:8765", "--out", str(tmp_path / "shots"),
    ]) == 0
    assert slept == [1]                     # the remote sleep loop was entered
    assert display.closed and touch.closed  # and the panel was released first
    assert "display failed repeatedly" in capsys.readouterr().err


def _viewfinder_kwargs(monkeypatch, tmp_path, extra):
    """Run main with --display fake and return the kwargs ViewfinderLoop got."""
    from pifilm.capture import app

    monkeypatch.setattr(app.Artifacts, "resolve", lambda _: Artifacts.default())
    seen = {}

    class LoopDouble:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def run(self, stop):
            return None

    monkeypatch.setattr(app, "ViewfinderLoop", LoopDouble)
    out = tmp_path / "shots"
    assert app.main(["--fake", "--display", "fake", "--no-preview", "--out", str(out), *extra]) == 0
    return seen


def test_idle_dimming_defaults_to_one_and_five_minutes(tmp_path, monkeypatch):
    kwargs = _viewfinder_kwargs(monkeypatch, tmp_path, [])
    assert kwargs["dim_after"] == 60.0 and kwargs["off_after"] == 300.0


def test_idle_dimming_delays_come_from_the_command_line(tmp_path, monkeypatch):
    kwargs = _viewfinder_kwargs(
        monkeypatch, tmp_path, ["--display-dim-after", "30", "--display-off-after", "0"],
    )
    assert kwargs["dim_after"] == 30.0 and kwargs["off_after"] == 0.0


def test_negative_idle_delays_are_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _viewfinder_kwargs(monkeypatch, tmp_path, ["--display-off-after", "-5"])


def test_record_carries_the_shutter_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    first = session.capture()
    session.camera.set_shutter(4000)
    second = session.capture()
    assert first.record["shutter_us"] is None
    assert second.record["shutter_us"] == 4000


# -- the DSI panel ----------------------------------------------------------------


def test_open_display_opens_the_dsi_panel_and_its_touch(monkeypatch):
    from pifilm.capture import app
    from pifilm.display import evtouch, kms

    calls = {}
    display = SimpleNamespace(width=640, height=480, close=lambda: None)

    def open_display(rotate):
        calls["display"] = rotate
        return display

    def open_touch(size, rotate):
        calls["touch"] = (size, rotate)
        return "touch"

    monkeypatch.setattr(kms, "open_waveshare35dsi", open_display)
    monkeypatch.setattr(evtouch, "open_goodix_touch", open_touch)
    args = argparse.Namespace(display="waveshare35dsi", display_rotate=180)
    assert app._open_display(args, Path("/unused")) == (display, "touch")
    assert calls == {"display": 180, "touch": ((640, 480), 180)}


def test_open_display_closes_the_dsi_panel_when_touch_fails(monkeypatch):
    from pifilm.capture import app
    from pifilm.display import DisplayError, evtouch, kms

    closed = []
    display = SimpleNamespace(width=640, height=480, close=lambda: closed.append(1))

    def no_touch(size, rotate):
        raise DisplayError("touch device not found")

    monkeypatch.setattr(kms, "open_waveshare35dsi", lambda rotate: display)
    monkeypatch.setattr(evtouch, "open_goodix_touch", no_touch)
    args = argparse.Namespace(display="waveshare35dsi", display_rotate=0)
    with pytest.raises(DisplayError, match="touch device not found"):
        app._open_display(args, Path("/unused"))
    assert closed == [1]


def test_dsi_display_warns_and_continues_on_v4l2(camera_cli, monkeypatch, capsys):
    monkeypatch.setattr(camera_cli, "V4L2Camera", lambda device: _CameraDouble())
    assert camera_cli.main(
        ["--camera", "v4l2", "--display", "waveshare35dsi", "--no-preview"]) == 0
    assert "warning: display unavailable" in capsys.readouterr().err


# -- the GPIO shutter button --------------------------------------------------------


class _ButtonDouble:
    def __init__(self):
        self.callback = None
        self.closed = False

    def on_press(self, callback):
        self.callback = callback

    def close(self):
        self.closed = True


class _ControllerDouble:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []
        self.delays = []

    def submit(self, request_id, *, delay_s=0.0):
        self.requests.append(request_id)
        self.delays.append(delay_s)
        return self.answer


def test_a_press_submits_one_new_request(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="queued", error_code=None))
    app._button_capture(controller)
    app._button_capture(controller)
    assert len(controller.requests) == 2
    assert controller.requests[0] != controller.requests[1]
    assert "button: capture requested" in capsys.readouterr().out


def test_a_press_asks_for_the_shutter_delay_read_at_the_press():
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="queued", error_code=None))
    delay = iter([0.5, 0.2])
    app._button_capture(controller)
    app._button_capture(controller, lambda: next(delay))
    app._button_capture(controller, lambda: next(delay))
    assert controller.delays == [0.0, 0.5, 0.2]


def test_a_press_while_busy_is_ignored_and_logged(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="failed", error_code="busy"))
    app._button_capture(controller)
    assert "button: press ignored (busy)" in capsys.readouterr().out


def test_a_press_after_close_is_ignored(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="failed", error_code="closed"))
    app._button_capture(controller)   # must not raise
    assert "button: press ignored (closed)" in capsys.readouterr().out


def test_main_opens_the_button_attaches_it_and_closes_it(camera_cli, monkeypatch, capsys):
    button = _ButtonDouble()
    opened = []

    def open_button(pin):
        opened.append(pin)
        return button

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    assert opened == [21]
    assert callable(button.callback)
    assert button.closed
    assert "Shutter button on BCM 21." in capsys.readouterr().out


def test_main_warns_and_continues_when_the_button_cannot_open(camera_cli, monkeypatch, capsys):
    from pifilm.capture.button import ButtonError

    def open_button(pin):
        raise ButtonError("cannot claim BCM 21: GPIO busy")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    assert (
        "warning: shutter button unavailable (cannot claim BCM 21: GPIO busy); "
        "continuing without it"
    ) in capsys.readouterr().err


def test_main_without_the_flag_never_opens_a_button(camera_cli, monkeypatch):
    def open_button(pin):
        raise AssertionError("the button must not be opened without --shutter-gpio")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview"]) == 0


@pytest.mark.parametrize("argv", [
    ["--fake", "--no-preview", "--shutter-gpio", "6", "--ups", "x728"],
    ["--fake", "--no-preview", "--shutter-gpio", "17", "--display", "waveshare28"],
    ["--fake", "--no-preview", "--shutter-gpio", "3"],
    ["--fake", "--no-preview", "--shutter-gpio", "28"],
    ["--fake", "--no-preview", "--shutter-gpio", "twenty"],
])
def test_shutter_gpio_refuses_reserved_pins(camera_cli, monkeypatch, argv):
    def open_button(pin):
        raise AssertionError("an argument error must come before any hardware is touched")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    monkeypatch.setattr(camera_cli, "build_ups", lambda name: None)
    with pytest.raises(SystemExit) as excinfo:
        camera_cli.main(argv)
    assert excinfo.value.code == 2


def test_a_reserved_pin_is_allowed_when_its_hardware_is_not_selected(camera_cli, monkeypatch):
    monkeypatch.setattr(camera_cli, "open_shutter_button", lambda pin: _ButtonDouble())
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "17"]) == 0


def test_main_still_shuts_down_when_the_button_does_not_close(camera_cli, monkeypatch, capsys):
    class StuckButton(_ButtonDouble):
        def close(self):
            raise RuntimeError("line busy")

    closed = []
    real_close = camera_cli.CaptureController.close

    def recording_close(self):
        closed.append(self)
        return real_close(self)

    monkeypatch.setattr(camera_cli.CaptureController, "close", recording_close)
    monkeypatch.setattr(camera_cli, "open_shutter_button", lambda pin: StuckButton())
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    assert len(closed) == 1
    assert (
        "warning: shutter button did not close cleanly (line busy)"
    ) in capsys.readouterr().err


def test_stalled_camera_exits_nonzero_even_with_the_remote_api(
    camera_cli, tmp_path, monkeypatch, capsys,
):
    """A dead screen costs the screen only, but a camera that stopped delivering
    frames leaves nothing to serve: the process must exit so systemd restarts it,
    and a clean-up that hangs on the same camera must not keep it alive."""
    from pifilm.capture.errors import CameraStalled

    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "token")
    armed = []

    class ServerDouble:
        def __init__(self, controller, token, listen, power=None):
            pass

        def start(self):
            return None

        def close(self):
            return None

    class Panel:
        closed = False

        def show(self, image):
            return None

        def read(self):
            return []

        def close(self):
            self.closed = True

    class StalledLoop:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, stop):
            raise CameraStalled("no preview frame within 5.0 s")

    display, touch = Panel(), Panel()
    monkeypatch.setattr(camera_cli, "RemoteCaptureServer", ServerDouble)
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: (display, touch))
    monkeypatch.setattr(camera_cli, "ViewfinderLoop", StalledLoop)
    monkeypatch.setattr(
        camera_cli, "_arm_exit_deadline", lambda seconds, code: armed.append((seconds, code)),
    )

    assert camera_cli.main([
        "--fake", "--display", "waveshare28", "--no-preview",
        "--remote-listen", "127.0.0.1:8765", "--out", str(tmp_path / "shots"),
    ]) == 1
    assert armed == [(camera_cli.STALL_EXIT_DEADLINE, 1)]
    assert display.closed and touch.closed
    assert "camera stopped delivering frames" in capsys.readouterr().err


def test_record_carries_the_white_balance_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    first = session.capture()
    session.camera.set_colour_temperature(6500)
    second = session.capture()
    assert first.record["colour_temperature_k"] is None
    assert second.record["colour_temperature_k"] == 6500


def test_double_exposure_records_carry_the_white_balance_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.camera.set_colour_temperature(5600)
    first = session.capture()
    second = session.capture()
    assert first.record["colour_temperature_k"] == 5600
    assert second.record["colour_temperature_k"] == 5600


def test_record_carries_the_iso_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    first = session.capture()
    session.camera.set_iso(400)
    second = session.capture()
    assert first.record["iso"] is None
    assert second.record["iso"] == 400


def test_double_exposure_records_carry_the_iso_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.camera.set_iso(800)
    first = session.capture()
    second = session.capture()
    assert first.record["iso"] == 800
    assert second.record["iso"] == 800


@pytest.mark.parametrize("argv", [
    ["--white-balance", "1000"], ["--white-balance", "12000"], ["--white-balance", "warm"],
    ["--white-balance", "6500", "--awb-lock"],
    ["--white-balance", "6500", "--colour-gains", "2.0,1.5"],
    ["--white-balance", "6500", "--camera", "v4l2"],
])
def test_bad_white_balance_arguments_are_refused(camera_cli, tmp_path, argv):
    with pytest.raises(SystemExit) as caught:
        camera_cli.main([*argv, "--no-preview", "--out", str(tmp_path / "shots")])
    assert caught.value.code == 2


def test_record_breaks_the_shot_down_by_stage(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    record = session.capture().record
    stages = record["stage_ms"]
    assert set(stages) == {
        "capture", "normalize", "lut", "grain", "save_original", "save_graded",
    }
    assert all(isinstance(v, float) and v >= 0.0 for v in stages.values())
    # The parts cannot exceed the whole (rounding to 0.1 ms aside), and the
    # existing totals stay in the record.
    assert sum(stages.values()) <= record["shutter_to_saved_ms"] + 1.0
    assert record["pipeline_ms"] >= 0.0


def test_camera_stage_timings_are_carried_into_the_record(tmp_path, pipeline):
    camera = FakeCamera([synthetic_frame(90, 160)])
    real_read = camera.read

    def read(**kwargs):
        frame = real_read(**kwargs)
        frame.timings = {"request": 400.0, "array": 30.0, "dng": 900.0}
        return frame

    camera.read = read
    stages = _session(tmp_path, pipeline, camera=camera).capture().record["stage_ms"]
    assert stages["capture_request"] == 400.0
    assert stages["capture_array"] == 30.0
    assert stages["capture_dng"] == 900.0


def test_a_saved_dng_is_timed(tmp_path, pipeline):
    camera = FakeCamera([synthetic_frame(90, 160)])
    real_read = camera.read

    def read(**kwargs):
        frame = real_read(**kwargs)
        frame.dng = b"raw"
        return frame

    camera.read = read
    record = _session(tmp_path, pipeline, camera=camera).capture().record
    assert record["stage_ms"]["save_dng"] >= 0.0


def test_double_exposure_records_are_broken_down_too(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    first = session.capture().record["stage_ms"]
    second = session.capture().record["stage_ms"]
    assert set(first) == {"capture", "save_original", "card"}
    assert set(second) == {
        "capture", "save_original", "composite", "normalize", "lut", "grain",
        "save_graded",
    }


def test_session_warms_the_pipeline_for_the_cameras_still_size(tmp_path):
    """The tile grain's tile is built at start-up, not during the first photograph."""
    import threading

    warmed = threading.Event()
    shapes = []

    class _Pipeline:
        def warm(self, shape):
            shapes.append(shape)
            warmed.set()

    camera = FakeCamera()
    camera.still_size = (4056, 3040)
    CaptureSession(camera, _Pipeline(), tmp_path)
    assert warmed.wait(5.0)
    assert shapes == [(3040, 4056)]


def test_session_without_a_still_size_does_not_warm(tmp_path):
    class _Pipeline:
        def warm(self, shape):
            raise AssertionError("no size to warm for")

    CaptureSession(FakeCamera(), _Pipeline(), tmp_path)


# --- Grading behind the shutter -------------------------------------------------

FIXED = datetime(2026, 9, 3, 21, 5, 7)
NEW_KEYS = {"shutter_to_ready_ms", "grade"}


def _earlier_keys(record):
    """The record's keys without what grading behind the shutter added, and without
    ``pipeline_backend``, which sits after ``stage_ms`` in an inline record and
    after ``grade`` in a background one."""
    return [k for k in record if k not in NEW_KEYS and k != "pipeline_backend"]


class _HeldPipeline:
    """A pipeline whose full grades each wait for one permit; quick grades pass.

    The quick grade is the only caller that asks for no grain, so that flag tells
    the two apart. Nothing waits longer than 5 s, so a broken test fails instead
    of hanging.
    """

    def __init__(self, inner):
        self._inner = inner
        self.entered = threading.Semaphore(0)  # one release per full grade started
        self._permits = threading.Semaphore(0)

    def process(self, rgb, *, grain=True, **kwargs):
        if grain:
            self.entered.release()
            assert self._permits.acquire(timeout=5), "the grade was never released"
        return self._inner.process(rgb, grain=grain, **kwargs)

    def release(self, grades=1):
        for _ in range(grades):
            self._permits.release()

    def wait_entered(self):
        assert self.entered.acquire(timeout=5), "no grade started"


@pytest.fixture
def background(tmp_path, pipeline):
    """Build background sessions; every one is closed, and every hold released, at the end."""
    made, held = [], []

    def make(camera=None, *, hold=False, root=None, state=None, **kwargs):
        pipe = pipeline
        if hold:
            pipe = _HeldPipeline(pipeline)
            held.append(pipe)
        log = kwargs.pop("log", [])
        session = CaptureSession(
            camera or FakeCamera([synthetic_frame(90, 160)]), pipe,
            root or tmp_path / "shots",
            now=kwargs.pop("now", None), seed_rng=np.random.default_rng(0),
            background=True, state_dir=state or tmp_path / "state", log=log.append, **kwargs,
        )
        session.test_log = log
        made.append(session)
        return session

    yield make
    for pipe in held:
        pipe.release(100)
    for session in made:
        assert session.close_grades(timeout=5), "a grade worker outlived its test"


def _lines(root):
    out = []
    for f in sorted(Path(root).glob("*/captures.jsonl")):
        out += [json.loads(line) for line in f.read_text().splitlines()]
    return out


def _settle(session):
    """Wait, at most 5 s, for every accepted grade to finish (the session stays open)."""
    deadline = time.monotonic() + 5
    while session.pending_grades:
        assert time.monotonic() < deadline, "grades did not finish"
        threading.Event().wait(0.005)


def test_inline_session_never_touches_a_state_directory(tmp_path, pipeline, monkeypatch):
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    session = _session(tmp_path, pipeline)
    result = session.capture()
    assert session.pending_grades == 0
    assert session.close_grades() is True
    assert result.graded == result.pifilm
    assert not home.exists()
    assert not any(t.name == "pifilm-grade" for t in threading.enumerate())


def test_background_graded_file_is_byte_identical_to_inline(tmp_path, pipeline, background):
    inline = _session(tmp_path / "inline", pipeline, now=lambda: FIXED).capture()
    session = background(now=lambda: FIXED)
    result = session.capture()
    assert session.close_grades(timeout=5)
    assert result.graded.name == inline.pifilm.name == "210507_graded.jpg"
    assert result.graded.read_bytes() == inline.pifilm.read_bytes()
    assert result.original.read_bytes() == inline.original.read_bytes()
    assert _lines(session.out_root)[0]["grain_seed"] == inline.record["grain_seed"]


def test_capture_returns_before_the_grade_finishes(tmp_path, background):
    session = background(hold=True)
    result = session.capture()
    session.pipeline.wait_entered()
    quick = load_rgb(result.pifilm)[0]
    assert quick.shape == (90, 160, 3)
    assert session.out_root.resolve() not in result.pifilm.resolve().parents
    assert result.pifilm.name.startswith("preview-")
    assert result.graded.parent.parent == session.out_root
    assert not result.graded.exists()
    assert session.pending_grades == 1
    assert _lines(session.out_root) == []
    session.pipeline.release()
    assert session.close_grades(timeout=5)
    assert result.graded.exists()
    assert session.pending_grades == 0
    assert session.close_grades() is True  # a second close is harmless


def test_background_record_is_the_inline_record_plus_the_new_fields(
    tmp_path, pipeline, background,
):
    inline = _session(tmp_path / "inline", pipeline, now=lambda: FIXED).capture().record
    assert not NEW_KEYS & set(inline)
    assert not {"preview", "queued"} & set(inline["stage_ms"])

    # Held until capture() has returned, so "ready" is always measured before "saved".
    session = background(hold=True, now=lambda: FIXED)
    result = session.capture()
    session.pipeline.release()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    # Today's fields in today's order, the new ones after them.
    assert _earlier_keys(record) == _earlier_keys(inline)
    assert list(record)[-3:] == ["shutter_to_ready_ms", "grade", "pipeline_backend"]
    assert record["grade"] == "background"
    assert "recovered" not in record
    assert {"capture", "save_original", "preview", "queued", "normalize", "lut", "grain",
            "save_graded"} <= set(record["stage_ms"])
    assert record["shutter_to_saved_ms"] >= record["shutter_to_ready_ms"] > 0
    for key in ("timestamp", "original", "pifilm", "wb_gains", "lut_sha1", "normalize_sha1",
                "ev_comp", "grain_seed", "params_version", "package_version", "width"):
        assert record[key] == inline[key], key
    # What the capture job carries: nothing from the grade, and its own copy of the stages.
    assert result.record["shutter_to_ready_ms"] == record["shutter_to_ready_ms"]
    assert "wb_gains" not in result.record and "shutter_to_saved_ms" not in result.record
    assert "preview" in result.record["stage_ms"]
    assert "save_graded" not in result.record["stage_ms"]
    assert result.record["pifilm"] == result.graded.name


def test_background_record_keeps_camera_metadata_and_dng_last(tmp_path, pipeline, background):
    class MetaCamera(FakeCamera):
        def read(self, full=True):
            frame = super().read()
            return Frame(frame.rgb, None, "picamera2", dng=b"raw", metadata={"Lux": 1.0})

    def make_camera():
        return MetaCamera([synthetic_frame(90, 160)])

    inline_session = CaptureSession(make_camera(), pipeline, tmp_path / "inline",
                                    now=lambda: FIXED, seed_rng=np.random.default_rng(0))
    inline = inline_session.capture().record
    session = background(make_camera(), now=lambda: FIXED)
    session.capture()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    assert _earlier_keys(record) == _earlier_keys(inline)
    assert record["dng"] == "210507.dng" and record["camera_metadata"] == {"Lux": 1.0}


def test_records_are_appended_in_shot_order(background):
    session = background(now=lambda: FIXED)
    results = [session.capture() for _ in range(3)]
    assert session.close_grades(timeout=5)
    names = ["210507_graded.jpg", "210507-2_graded.jpg", "210507-3_graded.jpg"]
    assert [r.graded.name for r in results] == names
    assert [line["pifilm"] for line in _lines(session.out_root)] == names
    assert all(r.graded.exists() for r in results)


def test_pending_entry_lives_exactly_as_long_as_the_grade(tmp_path, background):
    session = background(hold=True, now=lambda: FIXED)
    result = session.capture()
    session.pipeline.wait_entered()
    ((path, entry),) = read_entries(tmp_path / "state")
    assert path.name == "2026-09-03_210507.json"
    assert entry["out_root"] == str(session.out_root.absolute())
    assert entry["day"] == "2026-09-03" and entry["graded"] == "210507_graded.jpg"
    assert entry["originals"] == ["2026-09-03/210507_ungraded.jpg"]
    assert entry["seed"] == result.record["grain_seed"]
    assert entry["ev"] == 0.0 and entry["double"] is None and entry["attempts"] == 0
    assert entry["record"]["pifilm"] == "210507_graded.jpg"
    session.pipeline.release()
    assert session.close_grades(timeout=5)
    assert read_entries(tmp_path / "state") == []
    assert list((tmp_path / "state").iterdir()) == []


def _interrupted_shot(tmp_path, background, *, double=False):
    """Take a shot, keep a copy of its pending entry, and let the session finish.

    Returns (result, entry path, entry text): putting the text back is the state a
    power cut during the grade leaves behind.
    """
    frames = [synthetic_frame(90, 160), synthetic_frame(90, 160)[:, ::-1].copy()]
    session = background(FakeCamera(frames), hold=True, now=lambda: FIXED)
    if double:
        session.set_double_exposure(True)
        session.capture()
    result = session.capture()
    session.pipeline.wait_entered()
    ((path, _),) = read_entries(tmp_path / "state")
    text = path.read_text()
    session.pipeline.release()
    assert session.close_grades(timeout=5)
    assert not path.exists()
    return result, path, text


def test_an_unfinished_grade_is_finished_at_the_next_start(tmp_path, background):
    result, path, text = _interrupted_shot(tmp_path, background)
    first = result.graded.read_bytes()
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    again = background()
    assert again.test_log == ["recovering 1 unfinished grade(s)"]
    _settle(again)
    assert again.close_grades(timeout=5)
    assert not path.exists()
    (record,) = _lines(again.out_root)
    assert record["pifilm"] == result.graded.name
    assert record["recovered"] is True and record["grade"] == "background"
    assert "shutter_to_saved_ms" not in record and "shutter_to_ready_ms" not in record
    assert record["stage_ms"]["queued"] == 0.0 and "pipeline_ms" in record
    assert record["grain_seed"] == result.record["grain_seed"]
    # From the saved JPEG and the stored seed: the same picture within JPEG rounding.
    diff = np.abs(load_rgb(result.graded)[0].astype(int)
                  - np.asarray(Image.open(io.BytesIO(first)).convert("RGB")).astype(int))
    assert diff.mean() < 3.0


def test_recovery_does_not_record_a_finished_grade_twice(tmp_path, background):
    """The record was written and the power went before the entry was removed."""
    result, path, text = _interrupted_shot(tmp_path, background)
    before = _lines(result.graded.parents[1])
    result.graded.unlink()  # so a second grade would be visible as a new file
    path.write_text(text)

    again = background()
    _settle(again)
    assert again.close_grades(timeout=5)
    assert not path.exists()
    assert _lines(again.out_root) == before and len(before) == 1
    assert not result.graded.exists()


def test_recovery_leaves_other_folders_alone_and_survives_a_missing_original(
    tmp_path, background,
):
    result, path, text = _interrupted_shot(tmp_path, background)
    result.original.unlink()
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)
    foreign = write_entry(tmp_path / "state", "elsewhere", {
        **json.loads(text), "out_root": str(tmp_path / "another-folder"),
    })

    again = background(now=lambda: FIXED)
    assert again.test_log[0] == "recovering 1 unfinished grade(s)"
    shot = again.capture()  # start-up was not stopped, and neither is the next shot
    assert again.close_grades(timeout=5)
    assert shot.graded.exists()
    assert [line["pifilm"] for line in _lines(again.out_root)] == [shot.graded.name]
    assert sum(line.startswith(f"grade failed ({path.name}): ") for line in again.test_log) == 1
    assert json.loads(path.read_text())["attempts"] == 1
    assert json.loads(foreign.read_text())["attempts"] == 0
    assert {p.name for p in (tmp_path / "state").iterdir()} == {path.name, foreign.name}


def test_a_damaged_entry_fails_on_the_worker_not_at_start_up(tmp_path, background):
    root = tmp_path / "shots"
    for name, payload in (
        ("missing-keys", {}),
        ("escaping", {"day": "..", "graded": "x.jpg", "originals": ["../../x.jpg"], "seed": 1,
                      "ev": 0.0, "double": None, "record": {}}),
    ):
        write_entry(tmp_path / "state", name, {"out_root": str(root.absolute()), **payload})
    session = background()
    _settle(session)  # the two damaged entries are tried at start-up
    shot = session.capture()
    assert session.close_grades(timeout=5)
    assert shot.graded.exists()
    assert [line for line in session.test_log if line.startswith("grade failed")] == [
        "grade failed (escaping.json): pending entry has a bad name: '..'",
        "grade failed (missing-keys.json): 'day'",
    ]
    assert len(_lines(root)) == 1
    assert not list(tmp_path.glob("x.jpg")) and not list(tmp_path.parent.glob("x.jpg"))


def _double_frames():
    return [np.full((90, 160, 3), 10, np.uint8), np.full((90, 160, 3), 230, np.uint8)]


def test_background_double_exposure_matches_inline(tmp_path, pipeline, background):
    inline_session = _session(tmp_path / "inline", pipeline, FakeCamera(_double_frames()),
                              now=lambda: FIXED)
    inline_session.set_double_exposure(True)
    inline_session.capture()
    inline = inline_session.capture()

    session = background(FakeCamera(_double_frames()), now=lambda: FIXED)
    session.set_double_exposure(True)
    first = session.capture()
    assert first.exposure == (1, 2) and first.graded is None
    assert session.pending_grades == 0
    assert read_entries(tmp_path / "state") == []
    assert not list(session.out_root.glob("*/*_graded.jpg"))
    assert session.double_state == (True, 1)

    second = session.capture()
    assert second.exposure == (2, 2)
    assert session.double_state == (True, 0)
    assert load_rgb(second.pifilm)[0].shape == (90, 160, 3)
    assert session.out_root.resolve() not in second.pifilm.resolve().parents
    assert session.close_grades(timeout=5)
    assert second.graded.name == inline.pifilm.name == "210507-2_double_graded.jpg"
    assert second.graded.read_bytes() == inline.pifilm.read_bytes()
    one, two = _lines(session.out_root)
    assert one["double"] == {"index": 1, "of": 2} and "grade" not in one
    assert two["double"] == inline.record["double"] and two["double"]["index"] == 2
    assert two["grade"] == "background" and two["ev_comp"] == 0.0
    assert _earlier_keys(two) == _earlier_keys(inline.record)
    assert {"preview", "queued", "composite", "save_graded"} <= set(two["stage_ms"])
    assert read_entries(tmp_path / "state") == []


def test_an_unfinished_double_is_finished_from_both_originals(tmp_path, background):
    result, path, text = _interrupted_shot(tmp_path, background, double=True)
    entry = json.loads(text)
    assert entry["originals"] == ["2026-09-03/210507_ungraded.jpg",
                                  "2026-09-03/210507-2_ungraded.jpg"]
    assert entry["ev"] == 0.0 and entry["double"]["ev_comp"] == [0.0, 0.0]
    first = result.graded.read_bytes()
    result.graded.unlink()
    log = result.graded.parent / "captures.jsonl"
    log.write_text(log.read_text().splitlines()[0] + "\n")  # exposure 1's line only
    path.write_text(text)

    again = background()
    _settle(again)
    assert again.close_grades(timeout=5)
    record = _lines(again.out_root)[-1]
    assert record["recovered"] is True and record["double"]["index"] == 2
    assert "composite" in record["stage_ms"]
    diff = np.abs(load_rgb(result.graded)[0].astype(int)
                  - np.asarray(Image.open(io.BytesIO(first)).convert("RGB")).astype(int))
    assert diff.mean() < 3.0
    assert not path.exists()


def test_a_full_queue_makes_the_next_capture_wait(background):
    session = background(hold=True, queue_depth=1, now=lambda: FIXED)
    session.capture()
    session.pipeline.wait_entered()  # the worker has the first; nothing is waiting
    session.capture()  # takes the one place
    assert session.pending_grades == 2
    done, results = threading.Event(), []

    def third():
        results.append(session.capture())
        done.set()

    thread = threading.Thread(target=third, daemon=True)
    thread.start()
    # A bounded look for something that must not happen; it cannot fail by being slow.
    assert not done.wait(0.3), "the capture did not wait for a place in the queue"
    assert session.pending_grades == 2
    assert results == []
    session.pipeline.release()  # the first grade ends, the second leaves the queue
    assert done.wait(5), "the capture was not released by a free place"
    thread.join(5)
    assert results[0].original.exists()  # it waited; it did not fail or drop the shot
    session.pipeline.release(2)
    assert session.close_grades(timeout=5)
    assert [line["pifilm"] for line in _lines(session.out_root)] == [
        "210507_graded.jpg", "210507-2_graded.jpg", "210507-3_graded.jpg",
    ]


def test_closing_releases_a_capture_waiting_on_a_full_queue(background):
    """Stopping must not deadlock, and the waiting photograph is still finished."""
    session = background(hold=True, queue_depth=1, now=lambda: FIXED)
    session.capture()
    session.pipeline.wait_entered()
    session.capture()
    done, results = threading.Event(), []

    def third():
        results.append(session.capture())
        done.set()

    thread = threading.Thread(target=third, daemon=True)
    thread.start()
    assert not done.wait(0.3)
    assert session.close_grades(timeout=0) is False  # the worker is still grading
    session.pipeline.release(3)
    assert done.wait(5), "close did not release the waiting capture"
    thread.join(5)
    assert session.close_grades(timeout=5)
    assert results[0].graded.exists()
    # The refused shot was graded after the two the queue had accepted.
    assert [line["pifilm"] for line in _lines(session.out_root)] == [
        "210507_graded.jpg", "210507-2_graded.jpg", "210507-3_graded.jpg",
    ]
    assert all(line["grade"] == "background" for line in _lines(session.out_root))


def test_a_shot_refused_by_a_closing_queue_is_graded_inline(tmp_path, pipeline, background):
    inline = _session(tmp_path / "inline", pipeline, now=lambda: FIXED).capture()
    session = background(now=lambda: FIXED)
    assert session.close_grades(timeout=5)
    result = session.capture()
    # No worker is left, so the file and its record are there when capture returns.
    assert result.graded.read_bytes() == inline.pifilm.read_bytes()
    (record,) = _lines(session.out_root)
    assert record["pifilm"] == result.graded.name
    assert record["shutter_to_saved_ms"] >= record["shutter_to_ready_ms"]
    assert read_entries(tmp_path / "state") == []
    assert result.pifilm.exists() and result.pifilm != result.graded


class _CheapPipeline:
    """Returns the frame as given: for tests about sizes and files, not about colour."""

    def process(self, rgb, *, grain=True, rng=None, ev=0.0, timings=None, halation=True):
        return rgb, {"wb_gains": [1.0, 1.0, 1.0], "ev_comp": ev}


@pytest.mark.parametrize(("frame", "quick"), [
    ((3040, 4056), (760, 1014)),
    ((480, 640), (480, 640)),
    ((1080, 1920), (540, 960)),
])
def test_quick_grade_is_reduced_by_a_whole_factor(tmp_path, frame, quick):
    camera = FakeCamera([np.zeros((*frame, 3), np.uint8)])
    session = CaptureSession(camera, _CheapPipeline(), tmp_path / "shots",
                             background=True, state_dir=tmp_path / "state", log=lambda _: None)
    try:
        result = session.capture()
        assert load_rgb(result.pifilm)[0].shape == (*quick, 3)
    finally:
        assert session.close_grades(timeout=5)
    assert load_rgb(result.graded)[0].shape == (*frame, 3)


def test_only_the_newest_quick_grades_are_kept(tmp_path):
    camera = FakeCamera([np.zeros((6, 8, 3), np.uint8)])
    session = CaptureSession(camera, _CheapPipeline(), tmp_path / "shots", now=lambda: FIXED,
                             background=True, state_dir=tmp_path / "state", log=lambda _: None)
    try:
        results = [session.capture() for _ in range(19)]
    finally:
        assert session.close_grades(timeout=5)
    kept = sorted(results[0].pifilm.parent.glob("preview-*.jpg"))
    assert len(kept) == 16
    assert [r.pifilm.exists() for r in results] == [False] * 3 + [True] * 16
    assert all(r.graded.exists() for r in results)  # photographs are never pruned


def test_a_failed_grade_is_logged_counted_and_does_not_block_the_next(
    tmp_path, monkeypatch, background,
):
    from pifilm.capture import app

    real, failures = app.save_jpeg, []

    def flaky(rgb, path, *args, **kwargs):
        if Path(path).name.endswith("_graded.jpg") and not failures:
            failures.append(path)
            raise OSError("No space left on device")
        return real(rgb, path, *args, **kwargs)

    monkeypatch.setattr(app, "save_jpeg", flaky)
    session = background(now=lambda: FIXED)
    first = session.capture()
    _settle(session)
    second = session.capture()
    assert session.close_grades(timeout=5)
    assert session.test_log == [
        "grade failed (2026-09-03_210507.json): No space left on device",
    ]
    assert not first.graded.exists() and second.graded.exists()
    assert [line["pifilm"] for line in _lines(session.out_root)] == [second.graded.name]
    ((path, entry),) = read_entries(tmp_path / "state")
    assert entry["attempts"] == 1 and entry["graded"] == first.graded.name

    # The next start tries once more, and this time the disk has room.
    again = background()
    _settle(again)
    assert again.close_grades(timeout=5)
    assert first.graded.exists() and not path.exists()
    assert _lines(again.out_root)[-1]["recovered"] is True


def test_a_failed_quick_grade_still_grades_the_photograph(tmp_path, monkeypatch, background):
    from pifilm.capture import app

    real = app.save_jpeg

    def no_previews(rgb, path, *args, **kwargs):
        if Path(path).name.startswith("preview-"):
            raise OSError("temporary directory is full")
        return real(rgb, path, *args, **kwargs)

    monkeypatch.setattr(app, "save_jpeg", no_previews)
    session = background(now=lambda: FIXED)
    with pytest.raises(OSError, match="temporary directory"):
        session.capture()
    assert session.close_grades(timeout=5)
    assert (session.out_root / "2026-09-03" / "210507_graded.jpg").exists()
    assert len(_lines(session.out_root)) == 1
    assert read_entries(tmp_path / "state") == []


def test_an_unwritable_state_directory_costs_the_entry_not_the_photograph(tmp_path, background):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("in the way")
    session = background(state=blocked)
    result = session.capture()
    assert session.close_grades(timeout=5)
    assert result.graded.exists() and len(_lines(session.out_root)) == 1
    assert len(session.test_log) == 1
    assert session.test_log[0].startswith("warning: could not write the pending entry")
    assert blocked.read_text() == "in the way"


def test_a_new_shot_never_overwrites_another_folders_entry(tmp_path, background):
    theirs = write_entry(tmp_path / "state", "2026-09-03_210507",
                         {"out_root": str(tmp_path / "another-folder")})
    session = background(hold=True, now=lambda: FIXED)
    session.capture()
    session.pipeline.wait_entered()
    names = sorted(path.name for path, _ in read_entries(tmp_path / "state"))
    assert names == ["2026-09-03_210507-2.json", "2026-09-03_210507.json"]
    session.pipeline.release()
    assert session.close_grades(timeout=5)
    assert [path for path, _ in read_entries(tmp_path / "state")] == [theirs]


def test_the_controller_reports_and_finishes_the_sessions_grades(background):
    session = background(hold=True)
    controller = CaptureController(session)
    closed = False
    try:
        controller.submit("one")
        session.pipeline.wait_entered()
        deadline = time.monotonic() + 5
        while controller.status("one").state != "complete":
            assert time.monotonic() < deadline
            threading.Event().wait(0.005)
        # The capture job is over while its grade is still running.
        assert controller.snapshot().pending_grades == 1
        result = controller.status("one").result
        assert result.pifilm.exists() and not result.graded.exists()
        session.pipeline.release()
        controller.close()
        closed = True
        assert result.graded.exists()
        assert controller.snapshot().pending_grades == 0
    finally:
        session.pipeline.release(10)
        if not closed:
            controller.close()


def test_pipeline_process_is_safe_from_two_threads_at_once(pipeline):
    """The quick grade runs on the capture thread while the worker grades a full frame."""
    full, small = synthetic_frame(480, 640), synthetic_frame(120, 160)
    want_full, _ = pipeline.process(full, rng=np.random.default_rng(7))
    want_small, _ = pipeline.process(small, grain=False)
    start, wrong = threading.Barrier(2), []

    def run(frame, want, **kwargs):
        start.wait(timeout=5)
        for _ in range(8):
            got, _ = pipeline.process(frame, **dict(kwargs, rng=np.random.default_rng(7)))
            if not np.array_equal(got, want):
                wrong.append(kwargs)

    threads = [
        threading.Thread(target=run, args=(full, want_full)),
        threading.Thread(target=run, args=(small, want_small), kwargs={"grain": False}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert wrong == []


def test_announce_says_a_background_shot_is_still_grading(tmp_path, pipeline, background):
    session = background(now=lambda: FIXED)
    lines = []
    _announce(session.capture(), lines.append)
    assert session.close_grades(timeout=5)
    assert len(lines) == 1
    assert lines[0].startswith("Saved 210507_ungraded.jpg in ")
    assert lines[0].endswith(" ms; grading 210507_graded.jpg in the background")

    _announce(_session(tmp_path / "inline", pipeline, now=lambda: FIXED).capture(), lines.append)
    assert lines[1].startswith("Saved 210507_graded.jpg + 210507_ungraded.jpg in ")
    assert "wb=" in lines[1] and "background" not in lines[1]


@pytest.mark.parametrize(("argv", "background_expected"), [
    ([], True),
    (["--grade", "background"], True),
    (["--grade", "inline"], False),
])
def test_grade_flag_reaches_the_session(camera_cli, monkeypatch, tmp_path, argv,
                                        background_expected):
    session_kwargs = {}

    class RecordingSession:
        def __init__(self, camera, pipeline, out, **kwargs):
            session_kwargs.update(kwargs)
            self.camera = camera

        def capture(self, *, delay_s=0.0):
            raise AssertionError("capture should not be called in this test")

    monkeypatch.setattr(camera_cli, "CaptureSession", RecordingSession)
    assert camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path), *argv]) == 0
    assert session_kwargs["background"] is background_expected
    assert session_kwargs["state_dir"] is None  # the session resolves the default


def test_state_dir_flag_reaches_the_session(camera_cli, monkeypatch, tmp_path):
    session_kwargs, closes = {}, []

    class RecordingSession:
        def __init__(self, camera, pipeline, out, **kwargs):
            session_kwargs.update(kwargs)
            self.camera = camera

        def close_grades(self, timeout=None):
            closes.append(timeout)
            return True

    monkeypatch.setattr(camera_cli, "CaptureSession", RecordingSession)
    state = tmp_path / "pending"
    assert camera_cli.main(
        ["--fake", "--no-preview", "--out", str(tmp_path), "--state-dir", str(state)]
    ) == 0
    assert session_kwargs["state_dir"] == state
    assert closes, "main must finish the grades on the way out"


def test_an_unknown_grade_mode_is_refused(camera_cli, tmp_path):
    with pytest.raises(SystemExit):
        camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path), "--grade", "later"])


def test_main_finishes_the_grades_before_it_returns(tmp_path, monkeypatch, capsys):
    from pifilm.capture import app

    keys = iter([" ", "q"])
    terminal = Mock()
    terminal.read = lambda *args, **kwargs: next(keys)
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    out, state = tmp_path / "shots", tmp_path / "state"
    assert main(["--fake", "--no-preview", "--seed", "1", "--out", str(out),
                 "--state-dir", str(state)]) == 0
    assert len(list(out.glob("*/*_graded.jpg"))) == 1
    (record,) = _lines(out)
    assert record["grade"] == "background"
    assert read_entries(state) == []
    assert "in the background" in capsys.readouterr().out
    assert not any(t.name == "pifilm-grade" for t in threading.enumerate())


def test_main_inline_writes_no_state(tmp_path, monkeypatch):
    from pifilm.capture import app

    keys = iter([" ", "q"])
    terminal = Mock()
    terminal.read = lambda *args, **kwargs: next(keys)
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    out, state = tmp_path / "shots", tmp_path / "state"
    assert main(["--fake", "--no-preview", "--grade", "inline", "--out", str(out),
                 "--state-dir", str(state)]) == 0
    (record,) = _lines(out)
    assert not NEW_KEYS & set(record)
    assert not state.exists()


def _record_syncs(monkeypatch):
    """Record every ``os.fsync`` (as the file's inode) and every entry removal, in order."""
    from pifilm.capture import app

    events = []
    real_fsync, real_remove = os.fsync, app.remove_entry

    def fsync(fd):
        events.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    def remove(path):
        events.append("remove")
        return real_remove(path)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(app, "remove_entry", remove)
    return events


def test_a_background_grade_is_on_disk_before_its_entry_is_removed(monkeypatch, background):
    """The entry's removal can reach the disk before data that was only written.

    Without the two syncs a power cut just after a grade could leave no entry, no
    record and an empty graded file, with nothing left to say a grade is owed.
    """
    events = _record_syncs(monkeypatch)
    session = background(now=lambda: FIXED)
    result = session.capture()
    assert session.close_grades(timeout=5)
    graded = result.graded.stat().st_ino
    log = (result.graded.parent / "captures.jsonl").stat().st_ino
    assert [e for e in events if e in (graded, log, "remove")] == [graded, log, "remove"]


def test_a_recovered_grade_is_on_disk_before_its_entry_is_removed(
    tmp_path, monkeypatch, background,
):
    result, path, text = _interrupted_shot(tmp_path, background)
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)
    events = _record_syncs(monkeypatch)
    again = background()
    _settle(again)
    assert again.close_grades(timeout=5)
    graded = result.graded.stat().st_ino
    log = (result.graded.parent / "captures.jsonl").stat().st_ino
    assert [e for e in events if e in (graded, log, "remove")] == [graded, log, "remove"]


def test_an_inline_capture_syncs_nothing(tmp_path, pipeline, monkeypatch):
    events = _record_syncs(monkeypatch)
    session = _session(tmp_path, pipeline)
    session.capture()
    session.set_double_exposure(True)
    session.capture()
    session.capture()
    assert events == []


def test_a_recovered_record_never_joins_a_torn_line(tmp_path, background):
    """A power cut can leave half a line at the end of the log, with no newline."""
    result, path, text = _interrupted_shot(tmp_path, background)
    log = result.graded.parent / "captures.jsonl"
    whole = log.read_text()
    assert whole.endswith("\n")
    log.write_text(whole[: len(whole) // 2])  # the record was being written when it went
    path.write_text(text)

    again = background(now=lambda: FIXED)
    shot = again.capture()
    assert again.close_grades(timeout=5)
    lines = log.read_text().splitlines()
    assert len(lines) == 3 and lines[0] == whole[: len(whole) // 2]
    recovered, new = json.loads(lines[1]), json.loads(lines[2])
    assert recovered["recovered"] is True and recovered["pifilm"] == result.graded.name
    assert new["pifilm"] == shot.graded.name
    assert log.read_text().endswith("\n")
    assert "\n\n" not in log.read_text()  # a whole log gets no blank line


class _Signals:
    """Stands in for ``signal.signal``: keeps the handlers instead of installing them."""

    def __init__(self):
        self.previous = object()
        self.calls = []

    def __call__(self, number, handler):
        self.calls.append((number, handler))
        return self.previous

    def deliver(self):
        """What the kernel does on SIGTERM: run the handler installed now."""
        number, handler = self.calls[-1]
        assert callable(handler), "SIGTERM arrived with no handler of ours installed"
        handler(number, None)


def test_sigterm_ends_the_headless_service_cleanly(monkeypatch, tmp_path):
    """systemd stops the service with SIGTERM; the clean-up, which finishes the
    queued grades, has to run as it does on Ctrl-C."""
    import signal as signal_module

    from pifilm.capture import app

    signals, closed = _Signals(), []

    class Listener:
        def __init__(self, controller, token, listen, power=None):
            pass

        def start(self):
            pass

        def close(self):
            signals.deliver()  # a second SIGTERM, in the middle of the clean-up
            closed.append("remote")

    real_close = app.CaptureSession.close_grades

    def close_grades(self, timeout=None):
        signals.deliver()
        closed.append("grades")
        return real_close(self, timeout)

    def sleep(_):
        signals.deliver()
        raise AssertionError("SIGTERM did not end the loop")

    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "secret")
    monkeypatch.setattr(app, "RemoteCaptureServer", Listener)
    monkeypatch.setattr(app.CaptureSession, "close_grades", close_grades)
    monkeypatch.setattr(app, "has_display", lambda: False)
    monkeypatch.setattr(app.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(app.signal, "signal", signals)
    monkeypatch.setattr(app.time, "sleep", sleep)

    assert app.main(["--fake", "--no-preview", "--out", str(tmp_path / "shots"),
                     "--state-dir", str(tmp_path / "state"),
                     "--remote-listen", "127.0.0.1:8765"]) == 0
    assert closed == ["remote", "grades"]
    assert not any(t.name == "pifilm-grade" for t in threading.enumerate())
    # Installed for SIGTERM, silenced for the clean-up, and the handler that was
    # there before put back on the way out.
    assert [number for number, _ in signals.calls] == [signal_module.SIGTERM] * 3
    assert signals.calls[-1][1] is signals.previous


def test_sigterm_ends_the_terminal_loop_cleanly(monkeypatch, tmp_path):
    from pifilm.capture import app

    signals = _Signals()
    terminal = Mock()
    terminal.read = lambda *args, **kwargs: signals.deliver()
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(app.signal, "signal", signals)
    assert main(["--fake", "--no-preview", "--out", str(tmp_path / "shots"),
                 "--state-dir", str(tmp_path / "state")]) == 0
    assert not any(t.name == "pifilm-grade" for t in threading.enumerate())
    assert signals.calls[-1][1] is signals.previous


def test_main_off_the_main_thread_installs_no_signal_handler(monkeypatch, tmp_path):
    """``signal.signal`` raises outside the main thread; ``main`` must still run there."""
    from pifilm.capture import app

    signals = _Signals()
    terminal = Mock()
    terminal.read = lambda *args, **kwargs: "q"
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(app.signal, "signal", signals)
    codes = []
    thread = threading.Thread(target=lambda: codes.append(main(
        ["--fake", "--no-preview", "--out", str(tmp_path / "shots"),
         "--state-dir", str(tmp_path / "state")])))
    thread.start()
    thread.join(5)
    assert codes == [0]
    assert signals.calls == []


def test_private_pictures_stay_out_of_the_real_temporary_directory(
    background, tmp_path_factory,
):
    """The suite's own guard: a session's cards and quick grades land in pytest's tree."""
    result = background().capture()
    assert tmp_path_factory.getbasetemp().resolve() in result.pifilm.resolve().parents


class _MetaCamera(FakeCamera):
    """A Picamera2-style frame: no camera JPEG, a DNG and metadata."""

    def read(self, full=True):
        frame = super().read()
        return Frame(frame.rgb, None, "picamera2", dng=b"raw", metadata={"Lux": 1.0})


def test_a_background_original_is_synced_by_the_grade_worker_not_the_capture(
    tmp_path, monkeypatch, background,
):
    """Syncing 20 MB of DNG cost the Pi up to a second per shot, so the camera does
    not wait for it: the worker syncs the original and the DNG before it grades."""
    from pifilm.capture import app

    events = _record_syncs(monkeypatch)
    real_sync = app._sync_file
    threads = []

    def sync(path):
        threads.append(threading.current_thread().name)
        return real_sync(path)

    monkeypatch.setattr(app, "_sync_file", sync)
    session = background(_MetaCamera([synthetic_frame(90, 160)]), hold=True, now=lambda: FIXED)
    result = session.capture()
    session.pipeline.wait_entered()  # the full grade has begun: the syncs are done
    original = result.original.stat().st_ino
    dng = (result.original.parent / "210507.dng").stat().st_ino
    assert [e for e in events if e in (original, dng)] == [original, dng]
    assert threads == ["pifilm-grade", "pifilm-grade"]
    session.pipeline.release()
    session.close_grades()
    record = json.loads((result.original.parent / "captures.jsonl").read_text())
    assert "sync_original" in record["stage_ms"]


def test_both_originals_of_a_background_double_are_synced(monkeypatch, background):
    events = _record_syncs(monkeypatch)
    session = background(FakeCamera(_double_frames()), hold=True, now=lambda: FIXED)
    session.set_double_exposure(True)
    first = session.capture()
    second = session.capture()
    session.pipeline.wait_entered()
    assert first.original.stat().st_ino in events
    assert second.original.stat().st_ino in events
    session.pipeline.release()


def test_an_inline_capture_with_a_dng_syncs_nothing(tmp_path, pipeline, monkeypatch):
    events = _record_syncs(monkeypatch)
    session = CaptureSession(_MetaCamera([synthetic_frame(90, 160)]), pipeline, tmp_path / "shots")
    session.capture()
    assert events == []


def test_a_stop_leaves_the_recovery_backlog_for_the_next_start(tmp_path, background):
    """Stopping finishes this run's photographs, not everything an old run left."""
    first = background(hold=True, now=lambda: FIXED)
    results = [first.capture() for _ in range(3)]
    first.pipeline.wait_entered()
    texts = {path: path.read_text() for path, _ in read_entries(tmp_path / "state")}
    assert len(texts) == 3
    first.pipeline.release(3)
    assert first.close_grades(timeout=5)
    for result in results:
        result.graded.unlink()
    (results[0].graded.parent / "captures.jsonl").unlink()
    for path, text in texts.items():
        path.write_text(text)

    again = background(hold=True, now=lambda: FIXED)
    assert again.pending_grades == 3
    again.pipeline.wait_entered()  # the first recovered grade is running
    shot = again.capture()  # a photograph of this run, behind it
    assert again.close_grades(timeout=0) is False
    assert again.pending_grades == 2  # the running one and this run's; the rest are left
    again.pipeline.release(2)
    assert again.close_grades(timeout=5)
    assert again.pending_grades == 0
    # One recovered grade had started and was finished. (Which one: entries are
    # taken in name order, and "-2" sorts before the unsuffixed name.)
    assert sum(r.graded.exists() for r in results) == 1
    assert shot.graded.exists()
    assert len(read_entries(tmp_path / "state")) == 2
    assert "leaving 2 unfinished grade(s) for the next start" in again.test_log

    # And the next start does finish them.
    third = background()
    _settle(third)
    assert third.close_grades(timeout=5)
    assert all(r.graded.exists() for r in results)
    assert read_entries(tmp_path / "state") == []


def test_a_grade_does_not_wait_for_ever_for_the_capture_side(
    tmp_path, pipeline, monkeypatch, background,
):
    """An interrupt can end a capture between handing the job over and saying how
    long the shutter was held. The grade is then finished without that number."""
    from pifilm.capture import app

    monkeypatch.setattr(app, "READY_WAIT_S", 0.01, raising=False)
    session = background(now=lambda: FIXED)
    day = session.out_root / "2026-09-03"
    day.mkdir(parents=True)
    entry = write_entry(tmp_path / "state", "by-hand", {})
    work = app._GradeWork(
        "2026-09-03", "x_graded.jpg", ["2026-09-03/x_ungraded.jpg"], 1, 0.0,
        {"timestamp": "t", "pifilm": "x_graded.jpg", "ev_comp": 0.0},
        frames=(synthetic_frame(90, 160),), shutter=time.perf_counter(),
    )
    done = threading.Event()

    def grade():
        session._run_grade(app.GradeJob(payload=work, entry=entry))  # ``ready`` is never set
        done.set()

    threading.Thread(target=grade, daemon=True).start()
    assert done.wait(5), "the grade waited for ever for the capture side"
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    assert "shutter_to_ready_ms" not in record and "shutter_to_saved_ms" in record
    assert (day / "x_graded.jpg").exists()
    assert not entry.exists()


def test_a_twice_failed_grade_is_reported_as_given_up(tmp_path, background):
    result, path, text = _interrupted_shot(tmp_path, background)
    result.original.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)
    for _ in range(2):
        again = background()
        _settle(again)
        assert again.close_grades(timeout=5)
    assert again.test_log[0] == "recovering 1 unfinished grade(s)"
    assert again.test_log[1].startswith(f"grade failed ({path.name}): ")
    assert again.test_log[2] == (
        f"grade given up ({path.name}): failed 2 times, kept as {path.stem}.failed"
    )
    assert path.with_suffix(".failed").exists() and not path.exists()


class _Part:
    """A display, a touch reader or a server that is sent SIGTERM while it closes."""

    def __init__(self, signals, closed, name):
        self._signals, self._closed, self._name = signals, closed, name

    def start(self):
        return None

    def close(self):
        self._signals.deliver()
        self._closed.append(self._name)


def _viewfinder_main(camera_cli, monkeypatch, tmp_path, run):
    """Run ``main`` with a viewfinder whose loop is ``run``; SIGTERM arrives at every close."""
    signals, closed = _Signals(), []

    class Loop:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, stop):
            return run(signals, stop)

    class Camera(FakeCamera):
        def close(self):
            signals.deliver()
            closed.append("camera")

    real_close = camera_cli.CaptureSession.close_grades

    def close_grades(self, timeout=None):
        signals.deliver()
        closed.append("grades")
        return real_close(self, timeout)

    monkeypatch.setenv("PIFILM_REMOTE_TOKEN", "token")
    monkeypatch.setattr(camera_cli, "FakeCamera", Camera)
    monkeypatch.setattr(camera_cli.CaptureSession, "close_grades", close_grades)
    monkeypatch.setattr(camera_cli, "RemoteCaptureServer",
                        lambda *args, **kwargs: _Part(signals, closed, "remote"))
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: (
        _Part(signals, closed, "display"), _Part(signals, closed, "touch"),
    ))
    monkeypatch.setattr(camera_cli, "ViewfinderLoop", Loop)
    monkeypatch.setattr(camera_cli, "_arm_exit_deadline", lambda seconds, code: None)
    monkeypatch.setattr(camera_cli.signal, "signal", signals)
    code = camera_cli.main([
        "--fake", "--display", "waveshare28", "--no-preview",
        "--remote-listen", "127.0.0.1:8765", "--out", str(tmp_path / "shots"),
        "--state-dir", str(tmp_path / "state"),
    ])
    assert signals.calls[-1][1] is signals.previous
    return code, closed


def test_a_second_sigterm_cannot_break_into_the_viewfinders_clean_up(
    camera_cli, monkeypatch, tmp_path,
):
    """The viewfinder's own handler takes the first SIGTERM, so ``main``'s has not
    fired; a second one during the clean-up used to raise inside it and leave the
    camera open."""
    def run(signals, stop):
        signals.deliver()  # the first SIGTERM: the viewfinder's handler sets ``stop``
        assert stop.is_set()

    code, closed = _viewfinder_main(camera_cli, monkeypatch, tmp_path, run)
    assert code == 0
    assert closed == ["touch", "display", "remote", "grades", "camera"]


def test_sigterm_cannot_break_into_the_stall_exits_clean_up(camera_cli, monkeypatch, tmp_path):
    from pifilm.capture.errors import CameraStalled

    def run(signals, stop):
        raise CameraStalled("no preview frame within 5.0 s")

    code, closed = _viewfinder_main(camera_cli, monkeypatch, tmp_path, run)
    assert code == 1  # still the exit that makes systemd restart the service
    assert closed == ["touch", "display", "remote", "grades", "camera"]


# --- grading on the GPU ------------------------------------------------------


@pytest.fixture
def gpu(tmp_path, monkeypatch):
    """Background sessions whose grader is a real ``GpuGrader`` on a fake device.

    ``GpuGrader.open`` is replaced, so nothing here looks for a GPU. ``gpu.make``
    builds a session (closed at the end, like ``background``'s); ``gpu.opened``
    waits for the worker to have opened its grader and returns the device.
    """
    real_open = GpuGrader.open
    box = SimpleNamespace(threads=[], devices=[], graders=[], available=True,
                          pipeline=Pipeline(tile_artifacts()), max_texture=4096,
                          hold=None, entered=threading.Event(),
                          marker=tmp_path / "gpu-opening")
    done = threading.Event()
    made, held = [], []

    def fake_open(artifacts, *, log=print):
        box.threads.append(threading.current_thread().name)
        try:
            box.entered.set()
            if box.hold is not None:
                assert box.hold.wait(5), "the held open was never released"
            if not box.available:
                log("gpu grading off: no GPU in this test")
                return None
            device = FakeDevice(max_texture=box.max_texture)
            grader = real_open(artifacts, log=log, device_factory=lambda: device)
            box.devices.append(device)
            box.graders.append(grader)
            return grader
        finally:
            done.set()

    monkeypatch.setattr(GpuGrader, "open", staticmethod(fake_open))

    def make(camera=None, *, hold=False, grade_backend="auto", background=True, **kwargs):
        pipe = box.pipeline
        if hold:
            pipe = _HeldPipeline(pipe)
            held.append(pipe)
        log = []
        state = {}
        if background:
            state["state_dir"] = kwargs.pop("state_dir", tmp_path / "state")
        session = CaptureSession(
            camera or FakeCamera([synthetic_frame(90, 160)]), pipe, tmp_path / "shots",
            now=kwargs.pop("now", lambda: FIXED), seed_rng=np.random.default_rng(0),
            background=background, log=log.append, grade_backend=grade_backend,
            **state, **kwargs,
        )
        session.test_log = log
        made.append(session)
        return session

    def opened():
        assert done.wait(5), "the worker never opened a grader"
        return box.devices[-1] if box.devices else None

    box.make, box.opened = make, opened
    yield box
    if box.hold is not None:
        box.hold.set()
    for pipe in held:
        pipe.release(100)
    for session in made:
        assert session.close_grades(timeout=5), "a grade worker outlived its test"


def _gpu_bytes(tmp_path, seed, frame=None):
    """The file a GPU grade of the fixture's frame with ``seed`` must be."""
    pipe = Pipeline(tile_artifacts())
    grader = GpuGrader(pipe.artifacts, FakeDevice(), log=lambda _: None)
    frame = synthetic_frame(90, 160) if frame is None else frame
    graded, _ = pipe.process(frame, rng=np.random.default_rng(seed), grader=grader)
    assert graded.shape[2] == 4
    return save_jpeg(graded, tmp_path / "expected-gpu.jpg").read_bytes()


def _cpu_bytes(tmp_path, seed, frame=None):
    pipe = Pipeline(tile_artifacts())
    frame = synthetic_frame(90, 160) if frame is None else frame
    graded, _ = pipe.process(frame, rng=np.random.default_rng(seed))
    return save_jpeg(graded, tmp_path / "expected-cpu.jpg").read_bytes()


def test_the_grade_worker_opens_the_grader_and_grades_on_it(tmp_path, gpu):
    session = gpu.make()
    result = session.capture()
    device = gpu.opened()
    assert session.close_grades(timeout=5)
    assert gpu.threads == ["pifilm-grade"]
    (record,) = _lines(session.out_root)
    assert record["pipeline_backend"] == BACKEND
    assert list(record)[-1] == "pipeline_backend"
    assert {"normalize", "gpu", "gpu_upload", "gpu_shader", "gpu_readback"} <= set(
        record["stage_ms"])
    assert not {"lut", "grain"} & set(record["stage_ms"])
    seed = record["grain_seed"]
    assert result.graded.read_bytes() == _gpu_bytes(tmp_path, seed)
    assert result.graded.read_bytes() != _cpu_bytes(tmp_path, seed)
    # The self-test and the one photograph, both on the worker's thread.
    assert [run["shape"] for run in device.runs] == [(64, 96, 3), (90, 160, 3)]
    assert len(device.threads) == 1 and threading.get_ident() not in device.threads
    assert device.closed == 1 and gpu.graders[0].enabled is False
    assert session.test_log == ["gpu grading on (Fake GPU, self-test passed)"]
    # What the capture job carries says nothing of a grade that had not happened.
    assert "pipeline_backend" not in result.record


def test_without_a_gpu_the_worker_grades_on_the_cpu(tmp_path, gpu):
    gpu.available = False
    session = gpu.make()
    results = [session.capture() for _ in range(2)]
    assert session.close_grades(timeout=5)
    records = _lines(session.out_root)
    assert [r["pipeline_backend"] for r in records] == ["cpu", "cpu"]
    for result, record in zip(results, records, strict=True):
        assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    assert {"normalize", "lut", "grain"} <= set(records[0]["stage_ms"])
    assert gpu.threads == ["pifilm-grade"]
    assert session.test_log == ["gpu grading off: no GPU in this test"]


def test_a_photograph_the_gpu_fails_on_is_graded_on_the_cpu(tmp_path, gpu):
    session = gpu.make()
    gpu.opened().fail_next = 1
    first, second = session.capture(), session.capture()
    assert session.close_grades(timeout=5)
    one, two = _lines(session.out_root)
    assert (one["pipeline_backend"], two["pipeline_backend"]) == ("cpu", BACKEND)
    assert first.graded.read_bytes() == _cpu_bytes(tmp_path, one["grain_seed"])
    assert second.graded.read_bytes() == _gpu_bytes(tmp_path, two["grain_seed"])
    assert {"lut", "grain"} <= set(one["stage_ms"]) and "gpu" not in one["stage_ms"]
    assert session.test_log[1:] == [
        "gpu grade failed (210507_graded.jpg): GL error after grading: 0x505; "
        "graded on the CPU",
    ]
    assert read_entries(tmp_path / "state") == []


def test_three_failures_in_a_row_leave_the_rest_to_the_cpu_quietly(tmp_path, gpu):
    session = gpu.make()
    gpu.opened().fail_next = 3
    results = [session.capture() for _ in range(5)]
    assert session.close_grades(timeout=5)
    records = _lines(session.out_root)
    assert [r["pipeline_backend"] for r in records] == ["cpu"] * 5
    for result, record in zip(results, records, strict=True):
        assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    log = session.test_log
    assert sum(line.startswith("gpu grade failed") for line in log) == 3
    assert "gpu grading off: 3 failures in a row" in log
    assert len(log) == 5  # on, three failures, off: nothing for the last two


@pytest.mark.parametrize("kwargs", [
    {"grade_backend": "cpu"},
    {"grade_backend": "cpu", "background": False},
    {"grade_backend": "auto", "background": False},
])
def test_the_cpu_backend_and_inline_sessions_never_open_a_grader(tmp_path, gpu, kwargs):
    session = gpu.make(**kwargs)
    result = session.capture()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    assert record["pipeline_backend"] == "cpu"
    assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    assert gpu.threads == [] and session.test_log == []


def test_an_inline_record_names_its_backend_after_the_stages(tmp_path, pipeline):
    class MetaCamera(FakeCamera):
        def read(self, full=True):
            frame = super().read()
            return Frame(frame.rgb, None, "picamera2", dng=b"raw", metadata={"Lux": 1.0})

    session = CaptureSession(MetaCamera([synthetic_frame(90, 160)]), pipeline,
                             tmp_path / "shots", seed_rng=np.random.default_rng(0))
    record = session.capture().record
    assert list(record)[-4:] == ["stage_ms", "pipeline_backend", "camera_metadata", "dng"]
    assert record["pipeline_backend"] == "cpu"
    assert _lines(session.out_root) == [record]


def test_an_unknown_grade_backend_is_refused(tmp_path, pipeline):
    with pytest.raises(ValueError, match="grade_backend"):
        CaptureSession(FakeCamera([synthetic_frame(90, 160)]), pipeline, tmp_path / "shots",
                       grade_backend="gpu")


def test_a_shot_refused_by_a_closing_queue_is_graded_on_the_cpu(tmp_path, gpu):
    """It is graded on the capture's thread, which the graphics context is not."""
    session = gpu.make()
    gpu.opened()
    assert session.close_grades(timeout=5)
    result = session.capture()
    (record,) = _lines(session.out_root)
    assert record["pipeline_backend"] == "cpu"
    assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    assert len(gpu.devices[0].runs) == 1  # the self-test only


def test_a_grader_is_not_used_from_the_capture_thread(tmp_path, gpu, monkeypatch):
    """The same refused shot while the worker's grader is still open: the
    grader says no by thread, and nothing is logged about it."""
    session = gpu.make()
    device = gpu.opened()
    grader = gpu.graders[0]
    with monkeypatch.context() as patch:
        patch.setattr(session._queue, "submit", lambda job: False)
        # The worker, and so its grader, live on past this shot.
        patch.setattr(session._queue, "close", lambda timeout=None: True)
        result = session.capture()
    assert grader.enabled is True
    (record,) = _lines(session.out_root)
    assert record["pipeline_backend"] == "cpu"
    assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    assert len(device.runs) == 1
    assert session.test_log == ["gpu grading on (Fake GPU, self-test passed)"]


def test_a_recovered_grade_records_what_graded_it(tmp_path, gpu):
    first = gpu.make(hold=True, grade_backend="cpu")
    result = first.capture()
    first.pipeline.wait_entered()
    ((path, _),) = read_entries(tmp_path / "state")
    text = path.read_text()
    first.pipeline.release()
    assert first.close_grades(timeout=5)
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    again = gpu.make()
    gpu.opened()
    _settle(again)
    assert again.close_grades(timeout=5)
    (record,) = _lines(again.out_root)
    assert record["recovered"] is True and record["pipeline_backend"] == BACKEND
    assert list(record)[-3:] == ["grade", "recovered", "pipeline_backend"]
    assert again.test_log[0] == "recovering 1 unfinished grade(s)"
    # Graded from the saved JPEG, on the GPU.
    assert gpu.devices[0].runs[-1]["shape"] == (90, 160, 3)
    expected = _gpu_bytes(tmp_path, record["grain_seed"], load_rgb(result.original)[0])
    assert result.graded.read_bytes() == expected


def test_a_double_exposure_is_graded_on_the_gpu_and_exposure_1_names_no_backend(tmp_path, gpu):
    session = gpu.make(FakeCamera(_double_frames()))
    session.set_double_exposure(True)
    session.capture()
    session.capture()
    assert session.close_grades(timeout=5)
    one, two = _lines(session.out_root)
    assert "pipeline_backend" not in one and "params_version" not in one
    assert two["pipeline_backend"] == BACKEND and two["double"]["index"] == 2
    assert "composite" in two["stage_ms"] and "gpu" in two["stage_ms"]


def test_an_inline_double_exposure_records_the_cpu(tmp_path, pipeline):
    session = _session(tmp_path, pipeline, FakeCamera(_double_frames()))
    session.set_double_exposure(True)
    one, two = session.capture().record, session.capture().record
    assert "pipeline_backend" not in one
    assert two["pipeline_backend"] == "cpu"
    assert list(two)[-3:] == ["stage_ms", "pipeline_backend", "double"]


@pytest.mark.parametrize(("argv", "expected"), [
    ([], "auto"),
    (["--grade-backend", "auto"], "auto"),
    (["--grade-backend", "cpu"], "cpu"),
])
def test_grade_backend_flag_reaches_the_session(camera_cli, monkeypatch, tmp_path, argv,
                                                expected):
    session_kwargs = {}

    class RecordingSession:
        def __init__(self, camera, pipeline, out, **kwargs):
            session_kwargs.update(kwargs)
            self.camera = camera

    monkeypatch.setattr(camera_cli, "CaptureSession", RecordingSession)
    assert camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path), *argv]) == 0
    assert session_kwargs["grade_backend"] == expected


def test_grade_backend_flag_refuses_anything_else(camera_cli, tmp_path, capsys):
    with pytest.raises(SystemExit):
        camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path),
                         "--grade-backend", "gpu"])
    assert "--grade-backend" in capsys.readouterr().err


# --- the marker that keeps a hanging GPU from stopping every start ----------


def test_a_marker_left_by_the_last_start_keeps_the_gpu_closed(tmp_path, gpu):
    """An open that hangs holds the only grade worker for ever: no grade runs,
    the fourth shot waits, and without the marker the next start does the same."""
    gpu.marker.write_text("left by a start that never came back\n")
    session = gpu.make()
    results = [session.capture() for _ in range(2)]
    assert session.close_grades(timeout=5)
    assert gpu.threads == []
    assert session.test_log == [
        "gpu grading off: the last start did not finish opening the GPU; "
        f"delete {gpu.marker} to try again",
    ]
    records = _lines(session.out_root)
    assert [r["pipeline_backend"] for r in records] == ["cpu", "cpu"]
    for result, record in zip(results, records, strict=True):
        assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"])
    assert gpu.marker.exists()  # only a person removes it


@pytest.mark.parametrize("available", [True, False])
def test_the_marker_is_gone_once_the_open_has_returned(gpu, available):
    gpu.available = available
    session = gpu.make()
    gpu.opened()
    assert session.close_grades(timeout=5)
    assert gpu.threads == ["pifilm-grade"]
    assert not gpu.marker.exists()
    assert not any("did not finish" in line for line in session.test_log)


def test_the_marker_is_there_for_as_long_as_the_open_takes(gpu):
    gpu.hold = threading.Event()
    session = gpu.make()
    assert gpu.entered.wait(5)
    assert gpu.marker.is_file()
    assert not list(gpu.marker.parent.glob("state/*"))  # never among the entries
    gpu.hold.set()
    assert session.close_grades(timeout=5)
    assert not gpu.marker.exists()


def test_the_marker_is_written_where_no_state_directory_exists_yet(tmp_path, gpu):
    gpu.hold = threading.Event()
    gpu.marker = tmp_path / "deep" / "pifilm" / "gpu-opening"
    session = gpu.make(state_dir=tmp_path / "deep" / "pifilm" / "pending")
    assert gpu.entered.wait(5)
    assert gpu.marker.is_file()
    gpu.hold.set()
    assert session.close_grades(timeout=5)
    assert not gpu.marker.exists()


def test_a_marker_that_cannot_be_written_does_not_stop_the_open(tmp_path, gpu):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    session = gpu.make(state_dir=blocker / "pending")
    gpu.opened()
    session.capture()
    assert session.close_grades(timeout=5)
    assert gpu.threads == ["pifilm-grade"]
    assert "gpu grading on (Fake GPU, self-test passed)" in session.test_log


@pytest.mark.parametrize("kwargs", [
    {"grade_backend": "cpu"},
    {"grade_backend": "cpu", "background": False},
    {"grade_backend": "auto", "background": False},
])
def test_the_cpu_backend_and_inline_sessions_write_no_marker(gpu, monkeypatch, kwargs):
    written = []
    real_write = Path.write_text

    def spy(self, *args, **kw):
        written.append(self.name)
        return real_write(self, *args, **kw)

    monkeypatch.setattr(Path, "write_text", spy)
    session = gpu.make(**kwargs)
    session.capture()
    assert session.close_grades(timeout=5)
    assert "gpu-opening" not in written and not gpu.marker.exists()
    # And a marker already there is nothing to them: no line about the GPU.
    gpu.marker.write_text("stale\n")
    other = gpu.make(**kwargs)
    assert other.close_grades(timeout=5)
    assert other.test_log == []


def test_a_frame_wider_than_the_largest_texture_goes_to_the_cpu_quietly(tmp_path, gpu):
    gpu.max_texture = 256  # the tile (256 px) and the self-test fit; the frame does not
    frame = synthetic_frame(180, 320)
    session = gpu.make(FakeCamera([frame]))
    device = gpu.opened()
    results = [session.capture() for _ in range(3)]
    assert session.close_grades(timeout=5)
    records = _lines(session.out_root)
    assert [r["pipeline_backend"] for r in records] == ["cpu"] * 3
    for result, record in zip(results, records, strict=True):
        assert result.graded.read_bytes() == _cpu_bytes(tmp_path, record["grain_seed"], frame)
    assert len(device.runs) == 1  # the self-test only
    assert session.test_log == ["gpu grading on (Fake GPU, self-test passed)"]


# The tuning check (pifilm.tuning): once at start-up, warn only.

_LOOK_TUNING = "1" * 40
_OTHER_TUNING = "2" * 40


class _TunedCamera(FakeCamera):
    """A FakeCamera that, like Picamera2Camera, reports the tuning it opened with."""

    def __init__(self, sha1, frames=None, label="imx477.json"):
        super().__init__(frames or [synthetic_frame(90, 160)])
        self.tuning = Resolved(label=label, source=label, sha1=sha1,
                               reason=None if sha1 else "cannot load imx477.json")


def _tuned_pipeline(tmp_path, sha1=_LOOK_TUNING):
    d = tmp_path / "tuned-art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams(), tuning_sha1=sha1)
    return Pipeline(Artifacts.load(d))


def _tuned_session(tmp_path, camera, pipeline, **kwargs):
    log = []
    session = CaptureSession(camera, pipeline, tmp_path / "shots",
                             seed_rng=np.random.default_rng(0), log=log.append, **kwargs)
    session.test_log = log
    return session


def test_a_matching_tuning_is_one_line_and_marks_every_record(tmp_path):
    session = _tuned_session(tmp_path, _TunedCamera(_LOOK_TUNING), _tuned_pipeline(tmp_path))
    assert session.test_log == [f"tuning: matches the look ({_LOOK_TUNING[:12]})"]
    assert session.tuning_mismatch is False
    result = session.capture()
    assert result.record["tuning_matches"] is True
    (record,) = _lines(session.out_root)
    assert record["tuning_matches"] is True


def test_a_different_tuning_warns_and_the_photograph_is_still_graded(tmp_path):
    session = _tuned_session(tmp_path, _TunedCamera(_OTHER_TUNING), _tuned_pipeline(tmp_path))
    assert session.test_log == [
        f"tuning: DIFFERS from the look: camera {_OTHER_TUNING[:12]} (imx477.json), "
        f"look {_LOOK_TUNING[:12]}; colours may be off"
    ]
    assert session.tuning_mismatch is True
    result = session.capture()
    assert result.pifilm.exists() and result.original.exists()
    (record,) = _lines(session.out_root)
    assert record["tuning_matches"] is False


def test_tuning_matches_sits_right_after_the_camera_fields(tmp_path):
    camera = _TunedCamera(_LOOK_TUNING)
    camera._info = StreamInfo(
        width=160, height=90, fps=1.0, fourcc="RGB888", raw_mjpeg=False,
        tuning_file="imx477.json", tuning_sha1=_LOOK_TUNING,
    )
    session = _tuned_session(tmp_path, camera, _tuned_pipeline(tmp_path))
    keys = list(session.capture().record)
    assert keys.index("tuning_matches") == keys.index("tuning_sha1") + 1


def test_a_look_without_a_fingerprint_checks_nothing(tmp_path, pipeline):
    session = _tuned_session(tmp_path, _TunedCamera(_LOOK_TUNING), pipeline)
    assert session.test_log == []
    assert session.tuning_mismatch is False
    assert "tuning_matches" not in session.capture().record


def test_a_camera_without_a_tuning_checks_nothing(tmp_path):
    session = _tuned_session(tmp_path, FakeCamera([synthetic_frame(90, 160)]),
                             _tuned_pipeline(tmp_path))
    assert session.test_log == []
    assert session.tuning_mismatch is False
    assert "tuning_matches" not in session.capture().record


def test_a_tuning_that_could_not_be_fingerprinted_says_it_cannot_check(tmp_path):
    session = _tuned_session(tmp_path, _TunedCamera(None), _tuned_pipeline(tmp_path))
    assert session.test_log == [
        "tuning: cannot check the look's tuning (cannot load imx477.json)"
    ]
    assert session.tuning_mismatch is False
    assert "tuning_matches" not in session.capture().record


def test_every_session_has_a_tuning_mismatch_flag(tmp_path, pipeline):
    assert _session(tmp_path, pipeline).tuning_mismatch is False


def test_a_background_record_carries_tuning_matches(tmp_path):
    session = _tuned_session(tmp_path, _TunedCamera(_OTHER_TUNING), _tuned_pipeline(tmp_path),
                             background=True, state_dir=tmp_path / "state")
    result = session.capture()
    assert result.record["tuning_matches"] is False
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    assert record["tuning_matches"] is False
    assert record["grade"] == "background"


def test_a_recovered_grade_keeps_the_tuning_of_the_run_that_shot_it(tmp_path):
    tuned = _tuned_pipeline(tmp_path)
    held = _HeldPipeline(tuned)
    held.artifacts = tuned.artifacts
    first = _tuned_session(tmp_path, _TunedCamera(_LOOK_TUNING), held,
                           background=True, state_dir=tmp_path / "state")
    result = first.capture()
    held.wait_entered()
    ((path, payload),) = read_entries(tmp_path / "state")
    assert payload["record"]["tuning_matches"] is True  # stored at capture time
    text = path.read_text()
    held.release()
    assert first.close_grades(timeout=5)
    # Put the shot back as a power cut would have left it.
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    again = _tuned_session(tmp_path, _TunedCamera(_OTHER_TUNING), tuned,
                           background=True, state_dir=tmp_path / "state")
    assert again.tuning_mismatch is True
    _settle(again)
    assert again.close_grades(timeout=5)
    (recovered,) = _lines(again.out_root)
    assert recovered["recovered"] is True
    assert recovered["tuning_matches"] is True


@pytest.mark.parametrize("background", [False, True])
def test_a_double_exposure_carries_tuning_matches(tmp_path, background):
    frames = [synthetic_frame(90, 160), synthetic_frame(90, 160)[:, ::-1].copy()]
    kwargs = {"background": True, "state_dir": tmp_path / "state"} if background else {}
    session = _tuned_session(tmp_path, _TunedCamera(_LOOK_TUNING, frames),
                             _tuned_pipeline(tmp_path), **kwargs)
    session.set_double_exposure(True)
    session.capture()
    session.capture()
    assert session.close_grades(timeout=5)
    records = _lines(session.out_root)
    assert [r["double"]["index"] for r in records] == [1, 2]
    assert [r["tuning_matches"] for r in records] == [True, True]


# The disk-space warning (pifilm.capture.disk): the session holds it, main runs it.


def _full_vfs(percent):
    blocks = 1_000_000
    free = blocks - blocks * percent // 100
    return SimpleNamespace(f_blocks=blocks, f_bfree=free, f_bavail=free, f_frsize=4096)


def test_a_session_without_a_disk_monitor_has_no_disk_warning(tmp_path, pipeline):
    assert _session(tmp_path, pipeline).disk_warning == ()


def test_the_sessions_disk_monitor_watches_its_photo_folder_and_logs_through_it(
        tmp_path, pipeline):
    session = _tuned_session(tmp_path, FakeCamera([synthetic_frame(90, 160)]), pipeline)
    session.out_root.mkdir(parents=True)
    devices = {str(session.out_root): 2, "/": 1}
    full = {str(session.out_root): 93, "/": 50}
    monitor = session.watch_disk(
        90, statvfs=lambda p: _full_vfs(full[str(p)]),
        stat=lambda p: SimpleNamespace(st_dev=devices[str(p)]))
    assert session.disk_warning == ()  # nothing measured until it runs
    monitor.poll_once()
    assert session.disk_warning == (("PHOTOS", 93),)
    assert session.test_log[-1].startswith("disk: photos 93% full (")
    monitor.close()


def _disk_main(monkeypatch, tmp_path, *extra):
    from pifilm.capture import app

    made = []

    class FakeDisk:
        def __init__(self, threshold):
            self.threshold = threshold
            self.started = self.closed = False

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    def watch_disk(self, threshold):
        made.append(FakeDisk(threshold))
        return made[-1]

    monkeypatch.setattr(app.CaptureSession, "watch_disk", watch_disk)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = app.main(["--fake", "--no-preview", "--out", str(tmp_path), *extra])
    return code, made


def test_main_runs_the_disk_monitor_at_90_percent_and_stops_it(tmp_path, monkeypatch):
    code, made = _disk_main(monkeypatch, tmp_path)
    assert code == 2  # no terminal; the clean-up still runs
    (disk,) = made
    assert disk.threshold == 90 and disk.started and disk.closed


def test_disk_warn_percent_sets_the_threshold(tmp_path, monkeypatch):
    _, made = _disk_main(monkeypatch, tmp_path, "--disk-warn-percent", "40")
    assert made[0].threshold == 40


@pytest.mark.parametrize("value", ["0", "101", "-5", "abc", "90.5"])
def test_disk_warn_percent_outside_1_to_100_is_refused(tmp_path, monkeypatch, capsys, value):
    with pytest.raises(SystemExit) as exc:
        _disk_main(monkeypatch, tmp_path, "--disk-warn-percent", value)
    assert exc.value.code == 2
    assert "--disk-warn-percent" in capsys.readouterr().err


# The network addresses (pifilm.capture.network): the session holds them, main runs it.


def test_a_session_without_a_network_monitor_has_no_addresses(tmp_path, pipeline):
    assert _session(tmp_path, pipeline).network == ()


def test_the_sessions_network_monitor_logs_through_it(tmp_path, pipeline):
    session = _tuned_session(tmp_path, FakeCamera([synthetic_frame(90, 160)]), pipeline)
    monitor = session.watch_network(read=lambda: (("WLAN", "10.42.0.1"),))
    assert session.network == ()  # nothing read until it runs
    monitor.poll_once()
    assert session.network == (("WLAN", "10.42.0.1"),)
    assert session.test_log[-1] == "network: wlan0 10.42.0.1"
    monitor.close()


def test_main_runs_the_network_monitor_and_stops_it(tmp_path, monkeypatch):
    from pifilm.capture import app

    made = []

    class FakeNetwork:
        def __init__(self):
            self.started = self.closed = False

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    def watch_network(self):
        made.append(FakeNetwork())
        return made[-1]

    monkeypatch.setattr(app.CaptureSession, "watch_network", watch_network)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = app.main(["--fake", "--no-preview", "--out", str(tmp_path)])
    assert code == 2  # no terminal; the clean-up still runs
    (network,) = made
    assert network.started and network.closed


def test_session_save_dng_reaches_the_camera(tmp_path, pipeline):
    camera = FakeCamera([synthetic_frame(90, 160)])
    calls = []
    camera.set_save_dng = calls.append
    session = _session(tmp_path, pipeline, camera=camera)
    assert session.save_dng is True
    session.save_dng = False
    assert session.save_dng is False
    assert calls == [False]


def test_session_save_dng_tolerates_a_camera_without_the_setter(tmp_path, pipeline):
    camera = FakeCamera([synthetic_frame(90, 160)])
    assert not hasattr(camera, "set_save_dng")
    session = _session(tmp_path, pipeline, camera=camera)
    session.save_dng = False
    assert session.save_dng is False


# The settings menu wired into ``main``.


def _save_settings(**values):
    """Write a settings.json beside the default state directory (XDG_STATE_HOME)."""
    from pifilm.capture.gradequeue import default_state_dir
    from pifilm.capture.settings import SETTINGS_FILE, save

    path = default_state_dir().parent / SETTINGS_FILE
    save(path, values)
    return path


class _MenuCamera(_CameraDouble):
    """A Picamera2 double with the AE setters and what they support."""

    def __init__(self, *, constraint=True, metering=True, adjustable=True):
        super().__init__()
        self.ae_constraint_supported = constraint
        self.ae_metering_supported = metering
        self.exposure_adjustable = adjustable
        self.calls = []

    def set_ae_constraint(self, name):
        self.calls.append(("ae_constraint", name))

    def set_ae_metering(self, name):
        self.calls.append(("ae_metering", name))

    def set_save_dng(self, on):
        self.calls.append(("save_dng", on))


def _menu_main(camera_cli, monkeypatch, tmp_path, camera, *extra):
    """Run main on a Picamera2 double with a panel; return what ViewfinderLoop got."""
    seen = {}
    pair = (SimpleNamespace(close=lambda: None), SimpleNamespace(close=lambda: None))

    def open_picamera(tuning_file, **kwargs):
        seen["camera_kwargs"] = kwargs
        return camera

    class LoopDouble:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def run(self, stop):
            return None

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: pair)
    monkeypatch.setattr(camera_cli, "ViewfinderLoop", LoopDouble)
    argv = ["--camera", "picamera2", "--display", "waveshare28", "--no-preview",
            "--out", str(tmp_path / "shots"), *extra]
    assert camera_cli.main(argv) == 0
    return seen


def test_saved_ae_values_with_fake_do_not_trip_the_picamera2_only_check(
    camera_cli, capsys,
):
    _save_settings(ae_constraint="shadows", ae_metering="spot")
    assert camera_cli.main(["--fake", "--no-preview"]) == 0
    out = capsys.readouterr()
    assert "cannot be used with --fake" not in out.err


def test_saved_values_reach_the_picamera2_constructor_and_the_session(
    camera_cli, monkeypatch, tmp_path, capsys,
):
    _save_settings(ae_constraint="highlight", ae_metering="matrix", save_dng=False)
    camera = _MenuCamera()
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, camera)
    kwargs = seen["camera_kwargs"]
    assert kwargs["ae_constraint"] == "highlight"
    assert kwargs["ae_metering"] == "matrix"
    assert kwargs["save_dng"] is False
    assert seen["settings"].current.save_dng is False
    out = capsys.readouterr().out
    assert ("settings: dng off, ae-constraint highlight, ae-metering matrix "
            "(saved in the menu; the unit says dng on, ae-constraint normal, "
            "ae-metering centre)") in out


def test_the_menu_appliers_reach_the_camera_and_the_session(camera_cli, monkeypatch, tmp_path):
    camera = _MenuCamera()
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, camera)
    control = seen["settings"]
    assert control.set("ae_constraint", "shadows")
    assert control.set("ae_metering", "spot")
    assert control.set("save_dng", False)  # through the session, which tells the camera
    assert camera.calls == [
        ("ae_constraint", "shadows"), ("ae_metering", "spot"), ("save_dng", False),
    ]
    assert seen["menu_rows"] == ("save_dng", "halation", "ae_constraint", "ae_metering")


def test_menu_rows_leave_out_what_the_camera_cannot_use(camera_cli, monkeypatch, tmp_path):
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _MenuCamera(constraint=False))
    assert seen["menu_rows"] == ("save_dng", "halation", "ae_metering")


def test_menu_rows_leave_out_the_ae_rows_under_ae_lock(camera_cli, monkeypatch, tmp_path):
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _MenuCamera(adjustable=False),
                      "--ae-lock")
    assert seen["menu_rows"] == ("save_dng", "halation")


def test_fake_has_no_optional_menu_rows_and_an_empty_look(tmp_path, monkeypatch):
    kwargs = _viewfinder_kwargs(monkeypatch, tmp_path, [])
    assert kwargs["menu_rows"] is None
    assert kwargs["look"] == ""
    assert kwargs["upload"] is None
    assert kwargs["settings"] is not None


def test_the_look_is_the_artifacts_folder_name(tmp_path, monkeypatch):
    kwargs = _viewfinder_kwargs(
        monkeypatch, tmp_path, ["--artifacts", str(tmp_path / "imx477-v4-dark")],
    )
    assert kwargs["look"] == "imx477-v4-dark"


def test_the_bundled_starter_named_by_its_path_has_no_look_name(tmp_path, monkeypatch):
    # The shipped unit passes --artifacts .../pifilm/data; the row must not say "data".
    kwargs = _viewfinder_kwargs(
        monkeypatch, tmp_path, ["--artifacts", str(tmp_path / "pifilm" / "data")],
    )
    assert kwargs["look"] == ""


def test_nextcloud_env_with_a_display_makes_an_upload_closed_on_exit(tmp_path, monkeypatch):
    from pifilm.capture import app

    made = []

    class FakeUpload:
        def __init__(self, env_path, *, log=print):
            self.env_path, self.closed = env_path, False
            made.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(app, "Upload", FakeUpload)
    kwargs = _viewfinder_kwargs(
        monkeypatch, tmp_path, ["--nextcloud-env", str(tmp_path / "nc.env")],
    )
    (upload,) = made
    assert kwargs["upload"] is upload
    assert upload.env_path == tmp_path / "nc.env"
    assert upload.closed


def test_nextcloud_env_without_a_display_makes_no_upload(tmp_path, monkeypatch):
    from pifilm.capture import app

    made = []
    monkeypatch.setattr(app, "Upload", lambda *a, **k: made.append(1))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    app.main(["--fake", "--no-preview", "--out", str(tmp_path),
              "--nextcloud-env", str(tmp_path / "nc.env")])
    assert made == []


# --- halation in the records ---------------------------------------------------


def _halation_pipeline(enabled=True):
    look = dataclasses.replace(Artifacts.default(), halation=HalationParams(enabled=enabled))
    return Pipeline(look)


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_a_halation_look_names_its_halation_in_the_record(tmp_path, grade):
    pipe = _halation_pipeline()
    session = CaptureSession(FakeCamera([synthetic_frame(90, 160)]), pipe, tmp_path / "shots",
                             now=lambda: FIXED, seed_rng=np.random.default_rng(0),
                             background=grade == "background", state_dir=tmp_path / "state",
                             log=lambda _: None)
    session.capture()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    keys = list(record)
    assert record["halation_sha1"] == pipe.artifacts.halation.sha1
    assert keys.index("halation_sha1") == keys.index("normalize_sha1") + 1
    assert "halation" in record["stage_ms"]
    stages = list(record["stage_ms"])
    assert stages.index("normalize") < stages.index("halation") < stages.index("lut")


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_a_look_without_halation_records_none(tmp_path, grade):
    session = CaptureSession(FakeCamera([synthetic_frame(90, 160)]),
                             _halation_pipeline(enabled=False), tmp_path / "shots",
                             now=lambda: FIXED, seed_rng=np.random.default_rng(0),
                             background=grade == "background", state_dir=tmp_path / "state",
                             log=lambda _: None)
    session.capture()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    assert "halation_sha1" not in record and "halation" not in record["stage_ms"]


# --- the menu's HALATION, taken at the shutter ---------------------------------


class _SpyPipeline(_HeldPipeline):
    """A held pipeline that also records what each call asked of halation."""

    def __init__(self, inner):
        super().__init__(inner)
        self.artifacts = inner.artifacts
        self.calls = []  # (full grade?, halation asked)

    def process(self, rgb, *, grain=True, halation=True, **kwargs):
        self.calls.append((grain, halation))
        return super().process(rgb, grain=grain, halation=halation, **kwargs)

    def warm(self, shape):
        return None


def _halation_session(tmp_path, pipe, *, grade="background", frames=1, **kwargs):
    camera = FakeCamera([synthetic_frame(90, 160) for _ in range(frames)])
    return CaptureSession(camera, pipe, tmp_path / "shots", now=lambda: FIXED,
                          seed_rng=np.random.default_rng(0),
                          background=grade == "background", state_dir=tmp_path / "state",
                          log=lambda _: None, **kwargs)


def _off_record_checks(record, pipe):
    keys = list(record)
    assert "halation_sha1" not in record and record["halation"] == "off"
    assert keys.index("halation") == keys.index("normalize_sha1") + 1
    assert "halation" not in record["stage_ms"]
    assert pipe.artifacts.halation.active


def test_halation_off_at_the_shutter_holds_when_the_setting_flips_before_the_grade(tmp_path):
    spy = _SpyPipeline(_halation_pipeline())
    session = _halation_session(tmp_path, spy, halation=False)
    result = session.capture()
    spy.wait_entered()
    session.halation = True  # the row tapped again while the grade waits
    ((_, payload),) = read_entries(tmp_path / "state")
    assert payload["halation"] is False
    spy.release()
    assert session.close_grades(timeout=5)
    # The quick grade and the full grade both left it out.
    assert spy.calls == [(False, False), (True, False)]
    (record,) = _lines(session.out_root)
    _off_record_checks(record, spy)
    # The file is the inline grade with halation off.
    inline = _halation_session(tmp_path / "inline", _halation_pipeline(), grade="inline",
                               halation=False).capture()
    assert result.graded.read_bytes() == inline.pifilm.read_bytes()


def test_halation_on_at_the_shutter_holds_when_the_setting_flips_off(tmp_path):
    spy = _SpyPipeline(_halation_pipeline())
    session = _halation_session(tmp_path, spy)
    session.capture()
    spy.wait_entered()
    session.halation = False
    ((_, payload),) = read_entries(tmp_path / "state")
    assert payload["halation"] is True
    spy.release()
    assert session.close_grades(timeout=5)
    assert spy.calls == [(False, True), (True, True)]
    (record,) = _lines(session.out_root)
    assert record["halation_sha1"] == spy.artifacts.halation.sha1 and "halation" not in record


def test_halation_off_inline(tmp_path):
    pipe = _halation_pipeline()
    session = _halation_session(tmp_path, pipe, grade="inline", halation=False)
    session.capture()
    (record,) = _lines(session.out_root)
    _off_record_checks(record, pipe)


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_halation_off_for_a_look_without_halation_changes_no_record(tmp_path, grade):
    records = []
    for halation in (True, False):
        session = _halation_session(tmp_path / str(halation), _halation_pipeline(enabled=False),
                                    grade=grade, halation=halation)
        session.capture()
        assert session.close_grades(timeout=5)
        (record,) = _lines(session.out_root)
        assert "halation_sha1" not in record and "halation" not in record
        records.append(record)
    timing = {"pipeline_ms", "shutter_to_saved_ms", "shutter_to_ready_ms", "stage_ms"}
    on, off = ({k: v for k, v in r.items() if k not in timing} for r in records)
    assert list(on) == list(off) and on == off


@pytest.mark.parametrize(("stored", "applied"), [(None, True), (True, True), (False, False)])
def test_a_recovered_grade_uses_the_halation_of_its_entry(tmp_path, stored, applied):
    spy = _SpyPipeline(_halation_pipeline())
    first = _halation_session(tmp_path, spy)
    result = first.capture()
    spy.wait_entered()
    ((path, payload),) = read_entries(tmp_path / "state")
    spy.release()
    assert first.close_grades(timeout=5)
    # Put the shot back as a power cut would have left it, the entry as given.
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    if stored is None:
        del payload["halation"]  # an entry written before the setting existed
    else:
        payload["halation"] = stored
    path.write_text(json.dumps(payload))

    pipe = _halation_pipeline()
    again = _halation_session(tmp_path, pipe, halation=not applied)  # the menu now differs
    _settle(again)
    assert again.close_grades(timeout=5)
    (record,) = _lines(again.out_root)
    assert record["recovered"] is True
    if applied:
        assert record["halation_sha1"] == pipe.artifacts.halation.sha1
        assert "halation" not in record
    else:
        _off_record_checks(record, pipe)


def test_a_recovered_entry_with_a_bad_halation_fails(tmp_path):
    from pifilm.capture.app import _GradeWork
    payload = {"day": "2026-09-03", "graded": "x_graded.jpg", "originals": ["a/b.jpg"],
               "seed": 1, "ev": 0.0, "double": None, "record": {}, "halation": "off"}
    with pytest.raises(ValueError, match="halation"):
        _GradeWork.from_entry(payload)
    del payload["halation"]
    assert _GradeWork.from_entry(payload).halation is True


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_a_double_takes_halation_from_its_second_shutter(tmp_path, grade):
    spy = _SpyPipeline(_halation_pipeline())
    spy.release(10)
    session = _halation_session(tmp_path, spy, grade=grade, frames=2)
    session.set_double_exposure(True)
    session.capture()  # exposure 1, with halation on
    session.halation = False
    session.capture()  # exposure 2 makes the photograph
    assert session.close_grades(timeout=5)
    assert all(halation is False for _, halation in spy.calls)
    assert spy.calls[-1] == (True, False)
    record = _lines(session.out_root)[-1]
    assert record["double"]["index"] == 2
    _off_record_checks(record, spy)


def test_the_live_view_has_no_halation_whatever_the_menu_says(tmp_path):
    spy = _SpyPipeline(_halation_pipeline())
    session = _halation_session(tmp_path, spy, grade="inline")
    session.preview_frame()
    assert spy.calls == [(False, False)]


def test_the_halation_row_is_offered_and_its_applier_reaches_the_session(
    camera_cli, monkeypatch, tmp_path,
):
    made = []
    real = camera_cli.CaptureSession

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(camera_cli, "CaptureSession", recording)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _MenuCamera())
    assert seen["menu_rows"][:2] == ("save_dng", "halation")
    assert seen["look_halation"] is False  # camera_cli's pipeline has no look
    (session,) = made
    assert session.halation is True
    assert seen["settings"].set("halation", False)
    assert session.halation is False
    seen["settings"].reset()
    assert session.halation is True


def test_a_saved_halation_off_reaches_the_session(camera_cli, monkeypatch, tmp_path, capsys):
    _save_settings(halation=False)
    made = []
    real = camera_cli.CaptureSession

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(camera_cli, "CaptureSession", recording)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _MenuCamera())
    assert made[0].halation is False and seen["settings"].current.halation is False
    assert ("settings: halation off (saved in the menu; the unit says halation on)"
            in capsys.readouterr().out)


def test_a_saved_halation_off_is_not_taken_under_fake(camera_cli, monkeypatch, tmp_path):
    _save_settings(halation=False)
    made = []
    real = camera_cli.CaptureSession

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(camera_cli, "CaptureSession", recording)
    assert camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path / "s")]) == 0
    assert made[0].halation is True


@pytest.mark.parametrize("enabled", [True, False])
def test_the_viewfinder_is_told_whether_the_look_has_halation(tmp_path, monkeypatch, enabled):
    from pifilm.capture import app

    look = dataclasses.replace(Artifacts.default(), halation=HalationParams(enabled=enabled))
    monkeypatch.setattr(app.Artifacts, "resolve", lambda _: look)
    seen = {}

    class LoopDouble:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def run(self, stop):
            return None

    monkeypatch.setattr(app, "ViewfinderLoop", LoopDouble)
    out = tmp_path / "shots"
    assert app.main(["--fake", "--display", "fake", "--no-preview", "--out", str(out)]) == 0
    assert seen["look_halation"] is enabled


def _image_info(sharpness, contrast, saturation):
    """A Picamera2-like stream_info with the ISP's image controls in it."""
    return StreamInfo(width=160, height=90, fps=0.0, fourcc="RGB888", raw_mjpeg=False,
                      ae_constraint="normal", ae_metering="centre",
                      sharpness=sharpness, contrast=contrast, saturation=saturation)


def test_a_recovered_grade_keeps_the_image_values_of_the_run_that_shot_it(tmp_path):
    tuned = _tuned_pipeline(tmp_path)
    held = _HeldPipeline(tuned)
    held.artifacts = tuned.artifacts
    camera = FakeCamera([synthetic_frame(90, 160)], stream_info=_image_info(1.5, 1.0, 0.75))
    first = _tuned_session(tmp_path, camera, held, background=True,
                           state_dir=tmp_path / "state")
    result = first.capture()
    held.wait_entered()
    ((path, payload),) = read_entries(tmp_path / "state")
    stored = payload["record"]
    assert (stored["sharpness"], stored["contrast"], stored["saturation"]) == (1.5, 1.0, 0.75)
    text = path.read_text()
    held.release()
    assert first.close_grades(timeout=5)
    # Put the shot back as a power cut would have left it.
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    camera = FakeCamera([synthetic_frame(90, 160)], stream_info=_image_info(1.0, 0.5, 1.0))
    again = _tuned_session(tmp_path, camera, tuned, background=True,
                           state_dir=tmp_path / "state")
    _settle(again)
    assert again.close_grades(timeout=5)
    (recovered,) = _lines(again.out_root)
    assert recovered["recovered"] is True
    assert (recovered["sharpness"], recovered["contrast"], recovered["saturation"]) == (
        1.5, 1.0, 0.75,
    )
    keys = list(recovered)
    at = keys.index("ae_metering")
    assert keys[at + 1:at + 4] == ["sharpness", "contrast", "saturation"]


class _ImageMenuCamera(_MenuCamera):
    """A Picamera2 double with the image-control setter."""

    def __init__(self, supported=("sharpness", "contrast", "saturation"), **kwargs):
        super().__init__(**kwargs)
        self.image_controls_supported = frozenset(supported)

    def set_image_control(self, key, value):
        self.calls.append((key, value))


def test_saved_image_values_reach_the_picamera2_constructor(
    camera_cli, monkeypatch, tmp_path, capsys,
):
    _save_settings(contrast=1.25, saturation=0.5)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _ImageMenuCamera())
    kwargs = seen["camera_kwargs"]
    assert (kwargs["sharpness"], kwargs["contrast"], kwargs["saturation"]) == (1.0, 1.25, 0.5)
    assert ("settings: contrast 1.25, saturation 0.5 (saved in the menu; the unit says "
            "contrast 1.0, saturation 1.0)") in capsys.readouterr().out


def test_the_image_appliers_reach_the_camera_and_the_rows_follow_support(
    camera_cli, monkeypatch, tmp_path,
):
    camera = _ImageMenuCamera(supported=("sharpness", "saturation"))
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, camera)
    control = seen["settings"]
    assert control.set("sharpness", 1.5)
    assert control.set("saturation", 0.75)
    assert camera.calls == [("sharpness", 1.5), ("saturation", 0.75)]
    assert seen["menu_rows"] == (
        "save_dng", "halation", "ae_constraint", "ae_metering", "sharpness", "saturation",
    )


def test_saved_image_values_do_not_reach_a_fake_camera(tmp_path, monkeypatch):
    _save_settings(contrast=1.25)
    kwargs = _viewfinder_kwargs(monkeypatch, tmp_path, [])
    assert kwargs["menu_rows"] is None


class _StabiliseMenuCamera(_MenuCamera):
    """A Picamera2 double with the short-exposure setter."""

    def set_short_exposure(self, on):
        self.calls.append(("short_exposure", on))


def test_the_short_exposure_flag_defaults_to_off(camera_cli, monkeypatch, tmp_path):
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _StabiliseMenuCamera())
    assert seen["camera_kwargs"]["short_exposure"] is False


def test_a_saved_short_exposure_reaches_the_picamera2_constructor(
    camera_cli, monkeypatch, tmp_path, capsys,
):
    _save_settings(short_exposure=True, shutter_delay=0.5, multi_frame=4)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _StabiliseMenuCamera())
    assert seen["camera_kwargs"]["short_exposure"] is True
    current = seen["settings"].current
    assert (current.short_exposure, current.shutter_delay, current.multi_frame) == (
        True, 0.5, 4,
    )
    assert ("settings: short-exposure on, shutter-delay 0.5, multi-frame 4 (saved in the "
            "menu; the unit says short-exposure off, shutter-delay 0.0, multi-frame 1)"
            ) in capsys.readouterr().out


def test_the_short_exposure_applier_reaches_the_camera(camera_cli, monkeypatch, tmp_path):
    camera = _StabiliseMenuCamera()
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, camera)
    control = seen["settings"]
    assert control.set("short_exposure", True)
    # Not the camera's: the delay is read at each press, multi-frame is the session's.
    assert control.set("shutter_delay", 0.2)
    assert control.set("multi_frame", 2)
    assert camera.calls == [("short_exposure", True)]


def test_saved_stabilise_values_with_fake_start_as_usual(camera_cli, capsys):
    _save_settings(short_exposure=True, multi_frame=6)
    assert camera_cli.main(["--fake", "--no-preview"]) == 0
    assert "cannot be used with --fake" not in capsys.readouterr().err


# -- the shutter delay -------------------------------------------------------


def _delay_camera(source="picamera2"):
    return FakeCamera([synthetic_frame(90, 160)], source=source,
                      stream_info=_image_info(1.0, 1.0, 1.0))


def _after_stream(record):
    keys = list(record)
    return keys[keys.index("saturation") + 1]


def test_an_inline_record_carries_the_delay_after_the_stream_fields(tmp_path, pipeline):
    session = _session(tmp_path, pipeline, camera=_delay_camera())
    result = session.capture(delay_s=0.5)
    (record,) = _lines(session.out_root)
    assert record["shutter_delay_s"] == 0.5 and result.record["shutter_delay_s"] == 0.5
    assert _after_stream(record) == "shutter_delay_s"
    session.capture()
    assert _lines(session.out_root)[-1]["shutter_delay_s"] == 0.0


@pytest.mark.parametrize("source", ["fake", "raw-mjpeg", "decoded"])
def test_a_record_off_picamera2_has_no_delay(tmp_path, pipeline, source):
    session = _session(tmp_path, pipeline, camera=_delay_camera(source))
    session.capture(delay_s=0.2)
    assert "shutter_delay_s" not in _lines(session.out_root)[0]


def test_a_double_exposure_records_each_exposures_delay(tmp_path, pipeline):
    session = _session(tmp_path, pipeline, camera=_delay_camera())
    session.set_double_exposure(True)
    session.capture(delay_s=0.2)
    session.capture(delay_s=0.5)
    first, second = _lines(session.out_root)
    assert (first["shutter_delay_s"], second["shutter_delay_s"]) == (0.2, 0.5)
    assert _after_stream(first) == _after_stream(second) == "shutter_delay_s"


def test_a_background_and_a_recovered_record_carry_the_delay(tmp_path):
    tuned = _tuned_pipeline(tmp_path)
    held = _HeldPipeline(tuned)
    held.artifacts = tuned.artifacts
    first = _tuned_session(tmp_path, _delay_camera(), held, background=True,
                           state_dir=tmp_path / "state")
    result = first.capture(delay_s=0.2)
    assert result.record["shutter_delay_s"] == 0.2
    held.wait_entered()
    ((path, payload),) = read_entries(tmp_path / "state")
    assert payload["record"]["shutter_delay_s"] == 0.2
    text = path.read_text()
    held.release()
    assert first.close_grades(timeout=5)
    (finished,) = _lines(first.out_root)
    assert finished["shutter_delay_s"] == 0.2 and _after_stream(finished) == "shutter_delay_s"
    # Put the shot back as a power cut would have left it.
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    again = _tuned_session(tmp_path, _delay_camera(), tuned, background=True,
                           state_dir=tmp_path / "state")
    _settle(again)
    assert again.close_grades(timeout=5)
    (recovered,) = _lines(again.out_root)
    assert recovered["recovered"] is True
    assert recovered["shutter_delay_s"] == 0.2 and _after_stream(recovered) == "shutter_delay_s"


def _press_delay(camera_cli, monkeypatch):
    """Point ``_button_capture`` at a recorder; return what each press asked for."""
    asked = []
    monkeypatch.setattr(camera_cli, "_button_capture",
                        lambda controller, delay: asked.append(delay()))
    return asked


def test_the_button_reads_the_saved_delay_at_each_press_on_picamera2(
    camera_cli, monkeypatch, tmp_path,
):
    _save_settings(shutter_delay=0.5)
    button = _ButtonDouble()
    monkeypatch.setattr(camera_cli, "open_shutter_button", lambda pin: button)
    asked = _press_delay(camera_cli, monkeypatch)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _StabiliseMenuCamera(),
                      "--shutter-gpio", "21")
    button.callback()
    assert seen["settings"].set("shutter_delay", 0.2)
    button.callback()
    assert asked == [0.5, 0.2]


def test_the_button_never_delays_under_fake(camera_cli, monkeypatch):
    _save_settings(shutter_delay=0.5)
    button = _ButtonDouble()
    monkeypatch.setattr(camera_cli, "open_shutter_button", lambda pin: button)
    asked = _press_delay(camera_cli, monkeypatch)
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    button.callback()
    assert asked == [0.0]


# -- multi-frame photographs -------------------------------------------------


def _textured(seed, noise=0):
    """A 90 x 160 frame with enough texture for ECC; ``noise`` adds a frame's own grain."""
    from pifilm._cv2 import require_cv2

    rng = np.random.default_rng(1)
    base = require_cv2().GaussianBlur(rng.uniform(0, 255, (90, 160, 3)).astype(np.float32),
                                      (0, 0), 3)
    base = (base - base.min()) / (base.max() - base.min()) * 200 + 25
    if noise:
        base = base + np.random.default_rng(seed).normal(0, noise, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


class _BurstCamera(FakeCamera):
    """A Picamera2-like double whose bursts carry per-frame FocusFoM, exposure and
    gain, and a DNG on every frame (which a multi-frame shot must not save)."""

    def __init__(self, foms=(1.0, 5.0, 3.0, 2.0), *, split=True):
        self.burst_frames = [_textured(i, noise=4) for i in range(len(foms))]
        super().__init__([synthetic_frame(90, 160)], source="picamera2",
                         stream_info=_image_info(1.0, 1.0, 1.0))
        self.foms = foms
        self.split = split
        self.bursts = []
        self.reads = 0

    def read(self, *, full=True):
        self.reads += 1
        frame = super().read(full=full)
        frame.dng = b"raw"
        return frame

    def read_burst(self, n, *, split_exposure, dng=True):
        self.bursts.append((n, split_exposure, dng))
        frames = [
            Frame(self.burst_frames[i % len(self.burst_frames)].copy(), None, "picamera2",
                  metadata={"FocusFoM": self.foms[i % len(self.foms)],
                            "ExposureTime": 2500 + i, "AnalogueGain": 4.0 + i},
                  dng=b"raw")
            for i in range(n)
        ]
        info = {"exposure_us": 2500, "gain": 4.0, "split": self.split}
        return frames, info


STACK_KEYS = ["frames", "reference", "exposure_us", "gain", "requested_exposure_us",
              "requested_gain", "split", "method", "aligned", "align_ms", "merge_ms"]
STACK_NAMES = ["210507_original.jpg", "210507_frame2.jpg", "210507_frame3.jpg",
               "210507_frame4.jpg"]


def _nearest(image, candidates):
    """The index of the candidate ``image`` is closest to (mean absolute difference)."""
    return int(np.argmin([np.abs(image.astype(int) - c.astype(int)).mean()
                          for c in candidates]))


def _stack_session(tmp_path, pipe, camera, *, grade="inline", **kwargs):
    session = CaptureSession(camera, pipe, tmp_path / "shots", now=lambda: FIXED,
                             seed_rng=np.random.default_rng(0),
                             background=grade == "background", state_dir=tmp_path / "state",
                             log=lambda _: None, **kwargs)
    session.multi_frame = 4
    return session


def _check_stack_record(record, camera):
    keys = list(record)
    assert keys[keys.index("saturation") + 1] == "shutter_delay_s"
    assert keys[keys.index("shutter_delay_s") + 1] == "stack"
    stack = record["stack"]
    assert list(stack) == STACK_KEYS
    assert stack["frames"] == STACK_NAMES and stack["reference"] == STACK_NAMES[0]
    # The reference (FocusFoM 5.0) is capture frame 1: its own exposure, the asked one beside it.
    assert (stack["exposure_us"], stack["gain"]) == (2501, 5.0)
    assert (stack["requested_exposure_us"], stack["requested_gain"]) == (2500, 4.0)
    assert stack["split"] is True and stack["method"] == "ecc-euclidean-ghost-weighted-mean"
    assert len(stack["aligned"]) == 4 and stack["aligned"][0]["used"] is True
    assert all(set(f) == {"dx", "dy", "angle_deg", "used"} for f in stack["aligned"])
    assert "dng" not in record
    assert record["camera_metadata"]["FocusFoM"] == 5.0
    assert {"burst", "save_original", "save_frames", "merge"} <= set(record["stage_ms"])
    assert "capture" not in record["stage_ms"]
    assert camera.bursts == [(4, True, False)] and camera.reads == 0
    json.dumps(record)


def _expected_graded(pipe, camera, seed):
    """The graded picture: the frames merged onto the reference (capture frame 1),
    the others in capture order, then graded with the shot's seed."""
    from pifilm.stack import merge

    frames = camera.burst_frames
    merged, _ = merge([frames[1], frames[0], frames[2], frames[3]], reference=0)
    return pipe.process(merged, rng=np.random.default_rng(seed))[0]


def test_an_inline_multi_frame_shot_merges_the_burst(tmp_path, pipeline):
    camera = _BurstCamera()
    session = _stack_session(tmp_path, pipeline, camera)
    result = session.capture(delay_s=0.2)
    (record,) = _lines(session.out_root)
    _check_stack_record(record, camera)
    assert record["shutter_delay_s"] == 0.2
    assert result.record["stack"] == record["stack"]
    day = result.original.parent
    assert result.original.name == STACK_NAMES[0]
    assert sorted(p.name for p in day.glob("*.jpg")) == sorted(
        STACK_NAMES + ["210507_graded.jpg"])
    assert not list(day.glob("*.dng"))
    # Named in capture order without the reference: frame2 is capture frame 0.
    for name, index in zip(STACK_NAMES, (1, 0, 2, 3), strict=True):
        assert _nearest(load_rgb(day / name)[0], camera.burst_frames) == index, name
    # The merge is what was graded, and it is not the reference alone.
    used = sum(f["used"] for f in record["stack"]["aligned"])
    assert used >= 2
    expected = _expected_graded(pipeline, camera, record["grain_seed"])
    reference_only = pipeline.process(camera.burst_frames[1],
                                      rng=np.random.default_rng(record["grain_seed"]))[0]
    assert not np.array_equal(expected, reference_only)
    on_disk = load_rgb(result.graded)[0].astype(int)
    assert np.abs(on_disk - expected.astype(int)).mean() < \
        np.abs(on_disk - reference_only.astype(int)).mean()


def test_a_background_multi_frame_shot_is_the_inline_one(tmp_path, pipeline):
    inline = _stack_session(tmp_path / "inline", pipeline, _BurstCamera()).capture()
    camera = _BurstCamera()
    held = _SpyPipeline(pipeline)
    session = _stack_session(tmp_path, held, camera, grade="background")
    result = session.capture()
    held.wait_entered()
    ((_, entry),) = read_entries(tmp_path / "state")
    assert entry["originals"] == [f"2026-09-03/{name}" for name in STACK_NAMES]
    assert entry["stack"] == {"reference_index": 1} and entry["double"] is None
    assert list(entry["record"]["stack"]) == STACK_KEYS[:7]
    assert "dng" not in entry["record"]
    # Every frame is on disk while the grade has not run.
    assert all((result.original.parent / name).exists() for name in STACK_NAMES)
    held.release()
    assert session.close_grades(timeout=5)
    (record,) = _lines(session.out_root)
    _check_stack_record(record, camera)
    assert _earlier_keys(record) == _earlier_keys(inline.record)
    assert record["stack"]["aligned"] == inline.record["stack"]["aligned"]
    assert result.graded.read_bytes() == inline.graded.read_bytes()
    assert "merge" not in result.record["stage_ms"]  # the capture side never merges
    assert read_entries(tmp_path / "state") == []


def test_the_quick_grade_is_the_reference_frame_alone(tmp_path, pipeline, monkeypatch):
    from pifilm.capture import app

    seen = []
    real = app._quick_frame
    monkeypatch.setattr(app, "_quick_frame", lambda rgb: seen.append(rgb) or real(rgb))
    camera = _BurstCamera(foms=(1.0, 2.0))  # a two-frame burst must not be composited
    session = _stack_session(tmp_path, pipeline, camera, grade="background")
    session.multi_frame = 2
    result = session.capture()
    assert session.close_grades(timeout=5)
    assert len(seen) == 1 and np.array_equal(seen[0], camera.burst_frames[1])
    quick = [pipeline.process(f, grain=False)[0] for f in camera.burst_frames]
    assert _nearest(load_rgb(result.pifilm)[0], quick) == 1


def test_a_recovered_multi_frame_grade_merges_the_saved_frames(tmp_path, pipeline):
    camera = _BurstCamera()
    held = _SpyPipeline(pipeline)
    session = _stack_session(tmp_path, held, camera, grade="background")
    result = session.capture()
    held.wait_entered()
    ((path, _),) = read_entries(tmp_path / "state")
    text = path.read_text()
    held.release()
    assert session.close_grades(timeout=5)
    (first,) = _lines(session.out_root)
    result.graded.unlink()
    (result.graded.parent / "captures.jsonl").unlink()
    path.write_text(text)

    again = _stack_session(tmp_path, pipeline, FakeCamera([synthetic_frame(90, 160)]),
                           grade="background")
    _settle(again)
    assert again.close_grades(timeout=5)
    assert not path.exists()
    (record,) = _lines(again.out_root)
    assert record["recovered"] is True
    assert list(record["stack"]) == STACK_KEYS
    assert record["stack"]["frames"] == STACK_NAMES
    assert [f["used"] for f in record["stack"]["aligned"]] == \
        [f["used"] for f in first["stack"]["aligned"]]
    assert "merge" in record["stage_ms"]
    # From the JPEGs on disk: close to the grade from memory, not the same.
    expected = _expected_graded(pipeline, camera, record["grain_seed"])
    assert np.abs(load_rgb(result.graded)[0].astype(int) - expected.astype(int)).mean() < 3.0


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_a_double_exposure_ignores_multi_frame(tmp_path, pipeline, grade):
    camera = _BurstCamera()
    session = _stack_session(tmp_path, pipeline, camera, grade=grade)
    session.set_double_exposure(True)
    session.capture()
    session.capture()
    assert session.close_grades(timeout=5)
    assert camera.bursts == [] and camera.reads == 2
    records = _lines(session.out_root)
    assert len(records) == 2 and not any("stack" in r for r in records)
    assert not list(session.out_root.glob("*/*_frame*.jpg"))


@pytest.mark.parametrize("grade", ["inline", "background"])
def test_multi_frame_off_reads_one_frame(tmp_path, pipeline, grade):
    camera = _BurstCamera()
    session = _stack_session(tmp_path, pipeline, camera, grade=grade)
    session.multi_frame = 1
    session.capture()
    assert session.close_grades(timeout=5)
    assert camera.bursts == [] and camera.reads == 1
    (record,) = _lines(session.out_root)
    assert "stack" not in record and record["dng"] == "210507.dng"
    assert "capture" in record["stage_ms"] and "burst" not in record["stage_ms"]


def test_the_reference_is_frame_0_without_a_focus_figure(tmp_path, pipeline):
    camera = _BurstCamera()
    camera.foms = ("none",)  # not a number: no frame says how sharp it is
    session = _stack_session(tmp_path, pipeline, camera)
    session.capture()
    (record,) = _lines(session.out_root)
    assert record["stack"]["exposure_us"] == 2500  # capture frame 0's metadata


def test_an_entry_without_a_stack_is_one_frame():
    from pifilm.capture.app import _GradeWork

    payload = {"day": "2026-09-03", "graded": "x_graded.jpg", "originals": ["a/x.jpg"],
               "seed": 1, "ev": 0.0, "record": {}}
    assert _GradeWork.from_entry(payload).stack is None
    work = _GradeWork.from_entry({**payload, "stack": None})
    assert work.stack is None and work.entry(Path("/o"))["stack"] is None
    stacked = {**payload, "originals": ["a/x.jpg", "a/y.jpg"], "stack": {"reference_index": 1}}
    assert _GradeWork.from_entry(stacked).stack == {"reference_index": 1}
    # Two originals without a stack (nor a double) is no entry this build wrote.
    with pytest.raises(ValueError):
        _GradeWork.from_entry({**stacked, "stack": None})
    for bad in ({"reference_index": 2}, {"reference_index": True}, {}, [1]):
        with pytest.raises(ValueError):
            _GradeWork.from_entry({**stacked, "stack": bad})
    with pytest.raises(ValueError):
        _GradeWork.from_entry({**payload, "stack": {"reference_index": 0}})  # one frame
    with pytest.raises(ValueError):
        _GradeWork.from_entry({**stacked, "double": {"ev_comp": [0.0, 0.0]}})


def test_the_multi_frame_applier_reaches_the_session(camera_cli, monkeypatch, tmp_path):
    _save_settings(multi_frame=4)
    made = []
    real = camera_cli.CaptureSession

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(camera_cli, "CaptureSession", recording)
    seen = _menu_main(camera_cli, monkeypatch, tmp_path, _StabiliseMenuCamera())
    (session,) = made
    assert session.multi_frame == 4
    assert seen["settings"].set("multi_frame", 6)
    assert session.multi_frame == 6
    seen["settings"].reset()
    assert session.multi_frame == 1


def test_a_saved_multi_frame_is_not_taken_under_fake(camera_cli, monkeypatch, tmp_path):
    _save_settings(multi_frame=6)
    made = []
    real = camera_cli.CaptureSession

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(camera_cli, "CaptureSession", recording)
    assert camera_cli.main(["--fake", "--no-preview", "--out", str(tmp_path / "s")]) == 0
    assert made[0].multi_frame == 1
