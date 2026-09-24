import argparse
import builtins
import io
import json
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import (
    CaptureSession,
    _colour_gains,
    main,
    run_headless_loop,
    run_preview_loop,
)
from pifilm.capture.camera import CameraError, FakeCamera, Frame, StreamInfo, synthetic_frame
from pifilm.grain import GrainParams
from pifilm.imageio import load_rgb
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline


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
    }
    assert line["shutter_to_saved_ms"] >= line["pipeline_ms"]
    assert isinstance(line["grain_seed"], int)


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

    def delayed_capture():
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

    def delayed_capture():
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


@pytest.mark.parametrize("failed_attempt", [1, 2])
def test_capture_display_keeps_last_photo_after_failed_capture(
    tmp_path, pipeline, monkeypatch, capture_gui, capsys, failed_attempt,
):
    import cv2

    session = _session(tmp_path, pipeline)
    capture = session.capture
    attempts = 0

    def flaky_capture():
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

    def set_ev(self, value):
        self.ev = value

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
        (["--camera", "v4l2", "--ev", "-0.5"], "--ev"),
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
        ["--ev", "0.3"],
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
        "--ae-constraint", "highlight", "--ae-metering", "matrix", "--ev", "-0.5",
    ]) == 0

    assert backend_kwargs["ae_constraint"] == "highlight"
    assert backend_kwargs["ae_metering"] == "matrix"
    assert backend_kwargs["ev"] == -0.5


def test_default_flags_leave_ae_shaping_at_libcamera_defaults(camera_cli, monkeypatch):
    backend_kwargs = {}

    def open_picamera(tuning_file, **kwargs):
        backend_kwargs.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)

    assert camera_cli.main(["--camera", "picamera2", "--no-preview"]) == 0

    assert backend_kwargs["ae_constraint"] == "normal"
    assert backend_kwargs["ae_metering"] == "centre"
    assert backend_kwargs["ev"] == 0.0


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

        def capture(self):
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

    class StopSoon:
        def __init__(self):
            self.n = 0

        def is_set(self):
            self.n += 1
            return self.n > 2

    # Narrow the patch to app's own name: setattr on the real threading module
    # would also replace the Event that Thread.start() uses internally.
    monkeypatch.setattr(app, "threading", SimpleNamespace(Event=StopSoon))
    out = tmp_path / "shots"
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

    def viewfinder_double(camera, controller, display_pair, power, args):
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
