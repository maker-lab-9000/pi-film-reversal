import io

import cv2
import numpy as np
import pytest
from PIL import Image

from pifilm.capture import camera as camera_module
from pifilm.capture.camera import (
    CameraError,
    FakeCamera,
    Frame,
    StreamInfo,
    V4L2Camera,
    is_valid_jpeg,
    parse_device,
    synthetic_frame,
)


def _tagged_frame(index):
    """A synthetic frame with a unique first pixel, so frames are distinguishable."""
    frame = synthetic_frame(48, 64).copy()
    frame[0, 0] = (index, index, index)
    return frame


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_synthetic_frame_shape_and_colour():
    f = synthetic_frame(90, 160)
    assert f.shape == (90, 160, 3) and f.dtype == np.uint8
    assert f[..., 0].mean() != f[..., 2].mean()


def test_is_valid_jpeg_markers():
    good = _jpeg_bytes(synthetic_frame(16, 16))
    assert is_valid_jpeg(good)
    assert not is_valid_jpeg(good[:-2])        # truncated: no EOI
    assert not is_valid_jpeg(b"\x00\x01" + good[2:])  # no SOI
    assert not is_valid_jpeg(b"")
    assert not is_valid_jpeg(b"\xff\xd8\xff\xd9")     # too short to be a frame


def test_fake_camera_defaults_to_decoded_mode():
    cam = FakeCamera()
    frame = cam.read()
    assert isinstance(frame, Frame)
    assert frame.rgb.shape == (1080, 1920, 3) and frame.rgb.dtype == np.uint8
    assert frame.jpeg is None and frame.source == "decoded"
    assert isinstance(cam.stream_info, StreamInfo)
    cam.close()


def test_fake_camera_raw_mode_returns_bytes_that_decode_to_the_frame():
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded), "rgb must be the decode of the same buffer"


def test_fake_camera_records_ev_and_returns_metadata():
    camera = FakeCamera([synthetic_frame(8, 8)], metadata={"Lux": 100.0})
    assert camera.ev == 0.0
    camera.set_ev(-0.5)
    assert camera.ev == -0.5
    assert camera.read(full=False).metadata == {"Lux": 100.0}


def test_fake_camera_cycles_and_copies():
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    cam = FakeCamera(frames)
    assert cam.read().rgb.max() == 0
    assert cam.read().rgb.max() == 1
    assert cam.read().rgb.max() == 0
    cam.read().rgb[0, 0, 0] = 99
    assert frames[0].max() == 0


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (3, 3),
        ("3", 3),
        ("/dev/video7", 7),
        ("/dev/v4l/by-id/usb-Innomaker-video-index0", "/dev/v4l/by-id/usb-Innomaker-video-index0"),
    ],
)
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_garbage():
    with pytest.raises(CameraError):
        parse_device("camera")


class FakeCapture:
    """Stands in for ``cv2.VideoCapture`` so the real V4L2Camera can be driven.

    Without this, none of the raw-mode, fallback, negotiation or retry logic is
    executed by any test: a build where raw mode silently never engaged would
    pass the whole suite, and that logic is the project's headline promise.

    Three things to know when writing tests against it. Constructing a
    ``V4L2Camera`` consumes one buffer, because ``_enable_raw_mode`` reads a
    frame to verify the mode really works — so a queue must include that
    verification frame before the ones a test intends ``read()`` to return.
    Format properties are reported, not stored (see ``set``). And ``grab``
    consumes only when a test passes ``grab_consumes=True``, so that buffer
    counts stay independent of ``_drain``'s internal grab limit.
    """

    def __init__(
        self,
        *,
        buffers=None,
        decoded=None,
        set_convert_fails=False,
        set_raises=False,
        mutate_then_raise=False,
        restore_behaviour="ok",
        buffer_ndim=1,
        grab_consumes=False,
        props=None,
    ):
        self.buffers = list(buffers or [])
        self.decoded = decoded if decoded is not None else np.zeros((1080, 1920, 3), np.uint8)
        self.set_convert_fails = set_convert_fails
        self.set_raises = set_raises
        self.mutate_then_raise = mutate_then_raise
        self.restore_behaviour = restore_behaviour
        self.buffer_ndim = buffer_ndim
        self.grab_consumes = grab_consumes
        self.props = props or {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        self.convert_rgb = 1
        self.released = False
        self.requested = []
        self.grabs = 0

    def isOpened(self):
        return True

    def set(self, prop, val):
        """Accept a request but do not pretend it was honoured.

        A real camera reports the format it actually negotiated, which is the
        entire reason ``_negotiated`` reads the properties back. Storing the
        requested value here would make every negotiation check tautological:
        the code would read back exactly what it just wrote and never warn.
        So format writes are recorded and ignored; only CONVERT_RGB takes
        effect, because that one genuinely changes what ``read`` returns.
        """
        self.requested.append((prop, val))
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            if val == 1 and self.convert_rgb == 0:
                if self.restore_behaviour == "raise":
                    raise cv2.error("driver refused to leave raw mode")
                if self.restore_behaviour == "false":
                    return False
            if self.mutate_then_raise:
                self.convert_rgb = val
                raise cv2.error("device rejected the mode after applying it")
            if self.set_raises:
                raise cv2.error("simulated")
            if self.set_convert_fails:
                return False
            self.convert_rgb = val
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        if self.convert_rgb == 0:
            if not self.buffers:
                return False, None
            buf = np.frombuffer(self.buffers.pop(0), np.uint8)
            return True, buf.reshape(1, -1) if self.buffer_ndim == 2 else buf
        return True, self.decoded

    def grab(self):
        """Count every grab; consume a buffer only when a test asks it to.

        An unconditional ``True`` that consumes nothing makes ``_drain``
        untestable — a drain that grabbed the wrong number of frames, or was
        deleted outright, would still pass. But consuming unconditionally
        couples every fixture's buffer count to the literal ``4`` inside
        ``_drain``: change that constant and unrelated tests fail for reasons
        they do not assert on. So consumption is opt-in, and exactly one test
        opts in to verify the drain.
        """
        self.grabs += 1
        if not self.grab_consumes:
            return True
        if not self.buffers:
            return False
        self.buffers.pop(0)
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """Install a FakeCapture in place of cv2.VideoCapture and hand it back."""
    holder = {}

    def install(**kwargs):
        cap = FakeCapture(**kwargs)
        holder["cap"] = cap
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: cap)
        return cap

    return install


def test_raw_mode_engages_and_preserves_the_camera_bytes(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # One buffer is consumed by _enable_raw_mode's verification read.
    cap = fake_capture(buffers=[data] * 4)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is True
    assert cap.convert_rgb == 0

    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded)


def test_an_invalid_buffer_falls_back_once_and_stays_fallen_back(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # First buffer is eaten by the constructor's verification read, second is
    # the good frame the first read() returns, third is the truncated one.
    cap = fake_capture(
        buffers=[data, data, data[:-2]], decoded=np.zeros((48, 64, 3), np.uint8)
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    assert cam.read().source == "decoded"
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1
    warnings = capsys.readouterr().out.count("falling back to decoded frames")
    assert warnings == 1, "the fallback must announce itself once, not per frame"

    assert cam.read().source == "decoded"
    assert capsys.readouterr().out.count("falling back to decoded frames") == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"set_convert_fails": True}, {"set_raises": True}, {"mutate_then_raise": True}],
    ids=["set-returns-false", "set-raises", "set-mutates-then-raises"],
)
def test_every_raw_mode_failure_leaves_the_device_in_decoded_mode(fake_capture, kwargs):
    """The half-state is silently destructive, so no failure path may leave it."""
    cap = fake_capture(decoded=np.zeros((1080, 1920, 3), np.uint8), **kwargs)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1, "CONVERT_RGB left at 0 while the object thinks it is decoding"
    assert cam.read().source == "decoded"


def test_negotiation_mismatch_warns_but_does_not_abort(fake_capture, capsys):
    cap = fake_capture(
        decoded=np.zeros((720, 1280, 3), np.uint8),
        props={
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
        },
        set_convert_fails=True,
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    out = capsys.readouterr().out
    assert "negotiated 1280x720" in out
    assert "YUYV" in out
    assert cam.stream_info.width == 1280 and cam.stream_info.fps == 15.0
    assert cam.read().rgb.shape == (720, 1280, 3)
    assert cap is not None


def test_read_returns_the_newest_frame_not_a_stale_queued_one(fake_capture):
    """`_drain` must discard queued frames, or the preview lags behind reality."""
    buffers = [_jpeg_bytes(_tagged_frame(i)) for i in range(6)]
    cap = fake_capture(buffers=list(buffers), grab_consumes=True)
    cam = V4L2Camera(device=0, warmup_frames=0)

    frame = cam.read()
    assert cap.grabs == 4, "the drain loop should grab up to four queued frames"
    # Index 0 is eaten by the constructor's verification read, 1-4 by the drain,
    # so a correct drain leaves index 5. Without the drain this would be index 1.
    assert frame.jpeg == buffers[5], "read returned a stale frame instead of the newest"


def test_a_two_dimensional_raw_buffer_is_accepted(fake_capture):
    """Real V4L2 backends can hand back a 2-D buffer; 1-D is not the only shape."""
    data = _jpeg_bytes(synthetic_frame(48, 64))
    cap = fake_capture(buffers=[data] * 4, buffer_ndim=2)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is True
    assert cam.read().jpeg == data
    assert cap is not None


@pytest.mark.parametrize("restore", ["raise", "false"], ids=["raises", "returns-false"])
def test_a_device_that_will_not_leave_raw_mode_fails_honestly(fake_capture, restore, capsys):
    """Refusing the restore must not produce frames the session cannot trust.

    If the device keeps sending compressed buffers, claiming decoded mode
    would send them through ``cvtColor`` as BGR and yield silent garbage. The
    session must stay in raw mode and fail loudly instead.
    """
    good = _jpeg_bytes(synthetic_frame(48, 64))
    bad = good[:-2]
    cap = fake_capture(
        buffers=[good, good, bad, bad, bad, bad],
        decoded=np.zeros((48, 64, 3), np.uint8),
        restore_behaviour=restore,
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    with pytest.raises(CameraError, match="3 attempts"):
        cam.read()

    assert cam.stream_info.raw_mjpeg is True, "the mode must not change if the device refused"
    assert cap.convert_rgb == 0
    assert "refused to leave raw mode" in capsys.readouterr().out


def test_a_successful_restore_does_fall_back_to_decoded(fake_capture):
    """The ordinary case: the device complies, so the session degrades cleanly."""
    good = _jpeg_bytes(synthetic_frame(48, 64))
    fake_capture(
        buffers=[good, good, good[:-2], good],
        decoded=np.zeros((48, 64, 3), np.uint8),
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    frame = cam.read()
    assert frame.source == "decoded"
    assert frame.jpeg is None
    assert cam.stream_info.raw_mjpeg is False


def test_read_raises_after_three_failed_attempts(fake_capture):
    cap = fake_capture(buffers=[])
    cam = V4L2Camera(device=0, warmup_frames=0)
    cap.convert_rgb = 0
    cap.buffers = []
    with pytest.raises(CameraError, match="3 attempts"):
        cam.read()


def test_close_releases_the_device(fake_capture):
    cap = fake_capture(set_convert_fails=True)
    cam = V4L2Camera(device=0, warmup_frames=0)
    cam.close()
    assert cap.released is True


def test_v4l2_camera_reports_missing_device():
    with pytest.raises(CameraError, match="video"):
        V4L2Camera(device=99, warmup_frames=0)


def test_stream_info_to_dict():
    info = StreamInfo(width=1920, height=1080, fps=30.0, fourcc="MJPG", raw_mjpeg=True)
    assert info.to_dict() == {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "fourcc": "MJPG",
        "raw_mjpeg": True,
    }


def test_stream_info_to_dict_includes_available_sensor_details():
    info = StreamInfo(
        width=4608,
        height=2592,
        fps=14.25,
        fourcc="RGB888",
        raw_mjpeg=False,
        sensor_mode="4608x2592 SBGGR10_CSI2P",
        bit_depth=10,
        tuning_file="imx708_wide.json",
    )
    assert info.to_dict() == {
        "width": 4608,
        "height": 2592,
        "fps": 14.25,
        "fourcc": "RGB888",
        "raw_mjpeg": False,
        "sensor_mode": "4608x2592 SBGGR10_CSI2P",
        "bit_depth": 10,
        "tuning_file": "imx708_wide.json",
    }
