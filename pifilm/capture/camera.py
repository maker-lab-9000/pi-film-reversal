"""Camera access for the Innomaker U20CAM-1080P-WDR, plus a fake for tests.

Facts from the vendor manual that shaped this module:

* Standard UVC device, driven through OpenCV's V4L2 backend.
* 1920x1080 at 30 fps exists only in MJPEG; YUY2 falls to 5 fps at 1080p, so
  the FOURCC is forced to ``MJPG``.
* The camera runs its own auto exposure and white balance, which need a few
  frames to settle after opening (``warmup_frames``). Version 1 records
  those controls rather than locking them; see spec 7.5.

Byte-exact originals
--------------------
The headline promise is that the camera's own JPEG is saved. ``read()``
normally decodes MJPEG into BGR pixels, which would mean re-encoding a
second lossy JPEG and calling it the original. Instead the camera asks the
V4L2 backend for the compressed buffer with ``CAP_PROP_CONVERT_RGB = 0``,
and then:

* validates the buffer really is a complete JPEG (SOI at the front, EOI at
  the end, and it decodes) - OpenCV issue #23311 shows the backend can hand
  back truncated data on some devices;
* returns those exact bytes as ``Frame.jpeg`` and the decode of *that same
  buffer* as ``Frame.rgb``, so the saved original and the graded image
  provably come from one acquisition.

If raw mode is unsupported, or a buffer fails validation, the camera says so
once and falls back to decoded mode for the rest of the session. The app
then names its second file ``_ungraded.jpg`` rather than ``_original.jpg``,
so a filename never claims more than the bytes deliver.

Negotiation is verified, not assumed: FOURCC, size and rate are read back
after being set, a mismatch is warned about naming both values, and the
result is recorded in ``StreamInfo`` for the capture log.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .._cv2 import require_cv2
from .errors import CameraError

cv2 = require_cv2()

# CameraError is defined in .errors, which needs no OpenCV, and re-exported
# here so that `from .camera import CameraError` keeps working unchanged.

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_MIN_JPEG_BYTES = 128


@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float
    fourcc: str
    raw_mjpeg: bool
    sensor_mode: str | None = None
    bit_depth: int | None = None
    tuning_file: str | None = None
    autofocus: str | None = None

    def to_dict(self) -> dict:
        values = {
            "width": int(self.width),
            "height": int(self.height),
            "fps": round(float(self.fps), 2),
            "fourcc": self.fourcc,
            "raw_mjpeg": bool(self.raw_mjpeg),
        }
        if self.sensor_mode is not None:
            values["sensor_mode"] = self.sensor_mode
        if self.bit_depth is not None:
            values["bit_depth"] = int(self.bit_depth)
        if self.tuning_file is not None:
            values["tuning_file"] = self.tuning_file
        if self.autofocus is not None:
            values["autofocus"] = self.autofocus
        return values


@dataclass
class Frame:
    rgb: np.ndarray
    jpeg: bytes | None
    source: str  # "raw-mjpeg", "decoded", or "picamera2"
    metadata: dict | None = None  # per-shot camera metadata (picamera2)
    dng: bytes | None = None      # in-memory DNG for this frame (picamera2)


class Camera(Protocol):
    @property
    def stream_info(self) -> StreamInfo: ...

    def read(self, *, full: bool = True) -> Frame: ...

    def close(self) -> None: ...


def is_valid_jpeg(buf: bytes) -> bool:
    """A complete JPEG: start-of-image marker, end-of-image marker, plausible length."""
    return (
        len(buf) >= _MIN_JPEG_BYTES and buf[:2] == _SOI and buf[-2:] == _EOI
    )


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code else "----"


def synthetic_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """A grey gradient with a row of colour patches, for tests and ``--fake`` runs."""
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    frame = np.repeat(np.repeat(ramp[None, :, None], height, axis=0), 3, axis=2).astype(np.uint8)
    patches = [
        (220, 40, 40),
        (40, 180, 60),
        (50, 80, 220),
        (240, 200, 60),
        (230, 180, 150),
        (120, 120, 120),
    ]
    pw = max(1, width // len(patches))
    top, bottom = height // 4, height // 2
    for i, colour in enumerate(patches):
        frame[top:bottom, i * pw : (i + 1) * pw] = colour
    return frame


class FakeCamera:
    """Stands in for the hardware. Give it ``jpeg_bytes`` to exercise raw mode."""

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        jpeg_bytes: list[bytes] | None = None,
        source: str = "decoded",
        stream_info: StreamInfo | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._jpegs = jpeg_bytes
        self._metadata = metadata
        self.ev = 0.0
        if jpeg_bytes is not None:
            decoded = [
                cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jpeg_bytes
            ]
            self._frames = decoded
            self._frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self._frames]
        else:
            self._frames = frames if frames else [synthetic_frame()]
        self._source = source
        h, w = self._frames[0].shape[:2]
        self._info = stream_info or StreamInfo(
            width=w, height=h, fps=30.0, fourcc="MJPG", raw_mjpeg=source == "raw-mjpeg"
        )
        self._i = 0

    @property
    def stream_info(self) -> StreamInfo:
        return self._info

    def read(self, *, full: bool = True) -> Frame:
        # `full` is Picamera2-only (skips its autofocus cycle and DNG extraction);
        # this path is already cheap, so the flag has nothing to skip.
        idx = self._i % len(self._frames)
        self._i += 1
        jpeg = self._jpegs[idx % len(self._jpegs)] if self._jpegs else None
        return Frame(
            rgb=self._frames[idx].copy(),
            jpeg=jpeg,
            source=self._source,
            metadata=self._metadata,
        )

    def set_ev(self, value: float) -> None:
        # Mirrors Picamera2Camera.set_ev so viewfinder EV buttons can be driven
        # without hardware; there is no exposure to change, so it only records.
        self.ev = float(value)

    def close(self) -> None:
        return None


def parse_device(device: int | str | None) -> int | str | None:
    """Accept an index, ``/dev/videoN``, or a stable ``/dev/v4l/by-id/...`` path."""
    if device is None or isinstance(device, int):
        return device
    text = device.strip()
    if text.startswith("/dev/v4l/by-id/") or text.startswith("/dev/v4l/by-path/"):
        return text
    m = re.fullmatch(r"(?:/dev/video)?(\d+)", text)
    if not m:
        raise CameraError(
            f"Cannot parse camera device {device!r}; use an index, /dev/videoN, "
            "or a /dev/v4l/by-id/... path"
        )
    return int(m.group(1))


def list_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*")) + sorted(glob.glob("/dev/v4l/by-id/*"))


class V4L2Camera:
    def __init__(
        self,
        device: int | str | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        warmup_frames: int = 15,
        prefer_raw: bool = True,
    ) -> None:
        target = parse_device(device)
        candidates: list[int | str] = [target] if target is not None else list(range(10))
        self.cap = None
        chosen: int | str | None = None
        for candidate in candidates:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, _ = cap.read()
            if ok:
                self.cap = cap
                chosen = candidate
                break
            cap.release()
        if self.cap is None:
            found = list_video_devices()
            hint = f"found {', '.join(found)}" if found else "no /dev/video* devices exist"
            raise CameraError(
                f"No camera delivered a frame (tried {candidates[0]}"
                + (f"..{candidates[-1]}" if len(candidates) > 1 else "")
                + f"); {hint}. Pass --device N, /dev/videoN or a /dev/v4l/by-id/ path."
            )
        print(f"Using camera {chosen}")

        self._raw = self._enable_raw_mode() if prefer_raw else False
        self.stream_info = self._negotiated(width, height, fps)
        self._warned_fallback = False
        self._warned_stuck = False
        for _ in range(warmup_frames):
            self.cap.read()

    def _enable_raw_mode(self) -> bool:
        """Ask the backend for the compressed buffer; verify by reading one frame.

        Every failure path restores ``CAP_PROP_CONVERT_RGB``, including the
        ones where the ``set`` call itself failed or raised. OpenCV gives no
        guarantee that a failing ``set`` left the property untouched, and the
        half-state is silently destructive: the object would believe it is in
        decoded mode while the driver still hands back compressed buffers, so
        ``read`` would run ``cvtColor`` over JPEG bytes as though they were a
        BGR image and return plausible-looking garbage. Restoring
        unconditionally costs one ignored call and removes the question.
        """

        def give_up() -> bool:
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except cv2.error:
                pass
            return False

        try:
            if not self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                return give_up()
        except cv2.error:
            return give_up()
        ok, buf = self.cap.read()
        if not ok or buf is None or (buf.ndim != 2 and buf.ndim != 1):
            return give_up()
        if not is_valid_jpeg(np.asarray(buf, dtype=np.uint8).tobytes()):
            return give_up()
        return True

    def _negotiated(self, width: int, height: int, fps: int) -> StreamInfo:
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        if (actual_w, actual_h) != (width, height):
            print(f"warning: requested {width}x{height}, camera negotiated {actual_w}x{actual_h}")
        if fourcc != "MJPG":
            print(
                f"warning: requested MJPG, camera negotiated {fourcc}; "
                "1080p30 may be unavailable"
            )
        if actual_fps and abs(actual_fps - fps) > 1.0:
            print(f"warning: requested {fps} fps, camera reports {actual_fps:g} fps")
        if not self._raw:
            print("note: raw MJPEG unavailable; captures will be saved as re-encoded _ungraded.jpg")
        return StreamInfo(actual_w, actual_h, actual_fps, fourcc, self._raw)

    def _drain(self) -> None:
        """Discard queued frames so the next read is the newest available."""
        for _ in range(4):
            if not self.cap.grab():
                break

    def _fallback_to_decoded(self, reason: str) -> bool:
        """Try to leave raw mode. Only claim success if the device complied.

        Two failures are guarded here, and the second is subtler than it
        looks. Raising must not escape, or the error would propagate out of
        ``read`` and end the session this fallback exists to preserve.

        But swallowing the error is not enough either. If the device refuses
        to leave raw mode, it keeps sending compressed buffers — so declaring
        decoded mode anyway would send the next frame down the decoded branch,
        where ``cvtColor`` reads a JPEG buffer as though it were a BGR image.
        Measured: that returns a plausible ``(1, 504, 3)`` array rather than
        raising, so the session would serve silent garbage. That is the same
        belief-versus-reality split this class already guards against in
        ``_enable_raw_mode``.

        So the mode only changes when the device actually changed. If it did
        not, the session stays in raw mode and keeps validating buffers; a
        genuinely broken stream then exhausts ``read``'s retry budget and
        fails loudly, which is the honest outcome. A single bad buffer does
        not permanently downgrade a camera that cannot be downgraded.
        """
        try:
            left_raw_mode = bool(self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1))
        except cv2.error:
            left_raw_mode = False

        if not left_raw_mode:
            if not self._warned_stuck:
                print(
                    f"warning: {reason}, and the device refused to leave raw mode; "
                    "continuing to validate raw buffers"
                )
                self._warned_stuck = True
            return False

        if not self._warned_fallback:
            print(f"warning: {reason}; falling back to decoded frames (_ungraded.jpg)")
            self._warned_fallback = True
        self._raw = False
        self.stream_info.raw_mjpeg = False
        return True

    def read(self, *, full: bool = True) -> Frame:
        # `full` is Picamera2-only; this path is already cheap and ignores it.
        for _ in range(3):
            self._drain()
            ok, data = self.cap.read()
            if not ok or data is None:
                continue
            if self._raw:
                buf = np.asarray(data, dtype=np.uint8).tobytes()
                if not is_valid_jpeg(buf):
                    self._fallback_to_decoded("camera returned an incomplete JPEG buffer")
                    continue
                bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    self._fallback_to_decoded("camera buffer failed to decode")
                    continue
                return Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), buf, "raw-mjpeg")
            return Frame(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), None, "decoded")
        raise CameraError("Failed to read a frame after 3 attempts")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
