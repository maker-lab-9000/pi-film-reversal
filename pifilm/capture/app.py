"""``pifilm-capture``: live preview, press SPACE, get two JPEGs.

Structure
---------
``CaptureSession`` owns the camera, the pipeline and the output folder, and
knows how to take one capture or produce one preview frame. Two thin loops
drive it, both taking injectable key sources so the whole flow is testable
with ``FakeCamera``:

* ``run_preview_loop`` draws the graded feed, or only saved captures, in an OpenCV window.
  Capture display mode reads the camera only when SPACE is pressed. Every GUI
  call is inside the guard, because a build without GUI support can fail at
  ``imshow`` rather than at ``namedWindow``, and a failure there must fall
  back to headless rather than end the session.
* ``run_headless_loop`` reads single keys from the terminal.
* ``--display`` runs the LCD viewfinder (``pifilm/display/viewfinder.py``) on the
  Pi's SPI panel instead of any OpenCV window. It shares one
  ``CaptureController`` with the remote API, so a tap on the panel and a Stick
  request are the same kind of job and never two owners of the camera.

A dropped frame prints and continues in both loops. The spec promises the
session survives frame read failures, and that promise is only worth
anything if it also holds while the preview is running.

Capture semantics
-----------------
``SPACE`` acquires one fresh frame and saves exactly that frame; the
displayed frame is not re-used. ``capture()`` calls ``camera.read()`` once,
so the saved original and the graded image always come from one acquisition.

Output layout
-------------
``OUT/YYYY-MM-DD/HHMMSS_original.jpg`` holds the original for the shot.
For the USB camera (V4L2) it is the camera's own JPEG bytes, written verbatim.
For Picamera2, which has no separate camera JPEG, it is a single JPEG encode of
the ISP's RGB frame; that frame is always the camera's own full-quality
rendering, so it is named ``_original.jpg`` whether or not a DNG sidecar is
also saved. ``HHMMSS_graded.jpg`` is the graded version. ``_ungraded.jpg`` is
used only for the V4L2 fallback, when there is no camera JPEG to save
verbatim (raw mode unsupported, or a captured buffer failed validation). When
the backend supplies a raw frame and DNG saving is enabled, a ``<stem>.dng``
sidecar is written alongside the original; ``--no-dng`` only omits that
sidecar and never changes the original's name. One JSON line per capture
lands in ``captures.jsonl``.

That line is an audit record, not a status message. It carries the LUT hash,
the normalisation hash and the grain seed, which together let anyone regenerate
the graded file from the original — the LUT hash alone is not enough, because
normalisation is a per-artifact setting that changes the grade under an
unchanged LUT; the negotiated stream format, so a camera that quietly
dropped to a different mode is visible; two timings, because the pipeline cost
and the time from shutter to durable file are different numbers and only the
second is what the user waits for; ``camera_metadata`` (per-shot camera metadata)
when present; and ``dng`` (the sidecar filename) when a DNG was written.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from .._cv2 import require_cv2
from ..artifacts import PARAMS_VERSION, Artifacts, ArtifactsError
from ..display import DisplayError
from ..display.viewfinder import ViewfinderLoop
from ..imageio import load_rgb, save_jpeg
from ..pipeline import Pipeline
from .camera import Camera, CameraError, FakeCamera, V4L2Camera
from .controller import CaptureController, JobSnapshot
from .picamera import Picamera2Camera
from .power import UPS_CHOICES, PowerError, build_ups
from .remote import RemoteCaptureServer

cv2 = require_cv2()

DEFAULT_OUT = Path("~/Pictures/pifilm")
WINDOW_NAME = "Parr  [SPACE capture | P toggle grade | Q quit]"
CAPTURE_WINDOW_NAME = "Parr captures  [SPACE capture | Q quit]"


@dataclass
class CaptureResult:
    original: Path
    pifilm: Path
    record: dict


class CaptureSession:
    def __init__(
        self,
        camera: Camera,
        pipeline: Pipeline,
        out_root: str | Path,
        now: Callable[[], datetime] | None = None,
        seed_rng: np.random.Generator | None = None,
        package_version: str = __version__,
        save_dng: bool = True,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self.out_root = Path(out_root).expanduser()
        self._now = now or datetime.now
        self._seed_rng = seed_rng or np.random.default_rng()
        self._package_version = package_version
        self._save_dng = save_dng

    def _allocate(self, suffix: str) -> tuple[Path, str, datetime]:
        t = self._now()
        day_dir = self.out_root / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base = t.strftime("%H%M%S")
        stem, k = base, 1
        while (day_dir / f"{stem}_{suffix}.jpg").exists():
            k += 1
            stem = f"{base}-{k}"
        return day_dir, stem, t

    def capture(self) -> CaptureResult:
        shutter = time.perf_counter()
        frame = self.camera.read()

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(frame.rgb, rng=np.random.default_rng(seed))
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        has_original = frame.jpeg is not None or frame.source == "picamera2"
        suffix = "original" if has_original else "ungraded"
        day_dir, stem, t = self._allocate(suffix)
        original = day_dir / f"{stem}_{suffix}.jpg"
        if frame.jpeg is not None:
            original.write_bytes(frame.jpeg)
        else:
            save_jpeg(frame.rgb, original)
        dng_name = None
        if frame.dng is not None and self._save_dng:
            dng_path = day_dir / f"{stem}.dng"
            dng_path.write_bytes(frame.dng)
            dng_name = dng_path.name
        pifilm = save_jpeg(graded, day_dir / f"{stem}_graded.jpg")
        shutter_to_saved_ms = (time.perf_counter() - shutter) * 1000.0

        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": pifilm.name,
            "frame_source": frame.source,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round(shutter_to_saved_ms, 1),
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        with (day_dir / "captures.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return CaptureResult(original, pifilm, record)

    def preview_frame(self, graded: bool = True, size: tuple[int, int] = (640, 360)) -> np.ndarray:
        # full=False: the live preview must not pay Picamera2's per-frame autofocus
        # cycle or DNG extraction cost; capture() below keeps the full=True default.
        small = _resize_to_fit(self.camera.read(full=False).rgb, size)
        if graded:
            small, _ = self.pipeline.process(small, grain=False)
        return small


def _announce(result: CaptureResult, out: Callable[[str], None]) -> None:
    r = result.record
    clamps = [k for k, v in r["clamped"].items() if v]
    note = f" (clamped: {', '.join(clamps)})" if clamps else ""
    out(
        f"Saved {result.pifilm.name} + {result.original.name} in "
        f"{r['shutter_to_saved_ms']:.0f} ms; wb={r['wb_gains']} "
        + (f"levels gamma={r['levels']['gamma']} stretch={r['levels']['stretch']}"
           if r.get("levels") else f"exposure={r['exposure_gain']}")
        + note
    )


def run_headless_loop(
    session: CaptureSession,
    read_key: Callable[[], str | None],
    out: Callable[[str], None] = print,
    *,
    controller: CaptureController | None = None,
) -> int:
    out("Headless mode: SPACE to capture, Q to quit.")
    count = 0
    owned_controller = controller is None
    controller = controller or CaptureController(session)
    try:
        while True:
            key = read_key()
            if key is None:
                continue
            if key == " ":
                job = controller.submit(uuid.uuid4().hex)
                if job.error_code is not None:
                    out(f"error: {job.error_message or job.error_code}")
                    continue
                out("Processing photo...")
                completed = _wait_for_terminal_job(controller, job.request_id)
                if completed.state == "complete":
                    assert isinstance(completed.result, CaptureResult)
                    _announce(completed.result, out)
                    count += 1
                else:
                    out(f"error: {completed.error_message or completed.error_code}")
            elif key.lower() == "q":
                return count
    finally:
        if owned_controller:
            controller.close()


def _wait_for_terminal_job(controller: CaptureController, request_id: str) -> JobSnapshot:
    while True:
        job = controller.status(request_id)
        assert job is not None
        if job.state in {"complete", "failed"}:
            return job
        time.sleep(0.01)


def _resize_to_fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, fitted, interpolation=cv2.INTER_AREA)


def _fit_display(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit an image into the display, using black borders instead of cropping/stretching."""
    width, height = size
    fitted = _resize_to_fit(frame, size)
    image_height, image_width = fitted.shape[:2]
    left, top = (width - image_width) // 2, (height - image_height) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + image_height, left:left + image_width] = fitted
    return canvas


def _window_size(window_name: str, previous: tuple[int, int]) -> tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
        if width > 1 and height > 1:
            # GTK reports the fitted image rectangle, but exposes the entire widget's
            # width/height ratio through this property. Qt returns a ratio-mode enum.
            try:
                ratio = cv2.getWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO)
                if np.isfinite(ratio) and ratio > 0 and ratio != cv2.WINDOW_FREERATIO:
                    if width / height < ratio:
                        width = round(height * ratio)
                    else:
                        height = round(width / ratio)
            except cv2.error:
                pass
            return width, height
    except cv2.error:
        # Some backends cannot report geometry; let the window preserve image proportions.
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    return previous


def _capture_prompt(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    label = "SPACE to capture"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width / 640, height / 360)
    thickness = max(1, round(scale * 2))
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    position = ((width - text_width) // 2, (height + text_height) // 2)
    cv2.putText(screen, label, position, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return screen


def _capture_loading_screen(size: tuple[int, int]) -> np.ndarray:
    """Draw TV-style colour bars in OpenCV's BGR order."""
    width, height = size
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    colours = [
        (191, 191, 191), (0, 191, 191), (191, 191, 0), (0, 191, 0),
        (191, 0, 191), (0, 0, 191), (191, 0, 0),
    ]
    for index, colour in enumerate(colours):
        left, right = index * width // 7, (index + 1) * width // 7
        screen[:height * 3 // 4, left:right] = colour
    label = "Processing photo..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width / 640, height / 360) * 0.7
    thickness = max(1, round(scale))
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    position = ((width - text_width) // 2, (height * 7 // 8) + text_height // 2)
    cv2.putText(screen, label, position, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return screen


def run_preview_loop(
    session: CaptureSession,
    window_name: str = WINDOW_NAME,
    *,
    captures_only: bool = False,
    read_key: Callable[[], str | None] | None = None,
    controller: CaptureController | None = None,
) -> bool:
    """Run the windowed loop. Returns False if this build cannot show a window."""
    if controller is not None and not captures_only:
        raise ValueError("a shared capture controller requires capture-only display mode")
    owned_controller = captures_only and controller is None
    controller = controller or (CaptureController(session) if captures_only else None)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    except cv2.error:
        if owned_controller:
            assert controller is not None
            controller.close()
        return False
    graded = True
    try:
        try:
            # GTK's first imshow resets the drawable to the image's size. Let that
            # allocation finish BEFORE fullscreen, or it can leave a tiny drawable
            # inside a fullscreen (white) outer window. Do not read geometry yet.
            display_size = (640, 360)
            displayed_frame = _capture_prompt(display_size)
            cv2.resizeWindow(window_name, *display_size)
            cv2.imshow(window_name, displayed_frame)
            cv2.waitKey(50)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(50)
            captured_frame = None  # Keep the full-resolution photo for subsequent display resizes.
            active_request_id = None
            displayed_request_id = None
        except cv2.error:
            return False
        while True:
            try:
                size = _window_size(window_name, display_size)
                if size != display_size:
                    cv2.resizeWindow(window_name, *size)
                if captures_only and size != display_size:
                    displayed_frame = (
                        _capture_prompt(size) if captured_frame is None
                        else _fit_display(captured_frame, size)
                    )
                    cv2.imshow(window_name, displayed_frame)
                resized = size != display_size
                display_size = size
                if captures_only:
                    assert controller is not None
                    snapshot = controller.snapshot()
                    # Completed jobs may arrive entirely between GUI polls, so
                    # observing only the active request would miss fast remote shots.
                    job = snapshot.last_completed_job
                    new_photo = job is not None and job.request_id != displayed_request_id
                    if new_photo:
                        assert job is not None
                        assert isinstance(job.result, CaptureResult)
                        _announce(job.result, print)
                        frame, _ = load_rgb(job.result.pifilm)
                        captured_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        displayed_frame = _fit_display(captured_frame, display_size)
                        cv2.imshow(window_name, displayed_frame)
                        displayed_request_id = job.request_id
                    if active_request_id is not None:
                        tracked = controller.status(active_request_id)
                        if tracked is not None and tracked.state == "failed":
                            print(f"error: {tracked.error_message or tracked.error_code}")
                            cv2.imshow(window_name, displayed_frame)
                    active = snapshot.active_job
                    if active is not None:
                        if active.request_id != active_request_id or resized or new_photo:
                            cv2.imshow(window_name, _capture_loading_screen(display_size))
                        active_request_id = active.request_id
                    else:
                        active_request_id = None
                if not captures_only:
                    frame = session.preview_frame(graded)
                    cv2.imshow(window_name, _fit_display(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), display_size,
                    ))
                key = cv2.waitKey(30 if captures_only else 1) & 0xFF
            except CameraError as exc:
                print(f"error: {exc}")
                continue
            except cv2.error:
                return False
            if read_key is not None:
                terminal_key = read_key()
                if terminal_key:
                    key = ord(terminal_key)
            if key == ord(" "):
                try:
                    if captures_only:
                        if active_request_id is not None:
                            continue
                        assert controller is not None
                        job = controller.submit(uuid.uuid4().hex)
                        if job.error_code is not None:
                            print(f"error: {job.error_message or job.error_code}")
                            continue
                        active_request_id = job.request_id
                        cv2.imshow(window_name, _capture_loading_screen(display_size))
                        # imshow queues a repaint; pump GUI events before blocking on capture.
                        cv2.waitKey(1)
                    else:
                        try:
                            result = session.capture()
                            _announce(result, print)
                        except (CameraError, OSError) as exc:
                            print(f"error: {exc}")
                except cv2.error:
                    return False
            elif not captures_only and key in (ord("p"), ord("P")):
                graded = not graded
            elif key in (ord("q"), ord("Q"), 27):
                return True
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if owned_controller:
            assert controller is not None
            controller.close()


class TerminalKeys:
    """Put the terminal in cbreak mode and read single keys without Enter."""

    def __enter__(self) -> TerminalKeys:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


def has_display() -> bool:
    return sys.platform == "darwin" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def _remote_listen(value: str) -> tuple[str, int]:
    """Parse the intentionally simple IPv4/hostname ``host:port`` CLI value."""
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("expected host:port")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be a number") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return host, port


def _colour_gains(value: str) -> tuple[float, float]:
    """Parse the ``R,B`` fixed colour-gains CLI value."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected R,B")
    try:
        r, b = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("colour gains must be numbers") from exc
    if r <= 0 or b <= 0:
        raise argparse.ArgumentTypeError("colour gains must be positive numbers")
    return r, b


def _picamera2_available() -> bool:
    """Probe only the optional runtime package, without opening a camera."""
    try:
        __import__("picamera2")
    except (ImportError, OSError):
        return False
    return True


def _drop_display_on_v4l2(args: argparse.Namespace) -> None:
    """Downgrade ``--display`` to a warning when the backend is V4L2.

    The shipped service unit passes ``--display waveshare28``; on a deployment
    whose camera is USB, refusing the flag would exit 2 and systemd would
    restart the service forever. The viewfinder genuinely cannot run there (the
    V4L2 backend has no preview-mode split and the meter needs libcamera
    metadata), so say so once and serve the Stick.
    """
    if args.display == "none":
        return
    print(
        "warning: display unavailable (the V4L2 backend has no preview mode); "
        "continuing without it",
        file=sys.stderr,
    )
    args.display = "none"


def _reject_picamera2_only_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, reason: str,
) -> None:
    """Refuse Picamera2-only options whenever no real Picamera2 will be opened,
    whether that is V4L2 (however it was chosen) or ``--fake``.

    Covers ``--tuning-file``, ``--autofocus``, ``--af-range``, ``--ae-lock``,
    ``--awb-lock``, ``--colour-gains``, ``--ae-constraint``, ``--ae-metering``
    and ``--ev``. ``--device`` (which itself selects V4L2) and ``--no-dng``
    (wired independently into ``CaptureSession`` and meaningful on any backend)
    are deliberately excluded, and so is ``--display``: it is a warning rather
    than an error (see ``_drop_display_on_v4l2``), and it runs against
    ``FakeCamera`` under ``--fake`` so the panel and its touch mapping can be
    exercised without a sensor.
    """
    picamera_only = [
        ("--tuning-file", args.tuning_file is not None),
        ("--autofocus", args.autofocus is not None),
        ("--af-range", args.af_range is not None),
        ("--ae-lock", args.ae_lock),
        ("--awb-lock", args.awb_lock),
        ("--colour-gains", args.colour_gains is not None),
        ("--ae-constraint", args.ae_constraint is not None),
        ("--ae-metering", args.ae_metering is not None),
        ("--ev", args.ev is not None),
    ]
    offenders = [name for name, given in picamera_only if given]
    if offenders:
        parser.error(f"{', '.join(offenders)} {reason}")


def _open_display(args: argparse.Namespace, out_root: Path):
    """Open the requested display and touch, or None for --display none.

    Raises DisplayError; the caller downgrades that to a warning so a service
    with a broken or absent LCD keeps serving the Stick instead of crash-looping.
    """
    if args.display == "none":
        return None
    if args.display == "fake":
        from ..display.fake import FileDisplay, NoTouch
        return FileDisplay(Path(out_root).expanduser() / "viewfinder-last.png"), NoTouch()
    from ..display.cst3530 import open_waveshare28_touch
    from ..display.st7789 import open_waveshare28
    display = open_waveshare28(args.display_rotate)
    try:
        touch = open_waveshare28_touch(args.display_rotate)
    except DisplayError:
        display.close()
        raise
    return display, touch


def _close_display(display_pair) -> None:
    """Release the panel when the process is giving up before the loop starts."""
    if display_pair is None:
        return
    display, touch = display_pair
    touch.close()
    display.close()


def _serve_until_interrupt() -> int:
    """Hold the process open for the remote API with no terminal controls."""
    print("Remote capture API active without terminal controls. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


def _run_viewfinder(camera, controller, display_pair, power, args) -> int:
    display, touch = display_pair
    stop = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    loop = ViewfinderLoop(
        camera, controller, display, touch,
        power_snapshot=power.snapshot if power is not None else None,
        touch_debug=args.touch_debug,
    )
    print("LCD viewfinder running. Tap the shutter to capture; Ctrl-C to stop.")
    failure: DisplayError | None = None
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        pass
    except DisplayError as exc:
        failure = exc
    finally:
        signal.signal(signal.SIGTERM, previous)
        touch.close()
        display.close()
    if failure is None:
        return 0
    print(f"error: display failed repeatedly: {failure}", file=sys.stderr)
    # A dead screen must cost the screen only. Exiting here would be a restart
    # loop under systemd, and the Stick would get a few seconds of service per
    # cycle; without a remote API there is nothing left to serve, so 1 stands.
    if not args.remote_listen:
        return 1
    return _serve_until_interrupt()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pifilm-capture",
        description="Capture Parr-graded photos from a supported camera.",
    )
    parser.add_argument("--camera", choices=("v4l2", "picamera2"))
    parser.add_argument("--device", default=None, help="index, /dev/videoN or /dev/v4l/by-id/...")
    parser.add_argument(
        "--tuning-file",
        help="Picamera2 tuning filename or absolute path "
             "(default: libcamera's automatic choice for the detected sensor)",
    )
    parser.add_argument(
        "--autofocus", choices=("continuous", "auto", "manual"), default=None,
        help="Picamera2 autofocus mode (default: continuous)",
    )
    parser.add_argument(
        "--af-range", choices=("normal", "macro", "full"), default=None,
        help="Picamera2 autofocus range (default: normal)",
    )
    parser.add_argument(
        "--ae-lock", action="store_true",
        help=(
            "Picamera2: disable auto-exposure at whatever value it holds right "
            "before capture starts (not a converged/settled value); for controlled "
            "shoots (default: auto)"
        ),
    )
    parser.add_argument(
        "--awb-lock", action="store_true",
        help=(
            "Picamera2: disable auto white balance at whatever value it holds right "
            "before capture starts (not a converged/settled value); use --colour-gains "
            "instead to set a known white balance (default: auto)"
        ),
    )
    parser.add_argument(
        "--colour-gains", type=_colour_gains, default=None, metavar="R,B",
        help="Picamera2: fixed colour gains R,B; implies AWB disabled (controlled shoots)",
    )
    parser.add_argument(
        "--ae-constraint", choices=("normal", "highlight", "shadows"), default=None,
        help=(
            "Picamera2 AE constraint: highlight protects bright regions from clipping "
            "(default: normal)"
        ),
    )
    parser.add_argument(
        "--ae-metering", choices=("centre", "spot", "matrix"), default=None,
        help="Picamera2 AE metering (default: centre-weighted)",
    )
    parser.add_argument(
        "--ev", type=float, default=None, metavar="STOPS",
        help="Picamera2 exposure compensation in stops, -8 to 8 (default: 0)",
    )
    parser.add_argument(
        "--no-dng", action="store_true", help="disable the Picamera2 DNG sidecar output",
    )
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-preview", action="store_true", help="disable the live preview")
    parser.add_argument(
        "--show-captures", action="store_true",
        help="display each saved graded photo until the next capture (disables live preview)",
    )
    parser.add_argument("--fake", action="store_true", help="synthetic camera, no hardware")
    parser.add_argument("--seed", type=int, default=None, help="seed the grain seed generator")
    parser.add_argument(
        "--remote-listen", type=_remote_listen, metavar="HOST:PORT",
        help="serve the authenticated remote capture API (requires PIFILM_REMOTE_TOKEN)",
    )
    parser.add_argument(
        "--ups", choices=UPS_CHOICES, default="none",
        help="UPS to read the Pi's battery from and publish in /v1/status (default: none)",
    )
    parser.add_argument(
        "--display", choices=("none", "waveshare28", "fake"), default="none",
        help="LCD viewfinder with exposure meter (Picamera2 or --fake only); "
             "'fake' writes OUT/viewfinder-last.png instead of driving SPI",
    )
    parser.add_argument("--display-rotate", type=int, choices=(0, 180), default=0,
                        help="rotate the LCD image and touch mapping by 180 degrees")
    parser.add_argument("--touch-debug", action="store_true",
                        help="print raw and mapped touch coordinates (orientation check)")
    args = parser.parse_args(argv)

    if args.camera == "picamera2" and args.device is not None:
        parser.error("--device cannot be used with --camera picamera2")
    if args.camera == "v4l2":
        _drop_display_on_v4l2(args)
        _reject_picamera2_only_flags(parser, args, reason="cannot be used with --camera v4l2")

    autofocus = args.autofocus or "continuous"
    af_range = args.af_range or "normal"

    remote_token = os.environ.get("PIFILM_REMOTE_TOKEN") if args.remote_listen else None
    if args.remote_listen and not remote_token:
        print("error: --remote-listen requires PIFILM_REMOTE_TOKEN", file=sys.stderr)
        return 2

    power = None
    try:
        power = build_ups(args.ups)
    except PowerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        pipeline = Pipeline(Artifacts.resolve(args.artifacts))
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # The backend is resolved before the panel is opened, and the panel before
    # the camera. Preview mode is a continuous binned stream plus a mode switch
    # per shot, so it is only worth paying for once there is a panel to feed;
    # opening the camera first would leave a failed display with the camera
    # stuck in preview mode for the whole session, and opening the panel before
    # knowing the backend would claim the SPI bus only to abandon it on V4L2.
    backend = None
    if not args.fake:
        backend = args.camera
        if backend is None:
            backend = "v4l2" if args.device is not None else (
                "picamera2" if _picamera2_available() else "v4l2"
            )
        if backend == "v4l2":
            _drop_display_on_v4l2(args)
    display_pair = None
    if args.display != "none":
        try:
            display_pair = _open_display(args, args.out)
        except DisplayError as exc:
            print(f"warning: display unavailable ({exc}); continuing without it", file=sys.stderr)
    try:
        if args.fake:
            _reject_picamera2_only_flags(parser, args, reason="cannot be used with --fake")
            camera: Camera = FakeCamera()
        else:
            if backend == "picamera2":
                camera = Picamera2Camera(
                    args.tuning_file,
                    save_dng=not args.no_dng,
                    autofocus=autofocus,
                    af_range=af_range,
                    ae_lock=args.ae_lock,
                    awb_lock=args.awb_lock,
                    colour_gains=args.colour_gains,
                    ae_constraint=args.ae_constraint or "normal",
                    ae_metering=args.ae_metering or "centre",
                    ev=args.ev if args.ev is not None else 0.0,
                    preview=True if display_pair is not None else None,
                )
            else:
                _reject_picamera2_only_flags(
                    parser, args, reason="cannot be used with the selected V4L2 backend"
                )
                camera = V4L2Camera(args.device)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _close_display(display_pair)
        return 2

    session = CaptureSession(
        camera,
        pipeline,
        args.out,
        seed_rng=np.random.default_rng(args.seed),
        save_dng=not args.no_dng,
    )
    controller: CaptureController | None = None
    remote: RemoteCaptureServer | None = None
    if display_pair is not None:
        controller = CaptureController(session)
    try:
        if power is not None:
            power.start()
            print(f"Reading the {args.ups} UPS every 10 s.")
        if args.remote_listen:
            controller = controller or CaptureController(session)
            try:
                remote = RemoteCaptureServer(
                    controller, remote_token, args.remote_listen,
                    power=power.snapshot if power else None,
                )
                remote.start()
            except OSError as exc:
                print(f"error: could not start remote listener: {exc}", file=sys.stderr)
                return 2
            host, port = args.remote_listen
            print(f"Remote capture API listening on {host}:{port}.")
            try:
                if display_pair is not None:
                    assert controller is not None
                    return _run_viewfinder(camera, controller, display_pair, power, args)
                if has_display() and (args.show_captures or not args.no_preview):
                    if sys.stdin.isatty():
                        with TerminalKeys() as keys:
                            displayed = run_preview_loop(
                                session, CAPTURE_WINDOW_NAME, captures_only=True,
                                read_key=lambda: keys.read(timeout=0), controller=controller,
                            )
                    else:
                        displayed = run_preview_loop(
                            session, CAPTURE_WINDOW_NAME, captures_only=True, controller=controller,
                        )
                    if displayed:
                        return 0
                    print("Capture display unavailable; remote API remains active.")
                if sys.stdin.isatty():
                    with TerminalKeys() as keys:
                        run_headless_loop(session, keys.read, controller=controller)
                    return 0
                return _serve_until_interrupt()
            except KeyboardInterrupt:
                return 0
        if display_pair is not None:
            assert controller is not None
            return _run_viewfinder(camera, controller, display_pair, power, args)
        if args.show_captures and has_display():
            if sys.stdin.isatty():
                with TerminalKeys() as keys:
                    displayed = run_preview_loop(
                        session, CAPTURE_WINDOW_NAME, captures_only=True,
                        read_key=lambda: keys.read(timeout=0),
                    )
            else:
                displayed = run_preview_loop(session, CAPTURE_WINDOW_NAME, captures_only=True)
            if displayed:
                return 0
            print("Capture display unavailable; falling back to headless mode.")
        elif args.show_captures:
            print("No display attached; falling back to headless mode.")
        elif not args.no_preview and has_display():
            if run_preview_loop(session):
                return 0
            print("Preview unavailable (OpenCV built without GUI); falling back to headless mode.")
        if not sys.stdin.isatty():
            print(
                "error: stdin is not a terminal, so keys cannot be read. Run from a terminal, "
                "or use pifilm-process for files.",
                file=sys.stderr,
            )
            return 2
        with TerminalKeys() as keys:
            run_headless_loop(session, keys.read)
        return 0
    finally:
        if power is not None:
            power.close()
        if remote is not None:
            remote.close()
        if controller is not None:
            controller.close()
        else:
            camera.close()


if __name__ == "__main__":
    sys.exit(main())
