"""Native Raspberry Pi camera acquisition through the optional Picamera2 package.

Sensor-agnostic: the still stream is configured at whatever full resolution the
attached sensor reports (``Picamera2.sensor_resolution``) rather than one model's
hard-coded size, so an IMX708 (4608x2592) and an IMX477 (4056x3040) both capture
their own native frame. Picamera2's ``RGB888`` arrays use BGR byte order, so each
request is copied into an owned RGB array before the request is released.
Importing Picamera2 stays inside construction so USB and fake-camera use do not
require Raspberry Pi camera packages.

Tuning is left to libcamera, which selects the tuning file for the sensor it
detects; the recorded ``tuning_file`` is then ``auto:<Model>`` so a capture record
still names the colour science it was shot with. Passing ``tuning_file`` loads that
file explicitly and records its name, which is how a non-default variant (for
example ``imx708_wide.json`` on a wide Camera Module 3) is pinned.

Not every sensor has a focus motor: the IMX477 has a fixed C/CS mount and its
libcamera control set has no ``AfMode``. Autofocus controls are therefore applied
only when the camera advertises them, the recorded ``autofocus`` becomes ``"none"``,
and asking for a non-default ``autofocus``/``af_range`` on such a sensor is an
error rather than a silently ignored request.

Auto-exposure shaping (constraint mode, metering mode) is exposed as constructor
arguments whose defaults are libcamera's own defaults, so an unflagged capture
behaves exactly as it did before they existed.

``preview`` adds the LCD viewfinder's second, cheap stream. The still
configuration is created, applied and validated *first*, before the preview
configuration replaces it: ``camera_configuration()`` only ever describes what
is currently configured, and the record must describe the shot the camera will
take, not the viewfinder frames it discards. Each full read then uses
``switch_mode_and_capture_request``, which switches to the still configuration,
captures, and returns the camera to the preview configuration itself - so the
viewfinder keeps running afterwards without this module reconfiguring anything.

``preview=True`` derives the preview size from the sensor's own aspect ratio
rather than fixing 4:3: libcamera crops the sensor to the output aspect, so a
640x480 preview on a 16:9 IMX708 would show a centre crop of the frame the
still will actually record. The raw stream is simply asked for half the native
size and libcamera picks the nearest mode it has; ``Picamera2.sensor_modes`` is
deliberately never read, because enumerating it reconfigures the camera through
every raw mode and costs seconds at start-up.

Every ``configure`` resets Picamera2's controls to the configuration's own
``controls`` dict, and ``switch_mode_and_capture_request`` configures twice per
shot. The neutral ISP set therefore travels inside *both* configurations, not
only through ``set_controls``; otherwise the still would be taken with the
ISP's default sharpening and saturation and the viewfinder would come back
without them. A runtime control change has to reach those dicts as well as the
live camera for the same reason: ``set_shutter`` does so for a fixed exposure time
(shutter priority), and also widens and restores ``FrameDurationLimits``;
``set_iso`` does so for a fixed analogue gain (with both fixed the camera is in
full manual); ``set_colour_temperature`` for a fixed white balance.

There is no exposure compensation. ``ExposureValue`` is still sent, always 0.0, so
every capture records the same control set as before the viewfinder's EV buttons
were removed.

Previews and captures arrive on different threads (the viewfinder's frame
thread and the capture controller), so the request lock is held from acquiring
a request through releasing it: the pixel buffer belongs to the request, and a
second thread that captured or released in the middle would hand out or free
memory the first is still copying.

The control setters are called from a third thread (the viewfinder loop) and
take a second lock, the configuration lock, not the request lock. A preview
read holds the request lock for a whole exposure, a second at the slowest
shutter, and a setter that queued behind it would make a tap on ``+`` as slow
as the picture. A setter does not need that exclusion: it touches no request,
and Picamera2's ``set_controls`` is safe beside a capture in flight. What a
setter must not overlap is a still, which configures the camera twice from the
configuration dicts the setters write and whose controls are what the capture
record describes. So a still (``read(full=True)``) holds both locks, always
the request lock first and the configuration lock second, a preview read holds
only the request lock, and a setter holds only the configuration lock. No
thread ever waits for the request lock while holding the configuration lock,
so the two cannot deadlock. ``reassert_controls`` is a setter in this respect,
and ``restart_preview``, which stops and reconfigures the camera, is a still.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import sys
import tempfile
import threading
from typing import Any

import numpy as np

from .camera import CameraError, Frame, StreamInfo
from .errors import CameraStalled

DEFAULT_TUNING_FILE: str | None = None  # None: libcamera picks <sensor>.json itself
_MAIN_FORMAT = "RGB888"
_AUTOFOCUS_MODES = ("continuous", "auto", "manual")
_AF_RANGES = ("normal", "macro", "full")
_AE_CONSTRAINTS = ("normal", "highlight", "shadows")
_AE_METERING = ("centre", "spot", "matrix")
_SHUTTER_RANGE_US = (100, 1_000_000)
# A fixed exposure longer than a frame period is clipped by the sensor unless the
# frame is allowed to last at least that long, plus readout headroom.
_FRAME_MARGIN_US = 1000
# Manual white balance, in kelvin. Measured on the IMX477 on 2026-10-11: a request
# below 2360 K is clamped to 2360 K by the sensor's calibration, and 9000 K is
# accepted as asked; nothing above it was tried.
COLOUR_TEMPERATURE_RANGE_K = (2360, 9000)
# Manual ISO. ISO 100 is analogue gain 1.0, the convention the light meter already
# uses, so the range is gain 1.0 to 16.0; a sensor whose own gain limit is lower
# refuses the values above it (see ``set_iso``).
ISO_RANGE = (100, 1600)
# How long a read waits for its frame before the camera counts as stalled. The
# preview runs at tens of frames a second, so five seconds without one is not
# slowness; a still also pays for a mode switch in each direction. A fixed
# shutter stretches every frame, so its length is added several times over.
# Without a limit a read waits for ever while holding the request lock: seen on
# the Pi 4 on 2026-10-10, when the sensor kept streaming but Picamera2 stopped
# handing frames over, and the viewfinder froze with the shutter queued behind it.
_PREVIEW_TIMEOUT_S = 5.0
_STILL_TIMEOUT_S = 30.0
_TIMEOUT_SHUTTERS = 4

METADATA_KEYS = (
    "ExposureTime",
    "AnalogueGain",
    "DigitalGain",
    "ColourGains",
    "ColourTemperature",
    "Lux",
    "LensPosition",
    "AfState",
    "FocusFoM",
    "FrameDuration",
    "SensorTimestamp",
)


def serialisable_metadata(raw: dict) -> dict:
    out: dict = {}
    for key in METADATA_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, (tuple, list)) and all(isinstance(v, (int, float)) for v in value):
            out[key] = [float(v) for v in value]
        else:
            try:
                out[key] = int(value)  # libcamera enums (e.g. AfState) are int-like
            except (TypeError, ValueError):
                pass
    return out


class Picamera2Camera:
    """Acquire full-sensor RGB frames from any libcamera sensor through Picamera2."""

    def __init__(
        self,
        tuning_file: str | None = DEFAULT_TUNING_FILE,
        save_dng: bool = True,
        autofocus: str = "continuous",
        af_range: str = "normal",
        ae_lock: bool = False,
        awb_lock: bool = False,
        colour_gains: tuple[float, float] | None = None,
        ae_constraint: str = "normal",
        ae_metering: str = "centre",
        preview: bool | tuple[int, int] | None = None,
        colour_temperature: int | None = None,
    ) -> None:
        if autofocus not in _AUTOFOCUS_MODES:
            raise CameraError(
                f"Unknown autofocus mode {autofocus!r}; expected one of {_AUTOFOCUS_MODES}"
            )
        if af_range not in _AF_RANGES:
            raise CameraError(f"Unknown af_range {af_range!r}; expected one of {_AF_RANGES}")
        if ae_constraint not in _AE_CONSTRAINTS:
            raise CameraError(
                f"Unknown ae_constraint {ae_constraint!r}; expected one of {_AE_CONSTRAINTS}"
            )
        if ae_metering not in _AE_METERING:
            raise CameraError(
                f"Unknown ae_metering {ae_metering!r}; expected one of {_AE_METERING}"
            )

        if colour_temperature is not None:
            if awb_lock or colour_gains is not None:
                raise CameraError(
                    "colour_temperature cannot be combined with awb_lock or colour_gains"
                )
            colour_temperature = _colour_temperature_k(colour_temperature)

        self._request_lock = threading.Lock()
        # Held by the control setters and by a still; see the module docstring.
        # Order, where both are taken: request lock, then this one.
        self._config_lock = threading.Lock()
        self.colour_temperature_k: int | None = colour_temperature
        # False when white balance was fixed at start-up, or (set below) when this
        # libcamera has no settable ColourTemperature.
        self.wb_adjustable = not awb_lock and colour_gains is None
        self._wb_refusal = "white balance is fixed by --awb-lock or --colour-gains"
        # False under --ae-lock: with auto-exposure off there is nothing for a
        # fixed shutter or ISO to work against, and stepping one out and back to
        # auto would leave ``AeEnable: False`` beside a mode set to Auto in the
        # configurations. ``set_shutter`` and ``set_iso`` then refuse.
        self.exposure_adjustable = not ae_lock
        self._preview: tuple[int, int] | None = None
        self.preview_size: tuple[int, int] | None = None
        # Exposure compensation is gone, but the attribute stays: ``CaptureSession``
        # reads it, so every record still carries ``ev_comp: 0.0`` and old records
        # (which may hold another value) keep their meaning.
        self.ev = 0.0
        self.shutter_us: int | None = None
        self.iso: int | None = None
        self.max_gain: float | None = None
        self._shutter_mode = "legacy"
        self._iso_mode = "legacy"
        self._frame_limits: dict[str, tuple[int, int] | None] = {"still": None, "preview": None}
        self._still_config: Any | None = None
        self._preview_config: Any | None = None

        try:
            from picamera2 import Picamera2
        except (ImportError, OSError) as exc:
            raise CameraError(
                "Picamera2 is unavailable; install the Raspberry Pi OS python3-picamera2 package"
            ) from exc

        tuning = None
        if tuning_file is not None:
            try:
                tuning = Picamera2.load_tuning_file(tuning_file)
            except Exception as exc:
                raise CameraError(
                    f"Failed to load Picamera2 tuning file {tuning_file!r}: {exc}"
                ) from exc
        try:
            camera = Picamera2(tuning=tuning) if tuning is not None else Picamera2()
        except Exception as exc:
            raise CameraError(f"Failed to open Picamera2 camera: {exc}") from exc

        self._camera: Any | None = camera
        self._started = False
        self._save_dng = save_dng
        self._autofocus = autofocus
        start_attempted = False
        try:
            native = tuple(int(v) for v in camera.sensor_resolution)
            if len(native) != 2 or min(native) <= 0:
                raise CameraError(f"Picamera2 reported an unusable sensor resolution {native!r}")
            self._native_size: tuple[int, int] = (native[0], native[1])
            # Read straight off the camera, like sensor_resolution above: a
            # Picamera2 that cannot describe itself is a broken camera, and
            # defaulting these would report a missing control set as
            # "<model> has no autofocus", pointing the user at their --autofocus
            # flag instead of the real fault. A missing attribute reaches the
            # generic wrapper below instead.
            model = str(camera.camera_properties.get("Model", "unknown"))
            tuning_label = tuning_file if tuning_file is not None else f"auto:{model}"
            has_autofocus = "AfMode" in camera.camera_controls
            # Shutter priority: libcamera 0.5 (Trixie) ignores ExposureTime unless
            # ExposureTimeMode is Manual; older stacks take a non-zero ExposureTime with
            # AE on. Either way analogue gain stays automatic.
            self._shutter_mode = (
                "mode" if "ExposureTimeMode" in camera.camera_controls else "legacy"
            )
            print(
                "picamera2: shutter priority via "
                + ("ExposureTimeMode" if self._shutter_mode == "mode"
                   else "ExposureTime (legacy libcamera)"),
                file=sys.stderr,
            )
            # Manual ISO takes the same two paths: AnalogueGainMode where libcamera
            # has it, else a non-zero AnalogueGain (zero hands the gain back).
            self._iso_mode = (
                "mode" if "AnalogueGainMode" in camera.camera_controls else "legacy"
            )
            print(
                "picamera2: ISO via "
                + ("AnalogueGainMode" if self._iso_mode == "mode"
                   else "AnalogueGain (legacy libcamera)"),
                file=sys.stderr,
            )
            gain = camera.camera_controls.get("AnalogueGain")
            if isinstance(gain, (tuple, list)) and len(gain) >= 2:
                self.max_gain = float(gain[1])
            if not has_autofocus and (autofocus != "continuous" or af_range != "normal"):
                raise CameraError(
                    f"{model} has no autofocus; --autofocus and --af-range cannot be used"
                )
            if preview is True:
                self._preview = _preview_size(self._native_size)
            elif preview:
                self._preview = (int(preview[0]), int(preview[1]))
            self.preview_size = self._preview
            controls = _camera_controls(
                autofocus, af_range, ae_lock, awb_lock, colour_gains,
                ae_constraint, ae_metering, has_autofocus=has_autofocus,
            )
            if "ColourTemperature" not in camera.camera_controls:
                if colour_temperature is not None:
                    raise CameraError(
                        "this libcamera has no settable ColourTemperature control"
                    )
                if self.wb_adjustable:
                    self.wb_adjustable = False
                    self._wb_refusal = "this libcamera has no settable ColourTemperature control"
            if colour_temperature is not None:
                controls["ColourTemperature"] = colour_temperature
            config = camera.create_still_configuration(
                main={"size": self._native_size, "format": _MAIN_FORMAT},
                raw={"size": self._native_size},
                buffer_count=2,
                queue=False,
                controls=dict(controls),
            )
            camera.configure(config)
            actual = camera.camera_configuration()
            self._stream_info = _stream_info(actual, tuning_label, self._native_size)
            self._stream_info.autofocus = autofocus if has_autofocus else "none"
            self._still_config = config
            if self._preview is not None:
                preview_config = camera.create_preview_configuration(
                    main={"size": self._preview, "format": _MAIN_FORMAT},
                    raw={"size": _binned_mode(self._native_size)},
                    controls=dict(controls),
                )
                camera.configure(preview_config)
                self._preview_config = preview_config
            # A configuration built without FrameDurationLimits falls back to the
            # camera's own control range, so a long shutter can still widen it.
            default_limits = _control_frame_limits(camera.camera_controls)
            self._frame_limits = {
                key: (_config_frame_limits(config) or default_limits)
                if config is not None else None
                for key, config in (("still", self._still_config),
                                    ("preview", self._preview_config))
            }
            camera.set_controls(controls)
            start_attempted = True
            camera.start()
            self._started = True
        except Exception as exc:
            _cleanup_camera(camera, stop=start_attempted)
            self._camera = None
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Failed to configure or start Picamera2 camera: {exc}") from exc

    @property
    def stream_info(self) -> StreamInfo:
        return self._stream_info

    def read(self, *, full: bool = True) -> Frame:
        """Acquire one frame. ``full=False`` (preview) skips the autofocus cycle
        and DNG extraction, which otherwise make each read too slow for a live
        preview; the request is still captured, converted and released, and
        metadata is still read, so ``preview_frame`` keeps working.

        Every read holds the request lock. A still also holds the configuration
        lock, so no control setter changes the camera or the configurations
        under it; a preview read does not, so a setter never waits out a long
        preview exposure."""
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot read from a closed Picamera2 camera")

        config_lock = self._config_lock if full else contextlib.nullcontext()
        with self._request_lock, config_lock:
            if full and self._autofocus == "auto" and self._stream_info.autofocus != "none":
                try:
                    camera.autofocus_cycle()
                except Exception as exc:
                    raise CameraError(f"Autofocus cycle failed: {exc}") from exc

            still = full or self._preview is None
            timeout = self._frame_timeout(still)
            try:
                if full and self._preview is not None:
                    # Switches to the still configuration, captures, and restores
                    # the preview configuration itself.
                    request = camera.switch_mode_and_capture_request(
                        self._still_config, wait=timeout,
                    )
                else:
                    request = camera.capture_request(wait=timeout)
            except TimeoutError:
                raise self._stalled(camera, still, timeout) from None
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to capture a request: {exc}") from exc

            if full or self._preview is None:
                expected_shape = (self._native_size[1], self._native_size[0], 3)
            else:
                expected_shape = (self._preview[1], self._preview[0], 3)

            processing_failed = False
            try:
                try:
                    bgr = request.make_array("main")
                    if not isinstance(bgr, np.ndarray):
                        raise CameraError("Picamera2 main array is not a NumPy array")
                    if bgr.dtype != np.uint8 or bgr.shape != expected_shape:
                        raise CameraError(
                            "Picamera2 main array must be "
                            f"uint8 with shape {expected_shape}, got {bgr.dtype} {bgr.shape}"
                        )
                    rgb = np.array(bgr[..., ::-1], dtype=np.uint8, order="C", copy=True)
                    metadata = serialisable_metadata(request.get_metadata())
                    if full:
                        # stream_info is the audit of the still (it is written
                        # into every capture record), so only a still may set
                        # its measured rate: in preview mode the cheap reads
                        # come from the binned viewfinder stream and would
                        # otherwise make a record claim the wrong frame rate.
                        self._update_fps({"FrameDuration": metadata.get("FrameDuration", 0)})
                    dng = None
                    if full and self._save_dng:
                        # The ".dng" suffix is load-bearing: PiDNG (used by save_dng)
                        # appends ".dng" to a path that lacks it, which would write
                        # the payload to a different file than this one and leave
                        # this handle's read empty.
                        with tempfile.NamedTemporaryFile(suffix=".dng", delete=True) as tmp:
                            request.save_dng(tmp.name)
                            tmp.seek(0)
                            dng = tmp.read()
                    return Frame(
                        rgb=rgb, jpeg=None, source="picamera2", metadata=metadata, dng=dng
                    )
                except CameraError:
                    processing_failed = True
                    raise
                except Exception as exc:
                    processing_failed = True
                    raise CameraError(f"Picamera2 failed while reading a frame: {exc}") from exc
            finally:
                try:
                    request.release()
                except Exception as exc:
                    if not processing_failed:
                        raise CameraError(
                            f"Picamera2 failed to release a capture request: {exc}"
                        ) from exc

    def _frame_timeout(self, still: bool) -> float:
        base = _STILL_TIMEOUT_S if still else _PREVIEW_TIMEOUT_S
        return base + _TIMEOUT_SHUTTERS * (self.shutter_us or 0) / 1_000_000

    @staticmethod
    def _stalled(
        camera: Any, still: bool, timeout: float, what: str | None = None,
    ) -> CameraStalled:
        """Build the error for a read that timed out, and clear Picamera2's queue.

        ``what`` names the thing that did not arrive when it was not a frame
        (the live view restart).

        The timed-out job stays queued inside Picamera2; left there it would take
        the next frame and nobody would release it. The state goes into the
        message because the log is all there is to diagnose a stall from.
        """
        def state(name: str, convert: Any = lambda value: value) -> Any:
            try:
                return convert(getattr(camera, name))
            except Exception:
                return "?"

        detail = (
            f"started={state('started')} completed={state('completed_requests', len)} "
            f"frames={state('frames')}"
        )
        try:
            camera.cancel_all_and_flush()
        except Exception:
            pass
        what = what or ("still frame" if still else "preview frame")
        return CameraStalled(f"no {what} within {timeout:.1f} s ({detail})")

    def _update_fps(self, metadata: Any) -> None:
        try:
            duration = float(metadata.get("FrameDuration", 0))
        except (AttributeError, TypeError, ValueError):
            return
        if duration > 0 and math.isfinite(duration):
            self._stream_info.fps = 1_000_000.0 / duration

    def set_shutter(self, us: int | None) -> None:
        """Fix the exposure time (shutter priority), or return to auto with ``None``.

        Gain stays automatic unless ``set_iso`` has fixed it too. The controls go
        to the running camera and into both configurations, because every
        configure (including the two inside ``switch_mode_and_capture_request``)
        reapplies a configuration's own controls, so a change that only reached
        the camera would be undone by the next shot. A long shutter also widens
        the frame-duration limit, which is why the live view slows at long
        shutters, and ``None`` restores the limits each configuration was built
        with. Nothing changes if the camera rejects the controls or they cannot
        be built. Under ``--ae-lock`` (``exposure_adjustable`` False) every call
        is refused, ``None`` included, as ``set_colour_temperature`` refuses
        under a fixed white balance. ``us`` must be ``None`` or a whole number of microseconds: an
        integral float such as ``4000.0`` is accepted as ``4000``, while a bool,
        a string or a fractional number is a ``CameraError``.
        """
        if us is not None:
            us = _shutter_us(us)
            if not _SHUTTER_RANGE_US[0] <= us <= _SHUTTER_RANGE_US[1]:
                raise CameraError(f"shutter must be within {_SHUTTER_RANGE_US} us, got {us}")
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot set the shutter on a closed Picamera2 camera")
        if not self.exposure_adjustable:
            raise CameraError("Cannot set the shutter: auto-exposure is locked by --ae-lock")
        running = "preview" if self._preview_config is not None else "still"
        with self._config_lock:
            try:
                # Built inside the try, every one of them before anything is
                # sent: the mode enum is looked up lazily, and a lookup that
                # fails must be a CameraError with nothing changed.
                per_config = {
                    key: self._shutter_controls(us, self._frame_limits[key])
                    for key in ("still", "preview")
                }
                camera.set_controls(per_config[running])
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to set the shutter: {exc}") from exc
            for key, config in (("still", self._still_config),
                                ("preview", self._preview_config)):
                for name, value in per_config[key].items():
                    _set_config_control(config, name, value)
                if us is None and self._shutter_mode == "mode":
                    # Auto sends no ExposureTime, but the one a fixed shutter
                    # wrote would stay in the configuration and be reapplied
                    # beside ExposureTimeMode=Auto on every configure.
                    _drop_config_control(config, "ExposureTime")
        self.shutter_us = us

    def set_iso(self, iso: int | None) -> None:
        """Fix the ISO (analogue gain), or return to auto with ``None``.

        ISO 100 is ``AnalogueGain`` 1.0. Measured on the IMX477 with libcamera
        0.7.2 on 2026-10-11: a fixed gain holds while the exposure time stays
        automatic, and both can be fixed at once (full manual). Like
        ``set_shutter``, the controls go to the running camera and into both
        configurations, because every configure reapplies a configuration's own
        controls. Nothing changes if the camera rejects the controls or they
        cannot be built, and under ``--ae-lock`` every call is refused. ``iso`` must
        be ``None`` or a whole number inside ``ISO_RANGE`` whose gain the sensor
        can reach: an integral float such as ``400.0`` is accepted as ``400``,
        while a bool, a string or a fractional number is a ``CameraError``.
        """
        if iso is not None:
            iso = _iso(iso)
            if self.max_gain is not None and iso / 100 > self.max_gain:
                raise CameraError(
                    f"ISO {iso} needs analogue gain {iso / 100:g}, above this sensor's "
                    f"limit of {self.max_gain:g}"
                )
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot set the ISO on a closed Picamera2 camera")
        if not self.exposure_adjustable:
            raise CameraError("Cannot set the ISO: auto-exposure is locked by --ae-lock")
        with self._config_lock:
            try:
                # Built inside the try, like the shutter's: the mode enum is
                # looked up lazily and a failure there must be a CameraError.
                controls = self._iso_controls(iso)
                camera.set_controls(controls)
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to set the ISO: {exc}") from exc
            for config in (self._still_config, self._preview_config):
                for name, value in controls.items():
                    _set_config_control(config, name, value)
                if iso is None and self._iso_mode == "mode":
                    # Auto sends no AnalogueGain, but the one a fixed ISO wrote
                    # would stay in the configuration and be reapplied beside
                    # AnalogueGainMode=Auto on every configure.
                    _drop_config_control(config, "AnalogueGain")
        self.iso = iso

    def set_colour_temperature(self, kelvin: int | None) -> None:
        """Fix the white balance at ``kelvin``, or return to auto with ``None``.

        Measured on the IMX477 with libcamera 0.7.2 on 2026-10-11:
        ``ColourTemperature`` by itself fixes the colour gains, and ``AwbEnable``
        hands them back to the algorithm. Like ``set_shutter``, the control goes to
        the running camera and into both configurations, because every configure
        reapplies a configuration's own controls. The two controls are never left
        in a configuration together, so which one wins never depends on the order
        libcamera applies them in. Nothing changes if the camera rejects the control.
        """
        if kelvin is not None:
            kelvin = _colour_temperature_k(kelvin)
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot set the white balance on a closed Picamera2 camera")
        if not self.wb_adjustable:
            raise CameraError(f"Cannot set the white balance: {self._wb_refusal}")
        if kelvin is None:
            name, value, stale = "AwbEnable", True, "ColourTemperature"
        else:
            name, value, stale = "ColourTemperature", kelvin, "AwbEnable"
        with self._config_lock:
            try:
                camera.set_controls({name: value})
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to set the white balance: {exc}") from exc
            for config in (self._still_config, self._preview_config):
                _set_config_control(config, name, value)
                _drop_config_control(config, stale)
        self.colour_temperature_k = kelvin

    def reassert_controls(self) -> None:
        """Send the current selection to the running camera again, in one call.

        On the Pi 4 a control sent to the running camera has been seen not to
        take effect: a white balance was selected and the live view kept the
        old colour until a photo was taken (a still re-applies the whole
        configuration, which is why the photo was right). The cause is not
        known, so the viewfinder checks each frame's metadata against the
        selection and calls this when they keep disagreeing.

        What is sent is what the setters sent live: the shutter and ISO controls
        for ``shutter_us`` and ``iso`` (auto included), unless ``--ae-lock``
        fixed the exposure, and ``ColourTemperature`` or ``AwbEnable``, unless
        the white balance is not adjustable. The configurations already hold
        the selection and are not touched, nor is any attribute. With nothing
        adjustable nothing is sent.
        """
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot resend the controls on a closed Picamera2 camera")
        running = "preview" if self._preview_config is not None else "still"
        with self._config_lock:
            try:
                controls: dict[str, Any] = {}
                if self.exposure_adjustable:
                    controls.update(
                        self._shutter_controls(self.shutter_us, self._frame_limits[running])
                    )
                    controls.update(self._iso_controls(self.iso))
                if self.wb_adjustable:
                    if self.colour_temperature_k is None:
                        controls["AwbEnable"] = True
                    else:
                        controls["ColourTemperature"] = self.colour_temperature_k
                if controls:
                    camera.set_controls(controls)
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to resend the controls: {exc}") from exc

    def restart_preview(self) -> None:
        """Stop the camera, apply the preview configuration again and restart it.

        The remedy for a live control the camera ignores that a resend (see
        ``reassert_controls``) does not cure: on the Pi 4 a white balance sent
        after a photo is usually ignored, a fixed value and a return to auto
        alike, until the next photo. What the photo does that a resend does not
        is reconfigure the camera, and this is that reconfiguration without
        the photo; on 2026-10-12 it cured the fault 3 times out of 3. The
        viewfinder uses it at once for white balance and after two resends for
        shutter and ISO. The preview
        configuration's controls already hold the selection, because the
        setters write them there, so nothing else is sent.

        It stops the camera, so like a still it holds both locks, the request
        lock first: no read is in flight and no setter runs meanwhile. It
        therefore waits for a preview read in progress, up to one exposure.
        It waits as long as a still does; no answer in that time is a stalled
        camera. Only a camera with a preview stream has a live view to restart.
        """
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot restart the live view on a closed Picamera2 camera")
        if self._preview_config is None:
            raise CameraError("Cannot restart the live view: this camera has no preview stream")
        timeout = self._frame_timeout(True)
        with self._request_lock, self._config_lock:
            try:
                camera.switch_mode(self._preview_config, wait=timeout)
            except TimeoutError:
                raise self._stalled(camera, True, timeout, "live view restart") from None
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to restart the live view: {exc}") from exc

    def _shutter_controls(
        self, us: int | None, limits: tuple[int, int] | None,
    ) -> dict[str, Any]:
        controls: dict[str, Any] = {}
        if self._shutter_mode == "mode":
            from libcamera import controls as _lc

            mode = _lc.ExposureTimeModeEnum
            if us is None:
                controls["ExposureTimeMode"] = mode.Auto
            else:
                controls["ExposureTimeMode"] = mode.Manual
                controls["ExposureTime"] = us
        else:
            controls["ExposureTime"] = 0 if us is None else us
        if limits is not None:
            low, high = limits
            controls["FrameDurationLimits"] = (
                (low, high) if us is None else (low, max(high, us + _FRAME_MARGIN_US))
            )
        return controls

    def _iso_controls(self, iso: int | None) -> dict[str, Any]:
        if self._iso_mode == "mode":
            from libcamera import controls as _lc

            mode = _lc.AnalogueGainModeEnum
            if iso is None:
                return {"AnalogueGainMode": mode.Auto}
            return {"AnalogueGainMode": mode.Manual, "AnalogueGain": iso / 100}
        return {"AnalogueGain": 0.0 if iso is None else iso / 100}

    def close(self) -> None:
        camera = self._camera
        if camera is None:
            return
        self._camera = None
        started = self._started
        self._started = False

        first_error: Exception | None = None
        if started:
            try:
                camera.stop()
            except Exception as exc:
                first_error = exc
        try:
            camera.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise CameraError(f"Failed to close Picamera2 camera: {first_error}") from first_error


PREVIEW_WIDTH = 640


def _binned_mode(native: tuple[int, int]) -> tuple[int, int]:
    """Half the native size, for libcamera to match to a real binned mode.

    The viewfinder runs the sensor here: cheap frames at the sensor's binned
    rate, with the full-resolution still taken by a mode switch per shot.
    Asking ``Picamera2.sensor_modes`` which modes exist would be more precise
    and costs seconds, because reading it configures the camera once per raw
    mode; libcamera picks the nearest mode to this request by itself.
    """
    return (native[0] // 2, native[1] // 2)


def _preview_size(native: tuple[int, int]) -> tuple[int, int]:
    """A ``PREVIEW_WIDTH``-wide stream with the sensor's own aspect ratio.

    libcamera crops the sensor to the requested output aspect, so a fixed 4:3
    preview on a 16:9 sensor would frame the viewfinder differently from the
    still it is composing. The height is rounded to an even number because
    subsampled YUV/raw pipelines want even dimensions.
    """
    height = int(round(PREVIEW_WIDTH * native[1] / native[0]))
    return PREVIEW_WIDTH, max(2, height - (height % 2))


def _config_frame_limits(config: Any) -> tuple[int, int] | None:
    """The FrameDurationLimits a configuration was built with, if it has any."""
    try:
        low, high = config["controls"]["FrameDurationLimits"]
        return int(low), int(high)
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _control_frame_limits(camera_controls: Any) -> tuple[int, int] | None:
    """The camera's own FrameDurationLimits range, a Picamera2 (min, max, default)."""
    try:
        entry = camera_controls["FrameDurationLimits"]
        return int(entry[0]), int(entry[1])
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _colour_temperature_k(value: Any) -> int:
    """A whole number of kelvin inside the supported range, or ``CameraError``."""
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise CameraError(f"colour temperature must be a whole number of kelvin, got {value!r}")
    low, high = COLOUR_TEMPERATURE_RANGE_K
    if not low <= value <= high:
        raise CameraError(f"colour temperature must be within {low}-{high} K, got {value}")
    return int(value)


def _shutter_us(value: Any) -> int:
    """A whole number of microseconds, or ``CameraError`` for anything else."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise CameraError(f"shutter must be a whole number of microseconds, got {value!r}")


def _iso(value: Any) -> int:
    """A whole-number ISO inside ``ISO_RANGE``, or ``CameraError`` for anything else."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        iso = int(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        iso = int(value)
    else:
        raise CameraError(f"ISO must be a whole number, got {value!r}")
    low, high = ISO_RANGE
    if not low <= iso <= high:
        raise CameraError(f"ISO must be within {low}-{high}, got {iso}")
    return iso


def _set_config_control(config: Any, name: str, value: Any) -> None:
    """Update one control inside a Picamera2 configuration, if it has any."""
    if config is None:
        return
    try:
        config["controls"][name] = value
    except (TypeError, KeyError, IndexError):  # a configuration without controls
        pass


def _drop_config_control(config: Any, name: str) -> None:
    """Remove one control from a Picamera2 configuration, if it is there."""
    if config is None:
        return
    try:
        config["controls"].pop(name, None)
    except (TypeError, KeyError, IndexError, AttributeError):  # no controls dict
        pass


def _stream_info(actual: Any, tuning_file: str, native_size: tuple[int, int]) -> StreamInfo:
    try:
        main_size = tuple(actual["main"]["size"])
        main_format = actual["main"]["format"]
        raw_size = tuple(actual["raw"]["size"])
        raw_format = actual["raw"]["format"]
        bit_depth = actual["sensor"]["bit_depth"]
    except (KeyError, TypeError) as exc:
        raise CameraError(
            f"Picamera2 returned an incomplete negotiated configuration: {actual!r}"
        ) from exc

    mismatches = []
    if main_size != native_size:
        mismatches.append(f"main size {main_size!r}")
    if main_format != _MAIN_FORMAT:
        mismatches.append(f"main format {main_format!r}")
    if raw_size != native_size:
        mismatches.append(f"raw size {raw_size!r}")
    if not isinstance(raw_format, str) or not raw_format:
        mismatches.append(f"raw format {raw_format!r}")
    if not isinstance(bit_depth, int) or isinstance(bit_depth, bool) or bit_depth <= 0:
        mismatches.append(f"sensor bit depth {bit_depth!r}")
    if mismatches:
        details = ", ".join(mismatches)
        raise CameraError(f"Picamera2 negotiated an unsupported capture configuration: {details}")

    width, height = main_size
    return StreamInfo(
        width=width,
        height=height,
        fps=0.0,
        fourcc=main_format,
        raw_mjpeg=False,
        sensor_mode=f"{raw_size[0]}x{raw_size[1]} {raw_format}",
        bit_depth=bit_depth,
        tuning_file=tuning_file,
    )


def _warn_missing_ae_enum(control: str, flag: str, requested: str, default: str) -> None:
    """Report an AE control this libcamera cannot set, but only when it matters.

    The defaults are libcamera's own, so their absence changes nothing and stays
    silent; a requested non-default that cannot be applied is a surprise worth
    one stderr line.
    """
    if requested == default:
        return
    print(
        f"warning: this libcamera has no {control}, so {flag} {requested} was not applied; "
        f"the camera keeps its default ({default})",
        file=sys.stderr,
    )


def _camera_controls(
    autofocus: str,
    af_range: str,
    ae_lock: bool,
    awb_lock: bool,
    colour_gains: tuple[float, float] | None,
    ae_constraint: str,
    ae_metering: str,
    has_autofocus: bool,
) -> dict:
    """Neutral ISP rendering plus autofocus and AE shaping, as one control dict.

    Returned rather than applied so the same set can be handed to
    ``create_still_configuration`` and ``create_preview_configuration`` as well
    as to ``set_controls``: Picamera2 reapplies a configuration's controls on
    every ``configure``, and a shot in preview mode configures twice.

    Imports libcamera's control enums lazily so USB and fake-camera use never
    require the Raspberry Pi camera stack. ``autofocus``/``af_range``/
    ``ae_constraint``/``ae_metering`` are validated by the caller, so the dict
    lookups below cannot raise ``KeyError``.
    """
    from libcamera import controls as _lc

    cam_controls = {"Sharpness": 1.0, "Contrast": 1.0, "Saturation": 1.0}
    try:
        cam_controls["NoiseReductionMode"] = _lc.draft.NoiseReductionModeEnum.HighQuality
    except AttributeError:
        pass  # older libcamera: leave NR at its default, recorded as unset
    # A fixed-lens sensor (IMX477) has no AfMode control at all; setting one would
    # be rejected by libcamera, so the caller detects the capability and the two
    # focus controls are simply not part of this camera's control set.
    if has_autofocus:
        af_modes = {"continuous": _lc.AfModeEnum.Continuous,
                    "auto": _lc.AfModeEnum.Auto, "manual": _lc.AfModeEnum.Manual}
        af_ranges = {"normal": _lc.AfRangeEnum.Normal,
                     "macro": _lc.AfRangeEnum.Macro, "full": _lc.AfRangeEnum.Full}
        cam_controls["AfMode"] = af_modes[autofocus]
        cam_controls["AfRange"] = af_ranges[af_range]
    # Auto-exposure shaping. Centre-weighted metering with the normal constraint is
    # libcamera's own default, so the defaults here reproduce previous behaviour
    # exactly; "highlight" protects a bright sky or wall that centre-weighting would
    # blow out while metering a shaded foreground (measured: shot 171656 on the
    # IMX708 pilot). Enum lookups are guarded like NoiseReductionMode above because
    # older libcamera builds lack them.
    # A missing enum silently leaves libcamera's default in force. That is
    # harmless for the defaults (they *are* libcamera's defaults) but not for a
    # requested non-default: the user asked for highlight protection and would
    # otherwise believe they got it, so say so on stderr.
    try:
        constraint_modes = {"normal": _lc.AeConstraintModeEnum.Normal,
                            "highlight": _lc.AeConstraintModeEnum.Highlight,
                            "shadows": _lc.AeConstraintModeEnum.Shadows}
        cam_controls["AeConstraintMode"] = constraint_modes[ae_constraint]
    except AttributeError:
        _warn_missing_ae_enum("AeConstraintMode", "--ae-constraint", ae_constraint, "normal")
    try:
        metering_modes = {"centre": _lc.AeMeteringModeEnum.CentreWeighted,
                          "spot": _lc.AeMeteringModeEnum.Spot,
                          "matrix": _lc.AeMeteringModeEnum.Matrix}
        cam_controls["AeMeteringMode"] = metering_modes[ae_metering]
    except AttributeError:
        _warn_missing_ae_enum("AeMeteringMode", "--ae-metering", ae_metering, "centre")
    # 0.0 is neutral and nothing changes it any more (exposure compensation was
    # removed with the viewfinder's EV buttons), but it is still set so every
    # capture records the same control set as the ones taken before.
    cam_controls["ExposureValue"] = 0.0
    if colour_gains is not None:
        cam_controls["AwbEnable"] = False
        cam_controls["ColourGains"] = tuple(colour_gains)
    elif awb_lock:
        cam_controls["AwbEnable"] = False
    if ae_lock:
        cam_controls["AeEnable"] = False
    return cam_controls


def _cleanup_camera(camera: Any, *, stop: bool) -> None:
    """Best-effort constructor cleanup that cannot mask the setup failure."""
    if stop:
        try:
            camera.stop()
        except Exception:
            pass
    try:
        camera.close()
    except Exception:
        pass
