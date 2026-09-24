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

Auto-exposure shaping (constraint mode, metering mode, exposure compensation) is
exposed as constructor arguments whose defaults are libcamera's own defaults, so
an unflagged capture behaves exactly as it did before they existed.

``preview`` adds the LCD viewfinder's second, cheap stream. The still
configuration is created, applied and validated *first*, before the preview
configuration replaces it: ``camera_configuration()`` only ever describes what
is currently configured, and the record must describe the shot the camera will
take, not the viewfinder frames it discards. Each full read then uses
``switch_mode_and_capture_request``, which switches to the still configuration,
captures, and returns the camera to the preview configuration itself - so the
viewfinder keeps running afterwards without this module reconfiguring anything.

Previews and captures arrive on different threads (the viewfinder loop and the
capture controller), so one lock is held from acquiring a request through
releasing it: the pixel buffer belongs to the request, and a second thread that
captured or released in the middle would hand out or free memory the first is
still copying. ``set_ev`` takes the same lock so a control change never lands
between a capture and its release.
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
from typing import Any

import numpy as np

from .camera import CameraError, Frame, StreamInfo

DEFAULT_TUNING_FILE: str | None = None  # None: libcamera picks <sensor>.json itself
_MAIN_FORMAT = "RGB888"
_AUTOFOCUS_MODES = ("continuous", "auto", "manual")
_AF_RANGES = ("normal", "macro", "full")
_AE_CONSTRAINTS = ("normal", "highlight", "shadows")
_AE_METERING = ("centre", "spot", "matrix")
_EV_RANGE = (-8.0, 8.0)

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
        ev: float = 0.0,
        preview: tuple[int, int] | None = None,
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
        if not _EV_RANGE[0] <= float(ev) <= _EV_RANGE[1]:
            raise CameraError(f"ev must be within {_EV_RANGE}, got {ev}")

        self._request_lock = threading.Lock()
        self._preview = tuple(preview) if preview else None
        self.ev = float(ev)
        self._still_config: Any | None = None

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
            model = str(getattr(camera, "camera_properties", {}).get("Model", "unknown"))
            tuning_label = tuning_file if tuning_file is not None else f"auto:{model}"
            has_autofocus = "AfMode" in getattr(camera, "camera_controls", {})
            if not has_autofocus and (autofocus != "continuous" or af_range != "normal"):
                raise CameraError(
                    f"{model} has no autofocus; --autofocus and --af-range cannot be used"
                )
            config = camera.create_still_configuration(
                main={"size": self._native_size, "format": _MAIN_FORMAT},
                raw={"size": self._native_size},
                buffer_count=2,
                queue=False,
            )
            camera.configure(config)
            actual = camera.camera_configuration()
            self._stream_info = _stream_info(actual, tuning_label, self._native_size)
            self._stream_info.autofocus = autofocus if has_autofocus else "none"
            self._still_config = config
            if self._preview is not None:
                binned = _binned_mode(getattr(camera, "sensor_modes", []), self._native_size)
                preview_config = camera.create_preview_configuration(
                    main={"size": self._preview, "format": _MAIN_FORMAT},
                    raw={"size": binned},
                )
                camera.configure(preview_config)
            _apply_camera_controls(
                camera, autofocus, af_range, ae_lock, awb_lock, colour_gains,
                ae_constraint, ae_metering, ev, has_autofocus=has_autofocus,
            )
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
        metadata is still read, so ``preview_frame`` keeps working."""
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot read from a closed Picamera2 camera")

        with self._request_lock:
            if full and self._autofocus == "auto" and self._stream_info.autofocus != "none":
                try:
                    camera.autofocus_cycle()
                except Exception as exc:
                    raise CameraError(f"Autofocus cycle failed: {exc}") from exc

            try:
                if full and self._preview is not None:
                    # Switches to the still configuration, captures, and restores
                    # the preview configuration itself.
                    request = camera.switch_mode_and_capture_request(self._still_config)
                else:
                    request = camera.capture_request()
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

    def _update_fps(self, metadata: Any) -> None:
        try:
            duration = float(metadata.get("FrameDuration", 0))
        except (AttributeError, TypeError, ValueError):
            return
        if duration > 0 and math.isfinite(duration):
            self._stream_info.fps = 1_000_000.0 / duration

    def set_ev(self, value: float) -> None:
        """Change exposure compensation on the running camera (viewfinder EV buttons)."""
        value = float(value)
        if not _EV_RANGE[0] <= value <= _EV_RANGE[1]:
            raise CameraError(f"ev must be within {_EV_RANGE}, got {value}")
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot set EV on a closed Picamera2 camera")
        with self._request_lock:
            try:
                camera.set_controls({"ExposureValue": value})
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to set ExposureValue: {exc}") from exc
        self.ev = value

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


def _binned_mode(sensor_modes: Any, native: tuple[int, int]) -> tuple[int, int]:
    """The largest sensor mode no bigger than half the native size in each axis.

    The viewfinder runs the sensor here: cheap frames at the sensor's binned
    rate, with the full-resolution still taken by a mode switch per shot. When
    the driver lists no modes, half the native size is requested and libcamera
    picks the nearest.
    """
    half = (native[0] // 2, native[1] // 2)
    best: tuple[int, int] | None = None
    for mode in sensor_modes or []:
        try:
            w, h = (int(v) for v in mode["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= half[0] and h <= half[1] and (best is None or w * h > best[0] * best[1]):
            best = (w, h)
    return best or half


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


def _apply_camera_controls(
    camera: Any,
    autofocus: str,
    af_range: str,
    ae_lock: bool,
    awb_lock: bool,
    colour_gains: tuple[float, float] | None,
    ae_constraint: str,
    ae_metering: str,
    ev: float,
    has_autofocus: bool,
) -> None:
    """Neutral ISP rendering plus autofocus and AE shaping, applied once after
    ``configure``.

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
    # 0.0 is neutral, but it is set unconditionally so every capture records the
    # same control set regardless of the flags used.
    cam_controls["ExposureValue"] = float(ev)
    if colour_gains is not None:
        cam_controls["AwbEnable"] = False
        cam_controls["ColourGains"] = tuple(colour_gains)
    elif awb_lock:
        cam_controls["AwbEnable"] = False
    if ae_lock:
        cam_controls["AeEnable"] = False
    camera.set_controls(cam_controls)


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
