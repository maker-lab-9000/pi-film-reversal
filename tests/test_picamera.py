import builtins
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession
from pifilm.capture.camera import CameraError, Frame
from pifilm.capture.picamera import Picamera2Camera
from pifilm.capture.thumbnail import fitted_jpeg
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

NATIVE_SIZE = (4608, 2592)
RAW_FORMAT = "SBGGR10_CSI2P"


def _actual_configuration():
    return {
        "main": {"size": NATIVE_SIZE, "format": "RGB888", "stride": 13824},
        "raw": {"size": NATIVE_SIZE, "format": RAW_FORMAT, "stride": 5760},
        "sensor": {"output_size": NATIVE_SIZE, "bit_depth": 10},
        "buffer_count": 2,
        "queue": False,
    }


class FakeRequest:
    def __init__(
        self,
        array,
        *,
        metadata=None,
        make_array_error=None,
        metadata_error=None,
        release_error=None,
        mutate_on_release=False,
        dng_bytes=b"II*\x00fake-dng-payload",
        save_dng_error=None,
    ):
        self.array = array
        self.metadata = metadata if metadata is not None else {}
        self.make_array_error = make_array_error
        self.metadata_error = metadata_error
        self.release_error = release_error
        self.mutate_on_release = mutate_on_release
        self.dng_bytes = dng_bytes
        self.save_dng_error = save_dng_error
        self.release_count = 0
        self.requested_stream = None
        self.save_dng_calls = 0
        self.save_dng_path = None

    def make_array(self, stream):
        self.requested_stream = stream
        if self.make_array_error is not None:
            raise self.make_array_error
        return self.array

    def get_metadata(self):
        if self.metadata_error is not None:
            raise self.metadata_error
        return self.metadata

    def save_dng(self, path):
        self.save_dng_calls += 1
        self.save_dng_path = path
        if self.save_dng_error is not None:
            raise self.save_dng_error
        with open(path, "wb") as fh:
            fh.write(self.dng_bytes)

    def release(self):
        self.release_count += 1
        if self.mutate_on_release:
            self.array[...] = 0
        if self.release_error is not None:
            raise self.release_error


class _AfModeEnum:
    """Mimics ``libcamera.controls.AfModeEnum``: distinct sentinel values."""

    Continuous = "AfMode.Continuous"
    Auto = "AfMode.Auto"
    Manual = "AfMode.Manual"


class _AfRangeEnum:
    """Mimics ``libcamera.controls.AfRangeEnum``."""

    Normal = "AfRange.Normal"
    Macro = "AfRange.Macro"
    Full = "AfRange.Full"


class _NoiseReductionModeEnum:
    """Mimics ``libcamera.controls.draft.NoiseReductionModeEnum``."""

    HighQuality = "NoiseReductionMode.HighQuality"


class _AeConstraintModeEnum:
    """Mimics ``libcamera.controls.AeConstraintModeEnum``."""

    Normal = "AeConstraintMode.Normal"
    Highlight = "AeConstraintMode.Highlight"
    Shadows = "AeConstraintMode.Shadows"


class _AeMeteringModeEnum:
    """Mimics ``libcamera.controls.AeMeteringModeEnum``."""

    CentreWeighted = "AeMeteringMode.CentreWeighted"
    Spot = "AeMeteringMode.Spot"
    Matrix = "AeMeteringMode.Matrix"


def _fake_libcamera_module(*, with_noise_reduction=True, with_ae_enums=True):
    controls_ns = SimpleNamespace(AfModeEnum=_AfModeEnum, AfRangeEnum=_AfRangeEnum)
    if with_noise_reduction:
        controls_ns.draft = SimpleNamespace(NoiseReductionModeEnum=_NoiseReductionModeEnum)
    if with_ae_enums:
        controls_ns.AeConstraintModeEnum = _AeConstraintModeEnum
        controls_ns.AeMeteringModeEnum = _AeMeteringModeEnum
    return SimpleNamespace(controls=controls_ns)


@pytest.fixture
def install_picamera(monkeypatch):
    def install(
        *,
        actual=None,
        request=None,
        tuning_error=None,
        init_error=None,
        configure_error=None,
        start_error=None,
        capture_error=None,
        stop_error=None,
        close_error=None,
        autofocus_cycle_error=None,
        with_noise_reduction=True,
        with_ae_enums=True,
        sensor_resolution=NATIVE_SIZE,
        sensor_modes=None,
        fixed_lens=False,
        model="imx708_wide",
        missing_attributes=(),
    ):
        state = SimpleNamespace(instance=None, loaded_tuning=[])

        class FakePicamera2:
            @staticmethod
            def load_tuning_file(filename):
                state.loaded_tuning.append(filename)
                if tuning_error is not None:
                    raise tuning_error
                return {"loaded-from": filename}

            def __init__(self, *, tuning=None):
                self.tuning = tuning
                self.sensor_resolution = tuple(sensor_resolution)
                self.camera_properties = {"Model": model}
                self.camera_controls = {
                    "ExposureValue": (-8.0, 8.0, 0.0),
                    "AeConstraintMode": (0, 3, 0),
                }
                if not fixed_lens:
                    self.camera_controls["AfMode"] = (0, 2, 0)
                    self.camera_controls["AfRange"] = (0, 2, 0)
                self.sensor_modes = sensor_modes if sensor_modes is not None else [
                    {"size": (1536, 864)}, {"size": (2304, 1296)}, {"size": (4608, 2592)},
                ]
                self.created_config = None
                self.preview_config = None
                self.configured_with = None
                self.configure_calls = []
                self.switch_calls = []
                self.configure_count = 0
                self.start_count = 0
                self.capture_count = 0
                self.stop_count = 0
                self.close_count = 0
                self.set_controls_calls = []
                self.autofocus_cycle_count = 0
                for attribute in missing_attributes:
                    delattr(self, attribute)
                state.instance = self
                if init_error is not None:
                    raise init_error

            def create_still_configuration(self, **kwargs):
                self.created_config = kwargs
                return {"created": kwargs}

            def create_preview_configuration(self, **kwargs):
                self.preview_config = kwargs
                return {"preview": kwargs}

            def configure(self, config):
                self.configure_count += 1
                self.configured_with = config
                self.configure_calls.append(config)
                if configure_error is not None:
                    raise configure_error

            def camera_configuration(self):
                # Picamera2 only ever describes the configuration currently in
                # force, so the fake must too: with the preview configured it
                # reports the small main stream on the binned sensor mode. That
                # is what makes "stream_info describes the still" testable -
                # building it after the preview configure fails loudly.
                configured = self.configured_with
                if isinstance(configured, dict) and "preview" in configured:
                    preview = configured["preview"]
                    main_size = tuple(preview["main"]["size"])
                    raw_size = tuple(preview["raw"]["size"])
                    return {
                        "main": {
                            "size": main_size,
                            "format": preview["main"]["format"],
                            "stride": main_size[0] * 3,
                        },
                        "raw": {"size": raw_size, "format": RAW_FORMAT, "stride": raw_size[0] * 2},
                        "sensor": {"output_size": raw_size, "bit_depth": 10},
                        "buffer_count": 4,
                        "queue": True,
                    }
                return actual if actual is not None else _actual_configuration()

            def set_controls(self, controls):
                self.set_controls_calls.append(dict(controls))

            def start(self):
                self.start_count += 1
                if start_error is not None:
                    raise start_error

            def capture_request(self):
                self.capture_count += 1
                if capture_error is not None:
                    raise capture_error
                return request

            def switch_mode_and_capture_request(self, config):
                self.switch_calls.append(config)
                self.capture_count += 1
                if capture_error is not None:
                    raise capture_error
                return request

            def autofocus_cycle(self):
                self.autofocus_cycle_count += 1
                if autofocus_cycle_error is not None:
                    raise autofocus_cycle_error

            def stop(self):
                self.stop_count += 1
                if stop_error is not None:
                    raise stop_error

            def close(self):
                self.close_count += 1
                if close_error is not None:
                    raise close_error

        monkeypatch.setitem(sys.modules, "picamera2", SimpleNamespace(Picamera2=FakePicamera2))
        libcamera_module = _fake_libcamera_module(
            with_noise_reduction=with_noise_reduction, with_ae_enums=with_ae_enums
        )
        monkeypatch.setitem(sys.modules, "libcamera", libcamera_module)
        return state

    return install


def test_constructor_uses_explicit_native_still_configuration(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera("delivered-lens.json")

    assert state.loaded_tuning == ["delivered-lens.json"]
    assert state.instance.tuning == {"loaded-from": "delivered-lens.json"}
    assert state.instance.created_config == {
        "main": {"size": NATIVE_SIZE, "format": "RGB888"},
        "raw": {"size": NATIVE_SIZE},
        "buffer_count": 2,
        "queue": False,
    }
    assert state.instance.configured_with == {"created": state.instance.created_config}
    assert state.instance.start_count == 1
    assert camera.stream_info.to_dict() == {
        "width": 4608,
        "height": 2592,
        "fps": 0.0,
        "fourcc": "RGB888",
        "raw_mjpeg": False,
        "sensor_mode": "4608x2592 SBGGR10_CSI2P",
        "bit_depth": 10,
        "tuning_file": "delivered-lens.json",
        "autofocus": "continuous",
    }
    camera.close()


def test_default_tuning_is_libcameras_automatic_choice(install_picamera):
    state = install_picamera(model="imx477")
    camera = Picamera2Camera()
    assert state.loaded_tuning == []
    assert state.instance.tuning is None
    assert camera.stream_info.tuning_file == "auto:imx477"
    camera.close()


def test_explicit_tuning_file_is_still_loaded(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera("imx708_wide.json")
    assert state.loaded_tuning == ["imx708_wide.json"]
    assert camera.stream_info.tuning_file == "imx708_wide.json"
    camera.close()


IMX477_SIZE = (4056, 3040)


def _imx477_configuration():
    return {
        "main": {"size": IMX477_SIZE, "format": "RGB888", "stride": 12192},
        "raw": {"size": IMX477_SIZE, "format": "SRGGB12_CSI2P", "stride": 6112},
        "sensor": {"output_size": IMX477_SIZE, "bit_depth": 12},
        "buffer_count": 2,
        "queue": False,
    }


def test_native_size_comes_from_the_sensor(install_picamera):
    state = install_picamera(
        actual=_imx477_configuration(), sensor_resolution=IMX477_SIZE, fixed_lens=True,
        model="imx477",
    )
    camera = Picamera2Camera()
    assert state.instance.created_config["main"]["size"] == IMX477_SIZE
    assert state.instance.created_config["raw"]["size"] == IMX477_SIZE
    assert (camera.stream_info.width, camera.stream_info.height) == IMX477_SIZE
    assert camera.stream_info.sensor_mode == "4056x3040 SRGGB12_CSI2P"
    assert camera.stream_info.bit_depth == 12
    camera.close()


def test_fixed_lens_sensor_skips_autofocus_controls_and_records_none(install_picamera):
    state = install_picamera(
        actual=_imx477_configuration(), sensor_resolution=IMX477_SIZE, fixed_lens=True,
    )
    camera = Picamera2Camera()
    controls = state.instance.set_controls_calls[0]
    assert "AfMode" not in controls and "AfRange" not in controls
    assert camera.stream_info.autofocus == "none"
    assert camera.stream_info.to_dict()["autofocus"] == "none"
    camera.close()


def test_autofocus_sensor_still_records_its_mode(install_picamera):
    install_picamera()
    camera = Picamera2Camera(autofocus="auto")
    assert camera.stream_info.autofocus == "auto"
    camera.close()


@pytest.mark.parametrize("kwargs", [{"autofocus": "auto"}, {"af_range": "macro"}])
def test_fixed_lens_sensor_rejects_non_default_autofocus(install_picamera, kwargs):
    state = install_picamera(
        actual=_imx477_configuration(), sensor_resolution=IMX477_SIZE, fixed_lens=True,
    )
    with pytest.raises(CameraError, match="no autofocus"):
        Picamera2Camera(**kwargs)
    assert state.instance.close_count == 1


@pytest.mark.parametrize("attribute", ["camera_controls", "camera_properties"])
def test_missing_capability_attribute_is_not_diagnosed_as_a_fixed_lens(
    install_picamera, attribute
):
    """A camera that cannot describe itself is a broken camera, not a fixed lens.

    Defaulting the capability lookups would turn a missing ``camera_controls``
    into "<model> has no autofocus", sending the user after their --autofocus
    flag instead of the real fault; ``sensor_resolution`` on the neighbouring
    line already fails loudly, so these two do the same.
    """
    state = install_picamera(missing_attributes=(attribute,))
    with pytest.raises(CameraError) as exc_info:
        Picamera2Camera(autofocus="auto")
    message = str(exc_info.value)
    assert "no autofocus" not in message
    assert message.startswith("Failed to configure or start Picamera2 camera:")
    assert attribute in message
    assert state.instance.close_count == 1


def test_import_is_lazy_and_missing_picamera2_is_actionable(monkeypatch):
    monkeypatch.delitem(sys.modules, "picamera2", raising=False)
    real_import = builtins.__import__

    def missing_picamera(name, *args, **kwargs):
        if name == "picamera2":
            raise ModuleNotFoundError("No module named 'picamera2'", name="picamera2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_picamera)
    with pytest.raises(CameraError, match="Picamera2.*install"):
        Picamera2Camera()


def test_native_library_load_failure_is_wrapped_as_camera_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "picamera2", raising=False)
    real_import = builtins.__import__

    def broken_native_library(name, *args, **kwargs):
        if name == "picamera2":
            raise OSError("libcamera.so could not be loaded")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_native_library)
    with pytest.raises(CameraError, match="Picamera2.*install") as exc_info:
        Picamera2Camera()
    assert isinstance(exc_info.value.__cause__, OSError)


def test_tuning_load_failure_is_actionable_without_creating_camera(install_picamera):
    state = install_picamera(tuning_error=OSError("bad tuning data"))
    with pytest.raises(CameraError, match="imx708_wide.json") as exc_info:
        Picamera2Camera("imx708_wide.json")
    assert isinstance(exc_info.value.__cause__, OSError)
    assert state.instance is None


def test_configure_failure_closes_camera_and_preserves_primary_error(install_picamera):
    state = install_picamera(
        configure_error=RuntimeError("configure exploded"),
        close_error=RuntimeError("close exploded"),
    )
    with pytest.raises(CameraError, match="configure exploded") as exc_info:
        Picamera2Camera()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert state.instance.stop_count == 0
    assert state.instance.close_count == 1


def test_start_failure_stops_and_closes_camera_without_masking_error(install_picamera):
    state = install_picamera(
        start_error=RuntimeError("start exploded"),
        stop_error=RuntimeError("stop exploded"),
        close_error=RuntimeError("close exploded"),
    )
    with pytest.raises(CameraError, match="start exploded") as exc_info:
        Picamera2Camera()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert state.instance.stop_count == 1
    assert state.instance.close_count == 1


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("main", "size", (2304, 1296)),
        ("main", "format", "BGR888"),
        ("raw", "size", (2304, 1296)),
    ],
)
def test_negotiated_non_native_stream_is_rejected_before_start(
    install_picamera, section, field, value
):
    actual = _actual_configuration()
    actual[section][field] = value
    state = install_picamera(actual=actual)

    with pytest.raises(CameraError, match="negotiated"):
        Picamera2Camera()

    assert state.instance.start_count == 0
    assert state.instance.close_count == 1


def test_read_returns_owned_contiguous_rgb_and_updates_measured_fps(install_picamera):
    bgr = np.zeros((2592, 4608, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 40, 230]
    request = FakeRequest(
        bgr,
        metadata={"FrameDuration": 50_000},
        mutate_on_release=True,
    )
    state = install_picamera(request=request)
    camera = Picamera2Camera()

    frame = camera.read()

    assert isinstance(frame, Frame)
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.flags.c_contiguous
    assert frame.jpeg is None
    assert frame.source == "picamera2"
    assert frame.rgb[0, 0].tolist() == [230, 40, 10]
    assert request.requested_stream == "main"
    assert request.release_count == 1
    assert camera.stream_info.fps == 20.0
    assert state.instance.capture_count == 1
    camera.close()


def test_non_finite_frame_duration_does_not_claim_a_measured_fps(install_picamera):
    bgr = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(bgr, metadata={"FrameDuration": float("nan")})
    install_picamera(request=request)
    camera = Picamera2Camera()

    camera.read()

    assert camera.stream_info.fps == 0.0
    assert request.release_count == 1
    camera.close()


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((2592, 4608, 3), np.uint16),
        ((4, 4), np.uint8),
        ((4, 4, 4), np.uint8),
        ((4, 4, 3), np.uint8),
    ],
    ids=["wrong-dtype", "missing-channels", "too-many-channels", "wrong-size"],
)
def test_invalid_main_array_is_rejected_and_request_is_released(
    install_picamera, shape, dtype
):
    array = np.broadcast_to(np.zeros((1,) * len(shape), dtype=dtype), shape)
    request = FakeRequest(array)
    install_picamera(request=request)
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="main array"):
        camera.read()

    assert request.release_count == 1
    camera.close()


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"make_array_error": RuntimeError("array failed")}, "array failed"),
        ({"metadata_error": RuntimeError("metadata failed")}, "metadata failed"),
    ],
)
def test_request_processing_failure_is_wrapped_and_released(
    install_picamera, request_kwargs, message
):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array, **request_kwargs)
    install_picamera(request=request)
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match=message) as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert request.release_count == 1
    camera.close()


def test_release_failure_does_not_mask_request_processing_failure(install_picamera):
    request = FakeRequest(
        np.zeros((4, 4, 3), dtype=np.uint8),
        make_array_error=RuntimeError("array failed"),
        release_error=RuntimeError("release failed"),
    )
    install_picamera(request=request)
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="array failed") as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert request.release_count == 1
    camera.close()


def test_release_failure_after_successful_processing_is_actionable(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array, release_error=RuntimeError("release failed"))
    install_picamera(request=request)
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="release failed") as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert request.release_count == 1
    camera.close()


class _FakeLibcameraEnum:
    """Mimics a libcamera enum control value: int-like but not an int subclass."""

    def __init__(self, value):
        self._value = value

    def __int__(self):
        return self._value


def test_read_populates_serialisable_metadata_subset(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    raw_metadata = {
        "ExposureTime": 9995,
        "ColourGains": (1.8, 2.1),
        "AfState": _FakeLibcameraEnum(2),
        "FocusFoM": object(),  # non-serialisable: cannot be coerced, dropped
        "ScalerCrop": (10, 20, 100, 100),  # extra key: not in METADATA_KEYS, dropped
        "SomeVendorBlob": {"nested": True},  # extra key: dropped
    }
    request = FakeRequest(array, metadata=raw_metadata)
    install_picamera(request=request)
    camera = Picamera2Camera()

    frame = camera.read()

    assert frame.metadata["ExposureTime"] == 9995
    assert frame.metadata["ColourGains"] == [1.8, 2.1]
    assert frame.metadata["AfState"] == 2
    assert "ScalerCrop" not in frame.metadata
    assert "SomeVendorBlob" not in frame.metadata
    assert "FocusFoM" not in frame.metadata
    assert "LensPosition" not in frame.metadata  # absent from raw metadata entirely
    assert "SensorTimestamp" not in frame.metadata  # absent from raw metadata entirely
    camera.close()


def test_read_includes_dng_bytes_when_enabled(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    install_picamera(request=request)
    camera = Picamera2Camera()

    frame = camera.read()

    assert frame.dng[:4] == b"II*\x00"
    assert request.save_dng_calls == 1
    assert request.release_count == 1
    camera.close()


def test_read_omits_dng_when_disabled(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    install_picamera(request=request)
    camera = Picamera2Camera(save_dng=False)

    frame = camera.read()

    assert frame.dng is None
    assert request.save_dng_calls == 0
    assert request.release_count == 1
    camera.close()


def test_default_construction_applies_neutral_rendering_and_continuous_af(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()

    assert state.instance.set_controls_calls == [{
        "Sharpness": 1.0,
        "Contrast": 1.0,
        "Saturation": 1.0,
        "NoiseReductionMode": _NoiseReductionModeEnum.HighQuality,
        "AfMode": _AfModeEnum.Continuous,
        "AfRange": _AfRangeEnum.Normal,
        "AeConstraintMode": _AeConstraintModeEnum.Normal,
        "AeMeteringMode": _AeMeteringModeEnum.CentreWeighted,
        "ExposureValue": 0.0,
    }]
    camera.close()


def test_noise_reduction_mode_is_omitted_on_older_libcamera(install_picamera):
    state = install_picamera(with_noise_reduction=False)
    camera = Picamera2Camera()

    controls = state.instance.set_controls_calls[-1]
    assert "NoiseReductionMode" not in controls
    assert controls["Sharpness"] == 1.0
    camera.close()


@pytest.mark.parametrize(
    ("autofocus", "expected"),
    [
        ("continuous", _AfModeEnum.Continuous),
        ("auto", _AfModeEnum.Auto),
        ("manual", _AfModeEnum.Manual),
    ],
)
def test_autofocus_argument_maps_to_af_mode_enum(install_picamera, autofocus, expected):
    state = install_picamera()
    camera = Picamera2Camera(autofocus=autofocus)

    assert state.instance.set_controls_calls[-1]["AfMode"] == expected
    camera.close()


@pytest.mark.parametrize(
    ("af_range", "expected"),
    [
        ("normal", _AfRangeEnum.Normal),
        ("macro", _AfRangeEnum.Macro),
        ("full", _AfRangeEnum.Full),
    ],
)
def test_af_range_argument_maps_to_af_range_enum(install_picamera, af_range, expected):
    state = install_picamera()
    camera = Picamera2Camera(af_range=af_range)

    assert state.instance.set_controls_calls[-1]["AfRange"] == expected
    camera.close()


def test_unknown_autofocus_mode_raises_camera_error_without_opening_hardware(install_picamera):
    state = install_picamera()

    with pytest.raises(CameraError, match="autofocus"):
        Picamera2Camera(autofocus="turbo")

    assert state.instance is None


def test_unknown_af_range_raises_camera_error_without_opening_hardware(install_picamera):
    state = install_picamera()

    with pytest.raises(CameraError, match="af_range"):
        Picamera2Camera(af_range="wide")

    assert state.instance is None


def test_colour_gains_sets_gains_and_disables_awb(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(colour_gains=(1.8, 2.1))

    controls = state.instance.set_controls_calls[-1]
    assert controls["ColourGains"] == (1.8, 2.1)
    assert controls["AwbEnable"] is False
    camera.close()


def test_awb_lock_without_colour_gains_disables_awb_but_sets_no_gains(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(awb_lock=True)

    controls = state.instance.set_controls_calls[-1]
    assert controls["AwbEnable"] is False
    assert "ColourGains" not in controls
    camera.close()


def test_ae_lock_disables_ae(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(ae_lock=True)

    assert state.instance.set_controls_calls[-1]["AeEnable"] is False
    camera.close()


def test_default_leaves_ae_and_awb_auto(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()

    controls = state.instance.set_controls_calls[-1]
    assert "AeEnable" not in controls
    assert "AwbEnable" not in controls
    assert "ColourGains" not in controls
    camera.close()


def test_ae_constraint_metering_and_ev_reach_the_controls(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(ae_constraint="highlight", ae_metering="matrix", ev=-0.5)

    controls = state.instance.set_controls_calls[-1]
    assert controls["AeConstraintMode"] == _AeConstraintModeEnum.Highlight
    assert controls["AeMeteringMode"] == _AeMeteringModeEnum.Matrix
    assert controls["ExposureValue"] == -0.5
    camera.close()


def test_ae_enums_are_omitted_on_older_libcamera_but_ev_is_still_set(install_picamera, capsys):
    state = install_picamera(with_ae_enums=False)
    camera = Picamera2Camera(ae_constraint="highlight")

    controls = state.instance.set_controls_calls[-1]
    assert "AeConstraintMode" not in controls
    assert "AeMeteringMode" not in controls
    assert controls["ExposureValue"] == 0.0
    # A requested non-default must not be silently dropped: the user asked for
    # highlight protection and would otherwise believe they got it.
    warning = capsys.readouterr().err
    assert "highlight" in warning and "AeConstraintMode" in warning
    camera.close()


def test_default_ae_settings_warn_nothing_on_older_libcamera(install_picamera, capsys):
    """The defaults are libcamera's own, so their absence changes nothing."""
    install_picamera(with_ae_enums=False)
    camera = Picamera2Camera()

    assert capsys.readouterr().err == ""
    camera.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ae_constraint": "bright"}, "ae_constraint"),
        ({"ae_metering": "average"}, "ae_metering"),
        ({"ev": 9.0}, "ev"),
        ({"ev": -8.5}, "ev"),
    ],
)
def test_invalid_ae_settings_raise_camera_error(install_picamera, kwargs, match):
    state = install_picamera()

    with pytest.raises(CameraError, match=match):
        Picamera2Camera(**kwargs)

    assert state.instance is None


def test_autofocus_auto_triggers_exactly_one_cycle_per_read(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    state = install_picamera(request=request)
    camera = Picamera2Camera(autofocus="auto")

    camera.read()
    camera.read()
    camera.read()

    assert state.instance.autofocus_cycle_count == 3
    assert state.instance.capture_count == 3
    camera.close()


def test_read_full_false_skips_autofocus_cycle_and_dng_but_keeps_metadata(install_picamera):
    """The preview path (``full=False``) must stay cheap even in ``--autofocus
    auto`` with DNG saving on: no autofocus cycle, no DNG bytes, but the request
    is still captured, converted and released, and metadata is still populated.
    """
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array, metadata={"FrameDuration": 50_000})
    state = install_picamera(request=request)
    camera = Picamera2Camera(autofocus="auto")

    frame = camera.read(full=False)

    assert frame.dng is None
    assert request.save_dng_calls == 0
    assert state.instance.autofocus_cycle_count == 0
    assert state.instance.capture_count == 1
    assert request.release_count == 1
    assert frame.metadata == {"FrameDuration": 50_000}
    assert frame.rgb.shape == (2592, 4608, 3)
    camera.close()


def test_read_default_still_cycles_autofocus_and_extracts_dng(install_picamera):
    """``read()`` with no arguments (the default ``full=True``) must behave
    exactly as before: one autofocus cycle in ``--autofocus auto`` and a DNG
    extracted for every frame.
    """
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    state = install_picamera(request=request)
    camera = Picamera2Camera(autofocus="auto")

    frame = camera.read()

    assert frame.dng is not None
    assert request.save_dng_calls == 1
    assert state.instance.autofocus_cycle_count == 1
    camera.close()


def test_preview_frame_does_not_pay_the_full_capture_cost(install_picamera, tmp_path):
    """Important 3: ``CaptureSession.preview_frame`` must call ``camera.read(full=False)``
    so the live preview does not run a blocking autofocus cycle or DNG extraction
    per frame.
    """
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array, metadata={"FrameDuration": 40_000})
    state = install_picamera(request=request)
    camera = Picamera2Camera(autofocus="auto")

    artifact_dir = tmp_path / "artifact"
    write_artifact(
        artifact_dir,
        LUT3D.identity(2),
        NormalizeParams(white_balance=False),
        GrainParams(enabled=False),
    )
    session = CaptureSession(
        camera,
        Pipeline(Artifacts.load(artifact_dir)),
        tmp_path / "captures",
        seed_rng=np.random.default_rng(0),
    )
    try:
        frame = session.preview_frame()
    finally:
        camera.close()

    assert frame.shape[-1] == 3
    assert state.instance.autofocus_cycle_count == 0
    assert request.save_dng_calls == 0


@pytest.mark.parametrize("autofocus", ["continuous", "manual"])
def test_autofocus_continuous_and_manual_never_cycle(install_picamera, autofocus):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    state = install_picamera(request=request)
    camera = Picamera2Camera(autofocus=autofocus)

    camera.read()
    camera.read()

    assert state.instance.autofocus_cycle_count == 0
    camera.close()


def test_autofocus_cycle_failure_is_wrapped_as_camera_error(install_picamera):
    state = install_picamera(autofocus_cycle_error=RuntimeError("af hardware fault"))
    camera = Picamera2Camera(autofocus="auto")

    with pytest.raises(CameraError, match="Autofocus cycle failed") as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert state.instance.capture_count == 0
    camera.close()


def test_dng_extraction_failure_still_releases_the_request(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array, save_dng_error=RuntimeError("dng failed"))
    install_picamera(request=request)
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="dng failed") as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert request.release_count == 1
    camera.close()


def test_capture_failure_is_wrapped_without_a_request_to_release(install_picamera):
    state = install_picamera(capture_error=RuntimeError("capture failed"))
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="capture failed") as exc_info:
        camera.read()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert state.instance.capture_count == 1
    camera.close()


def test_close_is_idempotent_and_read_after_close_is_rejected(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()

    camera.close()
    camera.close()

    assert state.instance.stop_count == 1
    assert state.instance.close_count == 1
    with pytest.raises(CameraError, match="closed"):
        camera.read()
    assert state.instance.capture_count == 0


def test_close_calls_close_if_stop_fails_and_preserves_stop_error(install_picamera):
    state = install_picamera(
        stop_error=RuntimeError("stop failed"),
        close_error=RuntimeError("close failed"),
    )
    camera = Picamera2Camera()

    with pytest.raises(CameraError, match="stop failed") as exc_info:
        camera.close()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert state.instance.stop_count == 1
    assert state.instance.close_count == 1
    camera.close()
    assert state.instance.stop_count == 1
    assert state.instance.close_count == 1


def test_native_capture_session_saves_full_size_outputs_metadata_and_thumbnail(
    install_picamera, tmp_path,
):
    bgr = np.broadcast_to(
        np.array([[[25, 110, 220]]], dtype=np.uint8),
        (NATIVE_SIZE[1], NATIVE_SIZE[0], 3),
    )
    request = FakeRequest(bgr, metadata={"FrameDuration": 40_000})
    install_picamera(request=request)
    camera = Picamera2Camera("delivered-variant.json")

    artifact_dir = tmp_path / "artifact"
    write_artifact(
        artifact_dir,
        LUT3D.identity(2),
        NormalizeParams(white_balance=False),
        GrainParams(enabled=False),
    )
    session = CaptureSession(
        camera,
        Pipeline(Artifacts.load(artifact_dir)),
        tmp_path / "captures",
        seed_rng=np.random.default_rng(0),
    )
    try:
        result = session.capture()
    finally:
        camera.close()

    # Picamera2 has no camera JPEG, but supplies a DNG by default, so the saved
    # original is named "_original.jpg" (not "_ungraded.jpg") and a DNG sidecar
    # is written alongside it; see the module docstring in pifilm/capture/app.py.
    assert result.original.name.endswith("_original.jpg")
    with Image.open(result.original) as ungraded, Image.open(result.pifilm) as graded:
        assert ungraded.size == NATIVE_SIZE
        assert graded.size == NATIVE_SIZE

    day_dir = result.original.parent
    assert [p.name for p in day_dir.glob("*_original.jpg")] == [result.original.name]
    assert [p.name for p in day_dir.glob("*_graded.jpg")] == [result.pifilm.name]
    dng_names = [p.name for p in day_dir.glob("*.dng")]
    assert len(dng_names) == 1

    record_path, = (tmp_path / "captures").rglob("captures.jsonl")
    record = json.loads(record_path.read_text())
    assert record["frame_source"] == "picamera2"
    assert record["width"] == 4608
    assert record["height"] == 2592
    assert record["fps"] == 25.0
    assert record["sensor_mode"] == "4608x2592 SBGGR10_CSI2P"
    assert record["bit_depth"] == 10
    assert record["tuning_file"] == "delivered-variant.json"
    assert record["camera_metadata"] == {"FrameDuration": 40_000}
    assert record["dng"] == result.original.name.replace("_original.jpg", ".dng")
    assert record["dng"] == dng_names[0]
    dng_path = day_dir / record["dng"]
    assert dng_path.read_bytes()[:4] == b"II*\x00"

    with Image.open(io.BytesIO(fitted_jpeg(result.pifilm))) as thumbnail:
        assert thumbnail.size == (240, 135)


def test_native_capture_session_omits_dng_when_backend_disables_it(
    install_picamera, tmp_path,
):
    """A ``--no-dng`` run: Picamera2Camera(save_dng=False) never asks the request
    to extract a DNG, but the frame is still the camera's own ISP rendering, so
    the original stays named ``_original.jpg``; only the DNG sidecar is omitted.
    See the "Output layout" section of the module docstring in
    pifilm/capture/app.py.
    """
    bgr = np.broadcast_to(
        np.array([[[25, 110, 220]]], dtype=np.uint8),
        (NATIVE_SIZE[1], NATIVE_SIZE[0], 3),
    )
    request = FakeRequest(bgr, metadata={"FrameDuration": 40_000})
    install_picamera(request=request)
    camera = Picamera2Camera("delivered-variant.json", save_dng=False)

    artifact_dir = tmp_path / "artifact"
    write_artifact(
        artifact_dir,
        LUT3D.identity(2),
        NormalizeParams(white_balance=False),
        GrainParams(enabled=False),
    )
    session = CaptureSession(
        camera,
        Pipeline(Artifacts.load(artifact_dir)),
        tmp_path / "captures",
        seed_rng=np.random.default_rng(0),
    )
    try:
        result = session.capture()
    finally:
        camera.close()

    assert request.save_dng_calls == 0
    assert result.original.name.endswith("_original.jpg")

    day_dir = result.original.parent
    assert not list(day_dir.glob("*.dng"))
    assert [p.name for p in day_dir.glob("*_original.jpg")] == [result.original.name]
    assert [p.name for p in day_dir.glob("*_graded.jpg")] == [result.pifilm.name]

    record_path, = (tmp_path / "captures").rglob("captures.jsonl")
    record = json.loads(record_path.read_text())
    assert "dng" not in record
    assert record["camera_metadata"] == {"FrameDuration": 40_000}


def test_preview_mode_configures_binned_preview_after_validating_the_still(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=(640, 480))
    inst = state.instance
    assert inst.configure_calls[0] == {"created": inst.created_config}
    assert inst.configure_calls[1] == {"preview": inst.preview_config}
    assert inst.preview_config["main"] == {"size": (640, 480), "format": "RGB888"}
    assert inst.preview_config["raw"] == {"size": (2304, 1296)}
    assert (camera.stream_info.width, camera.stream_info.height) == NATIVE_SIZE
    camera.close()


def test_binned_mode_is_the_largest_at_or_below_half_native():
    from pifilm.capture.picamera import _binned_mode

    modes = [{"size": (1332, 990)}, {"size": (2028, 1080)}, {"size": (2028, 1520)},
             {"size": (4056, 3040)}]
    assert _binned_mode(modes, (4056, 3040)) == (2028, 1520)
    assert _binned_mode([], (4056, 3040)) == (2028, 1520)


def test_without_preview_the_configuration_calls_are_unchanged(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    assert len(state.instance.configure_calls) == 1
    assert state.instance.preview_config is None
    camera.close()


def test_preview_read_uses_the_preview_stream_and_full_read_switches_mode(install_picamera):
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small, metadata={"ExposureTime": 5000}))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    frame = camera.read(full=False)
    assert frame.rgb.shape == (480, 640, 3)
    assert frame.metadata == {"ExposureTime": 5000}
    assert state.instance.switch_calls == []
    state.instance.capture_request = lambda: (_ for _ in ()).throw(AssertionError("no switch"))
    inst = state.instance
    inst.switch_mode_and_capture_request = lambda cfg: (inst.switch_calls.append(cfg),
                                                        FakeRequest(big))[1]
    full = camera.read(full=True)
    assert full.rgb.shape == (NATIVE_SIZE[1], NATIVE_SIZE[0], 3)
    assert inst.switch_calls == [{"created": inst.created_config}]
    camera.close()


def test_set_ev_applies_exposure_value_at_runtime(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    camera.set_ev(1.0 / 3)
    assert state.instance.set_controls_calls[-1] == {"ExposureValue": pytest.approx(1 / 3)}
    assert camera.ev == pytest.approx(1 / 3)
    with pytest.raises(CameraError):
        camera.set_ev(9.0)
    camera.close()


def test_requests_are_serialised_by_a_lock(install_picamera):
    import threading

    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(big))
    camera = Picamera2Camera(save_dng=False)
    inside = threading.Event()
    release = threading.Event()
    overlaps = []

    original = state.instance.capture_request

    def slow_capture():
        if inside.is_set():
            overlaps.append(True)
        inside.set()
        release.wait(1.0)
        inside.clear()
        return original()

    state.instance.capture_request = slow_capture
    t = threading.Thread(target=camera.read)
    t.start()
    inside.wait(1.0)
    t2 = threading.Thread(target=camera.read)
    t2.start()
    release.set()
    t.join(2.0)
    t2.join(2.0)
    assert overlaps == []
    camera.close()


def test_preview_reads_do_not_overwrite_the_measured_still_fps(install_picamera):
    """``stream_info`` is the audit of the still, and ``fps`` is part of it: a
    binned viewfinder frame must not be what a capture record claims the still
    was shot at."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small, metadata={"FrameDuration": 20_000}))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    assert camera.stream_info.fps == 0.0
    camera.read(full=False)
    assert camera.stream_info.fps == 0.0
    inst = state.instance
    inst.switch_mode_and_capture_request = lambda cfg: FakeRequest(
        big, metadata={"FrameDuration": 50_000}
    )
    camera.read(full=True)
    assert camera.stream_info.fps == 20.0
    camera.close()
