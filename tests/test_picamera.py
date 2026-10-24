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
from pifilm.capture.errors import CameraStalled
from pifilm.capture.picamera import Picamera2Camera
from pifilm.capture.thumbnail import fitted_jpeg
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline
from pifilm.tuning import fingerprint

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
    Minimal = "NoiseReductionMode.Minimal"


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


class _ExposureTimeModeEnum:
    Auto = "exposure-time-auto"
    Manual = "exposure-time-manual"


class _AnalogueGainModeEnum:
    Auto = "analogue-gain-auto"
    Manual = "analogue-gain-manual"


def _fake_libcamera_module(
    *, with_noise_reduction=True, with_ae_enums=True, with_exposure_time_mode=False,
    with_analogue_gain_mode=False,
):
    controls_ns = SimpleNamespace(AfModeEnum=_AfModeEnum, AfRangeEnum=_AfRangeEnum)
    if with_noise_reduction:
        controls_ns.draft = SimpleNamespace(NoiseReductionModeEnum=_NoiseReductionModeEnum)
    if with_ae_enums:
        controls_ns.AeConstraintModeEnum = _AeConstraintModeEnum
        controls_ns.AeMeteringModeEnum = _AeMeteringModeEnum
    if with_exposure_time_mode:
        controls_ns.ExposureTimeModeEnum = _ExposureTimeModeEnum
    if with_analogue_gain_mode:
        controls_ns.AnalogueGainModeEnum = _AnalogueGainModeEnum
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
        with_exposure_time_mode=False,
        analogue_gain=None,
        frame_limits=None,
        control_frame_limits=None,
        with_colour_temperature=True,
        with_analogue_gain_mode=False,
        ae_controls=("AeConstraintMode", "AeMeteringMode"),
        image_controls=("Sharpness", "Contrast", "Saturation"),
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
                self.camera_controls = {"ExposureValue": (-8.0, 8.0, 0.0)}
                for name in ae_controls:
                    self.camera_controls[name] = (0, 3, 0)
                for name in image_controls:
                    self.camera_controls[name] = (0.0, 16.0, 1.0)
                if not fixed_lens:
                    self.camera_controls["AfMode"] = (0, 2, 0)
                    self.camera_controls["AfRange"] = (0, 2, 0)
                if with_exposure_time_mode:
                    self.camera_controls["ExposureTimeMode"] = (0, 1, 0)
                if with_analogue_gain_mode:
                    self.camera_controls["AnalogueGainMode"] = (0, 1, 0)
                if analogue_gain is not None:
                    self.camera_controls["AnalogueGain"] = analogue_gain
                if with_colour_temperature:
                    self.camera_controls["ColourTemperature"] = (100, 100000, None)
                if control_frame_limits is not None:
                    self.camera_controls["FrameDurationLimits"] = control_frame_limits
                self.sensor_modes = sensor_modes if sensor_modes is not None else [
                    {"size": (1536, 864)}, {"size": (2304, 1296)}, {"size": (4608, 2592)},
                ]
                self.created_config = None
                self.preview_config = None
                self.configured_with = None
                self.configure_calls = []
                self.configured_controls = []
                self.switch_calls = []
                self.switch_mode_calls = []
                self.configure_count = 0
                self.start_count = 0
                self.capture_count = 0
                self.waits = []
                self.cancel_count = 0
                self.started = True
                self.completed_requests = []
                self.frames = 0
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
                # Picamera2 puts the requested controls into the returned
                # configuration, and applies them again on every configure();
                # the control setters mutate that entry, so the fake must expose it.
                controls = kwargs.get("controls", {})
                if frame_limits is not None:
                    # Picamera2 merges its default frame limits into every configuration.
                    controls["FrameDurationLimits"] = frame_limits["still"]
                return {"created": kwargs, "controls": controls}

            def create_preview_configuration(self, **kwargs):
                self.preview_config = kwargs
                controls = kwargs.get("controls", {})
                if frame_limits is not None:
                    controls["FrameDurationLimits"] = frame_limits["preview"]
                return {"preview": kwargs, "controls": controls}

            def configure(self, config):
                self.configure_count += 1
                self.configured_with = config
                self.configure_calls.append(config)
                self.configured_controls.append(dict(config.get("controls", {})))
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

            def capture_request(self, wait=None):
                self.capture_count += 1
                self.waits.append(wait)
                if capture_error is not None:
                    raise capture_error
                return request

            def cancel_all_and_flush(self):
                self.cancel_count += 1

            def switch_mode_and_capture_request(self, config, wait=None):
                self.switch_calls.append(config)
                self.capture_count += 1
                self.waits.append(wait)
                if capture_error is not None:
                    raise capture_error
                return request

            def switch_mode(self, config, wait=None):
                self.switch_mode_calls.append((config, wait))

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
            with_noise_reduction=with_noise_reduction,
            with_ae_enums=with_ae_enums,
            with_exposure_time_mode=with_exposure_time_mode,
            with_analogue_gain_mode=with_analogue_gain_mode,
        )
        monkeypatch.setitem(sys.modules, "libcamera", libcamera_module)
        return state

    return install


def test_constructor_uses_explicit_native_still_configuration(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera("delivered-lens.json")

    assert state.loaded_tuning == ["delivered-lens.json"]
    assert state.instance.tuning == {"loaded-from": "delivered-lens.json"}
    # The controls entry is deliberate: Picamera2 resets its controls to the
    # configuration's own dict on every configure(), so the neutral ISP set has
    # to travel with the configuration, not only through set_controls().
    assert state.instance.created_config == {
        "main": {"size": NATIVE_SIZE, "format": "RGB888"},
        "raw": {"size": NATIVE_SIZE},
        "buffer_count": 2,
        "queue": False,
        "controls": state.instance.set_controls_calls[0],
    }
    assert state.instance.configured_with["created"] is state.instance.created_config
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
        "tuning_sha1": fingerprint({"loaded-from": "delivered-lens.json"}),
        "ae_constraint": "normal",
        "ae_metering": "centre",
        "sharpness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "autofocus": "continuous",
    }
    camera.close()


def test_default_tuning_is_libcameras_automatic_choice(install_picamera):
    state = install_picamera(model="imx477")
    camera = Picamera2Camera()
    # The camera is still opened without a tuning, so libcamera chooses; the
    # sensor's default file is loaded only to fingerprint it.
    assert state.instance.tuning is None
    assert state.loaded_tuning == ["imx477.json"]
    assert camera.stream_info.tuning_file == "auto:imx477"
    camera.close()


def test_explicit_tuning_fingerprints_the_dict_the_camera_opened_with(install_picamera):
    state = install_picamera(model="imx477")
    camera = Picamera2Camera("imx477.json")
    assert state.loaded_tuning == ["imx477.json"]  # loaded once, not re-read
    expected = fingerprint(state.instance.tuning)
    assert camera.stream_info.to_dict()["tuning_sha1"] == expected
    assert camera.stream_info.tuning_sha1 == expected
    assert camera.tuning.sha1 == expected
    assert camera.tuning.label == "imx477.json"
    camera.close()


def test_automatic_tuning_is_fingerprinted_from_the_sensor_default(install_picamera):
    install_picamera(model="imx477")
    camera = Picamera2Camera()
    expected = fingerprint({"loaded-from": "imx477.json"})
    assert camera.stream_info.to_dict()["tuning_sha1"] == expected
    assert camera.tuning.label == "auto:imx477"
    camera.close()


def test_automatic_tuning_ignores_the_environment_variable(install_picamera, monkeypatch,
                                                           tmp_path):
    # Picamera2 clears LIBCAMERA_RPI_TUNING_FILE when it opens without a tuning.
    path = tmp_path / "other.json"
    path.write_text('{"other": 1}')
    monkeypatch.setenv("LIBCAMERA_RPI_TUNING_FILE", str(path))
    install_picamera(model="imx477")
    camera = Picamera2Camera()
    assert camera.stream_info.tuning_sha1 == fingerprint({"loaded-from": "imx477.json"})
    camera.close()


def test_automatic_tuning_that_cannot_load_still_opens_the_camera(install_picamera, capsys):
    state = install_picamera(model="imx477", tuning_error=OSError("no such file"))
    camera = Picamera2Camera()
    assert state.instance.start_count == 1
    assert "tuning_sha1" not in camera.stream_info.to_dict()
    assert camera.tuning.sha1 is None
    err = capsys.readouterr().err
    assert "tuning: cannot fingerprint the tuning in use (" in err
    assert "no such file" in err
    camera.close()


def test_unexpected_resolve_failure_does_not_stop_the_camera(
    install_picamera, monkeypatch, capsys,
):
    import pifilm.capture.picamera as picamera_module

    def broken(*args, **kwargs):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(picamera_module, "resolve_tuning", broken)
    state = install_picamera(model="imx477")
    camera = Picamera2Camera()
    assert state.instance.start_count == 1
    assert "tuning_sha1" not in camera.stream_info.to_dict()
    assert camera.tuning.sha1 is None
    assert "tuning: cannot fingerprint the tuning in use (" in capsys.readouterr().err
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


def test_ae_constraint_and_metering_reach_the_controls_beside_a_zero_ev(install_picamera):
    """Exposure compensation is gone, but ``ExposureValue: 0.0`` is still sent so
    every capture records the same control set as before."""
    state = install_picamera()
    camera = Picamera2Camera(ae_constraint="highlight", ae_metering="matrix")

    controls = state.instance.set_controls_calls[-1]
    assert controls["AeConstraintMode"] == _AeConstraintModeEnum.Highlight
    assert controls["AeMeteringMode"] == _AeMeteringModeEnum.Matrix
    assert controls["ExposureValue"] == 0.0
    assert camera.ev == 0.0
    camera.close()


def test_exposure_compensation_is_gone_from_the_camera(install_picamera):
    install_picamera()
    with pytest.raises(TypeError):
        Picamera2Camera(ev=-0.5)
    camera = Picamera2Camera()
    assert not hasattr(camera, "set_ev")
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

    # Only the start-up lines naming the shutter and ISO control paths; no AE warning.
    assert capsys.readouterr().err == (
        "picamera2: shutter priority via ExposureTime (legacy libcamera)\n"
        "picamera2: ISO via AnalogueGain (legacy libcamera)\n"
    )
    camera.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ae_constraint": "bright"}, "ae_constraint"),
        ({"ae_metering": "average"}, "ae_metering"),
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
    assert record["tuning_sha1"] == fingerprint({"loaded-from": "delivered-variant.json"})
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
    # sensor_modes is deleted: reading it makes Picamera2 reconfigure the camera
    # through every raw mode, seconds of start-up, so the backend must not.
    state = install_picamera(missing_attributes=("sensor_modes",))
    camera = Picamera2Camera(preview=(640, 480))
    inst = state.instance
    assert inst.configure_calls[0]["created"] is inst.created_config
    assert inst.configure_calls[1]["preview"] is inst.preview_config
    assert inst.preview_config["main"] == {"size": (640, 480), "format": "RGB888"}
    assert inst.preview_config["raw"] == {"size": (2304, 1296)}
    assert (camera.stream_info.width, camera.stream_info.height) == NATIVE_SIZE
    camera.close()


def test_binned_mode_is_half_the_native_size():
    from pifilm.capture.picamera import _binned_mode

    assert _binned_mode((4056, 3040)) == (2028, 1520)
    assert _binned_mode(NATIVE_SIZE) == (2304, 1296)


@pytest.mark.parametrize(
    "native, expected",
    [(NATIVE_SIZE, (640, 360)), (IMX477_SIZE, (640, 480))],
)
def test_preview_true_derives_the_size_from_the_sensor_aspect(
    install_picamera, native, expected,
):
    """libcamera crops the sensor to the output aspect, so a fixed 4:3 preview on
    a 16:9 sensor would frame the viewfinder differently from the still."""
    state = install_picamera(
        actual=_imx477_configuration() if native == IMX477_SIZE else None,
        sensor_resolution=native,
        fixed_lens=native == IMX477_SIZE,
    )
    camera = Picamera2Camera(preview=True)
    assert camera.preview_size == expected
    assert state.instance.preview_config["main"]["size"] == expected
    camera.close()


def test_without_preview_there_is_no_preview_size(install_picamera):
    install_picamera()
    camera = Picamera2Camera()
    assert camera.preview_size is None
    camera.close()


def test_neutral_controls_travel_with_both_configurations(install_picamera):
    """Picamera2 resets its controls to the configuration's own dict on every
    configure, and switch_mode_and_capture_request configures twice; without the
    controls in both configurations the still would be taken with the ISP's
    defaults and the preview would come back without them."""
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    neutral = {
        "Sharpness": 1.0,
        "Contrast": 1.0,
        "Saturation": 1.0,
        "NoiseReductionMode": _NoiseReductionModeEnum.HighQuality,
        "AfMode": _AfModeEnum.Continuous,
        "AfRange": _AfRangeEnum.Normal,
        "AeConstraintMode": _AeConstraintModeEnum.Normal,
        "AeMeteringMode": _AeMeteringModeEnum.CentreWeighted,
        "ExposureValue": 0.0,
    }
    # The one difference: the live view's own (Picamera2's preview) noise reduction.
    preview = {**neutral, "NoiseReductionMode": _NoiseReductionModeEnum.Minimal}
    assert inst.created_config["controls"] == neutral
    assert inst.preview_config["controls"] == preview
    assert inst.configured_controls == [neutral, preview]
    camera.close()


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
    state.instance.capture_request = lambda **k: (_ for _ in ()).throw(
        AssertionError("no switch"))
    inst = state.instance
    inst.switch_mode_and_capture_request = lambda cfg, **k: (inst.switch_calls.append(cfg),
                                                        FakeRequest(big))[1]
    full = camera.read(full=True)
    assert full.rgb.shape == (NATIVE_SIZE[1], NATIVE_SIZE[0], 3)
    assert len(inst.switch_calls) == 1 and inst.switch_calls[0]["created"] is inst.created_config
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

    def slow_capture(**kwargs):
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


class _HeldCapture:
    """Replaces one of the fake camera's capture calls with one that waits.

    ``inside`` is set when the call has been entered (so the camera's lock is
    held by then) and the call returns once ``release`` is set.
    """

    def __init__(self, instance, name, request):
        import threading

        self.inside = threading.Event()
        self.release = threading.Event()
        self.order = []
        self._request = request
        setattr(instance, name, self._capture)

    def _capture(self, *args, **kwargs):
        self.order.append("entered")
        self.inside.set()
        assert self.release.wait(5.0), "the test never released this capture"
        self.order.append("returned")
        return self._request


def _in_thread(call):
    """Run ``call`` on a thread; the returned event is set when it has returned."""
    import threading

    done = threading.Event()
    errors = []

    def target():
        try:
            call()
        except Exception as exc:  # reported by the test's own assertions
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return done, errors


_SETTERS = [
    ("set_shutter", 4000, "shutter_us"),
    ("set_iso", 400, "iso"),
    ("set_colour_temperature", 5600, "colour_temperature_k"),
]


@pytest.mark.parametrize("setter, value, attribute", _SETTERS)
def test_a_control_setter_does_not_wait_for_a_preview_read(
    install_picamera, setter, value, attribute,
):
    """The viewfinder reads previews on their own thread, and at a 1 s shutter
    one read holds the request lock for a second. A tap on ``+`` must not wait
    behind it, or the controls are as slow as the picture."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    held = _HeldCapture(state.instance, "capture_request", FakeRequest(small))
    read_done, read_errors = _in_thread(lambda: camera.read(full=False))
    try:
        assert held.inside.wait(2.0)
        set_done, set_errors = _in_thread(lambda: getattr(camera, setter)(value))
        assert set_done.wait(2.0), f"{setter} waited for the preview read"
        assert set_errors == [] and getattr(camera, attribute) == value
        assert not read_done.is_set()
    finally:
        held.release.set()
    assert read_done.wait(2.0) and read_errors == []
    camera.close()


@pytest.mark.parametrize("setter, value, attribute", _SETTERS)
def test_a_control_setter_waits_for_a_still_capture(
    install_picamera, setter, value, attribute,
):
    """A still reconfigures the camera twice from the configurations the
    setters write, so a control change must not land in the middle of one."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    held = _HeldCapture(inst, "switch_mode_and_capture_request", FakeRequest(big))
    sent_before = len(inst.set_controls_calls)
    read_done, read_errors = _in_thread(lambda: camera.read(full=True))
    try:
        assert held.inside.wait(2.0)
        set_done, set_errors = _in_thread(lambda: getattr(camera, setter)(value))
        assert not set_done.wait(0.1), f"{setter} did not wait for the still"
        assert getattr(camera, attribute) is None
        assert len(inst.set_controls_calls) == sent_before
    finally:
        held.release.set()
    assert read_done.wait(5.0) and read_errors == []
    assert set_done.wait(2.0) and set_errors == []
    assert getattr(camera, attribute) == value
    assert len(inst.set_controls_calls) == sent_before + 1
    camera.close()


def test_a_still_capture_waits_for_a_preview_read_in_flight(install_picamera):
    """The pixel buffer belongs to the request: a mode switch while a preview
    request is still being copied would free memory that copy is reading."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    held = _HeldCapture(inst, "capture_request", FakeRequest(small))
    switched = []
    inst.switch_mode_and_capture_request = lambda config, **kwargs: (
        switched.append(list(held.order)), FakeRequest(big))[1]
    preview_done, preview_errors = _in_thread(lambda: camera.read(full=False))
    try:
        assert held.inside.wait(2.0)
        still_done, still_errors = _in_thread(lambda: camera.read(full=True))
        assert not still_done.wait(0.1), "the still did not wait for the preview read"
        assert switched == []
    finally:
        held.release.set()
    assert preview_done.wait(2.0) and preview_errors == []
    assert still_done.wait(5.0) and still_errors == []
    assert switched == [["entered", "returned"]]
    camera.close()


def test_a_still_without_a_preview_stream_also_excludes_the_setters(install_picamera):
    """Without ``preview`` a still is a plain ``capture_request``, but it is
    still the shot the record describes: its controls must not change under it."""
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(big))
    camera = Picamera2Camera(save_dng=False)
    held = _HeldCapture(state.instance, "capture_request", FakeRequest(big))
    read_done, read_errors = _in_thread(lambda: camera.read(full=True))
    try:
        assert held.inside.wait(2.0)
        set_done, set_errors = _in_thread(lambda: camera.set_shutter(4000))
        assert not set_done.wait(0.1)
        assert camera.shutter_us is None
    finally:
        held.release.set()
    assert read_done.wait(5.0) and read_errors == []
    assert set_done.wait(2.0) and set_errors == [] and camera.shutter_us == 4000
    camera.close()


def _controls_snapshot(inst):
    return [dict(config["controls"]) for config in (inst.created_config, inst.preview_config)]


def test_reassert_controls_resends_the_fixed_selection_in_one_call(install_picamera):
    """Seen on the Pi 4: a control sent to the running camera sometimes does not
    take effect. The resend is what each setter sent live, together, and it
    leaves the configurations and the selection alone."""
    state = install_picamera(frame_limits={"still": (100, 50_000), "preview": (100, 33_333)})
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    camera.set_shutter(250_000)
    camera.set_iso(400)
    camera.set_colour_temperature(5600)
    sent = inst.set_controls_calls[-3:]
    expected = {**sent[0], **sent[1], **sent[2]}
    assert expected == {
        "ExposureTime": 250_000, "FrameDurationLimits": (100, 251_000),
        "AnalogueGain": 4.0, "ColourTemperature": 5600,
    }
    configs, calls = _controls_snapshot(inst), len(inst.set_controls_calls)
    camera.reassert_controls()
    assert inst.set_controls_calls[calls:] == [expected]
    assert _controls_snapshot(inst) == configs
    assert (camera.shutter_us, camera.iso, camera.colour_temperature_k) == (250_000, 400, 5600)
    camera.close()


def test_reassert_controls_resends_auto_through_the_mode_controls(install_picamera):
    state = install_picamera(with_exposure_time_mode=True, with_analogue_gain_mode=True)
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    camera.reassert_controls()
    assert inst.set_controls_calls[-1] == {
        "ExposureTimeMode": _ExposureTimeModeEnum.Auto,
        "AnalogueGainMode": _AnalogueGainModeEnum.Auto,
        "AwbEnable": True,
    }
    camera.set_shutter(4000)
    camera.set_iso(800)
    camera.reassert_controls()
    assert inst.set_controls_calls[-1] == {
        "ExposureTimeMode": _ExposureTimeModeEnum.Manual, "ExposureTime": 4000,
        "AnalogueGainMode": _AnalogueGainModeEnum.Manual, "AnalogueGain": 8.0,
        "AwbEnable": True,
    }
    camera.close()


def test_reassert_controls_on_legacy_libcamera_resends_auto_as_zeroes(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    camera.reassert_controls()
    assert state.instance.set_controls_calls[-1] == {
        "ExposureTime": 0, "AnalogueGain": 0.0, "AwbEnable": True,
    }
    camera.close()


def test_reassert_controls_leaves_locked_exposure_alone(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True, ae_lock=True)
    camera.set_colour_temperature(6500)
    camera.reassert_controls()
    assert state.instance.set_controls_calls[-1] == {"ColourTemperature": 6500}
    camera.close()


def test_reassert_controls_leaves_a_locked_white_balance_alone(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True, awb_lock=True)
    camera.set_shutter(4000)
    camera.reassert_controls()
    assert state.instance.set_controls_calls[-1] == {"ExposureTime": 4000, "AnalogueGain": 0.0}
    camera.close()


def test_reassert_controls_sends_nothing_when_nothing_is_adjustable(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True, ae_lock=True, awb_lock=True)
    calls = len(state.instance.set_controls_calls)
    camera.reassert_controls()
    assert len(state.instance.set_controls_calls) == calls
    camera.close()


def test_reassert_controls_on_a_closed_camera_is_an_error(install_picamera):
    install_picamera()
    camera = Picamera2Camera(preview=True)
    camera.close()
    with pytest.raises(CameraError, match="closed"):
        camera.reassert_controls()


def test_reassert_controls_wraps_a_camera_failure(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    camera.set_shutter(4000)

    def refuse(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = refuse
    with pytest.raises(CameraError, match="failed to resend the controls: control rejected"):
        camera.reassert_controls()
    assert camera.shutter_us == 4000
    camera.close()


def test_reassert_controls_waits_for_a_still_capture(install_picamera):
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    held = _HeldCapture(inst, "switch_mode_and_capture_request", FakeRequest(big))
    calls = len(inst.set_controls_calls)
    read_done, read_errors = _in_thread(lambda: camera.read(full=True))
    try:
        assert held.inside.wait(2.0)
        sent, send_errors = _in_thread(camera.reassert_controls)
        assert not sent.wait(0.1)
        assert len(inst.set_controls_calls) == calls
    finally:
        held.release.set()
    assert read_done.wait(5.0) and read_errors == []
    assert sent.wait(2.0) and send_errors == []
    assert len(inst.set_controls_calls) == calls + 1
    camera.close()


def test_restart_preview_switches_to_the_preview_configuration_with_a_timeout(
    install_picamera,
):
    """Seen on the Pi 4: a live control is sometimes ignored for good and only a
    photo cures it, because the photo reconfigures the camera. This is that
    reconfiguration without the photo; the configuration holds the selection."""
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    camera.set_colour_temperature(6500)
    camera.set_shutter(500_000)
    configure_calls, live_calls = len(inst.configure_calls), len(inst.set_controls_calls)
    camera.restart_preview()
    assert len(inst.switch_mode_calls) == 1
    config, wait = inst.switch_mode_calls[0]
    assert config["preview"] is inst.preview_config
    assert config["controls"]["ColourTemperature"] == 6500
    assert config["controls"]["ExposureTime"] == 500_000
    assert wait == pytest.approx(30.0 + 4 * 0.5)
    assert len(inst.configure_calls) == configure_calls
    assert len(inst.set_controls_calls) == live_calls
    assert (camera.shutter_us, camera.colour_temperature_k) == (500_000, 6500)
    camera.close()


def test_restart_preview_that_times_out_is_a_stall(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    inst.switch_mode = lambda config, wait=None: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(CameraStalled, match="live view restart within 30.0 s"):
        camera.restart_preview()
    assert inst.cancel_count == 1
    # The locks were released: a control can still be set and a frame asked for.
    camera.set_shutter(4000)
    camera.close()


def test_restart_preview_wraps_a_camera_failure(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    state.instance.switch_mode = lambda config, wait=None: (_ for _ in ()).throw(
        RuntimeError("device busy"))
    with pytest.raises(CameraError, match="failed to restart the live view: device busy") as hit:
        camera.restart_preview()
    assert not isinstance(hit.value, CameraStalled)
    camera.close()


def test_restart_preview_needs_a_preview_stream(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    with pytest.raises(CameraError, match="no preview stream"):
        camera.restart_preview()
    assert state.instance.switch_mode_calls == []
    camera.close()


def test_restart_preview_on_a_closed_camera_is_an_error(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    camera.close()
    with pytest.raises(CameraError, match="closed"):
        camera.restart_preview()
    assert state.instance.switch_mode_calls == []


def test_restart_preview_waits_for_a_preview_read_in_flight(install_picamera):
    """It stops the camera, so like a still it must not start while a request
    is still being copied: the request lock comes first."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    held = _HeldCapture(inst, "capture_request", FakeRequest(small))
    read_done, read_errors = _in_thread(lambda: camera.read(full=False))
    try:
        assert held.inside.wait(2.0)
        restarted, restart_errors = _in_thread(camera.restart_preview)
        assert not restarted.wait(0.1)
        assert inst.switch_mode_calls == []
        # Waiting for the request lock, it does not yet hold the configuration
        # lock: a setter still gets through, as it does beside any preview read.
        set_done, set_errors = _in_thread(lambda: camera.set_iso(400))
        assert set_done.wait(2.0) and set_errors == []
    finally:
        held.release.set()
    assert read_done.wait(2.0) and read_errors == []
    assert restarted.wait(2.0) and restart_errors == []
    assert len(inst.switch_mode_calls) == 1
    camera.close()


def test_a_restart_in_progress_excludes_reads_and_setters(install_picamera):
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    held = _HeldCapture(inst, "switch_mode", None)
    restarted, restart_errors = _in_thread(camera.restart_preview)
    try:
        assert held.inside.wait(2.0)
        set_done, set_errors = _in_thread(lambda: camera.set_iso(400))
        read_done, read_errors = _in_thread(lambda: camera.read(full=False))
        assert not set_done.wait(0.1) and not read_done.is_set()
        assert camera.iso is None and inst.capture_count == 0
    finally:
        held.release.set()
    assert restarted.wait(2.0) and restart_errors == []
    assert set_done.wait(2.0) and set_errors == [] and camera.iso == 400
    assert read_done.wait(2.0) and read_errors == []
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
    inst.switch_mode_and_capture_request = lambda cfg, **k: FakeRequest(
        big, metadata={"FrameDuration": 50_000}
    )
    camera.read(full=True)
    assert camera.stream_info.fps == 20.0
    camera.close()


def test_shutter_priority_uses_exposure_time_mode_when_available(install_picamera, capsys):
    state = install_picamera(with_exposure_time_mode=True)
    camera = Picamera2Camera(preview=True)
    assert "ExposureTimeMode" in capsys.readouterr().err
    camera.set_shutter(4000)
    live = state.instance.set_controls_calls[-1]
    assert live["ExposureTimeMode"] == _ExposureTimeModeEnum.Manual
    assert live["ExposureTime"] == 4000
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["ExposureTimeMode"] == _ExposureTimeModeEnum.Manual
        assert config["controls"]["ExposureTime"] == 4000
    assert camera.shutter_us == 4000
    camera.set_shutter(None)
    assert state.instance.set_controls_calls[-1]["ExposureTimeMode"] == (
        _ExposureTimeModeEnum.Auto
    )
    assert camera.shutter_us is None
    camera.close()


def test_shutter_priority_falls_back_to_exposure_time_on_older_libcamera(
    install_picamera, capsys,
):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    assert "shutter priority via ExposureTime (legacy" in capsys.readouterr().err
    camera.set_shutter(250_000)
    assert state.instance.set_controls_calls[-1]["ExposureTime"] == 250_000
    assert "ExposureTimeMode" not in state.instance.set_controls_calls[-1]
    camera.set_shutter(None)
    assert state.instance.set_controls_calls[-1]["ExposureTime"] == 0
    assert state.instance.preview_config["controls"]["ExposureTime"] == 0
    camera.close()


def test_returning_to_auto_drops_the_stale_exposure_time_from_both_configurations(
    install_picamera,
):
    """Every configure reapplies a configuration's controls as one unordered set,
    so an ``ExposureTime`` left beside ``ExposureTimeMode=Auto`` would be sent
    again on every still; whether the IPA ignores it cannot be checked off the Pi."""
    state = install_picamera(with_exposure_time_mode=True)
    camera = Picamera2Camera(preview=True)
    camera.set_shutter(4000)
    camera.set_shutter(None)
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["ExposureTimeMode"] == _ExposureTimeModeEnum.Auto
        assert "ExposureTime" not in config["controls"]
    camera.close()


def test_a_long_shutter_widens_frame_limits_in_both_configurations_and_auto_restores(
    install_picamera,
):
    limits = {"still": (100, 1_000_000_000), "preview": (100, 83_333)}
    state = install_picamera(with_exposure_time_mode=True, frame_limits=limits)
    camera = Picamera2Camera(preview=True)
    camera.set_shutter(250_000)
    inst = state.instance
    assert inst.preview_config["controls"]["FrameDurationLimits"] == (100, 251_000)
    assert inst.created_config["controls"]["FrameDurationLimits"] == (100, 1_000_000_000)
    assert inst.set_controls_calls[-1]["FrameDurationLimits"] == (100, 251_000)
    camera.set_shutter(None)
    assert inst.preview_config["controls"]["FrameDurationLimits"] == (100, 83_333)
    assert inst.set_controls_calls[-1]["FrameDurationLimits"] == (100, 83_333)
    camera.close()


def test_a_short_shutter_keeps_the_frame_limits(install_picamera):
    limits = {"still": (100, 1_000_000_000), "preview": (100, 83_333)}
    state = install_picamera(with_exposure_time_mode=True, frame_limits=limits)
    camera = Picamera2Camera(preview=True)
    camera.set_shutter(4000)
    assert state.instance.preview_config["controls"]["FrameDurationLimits"] == (100, 83_333)
    camera.close()


@pytest.mark.parametrize("bad", [99, 1_000_001, -5, "x", 4000.5, True, "4000"])
def test_shutter_outside_the_range_is_refused(install_picamera, bad):
    install_picamera()
    camera = Picamera2Camera()
    with pytest.raises(CameraError):
        camera.set_shutter(bad)
    assert camera.shutter_us is None
    camera.close()


def test_integral_float_shutter_is_accepted_as_whole_microseconds(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    camera.set_shutter(4000.0)
    assert camera.shutter_us == 4000
    assert type(state.instance.set_controls_calls[-1]["ExposureTime"]) is int
    camera.close()


def test_a_configuration_without_frame_limits_falls_back_to_the_camera_range(
    install_picamera,
):
    state = install_picamera(
        with_exposure_time_mode=True, control_frame_limits=(100, 83_333, 33_333),
    )
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    camera.set_shutter(250_000)
    assert inst.set_controls_calls[-1]["FrameDurationLimits"] == (100, 251_000)
    assert inst.preview_config["controls"]["FrameDurationLimits"] == (100, 251_000)
    assert inst.created_config["controls"]["FrameDurationLimits"] == (100, 251_000)
    camera.set_shutter(None)
    assert inst.set_controls_calls[-1]["FrameDurationLimits"] == (100, 83_333)
    assert inst.preview_config["controls"]["FrameDurationLimits"] == (100, 83_333)
    camera.close()


def test_a_rejected_shutter_changes_nothing(install_picamera):
    state = install_picamera(with_exposure_time_mode=True)
    camera = Picamera2Camera(preview=True)
    before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError):
        camera.set_shutter(4000)
    assert camera.shutter_us is None
    assert state.instance.preview_config["controls"] == before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


def test_fixed_iso_uses_analogue_gain_mode_when_available(install_picamera, capsys):
    state = install_picamera(with_analogue_gain_mode=True)
    camera = Picamera2Camera(preview=True)
    assert "AnalogueGainMode" in capsys.readouterr().err
    assert camera.iso is None
    camera.set_iso(400)
    live = state.instance.set_controls_calls[-1]
    assert live == {"AnalogueGainMode": _AnalogueGainModeEnum.Manual, "AnalogueGain": 4.0}
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["AnalogueGainMode"] == _AnalogueGainModeEnum.Manual
        assert config["controls"]["AnalogueGain"] == 4.0
    assert camera.iso == 400
    camera.set_iso(None)
    assert state.instance.set_controls_calls[-1] == {
        "AnalogueGainMode": _AnalogueGainModeEnum.Auto
    }
    assert camera.iso is None
    camera.close()


def test_returning_to_auto_iso_drops_the_stale_gain_from_both_configurations(
    install_picamera,
):
    """The same hazard as a stale ``ExposureTime``: every configure would send the
    old ``AnalogueGain`` again beside ``AnalogueGainMode=Auto``."""
    state = install_picamera(with_analogue_gain_mode=True)
    camera = Picamera2Camera(preview=True)
    camera.set_iso(800)
    camera.set_iso(None)
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["AnalogueGainMode"] == _AnalogueGainModeEnum.Auto
        assert "AnalogueGain" not in config["controls"]
    camera.close()


def test_fixed_iso_falls_back_to_analogue_gain_on_older_libcamera(install_picamera, capsys):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    assert "ISO via AnalogueGain (legacy libcamera)" in capsys.readouterr().err
    camera.set_iso(1600)
    assert state.instance.set_controls_calls[-1] == {"AnalogueGain": 16.0}
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["AnalogueGain"] == 16.0
        assert "AnalogueGainMode" not in config["controls"]
    assert camera.iso == 1600
    camera.set_iso(None)
    assert state.instance.set_controls_calls[-1] == {"AnalogueGain": 0.0}
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"]["AnalogueGain"] == 0.0
    assert camera.iso is None
    camera.close()


@pytest.mark.parametrize("bad", [50, 3200, 99, 1601, -400, 400.5, True, "400", "x"])
def test_iso_outside_the_range_is_refused(install_picamera, bad):
    state = install_picamera()
    camera = Picamera2Camera()
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError):
        camera.set_iso(bad)
    assert camera.iso is None
    assert len(state.instance.set_controls_calls) == sent
    camera.close()


def test_iso_above_the_sensor_gain_limit_is_refused(install_picamera):
    state = install_picamera(analogue_gain=(1.0, 8.0, 1.0))
    camera = Picamera2Camera()
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError, match="gain"):
        camera.set_iso(1600)
    assert camera.iso is None
    assert len(state.instance.set_controls_calls) == sent
    camera.set_iso(800)
    assert camera.iso == 800
    camera.close()


def test_integral_float_iso_is_accepted_as_a_whole_number(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    camera.set_iso(200.0)
    assert camera.iso == 200 and type(camera.iso) is int
    assert state.instance.set_controls_calls[-1] == {"AnalogueGain": 2.0}
    camera.close()


def test_a_rejected_iso_changes_nothing(install_picamera):
    state = install_picamera(with_analogue_gain_mode=True)
    camera = Picamera2Camera(preview=True)
    before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError, match="failed to set the ISO"):
        camera.set_iso(400)
    assert camera.iso is None
    assert state.instance.preview_config["controls"] == before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


@pytest.mark.parametrize(("setter", "value", "enum", "attr", "kwargs"), [
    ("set_iso", 400, "AnalogueGainModeEnum", "iso", {"with_analogue_gain_mode": True}),
    ("set_shutter", 4000, "ExposureTimeModeEnum", "shutter_us",
     {"with_exposure_time_mode": True}),
])
def test_a_failure_building_the_controls_is_a_camera_error_and_changes_nothing(
    install_picamera, setter, value, enum, attr, kwargs,
):
    """The mode enums are looked up lazily on every call. A lookup that fails must
    not escape as an AttributeError: the viewfinder loop only survives CameraError."""
    import sys

    state = install_picamera(**kwargs)
    camera = Picamera2Camera(preview=True)
    before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])
    sent = len(state.instance.set_controls_calls)
    delattr(sys.modules["libcamera"].controls, enum)
    with pytest.raises(CameraError, match="Picamera2 failed to set the"):
        getattr(camera, setter)(value)
    assert getattr(camera, attr) is None
    assert len(state.instance.set_controls_calls) == sent
    assert state.instance.preview_config["controls"] == before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


def test_exposure_is_adjustable_unless_ae_lock_was_given(install_picamera):
    install_picamera()
    camera = Picamera2Camera()
    assert camera.exposure_adjustable is True
    camera.close()
    locked = Picamera2Camera(ae_lock=True)
    assert locked.exposure_adjustable is False
    locked.close()


@pytest.mark.parametrize(("setter", "value"), [
    ("set_shutter", 4000), ("set_shutter", None), ("set_iso", 400), ("set_iso", None),
])
def test_ae_lock_refuses_a_fixed_shutter_or_iso_and_changes_nothing(
    install_picamera, setter, value,
):
    """With auto-exposure off there is nothing for a fixed shutter or ISO to work
    against, and stepping out and back would leave ``AeEnable: False`` beside a
    mode set to Auto in the configurations."""
    state = install_picamera(with_exposure_time_mode=True, with_analogue_gain_mode=True)
    camera = Picamera2Camera(ae_lock=True, preview=True)
    before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError, match="--ae-lock"):
        getattr(camera, setter)(value)
    assert camera.shutter_us is None and camera.iso is None
    assert len(state.instance.set_controls_calls) == sent
    assert state.instance.preview_config["controls"] == before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


def test_iso_on_a_closed_camera_is_an_error(install_picamera):
    install_picamera()
    camera = Picamera2Camera()
    camera.close()
    with pytest.raises(CameraError, match="closed"):
        camera.set_iso(400)
    assert camera.iso is None


def test_max_gain_comes_from_the_sensor_controls(install_picamera):
    install_picamera(analogue_gain=(1.0, 22.26, 1.0))
    camera = Picamera2Camera()
    assert camera.max_gain == pytest.approx(22.26)
    camera.close()
    install_picamera()
    camera = Picamera2Camera()
    assert camera.max_gain is None
    camera.close()


def test_preview_read_that_gets_no_frame_is_a_stall_not_a_hang(install_picamera):
    """Seen on the Pi 4 on 2026-10-10: the sensor kept streaming but Picamera2
    stopped handing frames over, and a ``capture_request()`` with no timeout
    held the camera lock for good, so the screen froze and every later shutter
    press queued behind it."""
    state = install_picamera(capture_error=TimeoutError())
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    with pytest.raises(CameraStalled) as caught:
        camera.read(full=False)
    inst = state.instance
    assert inst.waits == [pytest.approx(5.0)]
    # The timed-out job is still queued inside Picamera2; left there, it would
    # take the next frame and never release it.
    assert inst.cancel_count == 1
    message = str(caught.value)
    assert "no preview frame within 5.0 s" in message
    assert "started=True" in message and "completed=0" in message and "frames=0" in message
    # The lock was released: the next read gets as far as the camera again.
    with pytest.raises(CameraStalled):
        camera.read(full=False)
    assert inst.capture_count == 2
    camera.close()


def test_still_that_gets_no_frame_is_a_stall_with_a_longer_limit(install_picamera):
    state = install_picamera(capture_error=TimeoutError())
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    with pytest.raises(CameraStalled, match="no still frame within 30.0 s"):
        camera.read(full=True)
    assert state.instance.waits == [pytest.approx(30.0)]
    assert state.instance.cancel_count == 1
    camera.close()


def test_a_long_shutter_lengthens_the_frame_timeout(install_picamera):
    """A one-second shutter makes every frame take a second; that is not a stall."""
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    camera.set_shutter(1_000_000)
    camera.read(full=False)
    assert state.instance.waits == [pytest.approx(5.0 + 4 * 1.0)]
    camera.close()


def test_a_stall_is_reported_even_when_the_diagnostics_are_unavailable(install_picamera):
    state = install_picamera(capture_error=TimeoutError())
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    inst = state.instance
    del inst.started, inst.completed_requests, inst.frames
    inst.cancel_all_and_flush = lambda: (_ for _ in ()).throw(RuntimeError("gone"))
    with pytest.raises(CameraStalled, match="started=\\? completed=\\? frames=\\?"):
        camera.read(full=False)
    camera.close()


def test_colour_temperature_reaches_the_camera_and_both_configurations(install_picamera):
    """Measured on the IMX477 on 2026-10-11: ``ColourTemperature`` alone fixes the
    gains, and ``AwbEnable`` returns to auto. Each configure reapplies its own
    controls, so the value has to be in both or the next photo undoes it."""
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    assert camera.wb_adjustable and camera.colour_temperature_k is None
    camera.set_colour_temperature(6500)
    inst = state.instance
    assert inst.set_controls_calls[-1] == {"ColourTemperature": 6500}
    for config in (inst.created_config, inst.preview_config):
        assert config["controls"]["ColourTemperature"] == 6500
        assert "AwbEnable" not in config["controls"]
    assert camera.colour_temperature_k == 6500
    camera.set_colour_temperature(None)
    assert inst.set_controls_calls[-1] == {"AwbEnable": True}
    for config in (inst.created_config, inst.preview_config):
        assert config["controls"]["AwbEnable"] is True
        assert "ColourTemperature" not in config["controls"]
    assert camera.colour_temperature_k is None
    camera.close()


def test_a_starting_colour_temperature_is_in_the_first_controls(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True, colour_temperature=5600)
    inst = state.instance
    assert camera.colour_temperature_k == 5600
    assert inst.created_config["controls"]["ColourTemperature"] == 5600
    assert inst.preview_config["controls"]["ColourTemperature"] == 5600
    assert inst.set_controls_calls[0]["ColourTemperature"] == 5600
    camera.close()


@pytest.mark.parametrize("bad", [2000, 9001, 5600.5, True, "5600"])
def test_colour_temperature_outside_the_range_is_refused(install_picamera, bad):
    install_picamera()
    camera = Picamera2Camera(preview=True)
    with pytest.raises(CameraError):
        camera.set_colour_temperature(bad)
    assert camera.colour_temperature_k is None
    camera.close()


def test_a_rejected_colour_temperature_changes_nothing(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    before = dict(state.instance.preview_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError):
        camera.set_colour_temperature(6500)
    assert camera.colour_temperature_k is None
    assert state.instance.preview_config["controls"] == before
    camera.close()


@pytest.mark.parametrize("kwargs", [
    {"awb_lock": True}, {"colour_gains": (2.0, 1.5)},
])
def test_a_fixed_white_balance_from_start_up_is_not_adjustable(install_picamera, kwargs):
    install_picamera()
    camera = Picamera2Camera(preview=True, **kwargs)
    assert camera.wb_adjustable is False
    with pytest.raises(CameraError, match="white balance is fixed"):
        camera.set_colour_temperature(6500)
    camera.close()


def test_a_camera_without_the_control_is_not_adjustable(install_picamera):
    install_picamera(with_colour_temperature=False)
    camera = Picamera2Camera(preview=True)
    assert camera.wb_adjustable is False
    with pytest.raises(CameraError, match="ColourTemperature"):
        camera.set_colour_temperature(6500)
    camera.close()


def test_start_up_colour_temperature_conflicts_are_refused(install_picamera):
    install_picamera()
    with pytest.raises(CameraError):
        Picamera2Camera(colour_temperature=6500, awb_lock=True)
    with pytest.raises(CameraError):
        Picamera2Camera(colour_temperature=1000)


def test_a_still_reports_how_long_the_camera_stages_took(install_picamera):
    """Step 1 of the pipeline roadmap: about two seconds of a shot are spent
    outside the grading pipeline, and only the backend can say on what."""
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    install_picamera(request=FakeRequest(big))
    camera = Picamera2Camera(save_dng=True)
    frame = camera.read(full=True)
    assert set(frame.timings) == {"request", "array", "dng"}
    assert all(isinstance(v, float) and v >= 0.0 for v in frame.timings.values())
    camera.close()


def test_a_still_without_a_dng_has_no_dng_stage(install_picamera):
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    install_picamera(request=FakeRequest(big))
    camera = Picamera2Camera(save_dng=False)
    assert set(camera.read(full=True).timings) == {"request", "array"}
    camera.close()


def test_a_preview_frame_carries_no_stage_timings(install_picamera):
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    install_picamera(request=FakeRequest(small))
    camera = Picamera2Camera(preview=(640, 480), save_dng=False)
    assert camera.read(full=False).timings is None
    camera.close()


@pytest.mark.parametrize(("setter", "attr", "control", "value", "expected"), [
    ("set_ae_constraint", "ae_constraint", "AeConstraintMode", "highlight",
     _AeConstraintModeEnum.Highlight),
    ("set_ae_metering", "ae_metering", "AeMeteringMode", "spot", _AeMeteringModeEnum.Spot),
])
def test_ae_mode_setter_sends_the_control_and_writes_both_configurations(
    install_picamera, setter, attr, control, value, expected,
):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    getattr(camera, setter)(value)
    assert state.instance.set_controls_calls[-1] == {control: expected}
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"][control] == expected
    assert getattr(camera, attr) == value
    camera.close()


def test_ae_mode_attributes_start_at_the_constructor_values(install_picamera):
    install_picamera()
    camera = Picamera2Camera(ae_constraint="shadows", ae_metering="matrix")
    assert camera.ae_constraint == "shadows"
    assert camera.ae_metering == "matrix"
    camera.close()


@pytest.mark.parametrize(("setter", "attr", "bad"), [
    ("set_ae_constraint", "ae_constraint", "bright"),
    ("set_ae_metering", "ae_metering", "center"),
])
def test_an_unknown_ae_mode_is_refused(install_picamera, setter, attr, bad):
    state = install_picamera()
    camera = Picamera2Camera()
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError, match="Unknown"):
        getattr(camera, setter)(bad)
    assert len(state.instance.set_controls_calls) == sent
    assert getattr(camera, attr) in ("normal", "centre")
    camera.close()


@pytest.mark.parametrize(("setter", "attr", "value", "before"), [
    ("set_ae_constraint", "ae_constraint", "highlight", "normal"),
    ("set_ae_metering", "ae_metering", "matrix", "centre"),
])
def test_a_rejected_ae_mode_changes_nothing(install_picamera, setter, attr, value, before):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    preview_before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError, match="failed to set"):
        getattr(camera, setter)(value)
    assert getattr(camera, attr) == before
    assert state.instance.preview_config["controls"] == preview_before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


@pytest.mark.parametrize(("setter", "control", "value"), [
    ("set_ae_constraint", "AeConstraintMode", "highlight"),
    ("set_ae_metering", "AeMeteringMode", "spot"),
])
def test_an_unsupported_ae_mode_is_refused(install_picamera, setter, control, value):
    state = install_picamera(ae_controls=())
    camera = Picamera2Camera()
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError):
        getattr(camera, setter)(value)
    assert len(state.instance.set_controls_calls) == sent
    camera.close()


@pytest.mark.parametrize(("setter", "value"), [
    ("set_ae_constraint", "highlight"), ("set_ae_metering", "spot"),
])
def test_an_ae_mode_on_a_closed_camera_is_an_error(install_picamera, setter, value):
    install_picamera()
    camera = Picamera2Camera()
    camera.close()
    with pytest.raises(CameraError, match="closed"):
        getattr(camera, setter)(value)


def test_ae_support_flags_follow_the_camera_controls(install_picamera):
    install_picamera()
    camera = Picamera2Camera()
    assert camera.ae_constraint_supported is True
    assert camera.ae_metering_supported is True
    camera.close()

    install_picamera(ae_controls=("AeMeteringMode",))
    camera = Picamera2Camera()
    assert camera.ae_constraint_supported is False
    assert camera.ae_metering_supported is True
    camera.close()


def test_ae_support_flags_are_false_without_the_libcamera_enums(install_picamera):
    install_picamera(with_ae_enums=False)
    camera = Picamera2Camera()
    assert camera.ae_constraint_supported is False
    assert camera.ae_metering_supported is False
    with pytest.raises(CameraError):
        camera.set_ae_constraint("highlight")
    camera.close()


def test_stream_info_records_the_ae_modes_opened_with(install_picamera):
    install_picamera()
    camera = Picamera2Camera(ae_constraint="shadows", ae_metering="matrix")
    assert camera.stream_info.ae_constraint == "shadows"
    assert camera.stream_info.ae_metering == "matrix"
    assert camera.stream_info.to_dict()["ae_metering"] == "matrix"
    camera.close()


def test_stream_info_leaves_an_unsupported_ae_mode_out(install_picamera):
    install_picamera(ae_controls=("AeMeteringMode",))
    camera = Picamera2Camera(ae_constraint="shadows", ae_metering="spot")
    assert camera.stream_info.ae_constraint is None
    assert camera.stream_info.ae_metering == "spot"
    assert "ae_constraint" not in camera.stream_info.to_dict()
    camera.close()

    install_picamera(with_ae_enums=False)
    camera = Picamera2Camera()
    assert not {"ae_constraint", "ae_metering"} & set(camera.stream_info.to_dict())
    camera.close()


def test_stream_info_records_the_ae_modes_under_ae_lock(install_picamera):
    install_picamera()
    camera = Picamera2Camera(ae_lock=True, ae_metering="spot")
    assert camera.stream_info.ae_constraint == "normal"
    assert camera.stream_info.ae_metering == "spot"
    camera.close()


@pytest.mark.parametrize(("setter", "field", "value"), [
    ("set_ae_constraint", "ae_constraint", "highlight"),
    ("set_ae_metering", "ae_metering", "matrix"),
])
def test_an_ae_mode_setter_updates_the_stream_info(install_picamera, setter, field, value):
    install_picamera()
    camera = Picamera2Camera()
    getattr(camera, setter)(value)
    assert getattr(camera.stream_info, field) == value
    camera.close()


@pytest.mark.parametrize(("setter", "field", "value", "before"), [
    ("set_ae_constraint", "ae_constraint", "highlight", "normal"),
    ("set_ae_metering", "ae_metering", "matrix", "centre"),
])
def test_a_rejected_ae_mode_leaves_the_stream_info(
    install_picamera, setter, field, value, before,
):
    state = install_picamera()
    camera = Picamera2Camera()

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError):
        getattr(camera, setter)(value)
    assert getattr(camera.stream_info, field) == before
    camera.close()


@pytest.mark.parametrize("background", [False, True])
def test_a_record_after_an_ae_change_carries_the_new_modes(
    install_picamera, tmp_path, background,
):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8), (NATIVE_SIZE[1], NATIVE_SIZE[0], 3),
    )
    install_picamera(request=FakeRequest(array))
    camera = Picamera2Camera(save_dng=False)
    artifact_dir = tmp_path / "artifact"
    write_artifact(
        artifact_dir, LUT3D.identity(2), NormalizeParams(white_balance=False),
        GrainParams(enabled=False),
    )
    session = CaptureSession(
        camera, Pipeline(Artifacts.load(artifact_dir)), tmp_path / "captures",
        seed_rng=np.random.default_rng(0), save_dng=False,
        background=background, state_dir=tmp_path / "state", log=lambda _: None,
    )
    try:
        first = session.capture()
        camera.set_ae_metering("matrix")
        camera.set_ae_constraint("shadows")
        second = session.capture()
        # A later change does not reach a photo already taken, even one still
        # being graded behind the shutter.
        camera.set_ae_metering("spot")
        assert session.close_grades(timeout=30)
    finally:
        camera.close()
    assert (first.record["ae_constraint"], first.record["ae_metering"]) == ("normal", "centre")
    assert (second.record["ae_constraint"], second.record["ae_metering"]) == (
        "shadows", "matrix",
    )
    record_path, = (tmp_path / "captures").rglob("captures.jsonl")
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert [(r["ae_constraint"], r["ae_metering"]) for r in records] == [
        ("normal", "centre"), ("shadows", "matrix"),
    ]


def test_set_save_dng_false_means_the_next_still_has_no_dng(install_picamera):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8),
        (2592, 4608, 3),
    )
    request = FakeRequest(array)
    install_picamera(request=request)
    camera = Picamera2Camera()
    camera.set_save_dng(False)
    frame = camera.read()
    assert frame.dng is None
    assert request.save_dng_calls == 0
    camera.set_save_dng(True)
    frame = camera.read()
    assert frame.dng[:4] == b"II*\x00"
    assert request.save_dng_calls == 1
    camera.close()


# --- the ISP's image controls: sharpness, contrast, saturation -----------------


def test_the_image_controls_open_with_the_values_given(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True, sharpness=0.5, contrast=1.25, saturation=1.5)
    inst = state.instance
    for controls in (inst.created_config["controls"], inst.preview_config["controls"],
                     inst.set_controls_calls[0]):
        assert (controls["Sharpness"], controls["Contrast"], controls["Saturation"]) == (
            0.5, 1.25, 1.5,
        )
    assert (camera.sharpness, camera.contrast, camera.saturation) == (0.5, 1.25, 1.5)
    info = camera.stream_info
    assert (info.sharpness, info.contrast, info.saturation) == (0.5, 1.25, 1.5)
    assert camera.image_controls_supported == {"sharpness", "contrast", "saturation"}
    camera.close()


@pytest.mark.parametrize("bad", [True, "1.0", -0.5, float("nan"), float("inf"), None])
def test_an_unusable_image_value_at_open_is_an_error(install_picamera, bad):
    install_picamera()
    with pytest.raises(CameraError, match="contrast"):
        Picamera2Camera(contrast=bad)


@pytest.mark.parametrize(("key", "control"), [
    ("sharpness", "Sharpness"), ("contrast", "Contrast"), ("saturation", "Saturation"),
])
def test_set_image_control_sends_it_and_writes_both_configurations(
    install_picamera, key, control,
):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    camera.set_image_control(key, 1.25)
    assert state.instance.set_controls_calls[-1] == {control: 1.25}
    for config in (state.instance.created_config, state.instance.preview_config):
        assert config["controls"][control] == 1.25
    assert getattr(camera, key) == 1.25
    assert getattr(camera.stream_info, key) == 1.25
    assert camera.stream_info.to_dict()[key] == 1.25
    camera.close()


def test_a_still_after_an_image_change_is_configured_with_it(install_picamera):
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(big))
    camera = Picamera2Camera(preview=True, save_dng=False)
    camera.set_image_control("saturation", 0.5)
    camera.read(full=True)
    (still,) = state.instance.switch_calls
    assert still["controls"]["Saturation"] == 0.5
    camera.close()


def test_an_unknown_image_control_or_value_is_refused(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError, match="Unknown image control"):
        camera.set_image_control("brightness", 1.0)
    with pytest.raises(CameraError, match="sharpness"):
        camera.set_image_control("sharpness", -1)
    assert len(state.instance.set_controls_calls) == sent
    assert camera.sharpness == 1.0
    camera.close()


def test_a_rejected_image_control_changes_nothing(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    preview_before = dict(state.instance.preview_config["controls"])
    still_before = dict(state.instance.created_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError, match="failed to set the contrast"):
        camera.set_image_control("contrast", 1.5)
    assert camera.contrast == 1.0
    assert camera.stream_info.contrast == 1.0
    assert state.instance.preview_config["controls"] == preview_before
    assert state.instance.created_config["controls"] == still_before
    camera.close()


def test_an_image_control_on_a_closed_camera_is_an_error(install_picamera):
    install_picamera()
    camera = Picamera2Camera()
    camera.close()
    with pytest.raises(CameraError, match="closed"):
        camera.set_image_control("contrast", 1.25)


def test_an_image_control_the_camera_does_not_list_is_refused_and_left_out(
    install_picamera, capsys,
):
    state = install_picamera(image_controls=("Sharpness", "Contrast"))
    camera = Picamera2Camera(saturation=1.5, contrast=0.75)
    assert camera.image_controls_supported == {"sharpness", "contrast"}
    err = capsys.readouterr().err
    assert "no Saturation control, so saturation 1.5 was not applied" in err
    # Sent at normal, as it always was, but not claimed in the record.
    assert state.instance.set_controls_calls[0]["Saturation"] == 1.0
    assert camera.saturation == 1.0
    assert camera.stream_info.saturation is None
    values = camera.stream_info.to_dict()
    assert "saturation" not in values
    assert values["contrast"] == 0.75
    sent = len(state.instance.set_controls_calls)
    with pytest.raises(CameraError, match="no Saturation control"):
        camera.set_image_control("saturation", 1.25)
    assert len(state.instance.set_controls_calls) == sent
    camera.close()


def test_a_missing_image_control_at_normal_is_silent(install_picamera, capsys):
    install_picamera(image_controls=())
    camera = Picamera2Camera()
    assert "warning" not in capsys.readouterr().err
    assert not {"sharpness", "contrast", "saturation"} & set(camera.stream_info.to_dict())
    camera.close()


def _image_session(tmp_path, camera, **kwargs):
    artifact_dir = tmp_path / "artifact"
    write_artifact(
        artifact_dir, LUT3D.identity(2), NormalizeParams(white_balance=False),
        GrainParams(enabled=False),
    )
    return CaptureSession(
        camera, Pipeline(Artifacts.load(artifact_dir)), tmp_path / "captures",
        seed_rng=np.random.default_rng(0), save_dng=False, log=lambda _: None, **kwargs,
    )


def _image_records(tmp_path):
    record_path, = (tmp_path / "captures").rglob("captures.jsonl")
    return [json.loads(line) for line in record_path.read_text().splitlines()]


def _image_fields(record):
    return record["sharpness"], record["contrast"], record["saturation"]


@pytest.mark.parametrize("background", [False, True])
def test_a_record_after_an_image_change_carries_the_new_values(
    install_picamera, tmp_path, background,
):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8), (NATIVE_SIZE[1], NATIVE_SIZE[0], 3),
    )
    install_picamera(request=FakeRequest(array))
    camera = Picamera2Camera(save_dng=False, contrast=1.25)
    session = _image_session(tmp_path, camera, background=background,
                             state_dir=tmp_path / "state")
    try:
        first = session.capture()
        camera.set_image_control("sharpness", 0.5)
        camera.set_image_control("saturation", 1.5)
        second = session.capture()
        # A later change does not reach a photo already taken.
        camera.set_image_control("contrast", 0.75)
        assert session.close_grades(timeout=30)
    finally:
        camera.close()
    assert _image_fields(first.record) == (1.0, 1.25, 1.0)
    assert _image_fields(second.record) == (0.5, 1.25, 1.5)
    records = _image_records(tmp_path)
    assert [_image_fields(r) for r in records] == [(1.0, 1.25, 1.0), (0.5, 1.25, 1.5)]
    for record in records:
        keys = list(record)
        at = keys.index("ae_metering")
        assert keys[at + 1:at + 4] == ["sharpness", "contrast", "saturation"]


@pytest.mark.parametrize("background", [False, True])
def test_both_records_of_a_double_carry_the_image_values_of_their_shutter(
    install_picamera, tmp_path, background,
):
    array = np.broadcast_to(
        np.array([[[10, 40, 230]]], dtype=np.uint8), (NATIVE_SIZE[1], NATIVE_SIZE[0], 3),
    )
    install_picamera(request=FakeRequest(array))
    camera = Picamera2Camera(save_dng=False)
    session = _image_session(tmp_path, camera, background=background,
                             state_dir=tmp_path / "state")
    try:
        session.set_double_exposure(True)
        session.capture()
        camera.set_image_control("contrast", 1.5)
        session.capture()
        assert session.close_grades(timeout=30)
    finally:
        camera.close()
    records = _image_records(tmp_path)
    assert [r["double"]["index"] for r in records] == [1, 2]
    assert [_image_fields(r) for r in records] == [(1.0, 1.0, 1.0), (1.0, 1.5, 1.0)]



# --- noise reduction: HighQuality for stills, Minimal for the live view ------------


def test_the_preview_takes_minimal_noise_reduction_and_the_still_high_quality(
    install_picamera,
):
    state = install_picamera()
    camera = Picamera2Camera(preview=True)
    inst = state.instance
    assert inst.created_config["controls"]["NoiseReductionMode"] == (
        _NoiseReductionModeEnum.HighQuality)
    assert inst.preview_config["controls"]["NoiseReductionMode"] == (
        _NoiseReductionModeEnum.Minimal)
    # The camera runs the preview, so it is started with the preview's mode.
    assert inst.set_controls_calls[0]["NoiseReductionMode"] == _NoiseReductionModeEnum.Minimal
    camera.close()


def test_without_a_preview_the_camera_runs_high_quality(install_picamera):
    state = install_picamera()
    camera = Picamera2Camera()
    assert state.instance.set_controls_calls[0]["NoiseReductionMode"] == (
        _NoiseReductionModeEnum.HighQuality)
    camera.close()


def test_an_older_libcamera_leaves_both_configurations_without_noise_reduction(
    install_picamera,
):
    state = install_picamera(with_noise_reduction=False)
    camera = Picamera2Camera(preview=True)
    assert "NoiseReductionMode" not in state.instance.created_config["controls"]
    assert "NoiseReductionMode" not in state.instance.preview_config["controls"]
    assert "NoiseReductionMode" not in state.instance.set_controls_calls[0]
    camera.close()


def test_runtime_changes_keep_each_configurations_noise_reduction(install_picamera):
    big = np.zeros((NATIVE_SIZE[1], NATIVE_SIZE[0], 3), dtype=np.uint8)
    state = install_picamera(request=FakeRequest(big), with_exposure_time_mode=True,
                             with_analogue_gain_mode=True)
    camera = Picamera2Camera(preview=True, save_dng=False)
    camera.set_ae_constraint("highlight")
    camera.set_ae_metering("spot")
    camera.set_image_control("contrast", 1.25)
    camera.set_shutter(10_000)
    camera.set_iso(400)
    camera.set_colour_temperature(5600)
    camera.reassert_controls()
    camera.restart_preview()
    inst = state.instance
    for sent in inst.set_controls_calls[1:]:
        assert "NoiseReductionMode" not in sent
    assert inst.created_config["controls"]["NoiseReductionMode"] == (
        _NoiseReductionModeEnum.HighQuality)
    assert inst.preview_config["controls"]["NoiseReductionMode"] == (
        _NoiseReductionModeEnum.Minimal)
    # A still from the live view is configured with the still's own mode.
    camera.read(full=True)
    (still,) = inst.switch_calls
    assert still["controls"]["NoiseReductionMode"] == _NoiseReductionModeEnum.HighQuality
    (restart, _wait), = inst.switch_mode_calls
    assert restart["controls"]["NoiseReductionMode"] == _NoiseReductionModeEnum.Minimal
    camera.close()
