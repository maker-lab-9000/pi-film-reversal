# LCD Viewfinder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live viewfinder with a light-meter readout on the Waveshare 2.8" SPI LCD, with an on-screen shutter and EV buttons, running inside `pifilm-capture` at boot, on both the IMX477 and the IMX708.

**Architecture:** A new `pifilm/display/` package holds an ST7789 frame-push driver, a CST3530 touch driver, a pure exposure-meter module, a pure Pillow renderer with hit-testing, and a `ViewfinderLoop` state machine that shares the existing `CaptureController` with the Stick. The Picamera2 backend gains a preview mode (binned sensor mode, full-resolution still via `switch_mode_and_capture_request`), a request lock, a runtime `set_ev`, and multi-sensor support (native size from the sensor, automatic tuning, fixed-lens autofocus skipped). `app.py` grows a `--display` flag that falls back to headless on any display fault.

**Tech Stack:** Python 3.11+, NumPy, Pillow ≥ 10.1, Picamera2/libcamera (apt), `spidev`, `smbus2`, `gpiozero` (apt, Pi only; all imported lazily), pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-lcd-viewfinder-design.md`

## Global Constraints

- Hardware libraries (`picamera2`, `libcamera`, `spidev`, `smbus2`, `gpiozero`) are imported only inside constructors or factory functions, never at module top; the whole suite runs on a Mac with none installed.
- `pifilm/display/` depends only on NumPy, Pillow and `pifilm` modules; no `cv2` (use `require_cv2()` if ever needed — it is not needed here).
- Without `--display`, `pifilm-capture` and the Picamera2 backend behave byte-for-byte as before: same configuration calls, same records. Existing tests keep passing unchanged unless a task says otherwise.
- Display is 320×240 landscape. Pins (BCM): DC 25, RST 27, BL 18, SPI0 CE0; touch I2C1 address `0x58`, TP_RST 17.
- All shots go through `CaptureController.submit`; nothing in `pifilm/display/` calls `CaptureSession.capture` directly.
- Never put secrets in code, commands or commits. `.env` is data only.
- TDD: write the failing test, watch it fail, implement, watch it pass, commit. `.venv/bin/pytest -q -m 'not slow'` green and `.venv/bin/ruff check .` clean at every commit.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
| --- | --- |
| `pifilm/capture/controller.py` (modify) | `finished_count` + `last_finished_job` on the snapshot |
| `pifilm/capture/camera.py` (modify) | `StreamInfo.autofocus`; `FakeCamera.set_ev`, `FakeCamera(metadata=...)` |
| `pifilm/capture/picamera.py` (modify) | multi-sensor, preview mode, request lock, `set_ev` |
| `pifilm/display/__init__.py` (create) | `DisplayError` |
| `pifilm/display/st7789.py` (create) | ST7789 driver, `open_waveshare28` |
| `pifilm/display/cst3530.py` (create) | touch driver, `decode_points`, `to_display`, `TapDetector`, `open_waveshare28_touch` |
| `pifilm/display/meter.py` (create) | `MeterReading`, `compute_reading`, formatters |
| `pifilm/display/ui.py` (create) | `render_live`, `render_review`, `render_message`, `hit`, `Action` |
| `pifilm/display/viewfinder.py` (create) | `ViewfinderLoop` |
| `pifilm/display/fake.py` (create) | `FileDisplay`, `NoTouch` for `--display fake` |
| `pifilm/capture/app.py` (modify) | flags, wiring, fallback |
| `pyproject.toml` (modify) | `Pillow>=10.1` |
| `deploy/pifilm-capture.service.example` (modify) | `--display waveshare28` |
| `docs/lcd-viewfinder.md` (create), `docs/setup.md`, `README.md`, `CLAUDE.md`, `docs/picamera2-bringup.md`, `docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md` (modify) | documentation |
| tests: `tests/test_capture_controller.py`, `tests/test_camera.py`, `tests/test_picamera.py`, `tests/test_app.py` (modify); `tests/test_display_st7789.py`, `tests/test_display_cst3530.py`, `tests/test_display_meter.py`, `tests/test_display_ui.py`, `tests/test_viewfinder.py` (create) | |

---

### Task 1: Controller reports finished jobs

**Files:**
- Modify: `pifilm/capture/controller.py`
- Test: `tests/test_capture_controller.py`

**Interfaces:**
- Produces: `ControllerSnapshot.finished_count: int` (increments on every job that ends, complete or failed) and `ControllerSnapshot.last_finished_job: JobSnapshot | None` (the most recent ended job, either state). `last_completed_job` keeps its current meaning (complete only) for the remote server.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_controller.py` (reuse the file's existing helpers for a session double; if it has none, use this one):

```python
import time

from pifilm.capture.camera import CameraError
from pifilm.capture.controller import CaptureController


class _Session:
    def __init__(self, fail=False):
        self.camera = None
        self.fail = fail

    def capture(self):
        if self.fail:
            raise CameraError("boom")
        return "result"


def _wait_finished(controller, request_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = controller.status(request_id)
        if job is not None and job.state in ("complete", "failed"):
            return job
        time.sleep(0.005)
    raise AssertionError("job did not finish")


def test_snapshot_counts_finished_jobs_of_both_states():
    controller = CaptureController(_Session())
    try:
        assert controller.snapshot().finished_count == 0
        assert controller.snapshot().last_finished_job is None
        controller.submit("a")
        _wait_finished(controller, "a")
        snap = controller.snapshot()
        assert snap.finished_count == 1
        assert snap.last_finished_job.request_id == "a"
        assert snap.last_finished_job.state == "complete"
    finally:
        controller.close()


def test_failed_job_counts_as_finished_but_not_completed():
    controller = CaptureController(_Session(fail=True))
    try:
        controller.submit("b")
        _wait_finished(controller, "b")
        snap = controller.snapshot()
        assert snap.finished_count == 1
        assert snap.last_finished_job.state == "failed"
        assert snap.last_completed_job is None
    finally:
        controller.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_capture_controller.py -k "finished"`
Expected: FAIL with `AttributeError: 'ControllerSnapshot' object has no attribute 'finished_count'`

- [ ] **Step 3: Implement**

In `pifilm/capture/controller.py`:

```python
@dataclass(frozen=True)
class ControllerSnapshot:
    """An immutable summary safe to render from another thread."""

    active_job: JobSnapshot | None
    last_completed_job: JobSnapshot | None
    closed: bool
    # Every job that ended, complete or failed. A display polls this counter to
    # notice a shot taken from any trigger (Stick, SPACE, LCD) without the
    # controller knowing that displays exist.
    finished_count: int = 0
    last_finished_job: JobSnapshot | None = None
```

In `__init__` add `self._finished_count = 0` and `self._last_finished_job: JobSnapshot | None = None`. In `snapshot()` return
`ControllerSnapshot(active, self._last_completed_job, self._closed, self._finished_count, self._last_finished_job)`.
In `_finish`, inside the lock after `self._active_request_id = None`, add:

```python
            self._finished_count += 1
            self._last_finished_job = job
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_capture_controller.py tests/test_remote.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/controller.py tests/test_capture_controller.py
git commit -m "Count finished capture jobs on the controller snapshot"
```

---

### Task 2: Picamera2 backend works on any sensor

**Files:**
- Modify: `pifilm/capture/picamera.py`
- Modify: `pifilm/capture/camera.py` (`StreamInfo.autofocus`)
- Modify: `pifilm/capture/app.py:643` (pass `args.tuning_file`, may be `None`)
- Test: `tests/test_picamera.py`, `tests/test_app.py`

**Interfaces:**
- Produces: `Picamera2Camera(tuning_file: str | None = None, ...)`; `DEFAULT_TUNING_FILE = None`; `StreamInfo.autofocus: str | None`; `_stream_info(actual, tuning_label, native_size)`; `_apply_camera_controls(..., has_autofocus: bool)`; fake `FakePicamera2` gains `sensor_resolution`, `camera_controls`, `camera_properties`, and the installer options `sensor_resolution=`, `fixed_lens=False`, `model="imx708_wide"`.

- [ ] **Step 1: Extend the fake**

In `tests/test_picamera.py` `install(...)` add parameters `sensor_resolution=NATIVE_SIZE, fixed_lens=False, model="imx708_wide"`. In `FakePicamera2`:

```python
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
                ...  # existing counters unchanged
```

- [ ] **Step 2: Write the failing tests**

Replace `test_default_tuning_file_matches_the_initial_camera_variant` with:

```python
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
```

Also update the existing test that reads the IMX708 frame shape if it hard-codes `NATIVE_SIZE` only through `_actual_configuration()` — it should still pass because the installer default `sensor_resolution=NATIVE_SIZE`.

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_picamera.py -k "tuning or native_size or fixed_lens or records_its_mode"`
Expected: FAIL (`load_tuning_file(None)` recorded, `AttributeError: autofocus`, etc.)

- [ ] **Step 4: Implement**

`pifilm/capture/camera.py`: add `autofocus: str | None = None` to `StreamInfo` after `tuning_file`, and in `to_dict()`:

```python
        if self.autofocus is not None:
            values["autofocus"] = self.autofocus
```

`pifilm/capture/picamera.py`:

```python
DEFAULT_TUNING_FILE: str | None = None  # None: libcamera picks <sensor>.json itself
_MAIN_FORMAT = "RGB888"
```

Remove `_NATIVE_SIZE`. Constructor signature `tuning_file: str | None = DEFAULT_TUNING_FILE`. Replace the tuning/open block with:

```python
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
```

Inside the existing `try:` after `self._autofocus = autofocus`:

```python
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
            _apply_camera_controls(
                camera, autofocus, af_range, ae_lock, awb_lock, colour_gains,
                ae_constraint, ae_metering, ev, has_autofocus=has_autofocus,
            )
```

In `read()`, `expected_shape = (self._native_size[1], self._native_size[0], 3)`. `_stream_info(actual, tuning_file, native_size)` compares against `native_size`. `_apply_camera_controls(..., has_autofocus: bool)` wraps the two AF lines:

```python
    if has_autofocus:
        af_modes = {...}
        af_ranges = {...}
        cam_controls["AfMode"] = af_modes[autofocus]
        cam_controls["AfRange"] = af_ranges[af_range]
```

Update the module docstring: no longer IMX708-specific; native size from the sensor; tuning automatic unless given; fixed-lens sensors skip AF. In `app.py` replace `args.tuning_file or DEFAULT_TUNING_FILE` with `args.tuning_file` and fix the `--tuning-file` help to say "(default: libcamera's automatic choice for the detected sensor)". Update any test in `tests/test_app.py` or `tests/test_picamera.py` that imported `DEFAULT_TUNING_FILE` expecting a string.

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/pytest -q tests/test_picamera.py tests/test_app.py tests/test_camera.py && .venv/bin/ruff check .`
Expected: PASS, clean

- [ ] **Step 6: Commit**

```bash
git add pifilm/capture/picamera.py pifilm/capture/camera.py pifilm/capture/app.py tests/test_picamera.py tests/test_app.py
git commit -m "Picamera2 backend: native size from the sensor, automatic tuning, fixed-lens sensors"
```

---

### Task 3: Picamera2 preview mode, request lock, runtime EV

**Files:**
- Modify: `pifilm/capture/picamera.py`
- Modify: `pifilm/capture/camera.py` (`FakeCamera.set_ev`, `FakeCamera(metadata=...)`)
- Test: `tests/test_picamera.py`, `tests/test_camera.py`

**Interfaces:**
- Consumes: Task 2's `self._native_size`, fake installer options.
- Produces: `Picamera2Camera(..., preview: tuple[int, int] | None = None)`; `Picamera2Camera.set_ev(value: float) -> None`; `Picamera2Camera.ev: float`; `FakeCamera.set_ev(value)` / `FakeCamera.ev`; `FakeCamera(metadata: dict | None = None)` returned on every frame; `_binned_mode(sensor_modes, native) -> tuple[int, int]`. Fake `FakePicamera2` gains `create_preview_configuration`, `switch_mode_and_capture_request`, `sensor_modes`; installer option `sensor_modes=`.

- [ ] **Step 1: Extend the fake**

In `install(...)` add `sensor_modes=None`. In `FakePicamera2.__init__`:

```python
                self.sensor_modes = sensor_modes if sensor_modes is not None else [
                    {"size": (1536, 864)}, {"size": (2304, 1296)}, {"size": (4608, 2592)},
                ]
                self.preview_config = None
                self.configure_calls = []
                self.switch_calls = []
```

Add methods:

```python
            def create_preview_configuration(self, **kwargs):
                self.preview_config = kwargs
                return {"preview": kwargs}

            def switch_mode_and_capture_request(self, config):
                self.switch_calls.append(config)
                self.capture_count += 1
                if capture_error is not None:
                    raise capture_error
                return request
```

and make `configure` append to `self.configure_calls` as well as setting `configured_with`.

- [ ] **Step 2: Write the failing tests**

```python
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
```

And in `tests/test_camera.py`:

```python
def test_fake_camera_records_ev_and_returns_metadata():
    camera = FakeCamera([synthetic_frame(8, 8)], metadata={"Lux": 100.0})
    assert camera.ev == 0.0
    camera.set_ev(-0.5)
    assert camera.ev == -0.5
    assert camera.read(full=False).metadata == {"Lux": 100.0}
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_picamera.py -k "preview or binned or set_ev or serialised or unchanged" tests/test_camera.py -k "fake_camera_records_ev"`
Expected: FAIL (`unexpected keyword argument 'preview'`, `ImportError _binned_mode`, ...)

- [ ] **Step 4: Implement**

`pifilm/capture/camera.py` `FakeCamera`: add `metadata: dict | None = None` parameter stored as `self._metadata`, `self.ev = 0.0`, return `metadata=self._metadata` in `Frame`, and:

```python
    def set_ev(self, value: float) -> None:
        self.ev = float(value)
```

`pifilm/capture/picamera.py`: `import threading`. Constructor parameter `preview: tuple[int, int] | None = None`. Before the `try:` block: `self._request_lock = threading.Lock()`, `self._preview = tuple(preview) if preview else None`, `self.ev = float(ev)`, `self._still_config = None`. In the configuration block, after `self._stream_info.autofocus = ...`:

```python
            self._still_config = config
            if self._preview is not None:
                binned = _binned_mode(getattr(camera, "sensor_modes", []), self._native_size)
                preview_config = camera.create_preview_configuration(
                    main={"size": self._preview, "format": _MAIN_FORMAT},
                    raw={"size": binned},
                )
                camera.configure(preview_config)
```

New helper:

```python
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
```

`read()`:

```python
        with self._request_lock:
            if full and self._autofocus == "auto" and self._stream_info.autofocus != "none":
                ...autofocus_cycle as before...
            try:
                if full and self._preview is not None:
                    request = camera.switch_mode_and_capture_request(self._still_config)
                else:
                    request = camera.capture_request()
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to capture a request: {exc}") from exc
            if full or self._preview is None:
                expected_shape = (self._native_size[1], self._native_size[0], 3)
            else:
                expected_shape = (self._preview[1], self._preview[0], 3)
            ... rest unchanged, indented under the lock ...
```

`set_ev`:

```python
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
```

Docstring addition to the module: why the still configuration is configured and validated first (the record describes the still, and `camera_configuration()` only describes what is currently configured), why one lock covers request and release (the buffer belongs to the request), and that `switch_mode_and_capture_request` returns the camera to the preview configuration itself.

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: PASS, clean

- [ ] **Step 6: Commit**

```bash
git add pifilm/capture/picamera.py pifilm/capture/camera.py tests/test_picamera.py tests/test_camera.py
git commit -m "Picamera2 preview mode with per-shot mode switch, request lock and runtime EV"
```

---

### Task 4: ST7789 display driver

**Files:**
- Create: `pifilm/display/__init__.py`, `pifilm/display/st7789.py`
- Test: `tests/test_display_st7789.py`

**Interfaces:**
- Produces: `pifilm.display.DisplayError(Exception)`; `st7789.WIDTH = 320`, `HEIGHT = 240`; `rgb565_bytes(rgb: np.ndarray) -> bytes`; `INIT_SEQUENCE`; `MADCTL = {0: 0x70, 180: 0xB0}`; `class ST7789Display(spi, dc, rst, bl, *, rotate=0, sleep=time.sleep)` with `show(image: PIL.Image.Image)`, `backlight(percent: int)`, `close()`; `open_waveshare28(rotate: int = 0) -> ST7789Display`.

- [ ] **Step 1: Write the failing tests**

`tests/test_display_st7789.py`:

```python
import numpy as np
import pytest
from PIL import Image

from pifilm.display import DisplayError
from pifilm.display.st7789 import (
    CHUNK, HEIGHT, INIT_SEQUENCE, MADCTL, WIDTH, ST7789Display, rgb565_bytes,
)


class FakePin:
    def __init__(self):
        self.level = 1
        self.history = []
        self.closed = False

    def on(self):
        self.level = 1
        self.history.append(1)

    def off(self):
        self.level = 0
        self.history.append(0)

    def close(self):
        self.closed = True


class FakeSpi:
    """Records each write together with the DC level in force, so command bytes
    (DC low) and data bytes (DC high) can be told apart afterwards."""

    def __init__(self, dc, fail=False):
        self.dc = dc
        self.writes = []      # bytes only
        self.tagged = []      # (dc_level, bytes)
        self.closed = False
        self.fail = fail

    def writebytes2(self, data):
        if self.fail:
            raise OSError(5, "spi")
        self.writes.append(bytes(data))
        self.tagged.append((self.dc.level, bytes(data)))

    def close(self):
        self.closed = True


class FakeBacklight:
    def __init__(self):
        self.value = 0.0
        self.closed = False

    def close(self):
        self.closed = True


def _display(rotate=0):
    dc, rst, bl = FakePin(), FakePin(), FakeBacklight()
    spi = FakeSpi(dc)
    disp = ST7789Display(spi, dc, rst, bl, rotate=rotate, sleep=lambda s: None)
    return disp, spi, dc, rst, bl


def test_rgb565_packs_big_endian_565():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    assert rgb565_bytes(rgb) == bytes([0xF8, 0x00, 0x07, 0xE0, 0x00, 0x1F, 0xFF, 0xFF])


def test_init_replays_the_vendor_sequence_then_sets_landscape_madctl():
    disp, spi, dc, rst, bl = _display()
    commands = [data[0] for level, data in spi.tagged if level == 0]
    vendor_commands = [cmd for cmd, _ in INIT_SEQUENCE] + [0x29]
    assert commands == vendor_commands + [0x36]  # MADCTL for rotation comes last
    data_after_madctl = spi.tagged[-1]
    assert data_after_madctl == (1, bytes([MADCTL[0]]))
    # the vendor data bytes travel with their commands, in order
    sent_data = [data for level, data in spi.tagged if level == 1]
    assert sent_data[0] == bytes([0x00]) and sent_data[1] == bytes([0x05])
    assert rst.history[:3] == [1, 0, 1]
    assert bl.value == pytest.approx(0.8)


def test_show_writes_one_frame_in_chunks():
    disp, spi, *_ = _display()
    spi.writes.clear()
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 0, 0))
    disp.show(image)
    data = b"".join(w for w in spi.writes if len(w) > 4)
    assert len(data) == WIDTH * HEIGHT * 2
    assert data[:2] == bytes([0xF8, 0x00])
    assert all(len(w) <= CHUNK for w in spi.writes)


def test_show_rejects_wrong_size():
    disp, *_ = _display()
    with pytest.raises(DisplayError, match="320x240"):
        disp.show(Image.new("RGB", (240, 320)))


def test_show_wraps_spi_errors():
    disp, spi, *_ = _display()
    spi.fail = True
    with pytest.raises(DisplayError):
        disp.show(Image.new("RGB", (WIDTH, HEIGHT)))


def test_rotate_180_uses_the_other_madctl():
    disp, spi, *_ = _display(rotate=180)
    assert spi.writes[-1] == bytes([MADCTL[180]])


def test_close_turns_the_panel_off_and_releases_everything():
    disp, spi, dc, rst, bl = _display()
    disp.close()
    commands = [w[0] for w in spi.writes[-4:]]
    assert 0x28 in commands and 0x10 in commands
    assert bl.value == 0
    assert spi.closed and dc.closed and rst.closed and bl.closed
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_display_st7789.py`
Expected: FAIL with `ModuleNotFoundError: pifilm.display`

- [ ] **Step 3: Implement**

`pifilm/display/__init__.py`:

```python
"""Optional SPI LCD viewfinder for the Pi.

Hardware libraries are imported only inside the ``open_*`` factories so that
the package imports cleanly on a Mac and in the test suite.
"""


class DisplayError(Exception):
    """The display or touch panel could not be opened or driven."""
```

`pifilm/display/st7789.py`:

```python
"""ST7789 driver for the Waveshare 2.8" Capacitive Touch LCD (V2).

Wiring (BCM GPIO → physical pin): MOSI 10 → 19, SCLK 11 → 23, CS 8 (SPI0 CE0) → 24,
DC 25 → 22, RST 27 → 13, BL 18 → 12, VCC 3.3 V → 1, GND → 6. MISO is not connected;
the panel is write-only. GPIO 5/6/12/16/20/26 belong to the X728 UPS and must not
be used.

The init sequence is the vendor's byte for byte. The frame push is not: the vendor
code converts the frame to a Python list and, in its landscape path, writes every
frame twice. Here the frame is packed to RGB565 with NumPy and written once in
``CHUNK``-byte pieces (spidev's transfer limit).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from . import DisplayError

WIDTH, HEIGHT = 320, 240
DC_PIN, RST_PIN, BL_PIN = 25, 27, 18
SPI_BUS, SPI_DEVICE, SPI_HZ = 0, 0, 40_000_000
CHUNK = 4096
MADCTL = {0: 0x70, 180: 0xB0}
BACKLIGHT_DEFAULT = 80

# (command, data bytes) — vendor ST7789_Init, then a 100 ms pause and DISPON (0x29).
INIT_SEQUENCE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (0x36, (0x00,)),
    (0x3A, (0x05,)),
    (0xB2, (0x0B, 0x0B, 0x00, 0x33, 0x35)),
    (0xB7, (0x11,)),
    (0xBB, (0x35,)),
    (0xC0, (0x2C,)),
    (0xC2, (0x01,)),
    (0xC3, (0x0D,)),
    (0xC4, (0x20,)),
    (0xC6, (0x13,)),
    (0xD0, (0xA4, 0xA1)),
    (0xD6, (0xA1,)),
    (0xE0, (0xF0, 0x06, 0x0B, 0x0A, 0x09, 0x26, 0x29, 0x33, 0x41, 0x18, 0x16, 0x15, 0x29, 0x2D)),
    (0xE1, (0xF0, 0x04, 0x08, 0x08, 0x07, 0x03, 0x28, 0x32, 0x40, 0x3B, 0x19, 0x18, 0x2A, 0x2E)),
    (0x21, ()),
    (0x11, ()),
)


def rgb565_bytes(rgb: np.ndarray) -> bytes:
    """Pack an (H, W, 3) uint8 array as big-endian RGB565, row-major."""
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    packed = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return packed.astype(">u2").tobytes()


class ST7789Display:
    def __init__(
        self, spi: Any, dc: Any, rst: Any, bl: Any, *, rotate: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rotate not in MADCTL:
            raise DisplayError(f"rotate must be one of {sorted(MADCTL)}, got {rotate}")
        self._spi, self._dc, self._rst, self._bl = spi, dc, rst, bl
        self._sleep = sleep
        self.width, self.height = WIDTH, HEIGHT
        self.backlight(BACKLIGHT_DEFAULT)
        self._reset()
        for command, data in INIT_SEQUENCE:
            self._command(command)
            if data:
                self._data(bytes(data))
        self._sleep(0.1)
        self._command(0x29)
        self._command(0x36)
        self._data(bytes([MADCTL[rotate]]))

    def _write(self, data: bytes) -> None:
        try:
            for i in range(0, len(data), CHUNK):
                self._spi.writebytes2(data[i:i + CHUNK])
        except OSError as exc:
            raise DisplayError(f"SPI write failed: {exc}") from exc

    def _command(self, command: int) -> None:
        self._dc.off()
        self._write(bytes([command]))

    def _data(self, data: bytes) -> None:
        self._dc.on()
        self._write(data)

    def _reset(self) -> None:
        self._rst.on()
        self._sleep(0.01)
        self._rst.off()
        self._sleep(0.01)
        self._rst.on()
        self._sleep(0.01)

    def _set_window(self) -> None:
        self._command(0x2A)
        self._data(bytes([0, 0, (WIDTH - 1) >> 8, (WIDTH - 1) & 0xFF]))
        self._command(0x2B)
        self._data(bytes([0, 0, (HEIGHT - 1) >> 8, (HEIGHT - 1) & 0xFF]))
        self._command(0x2C)

    def backlight(self, percent: int) -> None:
        self._bl.value = max(0, min(100, int(percent))) / 100

    def show(self, image: Image.Image) -> None:
        if image.size != (WIDTH, HEIGHT):
            raise DisplayError(f"frame must be {WIDTH}x{HEIGHT}, got {image.size[0]}x{image.size[1]}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        payload = rgb565_bytes(np.asarray(image))
        self._set_window()
        self._data(payload)

    def close(self) -> None:
        try:
            self._command(0x28)
            self._command(0x10)
        except DisplayError:
            pass
        finally:
            self.backlight(0)
            for obj in (self._spi, self._dc, self._rst, self._bl):
                close = getattr(obj, "close", None)
                if callable(close):
                    close()


def open_waveshare28(rotate: int = 0) -> ST7789Display:
    """Open the panel on SPI0 CE0 with gpiozero pins. Raises DisplayError with the cause."""
    try:
        import spidev
        from gpiozero import DigitalOutputDevice, PWMOutputDevice
    except ImportError as exc:
        raise DisplayError(
            "display libraries missing; install python3-spidev and python3-gpiozero from apt"
        ) from exc
    try:
        spi = spidev.SpiDev()
        spi.open(SPI_BUS, SPI_DEVICE)
        spi.max_speed_hz = SPI_HZ
        spi.mode = 0b00
    except PermissionError as exc:
        raise DisplayError(
            f"no permission for /dev/spidev{SPI_BUS}.{SPI_DEVICE}; add the service user to the "
            "spi group and log in again"
        ) from exc
    except OSError as exc:
        raise DisplayError(
            f"cannot open /dev/spidev{SPI_BUS}.{SPI_DEVICE}: {exc}; is dtparam=spi=on set?"
        ) from exc
    try:
        dc = DigitalOutputDevice(DC_PIN, active_high=True, initial_value=True)
        rst = DigitalOutputDevice(RST_PIN, active_high=True, initial_value=True)
        bl = PWMOutputDevice(BL_PIN, frequency=1000)
    except Exception as exc:  # gpiozero raises library-specific errors for a busy pin
        spi.close()
        raise DisplayError(
            f"cannot claim display GPIO {DC_PIN}/{RST_PIN}/{BL_PIN}: {exc}; remove any "
            "dtoverlay=fbtft line from config.txt and check the gpio group"
        ) from exc
    return ST7789Display(spi, dc, rst, bl, rotate=rotate)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_display_st7789.py && .venv/bin/ruff check pifilm/display tests/test_display_st7789.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display tests/test_display_st7789.py
git commit -m "ST7789 driver for the Waveshare 2.8\" LCD with a NumPy frame push"
```

---

### Task 5: CST3530 touch driver and tap detection

**Files:**
- Create: `pifilm/display/cst3530.py`
- Test: `tests/test_display_cst3530.py`

**Interfaces:**
- Produces: `RawPoint(x, y, strength)`, `TouchPoint(x, y, strength)`, `Tap(x, y)` (frozen dataclasses); `decode_points(buf: bytes, count: int) -> list[RawPoint]`; `to_display(point: RawPoint, rotate: int) -> TouchPoint`; `class CST3530Touch(i2c, *, rotate=0, address=0x58)` with `read() -> list[TouchPoint]`, `close()`; `class TapDetector(max_hold=0.6, max_move=20)` with `feed(points: list[TouchPoint], now: float) -> Tap | None`; `open_waveshare28_touch(rotate: int = 0) -> CST3530Touch`.

Design note (deviation from spec §3.2, recorded here): the driver polls the bus every step instead of gating on the INT pin. Capacitive controllers pulse INT per report rather than holding it, so a 10 Hz poll could miss the pulse; two short I2C transactions per step are negligible on the shared bus.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from pifilm.display.cst3530 import (
    ADDRESS, RawPoint, Tap, TapDetector, TouchPoint, CST3530Touch, decode_points, to_display,
)


def _frame(x, y, strength=40):
    # vendor layout: point i at buf[4+5i .. 8+5i]: x low, y low, strength, hi nibbles
    return bytes([x & 0xFF, y & 0xFF, strength, ((y >> 4) & 0xF0) | ((x >> 8) & 0x0F), 0x00])


def _buf(points):
    buf = bytearray(bytes([0, 0, 0, len(points)]) + b"".join(_frame(*p) for p in points))
    buf[8] = 0xA0  # the vendor treats buf[8] & 0xF0 == 0 as "no report"
    return bytes(buf)


def test_decode_single_point_matches_vendor_bit_layout():
    buf = _buf([(0x12C, 0x0F0)])  # x=300, y=240
    assert decode_points(buf, 1) == [RawPoint(300, 240, 40)]


def test_decode_two_points():
    buf = _buf([(10, 20), (200, 300)])
    assert decode_points(buf, 2) == [RawPoint(10, 20, 40), RawPoint(200, 300, 40)]


def test_to_display_rotations():
    raw = RawPoint(10, 50, 1)
    assert to_display(raw, 0) == TouchPoint(50, 229, 1)
    assert to_display(raw, 180) == TouchPoint(269, 10, 1)


class FakeI2C:
    def __init__(self, reports):
        self.reports = list(reports)  # each: (first9: bytes, extra: bytes)
        self.writes = []
        self.pending = b""
        self.closed = False

    def write_i2c_block_data(self, address, first, rest):
        assert address == ADDRESS
        reg = (first << 24) | (rest[0] << 16) | (rest[1] << 8) | rest[2]
        self.writes.append(reg)
        if reg == 0xD0070000:
            first9, extra = self.reports.pop(0) if self.reports else (bytes(9), b"")
            self.pending = first9
            self._extra = extra
        elif reg == 0xD0070900:
            self.pending = self._extra

    def read_byte(self, address):
        b, self.pending = self.pending[0], self.pending[1:]
        return b

    def close(self):
        self.closed = True


def test_read_returns_no_points_and_ends_the_read_on_an_empty_report():
    i2c = FakeI2C([(bytes(9), b"")])
    touch = CST3530Touch(i2c)
    assert touch.read() == []
    assert i2c.writes == [0xD0070000, 0xD00002AB]


def test_read_decodes_and_rotates_a_two_finger_report():
    buf = _buf([(10, 20), (200, 300)])
    i2c = FakeI2C([(buf[:9], buf[9:])])
    touch = CST3530Touch(i2c, rotate=0)
    assert touch.read() == [TouchPoint(20, 229, 40), TouchPoint(300, 39, 40)]
    assert i2c.writes == [0xD0070000, 0xD0070900, 0xD00002AB]


def test_close_releases_the_bus():
    i2c = FakeI2C([])
    CST3530Touch(i2c).close()
    assert i2c.closed


def test_tap_detector_emits_on_release_within_limits():
    det = TapDetector()
    assert det.feed([TouchPoint(100, 100, 5)], 0.0) is None
    assert det.feed([TouchPoint(104, 101, 5)], 0.1) is None
    assert det.feed([], 0.2) == Tap(100, 100)


def test_tap_detector_ignores_long_holds_and_drags():
    det = TapDetector()
    det.feed([TouchPoint(100, 100, 5)], 0.0)
    assert det.feed([], 1.0) is None
    det.feed([TouchPoint(100, 100, 5)], 2.0)
    det.feed([TouchPoint(150, 100, 5)], 2.1)
    assert det.feed([], 2.2) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_display_cst3530.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""CST3530 capacitive touch controller (Waveshare 2.8" V2) over I2C1.

Register protocol as in the vendor ``Touch_CST3530.py``: 32-bit register
addresses, the first byte sent as the SMBus "command" and the remaining three as
data; a report is 9 bytes at ``REG_DATA`` holding the point count in ``buf[3] &
0x0F`` and the first point, further points are 5 bytes each at ``REG_NEXT``, and
every read ends with a write to ``REG_END_READ``. Raw coordinates are in the
panel's 240x320 portrait space; ``to_display`` maps them to the 320x240 landscape
frame the viewfinder draws.

The bus (I2C1) is shared with the X728's fuel gauge (0x36) and RTC (0x68). Each
call here is one kernel ioctl and the controller's read pointer is not affected
by transactions to other addresses, so no cross-module locking is needed. The
driver polls instead of using the INT line: capacitive controllers pulse INT per
report rather than holding it, and a 10 Hz poll could miss the pulse.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from . import DisplayError

ADDRESS = 0x58
REG_DATA = 0xD0070000
REG_NEXT = 0xD0070900
REG_END_READ = 0xD00002AB
MAX_POINTS = 5
TP_RST_PIN = 17
PORTRAIT_W, PORTRAIT_H = 240, 320


@dataclass(frozen=True)
class RawPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


def decode_points(buf: bytes, count: int) -> list[RawPoint]:
    points = []
    for i in range(count):
        base = 4 + 5 * i
        hi = buf[base + 3]
        x = ((hi & 0x0F) << 8) | buf[base]
        y = ((hi & 0xF0) << 4) | buf[base + 1]
        points.append(RawPoint(x, y, buf[base + 2]))
    return points


def to_display(point: RawPoint, rotate: int) -> TouchPoint:
    if rotate == 0:
        return TouchPoint(point.y, PORTRAIT_W - 1 - point.x, point.strength)
    if rotate == 180:
        return TouchPoint(PORTRAIT_H - 1 - point.y, point.x, point.strength)
    raise DisplayError(f"rotate must be 0 or 180, got {rotate}")


def _split(reg: int) -> tuple[int, list[int]]:
    return (reg >> 24) & 0xFF, [(reg >> 16) & 0xFF, (reg >> 8) & 0xFF, reg & 0xFF]


class CST3530Touch:
    def __init__(self, i2c: Any, *, rotate: int = 0, address: int = ADDRESS) -> None:
        self._i2c = i2c
        self._rotate = rotate
        self._address = address

    def _read(self, reg: int, count: int) -> bytes:
        first, rest = _split(reg)
        self._i2c.write_i2c_block_data(self._address, first, rest)
        return bytes(self._i2c.read_byte(self._address) for _ in range(count))

    def _write(self, reg: int) -> None:
        first, rest = _split(reg)
        self._i2c.write_i2c_block_data(self._address, first, rest)

    def read(self) -> list[TouchPoint]:
        try:
            buf = self._read(REG_DATA, 9)
            count = buf[3] & 0x0F
            if count == 0 or count > MAX_POINTS or (buf[8] & 0xF0) == 0:
                self._write(REG_END_READ)
                return []
            extra = self._read(REG_NEXT, (count - 1) * 5) if count > 1 else b""
            self._write(REG_END_READ)
        except OSError as exc:
            raise DisplayError(f"touch I2C read failed: {exc}") from exc
        return [to_display(p, self._rotate) for p in decode_points(buf + extra, count)]

    def close(self) -> None:
        close = getattr(self._i2c, "close", None)
        if callable(close):
            close()


class TapDetector:
    """Turn per-step point lists into taps: down then up within ``max_hold`` s,
    moving less than ``max_move`` px. Holds and drags produce nothing."""

    def __init__(self, max_hold: float = 0.6, max_move: float = 20.0) -> None:
        self._max_hold, self._max_move = max_hold, max_move
        self._down: tuple[TouchPoint, float] | None = None
        self._last: TouchPoint | None = None

    def feed(self, points: list[TouchPoint], now: float) -> Tap | None:
        if points:
            if self._down is None:
                self._down = (points[0], now)
            self._last = points[0]
            return None
        if self._down is None:
            return None
        first, t0 = self._down
        last = self._last or first
        self._down, self._last = None, None
        moved = math.hypot(last.x - first.x, last.y - first.y)
        if now - t0 <= self._max_hold and moved < self._max_move:
            return Tap(first.x, first.y)
        return None


def open_waveshare28_touch(rotate: int = 0, sleep=time.sleep) -> CST3530Touch:
    try:
        from gpiozero import DigitalOutputDevice
        from smbus2 import SMBus
    except ImportError as exc:
        raise DisplayError(
            "touch libraries missing; install python3-smbus2 and python3-gpiozero from apt"
        ) from exc
    try:
        rst = DigitalOutputDevice(TP_RST_PIN, active_high=True, initial_value=True)
        rst.off()
        sleep(0.1)
        rst.on()
        sleep(0.5)
        rst.close()
    except Exception as exc:
        raise DisplayError(f"cannot reset the touch controller on GPIO {TP_RST_PIN}: {exc}") from exc
    try:
        bus = SMBus(1)
    except PermissionError as exc:
        raise DisplayError("no permission for /dev/i2c-1; add the service user to the i2c group") from exc
    except OSError as exc:
        raise DisplayError(f"cannot open /dev/i2c-1: {exc}; is dtparam=i2c_arm=on set?") from exc
    return CST3530Touch(bus, rotate=rotate)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_display_cst3530.py && .venv/bin/ruff check pifilm/display tests/test_display_cst3530.py`
Expected: PASS, clean

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/cst3530.py tests/test_display_cst3530.py
git commit -m "CST3530 touch driver with rotation mapping and tap detection"
```

---

### Task 6: Exposure meter

**Files:**
- Create: `pifilm/display/meter.py`
- Test: `tests/test_display_meter.py`

**Interfaces:**
- Produces: `MeterReading` (frozen dataclass: `shutter: str | None`, `iso: int | None`, `ev_comp: float`, `lux: float | None`, `deviation_ev: float`, `clip_pct: float`, `battery_percent: int | None`, `external_power: bool | None`); `format_shutter(exposure_us: float) -> str`; `format_ev(ev: float) -> str`; `compute_reading(metadata: dict | None, preview_rgb: np.ndarray, ev_comp: float, power) -> MeterReading` where `power` has `.percent` and `.external_power` or is `None`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from pifilm.display.meter import compute_reading, format_ev, format_shutter


def _grey(value, shape=(48, 64, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_format_shutter():
    assert format_shutter(4000) == "1/250"
    assert format_shutter(33333) == "1/30"
    assert format_shutter(1_300_000) == "1.3s"
    assert format_shutter(1_000_000) == "1.0s"


def test_format_ev():
    assert format_ev(0.0) == "0"
    assert format_ev(1 / 3) == "+0.3"
    assert format_ev(-4 / 3) == "-1.3"


def test_reading_from_metadata():
    meta = {"ExposureTime": 4000, "AnalogueGain": 2.0, "DigitalGain": 1.5, "Lux": 640.4}
    r = compute_reading(meta, _grey(118), 1 / 3, None)
    assert r.shutter == "1/250"
    assert r.iso == 300
    assert r.lux == pytest.approx(640.4)
    assert r.ev_comp == pytest.approx(1 / 3)
    assert r.battery_percent is None and r.external_power is None


def test_mid_grey_frame_reads_zero_deviation():
    # sRGB 118 ≈ 18 % linear reflectance
    r = compute_reading({}, _grey(118), 0.0, None)
    assert r.deviation_ev == pytest.approx(0.0, abs=0.05)
    assert r.clip_pct == 0.0


def test_dark_and_bright_frames_swing_and_clamp():
    assert compute_reading({}, _grey(0), 0.0, None).deviation_ev == -3.0
    assert compute_reading({}, _grey(255), 0.0, None).deviation_ev == pytest.approx(2.47, abs=0.05)


def test_clip_pct_counts_any_channel_at_ceiling():
    frame = _grey(50)
    frame[:24, :, 1] = 254
    assert compute_reading({}, frame, 0.0, None).clip_pct == pytest.approx(50.0)


def test_missing_metadata_gives_none_fields_without_raising():
    r = compute_reading(None, _grey(118), 0.0, None)
    assert r.shutter is None and r.iso is None and r.lux is None


def test_battery_fields_come_from_power():
    class P:
        percent = 87
        external_power = True

    r = compute_reading({}, _grey(118), 0.0, P())
    assert (r.battery_percent, r.external_power) == (87, True)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_display_meter.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Light-meter readout for the viewfinder, computed from a preview frame and its
libcamera metadata. Pure functions; nothing here touches hardware.

The needle (``deviation_ev``) is the scene's mean linear luminance against an
18 % mid-grey, in stops. With auto-exposure converged it rests near zero and only
swings when AE has run out of range or EV compensation is dialled in, which is
exactly what a camera's meter shows in auto mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..color import luminance, srgb_to_linear

MID_GREY = 0.18
DEVIATION_CLAMP = 3.0
CLIP_LEVEL = 254


@dataclass(frozen=True)
class MeterReading:
    shutter: str | None
    iso: int | None
    ev_comp: float
    lux: float | None
    deviation_ev: float
    clip_pct: float
    battery_percent: int | None
    external_power: bool | None


def format_shutter(exposure_us: float) -> str:
    seconds = exposure_us / 1_000_000
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"1/{round(1.0 / seconds)}"


def format_ev(ev: float) -> str:
    if abs(ev) < 0.05:
        return "0"
    return f"{ev:+.1f}"


def _number(metadata: dict, key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def compute_reading(
    metadata: dict | None, preview_rgb: np.ndarray, ev_comp: float, power: Any,
) -> MeterReading:
    meta = metadata or {}
    exposure = _number(meta, "ExposureTime")
    shutter = format_shutter(exposure) if exposure and exposure > 0 else None
    analogue = _number(meta, "AnalogueGain")
    digital = _number(meta, "DigitalGain")
    iso = None
    if analogue is not None and analogue > 0:
        iso = int(round(analogue * (digital if digital and digital > 0 else 1.0) * 100 / 10) * 10)
    lux = _number(meta, "Lux")

    small = preview_rgb[::4, ::4].astype(np.float32) / 255.0
    mean_luma = float(np.mean(luminance(srgb_to_linear(small))))
    deviation = math.log2(max(mean_luma, 1e-4) / MID_GREY)
    deviation = max(-DEVIATION_CLAMP, min(DEVIATION_CLAMP, deviation))
    clip_pct = float(np.mean(np.any(preview_rgb >= CLIP_LEVEL, axis=-1)) * 100.0)

    battery = int(power.percent) if power is not None else None
    external = bool(power.external_power) if power is not None else None
    return MeterReading(shutter, iso, float(ev_comp), lux, deviation, clip_pct, battery, external)
```

Check `pifilm.color.luminance` takes linear RGB and returns BT.709 Y; if its signature differs, adapt the call but keep the maths.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_display_meter.py && .venv/bin/ruff check pifilm/display/meter.py tests/test_display_meter.py`
Expected: PASS, clean. If the `_grey(255)` expectation is off by more than 0.05, compute `log2(1/0.18)` = 2.474 and correct the test to that constant, not the code.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/meter.py tests/test_display_meter.py
git commit -m "Exposure meter readout from preview frame and metadata"
```

---

### Task 7: Renderer and hit-testing

**Files:**
- Create: `pifilm/display/ui.py`
- Test: `tests/test_display_ui.py`

**Interfaces:**
- Consumes: `MeterReading` (Task 6).
- Produces: `Action` enum (`NONE`, `SHUTTER`, `EV_MINUS`, `EV_PLUS`); `render_live(frame_rgb: np.ndarray, reading: MeterReading, busy: bool) -> Image`; `render_review(graded_rgb: np.ndarray, caption: str) -> Image`; `render_message(title: str, detail: str) -> Image`; `hit(x: int, y: int) -> Action`; layout constants `BAR_TOP = 204`, `SHUTTER_CENTRE = (288, 102)`, `SHUTTER_RADIUS = 28`, `EV_BUTTON_W = 40`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from PIL import Image

from pifilm.display.meter import MeterReading
from pifilm.display.ui import (
    BAR_TOP, SHUTTER_CENTRE, SHUTTER_RADIUS, Action, hit, render_live, render_message,
    render_review,
)


def _reading(**over):
    base = dict(shutter="1/250", iso=100, ev_comp=0.0, lux=640.0, deviation_ev=0.0,
                clip_pct=3.2, battery_percent=87, external_power=False)
    base.update(over)
    return MeterReading(**base)


def test_render_live_is_320x240_rgb_with_a_darker_bar():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(), busy=False)
    assert isinstance(img, Image.Image) and img.size == (320, 240) and img.mode == "RGB"
    above = img.getpixel((160, BAR_TOP - 10))
    inside = img.getpixel((160, BAR_TOP + 4))
    assert sum(inside) < sum(above)


def test_render_live_letterboxes_a_16_9_frame():
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(), busy=False)
    assert img.getpixel((5, 5)) == (0, 0, 0)  # top band
    assert img.getpixel((160, 120)) != (0, 0, 0)


def test_busy_marker_is_drawn_only_when_busy():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    idle = render_live(frame, _reading(), busy=False).getpixel((10, 10))
    busy = render_live(frame, _reading(), busy=True).getpixel((10, 10))
    assert idle != busy and busy[0] > 200


def test_render_live_handles_missing_fields():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    img = render_live(frame, _reading(shutter=None, iso=None, lux=None,
                                       battery_percent=None, external_power=None), busy=False)
    assert img.size == (320, 240)


def test_render_review_and_message_sizes():
    graded = np.full((3040, 4056, 3), 90, dtype=np.uint8)[::8, ::8]
    assert render_review(graded, "1/250  ISO 100  0").size == (320, 240)
    assert render_message("Capture failed", "camera_error").size == (320, 240)


def test_hit_regions():
    cx, cy = SHUTTER_CENTRE
    assert hit(cx, cy) == Action.SHUTTER
    assert hit(cx + SHUTTER_RADIUS + 4, cy) == Action.SHUTTER   # 6 px margin
    assert hit(cx + SHUTTER_RADIUS + 20, cy) == Action.NONE
    assert hit(10, BAR_TOP + 10) == Action.EV_MINUS
    assert hit(310, BAR_TOP + 10) == Action.EV_PLUS
    assert hit(160, BAR_TOP + 10) == Action.NONE
    assert hit(160, 100) == Action.NONE
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_display_ui.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Pillow rendering for the 320x240 viewfinder and its hit regions. Pure.

Layout: the preview fills the screen letterboxed; a translucent bar along the
bottom carries the meter; a round shutter button sits at the right edge; EV
buttons occupy the bar's ends. ``hit`` mirrors the drawn regions plus a 6 px
margin so the two cannot drift apart: both read the same constants.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meter import MeterReading, format_ev

WIDTH, HEIGHT = 320, 240
BAR_TOP = 204
BAR_ALPHA = 153  # 60 %
SHUTTER_CENTRE = (288, 102)
SHUTTER_RADIUS = 28
EV_BUTTON_W = 40
HIT_MARGIN = 6
BUSY_CENTRE, BUSY_RADIUS = (10, 10), 5
AMBER = (255, 176, 0)
NEEDLE_X0, NEEDLE_X1, NEEDLE_Y = 200, 272, 222
NEEDLE_RANGE = 3.0


class Action(Enum):
    NONE = "none"
    SHUTTER = "shutter"
    EV_MINUS = "ev_minus"
    EV_PLUS = "ev_plus"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no bundled scalable default
        return ImageFont.load_default()


def _letterbox(rgb: np.ndarray, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    src = Image.fromarray(np.ascontiguousarray(rgb))
    scale = min(size[0] / src.width, size[1] / src.height)
    fitted = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))),
                        Image.BILINEAR)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _text_or_dash(value: str | None) -> str:
    return value if value else "—"


def render_live(frame_rgb: np.ndarray, reading: MeterReading, busy: bool) -> Image.Image:
    base = _letterbox(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, BAR_TOP, WIDTH, HEIGHT), fill=(0, 0, 0, BAR_ALPHA))
    # EV buttons
    draw.rectangle((0, BAR_TOP, EV_BUTTON_W, HEIGHT), outline=(255, 255, 255, 200))
    draw.rectangle((WIDTH - EV_BUTTON_W, BAR_TOP, WIDTH, HEIGHT), outline=(255, 255, 255, 200))
    big = _font(18)
    draw.text((EV_BUTTON_W // 2, BAR_TOP + 18), "−", fill=(255, 255, 255, 255), font=big, anchor="mm")
    draw.text((WIDTH - EV_BUTTON_W // 2, BAR_TOP + 18), "+", fill=(255, 255, 255, 255), font=big, anchor="mm")
    # readout
    font = _font(14)
    small = _font(11)
    iso = f"ISO {reading.iso}" if reading.iso is not None else "ISO —"
    line1 = f"{_text_or_dash(reading.shutter)}  {iso}  EV {format_ev(reading.ev_comp)}"
    lux = f"{reading.lux:.0f} lx" if reading.lux is not None else "— lx"
    line2 = f"{lux}   clip {reading.clip_pct:.0f}%"
    draw.text((EV_BUTTON_W + 6, BAR_TOP + 3), line1, fill=(255, 255, 255, 255), font=font)
    draw.text((EV_BUTTON_W + 6, BAR_TOP + 21), line2, fill=(220, 220, 220, 255), font=small)
    # needle: scale −3..+3 stops
    draw.line((NEEDLE_X0, NEEDLE_Y, NEEDLE_X1, NEEDLE_Y), fill=(255, 255, 255, 200), width=1)
    for k in range(-3, 4):
        x = _needle_x(float(k))
        draw.line((x, NEEDLE_Y - (4 if k == 0 else 2), x, NEEDLE_Y + (4 if k == 0 else 2)),
                  fill=(255, 255, 255, 200))
    nx = _needle_x(reading.deviation_ev)
    draw.polygon([(nx, NEEDLE_Y - 8), (nx - 5, NEEDLE_Y - 15), (nx + 5, NEEDLE_Y - 15)],
                 fill=AMBER + (255,))
    # shutter button
    cx, cy = SHUTTER_CENTRE
    draw.ellipse((cx - SHUTTER_RADIUS, cy - SHUTTER_RADIUS, cx + SHUTTER_RADIUS, cy + SHUTTER_RADIUS),
                 fill=(255, 255, 255, 60), outline=(255, 255, 255, 255), width=2)
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(255, 255, 255, 180))
    # busy marker
    if busy:
        bx, by = BUSY_CENTRE
        draw.ellipse((bx - BUSY_RADIUS, by - BUSY_RADIUS, bx + BUSY_RADIUS, by + BUSY_RADIUS),
                     fill=AMBER + (255,))
    # battery badge
    if reading.battery_percent is not None:
        label = f"{'AC ' if reading.external_power else ''}{reading.battery_percent}%"
        w = draw.textlength(label, font=small)
        draw.rounded_rectangle((WIDTH - w - 14, 4, WIDTH - 4, 20), radius=4, fill=(0, 0, 0, BAR_ALPHA))
        draw.text((WIDTH - w - 9, 6), label, fill=(255, 255, 255, 255), font=small)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _needle_x(deviation: float) -> int:
    t = (max(-NEEDLE_RANGE, min(NEEDLE_RANGE, deviation)) + NEEDLE_RANGE) / (2 * NEEDLE_RANGE)
    return int(round(NEEDLE_X0 + t * (NEEDLE_X1 - NEEDLE_X0)))


def render_review(graded_rgb: np.ndarray, caption: str) -> Image.Image:
    base = _letterbox(graded_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, HEIGHT - 22, WIDTH, HEIGHT), fill=(0, 0, 0, BAR_ALPHA))
    draw.text((6, HEIGHT - 19), caption, fill=(255, 255, 255, 255), font=_font(12))
    hint = "tap to continue"
    w = draw.textlength(hint, font=_font(11))
    draw.text((WIDTH - w - 6, HEIGHT - 18), hint, fill=(200, 200, 200, 255), font=_font(11))
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_message(title: str, detail: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, HEIGHT // 2 - 16), title, fill=(255, 255, 255), font=_font(18), anchor="mm")
    draw.text((WIDTH // 2, HEIGHT // 2 + 14), detail[:60], fill=(200, 200, 200), font=_font(12), anchor="mm")
    return img


def hit(x: int, y: int) -> Action:
    cx, cy = SHUTTER_CENTRE
    if math.hypot(x - cx, y - cy) <= SHUTTER_RADIUS + HIT_MARGIN:
        return Action.SHUTTER
    if y >= BAR_TOP - HIT_MARGIN:
        if x <= EV_BUTTON_W + HIT_MARGIN:
            return Action.EV_MINUS
        if x >= WIDTH - EV_BUTTON_W - HIT_MARGIN:
            return Action.EV_PLUS
    return Action.NONE
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_display_ui.py && .venv/bin/ruff check pifilm/display/ui.py tests/test_display_ui.py`
Expected: PASS, clean. If `anchor="mm"` or `textlength` is unavailable for the bitmap fallback font, guard with `try/except` and skip anchoring; the target is Pillow ≥ 10.1 where both work.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/ui.py tests/test_display_ui.py
git commit -m "Viewfinder renderer: live frame with meter bar, review, messages, hit regions"
```

---

### Task 8: Viewfinder loop

**Files:**
- Create: `pifilm/display/viewfinder.py`
- Test: `tests/test_viewfinder.py`

**Interfaces:**
- Consumes: `CaptureController` (+ Task 1 fields), `FakeCamera.set_ev`/`ev` (Task 3), `TapDetector`, `Tap`, `TouchPoint` (Task 5), `compute_reading`/`format_shutter`/`format_ev` (Task 6), `render_*`/`hit`/`Action` (Task 7), `pifilm.imageio.load_rgb`.
- Produces: `class ViewfinderLoop(camera, controller, display, touch, *, clock=time, power_snapshot=None, frame_period=0.1, review_timeout=30.0, max_display_failures=5, log=print, touch_debug=False)` with `step() -> None`, `run(stop: threading.Event) -> None`, `state: str` (`"LIVE"`/`"REVIEW"`), `ev_comp: float`; `EV_STEP = 1/3`, `EV_LIMIT = 2.0`.

- [ ] **Step 1: Write the failing tests**

```python
import threading
import time

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession
from pifilm.capture.camera import CameraError, FakeCamera, synthetic_frame
from pifilm.capture.controller import CaptureController
from pifilm.display import DisplayError
from pifilm.display.cst3530 import TouchPoint
from pifilm.display.ui import BAR_TOP, SHUTTER_CENTRE
from pifilm.display.viewfinder import EV_STEP, ViewfinderLoop
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline


class FakeDisplay:
    def __init__(self, fail_after=None):
        self.images = []
        self.fail_after = fail_after
        self.closed = False

    def show(self, image):
        if self.fail_after is not None and len(self.images) >= self.fail_after:
            raise DisplayError("spi")
        self.images.append(image)

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


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def controller(tmp_path):
    art = tmp_path / "art"
    write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    camera = FakeCamera([synthetic_frame(48, 64)], metadata={"ExposureTime": 4000,
                                                              "AnalogueGain": 1.0, "Lux": 500.0})
    session = CaptureSession(camera, Pipeline(Artifacts.load(art)), tmp_path / "shots",
                             seed_rng=np.random.default_rng(0))
    ctl = CaptureController(session)
    yield camera, ctl
    ctl.close()


def _loop(camera, ctl, touch=None, display=None, clock=None, **kw):
    touch = touch or ScriptedTouch()
    display = display or FakeDisplay()
    clock = clock or FakeClock()
    loop = ViewfinderLoop(camera, ctl, display, touch, clock=clock, log=lambda *a: None, **kw)
    return loop, touch, display, clock


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


def test_shutter_tap_submits_through_the_controller_and_reviews_the_result(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    touch.tap(*SHUTTER_CENTRE)
    loop.step(); loop.step()  # down, up → tap → submit
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    assert ctl.snapshot().last_finished_job.state == "complete"


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
    loop.step(); loop.step()
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
    loop.step(); loop.step()
    assert loop.ev_comp == pytest.approx(2.0 - EV_STEP)


def test_failed_job_shows_a_message_in_review(tmp_path):
    class Boom:
        camera = None

        def capture(self):
            raise CameraError("lens cap")

    ctl = CaptureController(Boom())
    try:
        camera = FakeCamera([synthetic_frame(48, 64)])
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("x")
        _until(lambda: ctl.snapshot().finished_count == 1)
        loop.step()
        assert loop.state == "REVIEW"
    finally:
        ctl.close()


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""The viewfinder state machine: LIVE ⇄ REVIEW.

Every shot goes through the shared ``CaptureController`` exactly like a Stick
request, so the LCD is one more trigger and one more screen, never a second
owner of the camera. A finished job is noticed through the controller's
``finished_count``, which is how a Stick shot appears here without the
controller knowing about displays.

Display and touch objects are duck-typed (``show``/``close``, ``read``/``close``)
and the clock is injectable, so the loop is tested without hardware.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from PIL import Image

from ..capture.camera import CameraError
from ..imageio import load_rgb
from . import DisplayError
from .cst3530 import Tap, TapDetector
from .meter import compute_reading, format_ev, format_shutter
from .ui import Action, hit, render_live, render_message, render_review

EV_STEP = 1.0 / 3.0
EV_LIMIT = 2.0
RATE_LOG_INTERVAL = 10.0


class ViewfinderLoop:
    def __init__(
        self, camera: Any, controller: Any, display: Any, touch: Any, *,
        clock: Any = time, power_snapshot: Callable[[], Any] | None = None,
        frame_period: float = 0.1, review_timeout: float = 30.0,
        max_display_failures: int = 5, log: Callable[..., None] = print,
        touch_debug: bool = False,
    ) -> None:
        self._camera, self._controller = camera, controller
        self._display, self._touch = display, touch
        self._clock, self._power = clock, power_snapshot
        self._frame_period, self._review_timeout = frame_period, review_timeout
        self._max_failures, self._log, self._touch_debug = max_display_failures, log, touch_debug
        self._taps = TapDetector()
        self.state = "LIVE"
        self.ev_comp = float(getattr(camera, "ev", 0.0))
        self._seen_finished = controller.snapshot().finished_count
        self._review_since = 0.0
        self._display_failures = 0
        self._frames = 0
        self._rate_since = clock.monotonic()

    # -- plumbing -------------------------------------------------------------

    def _poll_tap(self) -> Tap | None:
        points = self._touch.read()
        if self._touch_debug and points:
            self._log(f"touch: {[(p.x, p.y) for p in points]}")
        return self._taps.feed(points, self._clock.monotonic())

    def _show(self, image: Image.Image) -> None:
        try:
            self._display.show(image)
        except DisplayError as exc:
            self._display_failures += 1
            self._log(f"display: {exc} ({self._display_failures}/{self._max_failures})")
            if self._display_failures >= self._max_failures:
                raise
            return
        self._display_failures = 0

    def _set_ev(self, value: float) -> None:
        value = max(-EV_LIMIT, min(EV_LIMIT, round(value / EV_STEP) * EV_STEP))
        try:
            self._camera.set_ev(value)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            return
        self.ev_comp = value

    def _review_image(self, job: Any) -> Image.Image:
        if job.state != "complete" or job.result is None:
            return render_message("Capture failed", job.error_message or job.error_code or "")
        rgb, _ = load_rgb(job.result.pifilm)
        meta = job.result.record.get("camera_metadata") or {}
        parts = []
        if "ExposureTime" in meta:
            parts.append(format_shutter(float(meta["ExposureTime"])))
        if "AnalogueGain" in meta:
            parts.append(f"ISO {int(round(float(meta['AnalogueGain']) * 100 / 10) * 10)}")
        parts.append(f"EV {format_ev(self.ev_comp)}")
        return render_review(rgb, "  ".join(parts))

    # -- states -----------------------------------------------------------------

    def step(self) -> None:
        tap = self._poll_tap()
        snap = self._controller.snapshot()
        if snap.finished_count != self._seen_finished and snap.last_finished_job is not None:
            self._seen_finished = snap.finished_count
            self.state = "REVIEW"
            self._review_since = self._clock.monotonic()
            self._show(self._review_image(snap.last_finished_job))
            return
        if self.state == "REVIEW":
            if tap is not None or self._clock.monotonic() - self._review_since >= self._review_timeout:
                self.state = "LIVE"
            return
        if tap is not None:
            action = hit(tap.x, tap.y)
            if action is Action.SHUTTER:
                self._controller.submit(str(uuid.uuid4()))
            elif action is Action.EV_MINUS:
                self._set_ev(self.ev_comp - EV_STEP)
            elif action is Action.EV_PLUS:
                self._set_ev(self.ev_comp + EV_STEP)
        try:
            frame = self._camera.read(full=False)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            return
        power = self._power() if self._power is not None else None
        reading = compute_reading(frame.metadata, frame.rgb, self.ev_comp, power)
        self._show(render_live(frame.rgb, reading, busy=snap.active_job is not None))
        self._frames += 1
        now = self._clock.monotonic()
        if now - self._rate_since >= RATE_LOG_INTERVAL:
            self._log(f"viewfinder: {self._frames / (now - self._rate_since):.1f} fps")
            self._frames, self._rate_since = 0, now

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            started = self._clock.monotonic()
            self.step()
            elapsed = self._clock.monotonic() - started
            self._clock.sleep(max(0.0, self._frame_period - elapsed))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py && .venv/bin/ruff check pifilm/display tests/test_viewfinder.py`
Expected: PASS, clean. In `test_run_stops_on_event_and_paces_frames` the fake clock does not advance inside `step`, so each iteration sleeps the full period: 3 × 0.1 s.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/viewfinder.py tests/test_viewfinder.py
git commit -m "Viewfinder loop: live view, shutter and EV taps, review, Stick shots"
```

---

### Task 9: CLI wiring, fake display, service unit

**Files:**
- Create: `pifilm/display/fake.py`
- Modify: `pifilm/capture/app.py` (flags after `--ups`; `_reject_picamera2_only_flags`; `main` wiring), `pyproject.toml` (`Pillow>=10.1`), `deploy/pifilm-capture.service.example`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `ViewfinderLoop`, `open_waveshare28`, `open_waveshare28_touch`, `DisplayError`, `Picamera2Camera(preview=...)`.
- Produces: `FileDisplay(path)` (`show` saves PNG, `close` no-op), `NoTouch` (`read` → `[]`); flags `--display {none,waveshare28,fake}`, `--display-rotate {0,180}`, `--touch-debug`; `PREVIEW_SIZE = (640, 480)`; `_open_display(args, out_root) -> tuple[display, touch] | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app.py` (the `camera_cli` fixture exists; `_CameraDouble` too):

```python
def test_display_flag_is_rejected_on_v4l2(camera_cli, capsys):
    with pytest.raises(SystemExit):
        camera_cli.main(["--camera", "v4l2", "--display", "waveshare28"])
    assert "--display" in capsys.readouterr().err


def test_display_requests_preview_mode_from_picamera2(camera_cli, monkeypatch):
    opened = {}

    def open_picamera(tuning_file, **kwargs):
        opened.update(kwargs)
        return _CameraDouble()

    monkeypatch.setattr(camera_cli, "Picamera2Camera", open_picamera, raising=False)
    monkeypatch.setattr(camera_cli, "_open_display", lambda args, out: None)
    assert camera_cli.main(["--camera", "picamera2", "--display", "waveshare28", "--no-preview"]) == 0
    assert opened["preview"] == (640, 480)


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

    monkeypatch.setattr(app.threading, "Event", StopSoon)
    out = tmp_path / "shots"
    assert app.main(["--fake", "--display", "fake", "--no-preview", "--out", str(out)]) == 0
    assert (out / "viewfinder-last.png").exists()
```

If `_CameraDouble` lacks `set_ev`, add `def set_ev(self, value): self.ev = value` and `ev = 0.0` to it.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest -q tests/test_app.py -k display`
Expected: FAIL (`unrecognized arguments: --display`)

- [ ] **Step 3: Implement**

`pifilm/display/fake.py`:

```python
"""Hardware-free stand-ins for ``--display fake``: frames go to a PNG, no touch."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class FileDisplay:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def show(self, image: Image.Image) -> None:
        image.save(self.path)

    def close(self) -> None:
        return None


class NoTouch:
    def read(self) -> list:
        return []

    def close(self) -> None:
        return None
```

`pifilm/capture/app.py`:

- imports: `import signal`, `import threading`, `from ..display import DisplayError`, `from ..display.viewfinder import ViewfinderLoop`.
- `PREVIEW_SIZE = (640, 480)` near the other constants.
- flags after `--ups`:

```python
    parser.add_argument(
        "--display", choices=("none", "waveshare28", "fake"), default="none",
        help="LCD viewfinder with exposure meter (Picamera2 or --fake only); "
             "'fake' writes OUT/viewfinder-last.png instead of driving SPI",
    )
    parser.add_argument("--display-rotate", type=int, choices=(0, 180), default=0,
                        help="rotate the LCD image and touch mapping by 180 degrees")
    parser.add_argument("--touch-debug", action="store_true",
                        help="print raw and mapped touch coordinates (orientation check)")
```

- in `_reject_picamera2_only_flags` add `("--display", args.display != "none")` — but the `--fake` call site must not reject it: give the helper a keyword `allow_display: bool = False` and pass `allow_display=True` from the `--fake` branch. Update the docstring list.
- `Picamera2Camera(...)` call gains `preview=PREVIEW_SIZE if args.display != "none" else None`.
- helper:

```python
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
```

- wiring in `main`, after `session = CaptureSession(...)` and before the remote block: create the controller when a display is requested:

```python
    controller: CaptureController | None = None
    remote: RemoteCaptureServer | None = None
    display_pair = None
    if args.display != "none":
        try:
            display_pair = _open_display(args, args.out)
        except DisplayError as exc:
            print(f"warning: display unavailable ({exc}); continuing without it", file=sys.stderr)
    if display_pair is not None:
        controller = CaptureController(session)
```

In the remote block use `controller = controller or CaptureController(session)`. Then, after the remote server is started (and in the non-remote path too), before any OpenCV/terminal loop:

```python
        if display_pair is not None:
            assert controller is not None
            return _run_viewfinder(camera, controller, display_pair, power, args)
```

with:

```python
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
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        pass
    except DisplayError as exc:
        print(f"error: display failed repeatedly: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
        touch.close()
        display.close()
    return 0
```

Place the viewfinder branch so that it runs whether or not `--remote-listen` is given: simplest is one `if display_pair is not None:` block right after the remote server has started (inside the `if args.remote_listen:` branch, before the has_display check) and another identical one right after the `if args.remote_listen:` block for the non-remote path. `main`'s `finally` already closes controller/remote/power.

`pyproject.toml`: `"Pillow>=10.1"`. Service example `ExecStart` gains `--display waveshare28` with the comment: `# --display waveshare28 runs the LCD viewfinder; on a Pi without the panel the service logs a warning and continues for the Stick.`

Update the `app.py` module docstring with a fourth loop bullet: the LCD viewfinder (`pifilm/display/viewfinder.py`), which shares the controller with the remote API.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: PASS, clean

- [ ] **Step 5: Manual smoke without hardware**

Run: `.venv/bin/pifilm-capture --fake --display fake --no-preview --out /tmp/pifilm-vf & sleep 2; kill %1; ls /tmp/pifilm-vf`
Expected: `viewfinder-last.png` exists; open it and confirm the bar, needle, shutter circle and battery-less layout look like §3.4 of the spec.

- [ ] **Step 6: Commit**

```bash
git add pifilm/display/fake.py pifilm/capture/app.py pyproject.toml deploy/pifilm-capture.service.example tests/test_app.py
git commit -m "pifilm-capture --display: LCD viewfinder wiring with headless fallback"
```

---

### Task 10: Documentation

**Files:**
- Create: `docs/lcd-viewfinder.md`
- Modify: `docs/setup.md`, `README.md`, `CLAUDE.md`, `docs/picamera2-bringup.md`, `docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md`, `docs/superpowers/specs/2026-09-24-lcd-viewfinder-design.md` (§3.2 INT note)

- [ ] **Step 1: Write `docs/lcd-viewfinder.md`**

Sections, in this order, scannable like the other guides (tables and bullets, no long paragraphs):

1. **What it is** — two sentences; screenshot placeholder is NOT allowed: instead reference the `--display fake` PNG as the way to see the layout.
2. **Wiring** — the LCD pin / BCM GPIO / physical pin table and the X728 reserved-pin table from the spec §3 and the hardware handover verbatim; the warning that Waveshare lists BCM numbers.
3. **Pi setup** — `config.txt` needs `dtparam=spi=on` and `dtparam=i2c_arm=on`, no `dtoverlay=fbtft`; `sudo apt install python3-spidev python3-smbus2 python3-gpiozero python3-lgpio` (fall back to `python3-smbus` if `smbus2` is not packaged, and say the import name differs); `sudo usermod -aG spi,i2c,gpio george` then log out and in; verify with `ls -l /dev/spidev0.0 /dev/i2c-1` and `sudo i2cdetect -y 1` showing `58`.
4. **Running** — the three commands: terminal test, `--display-rotate 180`, `--touch-debug`; and the service line.
5. **The screen** — table of regions (live image, meter bar fields, needle, shutter, EV buttons, busy dot, battery badge, review screen) with what each shows and does. EV: 1/3 stops, ±2, resets on restart. Review: tap or 30 s.
6. **Reading the meter** — needle near 0 means AE is converged; hard left/right means AE has run out of range; `clip %` is the Phase 6 number, aim low outdoors; lux is libcamera's estimate.
7. **Hardware acceptance checklist** — the eight items from spec §4.
8. **Troubleshooting** — table: `display unavailable (...)` warning lines and their fixes (missing library, spi group, GPIO busy/fbtft, i2c group, no `58` on the bus → check cable/TP_RST).

- [ ] **Step 2: Cross-links**

- `docs/setup.md`: add a step after the Pi service step: "Optional: LCD viewfinder — see [docs/lcd-viewfinder.md](lcd-viewfinder.md)" with the one-line `usermod` and the apt line.
- `README.md`: in the capture-modes table add a row `| --display waveshare28 | LCD viewfinder with exposure meter and touch shutter ([guide](docs/lcd-viewfinder.md)) |`; under "From button press to displayed photo" add one bullet that the LCD shows Stick shots too and vice versa.
- `CLAUDE.md`: in "What this is" list `docs/lcd-viewfinder.md`; add a `### Display (pifilm/display/)` subsection under Architecture (five lines: drivers, meter, ui, loop, fake; all hardware imports lazy; the loop shares the controller); in the Picamera2 backend note that native size and tuning are now sensor-derived and the backend has a preview mode.
- `docs/picamera2-bringup.md`: a short "IMX477" subsection in §1: fixed lens (`autofocus none` in records), 4056×3040, 12-bit raw, `--tuning-file` unnecessary, the DNG acceptance test applies unchanged.
- Progress log: dated entry "2026-09-24 — LCD viewfinder implemented (hardware acceptance pending)" listing the checklist location.
- Spec §3.2: replace the INT-gating bullet with the polling decision and its reason (as in Task 5's design note).

- [ ] **Step 3: Verify links**

Run from the repo root:

```bash
.venv/bin/python - <<'PY'
import os, re
for f in ["docs/lcd-viewfinder.md", "docs/setup.md", "README.md", "CLAUDE.md",
          "docs/picamera2-bringup.md", "docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md"]:
    d = os.path.dirname(f)
    for m in re.finditer(r'\]\(([^)#h][^)#]*)', open(f).read()):
        q = os.path.normpath(os.path.join(d, m.group(1)))
        if not os.path.exists(q):
            print("BROKEN", f, m.group(1))
print("checked")
PY
```

Expected: only `checked`.

- [ ] **Step 4: Commit**

```bash
git add docs README.md CLAUDE.md
git commit -m "Document the LCD viewfinder: wiring, setup, screen, meter, acceptance"
```

---

## Self-review notes

- Spec §3.1–3.9 map to Tasks 4, 5, 6, 7, 8, 3+2, 1, 9, 9+10 respectively; §4 checklist is in Task 10's doc; §5 tests are spread across each task's Step 1.
- Deviation recorded: touch polls the bus instead of gating on INT (Task 5, Task 10 updates the spec).
- Names used across tasks: `finished_count`/`last_finished_job` (T1→T8), `set_ev`/`ev` (T3→T8, T9), `TouchPoint`/`Tap`/`TapDetector` (T5→T8), `MeterReading`/`compute_reading`/`format_*` (T6→T7, T8), `Action`/`hit`/`render_*`/`BAR_TOP`/`SHUTTER_CENTRE` (T7→T8), `ViewfinderLoop` (T8→T9), `_open_display`/`PREVIEW_SIZE` (T9 tests).
