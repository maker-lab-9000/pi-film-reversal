# Shutter Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shutter-speed `+`/`−` buttons to the LCD viewfinder. The Pi's
auto-exposure keeps choosing the gain (ISO) for the chosen shutter, and `A` returns to
full auto.

**Architecture:**
- A pure scale module (`pifilm/display/shutter.py`) steps through 1/3-stop shutter
  speeds.
- `Picamera2Camera.set_shutter` applies shutter priority through libcamera's
  `ExposureTimeMode`, falling back to a plain `ExposureTime` on older stacks. It
  mirrors `set_ev`: under the request lock, and written into both configurations.
- The meter flags a fixed shutter and a saturated gain.
- The UI draws two buttons around the shutter button, the viewfinder wires the taps,
  and every capture record gains `shutter_us`.

**Tech Stack:** Python 3.11+, NumPy, Pillow, Picamera2/libcamera (on the Pi only);
pytest; ruff (E, F, I, B, UP; line length 100).

**Spec:** `docs/superpowers/specs/2026-09-30-shutter-priority-design.md`

## Global Constraints

- Use `.venv/bin/pytest` and `.venv/bin/ruff`, from the repo root. Never a system Python.
- Nothing under `pifilm/display/` may import OpenCV, Picamera2 or libcamera. `picamera.py` imports `libcamera` lazily, inside functions, as it already does.
- Shutter series: 1/3-stop steps, 1/2000 s to 1 s, 34 steps, plus `A` (auto, `None`) at the slow end. `+` is faster; `−` is slower.
- `set_shutter` accepts `None` or an int in [100, 1_000_000] µs; anything else raises `CameraError`.
- Frame-duration margin: 1000 µs over the chosen shutter.
- `ISO MAX` threshold: `AnalogueGain >= 0.98 * max_gain`.
- Button boxes: `SHUTTER_PLUS_BOX = (264, 26, 312, 60)`, `SHUTTER_MINUS_BOX = (264, 144, 312, 178)`, `HIT_MARGIN` 6.
- ASCII-only labels: `A`, `1/250`, `0.3s`, `1s`, `S 1/250`, `ISO MAX`.
- Shutter state is memory only. It starts at `A` (`None`) on every start.
- Every existing test must pass. Where a test helper wraps a changed signature, extend the helper instead of weakening an assertion.
- End every commit message with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **A long shutter must survive a capture.** `switch_mode_and_capture_request`
   reconfigures twice, so the widened `FrameDurationLimits` and the shutter controls
   must be in both configuration dicts. Otherwise the live view snaps back to full rate
   and the next shot is on auto. Test in Task 2.
2. **Returning to `A` restores the original frame-duration limits.** Otherwise the live
   view stays at 4 fps after leaving 1/4 s. Test in Task 2.
3. **`set_shutter` failure changes nothing.** If `set_controls` raises, `shutter_us`
   and both configurations stay as they were. Test in Task 2.
4. **The first tap before any metered frame, or after a camera error.** It lands on
   1/125, not a crash. Tests in Task 1 and Task 5.
5. **A V4L2 camera.** No buttons are drawn and taps on the region do nothing. Test in
   Task 5.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `pifilm/display/shutter.py` | Create | Step series, `faster`/`slower`, `label` |
| `pifilm/capture/picamera.py` | Modify | `set_shutter`, `max_gain`, control-path detection, frame limits |
| `pifilm/capture/camera.py` | Modify | `FakeCamera.set_shutter`, `shutter_us`, `max_gain` |
| `pifilm/display/meter.py` | Modify | `shutter_fixed`, `iso_max` |
| `pifilm/display/ui.py` | Modify | Buttons, actions, readout |
| `pifilm/display/viewfinder.py` | Modify | Taps, metered exposure, render arguments |
| `pifilm/capture/app.py` | Modify | `shutter_us` in records |
| `tests/test_display_shutter.py` | Create | Scale tests |
| `tests/test_picamera.py`, `tests/test_camera.py`, `tests/test_display_meter.py`, `tests/test_display_ui.py`, `tests/test_viewfinder.py`, `tests/test_app.py`, `tests/test_double_capture.py` | Modify | New tests |
| `docs/lcd-viewfinder.md`, `docs/how-it-works.md`, `CLAUDE.md` | Modify | Docs |

---

### Task 1: Shutter scale (`pifilm/display/shutter.py`)

**Files:**
- Create: `pifilm/display/shutter.py`
- Test: `tests/test_display_shutter.py`

**Interfaces:**
- Produces:
  - `SHUTTER_STEPS: tuple[tuple[int, str], ...]`, as (µs, label) pairs, ascending.
  - `SHUTTER_STEPS_US: tuple[int, ...]`.
  - `DEFAULT_START_US = 8000`.
  - `faster(current_us: int | None, metered_us: float | None) -> int | None`.
  - `slower(current_us: int | None, metered_us: float | None) -> int | None`.
  - `label(us: int | None) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_display_shutter.py
import pytest

from pifilm.display.shutter import (
    DEFAULT_START_US,
    SHUTTER_STEPS,
    SHUTTER_STEPS_US,
    faster,
    label,
    slower,
)


def test_series_is_one_third_stops_from_1_2000_to_1_second():
    assert len(SHUTTER_STEPS_US) == 34
    assert SHUTTER_STEPS_US[0] == 500 and SHUTTER_STEPS_US[-1] == 1_000_000
    assert list(SHUTTER_STEPS_US) == sorted(set(SHUTTER_STEPS_US))
    assert [us for us, _ in SHUTTER_STEPS] == list(SHUTTER_STEPS_US)


def test_labels_are_ascii_and_conventional():
    assert label(None) == "A"
    assert label(4000) == "1/250"
    assert label(1563) == "1/640"
    assert label(300_000) == "0.3s"
    assert label(1_000_000) == "1s"
    assert all(text.isascii() for _, text in SHUTTER_STEPS)


def test_from_auto_the_first_tap_lands_on_the_metered_step():
    assert faster(None, 4100.0) == 4000
    assert slower(None, 4100.0) == 4000
    # nearest in stops, not in microseconds
    assert faster(None, 14_000.0) == 12_500


def test_from_auto_without_a_metered_value_starts_at_1_125():
    assert DEFAULT_START_US == 8000
    assert faster(None, None) == 8000
    assert slower(None, 0.0) == 8000


def test_steps_move_one_third_stop():
    assert faster(4000, None) == 3125
    assert slower(4000, None) == 5000


def test_fastest_step_stays_put_and_slowest_returns_to_auto():
    assert faster(500, None) == 500
    assert slower(1_000_000, None) is None


def test_an_off_series_value_snaps_to_the_nearest_step_first():
    assert faster(4100, None) == 3125
    assert slower(4100, None) == 5000


@pytest.mark.parametrize("fn", [faster, slower])
def test_every_step_is_reachable(fn):
    seen = set()
    value = 500 if fn is slower else 1_000_000
    for _ in range(40):
        seen.add(value)
        value = fn(value, None)
        if value is None:
            break
    assert set(SHUTTER_STEPS_US) <= seen
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_shutter.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pifilm.display.shutter'`

- [ ] **Step 3: Write the implementation**

```python
# pifilm/display/shutter.py
"""Shutter-priority steps for the viewfinder's shutter buttons. Pure.

The series is the standard 1/3-stop shutter scale from 1/2000 s to 1 s, as
(microseconds, label) pairs, with ``None`` meaning ``A``: full auto exposure. ``+``
is faster and ``-`` is slower; one step slower than 1 s is back to ``A``.

From ``A`` the first tap in either direction lands on the step nearest the exposure
auto-exposure is using right now, so switching into shutter priority never jumps the
picture; the next tap moves it. "Nearest" is measured in stops (log2), not
microseconds, because a stop is what the eye sees. Before any metered frame (the
first frame, or after a camera error) the start is 1/125, a safe hand-held speed.

The labels are ASCII only: the viewfinder's default font has no glyphs beyond it.
"""

from __future__ import annotations

import math

SHUTTER_STEPS: tuple[tuple[int, str], ...] = (
    (500, "1/2000"), (625, "1/1600"), (800, "1/1250"), (1000, "1/1000"),
    (1250, "1/800"), (1563, "1/640"), (2000, "1/500"), (2500, "1/400"),
    (3125, "1/320"), (4000, "1/250"), (5000, "1/200"), (6250, "1/160"),
    (8000, "1/125"), (10_000, "1/100"), (12_500, "1/80"), (16_667, "1/60"),
    (20_000, "1/50"), (25_000, "1/40"), (33_333, "1/30"), (40_000, "1/25"),
    (50_000, "1/20"), (66_667, "1/15"), (76_923, "1/13"), (100_000, "1/10"),
    (125_000, "1/8"), (166_667, "1/6"), (200_000, "1/5"), (250_000, "1/4"),
    (300_000, "0.3s"), (400_000, "0.4s"), (500_000, "0.5s"), (600_000, "0.6s"),
    (800_000, "0.8s"), (1_000_000, "1s"),
)
SHUTTER_STEPS_US: tuple[int, ...] = tuple(us for us, _ in SHUTTER_STEPS)
DEFAULT_START_US = 8000  # 1/125
_LABELS = dict(SHUTTER_STEPS)


def _nearest_index(us: float) -> int:
    target = math.log2(us)
    return min(range(len(SHUTTER_STEPS_US)),
               key=lambda i: abs(math.log2(SHUTTER_STEPS_US[i]) - target))


def _start(metered_us: float | None) -> int:
    if metered_us is None or not metered_us > 0 or not math.isfinite(metered_us):
        return DEFAULT_START_US
    return SHUTTER_STEPS_US[_nearest_index(metered_us)]


def faster(current_us: int | None, metered_us: float | None) -> int | None:
    if current_us is None:
        return _start(metered_us)
    i = _nearest_index(current_us)
    if SHUTTER_STEPS_US[i] > current_us:  # off-series value between steps: snap down
        return SHUTTER_STEPS_US[i - 1] if i > 0 else SHUTTER_STEPS_US[0]
    return SHUTTER_STEPS_US[max(0, i - 1)]


def slower(current_us: int | None, metered_us: float | None) -> int | None:
    if current_us is None:
        return _start(metered_us)
    i = _nearest_index(current_us)
    if SHUTTER_STEPS_US[i] < current_us:  # off-series value between steps: snap up
        i += 1
        return SHUTTER_STEPS_US[i] if i < len(SHUTTER_STEPS_US) else None
    return SHUTTER_STEPS_US[i + 1] if i + 1 < len(SHUTTER_STEPS_US) else None


def label(us: int | None) -> str:
    if us is None:
        return "A"
    if us in _LABELS:
        return _LABELS[us]
    return _LABELS[SHUTTER_STEPS_US[_nearest_index(us)]]
```

Check the off-series cases by hand against the tests:
- `faster(4100)`: the nearest step is 4000, which is below 4100. The code then takes
  one step faster, giving 3125.
- `slower(4100)`: the nearest step is 4000, which is below 4100, so it snaps up to
  5000.

Both match the tests.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/test_display_shutter.py && .venv/bin/ruff check pifilm/display/shutter.py tests/test_display_shutter.py`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/shutter.py tests/test_display_shutter.py
git commit -m "Shutter priority: 1/3-stop shutter scale with an auto position

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Camera backends (`set_shutter`)

**Files:**
- Modify: `pifilm/capture/picamera.py`: constants near `_EV_RANGE`, `__init__` (attributes before the `try`, detection inside it, frame limits after the configurations are built), a new `set_shutter` after `set_ev`, and a new helper function near `_set_config_control`
- Modify: `pifilm/capture/camera.py`: `FakeCamera`
- Test: `tests/test_picamera.py`, `tests/test_camera.py`

**Interfaces:**
- Produces:
  - `Picamera2Camera.set_shutter(us: int | None) -> None`.
  - `Picamera2Camera.shutter_us: int | None`.
  - `Picamera2Camera.max_gain: float | None`.
  - `FakeCamera.set_shutter`, `FakeCamera.shutter_us`, `FakeCamera.max_gain` (both `None` by default).
  - `V4L2Camera` has no `set_shutter`.

- [ ] **Step 1: Extend the `install_picamera` fixture** in `tests/test_picamera.py`.

  Read the fixture and `_fake_libcamera_module` first, then add three keyword
  parameters to `install(...)`:
  - `with_exposure_time_mode=False`
  - `analogue_gain=None`
  - `frame_limits=None`

  Wire them as follows:
  - Pass `with_exposure_time_mode` through to `_fake_libcamera_module`. When it is
    true, set `controls_ns.ExposureTimeModeEnum = _ExposureTimeModeEnum`, and in
    `FakePicamera2.__init__` add `self.camera_controls["ExposureTimeMode"] = (0, 1, 0)`.
  - When `analogue_gain` is given, set `self.camera_controls["AnalogueGain"] = analogue_gain`
    (a `(min, max, default)` tuple).
  - When `frame_limits` is given (a dict with `"still"` and `"preview"` tuples), make
    `create_still_configuration` / `create_preview_configuration` put
    `FrameDurationLimits` into the returned `controls` dict. This mimics Picamera2,
    which merges its default frame limits into every configuration.

  Add next to the other enum stand-ins:

```python
class _ExposureTimeModeEnum:
    Auto = "exposure-time-auto"
    Manual = "exposure-time-manual"
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_picamera.py`)

```python
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
    assert "legacy" in capsys.readouterr().err
    camera.set_shutter(250_000)
    assert state.instance.set_controls_calls[-1]["ExposureTime"] == 250_000
    assert "ExposureTimeMode" not in state.instance.set_controls_calls[-1]
    camera.set_shutter(None)
    assert state.instance.set_controls_calls[-1]["ExposureTime"] == 0
    assert state.instance.preview_config["controls"]["ExposureTime"] == 0
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


@pytest.mark.parametrize("bad", [99, 1_000_001, -5])
def test_shutter_outside_the_range_is_refused(install_picamera, bad):
    install_picamera()
    camera = Picamera2Camera()
    with pytest.raises(CameraError):
        camera.set_shutter(bad)
    assert camera.shutter_us is None
    camera.close()


def test_a_rejected_shutter_changes_nothing(install_picamera):
    state = install_picamera(with_exposure_time_mode=True)
    camera = Picamera2Camera(preview=True)
    before = dict(state.instance.preview_config["controls"])

    def boom(controls):
        raise RuntimeError("control rejected")

    state.instance.set_controls = boom
    with pytest.raises(CameraError):
        camera.set_shutter(4000)
    assert camera.shutter_us is None
    assert state.instance.preview_config["controls"] == before
    camera.close()


def test_max_gain_comes_from_the_sensor_controls(install_picamera):
    install_picamera(analogue_gain=(1.0, 22.26, 1.0))
    camera = Picamera2Camera()
    assert camera.max_gain == pytest.approx(22.26)
    camera.close()
    install_picamera()
    camera = Picamera2Camera()
    assert camera.max_gain is None
    camera.close()
```

If `set_controls_calls` is recorded by a `set_controls` method that the rejected-shutter
test replaces, the test still works: it only checks state after the failure.

Append to `tests/test_camera.py`:

```python
def test_fake_camera_records_the_shutter():
    camera = FakeCamera()
    assert camera.shutter_us is None and camera.max_gain is None
    camera.set_shutter(4000)
    assert camera.shutter_us == 4000
    camera.set_shutter(None)
    assert camera.shutter_us is None
```

(Match that file's existing `FakeCamera` import. Add it to the imports if missing.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_picamera.py tests/test_camera.py -k "shutter or max_gain"`
Expected: FAIL with `AttributeError: ... 'set_shutter'` (or the fixture's unexpected keyword before Step 1 is done).

- [ ] **Step 4: Implement in `pifilm/capture/picamera.py`**

Constants, next to `_EV_RANGE`:

```python
_SHUTTER_RANGE_US = (100, 1_000_000)
# A fixed exposure longer than a frame period is clipped by the sensor unless the
# frame is allowed to last at least that long, plus readout headroom.
_FRAME_MARGIN_US = 1000
```

In `__init__`, next to `self.ev = float(ev)`:

```python
        self.shutter_us: int | None = None
        self.max_gain: float | None = None
        self._shutter_mode = "legacy"
        self._frame_limits: dict[str, tuple[int, int] | None] = {"still": None, "preview": None}
```

Inside the `try`, right after `has_autofocus = ...`:

```python
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
            gain = camera.camera_controls.get("AnalogueGain")
            if isinstance(gain, (tuple, list)) and len(gain) >= 2:
                self.max_gain = float(gain[1])
```

After the preview configuration block, before `camera.set_controls(controls)`:

```python
            self._frame_limits = {
                "still": _config_frame_limits(self._still_config),
                "preview": _config_frame_limits(self._preview_config),
            }
```

After `set_ev`:

```python
    def set_shutter(self, us: int | None) -> None:
        """Fix the exposure time (shutter priority), or return to auto with ``None``.

        Gain stays automatic, so EV compensation still works through it. Like
        ``set_ev``, the controls go to the running camera and into both
        configurations, because every configure reapplies a configuration's own
        controls. A long shutter also widens the frame-duration limit, which is why
        the live view slows at long shutters, and ``None`` restores the limits each
        configuration was built with. Nothing changes if the camera rejects the
        controls.
        """
        if us is not None:
            us = int(us)
            if not _SHUTTER_RANGE_US[0] <= us <= _SHUTTER_RANGE_US[1]:
                raise CameraError(f"shutter must be within {_SHUTTER_RANGE_US} us, got {us}")
        camera = self._camera
        if camera is None:
            raise CameraError("Cannot set the shutter on a closed Picamera2 camera")
        running = "preview" if self._preview_config is not None else "still"
        with self._request_lock:
            try:
                camera.set_controls(self._shutter_controls(us, self._frame_limits[running]))
            except Exception as exc:
                raise CameraError(f"Picamera2 failed to set the shutter: {exc}") from exc
            for key, config in (("still", self._still_config),
                                ("preview", self._preview_config)):
                for name, value in self._shutter_controls(us, self._frame_limits[key]).items():
                    _set_config_control(config, name, value)
        self.shutter_us = us

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
```

Near `_set_config_control`:

```python
def _config_frame_limits(config: Any) -> tuple[int, int] | None:
    """The FrameDurationLimits a configuration was built with, if it has any."""
    try:
        low, high = config["controls"]["FrameDurationLimits"]
        return int(low), int(high)
    except (TypeError, KeyError, IndexError, ValueError):
        return None
```

Add a short paragraph to the module docstring after the `set_ev` sentence: "``set_shutter``
does the same for a fixed exposure time (shutter priority), and also widens and
restores ``FrameDurationLimits``."

In `pifilm/capture/camera.py` `FakeCamera.__init__`, next to `self.ev = 0.0`, add
`self.shutter_us: int | None = None` and `self.max_gain: float | None = None`. After
`set_ev`, add:

```python
    def set_shutter(self, us: int | None) -> None:
        # Mirrors Picamera2Camera.set_shutter for the viewfinder's shutter buttons.
        self.shutter_us = None if us is None else int(us)
```

- [ ] **Step 5: Run the tests**

The new start-up line goes to stderr. If an existing test asserts that construction
writes nothing to stderr (check with `grep -n "readouterr" tests/test_picamera.py`),
update that assertion to allow exactly this line and nothing else. Do not drop the
check.

Run: `.venv/bin/pytest -q tests/test_picamera.py tests/test_camera.py && .venv/bin/ruff check pifilm/capture tests/test_picamera.py tests/test_camera.py`
Expected: all PASS; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add pifilm/capture/picamera.py pifilm/capture/camera.py tests/test_picamera.py tests/test_camera.py
git commit -m "Shutter priority: set_shutter on Picamera2 and the fake camera

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Meter flags (`pifilm/display/meter.py`)

**Files:**
- Modify: `pifilm/display/meter.py` (`MeterReading`, `compute_reading`)
- Test: `tests/test_display_meter.py` (append)

**Interfaces:**
- Produces:
  - `MeterReading.shutter_fixed: bool = False`.
  - `MeterReading.iso_max: bool = False`.
  - `compute_reading(metadata, preview_rgb, ev_comp, power, *, focus=None, focus_peak=None, shutter_us=None, max_gain=None)`.

- [ ] **Step 1: Write the failing tests** (append; reuse the file's existing imports of `compute_reading` and numpy)

```python
def _frame():
    return np.full((48, 64, 3), 118, dtype=np.uint8)


def test_iso_max_flags_a_saturated_gain():
    meta = {"ExposureTime": 1000, "AnalogueGain": 21.9}
    assert compute_reading(meta, _frame(), 0.0, None, max_gain=22.26).iso_max is True
    meta["AnalogueGain"] = 21.0
    assert compute_reading(meta, _frame(), 0.0, None, max_gain=22.26).iso_max is False


def test_no_iso_max_without_a_known_maximum_or_gain():
    assert compute_reading({"AnalogueGain": 40.0}, _frame(), 0.0, None).iso_max is False
    assert compute_reading({}, _frame(), 0.0, None, max_gain=22.26).iso_max is False


def test_shutter_fixed_follows_the_chosen_value():
    meta = {"ExposureTime": 4000, "AnalogueGain": 2.0}
    assert compute_reading(meta, _frame(), 0.0, None, shutter_us=4000).shutter_fixed is True
    assert compute_reading(meta, _frame(), 0.0, None).shutter_fixed is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_meter.py -k "iso_max or shutter_fixed"`
Expected: FAIL with `TypeError: compute_reading() got an unexpected keyword argument 'max_gain'`

- [ ] **Step 3: Implement**

Add to `MeterReading`, after `focus_peak`:

```python
    # Shutter priority: the user fixed the exposure time, so the readout marks it.
    shutter_fixed: bool = False
    # Analogue gain is at the sensor's maximum: a fixed shutter this fast cannot be
    # compensated and the frame will be dark.
    iso_max: bool = False
```

Add a module-level constant `ISO_MAX_FRACTION = 0.98`. Change `compute_reading`'s
keyword-only parameters to
`*, focus: float | None = None, focus_peak: float | None = None, shutter_us: int | None = None, max_gain: float | None = None`.
Before the `return`, add:

```python
    iso_max = bool(
        max_gain is not None and analogue is not None
        and analogue >= ISO_MAX_FRACTION * max_gain
    )
```

Extend the `MeterReading(...)` call with `shutter_us is not None, iso_max` after
`focus, focus_peak`.

- [ ] **Step 4: Run the meter tests**

Run: `.venv/bin/pytest -q tests/test_display_meter.py && .venv/bin/ruff check pifilm/display/meter.py tests/test_display_meter.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/meter.py tests/test_display_meter.py
git commit -m "Shutter priority: meter flags a fixed shutter and a saturated gain

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Buttons and readout (`pifilm/display/ui.py`)

**Files:**
- Modify: `pifilm/display/ui.py`
- Test: `tests/test_display_ui.py` (append; add new names to its top import list)

**Interfaces:**
- Consumes: `MeterReading.shutter_fixed` and `iso_max` (Task 3).
- Produces:
  - `SHUTTER_PLUS_BOX`, `SHUTTER_MINUS_BOX`.
  - `Action.SHUTTER_FASTER`, `Action.SHUTTER_SLOWER`.
  - `render_live(frame_rgb, reading, double=None, shutter_buttons=False)`.

- [ ] **Step 1: Write the failing tests**

```python
def _box_centre(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) // 2, (y0 + y1) // 2


def test_shutter_buttons_hit_inside_their_boxes_and_margins():
    assert hit(*_box_centre(SHUTTER_PLUS_BOX)) is Action.SHUTTER_FASTER
    assert hit(*_box_centre(SHUTTER_MINUS_BOX)) is Action.SHUTTER_SLOWER
    x0, y0, x1, y1 = SHUTTER_PLUS_BOX
    assert hit(x0 - HIT_MARGIN, y0 - HIT_MARGIN) is Action.SHUTTER_FASTER
    x0, y0, x1, y1 = SHUTTER_MINUS_BOX
    assert hit(x1 + HIT_MARGIN, y1 + HIT_MARGIN) is Action.SHUTTER_SLOWER


def test_shutter_button_regions_overlap_no_other_control():
    cx, cy = SHUTTER_CENTRE
    for box, action in ((SHUTTER_PLUS_BOX, Action.SHUTTER_FASTER),
                        (SHUTTER_MINUS_BOX, Action.SHUTTER_SLOWER)):
        x0, y0, x1, y1 = box
        for x in range(x0 - HIT_MARGIN, min(WIDTH, x1 + HIT_MARGIN + 1)):
            for y in range(y0 - HIT_MARGIN, y1 + HIT_MARGIN + 1):
                assert hit(x, y) is action, (x, y)
    # the shutter button still owns its whole circle
    assert hit(cx, cy - SHUTTER_RADIUS - HIT_MARGIN) is Action.SHUTTER
    assert hit(cx, cy + SHUTTER_RADIUS + HIT_MARGIN) is Action.SHUTTER
    assert SHUTTER_MINUS_BOX[3] + HIT_MARGIN < BAR_TOP - HIT_MARGIN


def test_shutter_buttons_are_drawn_only_when_asked():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    plain = np.asarray(render_live(frame, _reading()))
    with_buttons = np.asarray(render_live(frame, _reading(), shutter_buttons=True))
    x0, y0, x1, y1 = SHUTTER_PLUS_BOX
    assert np.array_equal(plain[y0:y1 + 1, x0:x1 + 1], np.full_like(
        plain[y0:y1 + 1, x0:x1 + 1], 200))
    assert not np.array_equal(plain[y0:y1 + 1, x0:x1 + 1],
                              with_buttons[y0:y1 + 1, x0:x1 + 1])


def test_fixed_shutter_readout_is_prefixed(monkeypatch):
    drawn = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        drawn.append(text)
        return real(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    render_live(frame, _reading(shutter="1/250", shutter_fixed=True))
    assert any(t.startswith("S 1/250") for t in drawn)
    drawn.clear()
    render_live(frame, _reading(shutter="1/250"))
    assert any(t.startswith("1/250") for t in drawn)


def test_iso_max_is_amber(monkeypatch):
    calls = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        calls.append((text, k.get("fill")))
        return real(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    render_live(frame, _reading(iso_max=True))
    assert ("ISO MAX", AMBER + (255,)) in calls
```

(`ImageDraw` is already imported from PIL in that file. `_reading` accepts overrides,
and the new fields have defaults. Add `SHUTTER_PLUS_BOX`, `SHUTTER_MINUS_BOX` and
`HIT_MARGIN` to the imports if they are not there yet.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_ui.py -k "shutter_button or fixed_shutter or iso_max"`
Expected: FAIL with `ImportError: cannot import name 'SHUTTER_PLUS_BOX'`

- [ ] **Step 3: Implement in `pifilm/display/ui.py`**

Constants, after `DOUBLE_BOX`:

```python
# Shutter-priority buttons, stacked above and below the shutter button. With
# HIT_MARGIN the + region ends at y 66 and the - region starts at y 138, both clear
# of the shutter button's hit circle (y 68-136), and the - region ends at y 184,
# above the bar's hit zone (BAR_TOP - HIT_MARGIN = 198).
SHUTTER_PLUS_BOX = (264, 26, 312, 60)
SHUTTER_MINUS_BOX = (264, 144, 312, 178)
```

Add `SHUTTER_FASTER = "shutter_faster"` and `SHUTTER_SLOWER = "shutter_slower"` to
`Action`.

Update the module docstring's layout sentence to mention the shutter buttons.

Change the signature to
`def render_live(frame_rgb, reading, double=None, shutter_buttons: bool = False)`.
Keep the annotations and wrap at 100 columns.

Replace the readout `line1` block with:

```python
    shutter_text = text_or_dash(reading.shutter)
    if reading.shutter_fixed:
        shutter_text = f"S {shutter_text}"
    x = EV_BUTTON_W + 6
    if reading.iso_max:
        # Drawn in three runs so only the ISO turns amber.
        for text, colour in ((f"{shutter_text}  ", (255, 255, 255, 255)),
                             ("ISO MAX", AMBER + (255,)),
                             (f"  EV {format_ev(reading.ev_comp)}", (255, 255, 255, 255))):
            draw.text((x, BAR_TOP + 3), text, fill=colour, font=font)
            x += draw.textlength(text, font=font)
    else:
        line1 = f"{shutter_text}  {iso_label(reading.iso)}  EV {format_ev(reading.ev_comp)}"
        draw.text((x, BAR_TOP + 3), line1, fill=(255, 255, 255, 255), font=font)
```

Also remove the old `draw.text((EV_BUTTON_W + 6, BAR_TOP + 3), line1, ...)` line.
`line2` is unchanged.

After the shutter-button drawing:

```python
    if shutter_buttons:
        for box, glyph in ((SHUTTER_PLUS_BOX, "+"), (SHUTTER_MINUS_BOX, "-")):
            draw.rectangle(box, fill=(0, 0, 0, BAR_ALPHA), outline=(255, 255, 255, 200))
            draw.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), glyph,
                      fill=(255, 255, 255, 255), font=big, anchor="mm")
```

In `hit()`, after the `DOUBLE_BOX` check:

```python
    for box, action in ((SHUTTER_PLUS_BOX, Action.SHUTTER_FASTER),
                        (SHUTTER_MINUS_BOX, Action.SHUTTER_SLOWER)):
        x0, y0, x1, y1 = box
        if x0 - HIT_MARGIN <= x <= x1 + HIT_MARGIN and y0 - HIT_MARGIN <= y <= y1 + HIT_MARGIN:
            return action
```

- [ ] **Step 4: Run the UI tests**

Run: `.venv/bin/pytest -q tests/test_display_ui.py && .venv/bin/ruff check pifilm/display/ui.py tests/test_display_ui.py`
Expected: all PASS, including every existing test. The non-`iso_max` path draws the
same single string as before.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/ui.py tests/test_display_ui.py
git commit -m "Shutter priority: shutter buttons, S readout and amber ISO MAX

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Viewfinder wiring (`pifilm/display/viewfinder.py`)

**Files:**
- Modify: `pifilm/display/viewfinder.py`
- Test: `tests/test_viewfinder.py` (append; add `SHUTTER_PLUS_BOX`, `SHUTTER_MINUS_BOX` to the `pifilm.display.ui` import)

**Interfaces:**
- Consumes:
  - `shutter.faster` / `slower` (Task 1).
  - `camera.set_shutter`, `shutter_us`, `max_gain` (Task 2).
  - `compute_reading(..., shutter_us=, max_gain=)` (Task 3).
  - `render_live(..., shutter_buttons=)`, `Action.SHUTTER_FASTER/SLOWER` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
def _centre(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) // 2, (y0 + y1) // 2


def test_first_shutter_tap_from_auto_lands_on_the_metered_speed(controller):
    camera, ctl = controller  # FakeCamera with ExposureTime 4000 in its metadata
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()  # one live frame, so the loop has a metered exposure
    _tap(loop, touch, clock, _centre(SHUTTER_PLUS_BOX))
    assert camera.shutter_us == 4000
    _tap(loop, touch, clock, _centre(SHUTTER_PLUS_BOX))
    assert camera.shutter_us == 3125
    _tap(loop, touch, clock, _centre(SHUTTER_MINUS_BOX))
    _tap(loop, touch, clock, _centre(SHUTTER_MINUS_BOX))
    assert camera.shutter_us == 5000


def test_a_shutter_tap_before_any_metered_frame_starts_at_1_125(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    touch.tap(*_centre(SHUTTER_MINUS_BOX))
    camera.read = lambda **k: (_ for _ in ()).throw(CameraError("no frame"))
    loop.step()
    clock.t += 0.1
    loop.step()
    assert camera.shutter_us == 8000


def test_live_view_passes_shutter_state_to_meter_and_renderer(controller, monkeypatch):
    camera, ctl = controller
    seen = {}
    real_render, real_compute = viewfinder.render_live, viewfinder.compute_reading

    def render(frame, reading, double=None, shutter_buttons=False):
        seen["buttons"], seen["fixed"] = shutter_buttons, reading.shutter_fixed
        return real_render(frame, reading, double, shutter_buttons)

    def compute(*a, **k):
        seen["max_gain"] = k.get("max_gain")
        return real_compute(*a, **k)

    monkeypatch.setattr(viewfinder, "render_live", render)
    monkeypatch.setattr(viewfinder, "compute_reading", compute)
    camera.max_gain = 16.0
    camera.set_shutter(4000)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert seen == {"buttons": True, "fixed": True, "max_gain": 16.0}


def test_a_camera_without_set_shutter_draws_no_buttons_and_ignores_taps(
    controller, monkeypatch,
):
    camera, ctl = controller
    seen = []
    real_render = viewfinder.render_live

    def render(frame, reading, double=None, shutter_buttons=False):
        seen.append(shutter_buttons)
        return real_render(frame, reading, double, shutter_buttons)

    monkeypatch.setattr(viewfinder, "render_live", render)
    monkeypatch.setattr(type(camera), "set_shutter", None, raising=False)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    _tap(loop, touch, clock, _centre(SHUTTER_PLUS_BOX))
    assert seen and not any(seen)
    assert camera.shutter_us is None
    assert ctl.snapshot().finished_count == 0


def test_a_rejected_shutter_is_logged_not_fatal(controller):
    camera, ctl = controller
    lines = []

    def reject(us):
        raise CameraError("control rejected")

    camera.set_shutter = reject
    loop, touch, display, clock = _loop(camera, ctl, log=lines.append)
    loop.step()
    _tap(loop, touch, clock, _centre(SHUTTER_PLUS_BOX))
    assert any("control rejected" in line for line in lines)
    assert loop.state == "LIVE"
```

Notes:
- `_tap` and `_loop` already exist in the file, and so does `CameraError`.
- `monkeypatch.setattr(type(camera), "set_shutter", None)` makes
  `callable(getattr(camera, "set_shutter", None))` false for that test only.
- Existing tests spy on `render_live` with helpers that take `double=None`. Extend any
  such helper (for example `_spy_live`) to accept and forward `shutter_buttons=False`,
  and do not change their assertions.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py -k "shutter"`
Expected: FAIL. The first test fails with `camera.shutter_us` still `None`, and the
spy tests fail on the unexpected keyword.

- [ ] **Step 3: Implement in `pifilm/display/viewfinder.py`**

1. Imports: add `from . import shutter` (next to the other relative imports).
2. In `__init__`, next to `self.ev_comp = ...`:

   ```python
           # Shutter priority exists only on backends that can fix the exposure time
           # (Picamera2, the fake); on V4L2 the buttons are neither drawn nor live.
           self._shutter_ok = callable(getattr(camera, "set_shutter", None))
           self._metered_us: float | None = None
   ```

3. Next to `_set_ev`, add:

   ```python
       def _step_shutter(self, action: Action) -> None:
           step = shutter.faster if action is Action.SHUTTER_FASTER else shutter.slower
           value = step(getattr(self._camera, "shutter_us", None), self._metered_us)
           try:
               self._camera.set_shutter(value)
           except CameraError as exc:
               self._log(f"camera: {exc}")
   ```

4. In the LIVE tap dispatch, add after the EV branches:

   ```python
               elif (action in (Action.SHUTTER_FASTER, Action.SHUTTER_SLOWER)
                     and self._shutter_ok):
                   self._step_shutter(action)
   ```

5. In the live-frame section, right after `frame = self._camera.read(full=False)`
   succeeds:

   ```python
           exposure = (frame.metadata or {}).get("ExposureTime")
           if isinstance(exposure, (int, float)) and exposure > 0:
               self._metered_us = float(exposure)
   ```

6. Extend the `compute_reading(...)` call with
   `shutter_us=getattr(self._camera, "shutter_us", None), max_gain=getattr(self._camera, "max_gain", None)`.
   Extend the `render_live(...)` call with `shutter_buttons=self._shutter_ok`.

7. In the module docstring, add one sentence: shutter taps go straight to the camera,
   like EV taps, so they also work while a job is processing.

- [ ] **Step 4: Run the viewfinder tests and the fast suite**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py && .venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/viewfinder.py tests/test_viewfinder.py
git commit -m "Shutter priority: viewfinder shutter taps, metered start, meter and render wiring

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `shutter_us` in the capture record (`pifilm/capture/app.py`)

**Files:**
- Modify: `pifilm/capture/app.py`: `CaptureSession.capture()`, the single-shot record, and `_capture_double`'s `camera_fields`
- Test: `tests/test_app.py`, `tests/test_double_capture.py` (append)

**Interfaces:**
- Consumes: `camera.shutter_us` (Task 2).
- Produces: every `captures.jsonl` record has `"shutter_us": int | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app.py`, which already has the `pipeline` fixture and `_session`:

```python
def test_record_carries_the_shutter_setting(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    first = session.capture()
    session.camera.set_shutter(4000)
    second = session.capture()
    assert first.record["shutter_us"] is None
    assert second.record["shutter_us"] == 4000
```

Append to `tests/test_double_capture.py`:

```python
def test_both_double_exposure_records_carry_their_shutter(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.camera.set_shutter(250_000)
    first = session.capture()
    session.camera.set_shutter(None)
    second = session.capture()
    assert first.record["shutter_us"] == 250_000
    assert second.record["shutter_us"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_app.py tests/test_double_capture.py -k shutter`
Expected: FAIL with `KeyError: 'shutter_us'`

- [ ] **Step 3: Implement**

In `capture()`, next to the `ev = ...` line (and before `camera.read()`), add:

```python
        # Also read before the frame: the shutter the exposure was made at (None = auto).
        shutter_us = getattr(self.camera, "shutter_us", None)
```

- Pass `shutter_us` into `_capture_double`, adding it as a keyword parameter. That
  method already has a local named `shutter` holding the `perf_counter` start, so use
  the name `shutter_us` to avoid a clash.
- Add `"shutter_us": shutter_us,` to the single-shot record right after `**info,`.
- Add the same entry to `_capture_double`'s `camera_fields` dict.

Update the module docstring paragraph that lists what the audit line carries: add
"``shutter_us`` (the fixed exposure time under shutter priority, null on auto)".

- [ ] **Step 4: Run the capture tests**

Run: `.venv/bin/pytest -q tests/test_app.py tests/test_double_capture.py && .venv/bin/ruff check pifilm/capture/app.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/app.py tests/test_app.py tests/test_double_capture.py
git commit -m "Shutter priority: record shutter_us with every capture

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/lcd-viewfinder.md` (§5 table, a new §5 subsection, the §7 checklist)
- Modify: `docs/how-it-works.md` (the "At capture time, on the Pi" section)
- Modify: `CLAUDE.md` (the Display and Capture sections)

- [ ] **Step 1: `docs/lcd-viewfinder.md` §5 table.** Add these rows after the EV row:

```markdown
| Shutter `+` / `−` buttons (right edge, above and below the shutter button) | `+` and `-` | Shutter priority: `+` one 1/3 stop faster, `−` slower, 1/2000 to 1 s; slower than 1 s returns to auto (`A`). The first tap from auto starts at the speed auto-exposure is using. ISO stays automatic; EV still works. Picamera2 only; resets to auto on restart |
| Shutter readout `S 1/250` | `S` prefix when the shutter is fixed; otherwise the metered speed | — |
| `ISO MAX` (amber, in the readout) | Gain is at the sensor's maximum: the fixed shutter is too fast for the light and the photo will be dark | — |
```

- [ ] **Step 2: Add a subsection** after "### Double exposure" and before "## 6.":

```markdown
### Shutter priority

Tap the shutter `+` / `−` buttons to fix the shutter speed; the camera keeps choosing
the ISO to expose correctly, and EV compensation still shifts the result. Use a fast
speed (1/500 and up) to freeze motion and a slow one (1/15 and down) to blur it; the
readout shows `S` in front of a fixed speed. Keep tapping `−` past 1 s to return to
full auto.

- At long speeds the live view slows to match: at 1/4 s it shows about 4 frames a
  second. That is the exposure, not a fault.
- `ISO MAX` in amber means the camera has run out of gain for this speed. The photo
  will come out dark and grainy (it is graded like any dark frame); pick a slower
  speed or add light.
- Each photo's `captures.jsonl` line records `shutter_us` (null on auto); the real
  exposure time and gain are in `camera_metadata`.
- At start-up the journal names how the shutter is fixed:
  `picamera2: shutter priority via ExposureTimeMode` on current libcamera, or
  `... ExposureTime (legacy libcamera)` on older stacks.
```

- [ ] **Step 3: Append to the §7 checklist**, using the next number:

```markdown
12. Shutter priority: in a dim room tap `+` up to 1/1000: the readout shows
    `S 1/1000` and amber `ISO MAX`, and the shot is darker. Tap `−` down to 1/4 and pan
    across a scene: the live view drops to about 4 fps and a shot shows motion blur.
    Each shot's `camera_metadata.ExposureTime` is within a few percent of the chosen
    speed, and its `shutter_us` matches. Keep tapping `−` past 1 s: the readout loses
    the `S` and the live view returns to full rate. The journal line at start-up names
    the control path.
```

- [ ] **Step 4: `docs/how-it-works.md`.** Add a paragraph at the end of the "At capture
time, on the Pi" section, before any `###` subsection that follows it:

```markdown
Shutter priority (the viewfinder's shutter buttons) fixes the exposure time and
leaves gain to the Pi's auto-exposure, which meets its brightness target with gain
alone. EV compensation shifts that target, so it keeps working. The grade does not
change: a frame that ran out of gain (`ISO MAX`) is normalised like any dark frame.
```

- [ ] **Step 5: `CLAUDE.md`.**
  - In the Display paragraph, add after the `meter.py` clause: "`shutter.py` is the
    shutter-priority step scale (pure)".
  - In the Capture `picamera.py` bullet, add: "`set_shutter` gives shutter priority
    (`ExposureTimeMode` on libcamera 0.5, legacy `ExposureTime` otherwise), written into
    both configurations like `set_ev`."

- [ ] **Step 6: Verify and commit**

Run: `grep -n "Shutter priority" docs/lcd-viewfinder.md docs/how-it-works.md && .venv/bin/ruff check .`
Expected: the headings and rows are found; ruff clean.

```bash
git add docs/lcd-viewfinder.md docs/how-it-works.md CLAUDE.md
git commit -m "Shutter priority: document the buttons, readout, ISO MAX and acceptance

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
