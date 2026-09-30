# Shutter priority on the LCD viewfinder — design

**Date:** 2026-09-30
**Status:** approved in discussion; awaiting spec review.
**Depends on:** the LCD viewfinder (`docs/lcd-viewfinder.md`) and the Picamera2 backend.
Builds on `main` after double exposure (PR #39).

## 1. Goal

The viewfinder gets touch buttons that set the shutter speed. The Pi's auto-exposure
keeps choosing the gain (ISO) for the chosen shutter, so the photographer picks motion
blur or freeze and the camera keeps the exposure right. EV compensation keeps working
alongside it. A slow end position, `A`, returns to full auto.

## 2. Decisions taken in brainstorming

| Topic | Decision |
| --- | --- |
| Mode | Shutter priority: shutter fixed by the user, analogue gain automatic. Not full manual. |
| Camera mechanism | libcamera's `ExposureTimeMode=Manual` + `ExposureTime`, with `AnalogueGainMode` left on Auto. If those controls are absent (older libcamera), fall back to a fixed `ExposureTime` with AE on. |
| Buttons | A separate `+`/`−` pair on the right edge, stacked above and below the shutter button. The EV buttons in the bar are unchanged. |
| Steps | 1/3-stop shutter speeds from 1/2000 s to 1 s, plus `A` (auto) at the slow end. `+` is faster; `−` is slower. |
| Too little light | Warn only: the ISO readout turns amber `ISO MAX` when analogue gain is at the sensor's maximum. The photo is graded as today; normalisation treats it like any dark frame. |
| Persistence | Memory only; starts at `A` on every start, like EV. |
| Backends | Picamera2 and `FakeCamera`. On V4L2 the buttons are not drawn and taps on their area do nothing. |
| Out of scope | Full manual ISO, aperture, bulb exposures over 1 s, a shutter control on the Stick, carrying underexposure through the grade. |

## 3. Why these controls

The Raspberry Pi AGC holds a fixed exposure time and meets its brightness target by
varying gain alone; this is shutter priority. From libcamera 0.5 (Raspberry Pi OS
Trixie), manual exposure time is selected with `ExposureTimeMode`, and an `ExposureTime`
sent while the mode is Auto is ignored. So the explicit mode control is what makes the
fixed shutter take effect. Older stacks have no `ExposureTimeMode`; there, a non-zero
`ExposureTime` with `AeEnable` true gives the same behaviour, and `ExposureTime=0`
returns to auto. The backend detects which applies from `Picamera2.camera_controls` at
start-up and logs one line naming the path it chose.

EV compensation is an AE target offset. With the shutter fixed, AE meets the offset
through gain, so the EV buttons stay meaningful. The grade's existing EV handling
(`pifilm.pipeline`) is unchanged.

A shutter longer than a frame period needs a longer frame duration. The preview runs at
its configured rate, so `FrameDurationLimits` must allow the chosen time. Otherwise the
sensor clips the exposure to the frame period. The live view slows to match: 1/4 s gives
about 4 fps. That is expected and documented.

## 4. Components

### 4.1 Shutter scale (`pifilm/display/shutter.py`, new, pure)

- `SHUTTER_STEPS_US: tuple[int, ...]`: the standard 1/3-stop series from 1/2000 s to
  1 s, in microseconds, ascending:
  1/2000, 1/1600, 1/1250, 1/1000, 1/800, 1/640, 1/500, 1/400, 1/320, 1/250, 1/200,
  1/160, 1/125, 1/100, 1/80, 1/60, 1/50, 1/40, 1/30, 1/25, 1/20, 1/15, 1/13, 1/10, 1/8,
  1/6, 1/5, 1/4, 0.3, 0.4, 0.5, 0.6, 0.8, 1 s.
- `faster(current_us: int | None, metered_us: float | None) -> int | None`:
  - From `A` (`None`), start at the step nearest the metered exposure. With no
    metered value, start at 1/125.
  - Otherwise, move one step faster. The fastest step stays put.
- `slower(current_us: int | None, metered_us: float | None) -> int | None`:
  - From `A`, start at the step nearest the metered exposure.
  - Otherwise, move one step slower. Slower than 1 s returns `None` (`A`).
- `label(us: int | None) -> str`: `"A"`, `"1/250"`, `"0.3s"`, `"1s"`. ASCII only.
  This matches the meter's existing `format_shutter` style for fractions.

The first tap from `A` lands on the current metered value. So switching into shutter
priority never jumps the exposure, and the next tap moves it.

### 4.2 Picamera2 backend (`pifilm/capture/picamera.py`)

- At start-up, `self._shutter_mode` is `"mode"` if `"ExposureTimeMode"` is in
  `camera.camera_controls`, else `"legacy"`. The backend logs which one it chose.
- `self.max_gain: float | None` is the maximum of `camera_controls["AnalogueGain"]`, if
  present. The meter uses it for `ISO MAX`.
- `self.shutter_us: int | None = None`.
- `set_shutter(us: int | None) -> None` mirrors `set_ev`:
  1. Validate that `us` is `None` or within [100, 1_000_000].
  2. Under `self._request_lock`, apply the controls live.
  3. Write the same controls into both `_still_config` and `_preview_config` through
     `_set_config_control`, so the next `switch_mode_and_capture_request` keeps them.
  4. On success set `self.shutter_us`. On a failure, raise `CameraError`.

  Controls in `"mode"`:
  - Fixed: `ExposureTimeMode=Manual`, `ExposureTime=us`,
    `FrameDurationLimits=(min_default, max(max_default, us + margin))`.
  - Auto: `ExposureTimeMode=Auto`, and restore the configured default
    `FrameDurationLimits`.

  Controls in `"legacy"`: `ExposureTime=us` or `0`, with the same frame-duration
  handling.

  `min_default`/`max_default` are the `FrameDurationLimits` each configuration was
  built with. If a configuration has none, use the camera's control default. The margin
  is 1000 µs.
- `FakeCamera` (`camera.py`) gets `shutter_us = None`, `max_gain = None` and a
  `set_shutter` that records the value. This mirrors its `set_ev`.
- `V4L2Camera` has no `set_shutter`. The viewfinder checks `callable(getattr(camera,
  "set_shutter", None))` to decide whether the buttons exist.

### 4.3 Meter (`pifilm/display/meter.py`)

- `MeterReading` gains `shutter_fixed: bool = False` and `iso_max: bool = False`.
  Both have defaults, so existing constructions still work.
- `compute_reading(..., *, shutter_us: int | None = None, max_gain: float | None = None)`:
  - `shutter_fixed` is `shutter_us is not None`.
  - `iso_max` is `max_gain is not None and AnalogueGain >= 0.98 * max_gain`.
  - The shutter text stays the metered exposure from metadata, which is what the
    sensor really did. When fixed, the UI prefixes it with `S`.

### 4.4 Touch UI (`pifilm/display/ui.py`)

- Constants: `SHUTTER_PLUS_BOX = (264, 26, 312, 60)` and
  `SHUTTER_MINUS_BOX = (264, 144, 312, 178)`.
  - With `HIT_MARGIN` (6), `+` spans y 20–66 and `−` spans y 138–184.
  - The shutter button's hit circle spans y 68–136 (centre 102, radius 28 + 6).
  - The bar's hit zone starts at `BAR_TOP - HIT_MARGIN` = 198.
  - So none of these overlap. The battery badge (y 4–20) is not a touch target.
- `Action.SHUTTER_FASTER` and `Action.SHUTTER_SLOWER`. `hit()` checks the shutter circle
  first, then these boxes, then the other controls. The drawing and `hit()` read the
  same constants.
- `render_live(frame, reading, double=None, shutter_buttons=False)`:
  - When `shutter_buttons` is true, draw both buttons with the same style as the EV
    buttons: a translucent fill, a white outline, and `+`/`-` glyphs.
  - Readout line 1 is `S 1/250` when `reading.shutter_fixed`, else `1/250`.
  - When `reading.iso_max`, the ISO text is `ISO MAX` in `AMBER`.
- The review caption is unchanged. It shows the metered shutter and ISO of the shot.

### 4.5 Viewfinder (`pifilm/display/viewfinder.py`)

- `self._shutter_ok = callable(getattr(camera, "set_shutter", None))`.
- LIVE tap `SHUTTER_FASTER` or `SHUTTER_SLOWER`, only if `_shutter_ok`:
  1. Compute the new value with `shutter.faster` or `slower`, from the camera's
     current `shutter_us` and the last live frame's metered `ExposureTime`.
  2. Call `camera.set_shutter`, logging a `CameraError` as `_set_ev` does.
  3. Like EV taps, this also works while a job is processing. The camera lock
     serialises the change against a capture.
- Pass `shutter_buttons=self._shutter_ok`, `shutter_us` and `max_gain` into the render
  and meter calls.

### 4.6 Capture record (`pifilm/capture/app.py`)

- `CaptureSession.capture()` reads `shutter_us = getattr(self.camera, "shutter_us",
  None)` before the frame, as it reads `ev`.
- Every record (single, and both double-exposure records) gains
  `"shutter_us": <int|null>`. The real exposure and gain are already in
  `camera_metadata`.
- `double.shutter_us` is not added. Each exposure's own record carries it.

## 5. Failure handling

| Case | Result |
| --- | --- |
| `set_shutter` raises (control rejected) | Logged as `camera: ...`. The value is unchanged; the readout shows the old state. |
| libcamera without `ExposureTimeMode` | Legacy path; one start-up log line says so. Hardware acceptance checks the shot's `ExposureTime` matches. |
| Gain saturates | `ISO MAX` in amber on the live view; the photo is graded as today. |
| V4L2 camera | No buttons; the region is not a touch target. |
| Restart | `A` again. |
| Tap while the screen is off | Wakes only, per the idle rules. |

## 6. Testing

- **`tests/test_display_shutter.py`:**
  - The series is ascending, runs from 1/2000 to 1 s, and has 34 steps.
  - `faster` and `slower` from `A` land on the nearest metered step. With no metered
    value they land on 1/125.
  - `faster` stays put at 1/2000.
  - `slower` past 1 s gives `None`.
  - Labels are ASCII.
- **`tests/test_picamera.py`**, using the existing fake-Picamera2 pattern:
  - Mode path: the controls dict has `ExposureTimeMode` Manual, `ExposureTime` and
    widened `FrameDurationLimits`, and both configurations are updated.
  - Legacy path: when `ExposureTimeMode` is absent, `ExposureTime` is set and `0`
    restores auto.
  - Range validation, and `max_gain` read from `camera_controls`.
- **`tests/test_display_meter.py`:**
  - The `iso_max` threshold at 0.98 × max, and none when `max_gain` is `None`.
  - `shutter_fixed` is set.
- **`tests/test_display_ui.py`:**
  - The buttons are drawn only when asked.
  - `hit()` returns each action inside its box plus the margin.
  - The shutter circle still wins at its centre.
  - There is no overlap with the shutter circle, the bar, `DOUBLE_BOX` or the focus bar.
  - `S` prefix and amber `ISO MAX` pixels.
- **`tests/test_viewfinder.py`:**
  - Taps step the `FakeCamera` shutter.
  - The first tap from `A` uses the metered exposure.
  - A camera without `set_shutter` draws no buttons and ignores the taps.
- **`tests/test_app.py`:** `shutter_us` appears in the record, `null` and fixed.

## 7. Docs

- `docs/lcd-viewfinder.md`:
  - §5: rows for the two buttons, `S` prefix and `ISO MAX`.
  - A "Shutter priority" subsection: how to use it, the slower live view at long
    shutters, `A`, and what `ISO MAX` means.
  - §7: acceptance items.
    - 1/1000 s in a dim room gives amber `ISO MAX` and a darker frame.
    - 1/4 s on a moving subject blurs and the live view drops to about 4 fps.
    - The shot's `camera_metadata.ExposureTime` matches the chosen value.
    - The journal names the control path used.
- `docs/how-it-works.md`: one paragraph under capture on shutter priority and why EV
  still works.
- `CLAUDE.md`: note `pifilm/display/shutter.py` and `set_shutter` in the Display and
  Capture sections.
