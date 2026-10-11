# Manual Controls and SLR-Finder Viewfinder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace exposure compensation with manual ISO, drive shutter/ISO/WB from one select-and-adjust control set, and redraw the live screen in the approved "SLR finder" style with touch-down feedback.

**Architecture:** Pure step scales (`stepscale.py`, `iso.py`, `whitebalance.py`, existing `shutter.py`) decide values; `Picamera2Camera.set_iso` mirrors `set_shutter`; a new pure module `liveview.py` owns the live screen's geometry, drawing and hit-testing; `ViewfinderLoop` holds the selection and pressed state and wires taps to the scales. The old live-screen code in `ui.py` and all EV code are deleted in the task that switches the loop over, so every task leaves the suite green.

**Tech Stack:** Python 3.12, Pillow, NumPy, Picamera2/libcamera (lazy), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-11-manual-controls-ui-design.md` (read it; the mock-up is `docs/superpowers/mockups/2026-10-11-manual-controls-ui.png`).

## Global Constraints

- Always `.venv/bin/pytest` and `.venv/bin/ruff check .` (ruff: E, F, I, B, UP; line length 100). Finish every task with `.venv/bin/pytest -q -m 'not slow'` and ruff clean.
- No hardware imports at module top level; `pifilm/display/*` scale and drawing modules are pure.
- Match the surrounding code: module docstrings carry the rationale, comments explain why, tests are named as sentences.
- ASCII only in text drawn on the screen (use `-`, not a typographic minus).
- Never touch `.env`; never touch the untracked directory named `" "` at the repo root.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Do not change `Pipeline.process`, `pifilm/capture/batch.py`, the review/processing/message renderers' appearance, the Stick firmware, or `button.py`.
- `captures.jsonl` keeps `ev_comp` (now always `0.0`).

## Review Focus

- A control the camera cannot change (FakeCamera patched without a setter; `wb_adjustable = False`) must be unselectable and must not crash `+`/`-`.
- `+`/`-` with a camera that rejects the control (`CameraError`): logged as `camera: ...`, loop stays LIVE, value unchanged.
- A finger held on `+` across many frames fires exactly one step (on release), while the key stays lit.
- Metered values that are missing, zero, NaN or non-numeric (no metadata on the first frame) must fall back to the defaults, not raise.
- Scale 1 (the 320x240 fallback panel) must render and hit-test with the same geometry as scale 2.

---

### Task 1: Step scales for ISO and white balance

**Files:**
- Create: `pifilm/display/stepscale.py`, `pifilm/display/iso.py`
- Modify: `pifilm/display/whitebalance.py` (add functions; keep `next_step` and the old `label` behaviour working until Task 4 removes their callers)
- Test: `tests/test_display_stepscale.py`, `tests/test_display_iso.py`, `tests/test_display_whitebalance.py`

**Interfaces:**
- Produces, `stepscale.py` (pure, stdlib only):
  - `nearest(steps: Sequence[int], value: float) -> int`: the step with the smallest `abs(log2(step) - log2(value))`; ties go to the lower step.
  - `is_auto(value: float | None) -> bool`: `None`, non-finite, or `<= 0`. A `bool` or a non-number also counts as auto.
  - `up(steps, current, metered, default) -> int`: from auto, `nearest(steps, metered)` when `metered` is a usable positive finite number, else `default`. Otherwise the smallest step strictly greater than `current`, or `steps[-1]` when there is none.
  - `down(steps, current, metered, default) -> int | None`: from auto, the same start value as `up`. Otherwise the largest step strictly less than `current`, or `None` (auto) when there is none.
  - `steps` is ascending; `default` is one of the steps.
- Produces, `iso.py`: `ISO_STEPS = (100, 200, 400, 800, 1600)`, `DEFAULT_START = 400`, `higher(current: int | None, metered: float | None) -> int`, `lower(current, metered) -> int | None`, `label(iso: int) -> str` (`"400"`).
- Produces, `whitebalance.py` additions: `DEFAULT_START_K = 5600`, `warmer(current_k, metered_k) -> int`, `cooler(current_k, metered_k) -> int | None`, `value_label(kelvin: float) -> str` (`"5600K"`; rounds a metered value to the nearest 100 K, e.g. `4249 -> "4200K"`).

- [ ] **Step 1: Write the failing tests.** Cover, for `stepscale`: nearest in log2 (e.g. steps `(100, 200, 400)`: `140 -> 100`, `150 -> 200`, `283 -> 400` because log2 midpoint of 200/400 is 282.8; values below/above the ends clamp); `up`/`down` from auto with a metered value, without one (`None`, `0`, `nan`, `inf`, `"x"`, `True`), from a step, from between steps, from above the top, from below the bottom; `down` at the bottom returns `None`; `up` at the top stays. For `iso`: full walk `None -> higher (metered 380) -> 400 -> 800 -> 1600 -> 1600`, `lower` from 100 is `None`, default 400 with no metered value, `label`. For `whitebalance`: `warmer`/`cooler` walks, first step from auto with metered `4249` is `4000`, default 5600, `value_label(4249) == "4200K"`, `value_label(6500) == "6500K"`; the existing tests for `next_step`/`label` still pass.
- [ ] **Step 2: Run them and see them fail** (`.venv/bin/pytest -q tests/test_display_stepscale.py tests/test_display_iso.py tests/test_display_whitebalance.py`; expected: import errors).
- [ ] **Step 3: Implement.** `iso.py` and the new `whitebalance` functions are thin wrappers over `stepscale`. Module docstrings state the semantics above and why the start is the nearest step in log2 (a stop is what the eye sees; same reasoning as `shutter.py`).
- [ ] **Step 4: Full suite + ruff, then commit** (`display: step scales for ISO and warmer/cooler white balance`).

---

### Task 2: Manual ISO on the cameras and in the capture record

**Files:**
- Modify: `pifilm/capture/picamera.py`, `pifilm/capture/camera.py` (`FakeCamera`), `pifilm/capture/app.py` (`CaptureSession.capture`, `_capture_double`)
- Test: `tests/test_picamera.py`, `tests/test_app.py`

**Interfaces:**
- Produces: `Picamera2Camera.set_iso(iso: int | None) -> None` and attribute `iso: int | None` (None = auto; initial None). `FakeCamera.set_iso` and `FakeCamera.iso` mirror them (record only).
- Produces: capture records carry `"iso": <fixed ISO or None>` in the single-shot record and in both double-exposure records (put it in `camera_fields` beside `shutter_us` / `colour_temperature_k`), read from the camera *before* the frame, exactly as `shutter_us` and `wb_k` are.
- Do NOT remove `set_ev`, the `ev` parameter or `--ev` in this task (Task 4 does).

**Requirements for `set_iso`** (read `set_shutter`, `set_colour_temperature` and `_shutter_controls` first and mirror their structure, locking, error wrapping and docstring style):
- ISO 100 is `AnalogueGain` 1.0: `gain = iso / 100`.
- Validation: `None`, or a whole number (integral float accepted like `_shutter_us` does; bool, string, fractional refused) with `100 <= iso <= ISO_MAX` where module constant `ISO_RANGE = (100, 1600)`; additionally, when `self.max_gain` is known, `iso / 100 <= max_gain`. Anything else is a `CameraError`; nothing changes.
- Control path, chosen once in the constructor like `_shutter_mode`: `"mode"` when `"AnalogueGainMode" in camera.camera_controls`, else `"legacy"`.
  - mode, fixed: `{"AnalogueGainMode": AnalogueGainModeEnum.Manual, "AnalogueGain": gain}`; auto: `{"AnalogueGainMode": AnalogueGainModeEnum.Auto}` and `AnalogueGain` is dropped from both configurations (`_drop_config_control`), for the same reason `ExposureTime` is dropped on a return to auto.
  - legacy, fixed: `{"AnalogueGain": gain}`; auto: `{"AnalogueGain": 0.0}`.
  - The enum is `libcamera.controls.AnalogueGainModeEnum`, imported lazily like `ExposureTimeModeEnum`. The test module's fake libcamera needs the matching fake enum.
- Controls go to the running camera (`set_controls`) under `self._request_lock`, then into both `_still_config` and `_preview_config` (`_set_config_control`), then `self.iso` is updated. If `set_controls` raises: `CameraError("Picamera2 failed to set the ISO: ...")`, configurations and `self.iso` unchanged.
- Docstring records the measurement from the spec (fixed gain holds with automatic exposure time; both can be fixed; measured 2026-10-11 on the IMX477, libcamera 0.7.2).

- [ ] **Step 1: Write the failing tests**, mirroring the existing shutter tests in `tests/test_picamera.py` (`test_shutter_priority_uses_exposure_time_mode_when_available`, `..._falls_back_...`, `test_shutter_outside_the_range_is_refused`, `test_a_rejected_shutter_changes_nothing`): mode path sets both controls live and in both configs and auto drops `AnalogueGain`; legacy path; refused values (`50`, `3200`, `400.5`, `True`, `"400"`, and `1600` when `analogue_gain=(1.0, 8.0, 1.0)`); rejected control changes nothing; closed camera raises. Add a `with_analogue_gain_mode` option to `install_picamera` (default False so existing tests are unchanged). In `tests/test_app.py`: `test_record_carries_the_iso_setting` and the double-exposure equivalent, mirroring the white-balance ones at the end of the file.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Full suite + ruff, then commit** (`capture: manual ISO on Picamera2 and the fake camera, recorded per capture`).

---

### Task 3: The SLR-finder live screen (`liveview.py`)

**Files:**
- Create: `pifilm/display/liveview.py`, `tests/test_display_liveview.py`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]`: add `"data/fonts/*.ttf"`, `"data/fonts/OFL.txt"`)
- Already present (do not re-download): `pifilm/data/fonts/BarlowCondensed-SemiBold.ttf`, `BarlowCondensed-Medium.ttf`, `OFL.txt`
- Do NOT modify `ui.py` or `viewfinder.py` in this task; the loop keeps using the old screen until Task 4.

**Interfaces (Task 4 consumes these exactly):**

```python
class Action(Enum):
    NONE = "none"
    SHUTTER = "shutter"
    DOUBLE_TOGGLE = "double_toggle"
    SELECT_SHUTTER = "select_shutter"
    SELECT_ISO = "select_iso"
    SELECT_WB = "select_wb"
    PLUS = "plus"
    MINUS = "minus"

SELECT_ACTIONS = (Action.SELECT_SHUTTER, Action.SELECT_ISO, Action.SELECT_WB)  # index = cell

@dataclass(frozen=True)
class Cell:
    label: str            # "SHUTTER", "ISO", "WB"
    value: str            # "1/125", "400", "5200K", "MAX", "-"
    auto: bool = True     # draws "AUTO" beside the label
    enabled: bool = True  # False: drawn grey, cannot be selected

@dataclass(frozen=True)
class LiveState:
    cells: tuple[Cell, Cell, Cell]
    selected: int | None = 0                 # index into cells; None when nothing is selectable
    pressed: Action | None = None            # PLUS, MINUS or SHUTTER while lit
    double: tuple[bool, int] | None = None   # (enabled, exposures taken); None hides the 2x toggle

def render_live(frame_rgb: np.ndarray, reading: MeterReading, state: LiveState,
                *, scale: int = 1) -> Image.Image   # RGB, (320*scale, 240*scale)
def hit(x: int, y: int, scale: int = 1) -> Action
```

`render_live` reads from `reading` only: `deviation_ev`, `lux`, `clip_pct`, `focus`, `focus_peak`, `battery_percent`, `external_power` (do not read `ev_comp`, `shutter`, `iso`; Task 4 removes `ev_comp`).

**Geometry (base units; multiply by `scale`). These are constants in `liveview.py`:**

| Element | Constant | Value |
| --- | --- | --- |
| Screen | `WIDTH, HEIGHT` | 320, 240 |
| Info strip (solid black) | `BAR_TOP` | 208 (to 240) |
| Cell i (0..2) | `CELL_X0, CELL_PITCH` | text starts at x = 11 + 61*i; label centre y 215; value centre y 229; selected underline y 237..238, x 11+61i .. 57+61i |
| Cell hit | | y >= 204 and 8 + 61*i <= x < 69 + 61*i |
| Meter dots | `METER_X0, METER_X1, METER_Y` | 205, 306, 226; 13 dots, dot j at x0 + j*(x1-x0)/12 for -3.0 + 0.5*j stops; radii 1.5 (whole stops), 1.0 (half stops), 2.25 (lit) |
| Meter `-` / `+` glyphs | | centred 7 units left of x0 / right of x1, y 226 |
| Lux/clip text | | `"12 LX   CLIP 0%"` (lux `-` when None), centred on (255.5, 215) |
| Shutter button | `SHUTTER_CENTRE, SHUTTER_RADIUS` | (287, 39), ring 28, inner disc 21.5; hit radius 34 |
| `+` key | `PLUS_CENTRE, KEY_RADIUS` | (287, 125), 17; hit radius 21 |
| `-` key | `MINUS_CENTRE` | (287, 168), 17; hit radius 21 |
| `2x` toggle | `DOUBLE_BOX` | (6, 5, 30, 16), corner radius 5.5; hit box grown by 6, clipped to the screen |
| Battery tag | | box (35, 5, 35 + text width + 12, 16), same radius; absent when `battery_percent` is None |
| Focus gauge | | track x 6..9, y 22..198; filled from the bottom to `focus`; peak tick 4..11 wide, 1.5 tall at `focus_peak`; absent when `focus` is None |

Hit priority: SHUTTER, PLUS, MINUS, DOUBLE_TOGGLE, cells, else NONE. The hit regions do not overlap; a test must prove it by scanning every pixel of each region.

**Appearance (see the mock-up):**
- Colours: `AMBER = (255, 176, 0)`, `DIM_AMBER = (150, 106, 8)`, `WHITE = (245, 245, 240)`, `GREY = (125, 125, 120)`, unlit dot `(95, 95, 90)`.
- Fonts (base sizes, multiplied by scale): cell labels and lux/clip text Medium 5 with about 0.8 units of letter spacing; cell values SemiBold 18; `2x` and battery text SemiBold 6.5; `+`/`-` glyphs SemiBold 15 (ASCII `+` and `-`). Load with `importlib.resources.files("pifilm") / "data" / "fonts" / name`, cached by (name, pixel size); if the file cannot be loaded, fall back to `ImageFont.load_default(size)` rather than raising.
- Cells: selected = label AMBER, `AUTO` WHITE, value AMBER, amber underline. Unselected = label GREY, `AUTO` GREY, value DIM_AMBER. Disabled = everything GREY, no underline.
- Keys: translucent black fill `(0, 0, 0, 110)`, amber outline 0.75 wide, WHITE glyph; pressed = AMBER fill, black glyph.
- Shutter: fill `(0, 0, 0, 70)`, amber ring 1.25 wide, inner disc `(255, 255, 255, 215)`; pressed = inner disc AMBER.
- `2x` off: translucent black, amber outline, WHITE `2x`. On: AMBER fill, black `2x 0/2` / `2x 1/2` (widen the box to the text when on).
- Battery tag: translucent black, amber outline, WHITE `AC 88%` / `88%`.
- Focus gauge: track `(0, 0, 0, 110)`, fill WHITE, peak tick AMBER.
- Anti-aliasing: draw the overlay at `scale * 2` on an RGBA layer and downsample with `Image.LANCZOS` before compositing over the letterboxed frame. The static part (everything determined by `state`, the battery text and `scale`) must be cached (`functools.lru_cache`, small maxsize) so a frame only draws the meter dots, lux/clip text and focus gauge on top; the viewfinder runs this ten times a second on a Pi 4. Reuse `ui._letterbox` for the frame.

- [ ] **Step 1: Write the failing tests** in `tests/test_display_liveview.py` (use `tests/test_display_ui.py` as the style reference, including its `_reading()` helper pattern):
  - output size and mode at scale 1 and 2;
  - each hit region returns its action at its centre, at scale 1 and 2, and the regions are pairwise disjoint (scan);
  - the gap between the shutter's hit circle and the `+` key's is at least 20 base units;
  - a point in the image and a point in the meter area return `Action.NONE`;
  - the strip is solid black where nothing is drawn (sample a pixel), over a white frame;
  - the selected cell contains AMBER pixels and its underline row is AMBER; an unselected cell has no pure-AMBER pixels; a disabled cell has neither AMBER nor DIM_AMBER;
  - `pressed=Action.PLUS` makes the `+` key's centre region mostly AMBER and leaves `-` unchanged; likewise MINUS and SHUTTER;
  - the lit meter dot moves: deviation -3, 0, +3 light dots at x0, the middle, x1; +9 clamps to x1; NaN lights the middle and does not raise;
  - `double=None` draws nothing in `DOUBLE_BOX`; `(True, 1)` draws amber there;
  - `battery_percent=None` draws no tag;
  - `focus=None` leaves the gauge area untouched;
  - rendering twice with the same state returns equal images (the cache does not leak dynamic parts), and changing only `deviation_ev` changes only the meter area;
  - the bundled font files exist in the package and load.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement `liveview.py`** with a module docstring that states the design intent (full-screen image, solid strip, one select-and-adjust control set, shutter kept away from the keys and why) and the caching rationale.
- [ ] **Step 4: Render a check image** and look at it: `.venv/bin/python -c "..."` writing a scale-2 PNG with a synthetic frame to the scratch directory you were given; compare with the mock-up and fix anything clipped, overlapping or misaligned. Report what you saw.
- [ ] **Step 5: Full suite + ruff, then commit** (`display: the SLR-finder live screen (liveview.py) and its bundled font`). Include the font files and `pyproject.toml`.

---

### Task 4: Switch the viewfinder over; touch-down feedback; remove EV

**Files:**
- Modify: `pifilm/display/viewfinder.py`, `pifilm/display/ui.py`, `pifilm/display/meter.py`, `pifilm/display/whitebalance.py`, `pifilm/capture/app.py`, `pifilm/capture/picamera.py`, `pifilm/capture/camera.py`
- Test: `tests/test_viewfinder.py`, `tests/test_display_ui.py`, `tests/test_display_meter.py` (if present), `tests/test_display_whitebalance.py`, `tests/test_app.py`, `tests/test_picamera.py`

**Interfaces:**
- Consumes: `liveview.Action`, `SELECT_ACTIONS`, `Cell`, `LiveState`, `render_live`, `hit` (Task 3); `iso.higher/lower/label`, `whitebalance.warmer/cooler/value_label`, existing `shutter.faster/slower` and its label helper (Task 1); `camera.set_iso`, `camera.iso` (Task 2).
- Produces: `ViewfinderLoop.selected: int | None`.

**Viewfinder behaviour:**
- Availability per cell, decided once in `__init__`: shutter = `callable(camera.set_shutter)`; ISO = `callable(camera.set_iso)`; WB = `callable(camera.set_colour_temperature)` and `camera.wb_adjustable is not False`. `selected` starts at the first available cell (None when none is).
- A tap (on release, as now) in LIVE: `SHUTTER` submits; `DOUBLE_TOGGLE` as now; `SELECT_*` selects that cell if available; `PLUS`/`MINUS` step the selected control: shutter `+` = `shutter.faster`, `-` = `shutter.slower`; ISO `+` = `iso.higher`, `-` = `iso.lower`; WB `+` = `whitebalance.warmer`, `-` = `whitebalance.cooler`. Metered values: exposure time as now (`_metered_us`); ISO from the last reading's `iso`; colour temperature from the last frame's `metadata["ColourTemperature"]`. A `CameraError` from a setter is logged `camera: ...` and changes nothing.
- Cell contents each frame: SHUTTER fixed -> the scale's label for `camera.shutter_us`, auto -> `reading.shutter` or `-`; ISO fixed -> `iso.label(camera.iso)`, auto -> `MAX` when `reading.iso_max` else `str(reading.iso)` or `-`; WB fixed -> `whitebalance.value_label(camera.colour_temperature_k)`, auto -> `value_label(metered)` or `-`. `auto` is `value is None` on the camera; `enabled` is the availability.
- Pressed state: `_poll_tap` records the current finger position (first point, or None when no finger is down) on every poll, in an attribute the step reads (a plain assignment is the only cross-thread traffic; say so in a comment). Each step maps it through `liveview.hit`; if it is PLUS, MINUS or SHUTTER the key is lit. A tap that fires PLUS/MINUS/SHUTTER also lights its key until `now + PRESS_FLASH_SECONDS` (0.15), so a tap shorter than a frame is still seen. The lit key is passed as `LiveState.pressed`.
- A finger down on a key does not repeat: one step per release, as the tap detector already guarantees; add a test.
- The screen-off rule is unchanged: a tap on a dark screen only wakes it, and nothing is lit.

**EV removal:**
- `viewfinder.py`: `EV_STEP`, `EV_LIMIT`, `ev_comp`, `_set_ev`, the EV actions.
- `meter.py`: remove `ev_comp` from `MeterReading` and from `compute_reading`'s parameters; keep `format_ev` only if something else still uses it.
- Review caption (`_review_image`): `"{shutter}  ISO {iso}"` (with the existing dash handling), no EV part.
- `ui.py`: delete the old live screen (`render_live`, `hit`, `Action`, the EV/shutter/WB/double box constants, `_draw_double_badge`, `_draw_wb_badge`, `_draw_focus_bar` and constants only it used) and keep `render_processing`, `render_review`, `render_message`, `_letterbox`, `_ScaledDraw`, `_font` and what they need. `tests/test_display_ui.py` loses the tests of the deleted code; their intent (hit regions, drawn-only-when-asked) is already covered by `tests/test_display_liveview.py`.
- `whitebalance.py`: remove `next_step` and the old `label`; update its tests.
- `app.py`: remove `--ev` and its entry in `_reject_picamera2_only_flags` (and its docstring mention); stop passing `ev=`.
- `picamera.py`: remove the `ev` constructor parameter and `set_ev`; keep `self.ev = 0.0` (comment: kept so `CaptureSession` logs `ev_comp: 0.0` and old records keep their meaning) and keep sending `ExposureValue: 0.0` in `_camera_controls`.
- `camera.py` (`FakeCamera`): remove `set_ev`; keep `self.ev = 0.0`.
- `CaptureSession` is NOT changed.

- [ ] **Step 1: Write the failing tests** in `tests/test_viewfinder.py` (reuse `_loop`, `_tap`, `_centre`, `controller`, `ScriptedTouch`; a key's tap point is its centre constant from `liveview`):
  - SHUTTER is selected at start; tapping the ISO cell then `+` changes `camera.iso` and not `camera.shutter_us`; same for WB and shutter;
  - first `+` from auto lands on the metered step for each control (FakeCamera metadata gives ExposureTime 4000; add AnalogueGain/ColourTemperature to the metadata where needed);
  - `-` below the bottom of ISO and WB returns to None; `+` at the top stays;
  - an unavailable control cannot be selected and `+`/`-` do nothing when nothing is selectable (parametrize: no setter, `wb_adjustable = False`);
  - a rejected control is logged and not fatal, for each of the three;
  - `render_live` receives a `LiveState` whose cells show fixed values and `auto=False` after a step, and `MAX` when the reading says `iso_max`;
  - a finger held on `+` for several steps passes `pressed=Action.PLUS` each frame and changes the value exactly once, on release;
  - a quick tap passes `pressed` on the next frame and clears after `PRESS_FLASH_SECONDS`;
  - a tap on a dark screen wakes it and lights nothing;
  - the shutter tap still submits; the 2x toggle still toggles; review caption has no `EV`.
  Rewrite or delete existing tests that exercised EV, the old shutter boxes or the WB pill; do not leave skipped tests behind. In `tests/test_app.py`: `--ev 1` is now an argparse error (exit 2); records still carry `ev_comp == 0.0`. In `tests/test_picamera.py`: remove `set_ev`/`ev=` tests; assert `ExposureValue` is still `0.0` in the first controls.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Smoke-run** `.venv/bin/pifilm-capture --fake --display fake --no-preview --out <scratch dir>` for a few seconds (stop it with Ctrl-C / a timeout you implement with a background process and `kill`), confirm it starts, writes the PNG, and exits 0; look at the PNG.
- [ ] **Step 5: `grep -rn "ev_comp\|set_ev\|EV_PLUS\|EV_MINUS\|WB_NEXT\|SHUTTER_FASTER\|next_step" pifilm tests`** and justify every remaining hit in your report (only `ev_comp` in `CaptureSession`, the pipeline/batch code and their tests should remain).
- [ ] **Step 6: Full suite + ruff, then commit** (`viewfinder: select-and-adjust shutter/ISO/WB on the new live screen; EV removed`).

---

### Task 5: Documentation

**Files:**
- Modify: `docs/lcd-viewfinder.md`, `CLAUDE.md`, `README.md` (only if it mentions EV buttons, the WB pill or shutter `+`/`-`), `docs/setup.md` (same condition), `deploy/pifilm-capture.service.example` (only if it passes `--ev`)

**Requirements:**
- `docs/lcd-viewfinder.md` section 5 ("The screen"): rewrite the region table for the new screen (cells, `+`/`-`, shutter at top right, meter dots, `2x`, battery, focus gauge, review caption without EV). Replace the "Shutter priority" and "Manual white balance" subsections with one "Manual controls" subsection covering selection, the three scales, auto, first-step-from-auto, full manual (the meter is then the only guide), `MAX`, `--white-balance`, the record fields (`shutter_us`, `iso`, `colour_temperature_k`), and that EV compensation was removed on 2026-10-11 (`ev_comp` stays in records as 0; `--ev` no longer exists). Keep the still-true shutter notes (slow live view at long speeds, taps once per frame, the journal line naming the control path, the `--ae-lock` caveat) and the measured WB gains.
- Record the ISO measurements table from the spec.
- Hardware acceptance: replace the EV, shutter-priority and WB items with the spec's nine acceptance steps, marked "not yet run"; keep unrelated items and their recorded results. Do not renumber references from other documents without updating them (`grep -rn "lcd-viewfinder.md#" docs README.md CLAUDE.md`).
- Mention the font and its licence (`pifilm/data/fonts/OFL.txt`).
- `CLAUDE.md`: Display section (new `liveview.py`, `stepscale.py`, `iso.py`; `ui.py` now only the review/processing/message screens), Capture section (`set_iso`; no EV), and the commands block if it shows `--ev`.
- Check every statement against the code on the branch; do not describe anything that is not implemented.

- [ ] **Step 1: Make the edits.**
- [ ] **Step 2: `grep -rn -i "ev +\|ev -\|EV button\|--ev\b\|WB pill\|exposure compensation" docs README.md CLAUDE.md deploy`** and resolve every stale hit (dated experiment write-ups and `docs/superpowers/` are history; leave them).
- [ ] **Step 3: Full suite + ruff, then commit** (`docs: manual controls and the SLR-finder viewfinder`).
