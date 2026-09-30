# Double exposure — design

**Date:** 2026-09-30
**Status:** approved in discussion; awaiting spec review.
**Depends on:** the LCD viewfinder (`docs/lcd-viewfinder.md`) for the toggle. The capture
side works without a panel (Stick-only), but there is then no way to switch the mode on.

## 1. Goal

A touch toggle on the LCD puts the camera into double-exposure mode. In that mode the next
two shots are combined into one picture the way colour negative film combines two exposures
on one frame, and the screen shows the composite after the second. A badge shows progress:
`0/2`, `1/2`. The mode stays on, cycling 0/2 → 1/2 → composite → 0/2, until toggled off.

## 2. Decisions taken in brainstorming

| Topic | Decision |
| --- | --- |
| Grade before or after compositing | **Before grading.** The two ungraded frames are added as light, then the composite goes through `Pipeline.process` once. Film is exposed twice and developed once; `normalize → LUT → grain` is our development. |
| Composite maths | Linear light, one stop off each: `L = 0.5·L₁ + 0.5·L₂`, back to sRGB. |
| EV | Each frame is added as shot, its EV compensation included, so a frame shot at −1 EV contributes less light. The composite is graded with `ev=0`. |
| Alignment | None. Film does not align; hand-held offset is part of the look. |
| After exposure 1 | No review screen. A brief `Exposure 1/2` message, then straight back to live (option A). |
| After exposure 2 | The normal review screen with the graded composite. |
| Where state lives | `CaptureSession` owns the mode and the pending frame. Controller, remote protocol and firmware are unchanged in shape. |
| Triggers | Every trigger counts (LCD shutter, Stick, SPACE where it exists). The Stick shows a generated `Exposure 1/2` card after the first and the composite after the second. |
| Persistence | Mode and pending frame are memory only. A restart starts with the mode off at 0/2. |
| Out of scope | More than two exposures, per-exposure weight controls, a ghost overlay of exposure 1 on the live view, a Stick-side toggle, firmware changes. |

## 3. Why the composite is made before grading

Negative film records exposure `H` (light × time) at each point, and two exposures add:
`H = H₁ + H₂`. Dark areas add almost nothing, so where one frame is dark the other shows
at full strength (the silhouette filled with trees); where both are bright, detail
separation is lost. Colours add as light (red + green → yellow), not as paint. The
film's characteristic curve then acts once on the sum.

Mapped onto the pipeline:

- Adding sRGB codes, or blending after grading, is not addition of light. Grading each
  frame first would apply the LUT's tone curve twice and add values after the curve,
  which reads as a flat 50/50 blend, and would add grain twice.
- The capture camera auto-exposes each frame to a full exposure, so a straight sum would
  be one stop over and clip. Taking one stop off each (×0.5 in linear light) is the
  standard film starting point: where the two frames match, the total is one normal
  exposure. Because it is a mean it can never exceed 1.0, so nothing clips before the
  LUT's shoulder gets to act.
- Normalisation, LUT and grain run once, on the composite, exactly as for a single shot.
  Normalising each frame separately and again after compositing would break the
  "normalising twice is unsupported" rule in `pifilm/normalize.py`.
- Normalisation re-targets the composite's overall exposure, so the absolute EV of the
  pair is not preserved, but the ratio between the two exposures is. That is the
  control a film shooter uses: underexpose one exposure to make it recede.

## 4. Components

### 4.1 `pifilm/double.py` (new, pure NumPy)

```python
COMPOSITE_METHOD = "linear_mean"

def composite(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Two sRGB uint8 frames exposed onto one 'negative'; returns sRGB uint8."""
```

- sRGB → linear via `pifilm.color.srgb_to_linear` (float32), `0.5 * (a + b)`,
  `linear_to_srgb`, round and clip to uint8.
- Raises `ValueError` if either input is not `(H, W, 3)` uint8 or the shapes differ.
- The module docstring carries section 3's rationale.

### 4.2 `CaptureSession` (`pifilm/capture/app.py`)

New state, touched only on the controller's worker thread:

- `double_exposure: bool` (default `False`)
- `_pending: _PendingExposure | None`: the first frame's RGB array, its original's
  filename, its EV.

New method `set_double_exposure(enabled: bool)`. Turning the mode off drops `_pending`.
Turning it on when it is already on is a no-op, and keeps `_pending`.

`capture()` when `double_exposure` is on:

**Exposure 1** (`_pending is None`):
1. Read the frame and save `HHMMSS_original.jpg` (+ `.dng`) exactly as a single shot does.
2. Do not grade. Store the frame in `_pending`.
3. Append a `captures.jsonl` line with the usual camera fields (stream info,
   `camera_metadata`, `ev_comp`, timings) plus `"double": {"index": 1, "of": 2}`, and no
   `pifilm` key.
4. Return a `CaptureResult` whose `pifilm` points to the exposure card (§4.4) and whose
   new field `exposure` is `(1, 2)`.

**Exposure 2** (`_pending` set):
1. Read the frame and save its original (+ `.dng`) as usual.
2. `composite(pending.rgb, frame.rgb)`. Then `Pipeline.process(comp, rng=…, ev=0.0)`
   with a fresh grain seed.
3. Save `<stem>_double_graded.jpg`, where `<stem>` is exposure 2's stem.
4. Append one record: exposure 2's camera fields, plus the pipeline `info`, the
   `grain_seed`, `params_version` and `package_version`, `"pifilm": "<stem>_double_graded.jpg"` and

   ```json
   "double": {"index": 2, "of": 2, "method": "linear_mean",
              "originals": ["<exp1 original>", "<exp2 original>"],
              "ev_comp": [<exp1 ev>, <exp2 ev>]}
   ```

   With the hashes that is enough to regenerate the composite from the two originals.
   The top-level `ev_comp` is `0.0`, as the pipeline reports it.
5. Clear `_pending`. Return a `CaptureResult` with `pifilm` as the composite and
   `exposure=(2, 2)`.

With the mode off, `capture()` is byte-for-byte today's behaviour, and `exposure` is
`None`.

`CaptureResult` gains `exposure: tuple[int, int] | None = None`. Existing constructors
are unaffected.

A small read-only property, `double_state -> tuple[bool, int]` (enabled, exposures
taken), is what the controller snapshots.

### 4.3 `CaptureController` (`pifilm/capture/controller.py`)

- `set_double_exposure(enabled: bool) -> None` puts a toggle message on the existing
  work queue, so it runs on the worker thread between jobs and never races a capture.
  The queue's item type widens from `str | None` to a small union: a request id, a
  toggle message, or `None` for stop.
- The session's `double_state` is read at the end of every job and every toggle. The
  controller caches it under its lock.
- `ControllerSnapshot` gains `double_exposure: bool = False` and
  `exposures_taken: int = 0`. Both have defaults, so existing constructions still work.
- A session without `set_double_exposure` makes the toggle a no-op. This keeps the
  test doubles that implement only `capture()` working.

### 4.4 Exposure card

`render_exposure_card(index, of) -> bytes` lives in `pifilm/capture/thumbnail.py` beside
`fitted_jpeg`. It is a 240×135 JPEG with dimmed colour bars and the text
`Exposure 1/2`. The session writes it once, lazily, into a private temporary directory
(`tempfile.mkdtemp(prefix="pifilm-cards-")`), and never under `out_root`: the Nextcloud
sync uploads everything there. The existing `/image.jpg` path, which serves
`job.result.pifilm` through `fitted_jpeg`, then handles it unchanged, and the Stick
shows it like any photo.

### 4.5 Remote status (`pifilm/capture/remote.py`)

`GET /v1/status` gains `"double_exposure": {"enabled": bool, "taken": int}`. Current
firmware ignores unknown keys. No other endpoint changes.

### 4.6 Touch UI (`pifilm/display/ui.py`)

- New constants for a pill in the top-left corner: `DOUBLE_BOX = (4, 4, 74, 24)`. It
  sits clear of the focus bar's backing (which starts at y 36), the battery badge
  (top-right) and the shutter button.
- `render_live(frame, reading, double: tuple[bool, int] | None = None)`:
  - Off: an outlined pill reading `2x`.
  - On: an amber-filled pill with black text, `2x 0/2` or `2x 1/2`.
  - The label is ASCII only, because the default font has no `×`.
- `Action.DOUBLE_TOGGLE`: `hit()` returns it for taps inside `DOUBLE_BOX` expanded by
  `HIT_MARGIN`, using the same constants as the drawing.
- `render_processing(label)` is reused with `Exposure 1 of 2...` and
  `Developing double exposure...`.

### 4.7 Viewfinder (`pifilm/display/viewfinder.py`)

- **LIVE:** a `DOUBLE_TOGGLE` tap calls
  `controller.set_double_exposure(not snap.double_exposure)`. The badge is drawn from
  the snapshot on every live frame.
- **While a job is active:** the processing label depends on the snapshot. With the mode
  on and 1 taken, it is `Developing double exposure...`. With the mode on and 0 taken,
  it is `Exposure 1 of 2...`. Taps other than the shutter are ignored, as today.
- **On a finished job whose result has `exposure == (1, 2)`:** a new state, `NOTICE`,
  shows `render_message("Exposure 1/2", "frame the second exposure")` for
  `NOTICE_SECONDS = 1.2`, then goes to LIVE. A tap ends it early. It never enters REVIEW.
- **On a finished job whose result has `exposure == (2, 2)`:** REVIEW as today. The
  caption is prefixed with `2x`.
- **A failed job:** handled as today ("Capture failed").

## 5. Failure handling

| Failure | Result |
| --- | --- |
| Camera error on exposure 1 | Job fails. The count stays 0/2. |
| Camera error on exposure 2 (before the frame is read) | Job fails. `_pending` is kept, so the count stays 1/2 and the user retries. |
| Failure after exposure 2's frame is read (composite, grade, save) | Job fails. `_pending` is cleared and the count returns to 0/2. Both originals are on disk. |
| Mode toggled off at 1/2 | `_pending` is dropped. Exposure 1's original remains. |
| Toggle while a job runs | Not possible from the LCD (taps are ignored). A queued toggle from any future caller runs after the job. |
| Service restart | Mode off, 0/2. |
| Shape mismatch between frames | `ValueError` → the job fails, and the "after read" row applies. It cannot happen on one camera configuration; the guard keeps a bad merge out of the files. |

`pifilm-process` already skips `*_graded.*`, which includes `_double_graded.jpg`. It
does not try to rebuild composites. Regenerating one from its record is out of scope.

## 6. Testing

- **`tests/test_double.py`:**
  - black + black → black
  - black + white → the linear half-intensity code (≈188)
  - identical frames → unchanged (±1 code)
  - pure red + pure green → equal R and G, no B (yellow)
  - symmetric in its arguments
  - shape and dtype errors
- **Session (`FakeCamera`):**
  - Off: output identical to today.
  - On:
    - exposure 1 writes an original, no graded file, and a record with `double.index == 1`
    - exposure 2 writes its original, `_double_graded.jpg`, and a record naming both originals with one grain seed
    - the count cycles back to 0
  - Toggling off at 1/2 drops the pending frame.
  - Both failure rows from §5.
- **Controller:**
  - A toggle lands in the snapshot.
  - A toggle queued while a job runs applies after the job.
  - A session without `set_double_exposure` is tolerated.
- **Remote:** the status payload includes `double_exposure`. `/image.jpg` for exposure 1
  serves the card.
- **UI:**
  - The badge is rendered in the off, 0/2 and 1/2 states (pixel checks inside `DOUBLE_BOX`).
  - `hit()` returns `DOUBLE_TOGGLE` inside the box and never overlaps the shutter, EV or focus-bar regions.
- **Viewfinder (fake display, touch and clock):**
  - A toggle tap calls the controller.
  - Exposure 1 leads to NOTICE, then LIVE after 1.2 s, and never REVIEW.
  - Exposure 2 leads to REVIEW with the composite and the `2x` caption.
  - The processing labels match the state.

## 7. Docs

- `docs/lcd-viewfinder.md`: the toggle, badge and flow in the screen-layout section, and
  hardware acceptance items (toggle, 1/2 notice, composite review, Stick card, restart
  resets the mode).
- `docs/how-it-works.md`: a "Double exposure" subsection summarising section 3.
- `CLAUDE.md`: one line under Capture pointing to `pifilm/double.py`.
