# Stabilise: short exposure, shutter delay and multi-frame merge

Date: 2026-10-23. Branch `feat/stabilise`, stacked on `feat/image-controls`
(the IMAGE second menu page). No separate spec: the user's request in the
session is the authority ("do 2, 3, 4 and with the option to select them from
the menu"), with the design agreed in chat (below).

## Goal

Software help against camera shake on the Raspberry Pi HQ Camera (IMX477, no
gyro, no lens stabilisation), each part selectable from a new STABILISE page of
the LCD settings menu:

1. **Short exposure**: auto exposure prefers shorter shutter times and more gain.
2. **Shutter delay**: wait after a press on the LCD or the GPIO button so the
   jolt of the press has passed.
3. **Multi-frame**: take several shorter frames, align them, merge them into
   one picture (moving parts from the reference frame only), then grade.

## Global Constraints

- Read CLAUDE.md and the module docstrings of every file you change; keep the
  docstrings accurate. Python 3.11+, ruff (E, F, I, B, UP; line 100).
- Tests: `../martin-parr-project/.venv/bin/pytest -q -m 'not slow'` from the
  worktree (no venv in the worktree; pytest `pythonpath = ["."]`). Ruff:
  `../martin-parr-project/.venv/bin/ruff check .`. Every commit passes both.
- `cv2` only through `pifilm._cv2.require_cv2()`. Images only through
  `pifilm.imageio`. Hardware imports (libcamera, picamera2) stay lazy.
- Settings live in `pifilm/capture/settings.py` (`settings.json` VERSION 1;
  new keys ignored by older builds with one line). Settings apply on Picamera2
  only; V4L2 and `--fake` keep defaults and the STABILISE entry is inactive.
- Exact setting keys, values and defaults:
  - `short_exposure`: bool, default False.
  - `shutter_delay`: float, one of 0.0, 0.2, 0.5 (seconds), default 0.0.
  - `multi_frame`: int, one of 1, 2, 4, 6, default 1 (1 = off).
  Menu labels: SHORT EXPOSURE (`ON`/`OFF`), SHUTTER DELAY (`OFF`/`0.2 S`/`0.5 S`),
  MULTI-FRAME (`OFF`/`2`/`4`/`6`); a tap steps to the next value and wraps.
- Records gain, right after the image-control fields (`sharpness`, `contrast`,
  `saturation`), on Picamera2 records only: `short_exposure`, `shutter_delay_s`
  (the delay actually waited for this shot: 0.0 for the Stick), and, only for a
  multi-frame photograph, a `stack` object (see Task 5).
- Never read, print or source `.env`. No SSH to the Pi from implementer tasks.
- Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Tasks

### Task 1: settings and short exposure

- `settings.py`: the three keys above (`_valid`: exact type and value set;
  bool is not a number; `_describe`: `short-exposure on`, `shutter-delay 0.2`,
  `multi-frame 4`). Update the docstring's rationale.
- `picamera.py`: `AeExposureMode` Short when `short_exposure` is on, Normal
  when off, in the shared control dict (open) and a runtime setter that writes
  it into both configurations, following `_set_ae_mode` and its
  missing-enum warning. `stream_info` carries `short_exposure`; records copy it
  where the constraints say.
- `app.py` `main`: register the applier (Picamera2 only), pass saved values.
- Tests: settings validation/describe/load/save/reset; control dict; setter
  writes both configurations; missing enum warns; record placement.

### Task 2: shutter delay

- The delay is taken at the press, for presses from the LCD shutter and the
  GPIO button only. A Stick (remote API) capture never waits: the Stick is
  already hands-off. Give `CaptureController.submit` an optional keyword
  (e.g. `delay_s: float = 0.0`) carried on the job; the worker thread sleeps
  that long before calling the session, interruptibly (closing the controller
  must not wait out the delay; use an Event wait). The job stays `active`
  during the delay, so a second press is `busy` as today.
- The viewfinder and button pass the current setting; the remote server
  passes nothing. The record's `shutter_delay_s` is the delay waited.
- Tests: LCD/GPIO paths pass the delay, remote does not; worker waits (use a
  fake clock or a short real delay with an Event); close during the delay
  ends promptly and the capture does not run; busy during the delay.

### Task 3: burst capture

- `Picamera2Camera.read_burst(n, *, split_exposure: bool) -> list[Frame]`:
  switch to the still configuration once, capture `n` requests back to back,
  switch back to preview (same locks and timeouts as `read(full=True)`; a
  stall raises `CameraStalled` as `read` does). Each Frame keeps its metadata.
- `split_exposure`: when the SHUTTER and ISO cells are both auto, read the
  exposure time `t` and analogue gain `g` the AE is using (latest preview
  metadata), set manual `t/n` and gain `min(g * n, max_gain)` for the burst
  (max from `camera_controls["AnalogueGain"]`), then restore the previous
  exposure/gain modes exactly (auto or manual values) after the burst, also
  on failure (try/finally). When the user fixed the shutter or ISO in the
  viewfinder, the burst uses them unchanged (no split). Report what was used
  per burst (`exposure_us`, `gain`, `split: bool`).
- `FakeCamera` and the V4L2 camera: `read_burst` returns `n` frames from
  `read()` (fake: optionally shifted by a few pixels for tests) with
  `split=False`.
- Tests with the fake Picamera2 used by existing tests: one mode switch per
  burst, n requests, restore after success and after a stall, split maths,
  gain cap, no split with manual shutter/ISO.

### Task 4: the merge (`pifilm/stack.py`, pure, no camera)

- `merge(frames: Sequence[np.ndarray (H,W,3) uint8 sRGB], *, reference: int)
  -> tuple[np.ndarray, dict]`.
- Reference frame: the caller passes the sharpest (Task 5 uses the highest
  `FocusFoM` from metadata, falling back to frame 0).
- Alignment: each other frame to the reference with
  `cv2.findTransformECC` (`MOTION_EUCLIDEAN`) on luma reduced to about
  1000 px on the long side, a bounded iteration count and epsilon; scale the
  translation up; `cv2.warpAffine` at full size (`INTER_LINEAR`,
  `BORDER_REFLECT`). A frame whose ECC raises or does not converge, or whose
  shift exceeds 5% of the width or rotation 2 degrees, is dropped.
- Merge in linear light (reuse the sRGB<->linear used by `pifilm/double.py`;
  do not duplicate it — move it to a shared helper if needed), per pixel
  weight `w = exp(-(d / s)^2)` where `d` is the luma difference to the
  reference in linear light and `s = 0.05` (a module constant with a
  docstring saying why); the reference has weight 1. Output sRGB uint8.
- Process in bands of rows (as `halation.apply_cpu` does with
  `BAND_PIXELS`) so six 12 MP frames do not need several full-size float
  copies at once.
- Info dict: `reference`, per frame `{"dx","dy","angle_deg","used": bool}`,
  `method: "ecc-euclidean-ghost-weighted-mean"`, timings `align_ms`,
  `merge_ms`.
- Tests: synthetic textured frames shifted/rotated by known amounts are
  recovered within 0.5 px and merged image noise drops (std of a flat noisy
  area falls about as 1/sqrt(n)); a moving square appears only from the
  reference (no ghost); a frame with a huge shift is dropped; one frame in
  returns it unchanged; bands give the same result as one band.

### Task 5: session integration

- `CaptureSession.capture`: when `multi_frame > 1` (taken at the shutter like
  `halation`), call `read_burst(n, split_exposure=True)`; choose the reference
  frame (highest `FocusFoM`, else 0).
- Files: the reference frame is `HHMMSS_original.jpg`; the others
  `HHMMSS_frame2.jpg` ... `HHMMSS_frameN.jpg` (numbered in capture order,
  skipping the reference). All are written by the capture side before the
  camera is released (a power cut must not lose the shot); fsync stays on the
  grade worker as today. DNG (when on): reference frame only.
- Background grade: the pending entry's `originals` lists the reference first,
  then the others, plus a `stack` key `{"reference_index": i}`; the worker
  loads them, calls `stack.merge`, then grades the merge; recovery works from
  the entry alone; an older entry without `stack` is a single frame.
  The quick grade uses the reference frame only.
- Inline grade: the same merge, then grade.
- Double exposure: multi-frame is off for both exposures of a double (a
  ruling: the double already averages two frames and its pairing logic is not
  worth complicating); the record says nothing about a stack.
- Record: `stack` = `{"frames": [names, reference first], "reference": name,
  "exposure_us": ..., "gain": ..., "split": bool, "aligned": [per frame info
  from merge], "align_ms": ..., "merge_ms": ...}` placed after
  `shutter_delay_s`; `stage_ms` gains `burst` and `merge`.
- Tests: background/inline/recovered paths produce the merge and the record;
  quick grade from the reference; double ignores multi-frame; old entries.

### Task 6: `pifilm-process` and the STABILISE page

- `batch.py`: a record with `stack` regrades the merge of its listed frames
  (same reference), so the regrade reproduces the Pi's file; frames missing on
  disk: grade the reference alone and count it in a note. `--ignore-log`
  grades the reference alone. Never pick up `_frameN.jpg` files as inputs.
- Menu: a STABILISE page reached from page 1 exactly as the IMAGE page is
  (reuse its mechanism; one more entry must still fit page 1 at all scales
  and must not overlap DONE), with SHORT EXPOSURE, SHUTTER DELAY, MULTI-FRAME,
  a RESET for these three, and BACK. Inactive on V4L2/fake.
- Live view: an amber `STAB` tag (beside `IMG`, same pattern) whenever any of
  the three is not at its default.
- Tests for both.

### Task 7: docs and a Pi timing script

- `scripts/bench_stack.py`: given a capture folder and a record with a stack
  (or N JPEGs), time `stack.merge` (align, merge) on the machine it runs on,
  printing median of 3 runs; no camera, no `.env`. Used on the Pi to decide
  whether 6 frames stays.
- Docs: CLAUDE.md (settings, picamera, records, stack module, batch, display),
  `docs/lcd-viewfinder.md` (STABILISE page, acceptance steps: delay felt,
  short exposure lowers ExposureTime in records, a 4-frame shot handheld in
  dim light is sharper than a single), `docs/how-it-works.md` (a short section
  on the merge), `docs/known-issues.md` if anything is known-but-unfixed.
