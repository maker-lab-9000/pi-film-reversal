# Pi 4 / IMX708 roadmap progress

Roadmap: [2026-09-11-rpi4-imx708-roadmap.md](2026-09-11-rpi4-imx708-roadmap.md).
Dependency review: 2026-09-13, following highlight-protection commits
`d349fec` and `d44c08d` on `plan/rpi4-imx708-camera`.

## Phase status

| Phase | Status | Next prerequisite |
| --- | --- | --- |
| 1: Picamera2 acquisition | Implemented, merged, hardware-accepted | Done. |
| 2: capture controls and metadata | Implemented, merged, **hardware-accepted 2026-09-19** | Done: the sensor reports the full metadata set and autofocus moves `LensPosition` (`AfState=2`). |
| 3: original / DNG / graded output | Implemented, merged; DNG written on hardware (15 MB) | One manual check left: open a `.dng` in darktable/RawTherapee and confirm colour. |
| 4: benchmark | **Measured on the Pi 4, 2026-09-19** | Done: ~3.29 s to grade 12 MP, ~5.3 s shutter-to-saved, zero camera timeouts. Native 12 MP kept. |
| F: lighting experiment | **Dropped** | Superseded by the no-flash decision of 2026-09-14. |
| 6: gentler normalisation | **Done, frozen 2026-09-19** | Done: `levels_lift_highlight_ref=0.02`, `white_balance=False` in the bundled starter. |
| 5: IMX708 retraining | Not performed | Normalisation frozen 2026-09-19 (white balance off, ref 0.02); collect the corpus. |
| 7: grain | Not implemented | Phase 4 timing/resolution decision; independent of colour-LUT fitting. |
| 8: scene statistics | Not implemented | Acquisition metadata for full integration; not a blocker for the first retrain. |
| 9: baked highlight protection | Implemented and tested | Select the IMX708 value in Phase 5; current default remains zero. |
| 9: runtime blend | Not implemented; optional | Evidence of need and Phase 8 adaptive inputs. |
| 10: classifier | Not implemented; conditional | Labelled data and evidence simple scene rules are insufficient. |
| F′: final flash | **Dropped** | Superseded by the no-flash decision of 2026-09-14. |

## Dependency review completed

- Removed the dependency of baked highlight protection on scene statistics.
- Moved highlight-value selection into the first IMX708 training workflow;
  old-camera results do not select a new-camera default.
- Preserved separate controls for normalisation, LUT strength and highlights.
- Identified the scene-grouped CLI gap: `pifilm/experiments/train.py` accepts a
  `FitConfig` through its Python API but its CLI only exposes `--neutral-cap`.
  `pifilm-train` exposes highlights but partitions automatically by image.
- Added an explicit source/runtime parity requirement: pre-JPEG pixels and
  decoded saved-JPEG pixels are not identical inputs for exact replay.
- Clarified that new normalisation requires a refit, but suitable originals
  can be reused; DNG development is not needed for the current JPEG trainer.
- Corrected the assumption that 512-px training sampling makes all capture
  modes and pre-LUT resizing paths interchangeable.
- Moved final lighting establishment before the production flash corpus, or
  required a later recollection/refit if the light changes.

## Photo milestones

1. **Implementation can start now without a corpus.** Real shots are needed
   to accept acquisition, metadata and saved-file behaviour on hardware.
2. **First useful handoff: after Phases 1–3.** Target 20–30 pilot originals
   across 8–10 scenes, with capture logs and scene IDs. Include ten matched
   flash/ambient pairs if testing a light. Use these for F/6 decisions.
3. **Production collection: after capture and lighting are stable.** At least
   100 development originals over 25 scenes, plus 3–5 separate final-test
   scenes. Collection may overlap Phase 6 implementation; normalisation must
   be fixed before fitting. Prepare at least 200 coherent references separately.
4. **First production fit: Phase 5.** Use scene-held-out development data to
   select the fit and highlights; inspect the untouched final scenes only
   after settings are fixed. No IMX708 quality claim exists yet.

## Open: faces too dark under the starter (2026-09-20)

One indoor frame from the Pi 4 shows skin losing a third of its luminance under the
starter LUT; the normaliser was a no-op on it. Cause and a measured fix (the existing
`shadows` control in `candidates.py`) are in
[the face chromatic fix suggestion](../../experiments/2026-09-20-face-chromatic-fix-suggestion.md).
Not decided: the user is collecting more frames with faces, above all outdoors, first.
Can be done alongside the Phase 5 corpus collection since it changes only the starter.

## Next implementation unit

**Phase 5: IMX708 retraining, starting with the source corpus.** Phase 6 is
done and frozen (2026-09-19), so normalisation is stable and the LUT can be
fitted once against it. Collect at least 100 usable originals across at least
25 development scenes, plus 3–5 whole scenes reserved for the final held-out
test, then run `pifilm-train` with `--no-source-white-balance
--source-lift-highlight-ref 0.02` so the fit is trained on exactly the
normalisation the Pi will apply.

## 2026-09-19 — Phase 6 complete

Frozen after the [normalisation write-up](../../experiments/2026-09-19-imx708-normalisation.md)
and the design in
[the Phase 6 spec](../specs/2026-09-19-phase6-normalisation-design.md): the
108-shot IMX708 pilot showed the levels tone lift pushing already-clipped
outdoor frames further into clipping (median gamma 0.577, 53 of 108 shots
lifted while already at the ceiling) and a redundant grey-world white balance
on top of the ISP's own AWB (up to 1.6x blue on foliage/yellow scenes).

- **Clipping-aware lift** (`NormalizeParams.levels_lift_highlight_ref`):
  damps the levels gamma lift by the fraction of pixels already at the
  highlight ceiling; darkening is never damped. Single implementation in
  `compute_gains()`, so the Pi and the trainer stay in parity.
- **Frozen values**, bundled starter (`pifilm/preset.py` ->
  `pifilm/data/params.json`): `white_balance=False`,
  `levels_lift_highlight_ref=0.02`. LUT bytes and `lut_sha1` unchanged.
- **Decision**: chose `ref=0.02` over `ref=0.05`. On the aggregate outdoor
  medians the two are indistinguishable — both give median gamma 1.00 and
  median clip 8.17%. They differ on the specific problem shots, those with a
  moderate ceiling fraction, which `0.05` only partially damps and `0.02`
  removes entirely: 172404 gamma 0.86 (`ref=0.05+nowb`) vs 1.00
  (`ref=0.02+nowb`), 172258 0.83 vs 1.00, 171656 0.93 vs 1.00. The cost is
  that `0.02` removes *more* of the indoor lift (indoor median gamma
  0.764 -> 0.975) than `0.05` would (-> 0.856). The spec's proposed selection
  rule (indoor gamma within 0.02 of current, outdoor clip down to the indoor
  level) was not satisfiable by any candidate on this set, so the user chose on
  the contact sheet, preferring the outdoor problem shots fully corrected over
  keeping the indoor lift.
- **Capture-side**: `pifilm-capture --ae-constraint {normal,highlight,shadows}`,
  `--ae-metering {centre,spot,matrix}`, `--ev STOPS`, Picamera2-only, defaults
  unchanged from today's behaviour. `highlight` targets exactly the kind of
  in-camera clipping (a 171656-type shaded-foreground/bright-background scene)
  that normalisation cannot fix after the fact; still needs one hardware shot
  to accept (`docs/picamera2-bringup.md` section 4).
- **Trainer**: `pifilm-train --no-source-white-balance
  --source-lift-highlight-ref FRACTION`, recorded in `params.json`.

**In parallel, start collecting the Phase 5 source corpus** — it is the long
pole and needs no new code: at least 100 usable originals across at least 25
development scenes, plus 3–5 whole scenes reserved for the final held-out test.

## Validation of the September 13 roadmap review

Reviewed current highlight/config/provenance code, capture frame/session code,
normalisation and dataset handling, scene-partition training, training guide,
and the completed highlight experiment/progress records. Roadmap-relative
Markdown file links resolve and `git diff --check` passes. That review changed
documentation only; implementation progress is recorded below.

## 2026-09-14 — camera delivered; Phase 1 started

User reports the IMX708 autofocus camera has arrived. Connection to the Pi 4, SSH address, installed stack and exact lens variant are not yet confirmed.

Implementation plan: [Picamera2 acquisition](2026-09-14-picamera2-acquisition.md). Baseline camera/app/controller tests: 85 passed. Use fake Picamera2 requests for development; hardware acceptance remains separate. Preserve the prior roadmap review edits.

- Backend implemented: optional stream metadata, native BGR-to-RGB copy, measured FPS, request/close cleanup and lazy OS dependency. Focused camera tests: **50 passed**; Ruff and diff checks clean. Scoped spec and quality review passed; CLI integration in progress.

- Hardware checklist: [Picamera2 bring-up](../../picamera2-bringup.md). Hardware identity, SSH access and native capture acceptance remain pending. Existing deploy script/service examples omit backend selection; migration must set the intended backend/tuning explicitly.

- CLI selection checks: **63 app tests passed**. Explicit backend/tuning, automatic selection, fake bypass and dependency/conflict errors verified. Native session/thumbnail acceptance test and final suite remain in progress.

## 2026-09-14 — Phase 1 hardware acceptance on the Pi 4 (192.168.178.87)

Hardware confirmed and Phase 1 accepted on the real board.

- **Board / stack:** Raspberry Pi 4 Model B Rev 1.4, 8 GB, Raspberry Pi OS Trixie,
  kernel 6.18; Picamera2 0.3.37, libcamera 0.7.2 from apt.
- **Camera:** `imx708_wide` — the Wide variant, so `imx708_wide.json` is correct.
  Native 4608 × 2592, 10-bit `SBGGR10_CSI2P` (not 12-bit; the backend reads bit
  depth from the live config). `config.txt` already carries `dtoverlay=imx708`
  with `imx477` commented; RTC overlay and I2C present.
- **Capture test:** a direct `Picamera2Camera` + `CaptureSession` run saved two
  native 4608 × 2592 frames. `captures.jsonl` recorded `frame_source=picamera2`,
  `sensor_mode=4608x2592 SBGGR10_CSI2P`, `bit_depth=10`, `tuning_file`, and a
  measured `fps` of 14.35 from `FrameDuration`. The graded JPEG was pulled back
  and inspected: correct colour, red/blue not swapped, sharp, starter look
  applied. Colour management (BGR→RGB) verified on real pixels.
- **Phase 4 signal (informal):** at full 4608 × 2592 the pipeline took about
  3.4 s per frame and shutter-to-saved about 4.5 s on the Pi 4. This informs the
  Phase 4 full-res-versus-binned decision; a formal benchmark is still separate.

### Branch-lineage fix

`plan/rpi4-imx708-camera` was cut from `main` before the UPS battery work merged
(PR #9/#11), so it had the Picamera2 backend but not the `--ups` flag. The
installed unit (from the UPS deployment) passes `--ups x728`, so the first deploy
crash-looped with `unrecognized arguments: --ups x728`. Fixed by merging
`origin/main` into the branch (one import conflict in `pifilm/capture/app.py`,
resolved to keep both the `--camera` and `--ups` wiring). Full suite 475 passed,
ruff clean.

### Service state

The installed unit now runs `... --ups x728 --camera picamera2 --tuning-file
imx708_wide.json`. After the merge was pulled and the service restarted:
`active/running`, `NRestarts=0`, listening on `0.0.0.0:8765`, `/v1/status`
returns `ready:true` with `pi_battery` present and 401 without the token. The Pi
is on branch `plan/rpi4-imx708-camera`; it should return to `main` once this
branch merges.

### Still pending (manual / next)

- End-to-end Stick capture over the Pi's hotspot (this Pi was reached on the home
  network; the hotspot path was not exercised here).
- Autofocus control, per-shot AE/AWB metadata and original/DNG output are Phases
  2–3, not in this backend.
- A dedicated Phase 4 benchmark, and the pilot corpus (20–30 originals) once
  Phases 2–3 land.

## 2026-09-14 — Phases 2 and 3 implemented (fake-backed tests only)

Plan: [IMX708 Autofocus, Metadata and Original/DNG Output](2026-09-14-imx708-af-metadata-dng.md),
Tasks 1–4 on `plan/imx708-af-metadata-dng`.

- **Phase 2 (autofocus, AE/AWB control, metadata):** `Picamera2Camera` neutralises
  Sharpness/Contrast/Saturation and noise reduction, defaults to continuous
  autofocus at normal range, and leaves AE/AWB auto (no flash, per the 2026-09-14
  decision recorded in the implementation plan); `--ae-lock`, `--awb-lock` and
  `--colour-gains R,B` are available for controlled shoots. Each `read()` filters
  the request's metadata to a fixed, JSON-serialisable key set
  (`ExposureTime`, `AnalogueGain`, `DigitalGain`, `ColourGains`,
  `ColourTemperature`, `Lux`, `LensPosition`, `AfState`, `FocusFoM`,
  `FrameDuration`, `SensorTimestamp`) and drops the rest.
- **Phase 3 (original / DNG / graded output):** `CaptureSession.capture()` names
  the Picamera2 original `_original.jpg` always — a Picamera2 frame is the
  camera's own full-quality rendering whether or not a DNG sidecar is also
  saved — writes a `<stem>.dng` sidecar from the same request when `save_dng`
  is enabled, and records `camera_metadata` and `dng` in `captures.jsonl`.
  `--no-dng` disables the sidecar on both the backend and the session; it only
  omits the `.dng` file and never changes the original's name.
- **Task 4 (this entry):** added an end-to-end fake-backed test in
  `tests/test_picamera.py` that drives the real `Picamera2Camera` (via the
  injected fake `picamera2`/`libcamera` modules and a fake capture request)
  through a real `CaptureSession`, asserting the day folder holds
  `*_original.jpg`, `*.dng` and `*_graded.jpg`; that the last `captures.jsonl`
  record carries `frame_source == "picamera2"`, `camera_metadata`, `dng` and the
  sensor fields; and that `fitted_jpeg` on the graded file returns a
  240 × 135 thumbnail. A second test drives `Picamera2Camera(save_dng=False)`
  (the `--no-dng` path) through the same session and confirms the original is
  still saved as `_original.jpg`, with no `.dng` file, no `dng` key, and zero
  calls to the fake request's `save_dng`. Full suite:
  522 passed, 1 deselected (`-m 'not slow'`); Ruff clean. Documentation updated:
  `docs/picamera2-bringup.md`, `docs/setup.md`, `README.md`.
- **Hardware acceptance still pending** (fake-backed tests are not hardware
  acceptance): the DNG opens correctly in darktable/RawTherapee with correct
  colour; autofocus moves `LensPosition` and `AfState` is verified across
  subject distances on the sensor; `ExposureTime`, `AnalogueGain` and `Lux` are
  plausible for the scene; and the API-verification checklist from the
  implementation plan — `get_metadata()` key names actually present on this
  sensor, whether `save_dng(path)` also accepts a file object, presence of
  `AfModeEnum`/`AfRangeEnum`/`autofocus_cycle()` and whether the Wide module
  reports `AfState` and moves `LensPosition`, and whether the
  `NoiseReductionMode` enum path exists on the installed libcamera (0.7.2) or
  must be skipped.

## 2026-09-19 — Pi 4 bring-up: Phase 4 benchmarked, Phases 2/3 hardware-accepted

The SD card was migrated to a Raspberry Pi 4 Model B with the Camera Module 3
Wide attached. Everything below is measured on that hardware, not on fakes.

### Phase 4 — 12 MP benchmark

From `captures.jsonl` over three consecutive captures:

| Metric | Pi 3B | Pi 4 |
| --- | --- | --- |
| Grade one 12 MP frame (`pipeline_ms`) | ~20,400 ms | **3286.7 / 3284.8 / 3303.7 ms** |
| Shutter to saved (`shutter_to_saved_ms`) | — | **5325.6 / 5261.9 / 5372.2 ms** |
| `Camera frontend has timed out` | on the second capture | **0 for the whole boot** |

Roughly six times faster, and the second-capture timeout recorded in
`docs/known-issues.md` does not occur: the grade no longer starves the camera
long enough to trip libcamera's one-second dequeue watchdog. **Native 4608×2592
is kept**; the configurable/downscaled grading resolution considered during
diagnosis is not needed. That entry in `docs/known-issues.md` is now marked
resolved.

### Phases 2 and 3 — hardware acceptance

- **Metadata**: the sensor reports the full expected set — `AfState`,
  `AnalogueGain`, `ColourGains`, `ColourTemperature`, `DigitalGain`,
  `ExposureTime`, `FocusFoM`, `FrameDuration`, `LensPosition`, `Lux`,
  `SensorTimestamp`. Values are plausible for the scene (`Lux` 124–186,
  `ColourTemperature` 5198–5759 K, `AnalogueGain` ~2.0, `ExposureTime`
  43,379–59,994 µs).
- **Autofocus** genuinely moves: `LensPosition` varied 2.29 / 2.38 / 2.45 across
  shots with `AfState=2` (focused). The continuous-AF default works on the Wide
  module.
- **Output**: `<stem>.dng` 15 MB, `_original.jpg` 2.5 MB, `_graded.jpg` 3.1 MB —
  about 21 MB per shot, close to the ~28 MB documented estimate.
- **Still outstanding (manual)**: open a `.dng` in darktable/RawTherapee and
  confirm the colour is right. That is the only Phase 2/3 acceptance item left.

### Hotspot fix found during the same bring-up

The Stick appeared unable to connect, especially with Ethernet unplugged. It was
in fact associating and getting a lease (`10.42.0.70`) but re-associating every
few minutes, and the kernel logged `brcmf_cfg80211_set_power_mgmt: power save
enabled` on `wlan0` as eth0's carrier dropped. Wi-Fi power save on the AP radio
was degrading the hotspot. `scripts/pi_hotspot.py` now writes `powersave=2` into
the keyfile's `[wifi]` section. Ruled out along the way: the home-Wi-Fi client
profile stealing `wlan0` (`autoconnect=no`), an SSID mismatch, and the capture
service.

### Next

Phase 6 (gentler, configurable normalisation) is done and frozen — see the
2026-09-19 entry above. Next: Phase 5, retrain on IMX708 frames, starting
with the source corpus.
