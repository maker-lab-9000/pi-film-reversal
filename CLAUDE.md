# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A colour-grading pipeline that learns film and photographic looks from reference photographs
(the first being Martin Parr's saturated colour-negative look) with two runtimes:
a Mac side that trains 3D LUTs from reference photographs, and a Raspberry Pi side that captures
from a USB camera, grades, and saves. An M5Stack StickS3 acts as a wireless shutter button and
last-capture display, talking to the Pi over a token-authenticated HTTP API.

Forked from the user's `kodachrome-film` project (see `ORIGIN.md`). Package is `pifilm`, distribution is
`pi-film-reversal`, commands are `pifilm-*`. `docs/setup.md` is the ordered install guide (Mac, `.env`,
Pi network, Pi service, Stick, optional training) and says which machine each step runs on and why;
`docs/sticks3-remote.md` holds the network and service detail; `firmware/sticks3/README.md` the Stick;
`docs/training.md` is the step-by-step training procedure and explains every report gate;
`docs/x728-ups.md` covers the Geekworm X728 UPS shield (pins, I2C, services, shutdown policy, RTC);
`docs/how-it-works.md` explains the colour model (capture pipeline, training steps, libraries, dependency-by-role table);
`docs/nextcloud-sync.md` the optional Pi→Nextcloud photo archive; `docs/photo-volume.md` the fixed-size photo
volume that keeps captures from filling the Pi's root filesystem; `docs/lcd-viewfinder.md` the optional
Waveshare LCD viewfinder (the 3.5" DSI panel and the fallback 2.8" SPI panel: setup, screen layout, settings menu,
manual controls, meter, hardware acceptance); and
`docs/known-issues.md` tracks understood-but-unfixed defects. The bundled LUT in `pifilm/data/` is a handcrafted,
untrained starter preset (`trained: false`). Trained artifacts, training data, `data/`, `artifacts/`,
`ektar100/`, `velvia/` are all gitignored and not in a clone.

## Commands

Python 3.11+ required (the venv is 3.12). Always use `.venv/bin/...`.

```bash
# Setup (Mac)
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[train,dev]'     # add ,deploy for scripts/deploy_remote.py

# Tests. pyproject sets pytest `pythonpath = ["."]` because tests import `scripts/` and
# `firmware/sticks3/scripts/` as namespace packages from the repo root.
.venv/bin/pytest -q -m 'not slow'                     # ~85s, ~1430 tests
.venv/bin/pytest -q tests/test_lut.py                 # one file
.venv/bin/pytest -q tests/test_lut.py -k identity     # one test by name
.venv/bin/pytest -q -m slow                           # builds a wheel, installs into a temp venv, runs pifilm-process

# Lint (ruff: E, F, I, B, UP; line length 100)
.venv/bin/ruff check .

# CLI entry points (pyproject [project.scripts])
.venv/bin/pifilm-preset --out artifacts/starter-v1                 # regenerate the untrained starter
.venv/bin/pifilm-process IN_DIR OUT_DIR [--artifacts DIR]          # batch regrade
.venv/bin/pifilm-capture --fake --no-preview                       # capture loop w/o hardware
.venv/bin/pifilm-battery                                           # print the X728 battery level (Pi only)
.venv/bin/pifilm-gpu-check --artifacts DIR [--image FILE] [--runs N]   # GPU grade against its reference (Pi only)
.venv/bin/pifilm-tuning [--tuning-file NAME] [--stamp DIR]          # tuning fingerprint; stamp a look (Pi; --sha1 HEX on the Mac)
.venv/bin/pifilm-train --source data/source --target data/references --out artifacts/pifilm-v1
.venv/bin/pifilm-fetch --category 'Category:...'                   # Wikimedia Commons corpus fetch

# Experiments have no console scripts; run as modules from the repo root
.venv/bin/python -m pifilm.experiments.{regression,candidates,train,evaluate} ...

# StickS3 firmware (PlatformIO; reads repo-root .env via pre-build script)
pio run -d firmware/sticks3                # build
pio run -d firmware/sticks3 -t upload      # flash
pio test -d firmware/sticks3 -e native     # host-side Unity tests of the state machine only
python3 firmware/sticks3/scripts/generate_config.py --env .env   # validate .env w/o printing secrets

# Pi deployment (needs [deploy] extra, .env, and the Pi host key already in known_hosts)
.venv/bin/python scripts/deploy_remote.py --env .env             # inspect only
.venv/bin/python scripts/deploy_remote.py --env .env --restart   # guarded systemd restart

# Pi hotspot (run ON the Pi, over Ethernet; dry run prints the redacted keyfile)
.venv/bin/python scripts/pi_hotspot.py --env .env
sudo .venv/bin/python scripts/pi_hotspot.py --env .env --apply

# Pi photo volume (run ON the Pi; dry run prints the steps)
python3 scripts/pi_photo_volume.py
sudo python3 scripts/pi_photo_volume.py --apply

# Pi journal kept across reboots, and the capture service's restarts (run ON the Pi)
python3 scripts/pi_journal.py              # dry run
sudo python3 scripts/pi_journal.py --apply
python3 scripts/pi_journal.py --report     # starts, stalls, error exits per boot

# Nextcloud sync (run ON the Pi; dry run without --apply; the timer runs scripts/nextcloud_sync.py)
.venv/bin/python -m pifilm.nextcloud --env .env [--apply]
```

`pifilm-train` exit code 3 means an artifact was written but a quality gate failed.

## Architecture

### Processing core (`pifilm/`, runs on the Pi; NumPy + Pillow + OpenCV only)

`Pipeline.process()` in `pipeline.py` is the single grading path for captures, previews, and batch:
**normalize → LUT → grain**, in that fixed order. Normalization runs first because the LUT was
fitted on normalized input; grain runs last because it models developed film.
`process(..., grader=, used=)` can hand the three steps to a GPU grader as one pass; the statistics
stay on the CPU, so the audit fields are the same on both paths. The grader is asked first
(`can_grade`), a `GpuError` falls back to the CPU for the same frame with the generator's state
restored (byte-identical to no GPU), `used["backend"]` is `"gpu-1"` or `"cpu"`, and a GPU result is
(H, W, 4) with an unused fourth byte. Without a grader nothing differs.

- `normalize.py`: per-image white balance and exposure/levels normalization. The trainer applies the
  identical maths in float (`normalize_float`); the Pi uses 256-entry lookup tables (`normalize_u8`).
  Tone is applied to luma only by default to avoid per-channel gamma inflating saturation.
  Normalizing an image twice is unsupported.
- `grain.py`: `add_grain`, luminance-only. `model="gaussian"` is the original hand-set grain;
  `model="two_scale"` is Ilford HP5 Plus 400 measured from scans (fine + coarse clumps in mm of a
  36 mm frame, measured brightness envelope; `docs/experiments/2026-10-01-hp5-grain.md`).
  `model="two_scale_tile"` is the same HP5 grain laid down from a noise tile built once
  (`tile_px` square, seamless, fixed seed) and placed per shot by the seed (offset and one of
  eight orientations); about a tenth of the time, different pixels, so its own model and version.
- `lut.py`: `LUT3D` with `apply_numpy` (reference, used in tests/trainer) and `apply_pillow`
  (C fast path, used on the Pi). Both `.cube` and Pillow order the flat table red-fastest.
  `sha1_hex` is the LUT content identity.
- `artifacts.py`: an artifact is `pifilm.cube` + `params.json`. Loading verifies `lut_sha1` matches the
  cube on disk so a half-written pair cannot load. `publish()` stages then swaps the directory with
  `os.replace`. `Artifacts.default()` resolves the packaged `pifilm/data/`; `--artifacts DIR` overrides.
  An optional top-level `tuning_sha1` (forty lowercase hex, validated; `Artifacts.tuning_sha1`) is
  the libcamera tuning the look was trained under; `stamp_tuning` writes it into `params.json`
  (temp file + `os.replace`, other keys and the file mode kept, loaded before and after).
  Older builds ignore it, so no params version change.
- `tuning.py`: the tuning fingerprint (SHA-1 of the parsed JSON with sorted keys and no
  whitespace), `resolve` (the dict the camera is opened with for `--tuning-file`, else
  `<model>.json`; a failure there is a reason, not an error; `LIBCAMERA_RPI_TUNING_FILE` is
  ignored because Picamera2 clears it when it opens without a tuning)
  and `check` (the `tuning:` log line). No Picamera2 import; `pifilm-tuning` is
  `capture/tuningcli.py`.
- `gpu/`: the grade as one pass on the Pi 4's GPU. `shader.py` is the GLSL and `SHADER_VERSION`
  (`BACKEND` is `"gpu-1"`; any change that can move pixels raises it). `reference.py` is the same
  pass in NumPy, the definition the tests and the self-test hold the GPU to. `grader.py` is
  `GpuGrader`: a self-test at set-up (a 96 x 64 frame against the reference, at most 1 level),
  one thread only (another thread is told it cannot grade), every failure a `GpuError`, three in
  a row switch it off. `egl.py` is the only file that loads `libEGL`/`libGLESv2`, through
  `ctypes`, inside functions, so the package imports on a Mac. `check.py` is `pifilm-gpu-check`.
  Only `two_scale_tile` grain (or none), EV 0 and frames up to 4096 wide go to the GPU. A GPU
  grade is not byte-identical to the CPU's: a few levels on ordinary colours, 10 to 20 on strongly
  saturated ones (the CPU rounds to bytes before the LUT). Design:
  `docs/superpowers/specs/2026-10-13-gpu-grading-design.md`.
- `raw.py`: reads a Picamera2 DNG and develops it with the camera's own libcamera tuning file
  (black level, lens shading, digital gain and white balance, OpenCV demosaic, colour matrix,
  tone curve). Offline only: a look trained on developed DNGs showed no visible gain
  (`docs/experiments/2026-10-14-raw-look.md`), so capture does not use it and the example unit
  passes `--no-dng`. Full-field sensor modes only.
- `color.py`: sRGB ↔ linear ↔ Oklab/Oklch. All trainer statistics are computed in Oklab.
- `imageio.py`: the only place pixels enter. Applies EXIF orientation and converts embedded ICC
  profiles to sRGB. Do not load images elsewhere with raw `Image.open`.
- `_cv2.py`: the single `cv2` import site. Use `require_cv2()`; never `import cv2` directly.
  OpenCV is deliberately not a base dependency (Pi uses apt `python3-opencv` for GTK).
- `nextcloud.py`: the Pi→Nextcloud sync (`python -m pifilm.nextcloud`; `scripts/nextcloud_sync.py`
  is a wrapper the timer unit still runs), one at a time under a `flock` on
  `nextcloud-sync.lock` in the state directory. `envfile.py` is its `.env` parser
  (`firmware/sticks3/scripts/generate_config.py` keeps its own copy, as it runs without the package).

### Capture (`pifilm/capture/`)

- `camera.py`: V4L2 UVC camera forcing MJPEG at 1920×1080. Grabs the raw compressed buffer so the
  saved `*_original.jpg` is the camera's own bytes; falls back to decoded mode (file becomes
  `*_ungraded.jpg`) if raw mode fails. `FakeCamera` for tests and `--fake`.
- `picamera.py`: Picamera2 backend, sensor-agnostic (native size and tuning are derived from the
  attached sensor via `Picamera2.sensor_resolution`/libcamera's automatic tuning, not hard-coded
  per model). An optional `preview` mode adds a second, cheap stream for the LCD viewfinder;
  `read(full=True)` switches to the still configuration, captures, and returns to preview.
  `set_shutter` gives shutter priority (`ExposureTimeMode` on libcamera 0.5, legacy `ExposureTime`
  otherwise), written into both configurations so a later configure does not undo it.
  `set_iso` fixes the analogue gain the same way (ISO 100 is gain 1.0; `AnalogueGainMode` on
  current libcamera, legacy `AnalogueGain` otherwise) and `set_colour_temperature` fixes the
  white balance in kelvin (the viewfinder's `WB` cell, `--white-balance K`). There is no
  exposure compensation: `ExposureValue` is sent as 0.0 so `ev_comp` stays in records. Every read waits a bounded time
  for its frame and raises `CameraStalled` (see `docs/known-issues.md`); three stalled preview
  reads end the viewfinder and the process exits 1 for a systemd restart.
  The tuning is fingerprinted at open (`tuning` property; `tuning_sha1` after `tuning_file` in
  records); one that cannot be is one `tuning: cannot fingerprint ...` line, and the camera opens.
- `app.py`: `CaptureSession` owns camera + pipeline + output dir. Three loops: live preview,
  captures-only display (`--show-captures`, shows TV colour bars while processing), and headless
  terminal. `--display waveshare35dsi` (or `waveshare28`) runs the LCD viewfinder instead of any
  OpenCV window. Output:
  `~/Pictures/pifilm/YYYY-MM-DD/HHMMSS_{original|ungraded,graded|double_graded}.jpg` plus an audit
  line in `captures.jsonl` (LUT hash, normalisation hash and grain seed allow regenerating the
  graded file). `--grade background` (the default) splits a shot into two sides: the
  capture side saves the original, writes a pending entry, makes a quick grade (reduced by a whole
  factor to about 1,000 px on the long side, no grain; what the review screen and the Stick show) and frees the camera;
  the grade side (`_run_grade`, on the queue's worker) grades at full size, saves the graded
  file, appends the record and removes the entry, in that order. Such a record gains
  `shutter_to_ready_ms`, `grade: "background"`, stages `preview` and `queued`, and
  `recovered: true` when it was finished at a later start from the saved JPEG.
  `--grade inline` is the earlier behaviour, with no queue and none of those fields (its records
  do carry `pipeline_backend`, always `"cpu"`); `--state-dir`
  moves the pending entries (default `~/.local/state/pifilm/pending`). The grade
  worker fsyncs the original (and DNG) as its first step, not the capture (that cost up to 1 s a
  shot on the Pi's SD card), and the graded file and the log line before the entry is removed.
  SIGTERM ends every loop the way Ctrl-C does (`main` raises `_Terminated` in the main thread;
  the viewfinder's own handler sets its stop event instead), and every clean-up starts with
  `_ignore_sigterm`, so a later SIGTERM cannot break into it. A stop finishes the photographs
  taken in this run; recovered grades not yet started keep their entries for the next start,
  as does anything a power cut or a kill interrupts. Just after a restart with many unfinished
  grades the recovered ones go first, and a fourth new shot waits for them.
  `--grade-backend auto` (the default; `cpu` is the other choice) lets the grade worker open a
  `GpuGrader` on its own thread before its first job and close it after its last; a session
  grading inline never opens one, and the quick grade and a shot a closing queue refused stay on
  the CPU. A photograph the GPU fails on is graded on the CPU in the same call and logged as
  `gpu grade failed (...)`. An open that hangs in the driver would hold the only grade worker
  at every start, so the worker writes a marker, `gpu-opening`, beside the state directory
  (`~/.local/state/pifilm/gpu-opening`) before the open and removes it when the open returns; a
  start that finds the marker logs one `gpu grading off: ...` line naming it, grades on the CPU
  and leaves it for a person to delete. Every record of a graded photograph has `pipeline_backend`
  (`"gpu-1"` or `"cpu"`): after `stage_ms` in an inline record, last in a background one. Each record also carries `stage_ms`, the
  milliseconds each stage of the shot took (camera request, array copy, DNG, normalise, LUT,
  grain, each file write); `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`
  is the plan those numbers feed. At start-up the session checks the camera's tuning
  fingerprint against the look's `tuning_sha1` (warn only, never refuses to start): one
  `tuning:` line, `tuning_matches` right after the camera fields in every record when both
  have a fingerprint (a recovered grade keeps the capture run's value), and
  `CaptureSession.tuning_mismatch` (True only when they differ) for the viewfinder's tag.
  `disk.py`'s `DiskMonitor` (a `pifilm-disk` thread, every 60 s, started for every loop beside
  the UPS poller) checks df's Use% of the photo folder and `/` against `--disk-warn-percent`
  (default 90), logs `disk:` lines on each crossing and feeds `CaptureSession.disk_warning`
  (`(label, percent)` pairs, `PHOTOS` then `SD`; warn only). `network.py`'s `NetworkMonitor`
  (a `pifilm-network` thread, every 10 s, started beside the disk monitor) reads the IPv4
  address of `eth0` and `wlan0` while sysfs reports them up (`SIOCGIFADDR` ioctl; the hotspot's
  address counts), logs `network:` lines on each change and feeds `CaptureSession.network`
  (`(label, address)` pairs, `ETH` then `WLAN`), which the settings menu shows.
- `settings.py`: the three settings the LCD menu changes (DNG saving, AE constraint, AE
  metering), in `settings.json` beside the state directory; read at start-up before the camera
  opens, a saved value wins over the unit's flag on Picamera2 only (one `settings:` line), and
  `SettingsControl` applies, saves atomically and resets them. `upload.py`: `Upload`, the menu's
  UPLOAD row, runs `python -m pifilm.nextcloud` as a child (`--nextcloud-env PATH`, never read
  by the service; the token stripped from its environment) and reads its last line for the status.
- Double exposure: `CaptureSession` owns the mode and pending first frame; `pifilm/double.py`
  adds the two ungraded frames in linear light (one stop down each), and the composite is
  graded once. Exposure 1's `CaptureResult.pifilm` is an "Exposure 1/2" card in a temp dir
  (never under the synced output folder); toggled via `CaptureController.set_double_exposure`.
- `gradequeue.py`: `GradeQueue`, one `pifilm-grade` worker with three waiting places (a full
  queue makes the next capture wait), and the pending entries: atomically written JSON files
  that let the next start finish a grade a power cut interrupted. A failed grade is logged by its entry's name; an entry that fails twice is
  renamed `*.failed` (a second log line says so). `close()` finishes the running job and
  those accepted by `submit`, and drops the recovery backlog not yet started. `on_start` and
  `on_stop` run on the worker thread before the first job and after the last (the GPU grader is
  opened and closed there); one that raises is logged. No camera, pipeline or display in it.
- `controller.py`: `CaptureController` serializes captures on one worker thread, one active job at
  a time, idempotent by `request_id`. Shared by the local SPACE key, the LCD, the GPIO shutter
  button and the remote API. `ControllerSnapshot.pending_grades` is read live from the session
  (it includes a recovery backlog), and closing finishes this run's grades before the camera closes.
- `button.py`: optional physical shutter button (`--shutter-gpio BCM`, documented on BCM 21 with
  ground on physical pin 39). gpiozero `Button` with the internal pull-up, lazy import; a press
  calls `CaptureController.submit` like an LCD tap, so a press during a capture is dropped as
  `busy`. Pins of the X728 and of the 2.8" LCD are refused when that hardware is selected; a pin
  that cannot be claimed is one warning line, not a failure to start.
- `remote.py`: `RemoteCaptureServer`, a stdlib `http.server` with bearer-token auth
  (`PIFILM_REMOTE_TOKEN`). Endpoints: `GET /v1/status`, `POST /v1/captures` (`{"request_id": uuid}`,
  409 when busy), `GET /v1/captures/{id}`, `GET /v1/captures/{id}/image.jpg`. Only jobs submitted
  through this server instance are visible; `instance_id` changes on restart.
- `thumbnail.py`: 240×135 letterboxed JPEG, ≤64 KiB, served to the Stick.
- `batch.py`: `pifilm-process`. Skips `*_graded.*`, defaults to only `_original`/`_ungraded` files in a
  capture folder, refuses an output dir equal to or inside the input. When the folder has a
  `captures.jsonl`, each frame is graded with its logged `ev_comp` and `grain_seed`, so the same
  artifact reproduces the Pi's graded file (`--ignore-log` grades at EV 0): pixel for pixel
  for a USB camera, whose original is the camera's own JPEG; closely for the Pi camera, whose
  original is a quality-95 JPEG of the frame the Pi graded (1 to 2 levels on average).
  It always grades on the CPU, so a record the Pi graded on its GPU (`pipeline_backend` starting
  `gpu`) is reproduced closely, not byte for byte; the tool prints one note per run when it
  regrades such records.

### Display (`pifilm/display/`)

Optional LCD viewfinder, wired up by `--display waveshare35dsi` (Waveshare 3.5" DSI LCD (E),
640×480) or `--display waveshare28` (the fallback 2.8" SPI panel, 320×240). `kms.py` drives the
DSI panel through DRM/KMS (`pykms`, two swapped buffers; the Pi must boot to the console so the
service owns the screen) and `evtouch.py` reads its Goodix touch from kernel input events, found
by device name; `st7789.py` and `cst3530.py` drive the SPI panel and its touch controller.
`touch.py` holds the shared `TouchPoint`/`Tap`/`TapDetector`. `meter.py` computes the light-meter
readout (pure, from preview metadata + pixels). The manual controls are pure step scales:
`shutter.py` (shutter speed), `iso.py` (100 to 1600), `whitebalance.py` (3200 to 7500 K), the
last two over the generic `stepscale.py` (an auto end, and a first step from auto that lands on
what the camera is metering). `liveview.py` draws the live screen, an SLR-finder style black
strip of three selectable cells (SHUTTER, ISO, WB) with one `+`/`-` pair, shutter button at the
top right, meter dots, `2x`, battery, the developing count (an amber number right of the
battery tag: `LiveState.developing`, from the snapshot's `pending_grades` at every frame,
absent at zero), an amber `TUNING` tag last in that row (`LiveState.tuning_mismatch`, from the
snapshot's `tuning_mismatch`), amber `PHOTOS NN%`/`SD NN%` tags in their own tile between that
row and the shutter (`LiveState.disk_warning`, from the snapshot's `disk_warning`), a `MENU`
key between the shutter and `+` that opens the settings menu, and focus gauge, from cached anti-aliased tiles in the
bundled Barlow Condensed font (`pifilm/data/fonts/`, OFL), and hit-tests its taps; `ui.py` now
holds only the review, processing and message screens; `menu.py` draws the settings menu page
(rows, `DONE` where the shutter is) and hit-tests it. Both work in 320×240 base units at an
integer `scale` (pure). `viewfinder.py`'s `ViewfinderLoop` is the LIVE/REVIEW/MENU state machine
(it also lights `+`/`-`/shutter while a finger is on them) and takes the scale from the display's
width. Under `run()` it reads preview frames on a `pifilm-frames` thread (touch is on `pifilm-touch`)
and redraws the latest frame every step, so the controls stay responsive at a 1 s shutter;
`Picamera2Camera` backs this with two locks: a preview read holds only the request lock, the
control setters only the configuration lock, and a still holds both (request first). Each new
frame's metadata is checked against the fixed shutter/ISO/white balance, because the Pi 4 ignores
live white-balance changes after a photo (`docs/known-issues.md`): a white-balance mismatch lasting
1.5 s and 6 frames restarts the live view at once (`restart_preview`), a shutter/ISO one lasting 3 s
is resent (`reassert_controls`) and restarted after two resends, one restart per control per selection; each
successful step also logs one `control: ...` trace line about 2 s later (selection, what the camera
reports, the picture's R/G and B/G). The DSI panel has no brightness control, so it
skips the idle dim step and only turns off. All hardware imports (`pykms`, `spidev`, `gpiozero`,
`smbus2`) are lazy, inside the `open_*` factories, so the package imports on a Mac and in tests.
`fake.py` (`--display fake`) writes frames to a PNG instead. The loop shares the
`CaptureController` with the Stick's remote server, so an LCD tap and a Stick request are the
same kind of job.

### Trainer (`pifilm/train/`, Mac only; needs `[train]` extra: SciPy, requests)

`fit.py` sequence: `dataset.build_corpus` (split **by image** before pixel sampling, then normalize
and sample in Oklab) → `transport.hue_weights` (reweight target hues toward source to reduce content
bias) → `transport.iterative_distribution_transfer` (Pitié IDT; gives every source pixel a target
partner) → `lutfit.fit_lut` (sparse regularised least squares with smoothness + identity terms,
solved by CG, then monotone projection) → `evaluate.evaluate` + `check_gates` (paired SWD metrics on
held-out images, plus safety gates: grey-axis monotone, neutral tint cap, channel monotone, gamut
clip) → `report.write_report` → `artifacts.publish`. `FitConfig` defaults are the single source of
truth for CLI defaults. Targets default to no white balance and no levels stretch (exposure match
only); sources get the same normalization the Pi applies.

`fetch.py` (`pifilm-fetch`) downloads a Wikimedia Commons category with per-file licence checks and a
`manifest.json`; it is a generic corpus utility, not a Parr reference set.

### Experiments (`pifilm/experiments/`)

Offline grading experiments that never change production defaults. `partitions.py` enforces
checksum-verified, scene-grouped train/validation splits; `regression.py` freezes immutable
snapshots of ungraded/graded triplets; `candidates.py` bakes starter-derived controls into
deployable LUTs; `train.py`/`evaluate.py` run pilots and paired comparisons. Results are written up
in `docs/experiments/`. `scripts/prepare_refinement.py` is a dated curation recipe for one run.

### StickS3 firmware (`firmware/sticks3/`, PlatformIO, ESP32-S3, M5Unified)

`capture_client.{h,cpp}` is a pure state machine (Connecting → Ready → Requesting → Processing →
Downloading → Photo/Error) with no Arduino/Wi-Fi/display dependency so it can run under the `native`
Unity test env; `display_power.{h,cpp}` is the second such pure module, the BtnB screen-off/on toggle
(a debounced click toggles; `StickDisplay::setPower` sleeps the panel and forces a repaint on wake,
and captures continue while the screen is dark). `main.cpp` runs networking in a FreeRTOS task and the UI loop on the main task;
`display.cpp` draws colour bars, status, and the decoded JPEG. Config enters only as four
`STICKS3_*` compile defines appended by `scripts/generate_config.py` from `.env`; empty defaults on a
clean clone leave the device unable to join a network. Build outputs contain credentials.
Every capture step is logged on the serial port (`STICK_LOG` in `main.cpp`, `[display]` lines in
`display.cpp`), including the return value of each `drawJpg`; `clientStateName()` names states for
the log and is unit-tested. `firmware/sticks3/README.md` has an annotated healthy log.

### Deployment (`scripts/deploy_remote.py`, `deploy/pifilm-capture.service.example`)

Topology: the Pi runs its own WPA2 hotspot (`scripts/pi_hotspot.py`, a NetworkManager keyfile
generated from `.env`) and the Stick joins it directly; SSH and photo transfer go over Ethernet
(`parr.local` via avahi). `WIFI_SSID`/`WIFI_PASSWORD` in `.env` are therefore the hotspot's
credentials, `PIFILM_REMOTE_URL`'s host is the hotspot address, `PI_HOST` is the SSH address, and
`PIFILM_LISTEN` (default `0.0.0.0:8765`) is where `pifilm-capture` binds.

Facts verified on the real Pi (2026-09-08): Raspberry Pi OS Trixie, NetworkManager 1.52 with
netplan-generated profiles (`netplan-eth0`, `netplan-wlan0-<SSID>`; `nmcli` changes persist as
`/etc/netplan/90-NM-*.yaml`). The Pi 3B radio (BCM43430) has no management-frame protection, so the
hotspot keyfile must set `pmf=1` or the AP never starts. The deployed look is the bundled starter
(`--artifacts .../pifilm/data`); the example unit and `.env.example` match that. The user account has
no passwordless sudo, so `deploy_remote.py --restart` needs the narrow sudoers rule from the guide.

`deploy_remote.py` uses Paramiko with system known_hosts and `RejectPolicy`; it inspects the Pi
(project dir, artifact, running `pifilm-capture`, `/dev/video*` owners, desktop sessions) and refuses
to restart when a foreign process owns the camera. The systemd unit runs the LCD viewfinder when a
panel is present and falls back to headless Stick-only otherwise (a missing panel, or a V4L2
backend, is one warning line, not a failure to start); it reads `PIFILM_REMOTE_TOKEN` from
`/etc/pifilm-capture.env`. Two-screen mode (`--show-captures`) must
be launched from the Pi's desktop session, not over SSH. Full guide: `docs/sticks3-remote.md`.

## Conventions and constraints

- `.env` holds Pi/Wi-Fi/token secrets and is gitignored. `.env.example` documents keys. Scripts parse
  it as `KEY=VALUE` data only; never source it or echo values. Never put the token, Wi-Fi password,
  or SSH password into a command line, source file, service unit, or commit.
- `params.json` `version` follows the grain model (`GRAIN_MODEL_VERSIONS`): 2 for Gaussian,
  3 for `two_scale`, 4 for `two_scale_tile`, so an older build refuses grain it would misread.
  The loader reads 1 to `PARAMS_VERSION` (4). `lut_sha1` is required.
- Module docstrings carry the design rationale (why an order is fixed, why a check exists). Read them
  before changing behaviour; many encode failures found on real hardware.
- Reference photographs must be credited to Martin Parr from primary sources with permission;
  fan-submission groups are excluded. See `docs/reference-sources.md`. Training reports live in
  `docs/training-*.md`.
- `ektar100/` and `velvia/` are downloaded film-reference corpora (gitignored); `velvia-attribution.md`
  records their licences.
- Agent scratch space (`.superpowers/`) is gitignored; implementation plans are committed under
  `docs/superpowers/plans/`.
