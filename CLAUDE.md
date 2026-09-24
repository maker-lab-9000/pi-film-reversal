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
`docs/nextcloud-sync.md` the optional Pi→Nextcloud photo archive; `docs/lcd-viewfinder.md` the optional
Waveshare LCD viewfinder (wiring, setup, screen layout, meter, hardware acceptance); and
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
.venv/bin/pytest -q -m 'not slow'                     # ~40s, ~380 tests
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
```

`pifilm-train` exit code 3 means an artifact was written but a quality gate failed.

## Architecture

### Processing core (`pifilm/`, runs on the Pi; NumPy + Pillow + OpenCV only)

`Pipeline.process()` in `pipeline.py` is the single grading path for captures, previews, and batch:
**normalize → LUT → grain**, in that fixed order. Normalization runs first because the LUT was
fitted on normalized input; grain runs last because it models developed film.

- `normalize.py`: per-image white balance and exposure/levels normalization. The trainer applies the
  identical maths in float (`normalize_float`); the Pi uses 256-entry lookup tables (`normalize_u8`).
  Tone is applied to luma only by default to avoid per-channel gamma inflating saturation.
  Normalizing an image twice is unsupported.
- `lut.py`: `LUT3D` with `apply_numpy` (reference, used in tests/trainer) and `apply_pillow`
  (C fast path, used on the Pi). Both `.cube` and Pillow order the flat table red-fastest.
  `sha1_hex` is the LUT content identity.
- `artifacts.py`: an artifact is `pifilm.cube` + `params.json`. Loading verifies `lut_sha1` matches the
  cube on disk so a half-written pair cannot load. `publish()` stages then swaps the directory with
  `os.replace`. `Artifacts.default()` resolves the packaged `pifilm/data/`; `--artifacts DIR` overrides.
- `color.py`: sRGB ↔ linear ↔ Oklab/Oklch. All trainer statistics are computed in Oklab.
- `imageio.py`: the only place pixels enter. Applies EXIF orientation and converts embedded ICC
  profiles to sRGB. Do not load images elsewhere with raw `Image.open`.
- `_cv2.py`: the single `cv2` import site. Use `require_cv2()`; never `import cv2` directly.
  OpenCV is deliberately not a base dependency (Pi uses apt `python3-opencv` for GTK).

### Capture (`pifilm/capture/`)

- `camera.py`: V4L2 UVC camera forcing MJPEG at 1920×1080. Grabs the raw compressed buffer so the
  saved `*_original.jpg` is the camera's own bytes; falls back to decoded mode (file becomes
  `*_ungraded.jpg`) if raw mode fails. `FakeCamera` for tests and `--fake`.
- `picamera.py`: Picamera2 backend, sensor-agnostic (native size and tuning are derived from the
  attached sensor via `Picamera2.sensor_resolution`/libcamera's automatic tuning, not hard-coded
  per model). An optional `preview` mode adds a second, cheap stream for the LCD viewfinder;
  `read(full=True)` switches to the still configuration, captures, and returns to preview.
- `app.py`: `CaptureSession` owns camera + pipeline + output dir. Three loops: live preview,
  captures-only display (`--show-captures`, shows TV colour bars while processing), and headless
  terminal. `--display waveshare28` runs the LCD viewfinder instead of any OpenCV window. Output:
  `~/Pictures/pifilm/YYYY-MM-DD/HHMMSS_{original|ungraded,pifilm}.jpg` plus an audit
  line in `captures.jsonl` (LUT hash, normalisation hash and grain seed allow regenerating the
  graded file).
- `controller.py`: `CaptureController` serializes captures on one worker thread, one active job at
  a time, idempotent by `request_id`. Shared by the local SPACE key and the remote API.
- `remote.py`: `RemoteCaptureServer`, a stdlib `http.server` with bearer-token auth
  (`PIFILM_REMOTE_TOKEN`). Endpoints: `GET /v1/status`, `POST /v1/captures` (`{"request_id": uuid}`,
  409 when busy), `GET /v1/captures/{id}`, `GET /v1/captures/{id}/image.jpg`. Only jobs submitted
  through this server instance are visible; `instance_id` changes on restart.
- `thumbnail.py`: 240×135 letterboxed JPEG, ≤64 KiB, served to the Stick.
- `batch.py`: `pifilm-process`. Skips `*_graded.*`, defaults to only `_original`/`_ungraded` files in a
  capture folder, refuses an output dir equal to or inside the input.

### Display (`pifilm/display/`)

Optional SPI/I2C LCD viewfinder, wired up by `--display waveshare28`: `st7789.py` and `cst3530.py`
drive the panel and its touch controller; `meter.py` computes the light-meter readout (pure,
from preview metadata + pixels); `ui.py` renders the live/review/message screens and hit-tests
taps (pure); `viewfinder.py`'s `ViewfinderLoop` is the LIVE/REVIEW state machine. All hardware
imports (`spidev`, `gpiozero`, `smbus2`) are lazy, inside the `open_*` factories, so the package
imports on a Mac and in tests. `fake.py` (`--display fake`) writes frames to a PNG instead of SPI.
The loop shares the `CaptureController` with the Stick's remote server, so an LCD tap and a Stick
request are the same kind of job.

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
Unity test env. `main.cpp` runs networking in a FreeRTOS task and the UI loop on the main task;
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
to restart when a foreign process owns the camera. The systemd unit is the headless Stick-only mode
and reads `PIFILM_REMOTE_TOKEN` from `/etc/pifilm-capture.env`. Two-screen mode (`--show-captures`) must
be launched from the Pi's desktop session, not over SSH. Full guide: `docs/sticks3-remote.md`.

## Conventions and constraints

- `.env` holds Pi/Wi-Fi/token secrets and is gitignored. `.env.example` documents keys. Scripts parse
  it as `KEY=VALUE` data only; never source it or echo values. Never put the token, Wi-Fi password,
  or SSH password into a command line, source file, service unit, or commit.
- `params.json` `version` is currently 2 (`PARAMS_VERSION`). `lut_sha1` is required.
- Module docstrings carry the design rationale (why an order is fixed, why a check exists). Read them
  before changing behaviour; many encode failures found on real hardware.
- Reference photographs must be credited to Martin Parr from primary sources with permission;
  fan-submission groups are excluded. See `docs/reference-sources.md`. Training reports live in
  `docs/training-*.md`.
- `ektar100/` and `velvia/` are downloaded film-reference corpora (gitignored); `velvia-attribution.md`
  records their licences.
- Agent scratch space (`.superpowers/`) is gitignored; implementation plans are committed under
  `docs/superpowers/plans/`.
