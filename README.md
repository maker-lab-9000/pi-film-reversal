# Introduction
This entire idea took shape from trying to find a kid friendly camera for my son, which escalated into a full blown project consisting of Raspberry Pi 4 portable camera with IMX708 Sensor and C mount lens, 3D LUT grading via machine learning using some hand selected sources and 3d printed case.

The prototype phase is still in progress, the repo is super messy hence I'm experimenting with various hardware, software, and photo grading techniques.


# Pi Film Reversal

One experiment in pushing the Raspberry Pi camera to recreate film and photographic looks with machine learning: it learns a 3D-LUT colour grade from a set of reference photographs, then applies it on-device at capture time.

The bundled look is a **handcrafted, untrained starter preset**; no reference photographs were used to train that preset. 

See [the training guide](docs/training.md) for how a run works and what the gates mean.

3D LUT Artifacts and training data are gitignored for the moment and are not included in a clone.

To test a sample trained model after installing the package, explicitly select it:

```bash
pifilm-process /path/to/ungraded-photos /path/to/results \
  --artifacts artifacts/personal-collection-01-v1
```

Without `--artifacts`, commands continue to use the untrained starter preset.

## Sample results

Photos captured and graded on the Raspberry Pi with the Camera Module 3 Wide (IMX708):
<p>
  <img width="300" alt="Sample photo 2" src="https://github.com/user-attachments/assets/a626dde2-58c7-4406-ae52-45b3a5e5780a" />
  <img width="300" alt="Sample photo 3" src="https://github.com/user-attachments/assets/0bc29cdd-6456-48be-8309-467c11d3afa7" />
  <img width="300" alt="Sample photo 4" src="https://github.com/user-attachments/assets/7ea2ab9f-f598-4074-89c9-34527727c80f" />
  <img width="300" alt="Sample photo 5" src="https://github.com/user-attachments/assets/e923aaf4-8214-4a79-a673-c2b23f084b38" />
  <img width="300" alt="Sample photo 6" src="https://github.com/user-attachments/assets/6f165bc7-04f7-4809-89df-dad4568b2e50" />
  <img width="300" alt="Sample photo 7" src="https://github.com/user-attachments/assets/4815dfc8-fd7f-4b68-8855-3100038680f7" />
</p>

## Current build
<img width="350" alt="Current Raspberry Pi camera build" src="https://github.com/user-attachments/assets/0fa1a043-e4fe-44cd-b581-5edf5939938f" />

## 3D printed case prototype
<img width="450" alt="3D printed camera enclosure prototype" src="https://github.com/user-attachments/assets/df29fb9c-6b82-45ab-9d3a-7b00bcd2097e" />

I will commit the STL files once final design is approved and tested.

## Hardware
- Raspberry Pi 4, 4GB RAM
- M5Stack StickS3 (display, trigger)
- Waveshare 2.8 inch LCD Display Module and touchscreen (light meter, focus meter, trigger, exposure controls, battery meter)
- x728 UPS Shield + Geekworm X728-C1 Metal Case
- 4x 18650 Li-ion cells, estimated capacity around 10000mAh
- 64GB SD Card
- Arducam 8-50mm C-Mount Zoom Lens for IMX477
- Raspberry Pi Camera Module 3, Wide, IMX708 sensor

## Getting started

Full walkthrough: **[the setup guide](docs/setup.md)** — ordered by dependency, and it names the machine for every step. In brief:

More detail lives in the docs: the [network & deployment guide](docs/sticks3-remote.md)
(hotspot, Ethernet, service, rollback) and the [firmware README](firmware/sticks3/README.md)

| Last captured photo | Ready to capture |
| --- | --- |
| <img src="docs/images/sticks3-ready.jpg" alt="M5Stack StickS3 showing TV colour bars and the READY prompt" height="280"> | <img src="docs/images/sticks3-captured-photo.jpg" alt="M5Stack StickS3 displaying a captured room photo" height="280"> |

## From button press to displayed photo

The Pi captures, grades and stores. 
The M5Stack StickS3 is the wireless shutter and **last-capture display** — not a live viewfinder. It joins the Pi's own Wi-Fi hotspot and they talk over a token-authenticated HTTP API, so no home network or cloud is needed in the field. 
Ethernet/SSH is only for admin and photo transfer.

```text
Stick joins hotspot → checks readiness → shows READY
    ↓ button press
POST /v1/captures → Pi accepts (one at a time) → Stick sounds shutter
    ↓ colour bars while waiting
Pi grabs one frame → normalize → LUT → grain
    ↓
Pi saves original + graded JPEG + captures.jsonl → job complete
    └─ Stick polls, downloads the graded thumbnail, displays it
```

- **Ready** — the Pi app runs with its remote listener; the Stick joins the hotspot and polls `GET /v1/status` before showing READY.
- **Shutter** — a debounced press sends `POST /v1/captures` with a unique ID. The Pi runs one capture at a time (extra requests get a busy `409`); the shutter tone means "accepted", not "saved".
- **Feedback** — the Stick shows colour bars and polls `GET /v1/captures/{id}` while it works (two-screen mode shows the same bars on the Pi).
- **Capture + grade** — one frame becomes both outputs: normalize → chosen 3D LUT → grain. `--artifacts DIR` picks a different look.
- **Save** — to `~/Pictures/pifilm/YYYY-MM-DD/`: `*_original.jpg` (or `*_ungraded.jpg`), the full-res `*_graded.jpg`, and a `captures.jsonl` line (LUT hash, gains, grain seed, timings). The job completes only after these writes.
- **Preview to Stick** — the Stick fetches `GET /v1/captures/{id}/image.jpg`; the Pi returns a 240×135, ≤64 KiB letterboxed JPEG of the graded file. No re-grading on the Stick.
- **Display** — shown until the next capture. Full-res stays on the Pi (the Stick isn't the archive); failures show an error, not a fake success.
- **LCD too** — an optional Waveshare panel on the Pi (`--display waveshare28`) shows the same live view and shutter; a Stick shot appears on the LCD and an LCD shot appears on the Stick, since both go through the same controller.

**Running modes:**

| Mode | Command | Notes |
| --- | --- | --- |
| Headless (field) | `pifilm-capture --no-preview --remote-listen 0.0.0.0:8765` | battery, no monitor; the Stick still gets the preview |
| Two-screen | `pifilm-capture --show-captures` | run in a Pi desktop session; the Pi also shows colour bars + the graded photo |

Full config and rollback: the [deployment guide](docs/sticks3-remote.md).

## Using the tools directly

```bash
pifilm-process /path/to/originals /path/to/results   # batch re-grade a folder
pifilm-capture --no-preview --show-captures          # capture on the Pi
```

Capture modes:

| Flag | Behaviour |
| --- | --- |
| `--show-captures` | fullscreen window; `SPACE` grades a shot (colour bars while it works) and shows it until the next; `Q`/Esc quits |
| `--no-preview` | terminal controls, no window |
| `--fake` | synthetic frames, no camera (for testing) |
| `--device /dev/videoN` | select a specific USB camera |
| `--display waveshare28` | LCD viewfinder with exposure meter and touch shutter ([guide](docs/lcd-viewfinder.md)) |

With the default artifact, `pifilm-process` now grades any folder — including old USB captures — without white balance and with the damped highlight lift (see [Camera backends](#camera-backends)).

Output lands in `~/Pictures/pifilm/YYYY-MM-DD/`: `*_original.jpg` (or `*_ungraded.jpg`), `*_graded.jpg`, and a `captures.jsonl` line. The USB (V4L2) path captures a 1920×1080 MJPEG stream at 30 fps; window size does not change capture resolution.

### Camera backends

`--camera` picks the backend. With no flag, `--device` implies `v4l2`; otherwise Picamera2 is used when it imports, falling back to V4L2. Picamera2 and libcamera come from Raspberry Pi OS apt (never pip), so the venv uses `--system-site-packages`.

| Command | What it captures |
| --- | --- |
| `pifilm-capture --camera v4l2 --device /dev/video0` | USB (UVC) camera at 1920×1080 |
| `pifilm-capture --camera picamera2` | Pi Camera Module 3 (IMX708) at native 4608×2592 |

> The tuning file is libcamera's own automatic choice for the detected sensor;
> `--tuning-file imx708_wide.json` pins it, for a module variant whose tuning
> libcamera would not pick. Once Picamera2 is installed it becomes the default,
> so a Pi still on the USB camera must pass `--camera v4l2` until it is migrated. First-time bring-up
> (`rpicam-hello --list-cameras`, an RGB patch check, dimensions and metadata) is
> in [the Picamera2 bring-up checklist](docs/picamera2-bringup.md).

The bundled starter now assumes the IMX708 ISP's AWB (white balance off, clipping-aware lift on), so a USB/V4L2 camera should point `--artifacts` at a copy with `"white_balance": true` in its `params.json` — redeploying the bundled starter to a USB Pi would otherwise drop white balance entirely ([how-it-works](docs/how-it-works.md#at-capture-time-on-the-pi)).

**Picamera2 saves three files** per shot (two on V4L2):

- `*_original.jpg` — the ISP's own full-quality render (always this name).
- `<stem>.dng` — raw sidecar from the same frame (~18 MB; ~28 MB/shot total).
- `*_graded.jpg` — the graded result.

`captures.jsonl` also records `camera_metadata` (`ExposureTime`, `AnalogueGain`,
`Lux`, `LensPosition`, `AfState`, …) and the `dng` filename when present.

Capture flags:

| Flag | Action |
| --- | --- |
| `--autofocus {continuous,auto,manual}` | autofocus mode (default `continuous`) |
| `--af-range {normal,macro,full}` | autofocus range (default `normal`) |
| `--no-dng` | skip the raw sidecar for long sessions (keeps `_original.jpg`) |
| `--ae-lock` / `--awb-lock` / `--colour-gains R,B` | lock exposure / white balance for controlled reference shoots |
| `--ae-constraint {normal,highlight,shadows}` | AE constraint mode (default `normal`); `highlight` protects bright regions from clipping |
| `--ae-metering {centre,spot,matrix}` | AE metering mode (default `centre`) |
| `--ev STOPS` | exposure compensation, -8 to 8 (default `0`) |


The preset increases midtone color and contrast with a smooth tone curve,
compresses out-of-gamut chroma, and adds subtle grain (`0.004`). Rebuild it with:

```bash
.venv/bin/pifilm-preset --out artifacts/starter-v1
```

Pass `--highlights 0.6` to hold coloured highlights back from clipping; `0`
(the default) is the current look.

## Archiving photos to Nextcloud

Optional off-device backup: the Pi pushes everything under `~/Pictures/pifilm` to a Nextcloud folder with `rclone copy` over WebDAV (the rsync-equivalent for Nextcloud — plain `rsync` can't talk to it).

How it behaves:

- **Copy, never delete** — only adds/updates on Nextcloud, so cleaning the Pi's SD card never removes the cloud copies.
- **No-op off the wired LAN** — each run exits at once unless `eth0` has a home-LAN IP and Nextcloud answers a TCP connect, so a frequent timer is cheap.
- **Incremental** — syncs the originals, `_graded.jpg`, `.dng` raws and `captures.jsonl`, skipping anything already uploaded.

Scheduling is a **systemd timer, not cron**: `pifilm-nextcloud-sync.timer` runs the oneshot a few minutes after boot, then every 15 min (`OnUnitActiveSec=15min`, `Persistent=true`). Nothing runs until you install and enable it.

Setup, on the Pi:

```bash
sudo apt install rclone                                # one-time
# then fill NEXTCLOUD_* in .env — use a Nextcloud app password, not your login

.venv/bin/python scripts/nextcloud_sync.py --env .env          # dry run (no transfer)
.venv/bin/python scripts/nextcloud_sync.py --env .env --apply  # real copy

# enable the 15-minute timer (drops the .example suffix)
sudo cp deploy/pifilm-nextcloud-sync.service.example /etc/systemd/system/pifilm-nextcloud-sync.service
sudo cp deploy/pifilm-nextcloud-sync.timer.example   /etc/systemd/system/pifilm-nextcloud-sync.timer
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-nextcloud-sync.timer
systemctl list-timers pifilm-nextcloud-sync.timer      # confirm NEXT / LAST
```

Full guide (app password, every key, troubleshooting): [docs/nextcloud-sync.md](docs/nextcloud-sync.md).

## Training a reference-derived look

Training is optional — the camera works with the bundled starter. Point `pifilm-train` at two local image folders (source frames, and reference photos you have the rights to use); it fits a 33³ `.cube` LUT by colour-distribution transfer and regularised least squares, evaluates on held-out images, and writes a report with quality gates. A LUT only shifts global colour and tone.

- **How to train, gate by gate:** [the training guide](docs/training.md) (short outline in [setup §6](docs/setup.md#6-training-a-look-optional)).

## How the colour model works

The "model" is a **33 × 33 × 33 colour lookup table** (a standard `.cube`), not a neural network — fitted with classical colour statistics (distribution matching, regularised least squares, safety gates) and run on the Pi with no ML runtime.

- **At capture:** normalize → 3D LUT (trilinear) → film-like grain — a fixed pipeline in `pifilm/pipeline.py`.
- Normalisation is defined per artifact, not per camera: the bundled starter now targets the IMX708 (the default mount), with source white balance off and a clipping-aware highlight lift — see [the write-up](docs/experiments/2026-09-19-imx708-normalisation.md).
- **At training:** `pifilm-train` samples two image corpora in Oklab, reweights hues, matches distributions (Pitié IDT), fits the LUT by regularised least squares, then passes five numerical gates before publishing.

Full detail — the maths, the libraries, the dependency-by-role table, and what a LUT can't do — is in **[docs/how-it-works.md](docs/how-it-works.md)**.

## Known issues

Understood-but-unfixed limitations are tracked in [docs/known-issues.md](docs/known-issues.md) 

See [ORIGIN.md](ORIGIN.md) for the code's starting point and what is independent.
