# IMX708 / Picamera2 bring-up

The camera has arrived; hardware acceptance is not yet recorded. This checklist
separates the local fake-backed software tests from tests on the actual Pi 4.
See [roadmap progress](superpowers/plans/2026-09-13-rpi4-imx708-progress.md).

## 1. Identify the board and camera

With the Pi powered off, connect the camera using the cable/connector procedure
for the delivered module, then boot the Pi. Record the product/variant from its
packaging: an IMX708 sensor alone does not identify the lens or tuning file.

Run on the Pi and retain the output with the hardware test record:

```sh
tr -d '\0' < /proc/device-tree/model
cat /etc/os-release
uname -a
rpicam-hello --list-cameras
/usr/bin/python3 -c 'from importlib.metadata import version; print("Picamera2", version("picamera2"))'
dpkg-query -W 'python3-picamera2' 'python3-libcamera' 'libcamera*' 'rpicam-apps*'
systemctl is-active pifilm-capture.service
```

If `rpicam-hello` is unavailable, inspect installed camera packages first; an
older OS may use the `libcamera-hello` command name. Do not assume the hardware
is absent from a missing command. Record detected model and supported modes,
including native dimensions and bit depth. No fixed FPS or raw bit depth is
assumed by the new backend.

Picamera2 must use the libcamera build supplied with the OS. Install the OS's
`python3-picamera2` package if missing, and expose system packages to the project
venv using `--system-site-packages`, as in the [setup guide](setup.md). Do not
install Picamera2 from pip into an otherwise isolated venv. Confirm imports
using the same interpreter that will run the service.

### IMX477

The Sony IMX477 (Raspberry Pi HQ Camera) is a fixed-lens sensor: it has no
focus motor, so `AfMode` is absent from its libcamera control set and the
backend records `"autofocus": "none"` in `camera_metadata` — a non-default
`--autofocus`/`--af-range` on this sensor is rejected as an error, not silently
ignored. Native resolution is 4056 × 3040, 12-bit raw
(`SBGGR12_CSI2P`). `--tuning-file` is unnecessary: the backend is
sensor-agnostic and libcamera selects its own tuning file for the detected
sensor by default (recorded as `tuning_file: auto:imx477`), the same as for
the IMX708. The [DNG acceptance test](#dng-acceptance-test) below applies
unchanged: its red-is-red check is exactly what catches a wrong
colour-filter-order tag on a new sensor.

## 2. Establish tuning and exclusive camera ownership

Confirm the module's tuning file before acquiring it. The roadmap assumes
`imx708_wide.json`; a standard, NoIR or third-party module may require a
different file. Use the module/vendor guidance and installed libcamera tuning
files to select it. Keep the chosen filename in the capture log.

Check whether an existing capture service or preview owns the camera. Stop the
known owner before manual capture, then restore the intended service after the
test. Do not launch a second camera application to work around an acquisition
error. A Pi still running the USB setup should select `--camera v4l2` explicitly
once the new code is installed: availability of Picamera2 can change automatic
backend selection.

## 3. Phase 1 capture test

After the new backend is installed on the identified Pi, run from its checkout.
This example assumes the Wide variant has been confirmed:

```sh
.venv/bin/pifilm-capture --camera picamera2 --tuning-file imx708_wide.json \
  --no-preview --out ~/Pictures/pifilm-imx708-bringup
```

Use a terminal, press SPACE to capture, and Q to quit. Take a frame containing
red, green and blue objects plus a neutral patch; repeat several captures to
check that requests are returned and the camera does not stall. Confirm:

- `_original.jpg` (plus `.dng`) and `_graded.jpg` both have 4608 × 2592 pixels.
- Red and blue are not swapped. Picamera2 calls the BGR byte layout `RGB888`;
  the backend must convert it to the pipeline's RGB convention.
- `captures.jsonl` records the actual raw sensor mode, bit depth and tuning
  filename; a measured frame duration supplies FPS rather than a preset guess.
- The graded frame loads and shows the current artifact's look; this is not an
  IMX708-trained look yet.
- Normal shutdown releases the camera so a second run can acquire it.
- When the usual authenticated remote service is tested, the Stick receives
  its existing thumbnail. Do not expose an unauthenticated test listener.

Phase 1 originally produced `_ungraded.jpg`, not an original JPEG/DNG pair,
before autofocus control, per-shot exposure/AWB metadata and retained request
saves landed in Phases 2–3 below; this same command now produces the Phase 2/3
output described in Section 4 (`_original.jpg` + `.dng` + `_graded.jpg`). Do not
label these files as the production training corpus before the capture
pipeline and source/runtime input boundary are fixed.

## 4. Phase 2/3 capture test: metadata, autofocus, original + DNG

Phases 2 and 3 are implemented and covered by fake-backed tests (see
`tests/test_picamera.py`); the hardware acceptance below is still pending.

The Picamera2 backend now defaults to continuous autofocus and auto
exposure/AWB (there is no flash mode, so both stay auto unless a controlled
shoot needs otherwise), and every capture writes three files instead of one:

```sh
.venv/bin/pifilm-capture --camera picamera2 --tuning-file imx708_wide.json \
  --no-preview --out ~/Pictures/pifilm-imx708-phase23
```

- `_original.jpg`: the ISP's own 8-bit rendering, always saved under this name
  — a Picamera2 frame is the camera's own full-quality capture whether or not
  a DNG sidecar is also saved.
- `<stem>.dng`: a raw sidecar saved from the same capture request, enabled by
  default. Pass `--no-dng` to skip it for long sessions; it costs storage
  (about 18 MB DNG at 4608 × 2592 10-bit raw, roughly 28 MB per shot in total
  once the two JPEGs are added). `--no-dng` only omits this sidecar; it does
  not change the original's name.
- `_graded.jpg`: the graded output, as before.

`captures.jsonl` gains two keys when the frame carries them: `camera_metadata`
(a filtered, JSON-serialisable subset of the request's metadata — for
example `ExposureTime`, `AnalogueGain`, `Lux`, `LensPosition`, `AfState`) and
`dng` (the sidecar's filename, only present when one was written).

For a controlled reference-matching shoot, AE and AWB can be locked instead of
left auto: `--ae-lock` and `--awb-lock` disable the algorithm at whatever value
it holds right before `start()` — the initial, pre-convergence value, not a
settled one — and `--colour-gains R,B` fixes explicit, known gains (which also
disables AWB) rather than relying on whatever AWB happened to land on.
`--autofocus {continuous,auto,manual}` and `--af-range {normal,macro,full}`
control the lens; the default is continuous AF at normal range.

Three more controls, all Picamera2-only (rejected on V4L2, `--device` and
`--fake`, and their defaults reproduce today's behaviour exactly):
`--ae-constraint {normal,highlight,shadows}` (default `normal`) sets
libcamera's `AeConstraintMode`; `highlight` protects bright regions from
clipping instead of exposing for the average scene. `--ae-metering
{centre,spot,matrix}` (default `centre`) sets `AeMeteringMode`. `--ev STOPS`
(default `0`, range -8 to 8) sets `ExposureValue`, always applied so 0 is an
explicit, deterministic neutral rather than an unset control. The effect of a
non-default value shows up in the already-recorded `ExposureTime` and `Lux`
in `camera_metadata`.

Confirm on real hardware:

- The DNG passes the [DNG acceptance test](#dng-acceptance-test) below.
- Repeated captures at different subject distances move `LensPosition` in
  `camera_metadata`, and `AfState` reflects the autofocus state machine
  (searching vs. focused).
- `ExposureTime`, `AnalogueGain` and `Lux` in `camera_metadata` are plausible
  for the scene, and `--no-dng` reliably removes the `.dng` file while the
  original is still saved as `_original.jpg`.
- With `--ae-lock`, confirm the locked frame's exposure matches an unlocked
  frame of the same scene before relying on it — the lock freezes whatever
  value AE held before capture started, which is not guaranteed to be settled.
- Shoot a 171656-type strong-light scene (bright background metered against a
  shaded foreground, the kind that blows highlights under default
  centre-weighted AE — see
  [the normalisation write-up](experiments/2026-09-19-imx708-normalisation.md#what-this-does-not-fix))
  with `--ae-constraint highlight` and confirm the `_original.jpg` keeps its
  highlights, by comparing `ExposureTime`/`Lux` in `camera_metadata` against a
  default (`--ae-constraint normal`) shot of the same scene.

### DNG acceptance test

The DNG is the archival, re-processable source for later work (Phase 5
retraining, reprocessing experiments). The saved JPEGs come from the ISP and do
not depend on the DNG's tags, so a subtly wrong DNG breaks nothing today — but
every future raw-based experiment would be built on bad data. Check it once,
here.

**Test shot.** In even daylight, photograph a scene containing a white or grey
sheet of paper, something clearly **red**, and something clearly **blue**. One
frame then exposes white balance, channel order and black level at a glance.
Open its `<stem>.dng` and work down this table.

| Check | Pass | A failure means |
| --- | --- | --- |
| Opens; pixel dimensions | Opens cleanly at **4608 × 2592** | Will not open, half size or garbled → PiDNG or stride problem |
| Is the red object red? | Correct colours | **Red and blue swapped** → wrong CFA pattern tag. The sensor is **BGGR**; tagged RGGB, skin goes cyan and sky goes orange. The most likely failure |
| "As shot" white balance | Greys and whites look roughly neutral | A heavy **green cast** that as-shot WB does not fix → `AsShotNeutral` or the colour matrix is missing or wrong |
| Shadows | Fall to clean black, no colour tint | Milky, or **magenta/green shadows** → wrong `BlackLevel` |
| Highlights | Clip to white | Clip to **pink or cyan** → wrong per-channel white level |
| 100 % zoom | Smooth grain, no pattern | **Vertical stripes, banding or a screen-door texture** → the 10-bit `SBGGR10_CSI2P` packed data is being unpacked wrongly |
| Against `<stem>_original.jpg` | Same framing, same moment, same focus; colours in the same family | A different frame → the DNG did not come from the same capture request |
| EXIF in the raw developer | Matches that capture's `captures.jsonl` record (`ExposureTime`, `AnalogueGain`, colour temperature) | Mismatch → the metadata is not reaching the DNG |

**Expected, and not a fault:**

- The raw looks **flat and desaturated** beside the graded JPEG. That is
  correct: the ISP is deliberately neutralised (sharpness, contrast and
  saturation at 1.0) and the JPEG additionally carries the tone curve and LUT.
- **darktable applies its own scene-referred tone mapping by default**, so it
  will not match the JPEG. Judge the raw with a minimal profile; RawTherapee's
  **Neutral** profile is the easier choice for this check.

**Pass criteria:** opens at full size, red is red, as-shot white balance looks
neutral, blacks are clean, no repeating pattern at 100 %, and it is recognisably
the same frame as the original JPEG. Record the result in the progress document.

## 5. Record acceptance and collect the pilot later

Append the actual board/OS/packages, camera product, tuning, command, saved
sizes, colour observation, repeated-capture result and remaining failures to
the separate progress document. Passing fake-backed tests is not evidence of
hardware acceptance.

After Phases 1–3, the first useful photo handoff is 20–30 pilot originals across
8–10 scenes with capture logs and scene IDs. Use them to choose lighting and
normalisation before collecting the production corpus. The complete collection
and retraining gates are in the [roadmap](superpowers/plans/2026-09-11-rpi4-imx708-roadmap.md#when-new-camera-source-photos-are-needed).

## API references

- [Official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [Picamera2 project and installation guidance](https://github.com/raspberrypi/picamera2)
- [Raspberry Pi camera software documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)

Upstream APIs were checked on 2026-09-14; compare them with the versions actually
installed on the Pi before marking hardware acceptance complete.
