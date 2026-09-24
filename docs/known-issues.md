# Known issues

Understood-but-unfixed defects and hardware limitations. Each entry records the
symptom, the verified root cause, what was ruled out, the current decision, and
how to confirm it is resolved.

## IMX708 second-capture "Camera frontend has timed out" on the Raspberry Pi 3B

**Status:** **resolved 2026-09-19** by the move to the Raspberry Pi 4. No code
change was needed and native 12 MP capture was kept — see
[Resolution](#resolution) for the measurements. The entry stays because the
failure mode is instructive and a Pi 3B is still affected.

**First seen:** 2026-09-18, Raspberry Pi 3B + Camera Module 3 Wide (IMX708),
`pifilm-capture` on `main`.

### Symptom

The first capture succeeds; the second (and later) captures log a libcamera
frontend timeout and can stall:

```
WARN  V4L2 /dev/video0[..:cap]: Dequeue timer of 1000000.00us has expired!
ERROR RPI  Camera frontend has timed out!
ERROR RPI  Please check that your camera sensor connector is attached securely.
ERROR RPI  Alternatively, try another cable and/or sensor.
```

In an interactive run both photos still saved (~20 s each); under the systemd
service / remote (Stick) path the stall escalated to a capture that never
returned (observed waiting >150 s).

### Root cause (verified)

It is **CPU/processing time at full resolution**, not memory and not the DNG:

- The camera is configured for the IMX708's native **4608×2592 (~12 MP)** with an
  RGB grading stream plus a raw stream (`pifilm/capture/picamera.py`,
  `create_still_configuration(..., buffer_count=2)`).
- Grading one 12 MP frame — normalize → 3D LUT → grain — takes **~20 s on the
  Pi 3B** (measured: `Saved … in 20394 ms`). During that grade the capture loop
  does not service the camera, so libcamera's 1-second dequeue watchdog on the
  CSI frontend expires and logs "frontend has timed out". The free-running camera
  then stalls; sometimes it recovers (the photo still saves), sometimes the next
  `capture_request()` blocks — the >150 s hang.

### Ruled out during diagnosis

- **DNG / raw write.** Reproduced with `--no-dng`; the raw stream is still
  configured and the second capture still timed out, so writing the DNG is not
  the trigger.
- **CMA / RAM.** The 256 MiB CMA pool had ~68 MiB free and `dmesg` showed no
  allocation failures; both photos saved. Memory pressure is not the cause.

### Environment

Raspberry Pi 3B, Raspberry Pi OS Trixie, libcamera 0.7.2, picamera2 0.3.37,
IMX708 (`imx708_wide`) at 4608×2592, `buffer_count=2`, CMA 256 MiB shared with
`vc4-kms-v3d`.

### Resolution

Measured on a Raspberry Pi 4 Model B running the migrated SD, 2026-09-19, with
the same 4608×2592 configuration and `buffer_count=2`:

| Metric | Pi 3B | Pi 4 |
|---|---|---|
| Grade one 12 MP frame (`pipeline_ms`) | ~20,400 ms | **~3,290 ms** |
| Shutter to saved (`shutter_to_saved_ms`) | — | **~5,300 ms** |
| `Camera frontend has timed out` | on the second capture | **0 across every capture in the boot** |

The Pi 4 grades a 12 MP frame roughly six times faster, so the capture loop no
longer leaves the camera unserviced long enough to trip libcamera's one-second
dequeue watchdog. Repeated captures complete with consistent timings
(3286.7 / 3284.8 / 3303.7 ms across three shots).

Native 12 MP is therefore kept, and the configurable grading resolution
considered during diagnosis was **not** needed.

### If it regresses — the fallback

Stop grading at full 12 MP: make the capture/grading resolution configurable (for
example a half-res `2304×1296` binned stream, or `1920×1080`), keeping full
resolution only for the saved original and DNG. Grading 2–3 MP is several times
faster, which closes the camera-servicing gap that trips the watchdog. Raising
`buffer_count` and/or CMA is a secondary lever where there is room.

## A libcamera update could silently retune a deployed trained LUT

**Status:** understood, mitigated by a documented deployment step, not enforced
in code.

**Description:** the Picamera2 backend defaults `--tuning-file` to `None`,
meaning libcamera's own automatic tuning choice for the detected sensor
(recorded as `tuning_file: auto:<model>` in `captures.jsonl`); the example
systemd unit runs with this default. A trained LUT is fitted against whatever
colour science its source corpus was captured under. If the automatic tuning
file libcamera picks ever changes — an OS or libcamera package update, or a
sensor firmware/variant change read differently — a deployment running the
default would start feeding a **different** colour response through an
unchanged LUT, with no error and no changed `lut_sha1` to flag it.

**Mitigation:** when deploying a trained (not the bundled starter) LUT, pin
`--tuning-file` to the exact file the corpus was captured under (see
[docs/training.md](training.md), Step 7). This is a documentation-only
safeguard; nothing currently checks that a running service's tuning file
matches what a deployed artifact was trained against.

## Capture records written before 2026-09-19 cannot be told apart by `lut_sha1`

**Status:** understood, not fixed retroactively. New records carry the missing
field; old ones cannot be rewritten without guessing.

**First seen:** 2026-09-19, reviewing the Phase 6 normalisation freeze
([write-up](experiments/2026-09-19-imx708-normalisation.md)).

### Symptom

Two `captures.jsonl` records — one from before the Phase 6 freeze, one from
after — can carry the **same `lut_sha1`** and still have been graded
differently. The bundled starter's LUT did not change in Phase 6; its
normalisation did (`white_balance` true -> false,
`levels_lift_highlight_ref` null -> 0.02).

### Root cause

`lut_sha1` is the content hash of the `.cube` only. Normalisation lives in the
artifact's `params.json` and was never hashed into the record, so the record
identified half of the grade. `Pipeline.process` now also records
`normalize_sha1` (a SHA-1 of `NormalizeParams.to_dict()`, sorted keys), and
`CaptureSession` writes it into every line — but only for captures taken from
2026-09-19 onward.

### Workaround for old records

The record still carries the *applied* `wb_gains`, `exposure_gain` and
`levels`, which is what the normaliser actually did to that frame. Feeding
those values back reconstructs the grade without knowing which parameter set
produced them; the missing piece is only the parameter identity, not the
result.

### How to confirm it is resolved for new captures

`normalize_sha1` is present in every new `captures.jsonl` line
(`tests/test_app.py::test_log_line_carries_full_provenance`) and differs
between two artifacts whose normalisation differs
(`tests/test_pipeline.py::test_normalize_sha1_identifies_the_normalisation`).
