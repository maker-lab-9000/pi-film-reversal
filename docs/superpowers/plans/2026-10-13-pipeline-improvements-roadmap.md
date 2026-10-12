# Pipeline improvements: roadmap

Date: 2026-10-13. Agreed with the owner in conversation, in response to an outside
review of the capture and grading pipeline.

## How this work is organised

- Integration branch: `pipeline-improvements`, cut from `main` on 2026-10-13.
- Each step is its own branch off `pipeline-improvements`, merged back by a pull
  request whose base is `pipeline-improvements`. `main` gets one merge when the
  owner decides the set is ready (or at agreed points along the way).
- Steps 1 to 3 do not change how photographs look. Steps 4 and 5 do, and stay
  experiments until the owner has judged a gallery (looks are decided from many
  rendered examples, not from metrics).
- Every step that changes pixels gets a new version marker, so photographs taken
  before it still regrade exactly as they were (`pifilm-process` reproduces the
  Pi's graded file from `captures.jsonl`).

## Where things stand (measured)

Pi 4 + IMX477, 12 MP, 2026-10-12, from `captures.jsonl`: `pipeline_ms` about
4,800 and `shutter_to_saved_ms` about 6,800 to 6,900.

One 12 MP frame on a laptop (proportions only, 2026-10-13):

| Stage | Gaussian grain | HP5 grain |
| --- | --- | --- |
| Normalise + LUT | 0.88 s | 0.75 s |
| Grain | 1.11 s | 1.80 s |
| JPEG encode, one file | 0.82 s | 0.74 s |

About two seconds of the Pi's shutter-to-saved time are outside the grading
pipeline (mode switch and capture, DNG extraction, two JPEG encodes, file writes)
and are not broken down anywhere yet.

## Steps

### 1. Measure: per-stage timings in `captures.jsonl`

Branch `perf/stage-timings`. Record, for every capture, how long each stage took
(`stage_ms`): camera request, array copy, DNG extraction, composite (double
exposure), normalise, LUT, grain, each file write. `pipeline_ms` and
`shutter_to_saved_ms` stay. No change to pixels.

Done when: a capture on the Pi shows `stage_ms`, and its parts add up to
`shutter_to_saved_ms` within a few percent. The Pi numbers then set the order of
step 2.

### 2. Speed without changing the look

Branch per change, each judged against step 1's numbers.

- **Grain.** The largest stage. Candidates: a pre-generated grain tile placed with
  seeded offsets, cheaper random generation, float32 throughout. Any change in
  the grain's pixels is a new grain model name and params version, so existing
  artifacts and old records keep the present code path. Checked by a gallery at
  100 % as well as by time.
- **File writes.** Measure first. Pillow here already uses libjpeg-turbo, so the
  lever is settings (quality 95, 4:4:4), which is a quality decision for the owner.
- **DNG extraction** and the second JPEG, if step 1 shows they matter.

### 3. Integrity: tuning-file identity

Branch `feat/tuning-hash`. Record a hash of the libcamera tuning in use in each
capture record and in the artifact a look was trained against; warn (or refuse,
owner's choice) when they differ. This turns the known issue "a libcamera update
could silently retune a deployed trained LUT" into a check.

### 4. Look experiments (offline, gallery-judged)

No change to defaults until the owner has chosen from a gallery.

- **Exposure from metadata.** Today normalisation stretches levels and moves the
  median from image statistics. That pulled night scenes toward mid-grey, and it
  now works against a deliberate manual exposure. Experiment: derive the
  exposure placement from `Lux`, `ExposureTime` and `AnalogueGain` in the record
  instead, and compare on existing captures (night, snow-like bright scenes,
  manual exposures) with `pifilm-process`.
- **Adaptive contrast off.** libcamera's `rpi.contrast` enhancement is
  scene-dependent, which one global LUT cannot follow. Experiment: capture pairs
  with it on and off, grade both, compare. It changes what the LUT is fed, so a
  retrain may follow.

### 5. Background grading

Branch `feat/background-grade`. Save the original and the DNG, show a fast
low-resolution grade for review and for the Stick, and grade the full frame on a
queue behind the shutter, so the next photograph does not wait for the last. Needs
decisions on queue depth and memory, what a power loss leaves behind (a journal of
pending grades), double exposure, and what the Stick is told. Gets its own design
document before any code.

### 6. GPU grading (only if steps 2 and 5 are not enough)

The Pi 4's GPU can do the LUT lookup in hardware. It needs a graphics context
beside the display driver and ends pixel-exact agreement between the Pi and
`pifilm-process` on a Mac, so it waits until the cheaper steps have been measured.
Gets its own design document and a hardware spike first.

## Considered and not planned

- **Grading from the raw stream.** Right in principle (the LUT currently sits on
  the ISP's own tone curve), but it means rebuilding demosaic, lens shading,
  white balance, colour matrix and denoise, and retraining every look, including
  the approved `imx477-2026-09-v4-dark`.
- **A per-sensor input LUT plus a shared look LUT.** Only pays off with one look
  across two sensors.
- **A log shaper and tetrahedral interpolation.** For linear input; the LUT input
  is 8-bit sRGB. Tetrahedral would cost the Pillow fast path for a small gain.
- **simplejpeg.** Pillow already links libjpeg-turbo.

## Corrections to the outside review

- Its 3.3 s baseline is the Pi 3B-to-Pi 4 migration figure; the current pipeline
  takes about 4.8 s.
- Background grading would not have prevented the 2026-10-10 preview stall, which
  was Picamera2 not delivering preview frames and did not involve grading.
