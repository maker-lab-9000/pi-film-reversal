# Pipeline improvements: roadmap

Date: 2026-10-13. Agreed with the owner in conversation, in response to an outside
review of the capture and grading pipeline.

## How this work is organised

- Integration branch: `pipeline-improvements`, cut from `main` on 2026-10-13.
- Each step is its own branch off `pipeline-improvements`, merged back by a pull
  request whose base is `pipeline-improvements`. `main` gets one merge when the
  owner decides the set is ready (or at agreed points along the way).
- Steps 1, 3 and 4 do not change how photographs look. Step 5 does, and stays
  an experiment until the owner has judged a gallery (looks are decided from many
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
`shutter_to_saved_ms` within a few percent. The Pi numbers then say how much of
a shot the GPU can touch.

### 2. GPU spike (hardware, throwaway)

Moved up on 2026-10-13: the owner wants GPU grading for the speed. Before any
design, a script on the Pi uploads a 12 MP frame, applies a 33-point LUT in a
shader, reads the result back, and reports:

- time for upload, lookup and read-back, against step 1's CPU numbers;
- graphics memory used, and whether a full-resolution still capture still works
  while the textures are loaded (the GPU, the display and the camera share one
  reserved memory pool on the Pi 4);
- the largest texture the GPU accepts (4096 is expected: the IMX477 at 4056 fits,
  the IMX708 at 4608 does not);
- whether a windowless graphics context runs beside the display the app owns;
- how far the result is from the CPU reference, pixel by pixel.

Nothing from the spike is kept as product code. Its numbers decide whether step 3
speeds grain up on the CPU or goes straight to the GPU, and feed step 7's design.

Known costs of GPU grading, accepted by the owner as the price of the speed, to
be designed around: it speeds up only the grading part of a shot; the Pi and
`pifilm-process` on a Mac stop agreeing pixel for pixel (the CPU path stays as the
reference and the tests become "within one level"); two grading paths to keep in
step; the GPU path can only be tested on the Pi; a new dependency on the Pi's
graphics driver.

### 3. Speed without changing the look

Branch per change, each judged against step 1's numbers and step 2's verdict on
where grain should run.

- **Grain.** The largest stage. Candidates: a pre-generated grain tile placed with
  seeded offsets, cheaper random generation, float32 throughout. Any change in
  the grain's pixels is a new grain model name and params version, so existing
  artifacts and old records keep the present code path. Checked by a gallery at
  100 % as well as by time.
- **File writes.** Measure first. Pillow here already uses libjpeg-turbo, so the
  lever is settings (quality 95, 4:4:4), which is a quality decision for the owner.
- **DNG extraction** and the second JPEG, if step 1 shows they matter.

### 4. Integrity: tuning-file identity

Branch `feat/tuning-hash`. Record a hash of the libcamera tuning in use in each
capture record and in the artifact a look was trained against; warn (or refuse,
owner's choice) when they differ. This turns the known issue "a libcamera update
could silently retune a deployed trained LUT" into a check.

### 5. Look experiments (offline, gallery-judged)

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

### 6. Background grading

Branch `feat/background-grade`. Save the original and the DNG, show a fast
low-resolution grade for review and for the Stick, and grade the full frame on a
queue behind the shutter, so the next photograph does not wait for the last. Needs
decisions on queue depth and memory, what a power loss leaves behind (a journal of
pending grades), double exposure, and what the Stick is told. Gets its own design
document before any code.

### 7. GPU grading

Wanted by the owner (2026-10-13). Built from step 2's measurements, with its own
design document: where the context lives, what is done on the GPU (normalise,
LUT, grain), the fallback to the CPU path, how memory is kept clear of the
camera's still capture, the IMX708's oversize frame, and the tolerance the tests
hold it to. A graded live view becomes possible once it exists.

Quality requirements for the design (from a second outside note, 2026-10-13,
checked against this codebase):

- Stay in float from input to output and round once, with dithering. Today the
  frame is rounded to 8 bits after normalisation and again after the LUT.
- Fetch LUT texels with `texelFetch` and interpolate in `highp` float in the
  shader; never the GPU's built-in filtering, whose blend weights are coarse.
  With `texelFetch` there is no texel-centre offset to get wrong.
- LUT texture in half or full float, never RGBA8.
- Tetrahedral interpolation is then a few lines; whether to adopt it is a look
  decision (it changes pixels against the trilinear reference), judged by gallery.
- Record the backend and its version in `captures.jsonl` (`pipeline_backend`), so
  a regrade knows which path made a photograph. GPU grain will not match NumPy's
  for the same `grain_seed`.
- Frames wider than the GPU's largest texture are graded in tiles; the LUT has no
  seams, grain and any neighbourhood effect need overlap.

Two statements in that note do not hold here: grain is already scaled by
luminance (both grain models have an envelope), and there is no "preview equals
capture" guarantee to keep, because the live view is ungraded today. A graded
live view from the same shader would be new.

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
