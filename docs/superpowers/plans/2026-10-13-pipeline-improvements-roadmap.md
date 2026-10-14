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
  before it still regrade as they were (`pifilm-process` reproduces the Pi's
  graded file from `captures.jsonl`: exactly for a USB camera, within JPEG
  rounding of the original for the Pi camera).

## Where things stand (measured)

Pi 4 + IMX477, 12 MP, 2026-10-12, from `captures.jsonl`: `pipeline_ms` about
4,800 and `shutter_to_saved_ms` about 6,800 to 6,900.

One 12 MP frame on a laptop (proportions only, 2026-10-13):

| Stage | Gaussian grain | HP5 grain |
| --- | --- | --- |
| Normalise + LUT | 0.88 s | 0.75 s |
| Grain | 1.11 s | 1.80 s |
| JPEG encode, one file | 0.82 s | 0.74 s |

One shot on the Pi, by stage (step 1's `stage_ms`, 2026-10-13, one capture, HP5
grain, shutter to saved 6,838 ms):

| Stage | Time | Share |
| --- | --- | --- |
| Grain | 3,626 ms | 53 % |
| Capture: camera request | 431 ms | 6 % |
| Capture: array copy | 322 ms | 5 % |
| Capture: DNG extraction | 501 ms | 7 % |
| LUT | 825 ms | 12 % |
| Normalise | 328 ms | 5 % |
| Save original JPEG | 389 ms | 6 % |
| Save graded JPEG | 369 ms | 5 % |
| Save DNG | 27 ms | 0 % |

The parts add up to 6,827 ms, so nothing of a shot is unaccounted for. What it
says: grain is more than half of a shot and four times the LUT. Moving only the
LUT to the GPU would save at most 0.8 s; the GPU pays off only if grain (and
normalisation) move with it. Grain is therefore first, on the CPU, whatever the
spike says, since the CPU path stays as the reference and as `pifilm-process`.

After the byte-identical grain speed-up (PR #56), two captures on the Pi: grain
2,501 and 2,519 ms, shutter to saved 5,678 and 5,388 ms. What is left of grain,
timed part by part on the Pi (12 MP, HP5): two random fields about 660 ms, two
blurs 340 ms, three standard deviations 370 ms, two table lookups 280 ms, the
arithmetic 300 ms and more. No single part dominates, but everything that builds
the noise field (random numbers, blur, normalising) is over half of it.

GPU spike result (step 2, 2026-10-13, `scripts/experiments/gpu_lut_spike.py` on
branch `spike/gpu-lut`):

- It works: V3D 4.2, OpenGL ES 3.1, Mesa 26.2, a windowless (surfaceless) context,
  no new packages. Largest texture 4096, so the IMX477 fits and the IMX708 needs
  tiles.
- Memory is not a problem: free graphics memory stayed at about 497 MB with the
  12 MP textures loaded, and two full-resolution stills and a preview frame were
  taken with them loaded.
- Accuracy: never more than one level from the CPU reference; 0.04 % to 3.5 % of
  values differ by one.
- Speed: the LUT alone is no faster. GPU 890 ms against the CPU's 845 ms: the
  shader takes 118 ms, but packing RGB to RGBA (310 ms), reading back (180 ms)
  and unpacking (260 ms) cost the rest.

Verdict: GPU grading is worth building only as one pass that does normalise, LUT
and grain together, with the packing avoided (the camera can deliver four-byte
pixels; the JPEG encoder can read them). Then one upload, one shader and one
read-back replace about 3.6 s of CPU work. A GPU grain needs a noise texture, so
step 3's grain tile is designed to be that texture: one grain, two backends.

Tile grain (`two_scale_tile`, PR #57) on the Pi, 2026-10-13: grain 706 ms, and
1,266 ms on the first photograph, which also built the tile (now built at
start-up). That shot was 3,651 ms from shutter to saved: capture 1,099, LUT 808,
grain 706, the two JPEGs 714, normalise 290. From 6,838 ms at the start of the day.

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

Built and passed on the Pi (2026-10-14, PR #63). The owner chose warn only. The design is
[2026-10-14-tuning-identity-design.md](../specs/2026-10-14-tuning-identity-design.md),
the plan [2026-10-14-tuning-identity.md](2026-10-14-tuning-identity.md). The Pi
fingerprints its tuning at open (`pifilm/tuning.py`) and records it as `tuning_sha1`;
a look carries an optional `tuning_sha1`, written by `pifilm-tuning --stamp` or
`pifilm-train --tuning-sha1`; at start-up a difference is one journal line,
`tuning_matches: false` in every record and an amber `TUNING` tag on the viewfinder.
The deployed `imx477-2026-09-v4-dark` look has no fingerprint yet: it is stamped in
the hardware acceptance (`docs/lcd-viewfinder.md`, section 7).

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

Built: the design and its decisions are in
[2026-10-13-background-grading-design.md](../specs/2026-10-13-background-grading-design.md),
the implementation plan in
[2026-10-13-background-grading.md](2026-10-13-background-grading.md). The queue is
`pifilm/capture/gradequeue.py`; `--grade inline` switches it off.

Pi numbers, 2026-10-13, eight shots with the originals synced before the camera was
free: shutter to ready 2,537 to 3,689 ms, against 3,651 ms inline. The two syncs took
most of the gain: `save_dng` went from 27 ms to 557 to 1,114 ms and `save_original`
from 343 ms to 480 to 1,107 ms. The syncs then moved to the grade worker; without
them the same shots would have been ready in about 1.8 to 2.0 s (capture 1.2 to 1.5 s,
original 0.34 s, quick grade 0.18 s). To be measured again.

Behind the shutter the grade is slower than inline, as expected with the camera and
live view running beside it: shutter to saved 4.8 to 6.4 s.

### 7. GPU grading

Second spike, 2026-10-13 (`scripts/experiments/gpu_pass_spike.py`, branch
`spike/gpu-pass`): the whole grade in one pass takes about 665 ms on the Pi against
1,728 ms on the CPU, within 5 levels of the CPU's result. With the camera already
free after 1.8 to 2.0 s (background grading, measured the same day), this shortens
the background job by about a second. Design:
[2026-10-13-gpu-grading-design.md](../specs/2026-10-13-gpu-grading-design.md).

Built (2026-10-13): the plan is
[2026-10-13-gpu-grading.md](2026-10-13-gpu-grading.md). The pass, its NumPy
reference, the EGL device and the grader are in `pifilm/gpu/`. The grade worker
opens the grader on its own thread; `--grade-backend cpu` switches it off; records
carry `pipeline_backend`; `pifilm-gpu-check` is the hardware check. The hardware
acceptance table is in `docs/lcd-viewfinder.md`, section 7. Not built, by design:
dithering, tetrahedral interpolation and tiles for frames wider than 4096 (the
IMX708 is graded on the CPU). The grain does land in the same place as NumPy's for
the same `grain_seed`, because both paths place the same tile.

On the Pi, 2026-10-14 (PR #60, merged): every record `gpu-1`, the self-test passed at
every start. Grade 0.79 to 0.85 s (normalise 0.12, GPU 0.67 to 0.70: upload 215,
shader 250, read-back 183 ms) against 1.4 to 2.1 s on the CPU behind the shutter.
Shutter to saved 4.2 to 5.6 s; shutter to ready unchanged at 1.83 to 2.03 s. What is
left of the background job is mostly the SD card (syncing the original and DNG 0.85
to 1.6 s, the graded file 0.6 to 1.8 s). One preview stall during the test, the known
one, contained by the restart. Regraded with `pifilm-process` and compared with the
Pi's files: 0.7 to 2.1 levels on average, at most 18 to 27, nothing visible. That
includes the regrade starting from the quality-95 original rather than the frame the
Pi graded, so the Pi camera's photographs never regraded exactly, GPU or not.

One number in the design was corrected after the reference was measured: a GPU grade
is not always within 6 to 8 levels of the CPU regrade. On strongly saturated areas
it is 10 to 20 levels away, because the CPU rounds to bytes before a steep part of
the LUT. The GPU's file is the more exact one.

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
