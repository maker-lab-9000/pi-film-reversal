# How the colour model works

There is no neural network in this project. The "model" is a 33 × 33 × 33
colour lookup table, the same `.cube` format that DaVinci Resolve or Photoshop
would open, fitted with classical colour statistics: distribution matching,
regularised least squares and a handful of safety checks. The training data is a
few hundred images with no pairs between camera frames and reference
photographs, and the result has to run on a Raspberry Pi 4 with no
machine-learning runtime installed.

## At capture time, on the Pi

Every graded frame goes through three steps, in a fixed order, in
`pifilm/pipeline.py`:

1. **Normalisation** (`pifilm/normalize.py`). Per-frame white balance (when the
   artifact enables it) and an exposure or levels correction, computed in
   linear light and applied to
   the luma channel only, so contrast changes do not inflate saturation. It
   compiles into three 256-entry tables plus one for tone, applied with
   OpenCV's `LUT` and two `cvtColor` calls: about 100 ms for 1080p on the Pi.
   The same code, in floating point, prepared every training image, so the LUT
   sees the input distribution it was fitted on. White balance and the levels
   tone lift are per-artifact settings, not per-camera ones: the bundled
   starter targets the IMX708 (the default mount), whose ISP already runs its
   own AWB, so its `params.json` turns source white balance off and caps the
   levels lift by `levels_lift_highlight_ref` — the fraction of a frame's
   pixels already at the highlight ceiling — so a frame that is already
   clipped is not lifted further while a genuinely dark frame keeps its full
   lift. On a 108-shot IMX708 pilot this took outdoor shots' median tone
   gamma from a 0.577 lift to 1.0 (no lift). The indoor lift is also largely
   removed at this setting (median gamma 0.764 to 0.975); `ref=0.05` would have
   kept more of the indoor lift (0.856) but only partially damps the worst
   shots, and the user chose 0.02 to have those fully corrected. See
   [the write-up](experiments/2026-09-19-imx708-normalisation.md) for the full
   numbers and the trade-off made. A USB/V4L2 camera has no ISP AWB, so it needs
   an artifact with white balance back on. There is no flag for this:
   copy the artifact directory (or write a fresh one with `pifilm-preset --out
   DIR`), then edit its `params.json` — set `"white_balance": true` under
   `normalize` and delete (or set to `null`) `"levels_lift_highlight_ref"` —
   and pass `--artifacts DIR`. Editing `params.json` is safe here because
   `lut_sha1` covers only the `.cube`, not the normalisation block.
2. **The 3D LUT** (`pifilm/lut.py`). Trilinear interpolation over the 35,937
   nodes, executed by Pillow's `ImageFilter.Color3DLUT` in C with 16-bit fixed
   point. A NumPy reference implementation exists for tests and the trainer.
3. **Grain** (`pifilm/grain.py`). Gaussian noise on luminance only, blurred by
   `blur_sigma` so it clumps like film rather than looking like sensor noise,
   scaled by `4Y(1 − Y)` so it vanishes in deep shadow and blown highlights.
   The seed is recorded per capture, so any graded file can be regenerated
   from its original.

Manual controls (the viewfinder's shutter, ISO and white-balance cells) fix the
exposure time, the analogue gain or the colour temperature. A fixed shutter leaves
gain to the Pi's auto-exposure, which meets its brightness target with gain alone;
with shutter and ISO both fixed there is no auto-exposure and the meter is the guide.
There is no exposure compensation (removed 2026-10-11). The grade does not change: a
frame that ran out of gain (`MAX` in the `ISO` cell) is normalised like any dark frame.

### The order of a shot

The three steps above take about 2 s on a full-resolution frame on the Pi 4. By
default (`--grade background`) the camera does not wait for them:

1. The frame is read and the original JPEG is saved (and the DNG, when enabled),
   The grade worker forces both onto the disk as its first step, usually within a
   second; a power cut before that can lose the photograph. The camera does not wait
   for it, because on the Pi's SD card that cost up to a second a shot.
2. A pending entry is written: a small JSON file outside the photo folder
   (`~/.local/state/pifilm/pending/`, or `--state-dir PATH`) with the original's
   name, the grain seed, the EV and the camera fields of the record.
3. A quick grade is made for the review screen and the Stick: the frame reduced
   by a whole factor to about 1,000 pixels on the long side (four times each way,
   1014 × 760, from the IMX477), through the same pipeline without grain. The camera is now free for the next photograph.
4. Behind the shutter, one worker thread grades the full frame, saves
   `*_graded.jpg`, appends the record to `captures.jsonl` and deletes the pending
   entry, in that order. Photographs are graded in the order they were shot.

The record is therefore written when the graded file is saved, not when the shutter
is pressed. It gains `shutter_to_ready_ms` (shutter to the camera being free),
`grade: "background"`, and the stages `preview` and `queued` in `stage_ms`;
`shutter_to_saved_ms` now includes the time spent waiting in the queue. The graded
file is byte for byte the one inline grading would write: same frame, seed and
pipeline.

The quick grade is not the final file. It has no grain and its normalisation is
measured on the reduced frame, so its tone can differ slightly. It is kept in a
private temporary directory, never in the photo folder.

At most three grades wait at once, each holding its frame in memory. When the queue
is full the next capture waits for a place. The viewfinder shows how many
photographs are still developing.

**Recovery.** At start-up the pending entries that belong to the same output folder
are finished first, on the same worker. A recovered grade starts from the saved JPEG,
not from the frame in memory, so it can differ from the uninterrupted grade by JPEG
rounding; its record carries `recovered: true` and has no shutter timings. Just after
a restart with many unfinished grades, the recovered ones go first: three new
photographs can be taken at once and a fourth waits until they are done. A grade
that fails is reported in the log by its entry's name; an entry that fails twice is
renamed `*.failed` and a second line says so.

**Stopping.** A stop finishes the photographs taken in this run (the grade that is
running and those waiting) before the process exits, with or without the LCD
viewfinder: SIGTERM (how systemd stops the service) ends the loops the way Ctrl-C
does, and once the clean-up has begun a further SIGTERM is ignored. Recovered
grades that have not started are not waited for: their entries stay and the next
start finishes them, so a stop is never longer than this run's queue. What is not
finished, because the power went or the process was killed, is also completed at
the next start.

**Order in the log.** Records are written when a grade is saved, so they are in
shot order. One exception: exposure 1 of a double is logged at once, so its line
can land ahead of the photographs still waiting to be graded (up to four: the one
being graded and three waiting).

`--grade inline` restores the earlier order: the full grade is saved before the
camera is free, there is no queue and no pending entry, and records have none of the
new fields above. They do have `pipeline_backend`, always `cpu` (see the next section).

### Grading on the GPU

On the Pi 4 the grade worker can do the three steps as one pass on the GPU
(`pifilm/gpu/`). Measured on 2026-10-13 with the IMX477 at 12 MP: about 0.67 s
against 1.73 s on the CPU. The camera is already free by then, so the wait between
photographs does not change. The graded file exists about a second sooner and a
burst drains faster.

**What runs where.** The GPU grades the worker's full-resolution grade when all of
these hold: a GPU grader was set up at start, the look's grain is `two_scale_tile`
(or grain is off), the frame is no wider than the GPU's largest texture (4096; the
IMX477's 4056 fits, the IMX708's 4608 does not), and the EV is 0. Everything else
is graded on the CPU as before: the quick grade for the screen, `--grade inline`,
a shot taken while the service is stopping, `pifilm-process`, the trainer, and
looks with the Gaussian or `two_scale` grain. A double exposure is composited on
the CPU and the composite is then graded like any other frame.

**The pass.** The statistics (white balance, levels) are measured on the CPU by the
same code as before, so the audit fields of a record are the same whichever path
graded. The shader then does, per pixel and in float: the per-channel
normalisation tables, the luma tone curve, the 3D LUT with trilinear interpolation
written out in the shader (never the GPU's own filtering), and the grain, from the
same tile placed by the same seed as on the CPU. It rounds to bytes once, at the
end. The CPU path rounds after normalising, after the tone curve, after the LUT
and in the grain.

**The self-test.** The GPU code cannot run in the test suite, so the service checks
it at every start. The grader grades a 96 × 64 frame with a 64 px grain tile and
compares it with a NumPy version of the same pass (`pifilm/gpu/reference.py`).
More than 1 level of difference anywhere, or any graphics error, and GPU grading is
off for this run. The log has one line either way: `gpu grading on (...)` or
`gpu grading off: ...` with the reason. The service starts in both cases.

**If opening the GPU hangs.** A driver that blocks while the GPU is opened would
hold the only grade worker: no photograph would be graded, and the next start
would hang the same way. So the worker writes a marker file, `gpu-opening`, beside
the pending entries (`~/.local/state/pifilm/gpu-opening`) before it opens the GPU,
and removes it when the open returns, whether or not it worked. A start that finds
the marker does not open the GPU. It logs
`gpu grading off: the last start did not finish opening the GPU; delete <path> to try again`
and grades on the CPU until the file is deleted.

**Fallback.** A photograph the GPU fails on is graded on the CPU at once, from the
same seed, so it is never lost and never waits. The log names it
(`gpu grade failed (...)`). Three failures in a row switch the GPU off for the run.
Nothing shows on the screen.

**The record.** Every record of a graded photograph has `pipeline_backend`: `gpu-1`
or `cpu`. The number is the shader's version and rises with any change that can
move pixels. A GPU grade adds the stages `gpu`, `gpu_upload`, `gpu_shader` and
`gpu_readback` to `stage_ms` and has no `lut` or `grain` stage.

**Reproducibility.** `pifilm-process` regrades on the CPU. A record marked `cpu`
still reproduces pixel for pixel. A record marked `gpu-1` does not: the GPU's file is
the more exact of the two, because the CPU rounds colours to bytes before the LUT.
Against a CPU regrade it differs by about 0.2 to 0.6 levels on average and by a few
levels at most on ordinary colours. On strongly saturated areas, where the look's
LUT is steep, it differs by 10 to 20 levels. Measured on 2026-10-13: at most 5
levels on a real photograph with the trained IMX477 look on the Pi, and up to 20 on
synthetic saturated frames with the bundled starter look. `pifilm-process` says so
once when the folder's log has such records. The grain lands in the same place on
both paths.

**The switch.** `--grade-backend auto` (the default) uses the GPU when it can.
`--grade-backend cpu` never opens the GPU and is the earlier behaviour exactly.

**Checking it.** `pifilm-gpu-check --artifacts DIR [--image FILE] [--runs N]` grades
a full frame on the GPU, compares it with the reference and with the CPU, and
prints the timings. It exits 0 when the GPU is within 1 level of the reference.
The command appears only after `pip install -e .` has been run again in the Pi's
venv; `python -m pifilm.gpu.check` with the same arguments works without that.
The bundled starter look has Gaussian grain, so it is always graded on the CPU:
pass a look with the tile grain.

**Memory.** The textures for a 12 MP frame take about 110 MB of graphics memory,
held while the service runs.

### Double exposure

With the viewfinder's `2x` toggle on, two captures make one picture
(`pifilm/double.py`). Film adds exposures, H = H₁ + H₂, and is developed once, so the
two *ungraded* frames are added in linear light and the sum goes through the three
steps above once. Grading each frame and blending the results would apply the LUT's
tone curve twice, add values after the curve instead of light before it, lay grain
down twice, and normalise twice, which normalisation does not support. Each frame is
taken one stop down before the sum (`0.5·L₁ + 0.5·L₂`), the usual film starting
point: the camera auto-exposes every frame fully, so a straight sum would sit a stop
over and clip before the LUT's shoulder could act. Each frame keeps the EV it was
shot at, so a frame shot darker contributes less light. The composite is graded at
EV 0, and normalisation sets the pair's overall exposure.

## At training time, on the Mac

`pifilm-train` (`pifilm/train/`) turns two folders of images into that LUT:

1. **Corpus building** (`dataset.py`). Each corpus is split **by image**
   before any pixel is sampled, 20 % held out. Every image is oriented from
   EXIF, colour-managed to sRGB from its embedded ICC profile, cropped 6 % per
   edge, downscaled to 512 px, normalised, and sampled: 3000 pixels per image,
   capped at 400,000 per pool, kept only where Oklab lightness lies in
   (0.02, 0.98).
2. **Oklab** (`color.py`). All statistics are computed in Oklab, Björn
   Ottosson's perceptual space, so "match the distribution" means "match what
   the eye sees" and hue angles behave in the blues.
3. **Hue reweighting** (`transport.py`). The two corpora show different
   subjects. Target pixels are reweighted across 24 hue bins so the target's
   hue histogram matches the source's, which removes the largest content bias.
   Weights are clipped to 0.2 to 5, so this is a heuristic, and the residual
   is published in the report.
4. **Iterative Distribution Transfer** (Pitié, Kokaram and Dahyot, 2005).
   Forty rounds of: pick a random 3D rotation, project both pixel clouds onto
   its axes, match the source's marginal to the weighted target's along each
   axis by quantile mapping, rotate back. Each source pixel ends up with a
   partner colour in the target distribution. `--strength` blends between the
   original and the moved colour, and may extrapolate up to 2.
5. **LUT fit** (`lutfit.py`). A trilinear LUT is linear in its node values, so
   fitting is ordinary least squares with a sparse design matrix of eight
   non-zeros per pixel. Two regularisers are added: a second-difference
   smoothness term along all three grid axes, and an identity term that holds
   nodes no source pixel ever touched in place (71 % of the cube, in the first
   real fit). The normal equations are solved with SciPy's conjugate-gradient
   solver and a Jacobi preconditioner. A neutral-axis cap limits tint on greys,
   and monotonicity is then enforced exactly with SciPy's isotonic regression.
6. **Evaluation** (`evaluate.py`). The headline metric is a **sliced
   Wasserstein distance** from the graded held-out source pixels to the target
   cloud, using 256 fixed random projections and fixed samples so "before" and
   "after" differ only by the LUT. It is repeated over five seeds and the
   spread reported, so an improvement can be compared against noise. Five
   gates then pass or fail with thresholds fixed before any tuning:
   improvement above three times the seed spread, grey axis monotone, each
   channel monotone, neutral input staying neutral, and the cube interior off
   the gamut boundary.
7. **Report and publish** (`report.py`, `artifacts.py`). Contact sheets of
   held-out frames, tone ramps, normalisation diagnostics and `metrics.json`,
   then an atomic swap of the finished artifact into place so a crash can never
   leave a new LUT beside old parameters.

The full procedure, from an empty `data/` folder to a verified deployment, is
[the training guide](training.md).

## Libraries

| Library | Where | What it does here |
|---|---|---|
| NumPy | everywhere | All array maths: colour conversions, sampling, transport, evaluation |
| Pillow | runtime and trainer | Image I/O, EXIF orientation, ICC conversion (`ImageCms`), the `Color3DLUT` filter, `.cube` round-trips, contact sheets |
| OpenCV (`cv2`) | runtime and trainer | Colour-space conversions, the 256-entry `LUT` tables, Gaussian blur for grain, resizing, and V4L2 camera capture on the Pi |
| SciPy | trainer only | Sparse matrices, the conjugate-gradient solver, isotonic regression |
| requests, tqdm | `pifilm-fetch` only | Wikimedia Commons downloads with licence checks |
| smbus2, gpiozero | Pi only, `--ups x728` | Reading the UPS fuel gauge over I2C and the power-loss pin |
| `libEGL`, `libGLESv2` (through `ctypes`) | Pi only, `--grade-backend auto` | The GPU grade: a windowless context and one shader pass. System libraries, loaded when the grade worker starts, no Python package |
| Paramiko | Mac only, deploy script | SSH to the Pi with a reject-unknown-hosts policy |
| pytest, ruff, build | development | Tests, lint, wheel build |

Not used: PyTorch, TensorFlow or scikit-learn. The trainer uses no GPU; the Pi's
GPU runs only the grade described above. Nothing is downloaded at runtime and no
data leaves the machine.

## Dependencies by role

`pyproject.toml` keeps the base install small and puts the rest behind extras:

| Install | Pulls in | Who needs it |
|---|---|---|
| `pip install -e .` | NumPy, Pillow | The Pi, which gets OpenCV from apt |
| `.[opencv]` | + `opencv-python` | Any non-Pi machine that runs the pipeline |
| `.[train]` | + SciPy, requests, tqdm, `opencv-python` | The Mac, for `pifilm-train` and `pifilm-fetch` |
| `.[deploy]` | + Paramiko | The Mac, for `scripts/deploy_remote.py` |
| `.[dev]` | + pytest, ruff, build, `opencv-python` | Anyone running the tests |

On the Pi, `python3-opencv`, `python3-numpy` and `python3-pil` come from apt
and the venv is created with `--system-site-packages`, because the apt OpenCV
is built with GTK for the two-screen mode and the pip wheel is not; `smbus2`
and `gpiozero` are preinstalled on Raspberry Pi OS. OpenCV is imported through
one guard, `pifilm/_cv2.py`, which explains the right fix when it is missing.

## What a LUT cannot do

It maps each input colour to one output colour, everywhere in the frame. It
cannot add flash lighting, local contrast, sharpness or depth, and it cannot
invent colour: an unlit grey room comes out grey through every LUT, as measured
in `todo.md` section 1. Those limits, and the roadmap for better fits, are
covered in the README's "The look and its limits" section and in
[the training guide](training.md).
