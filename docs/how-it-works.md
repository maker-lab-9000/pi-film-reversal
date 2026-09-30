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

Shutter priority (the viewfinder's shutter buttons) fixes the exposure time and
leaves gain to the Pi's auto-exposure, which meets its brightness target with gain
alone. EV compensation shifts that target, so it keeps working. The grade does not
change: a frame that ran out of gain (`ISO MAX`) is normalised like any dark frame.

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
| Paramiko | Mac only, deploy script | SSH to the Pi with a reject-unknown-hosts policy |
| pytest, ruff, build | development | Tests, lint, wheel build |

Not used: PyTorch, TensorFlow, scikit-learn, or any GPU. Nothing is downloaded
at runtime and no data leaves the machine.

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
