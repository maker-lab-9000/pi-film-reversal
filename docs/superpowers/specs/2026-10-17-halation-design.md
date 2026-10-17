# Halation: design

Date: 2026-10-17. Branch `feat/halation`, pull request into `main`. Chosen by the owner
from two galleries on 2026-10-17 (`scripts/experiments/halation_gallery.py`): model C,
the global mix with 2x headroom, strength 0.15, 0.25 mm. Approved in conversation the
same day: the CPU makes a small blurred copy and the GPU mixes it in (option 1), and
the look that gets it is `imx477-2026-09-v4-dark`.

## Goal

On colour film, light passes through the emulsion, reflects off the film base and
comes back through the red-sensitive layer (the deepest), so bright highlights grow a
soft red-orange glow. The grade gains that as a step of the look, set in
`params.json`, off unless a look turns it on, and graded on the Pi's GPU like the
rest of the photograph.

## What the galleries showed (2026-10-17)

- A threshold model (only highlights above a level, blurred, tinted, added) barely
  shows round a lamp and turns a sunlit wall cream.
- A global mix (a blurred copy of the whole frame mixed into red and a little green,
  no threshold) leaves flat areas exactly as they were and shifts no colour, but on
  the camera's JPEG a clipped lamp is no brighter than white paper, so the halo is
  weak, and inside the clipped area the red is pulled down and the lamp turns
  blue-white.
- The mix with the clipped highlights expanded before it and compressed back after
  (x / (1 - c x), then y / (1 + c y)) gives the warm glow and keeps the lamp white.
  At 10x (c = 0.9) a sunlit facade floods its dark windows with orange; at 2x
  (c = 0.5) it does not. The owner chose 2x.

The write-up goes in `docs/experiments/2026-10-17-halation.md` with the script.

## The model

Halation runs after normalisation and before the LUT: it happens when the film is
exposed, and the LUT is the look's roll-off. The order of the grade becomes
**normalise → halation → LUT → grain**. Grain stays last.

All in linear light, per channel, with `lin` the sRGB-decoded normalised frame:

1. **Expand:** `x = lin / (1 - c * lin)`, with `c = headroom`. White becomes
   `1 / (1 - c)`, 2 at c = 0.5. `c = 0` is no expansion.
2. **The glow, `B`:** `x`, reduced by a whole factor `k` (an area mean in linear
   light, so a small lamp keeps its energy), blurred with the fall-off, and read back
   at full size with bilinear interpolation (half-pixel centres, edges clamped).
   - The fall-off is three Gaussians: `sigma`, `2 sigma`, `4 sigma`, weighted 0.5,
     0.3 and 0.2 (about an exponential), each with reflected borders.
   - `sigma` in pixels is `sigma_mm * max(H, W) / frame_width_mm`, the grain's scale,
     so the glow is the same share of the frame at any size (quick grade, batch,
     full photograph).
   - `k = max(1, floor(sigma_px / 4))`: at least four reduced pixels of sigma. At
     12 MP and 0.25 mm that is `k = 7`.
3. **Mix:** `x' = x + strength * weights * (B - x)`, the weights per channel. A flat
   area equals its own blur, so it is unchanged.
4. **Compress:** `lin' = x' / (1 + c * x')`, the exact inverse of step 1, then back
   to sRGB.

The fall-off, the reduction rule and the interpolation are fixed in code under the
model name `"mix"`; changing any of them later is a new model name, as the grain
models are.

## `params.json`

A new optional top-level section:

```json
"halation": {
  "enabled": true,
  "model": "mix",
  "strength": 0.15,
  "sigma_mm": 0.25,
  "headroom": 0.5,
  "weights": [1.0, 0.35, 0.0],
  "frame_width_mm": 36.0
}
```

- `HalationParams` (new `pifilm/halation.py`), validated at load like the grain:
  `strength` 0 to 1, `sigma_mm` above 0 and at most 5, `headroom` 0 to 0.9,
  each weight 0 to 1, `frame_width_mm` above 0, all finite, `model` exactly `"mix"`.
  A bad value fails the load with the file named.
- No section, or `enabled: false`, or `strength: 0`: no halation, and the pixels,
  `normalize_sha1` and every record are what they are today.
- **Version 5.** `PARAMS_VERSION` becomes 5. A look with halation on needs version 5,
  so an older build refuses it instead of grading without the glow. A look without
  halation keeps its version. The loader reads 1 to 5; halation on with a version
  below 5 is refused, as the grain models are.
- `write_artifact` takes `halation=` and writes the larger of the grain's version and
  5 when halation is on.

## Turning it on for a look

`stamp_halation(dir, params)` in `pifilm/artifacts.py`, the same pattern as
`stamp_tuning`: the artifact is loaded before and after, every other key and the file
mode are kept, the version is raised to 5 when halation is on (never lowered), and
the new `params.json` is written beside the old, fsynced and moved over it.

`scripts/stamp_halation.py --artifacts DIR [--strength S] [--sigma-mm M]
[--headroom C] [--off]`, defaults the chosen values; prints the section it wrote.
For this change it is run on `artifacts/imx477-2026-09-v4-dark` on the Mac and on
the Pi (artifacts are not in git).

## The two paths

One function makes the glow for both: `halation.glow(normalised_u8, params)`
returns the reduced, blurred `B` as float32 `(h, w, 3)`, from the frame the LUT is
about to be given (the output of `normalize_u8`, after any EV compensation). The
reduction and blur run on the CPU with OpenCV (`cv2.LUT` with a float table from
code to expanded linear, `cv2.resize` with `INTER_AREA`, `cv2.GaussianBlur`).

**CPU (`Pipeline.process`).** After `normalize_u8`: the glow, then steps 1, 3 and 4
on the full frame (channels whose weight is 0 are skipped), worked in bands of rows
so a 12 MP frame does not need a full-size float copy, rounded to bytes, then the LUT
and grain as today. Stage `halation` in `stage_ms`.

**GPU.** When halation is on, the grader is given the glow with the tables: the CPU
runs `normalize_u8` on the frame (it has only the statistics today), makes the glow,
and uploads it as a float texture. Without halation the CPU does no more than today. The shader, after the tone step: decode to linear, expand, read `B`
by hand with `texelFetch` and bilinear weights (the shader reads no texture through
the GPU's filtering), mix, compress, encode to sRGB, then the LUT and grain as
today. `SHADER_VERSION` becomes 2 (records say `gpu-2`) for every photograph, with
halation or without, since the pass changed. Stage `halation` covers the CPU's part.

- `reference.py` gains the same steps, in float32 with the shader's names; the
  self-test runs them when the look has halation, at the same one-level tolerance.
- The glow texture's size per photograph is under the device's largest texture by
  construction (`k >= 1`); a 1 x 1 placeholder is bound when halation is off.
- `can_grade` is unchanged: EV other than 0 still goes to the CPU.

**Which frames get it.** The full grade (inline and background), the quick grade
(review screen and Stick), `pifilm-process`, and `pifilm-gpu-check` all do.
`Pipeline.process(..., halation=True)` is the new default; the live view's
`preview_frame` passes `halation=False`, as it passes `grain=False`.

## Records

When halation is on, each graded record gains `halation_sha1` (SHA-1 of the
section's `to_dict()`, sorted keys) right after `normalize_sha1`. Without it the
record is unchanged. `pifilm-process` regrades with the artifact's halation, so a
record graded with halation is reproduced by the same artifact; it says nothing when
`halation_sha1` differs from the artifact's (the same as `normalize_sha1` today).

## Costs to check on the hardware

Estimated, to be measured on the Pi 4 at 12 MP:

- GPU path: the extra `normalize_u8` (about 120 ms), the float table and area
  reduction (about 100 ms), the blur on the small copy (tens of ms), the upload of a
  579 x 434 float texture, and the extra shader work. Aim: under 400 ms added to a
  shot's grade (today 650 ms). It is the background grade, so the shutter is not
  slower.
- CPU path (the fallback, EV other than 0, `pifilm-process`): seconds, not measured
  yet. Recorded, not a gate.

## The look

`imx477-2026-09-v4-dark` has the deployed `-hp5-tile` look's LUT (`98e3671`) and
normalisation; only its grain, training notes and tuning stamp differed. Decided at
spec review (2026-10-17) and done the same day, on the Mac and the Pi, before any code:

- **Grain off** (it was the Gaussian model at 0.004, barely visible, and the GPU pass
  makes only the tile grain or none), so the GPU grades it. With grain off it is the
  latest look without grain, and the owner may run it before halation exists.
- **Stamped** with the Pi's tuning, `5404134af8256b674e5a8f93fe1bb616a1e0407e`
  (`pifilm-tuning --stamp`), so the tuning check covers it.
- **Changed in place** by `stamp_halation`, which raises it to version 5: builds from
  before this branch refuse it. Nothing used it on 2026-10-17.

## Code

- New `pifilm/halation.py`: `HalationParams`, `glow`, the CPU steps 1, 3, 4, and
  the fixed fall-off.
- `pifilm/artifacts.py`: the section, version 5, `write_artifact(halation=)`,
  `stamp_halation`.
- `pifilm/pipeline.py`: the step, `halation=` on `process`, `halation_sha1`.
- `pifilm/gpu/shader.py`, `reference.py`, `grader.py`, `egl.py`: the texture, the
  uniforms, the steps, version 2.
- `pifilm/capture/app.py`: the live view passes `halation=False`.
- `scripts/stamp_halation.py`; `scripts/experiments/halation_gallery.py` and its
  write-up.
- Docs: `docs/how-it-works.md` (the step and its order), `CLAUDE.md`, the record
  fields where they are listed.

## Tests

- The maths: a flat frame is unchanged; expand and compress round-trip; halation off
  is byte-identical to today; the glow is the same share of the frame at two sizes;
  a small bright spot in a dark frame grows a red halo larger than its green one,
  and none in blue.
- The CPU path against the reference (with and without halation), at the existing
  tolerances.
- Loading: the section's validation, version 5 needed, older versions load as
  before, a look without the section unchanged; `stamp_halation` keeps other keys,
  raises the version, is atomic.
- Records: `halation_sha1` present only when on, in its place.
- The grader with the fake device: the glow passed, the self-test with halation.

## Acceptance (hardware)

1. The shader compiles and `pifilm-gpu-check --artifacts .../imx477-2026-09-v4-dark`
   passes, with the halation stage
   timed.
2. A photograph of a lamp in a dark room and one of a sunlit wall look like the
   gallery's 2x column on the review screen and in the graded file.
3. `stage_ms` shows `halation`; the record has `halation_sha1`.
4. A look without halation grades as before (`gpu-2`, pixels within the self-test
   tolerance of a `gpu-1` grade).

## Not in this step

- The blur on the GPU (several render passes).
- Halation in the live view.
- Training or fitting halation; it is set by hand, like the grain.
- A settings-menu row for it.
