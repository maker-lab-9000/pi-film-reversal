# Exposure placed from the camera's Lux reading

Date: 2026-10-14. Step 5 of `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`.
Script: `scripts/experiments/exposure_from_lux.py`. Gallery: `artifacts/exposure-gallery/`
(gitignored), 144 IMX477 photographs from 2026-09-24 to 2026-10-12.

**Result: no change.** The owner judged the gallery and kept the deployed look
(`imx477-2026-09-v4-dark`, tile grain): outdoors it is the most balanced and the
closest to the target look, and indoors it gives the best balance of contrast,
detail and saturation.

## Why it was tried

The look moves every photograph's median towards 0.25 linear and may only darken
(levels gamma 1.0 to 1.25, stretch up to 1.5). Measured on 328 IMX477 photographs,
the result does not follow the scene's brightness:

| Scene (camera Lux) | Photos | Camera median | After the look |
| --- | --- | --- | --- |
| Night, under 10 | 47 | 0.031 | 0.034 |
| Dim, 10 to 50 | 104 | 0.115 | 0.142 |
| Indoor, 50 to 200 | 63 | 0.249 | 0.239 |
| Bright indoor, 200 to 1,000 | 47 | 0.149 | 0.140 |
| Daylight, 1,000 to 5,000 | 43 | 0.076 | 0.072 |
| Bright daylight, over 5,000 | 24 | 0.107 | 0.115 |

Daylight ends darker than a dim room: the highlight-protecting AE exposes it for
the sky, and the look may not lift it back.

## What was compared

Four columns per photograph, the same LUT and grain in all of them:

- **today**: the deployed look.
- **lux**: the levels gamma puts the median on a target that rises with Lux
  (3 lux 0.05, 30 lux 0.12, 300 lux 0.18, 3,000 lux and above 0.22, interpolated in
  log Lux), lifting allowed (gamma 0.6 to 1.25), the clipping-aware damping kept.
- **lux-gain**: the same target reached by a plain gain (0.5 to 3), capped so at
  most 0.5 % of the frame reaches the ceiling; a frame already at the ceiling is
  left at gain 1.
- **camera**: no stretch and no gamma.

Manual shots kept the camera's exposure in both lux columns.

Median of the graded photographs (automatic exposures only):

| Scene | today | lux | lux-gain | camera |
| --- | --- | --- | --- | --- |
| Night | 0.031 | 0.026 | 0.025 | 0.021 |
| Dim | 0.162 | 0.102 | 0.093 | 0.109 |
| Indoor | 0.249 | 0.249 | 0.149 | 0.270 |
| Bright indoor | 0.150 | 0.137 | 0.140 | 0.148 |
| Daylight | 0.077 | 0.157 | 0.080 | 0.071 |
| Bright daylight | 0.116 | 0.189 | 0.107 | 0.105 |

What it showed:

- Only a tone curve can brighten daylight here: in those frames the sky is
  already at the ceiling, so the protected gain stays near 1. The curve raises the
  shadows most, the milky lift rejected on 2026-09-30.
- Placing indoor scenes lower made faces too dark (the gain column on the
  2026-09-30 portrait), the problem the v4 limits were made to fix.
- The measured "daylight is darker than a dim room" is a statistic of the frames,
  not something the owner sees as wrong in the pictures.

## If it is revisited

The script regenerates the gallery from any Pi capture folders with
`captures.jsonl`. The second step of the roadmap's look experiments (libcamera's
adaptive contrast off) needs new photo pairs from the Pi.
