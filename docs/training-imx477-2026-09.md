# IMX477 training, September 2026

The first look trained for the Raspberry Pi HQ Camera (IMX477). Candidate v3 was
deployed to the Pi 4 on 2026-09-30 at 16:40. The procedure is `docs/training.md`,
and `docs/retraining-imx477.md` has the IMX477-specific steps.

## Deployed artifact

| | |
|---|---|
| Artifact | `artifacts/imx477-2026-09-v3-highlight` (gitignored; also on the Pi under `~/repos/pi-film-reversal/artifacts/`) |
| `lut_sha1` | `a2c9682660875b9b811455930a0f6b2e9b6ba62d` |
| Command | `pifilm-train --source data/source-imx477-v3-highlight --target data/references-merged-2026-09-29 --out artifacts/imx477-2026-09-v3-highlight --no-source-white-balance --source-lift-highlight-ref 0.02` |
| Code | `90e5f34` on branch `train/imx477-2026-09`, which lowers `MIN_TARGET_IMAGES` from 200 to 150 (see References) |
| Service flags | `--ae-constraint highlight --tuning-file imx477.json --artifacts …/imx477-2026-09-v3-highlight` |

## Sources

141 `_original.jpg` frames from 2026-09-30, all shot at EV 0 with
`--ae-constraint highlight` and libcamera's IMX477 tuning. They cover daylight
streets, cars and buildings (13:03–15:55, up to 26 000 lux) and indoor scenes
(15:55–16:16, mostly under 100 lux, 2400–5500 K). Source corpus SHA-1:
`705ab38410c60c5f4a637a40c5a3a404e54919a2`.

The highlight constraint matters. Under libcamera's default (`normal`)
constraint, 35 of 98 EV-0 frames from 2026-09-24 to 09-29 clipped more than 1%
of pixels at the sensor, up to 12%, measured on the DNGs. With `highlight`, none
of the 2026-09-30 frames clipped more than 1%. Mid-tones were placed normally in
both sets; only highlights were lost. Frames shot with EV compensation were
excluded because the trainer would learn a −0.67 EV frame as a normal exposure.

## References

`data/references-merged-2026-09-29`: every local Parr set merged, with near-duplicates
removed by average hash. `references-personal-01-curated-v1` got first pick; together
the four sets hold 353 files but only 168 unique images. That is below the trainer's
200-image minimum, so the training branch lowered `MIN_TARGET_IMAGES` to 150. This
branch keeps 200. A larger reference set is the next improvement.

## Candidates

All runs used the same references and flags. "Distance" is the held-out sliced
Wasserstein distance to the references in Oklab (lower is closer).

| Run | Source frames | Distance before → after | Exit |
|---|---|---|---|
| v1 | 85, `normal` constraint | 0.0218 → 0.0283 (worse) | 3 |
| v2 mixed | 113, both constraints | 0.0182 → 0.0171 (−6%) | 0, margin 0.00003 |
| v2 highlight | 57, `highlight` | 0.0277 → 0.0250 (−10%) | 3, 11 validation frames too noisy |
| **v3 highlight** | **141, `highlight`** | **0.0225 → 0.0151 (−33%)** | **0**, margin 7× the noise threshold |
| v3 mixed | 226, both constraints | 0.0272 → 0.0269 (−1%) | 3 |

v3 highlight passes every safety gate: the grey axis is monotone, channels are
monotone, neutral-axis chroma is 0.0054 (cap 0.02), and no interior volume is
clipped. Both mixed runs did badly, so frames shot under different AE constraints
should not share a source set.

## Look

A subtle grade: slightly more contrast, deeper shadows, and richer reds and
oranges (brickwork, cars), with a peachy warmth indoors under tungsten rather
than a neutral correction. It is gentler than the handcrafted starter, which
reads as punchier mainly through saturation.

## v4: tone limits

v3's normalisation used the default levels gamma clamp (0.5 to 2.0). Two frames
from 2026-09-30 showed both ends failing. `160707`, a deliberately dark street,
was lifted to slate grey at the 0.5 floor. `155821`, a face against a white wall,
was darkened at the 2.0 ceiling until the face fell from 0.25 to 0.10 brightness.
v4 retrains on the same 141 frames and references with
`--source-gamma-min 0.65 --source-gamma-max 1.25 --allow-small` (the new flags are
recorded in `params.json` as `levels_gamma_min`/`levels_gamma_max`).

| | v3 | v4 |
|---|---|---|
| `lut_sha1` | `a2c96826…` | `98e36716…` |
| Distance before → after | 0.0225 → 0.0151 | 0.0285 → 0.0187 (−34%) |
| Gates | all pass | all pass |

v4's distances are higher in both columns because the references are compared
after exposure matching: a frame the limits keep dark, or leave brighter, is
further from the references' median by construction. The metric cannot say
whether a dark scene should stay dark, so the choice between v3 and v4 is
made on the pictures. Frames the median rule already handles (gamma between
0.65 and 1.25) get the same tone curve under both.

## Rollback

Point `--artifacts` in the unit's `ExecStart` back at
`/home/george/repos/pi-film-reversal/pifilm/data` and restart (`docs/training.md` §8).
