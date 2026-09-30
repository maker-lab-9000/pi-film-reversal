# IMX477 training, September 2026

The first look trained for the Raspberry Pi HQ Camera (IMX477). Candidate v3 was
deployed to the Pi 4 on 2026-09-30 at 16:40, then v4, then v4-dark (below), which
is the deployed look. The procedure is `docs/training.md`,
and `docs/retraining-imx477.md` has the IMX477-specific steps.

## Deployed artifact

| | |
|---|---|
| Artifact | `artifacts/imx477-2026-09-v4-dark` (gitignored; also on the Pi under `~/repos/pi-film-reversal/artifacts/`) |
| `lut_sha1` | `98e3671607763e659ae2e8755a6098e12422df8f` (v4's LUT) |
| Normalisation | levels gamma 1.0–1.25, stretch cap 1.5, no white balance, lift highlight ref 0.02 |
| LUT trained by | `pifilm-train --source data/source-imx477-v3-highlight --target data/references-merged-2026-09-29 --out artifacts/imx477-2026-09-v4-limits --no-source-white-balance --source-lift-highlight-ref 0.02 --source-gamma-min 0.65 --source-gamma-max 1.25 --allow-small` |
| Service flags | `--ae-constraint highlight --tuning-file imx477.json --artifacts …/imx477-2026-09-v4-dark` |

v3 (`a2c96826…`) was trained at `90e5f34` on `train/imx477-2026-09`, which lowered
`MIN_TARGET_IMAGES` to 150; later runs use `--allow-small` instead.

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

## v4-dark: keep dark scenes dark (deployed)

v4 still lifted night frames: with the 0.65 floor, 10 of 15 night frames from the
evening of 2026-09-30 sat on the floor, and the black-to-white stretch (default cap
4.0) was a second lift, brightening one frame (`213816`) by two stops by itself.
Removing both lifts at normalisation keeps a dark scene as the camera exposed it.

**v5, trained with the lifts removed, failed.** It was trained with
`--source-gamma-min 1.0 --source-gamma-max 1.25 --source-max-stretch 1.5` and exited with 3: held-out distance
0.0152 → 0.0244, training pool 0.0349 → 0.0068. The references are exposure-matched
and the sources no longer were, so the fit learned to brighten: its grey ramp lifts
the shadows and its hue sweeps are blotchy. The trainer compares at one exposure, so
it cannot learn a look for frames that are deliberately left at another.

**v4-dark is an untrained pairing.** It keeps v4's LUT unchanged (`lut_sha1`
`98e36716…`) and changes only three normalisation limits in `params.json`:

| | v4 | v4-dark |
|---|---|---|
| `levels_gamma_min` (brightening floor) | 0.65 | 1.0 (never brightens) |
| `levels_gamma_max` (darkening cap) | 1.25 | 1.25 |
| `levels_max_stretch` | 4.0 | 1.5 |

`params.json` records the derivation under `training.derived`; the metrics in the
file are v4's, not this pairing's. It was judged on pictures: all 316 IMX477
frames from 2026-09-24 to 09-30 rendered as original / v4 / v4-dark. 134 came out
darker than v4, none brighter. The typical brightness difference from the camera
original was 0.026, against v4's 0.057. Daylight shot with the highlight AE
constraint stays near the camera's exposure with more contrast; night scenes keep
their lamps and screens as the light sources. Two things to expect:

- The LUT was fitted to brighter input and slightly deepens the darkest tones, so
  a night frame can end up a little darker than the camera original.
- Normalisation no longer rescues underexposure; a backlit subject stays a
  silhouette unless it is shot with positive EV.

## Rollback

Point `--artifacts` in the unit's `ExecStart` back at
`/home/george/repos/pi-film-reversal/pifilm/data` and restart (`docs/training.md` §8).
