# libcamera's adaptive contrast, on and off

Date: 2026-10-14. Step 5 of `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`,
the second look experiment. Script: `scripts/experiments/contrast_pairs.py`. Sheet:
`artifacts/contrast-pairs/contrast-pairs.jpg` (gitignored).

**Result: no change recommended.** The adaptive contrast enhancement (`rpi.contrast`,
`ce_enable` in the tuning file) changes the IMX477's pictures by 1 to 4 levels out
of 255, too little to matter beside what the look does, and turning it off would
feed the trained LUT pictures unlike the ones it was trained on.

## How the pairs were taken

On the Pi with the service stopped, four scenes, each photographed twice about
3 s apart: first as the service would (automatic exposure with the highlight
constraint, automatic white balance, `imx477.json`), then with the same tuning
except `ce_enable: 0`, the first shot's `ExposureTime`, `AnalogueGain` and
`ColourGains` fixed. Within each pair the exposure differed by under 1 % and the
colour gains were identical. Both shots were graded with the deployed look
(`imx477-2026-09-v4-dark`, tile grain).

## What it changed

Mean change in luminance, contrast on minus off, in levels of 255, by tone band
(share of the frame in brackets):

| Pair | Lux | Shadows 0-32 | 32-96 | 96-160 | 160-224 | 224-255 |
| --- | --- | --- | --- | --- | --- | --- |
| 162934, backlit room | 771 | 0.0 (93 %) | +1.1 | +2.0 | +2.0 | +1.8 |
| 162946, lit room | 142 | -0.1 (8 %) | +0.5 (34 %) | +1.5 (46 %) | +2.6 (11 %) | +1.3 |
| 163007, backlit room | 267 | 0.0 (94 %) | +1.0 | +2.3 | +2.3 | +0.6 |
| 163015, backlit room, a person | 369 | -1.3 (95 %) | +1.1 | +3.3 | +3.8 | +1.0 |

The enhancement lifts mid-tones and highlights by a few levels and leaves shadows
alone. In the last pair the person moved between the two shots, so its shadow
figure is not only the enhancement.

## Limits

Four pairs, three of them backlit rooms whose interior is near black. The
enhancement stretches a narrow histogram, so a flat scene (overcast, fog, haze)
is where it should do most; none was in this set. If such scenes ever look too
flat or too punchy, take pairs of them with the same script before changing
anything.

## Seen on the way

In the three backlit rooms the highlight-protecting exposure puts the interior
near black (median 0.001 to 0.002 linear), and the look darkens it further. That
is the camera's exposure choice, not the contrast enhancement, and the owner is
content with the look as it is.
