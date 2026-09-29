Retraining for the IMX477

Yes, you're right: it starts with photos from this sensor and lens. docs/training.md is the full procedure. For this camera it comes down to:

1. Shoot the source set on the Pi. Aim for 60–100 frames across at least 20 different scenes: tungsten indoors, window light, overcast and sunny outdoors, people and skin, food, signs, fabrics. Add 10–20 frames of deliberately colourful things (book spines, toys, a colour chart) so the rarer colours are covered.
- Keep the EV buttons at 0. The trainer doesn't know about EV compensation, so a −1 frame would be learned as a normal exposure.
- Leave auto exposure and auto white balance on, as the service uses them. Let each frame settle for a second.
- Focus properly with the focus bar, and skip frames with blown highlights or near-black ones.

2. Pull and sort on the Mac:
rsync -av --include='*/' --include='*_original.jpg' --exclude='*' \
  george@parr.local:Pictures/pifilm/ source_imx477/
- Copy the frames you want, flat, into data/source-imx477/, and drop near-duplicates.
- Set 3–5 whole scenes aside in data/test-scenes/ and don't look at them until the end.

3. References. The trainer wants at least 200, and none of your local sets has that many:

┌───────────────────────────────────┬────────┐
│                Set                │ Images │
├───────────────────────────────────┼────────┤
│ references-personal-01-curated-v1 │ 150    │
├───────────────────────────────────┼────────┤
│ references-personal-collection-01 │ 135    │
├───────────────────────────────────┼────────┤
│ references-expanded-2026-09-06    │ 56     │
├───────────────────────────────────┼────────┤
│ references-pilot-2026-09-06       │ 12     │
└───────────────────────────────────┴────────┘

You'll need to grow one to 200+, or merge sets if they share one era and lighting style (check for the same photo appearing in both).

4. Train:
.venv/bin/pifilm-train --source data/source-imx477 --target data/references-<set> \
  --out artifacts/imx477-2026-10-v1 \
  --no-source-white-balance --source-lift-highlight-ref 0.02
The two flags match how the Pi currently adjusts each frame: the camera does its own white balance. Only use a result with exit code 0; code 3 means a quality check failed. Start with report/summary.txt and contact_sheet.png.

5. Compare against the current look on the frames you set aside:
.venv/bin/pifilm-process data/test-scenes out/starter
.venv/bin/pifilm-process data/test-scenes out/imx477 --artifacts artifacts/imx477-2026-10-v1

6. Deploy by rsyncing the artifact to the Pi and pointing --artifacts in the service at it (docs/training.md §8). Also add --tuning-file imx477.json to the service command, so a libcamera update can't shift the camera's colour under the trained look.