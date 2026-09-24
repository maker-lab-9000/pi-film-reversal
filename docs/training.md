# Training a new look from scratch

Step-by-step: from an empty `data/` folder to a trained artifact running on the
Pi, with a record of what you did. Read section 1 once; it explains what each
later step is for. Commands run on the Mac unless marked otherwise.

Prerequisites: the Mac workstation from [the setup guide](setup.md), section 1,
installed with the `train` extra, and a working camera so you can shoot source
frames. A full run on a laptop takes minutes: the last passing run fitted
400,000 source and 360,000 reference pixels in 30 seconds, plus a few minutes of
image loading and report rendering.

Everything under `data/` and `artifacts/` is gitignored and stays on your
machine. Nothing here uploads or commits a photograph.

## 0. Should you train at all?

The camera works with the bundled starter preset. Train only if you want a look
derived from reference photographs, and keep the starter deployed until a
candidate passes both the numeric gates and a side-by-side visual check.

What the trainer learns: a 33³ colour lookup table that moves the colour
distribution of *your camera's frames* toward the colour distribution of a
folder of *reference photographs*. What it cannot learn: flash lighting,
composition, subjects, local contrast, or colour that is not in the frame. An
indoor scene with almost no colour looks the same through every LUT; see
`todo.md`, section 1, for the measurements behind that and for the roadmap
(real camera frames, a coherent reference set, a flash).

## 1. How a run works

Understanding this makes every flag and every gate in the report meaningful.

1. **Two corpora.** `--source` is a folder of ungraded camera frames.
   `--target` is a folder of reference photographs. Each is split **by image**
   into training and validation before any pixel is sampled, 20% held out by
   default. Splitting after sampling would leak: pixels from one photo are
   highly correlated, and the held-out score would flatter the fit.
2. **Per image.** EXIF orientation is applied and any embedded ICC profile is
   converted to sRGB. 6% is cropped from every edge (scanner beds, rebate,
   vignetting). The long side is downscaled to 512. Source frames get the same
   white balance and levels normalisation the Pi applies at capture; reference
   photographs get exposure matching only, no white balance and no levels
   stretch, so their colour balance is preserved. `--no-source-levels` and
   `--no-source-white-balance` turn off the respective source-side step
   (mirroring `NormalizeParams.levels` and `.white_balance`), and
   `--source-lift-highlight-ref FRACTION` sets the source's clipping-aware
   levels lift; all three are recorded in the artifact's `params.json` so the
   Pi applies exactly what was trained. Use `--no-source-white-balance` and
   `--source-lift-highlight-ref` together when the source camera has its own
   ISP-side AWB (an IMX708/Picamera2 corpus, for example) — the bundled
   starter's own values are `white_balance=False` and
   `levels_lift_highlight_ref=0.02`; see
   [the normalisation write-up](experiments/2026-09-19-imx708-normalisation.md)
   for why. 3000 pixels per image are
   sampled with Oklab lightness between 0.02 and 0.98, capped at 400,000 per
   pool. The white-balance and exposure gains, how often they hit their clamps,
   and which ICC profiles were seen travel with the pool into the report.
3. **Hue reweighting.** The two corpora show different things. Target pixels
   are reweighted over 24 hue bins so the target's hue histogram matches the
   source's, which removes the largest content bias (a wall does not turn green
   because the references have more foliage). Weights are clipped to 0.2 to 5,
   so this is a heuristic, not a guarantee.
4. **Transport.** Iterative distribution transfer, 40 rounds: every source
   pixel is moved to a partner colour in the target distribution. `--strength`
   blends between the original (0) and the fully moved colour (1) and may
   extrapolate up to 2.
5. **LUT fit.** A 33³ trilinear LUT is fitted to the (source, partner) pairs by
   regularised least squares: a smoothness term (default 0.01) stops nodes
   chasing noise, an identity term (default 1.0) keeps nodes no source pixel
   ever touches where they are, a neutral-axis cap (default 0.005 Oklab chroma)
   limits tint on greys, and a final projection enforces monotonicity.
6. **Evaluation.** On the held-out images, the distance from graded source
   pixels to the target cloud is measured before and after, with the same
   samples and projections both times so the two numbers are comparable. The
   measurement is repeated over five seeds and the spread reported, so an
   "improvement" can be compared against noise. Five gates then pass or fail.
7. **Report and publish.** Everything is written to a staging directory and
   swapped into `--out` in one move, so an interrupted run never leaves a new
   LUT beside old parameters.

Exit codes: `0` all gates passed; `1` the folders, arguments or corpus size were
unusable and nothing was written; `3` an artifact **was** written but at least
one gate failed. Never deploy a `3`.

## 2. Step 1: collect source frames from the camera

The source corpus must be this camera's own output. The only run so far that
passed every gate used 399 web photographs as a stand-in source, so it learned
"web photo to reference", not "Innomaker frame to reference". Its `params.json`
records `"proxy": true` for that reason.

**Plan the shoot.** Minimum 30 frames; aim for 60 to 100 across at least 20
distinct scenes: tungsten interior, window daylight, overcast and sunny
exterior, people and skin, food, signage, painted surfaces, fabrics. Vary
distance and exposure. Add 10 to 20 frames of deliberately colourful things (a
colour chart, toys, book spines): they are not photographs, they are coverage
for regions of the colour cube that ordinary scenes never visit. If the camera
has a flash, use it; the references were shot with one.

**Configure the camera exactly as it is used at capture.** The LUT is fitted to
the camera plus the pipeline's normalisation as one system, so training frames
must come through the same camera settings as real captures. The deployed
service sets no controls, so auto exposure and auto white balance are on at
capture time; leave them on for the shoot too, and give each frame a second to
settle before pressing.

Do **not** lock one white balance across different lighting. What absorbs
tungsten versus daylight is either the pipeline's per-frame white balance,
applied identically in training and at capture, or — when the artifact turns it
off, as the IMX708 starter does — the camera's own AWB. The pipeline's gains
are clamped to 0.6 to 1.6.
A tungsten room shot with a daylight value locked in leaves a cast that range
cannot remove; the trainer then learns to cool everything, and every daylight
capture comes out wrong. If auto white balance visibly hunts within one scene,
freeze it at the value it settled on for that scene only and re-enable it before
moving to different light:

```sh
v4l2-ctl -d /dev/video0 --list-ctrls                             # names differ per camera
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=0     # freeze at the settled value
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=1     # re-enable before the next scene
```

Variety of lighting is good; extremes are not. Blown highlights and near-black
frames cannot be normalised back. After training, `diagnostics.png` shows the
white-balance gains the corpus needed; centred near 1.0 with a low clamp rate
means the shoot was well behaved.

**Shoot.** Use the Stick, or on the Pi `pifilm-capture --no-preview`. Frames land
in `~/Pictures/pifilm/YYYY-MM-DD/` as `*_original.jpg` (the camera's own bytes)
or `*_ungraded.jpg` (re-encoded when raw MJPEG was unavailable). Either is
usable; prefer `_original` when both exist for a frame.

**Pull and select.** From the Mac:

```sh
rsync -av --include='*/' --include='*_original.jpg' --include='*_ungraded.jpg' --exclude='*' \
  george@parr.local:Pictures/pifilm/ data/source-raw/
mkdir -p data/source data/test-scenes
```

Copy the frames you want into `data/source/`, flat, one folder. Drop
near-duplicates: bursts of the same framing teach nothing and leak across the
split. Then set aside **three to five whole scenes** into `data/test-scenes/`
and do not touch them again until section 8. They are your untouched final
test; the trainer's own held-out split is for choosing settings, and anything
you look at while choosing is no longer a test.

Write down which files belong to which scene. You will need it for scene-grouped
validation in section 7.

## 3. Step 2: curate reference photographs

`data/references/`, flat, at least 200 images. The trainer refuses fewer without
`--allow-small`, and the two runs that used 12 and 56 references both failed
the held-out gate.

**Coherence.** One period and one lighting style. The look this project aims at
is the flash-lit saturated 1990s colour work; mixing in early monochrome, muted
scans or later digital work gives the transport contradictory targets.

**Quality bar.** Short side at least 800 pixels. No visible page texture,
borders, watermarks, screen photographs or photographed prints. The loader
converts embedded ICC profiles to sRGB and assumes sRGB when there is none; the
report lists the profiles it saw, and a corpus that is secretly half Adobe RGB
shows up there.

**Deduplicate by scene, not by file.** Different crops or re-encodes of the same
photograph must not land on both sides of the split. Balance series so none
exceeds about a third of the corpus.

**Rights and provenance.** Follow [the reference guide](reference-sources.md):
photographs credited to the photographer from primary sources, files you have
permission to use, fan-submission groups excluded. Keep the source URL, author,
licence and checksum for every file. If you write a `manifest.json` into the
folder with `licence_policy` and `licences` keys, the trainer copies them into
the artifact's `params.json`:

```json
{
  "licence_policy": "Personal research corpus; rights recorded per file; no redistribution",
  "licences": {"publisher-page": 120, "gallery-page": 80},
  "images": [{"file": "example-001.jpg", "sha256": "...", "source": "https://...", "author": "..."}]
}
```

`pifilm-fetch --category 'Category:...'` downloads a Wikimedia Commons category
with a per-file licence allowlist, validation and a manifest. It is a generic
corpus tool; it does not produce a reference set for this look.

## 4. Step 3: sanity checks

Count, and confirm the files load with the profiles you expect:

```sh
ls data/source | wc -l; ls data/references | wc -l
.venv/bin/python - <<'PY'
from pathlib import Path
from pifilm.imageio import list_images, load_rgb
for folder in ("data/source", "data/references"):
    paths = list_images(Path(folder))
    seen = {}
    for p in paths:
        rgb, meta = load_rgb(p)
        seen[meta.profile] = seen.get(meta.profile, 0) + 1
        if meta.profile_error: print("profile error:", p.name, meta.profile_error)
    print(folder, len(paths), "images; profiles:", seen)
PY
```

Optionally run one exploratory fit with `--allow-small` on a subset to see the
timings and the report layout. Treat its numbers as meaningless; `--allow-small`
also marks the held-out evaluation unavailable when the corpus is too small to
hold images back.

## 5. Step 4: train

Run from this repository's own virtual environment, so the git revision is
recorded in `params.json` (earlier runs from a sibling environment show
`"code_revision": "unknown"`). Use a fresh, dated `--out` for every run: if the
directory already exists, `pifilm-train` replaces it atomically and the previous
artifact and report are gone.

```sh
.venv/bin/pifilm-train --source data/source --target data/references --out artifacts/pifilm-2026-09-v1
```

Progress lines, in order: `source: N train, M validation images`,
`target: ...`, `reweighting target hues toward the source histogram`,
`iterative distribution transfer, 40 rounds`, `fitting 33^3 LUT by regularised
least squares`, `evaluating on held-out images`, `writing report`,
`published artifacts/... ; report in .../report`. Then the headline distance,
the seed spread, and one `[PASS]` or `[FAIL]` line per gate.

Output:

```text
artifacts/pifilm-2026-09-v1/
  pifilm.cube            the 33³ LUT, hashed into params.json
  params.json          normalisation, grain, provenance, corpus hashes, metrics
  report/
    summary.txt        headline numbers and the gates in plain language
    metrics.json       every metric, plus per-hue-bin colour changes
    contact_sheet.png  held-out source frames, normalised beside graded, with a reference strip
    ramps.png          grey ramp and hue sweeps, before over after
    diagnostics.png    white-balance and exposure gain histograms, clamp rates
```

## 6. Step 5: read the report

Start with `summary.txt`. From the one run that passed:

```text
held-out distance to Parr: 0.01799 before -> 0.00875 after (seed spread 0.00030)
training-pool distance:          0.01903 -> 0.00390
transport clipped out of gamut:  0.00168 dE
LUT fit residual:                0.01235 dE
```

- **Held-out distance.** The number that matters. It must fall, and by more
  than three times the seed spread.
- **Training-pool distance.** Always falls more than held-out. A large gap
  between the two means the fit memorised its training images.
- **Transport clipped out of gamut** says the transport asked for colours sRGB
  cannot hold. **LUT fit residual** says how far a smooth LUT is from what was
  asked. They are reported separately so a gamut problem is not mistaken for a
  fitting problem.

The five gates, with fixed thresholds that are never relaxed to make a run pass:

| Gate | Passes when | A failure usually means | First thing to try |
|---|---|---|---|
| `improvement_exceeds_noise` | held-out improvement > 3 × seed spread | Overfitting or a source/target mismatch; too few or incoherent references | More and more coherent references; more diverse camera frames; compare train vs held-out |
| `grey_axis_monotone` | neutral greys never darken as input brightens | Sparse cube, nodes drifting where no source pixel landed | Raise `--lambda-identity`; add colour-chart frames; more source coverage |
| `channel_monotone` | each output channel rises with its own input | Same as above, or an over-strong fit | Raise `--lambda-smooth`; lower `--strength` |
| `neutral_axis_chroma` | max Oklab chroma on the grey axis < 0.02 | Greys picking up a tint from the references' white balance | Lower `--neutral-axis-cap`; check the references are not uniformly warm or cool |
| `clipped_volume` | < 5% of the cube's interior sits on the gamut boundary | The grade pushes ordinary colours out of sRGB | Lower `--strength` |

Then the pictures. `contact_sheet.png` shows **held-out** frames: the question
is whether the graded row belongs in the same family as the reference strip
underneath. If the corpus was too small to hold images back, the labels say
TRAINING instead and the sheet is flattering. `ramps.png`: the grey ramp is the
learned tone curve and the hue sweeps show saturation and hue movement; a wobble
or banding means smoothness is too low. `diagnostics.png`: if normalisation is
clamping often, the LUT was fitted on input the Pi will rarely produce, and the
camera frames or their exposure need attention.

`metrics.json` also carries `hue_bins`: for each of 24 hues, the change in
lightness, the chroma ratio and the hue shift the LUT applies. It tells you in
numbers what the look does to reds, skin, greens and blues.

## 7. Step 6: iterate deliberately

Rules that keep the result honest:

- One change per run, a new `--out` per run, and keep the failures. Their
  reports are evidence.
- Choose settings on the trainer's held-out split. Never look at
  `data/test-scenes/` until you have chosen.
- Do not tune until a gate happens to pass. If a gate fails, understand why from
  the table above.

The knobs:

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 1.0 | 0 = identity, 1 = the learned look, up to 2 exaggerates it. The first real fit had the right shape at about a third of the expected amplitude, so above 1 is legitimate. |
| `--lambda-smooth` | 0.01 | Higher = smoother LUT, less banding, less detail in the grade. |
| `--lambda-identity` | 1.0 | Higher = untouched cube regions stay closer to identity. Protects greys and rare colours. |
| `--neutral-axis-cap` | 0.005 | Max tint allowed on neutral input; 0 forces greys fully neutral. Not a hard bound after the later projections; check `neutral_axis_max_chroma`. |
| `--highlights` | 0.0 | Protects coloured highlights by reducing the LUT's tone lift at the top of the range; 0 is off. Applied after the fit and before the final monotone projection. |
| `--hue-bins` | 24 | Resolution of the content-bias reweighting. |
| `--iterations` | 40 | Transport rounds. More is slower and rarely changes the result. |
| `--target-levels` | off | Stretch reference black and white points. For flat scans only. |
| `--target-median` | exposure target | Median the references are gamma-normalised to. Lower keeps more of the film's density. |
| `--val-fraction`, `--seed` | 0.2, 0 | The split. Change the seed to check a result is not a lucky split. |
| `--max-side`, `--pixels-per-image`, `--max-pixels` | 512, 3000, 400000 | Sampling. Defaults are adequate; raising them costs time, not quality. |
| `--grain-strength` | 0.004 | Grain recorded in the artifact; 0 disables. Independent of the fit. |
| `--allow-small`, `--proxy-source` | off | Exploration only. Both are recorded in `params.json` so nobody mistakes the result for a real fit. |

**Scene-grouped validation.** The production trainer splits by image, so two
frames of the same room can sit on both sides of the split and inflate the
held-out score. `pifilm.experiments.train` uses explicit, checksum-verified
partitions grouped by scene instead. Prepare an inputs directory:

```text
inputs/
  camera.json        {"records": [{"file": "camera/x.jpg", "sha256": "...", "role": "train", "group": "kitchen"}, ...]}
  references.json    same shape; roles train or validation; group = series or scene
  exclusions.json    {"regression_sha256": [ ...hashes that must never be trained on... ]}
  baseline/          an artifact directory (params.json + pifilm.cube) whose normalisation is reused
  camera/ references/   the image files named in the manifests
```

Roles are `train`, `validation`, `regression` or `excluded`; no group may appear
on both sides, no hash may appear twice, and every train file is checked against
the exclusions. Then:

```sh
.venv/bin/python -m pifilm.experiments.train --inputs inputs --out artifacts/candidate-a --neutral-cap 0.005
```

`docs/experiments/*.json` are real examples of the manifests, and
`scripts/prepare_refinement.py` is a dated recipe that builds such a directory
from a capture folder and a reference manifest. `pifilm.experiments.regression`
freezes a checksum-verified snapshot of ungraded/graded pairs, and
`pifilm.experiments.evaluate` renders paired comparison sheets and per-region
colour metrics for several candidates against that snapshot; the September 7
refinement report in `docs/experiments/` shows all of them in use.

**Visual comparison.** Grade the same frames with the starter and the candidate
and look at them side by side, ideally without knowing which is which:

```sh
.venv/bin/pifilm-process data/test-scenes out/starter
.venv/bin/pifilm-process data/test-scenes out/candidate --artifacts artifacts/pifilm-2026-09-v1
```

## 8. Step 7: deploy to the Pi and verify

Only for a candidate with exit code 0 that also won the visual comparison.

```sh
rsync -av artifacts/pifilm-2026-09-v1/ george@parr.local:repos/pi-film-reversal/artifacts/pifilm-2026-09-v1/
```

On the Pi, point the service at it and restart:

```sh
sudo sed -i 's#--artifacts [^ ]*#--artifacts /home/george/repos/pi-film-reversal/artifacts/pifilm-2026-09-v1#' /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload
sudo systemctl restart pifilm-capture.service
```

Take one capture and confirm the record names the new LUT:

```sh
tail -1 ~/Pictures/pifilm/$(date +%F)/captures.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["lut_sha1"])'
grep lut_sha1 ~/repos/pi-film-reversal/artifacts/pifilm-2026-09-v1/params.json
```

Update `PI_ARTIFACT_DIR` in `.env` on the Mac to match. Rollback is the same
edit pointing back at `pifilm/data`.

**Pin the tuning file for a trained look.** The Picamera2 backend defaults to
libcamera's automatic tuning for the detected sensor (`tuning_file:
auto:<model>` in `captures.jsonl`), which is what the deployed systemd unit
uses. Once you deploy a **trained** LUT, add `--tuning-file` to the unit's
`ExecStart` naming the exact tuning file the source corpus was captured
under (for example `imx708_wide.json`). Otherwise a future libcamera or OS
update could silently change the default tuning and shift colour science out
from under a LUT that was fitted against the old one.

## 9. Step 8: write it down

Add `docs/training-<date>-<name>.md`, in the style of the three existing
reports. Include: the corpus (counts, where the files came from, rights
status, corpus hashes from `params.json`), the exact command, dependency
versions and code revision (all in `params.json`), the split sizes, the
`summary.txt` block, each gate's result, what you saw on the contact sheet, and
the decision: deployed, rejected, or kept as evidence. A rejected run with a
clear reason is worth as much as a passing one.

## 10. Troubleshooting

- **`source corpus has N images, fewer than the 30 needed`** (or 200 for the
  target): add images, or use `--allow-small` for an exploratory run only.
- **`source and target are the same folder`**: the two paths resolve to one
  directory.
- **A previous run's report has vanished**: you reused its `--out`. `pifilm-train`
  replaces an existing artifact directory in one atomic move; only `pifilm-preset`
  and the experiments modules refuse to overwrite. Use a new name per run.
- **`WARNING: no held-out images for the ... corpus`** and `held_out_eval:
  false` in `params.json`: the corpus is smaller than `1 / val_fraction`
  images; every number is a training-pool number and measures memorisation.
- **High `clamp_rate` in the report**: normalisation hit its gain limits on many
  images. The frames are badly exposed or unusually lit; fix the shoot rather
  than the flags.
- **Several ICC profiles in the target's profile table**: expected for web
  material; the loader converts them. A profile listed with an error was
  treated as sRGB; check that file.
- **Exit code 3**: the artifact exists and can be inspected with
  `pifilm-process --artifacts`, but a gate failed. See the table in section 6.
- **A run that passes every gate but looks wrong**: gates are safety checks,
  not taste. The visual comparison in section 7 decides.

## Appendix: defaults at a glance

| Setting | Value |
|---|---|
| Minimum images | 30 source, 200 target |
| Validation split | 20% of images, by image, seed 0 |
| Crop, downscale, sample | 6% per edge, long side 512, 3000 px/image, 400,000 px cap, L in (0.02, 0.98) |
| Hue reweighting | 24 bins, weights clipped 0.2 to 5.0, chroma floor 0.03 |
| Transport | 40 IDT rounds, strength 1.0 |
| LUT | 33³, smoothness 0.01, identity 1.0, neutral cap 0.005, monotone projection |
| Gates | improvement > 3 × seed spread; grey axis and channels monotone; neutral chroma < 0.02; clipped volume < 5% |
| Grain | strength 0.004, blur 0.7 |
