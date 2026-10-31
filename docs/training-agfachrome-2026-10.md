# Agfachrome training, October 2026

A slide-film look after Agfachrome 50S, trained for the Raspberry Pi HQ Camera
(IMX477). The user chose candidate B from a gallery on 2026-10-18, then, finding
it a little flat, candidate D on 2026-10-23 (see "Candidates C and D"); D is
deployed. The procedure
is `docs/training.md`; the IMX477 settings are those of
`docs/training-imx477-2026-09.md`.

## Chosen artifact

| | |
|---|---|
| Artifact | `artifacts/agfachrome-2026-10-d-dark` (gitignored), deployed 2026-10-23 |
| `lut_sha1` | `ae4fd640…` (trained as `artifacts/agfachrome-2026-10-d`) |
| Normalisation | from the deployed v4-dark look: levels gamma 1.0–1.25, stretch cap 1.5, no white balance, lift highlight ref 0.02 |
| Grain | off, as in `imx477-2026-09-v4-dark` (the user's choice, 2026-10-18), so the GPU grades it |
| References | the 124 left after the review of 2026-10-19 (below), each stretched to full black and white (`--target-levels`) |
| Halation | on, `scripts/stamp_halation.py` defaults (strength 0.15, sigma 0.25 mm, headroom 0.5); params version 5 |
| `tuning_sha1` | `5404134a…` (libcamera `imx477.json`, the same as the 2026-10-15 captures) |
| LUT trained by | `pifilm-train --source data/source-imx477-agfa-2026-10 --target data/references-agfachrome-commons --out artifacts/agfachrome-2026-10-d --target-levels --no-source-white-balance --source-lift-highlight-ref 0.02 --source-gamma-min 0.65 --source-gamma-max 1.25 --allow-small` |

As with v4-dark, the LUT was fitted with the v4 normalisation limits and is
deployed with the darker ones; `params.json` records this under
`training.derived`, and the metrics in it are the trained run's.

## References

No open archive holds enough Agfachrome 50S: Wikimedia Commons had about six
frames and openly licensed Flickr photographs about five (2026-10-17). The
reference set is therefore the Agfachrome family:

- A Commons file search for "Agfachrome": 402 candidates, 370 passed the licence
  (public domain, CC BY, CC BY-SA; no NC or ND) and size checks, 355 downloaded
  as 1280 px thumbnails. The rest were too small, greyscale, not a photograph
  format, or packaging. Mixed stocks: 50S, 50, 50RS, CT18, CT100 and many unnamed.
- Removed by review: 18 photographs of film boxes and cartridges, 56 dated 1960 or
  earlier (Agfacolor, which the search also matched), and 23 more the user removed.
  They are in `excluded/`, and `manifest.json` records why.
- Added: the user's own 29 Agfa 50S scans, weighted 3× through `weights.json`, so
  they give about a quarter of the reference pixels.

287 references in all: 258 from Commons and 29 of the user's. `manifest.json` and
`ATTRIBUTION.md` in the folder credit every Commons file. The download script,
`.fetch.py`, honours `Retry-After` on HTTP 429; Commons asked for ten-minute
pauses three times.

## Sources

229 IMX477 originals, all at EV 0 with libcamera's `imx477.json` tuning:

- the 141 frames from 2026-09-30 that trained v4 (highlight AE constraint);
- 88 frames from 2026-10-15, mostly manual exposure (ISO 800, 5000 K), 27 to 600 lux.

## Candidates

| Run | Sources | Distance before → after | Exit |
|---|---|---|---|
| A | 141 (2026-09-30) | 0.0251 → 0.0163 (−35%) | 0 |
| **B** | **229 (+ 2026-10-15)** | **0.0265 → 0.0119 (−55%)** | **0** |

Both pass every safety gate. The gallery (`artifacts/agfachrome-gallery`, 277
frames: camera, deployed look, A, B, each with its logged EV and grain seed) put
the frames neither run trained on first.

## Look

Softer contrast than the IMX477 look, lighter shadows, a warmer and more pastel
palette, and yellower greens: the look of a scanned slide. A stays close to the
previous look, slightly cooler. D keeps B's warm palette and yellower greens with deeper
blacks and brighter whites, so it reads less flat.

## Candidates C and D

The user found B a little flat. Two changes, one at a time:

- **C, faded references removed (2026-10-19).** Every Commons reference was scored
  by the 90th percentile of its Oklab chroma. 170 were flagged: below 0.057 (what
  95% of the user's own 50S scans exceed), or a year of 1960 or earlier in the
  title or description (the earlier pre-1960 review had read the manifest's
  `date`, often the upload date). The user kept 11 after looking at them; the other
  159 moved to `excluded/` with their reason in `manifest.json`. 124 references
  remain: 95 from Commons and the user's 29 scans, still weighted 3×. C:
  0.0213 → 0.0149, all gates pass. It looked much like B: the flatness was not
  weak colour.
- **D, reference blacks stretched (2026-10-23).** The references' blacks are
  lifted, as scans of slides often are (Oklab L at the 2nd percentile: 0.214 for
  the user's scans, 0.171 for Commons). D is C trained with `--target-levels`,
  which stretches each reference to full black and white before fitting. D:
  0.0297 → 0.0197, all gates pass.

On the 64 frames of 2026-10-19 (none trained on), median Oklab L at the 2nd
percentile (blacks) and 90th-percentile chroma:

| Look | Blacks | Whites (98th) | Chroma |
|---|---|---|---|
| v4-dark | 0.092 | 0.923 | 0.088 |
| B | 0.150 | 0.932 | 0.086 |
| C | 0.147 | 0.939 | 0.088 |
| **D** | **0.095** | **0.960** | **0.088** |

D keeps the Agfachrome colour with blacks as deep as v4-dark's. Each distance is
measured against its own reference set, so the numbers do not compare across
candidates.

## Rollback

Point `--artifacts` in the unit's `ExecStart` back at
the previous look and restart.
