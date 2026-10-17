# Agfachrome training, October 2026

A slide-film look after Agfachrome 50S, trained for the Raspberry Pi HQ Camera
(IMX477). The user chose candidate B from a gallery on 2026-10-18. The procedure
is `docs/training.md`; the IMX477 settings are those of
`docs/training-imx477-2026-09.md`.

## Chosen artifact

| | |
|---|---|
| Artifact | `artifacts/agfachrome-2026-10-b-dark-tile` (gitignored) |
| `lut_sha1` | `ee2a3b78…` (trained as `artifacts/agfachrome-2026-10-b`) |
| Normalisation and grain | copied from the deployed `imx477-2026-09-v4-dark-hp5-tile`: levels gamma 1.0–1.25, stretch cap 1.5, no white balance, lift highlight ref 0.02, HP5 tile grain 0.025 (params version 4) |
| `tuning_sha1` | `5404134a…` (libcamera `imx477.json`, the same as the 2026-10-15 captures) |
| LUT trained by | `pifilm-train --source data/source-imx477-agfa-2026-10 --target data/references-agfachrome-commons --out artifacts/agfachrome-2026-10-b --no-source-white-balance --source-lift-highlight-ref 0.02 --source-gamma-min 0.65 --source-gamma-max 1.25 --allow-small` |

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
previous look, slightly cooler.

## Rollback

Point `--artifacts` in the unit's `ExecStart` back at
`/home/george/repos/pi-film-reversal/artifacts/imx477-2026-09-v4-dark-hp5-tile` and restart.
