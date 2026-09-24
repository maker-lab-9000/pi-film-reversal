# Kodachrome look on the IMX477 (experiment, 2026-09-24)

Branch `exp/kodachrome-look`. The trained Kodachrome LUT from the sibling
`kodachrome-film` repository, run on the IMX477 without retraining, to see how
far a look trained for another camera carries before an IMX477 corpus exists.

## What was copied

On this branch the Kodachrome look **replaces the bundled starter** in
`pifilm/data/` (`params.json`, `pifilm.cube`, `training-summary.txt`), and the
starter moves to `pifilm/data/looks/starter/`. Everything that loads the default
look - the deployed service, `pifilm-process` without `--artifacts` - gets
Kodachrome. `main` is unchanged.

The files are `params.json` and the cube from
`kodachrome-film/artifacts/` (training code revision `e4729e8`, LUT
`b8ccf30cc719241c98ff3c4a9f73a1cdf8cb000b`), unchanged except that the cube is
renamed `pifilm.cube` (what `scripts/deploy_remote.py` checks for) and
`training.origin` records where it came from. `training-summary.txt` is that
run's own report: all five gates passed, held-out distance 0.02009 -> 0.01536.

Caveats carried with it:

- It was trained with `--proxy-source` on web photographs, not on frames from
  any camera: it learned "web photo -> Kodachrome K-14 scans".
- Its normalisation is the one it was trained with: `white_balance: true` and no
  clipping-aware lift (`levels_lift_highlight_ref` absent, so `None`). The IMX477
  already white-balances in its ISP, so this look balances a second time; the
  starter does not.
- Tuning file: unchanged, libcamera's automatic `imx477` tuning, as the deployed
  unit already uses. Nothing is pinned, on purpose, for an experiment.

## First look, on 68 IMX477 captures from 2026-09-24

Graded on the Mac with the capture pipeline (grain off, each frame's `ev_comp`
honoured), beside the camera original and the deployed starter:

- **Tungsten warmth is removed.** The per-frame white balance neutralises the
  orange of evening interiors; walls go grey-white to slightly green where the
  starter keeps them warm.
- **Flatter and paler.** Shadows are lifted (a black curtain goes charcoal),
  saturation is lower than the starter on reds and yellows; the red floor and
  yellow chair lose most of their punch.
- **Daylight frames suffer least**, but still read softer than the starter.

That is roughly what the caveats predict: a proxy-trained, K-14-scan look with
its own white balance, on a camera whose ISP has already balanced. It is a
reference point for the IMX477 retrain (`docs/training.md`), not a candidate to
keep.

## Running it on the Pi

The deployed unit already points `--artifacts` at `pifilm/data`, so checking out
the branch and restarting is all it takes:

```sh
cd ~/repos/pi-film-reversal && git fetch && git checkout exp/kodachrome-look
sudo systemctl restart pifilm-capture.service
```

`lut_sha1` in `captures.jsonl` then reads `b8ccf30c...`. Roll back with
`git checkout main` (or `feat/lcd-focus-bar`) and the same restart, or point
`--artifacts` at `pifilm/data/looks/starter` to keep this branch's code.
