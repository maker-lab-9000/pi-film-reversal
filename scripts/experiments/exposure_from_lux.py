"""Experiment: place each photograph's brightness from the camera's Lux reading.

Step 5 of docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md.
Offline only; nothing here changes a default. The owner judges the gallery.

Today the levels gamma of the deployed look (``imx477-2026-09-v4-dark``) may
only darken (gamma 1.0 to 1.25) towards one median, 0.25 linear, whatever the
scene. Measured on 328 IMX477 photographs (2026-10-14), the result does not
follow the scene: night 0.03, a dim room 0.14, a lit room 0.24, daylight
0.07 to 0.12 (the highlight-protecting AE exposes daylight for its sky and the
look may not lift it back).

The variants keep the look's LUT, grain and stretch and change only where the
median goes:

* ``today``: the deployed look, unchanged.
* ``lux``: the median is placed on a target that rises with the scene's Lux
  (``TARGET``), and the gamma may lift as well as darken (``GAMMA``). The
  clipping-aware damping of the lift stays, so a frame with a blown sky is not
  lifted further.
* ``lux-gain``: the same target reached by a plain gain in linear light, as a
  longer or shorter exposure would (a gamma lift raises the shadows most, which
  read as milky on the 2026-09-30 night shots). No stretch. The gain is held to
  0.5 to 3 and capped so that at most 0.5 % of the frame reaches the ceiling.
* ``camera``: no stretch and no gamma at all: what the camera's own exposure
  gives, the reference for both.

A photograph with a manual shutter or ISO keeps the camera's exposure in the
``lux`` variants (no stretch, no gamma): the photographer chose it.

Usage (from the repo root):

    .venv/bin/python scripts/experiments/exposure_from_lux.py \\
        --look artifacts/imx477-2026-09-v4-dark-hp5-tile --out artifacts/exposure-gallery \\
        FOLDER [FOLDER ...]

Each FOLDER is a Pi capture folder with ``captures.jsonl``. Writes
``index.html`` and the JPEGs into ``--out``, and ``summary.json`` with the
measured medians.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
from pathlib import Path

import numpy as np

from pifilm.artifacts import Artifacts
from pifilm.color import LUMA_709, srgb_to_linear
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

# Target linear median by scene Lux, interpolated in log10(Lux) and held flat
# beyond the ends. Points: 3, 30, 300 and 3,000 lux and above.
LOG_LUX = [0.5, 1.5, 2.5, 3.5]
TARGET = [0.05, 0.12, 0.18, 0.22]
GAIN = (0.5, 3.0)
# The gain may take at most this share of pixels to the ceiling.
GAIN_CLIP_PCT = 99.5
# The lux variants may lift (gamma below 1) as well as darken.
GAMMA = (0.6, 1.25)
BINS = [
    (0, 10, "Night, under 10 lux"),
    (10, 50, "Dim, 10 to 50 lux"),
    (50, 200, "Indoor, 50 to 200 lux"),
    (200, 1000, "Bright indoor or overcast, 200 to 1,000 lux"),
    (1000, 5000, "Daylight, 1,000 to 5,000 lux"),
    (5000, math.inf, "Bright daylight, over 5,000 lux"),
]
PER_BIN = 24
VARIANTS = ["today", "lux", "lux-gain", "camera"]


def target_for(lux: float, points: list[float]) -> float:
    return float(np.interp(math.log10(max(lux, 1e-3)), LOG_LUX, points))


def exposure_gain(rgb_u8: np.ndarray, base: NormalizeParams, target: float) -> float:
    """The lux-gain variant's gain: the stats median to ``target``, clip-capped."""
    lin = srgb_to_linear(rgb_u8[::4, ::4].astype(np.float32) / 255).reshape(-1, 3)
    lum = lin @ LUMA_709
    sel = lum[(lum >= base.stats_lum_min) & (lum <= base.stats_lum_max)]
    median = float(np.median(sel if sel.size > lum.size // 100 else lum))
    cap = 0.97 / max(float(np.percentile(lum, GAIN_CLIP_PCT)), 1e-6)
    # The cap only limits a lift: a frame already at the ceiling is left alone (gain 1),
    # never darkened for it. Darkening towards a lower target is not capped.
    return float(np.clip(target / max(median, 1e-6), GAIN[0], max(min(GAIN[1], cap), 1.0)))


def params_for(
    variant: str, base: NormalizeParams, lux: float, manual: bool, rgb_u8: np.ndarray
) -> NormalizeParams:
    if variant == "today":
        return base
    keep = dataclasses.replace(
        base, levels_max_stretch=1.0, levels_gamma_min=0.999999, levels_gamma_max=1.000001
    )
    if variant == "camera" or manual:
        return keep
    target = target_for(lux, TARGET)
    if variant == "lux-gain":
        g = exposure_gain(rgb_u8, base, target)
        # levels off: one scalar gain, pinned to g by its own limits.
        return dataclasses.replace(
            base, levels=False, exposure_gain_min=g * 0.999999, exposure_gain_max=g * 1.000001
        )
    return dataclasses.replace(
        base, levels_target_median=target, levels_gamma_min=GAMMA[0], levels_gamma_max=GAMMA[1]
    )


def median_lum(rgb_u8: np.ndarray) -> float:
    lin = srgb_to_linear(rgb_u8[::4, ::4, :3].astype(np.float32) / 255).reshape(-1, 3)
    return float(np.median(lin @ LUMA_709))


def collect(folders: list[Path]) -> list[dict]:
    shots = []
    for d in folders:
        for line in (d / "captures.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            md = r.get("camera_metadata") or {}
            src = d / r.get("original", "")
            if (
                "imx477" not in str(r.get("tuning_file"))
                or "Lux" not in md
                or r.get("double")
                or not src.exists()
            ):
                continue
            shots.append(
                {
                    "src": src,
                    "folder": d.name,
                    "lux": float(md["Lux"]),
                    "exposure_us": md.get("ExposureTime"),
                    "gain": md.get("AnalogueGain"),
                    "manual": bool(r.get("shutter_us") or r.get("iso")),
                    "seed": r.get("grain_seed") or 0,
                }
            )
    return shots


def choose(shots: list[dict]) -> list[tuple[str, list[dict]]]:
    """Up to PER_BIN a bin, spread over the bin's shots, every manual one kept."""
    out = []
    for lo, hi, title in BINS:
        b = sorted(
            (s for s in shots if lo <= s["lux"] < hi), key=lambda s: (s["folder"], s["src"].name)
        )
        manual = [s for s in b if s["manual"]]
        auto = [s for s in b if not s["manual"]]
        room = max(PER_BIN - len(manual), 0)
        if len(auto) > room:
            idx = np.linspace(0, len(auto) - 1, room).round().astype(int) if room else []
            auto = [auto[i] for i in sorted(set(idx))]
        if manual or auto:
            out.append((title, sorted(manual + auto, key=lambda s: s["lux"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--look", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--long-edge", type=int, default=1600, help="size of the gallery JPEGs")
    ap.add_argument("folders", nargs="+", type=Path)
    args = ap.parse_args()

    look = Artifacts.load(args.look)
    args.out.mkdir(parents=True, exist_ok=True)
    groups = choose(collect(args.folders))
    pipelines: dict = {}
    summary = []
    sections = []
    for title, shots in groups:
        rows = []
        for s in shots:
            rgb = load_rgb(s["src"])[0][::2, ::2]  # half size: grain is set in mm of the frame
            stem = (
                f"{s['folder']}_{s['src'].stem}".replace(" ", "_").replace("(", "").replace(")", "")
            )
            cells, meds = [], {}
            for v in VARIANTS:
                p = params_for(v, look.normalize, s["lux"], s["manual"], rgb)
                key = json.dumps(dataclasses.asdict(p), sort_keys=True)
                if key not in pipelines:
                    pipelines[key] = Pipeline(dataclasses.replace(look, normalize=p))
                graded, _ = pipelines[key].process(rgb, rng=np.random.default_rng(s["seed"]))
                h, w = graded.shape[:2]
                k = max(1, math.ceil(max(h, w) / args.long_edge))
                name = f"{stem}_{v}.jpg"
                save_jpeg(np.ascontiguousarray(graded[::k, ::k, :3]), args.out / name, quality=88)
                meds[v] = median_lum(graded)
                cells.append(
                    f'<figure><a href="{name}"><img loading="lazy" src="{name}"></a>'
                    f"<figcaption>{v} &middot; median {meds[v]:.3f}</figcaption></figure>"
                )
            info = (
                f"{html.escape(s['folder'])}/{html.escape(s['src'].name)} &middot; "
                f"{s['lux']:.0f} lux &middot; {s['exposure_us']} &micro;s &middot; "
                f"gain {s['gain']:.2f}"
                + (" &middot; <b>manual</b>" if s["manual"] else "")
            )
            rows.append(
                f'<section class="shot"><p>{info}</p>'
                f'<div class="row">{"".join(cells)}</div></section>'
            )
            summary.append(
                {
                    "name": f"{s['folder']}/{s['src'].name}",
                    "lux": s["lux"],
                    "manual": s["manual"],
                    "median": meds,
                }
            )
            print(
                f"{s['lux']:8.0f} lux  " + "  ".join(f"{v} {meds[v]:.3f}" for v in VARIANTS),
                flush=True,
            )
        sections.append(f"<h2>{html.escape(title)} ({len(shots)})</h2>" + "".join(rows))

    curve = ", ".join(f"{10**x:.0f} lux {t}" for x, t in zip(LOG_LUX, TARGET, strict=True))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Exposure from Lux</title>
<style>
body{{margin:0;padding:16px;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
h1{{font-size:20px}} h2{{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}}
figure{{margin:0}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#aaa;padding:2px 0 8px}}
.shot p{{margin:16px 0 6px;color:#bbb}}
@media (max-width:800px){{.row{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<h1>Exposure from the camera's Lux reading</h1>
<p>Columns: <b>today</b> (the deployed look), <b>lux</b> (the median placed from the scene's Lux by
the levels gamma, lifting allowed), <b>lux-gain</b> (the same target by a plain gain, like a longer
exposure, highlights protected), <b>camera</b> (no stretch or gamma: the camera's own exposure).
Manual shots keep the camera's exposure in the lux columns. Median = linear luminance of the
graded picture. Target median: {html.escape(curve)}. Click a picture for the larger file.</p>
{"".join(sections)}
</body></html>"""
    (args.out / "index.html").write_text(page)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out / 'index.html'} ({len(summary)} photographs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
