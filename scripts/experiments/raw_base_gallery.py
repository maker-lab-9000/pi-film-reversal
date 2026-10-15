"""Experiment: the IMX477 DNGs developed by ``pifilm.raw``, beside the camera's JPEG.

Task 3 of docs/superpowers/plans/2026-10-14-raw-developer.md (the raw look,
phase 1). Offline only; nothing here changes a default. The owner judges from
the gallery whether a developed DNG is a good base to train a look on.

Each photograph is developed with each tone curve of ``pifilm.raw.TONE_CURVES``
from the shot's own ``ColourGains`` and ``ColourTemperature`` and the camera's
tuning file. Nothing is graded: no LUT, no grain.

Two checks come with the pictures:

* Validation. The ``isp`` column should come close to the camera's JPEG: same
  tables, same order, without the ISP's adaptive contrast, denoise, sharpening
  and adaptive lens shading. Per channel, the mean absolute and the mean signed
  difference against the JPEG (a cast shows as a signed mean), and the
  difference in centre-to-corner brightness and in corner-minus-centre
  log(R/G) and log(B/G) (a lens-shading error, of brightness or of colour, shows there),
  over all photographs and per Lux bin.
* Does 8 bits hold what the raw has? For the 6 darkest and the 6 brightest
  scenes by Lux, a 600 x 400 crop of the full-size ``soft-1.5`` development
  (the deepest shadow area, or the brightest area that is not clipped) is
  pushed by +2 or -2 stops after development, once from the uint8 picture and
  once from the same development in float32 (``develop(..., as_float=True)``).
  Banding or lost texture in the uint8 crop that the float crop does not have
  means grading needs a float path. Under each, the number of distinct levels
  in the crop's green channel after the push. The crops are PNG, so JPEG
  compression neither hides nor adds banding.

Usage (from the repo root; the training photographs, then the 13 held-out
scenes named in ``data/test-scenes-imx477/`` from their capture folders):

    .venv/bin/python scripts/experiments/raw_base_gallery.py \\
        --tuning data/tuning/imx477.json --out artifacts/raw-base-gallery \\
        --held-out data/test-scenes-imx477 --photos ~/Downloads/PiCameraPhotos \\
        --jobs 4 ~/Downloads/PiTrainingSource/2026-09-30

Writes ``index.html``, the JPEGs, the PNG crops and ``summary.json`` into
``--out``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pifilm._cv2 import require_cv2
from pifilm.color import LUMA_709, linear_to_srgb, srgb_to_linear
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.raw import TONE_CURVES, develop, read_dng, tuning_tables

# The Lux bins of exposure_from_lux.py.
BINS = [
    (0, 10, "Night, under 10 lux"),
    (10, 50, "Dim, 10 to 50 lux"),
    (50, 200, "Indoor, 50 to 200 lux"),
    (200, 1000, "Bright indoor or overcast, 200 to 1,000 lux"),
    (1000, 5000, "Daylight, 1,000 to 5,000 lux"),
    (5000, math.inf, "Bright daylight, over 5,000 lux"),
]
COLUMNS = ["camera", *TONE_CURVES]
CROP = (400, 600)  # rows, columns
CROP_TONE = "soft-1.5"
PUSH_STOPS = 2.0
PUSHED_PER_END = 6
# A highlight crop may hold at most this share of pixels at the ceiling.
CROP_CLIP_SHARE = 0.001


def bin_of(lux: float) -> int:
    return next(i for i, (lo, hi, _) in enumerate(BINS) if lo <= lux < hi)


def lum_linear(rgb: np.ndarray) -> np.ndarray:
    """Linear luminance of a display picture, uint8 or float 0..1."""
    x = rgb.astype(np.float32) / 255 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
    return srgb_to_linear(x) @ LUMA_709


def stats(rgb_u8: np.ndarray) -> dict:
    """Median linear luminance (of every fourth pixel) and the share at 255."""
    med = float(np.median(lum_linear(rgb_u8[::4, ::4])))
    clipped = float(np.mean(np.any(rgb_u8 == 255, axis=2)))
    return {"median": med, "at_255": clipped}


def centre_to_corner(rgb_u8: np.ndarray) -> float:
    """Mean linear luminance of the centre fifth over that of the four corner tenths."""
    cv2 = require_cv2()
    h, w = rgb_u8.shape[:2]
    small = cv2.resize(rgb_u8, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    lum = lum_linear(small)
    sh, sw = lum.shape
    ch, cw = sh // 5, sw // 5
    centre = lum[(sh - ch) // 2 : (sh + ch) // 2, (sw - cw) // 2 : (sw + cw) // 2].mean()
    kh, kw = sh // 10, sw // 10
    corners = np.mean(
        [lum[:kh, :kw].mean(), lum[:kh, -kw:].mean(), lum[-kh:, :kw].mean(), lum[-kh:, -kw:].mean()]
    )
    return float(centre / max(corners, 1e-6))


def colour_shading(rgb_u8: np.ndarray) -> tuple[float, float]:
    """Corner minus centre of log(R/G) and log(B/G), in linear light.

    A colour lens-shading error (a red or blue tint growing towards the
    corners) shows here, where the brightness ratio above cannot see it.
    """
    cv2 = require_cv2()
    h, w = rgb_u8.shape[:2]
    small = cv2.resize(rgb_u8, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    lin = srgb_to_linear(small.astype(np.float32) / 255)
    sh, sw = lin.shape[:2]
    ch, cw = sh // 5, sw // 5
    kh, kw = sh // 10, sw // 10
    centre = lin[(sh - ch) // 2 : (sh + ch) // 2, (sw - cw) // 2 : (sw + cw) // 2]
    corners = [lin[:kh, :kw], lin[:kh, -kw:], lin[-kh:, :kw], lin[-kh:, -kw:]]

    def log_ratios(region: np.ndarray) -> np.ndarray:
        r, g, b = (max(float(v), 1e-6) for v in region.reshape(-1, 3).mean(axis=0))
        return np.array([math.log(r / g), math.log(b / g)])

    corner = np.mean([log_ratios(c) for c in corners], axis=0)
    d = corner - log_ratios(centre)
    return float(d[0]), float(d[1])


def shrink(rgb_u8: np.ndarray, long_edge: int) -> np.ndarray:
    cv2 = require_cv2()
    h, w = rgb_u8.shape[:2]
    k = long_edge / max(h, w)
    if k >= 1:
        return rgb_u8
    return cv2.resize(rgb_u8, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def collect(folders: list[Path], held_out: Path | None, photos: Path | None) -> list[dict]:
    """Every usable record of ``folders``, and the held-out scenes from ``photos``."""
    wanted: dict[Path, set[str]] = {}
    for f in sorted(held_out.glob("*_original.jpg")) if held_out else []:
        day, hms, _ = f.name.split("_")
        wanted.setdefault(photos / day, set()).add(f"{hms}_original.jpg")
    sources = [(d, None, False) for d in folders] + [
        (d, names, True) for d, names in sorted(wanted.items())
    ]
    shots = []
    for d, names, is_held_out in sources:
        for line in (d / "captures.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            md = r.get("camera_metadata") or {}
            original = d / r.get("original", "")
            dng = d / str(r.get("dng"))
            if names is not None and original.name not in names:
                continue
            if (
                not r.get("dng")
                or "imx477" not in str(r.get("tuning_file"))
                or not dng.exists()
                or not original.exists()
                or "ColourGains" not in md
                or "ColourTemperature" not in md
            ):
                continue
            shots.append(
                {
                    "dng": str(dng),
                    "original": str(original),
                    "folder": d.name,
                    "stem": f"{d.name}_{original.stem.removesuffix('_original')}",
                    "lux": float(md.get("Lux", math.nan)),
                    "exposure_us": md.get("ExposureTime"),
                    "gain": md.get("AnalogueGain"),
                    "colour_gains": tuple(md["ColourGains"]),
                    "ct": float(md["ColourTemperature"]),
                    "digital_gain": float(md.get("DigitalGain", 1.0)),
                    "held_out": is_held_out,
                }
            )
    return shots


_TUNING = None


def _tuning(path: str):
    global _TUNING
    if _TUNING is None:
        _TUNING = tuning_tables(json.loads(Path(path).read_text()))
    return _TUNING


def _develop(s: dict, tuning_path: str, tone: str, as_float: bool = False) -> np.ndarray:
    frame = read_dng(Path(s["dng"]).read_bytes())
    return develop(
        frame,
        _tuning(tuning_path),
        colour_gains=s["colour_gains"],
        colour_temperature=s["ct"],
        tone=tone,
        as_float=as_float,
        digital_gain=s["digital_gain"],
    )


def develop_shot(s: dict, tuning_path: str, out: Path, long_edge: int) -> dict:
    """The gallery JPEGs of one shot, its stats, and the isp-against-camera numbers."""
    cv2 = require_cv2()
    camera = load_rgb(s["original"])[0]
    result = {"stats": {}, "files": {}}
    for column in COLUMNS:
        rgb = camera if column == "camera" else _develop(s, tuning_path, column)
        name = f"{s['stem']}_{column}.jpg"
        save_jpeg(np.ascontiguousarray(shrink(rgb, long_edge)), out / name, quality=90)
        result["files"][column] = name
        result["stats"][column] = stats(rgb)
        if column == "isp":
            ref = camera
            if ref.shape != rgb.shape:
                ref = cv2.resize(ref, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
            diff = rgb.astype(np.float32) - ref.astype(np.float32)
            result["signed"] = diff.reshape(-1, 3).mean(axis=0).tolist()
            result["absolute"] = np.abs(diff).reshape(-1, 3).mean(axis=0).tolist()
            result["shading"] = {"isp": centre_to_corner(rgb), "camera": centre_to_corner(ref)}
            isp_c, cam_c = colour_shading(rgb), colour_shading(ref)
            result["colour_shading"] = {
                "isp": isp_c,
                "camera": cam_c,
                "isp_minus_camera": [isp_c[0] - cam_c[0], isp_c[1] - cam_c[1]],
            }
            result["same_size"] = camera.shape == rgb.shape
            del diff
    return result


def find_crop(picture: np.ndarray, shadow: bool) -> tuple[int, int]:
    """The top-left of the darkest, or the brightest unclipped, CROP window."""
    cv2 = require_cv2()
    k = 8
    lum = lum_linear(picture[::k, ::k])
    clip = np.any(picture[::k, ::k] >= 0.999, axis=2).astype(np.float32)
    wh, ww = CROP[0] // k, CROP[1] // k
    mean = cv2.boxFilter(lum, -1, (ww, wh), normalize=True, borderType=cv2.BORDER_REFLECT)
    share = cv2.boxFilter(clip, -1, (ww, wh), normalize=True, borderType=cv2.BORDER_REFLECT)
    # Only windows that lie wholly inside the frame: centres at least half a window in.
    valid = np.zeros_like(mean, dtype=bool)
    valid[
        wh // 2 : lum.shape[0] - (wh - wh // 2) + 1, ww // 2 : lum.shape[1] - (ww - ww // 2) + 1
    ] = True
    if shadow:
        score = np.where(valid, mean, np.inf)
        cy, cx = np.unravel_index(np.argmin(score), score.shape)
    else:
        score = np.where(valid & (share <= CROP_CLIP_SHARE), mean, -np.inf)
        if not np.isfinite(score).any():
            score = np.where(valid, -share, -np.inf)
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
    top = int(np.clip((cy - wh // 2) * k, 0, picture.shape[0] - CROP[0]))
    left = int(np.clip((cx - ww // 2) * k, 0, picture.shape[1] - CROP[1]))
    return top, left


def push(display: np.ndarray, stops: float) -> np.ndarray:
    """``stops`` of exposure in linear light on a display picture, back to uint8."""
    x = display.astype(np.float32) / 255 if display.dtype == np.uint8 else display
    lin = srgb_to_linear(x) * np.float32(2.0**stops)
    out = linear_to_srgb(np.clip(lin, 0.0, 1.0))
    return np.rint(out * 255).astype(np.uint8)


def pushed_crops(s: dict, tuning_path: str, out: Path, shadow: bool) -> dict:
    cv2 = require_cv2()
    picture = _develop(s, tuning_path, CROP_TONE, as_float=True)
    as_u8 = np.rint(picture * np.float32(255)).astype(np.uint8)  # what develop returns
    top, left = find_crop(picture, shadow)
    window = (slice(top, top + CROP[0]), slice(left, left + CROP[1]))
    stops = PUSH_STOPS if shadow else -PUSH_STOPS
    result = {"top": top, "left": left, "stops": stops, "files": {}, "levels": {}}
    for kind, source in (("uint8", as_u8), ("float", picture)):
        crop = push(source[window], stops)
        name = f"{s['stem']}_push_{kind}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        result["files"][kind] = name
        result["levels"][kind] = int(np.unique(crop[..., 1]).size)
    result["levels"]["before_push"] = int(np.unique(as_u8[window][..., 1]).size)
    return result


def aggregate(results: list[tuple[dict, dict]]) -> dict:
    def block(items):
        if not items:
            return None
        signed = np.mean([r["signed"] for _, r in items], axis=0)
        absolute = np.mean([r["absolute"] for _, r in items], axis=0)
        shading = np.mean([r["shading"]["isp"] - r["shading"]["camera"] for _, r in items])
        colour = np.mean([r["colour_shading"]["isp_minus_camera"] for _, r in items], axis=0)
        return {
            "n": len(items),
            "signed_rgb": [round(float(v), 2) for v in signed],
            "absolute_rgb": [round(float(v), 2) for v in absolute],
            "centre_to_corner_isp_minus_camera": round(float(shading), 3),
            "colour_shading_isp_minus_camera": [round(float(v), 4) for v in colour],
        }

    out = {"all": block(results)}
    for i, (_, _, title) in enumerate(BINS):
        b = block(
            [(s, r) for s, r in results if not math.isnan(s["lux"]) and bin_of(s["lux"]) == i]
        )
        if b:
            out[title] = b
    unknown = [(s, r) for s, r in results if math.isnan(s["lux"])]
    if unknown:
        out["Lux unknown"] = block(unknown)
    return out


def figure(name: str, caption: str) -> str:
    return (
        f'<figure><a href="{name}"><img loading="lazy" src="{name}"></a>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tuning", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--long-edge", type=int, default=1600, help="size of the gallery JPEGs")
    ap.add_argument("--held-out", type=Path, help="folder of <day>_<HHMMSS>_original.jpg scenes")
    ap.add_argument("--photos", type=Path, help="the capture folders of the held-out scenes")
    ap.add_argument("--jobs", type=int, default=1, help="photographs developed at once")
    ap.add_argument("--limit", type=int, help="only the first N photographs, for a quick try")
    ap.add_argument("folders", nargs="*", type=Path)
    args = ap.parse_args()
    if args.held_out and not args.photos:
        ap.error("--held-out needs --photos")

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    shots = collect(args.folders, args.held_out, args.photos)
    if args.limit:
        shots = shots[: args.limit]
    tuning = str(args.tuning)
    _tuning(tuning)  # a bad tuning file fails before any work
    print(f"{len(shots)} photographs ({sum(s['held_out'] for s in shots)} held out)", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(develop_shot, s, tuning, args.out, args.long_edge) for s in shots]
        results = []
        for s, f in zip(shots, futures, strict=True):
            r = f.result()
            results.append((s, r))
            print(
                f"{s['lux']:8.1f} lux  {s['stem']:28s} signed "
                + " ".join(f"{v:+5.1f}" for v in r["signed"])
                + "  abs "
                + " ".join(f"{v:4.1f}" for v in r["absolute"]),
                flush=True,
            )

        by_lux = sorted((s for s in shots if not math.isnan(s["lux"])), key=lambda s: s["lux"])
        ends = [(s, True) for s in by_lux[:PUSHED_PER_END]] + [
            (s, False) for s in by_lux[-PUSHED_PER_END:]
        ]
        push_futures = [pool.submit(pushed_crops, s, tuning, args.out, sh) for s, sh in ends]
        pushed = [(s, sh, f.result()) for (s, sh), f in zip(ends, push_futures, strict=True)]

    validation = aggregate(results)
    print(
        "isp minus camera: signed R/G/B | absolute R/G/B | brightness shading | log(R/G) log(B/G)"
    )
    for k, v in validation.items():
        print(
            f"{k:45s} n={v['n']:3d}  "
            + " ".join(f"{x:+5.1f}" for x in v["signed_rgb"])
            + " | "
            + " ".join(f"{x:4.1f}" for x in v["absolute_rgb"])
            + f" | {v['centre_to_corner_isp_minus_camera']:+.3f} | "
            + " ".join(f"{x:+.3f}" for x in v["colour_shading_isp_minus_camera"]),
            flush=True,
        )
    for s, _shadow, p in pushed:
        print(
            f"push {p['stops']:+.0f}  {s['lux']:8.1f} lux  {s['stem']:28s} green levels "
            f"uint8 {p['levels']['uint8']:4d}  float {p['levels']['float']:4d}",
            flush=True,
        )

    sections = []
    for i, (_, _, title) in enumerate([*BINS, (0, 0, "Lux unknown")]):
        rows = []
        for s, r in results:
            in_bin = (
                math.isnan(s["lux"])
                if i == len(BINS)
                else (not math.isnan(s["lux"]) and bin_of(s["lux"]) == i)
            )
            if not in_bin:
                continue
            cells = [
                figure(
                    r["files"][c],
                    f"{c} &middot; median {r['stats'][c]['median']:.3f} &middot; "
                    f"at 255 {100 * r['stats'][c]['at_255']:.2f}%",
                )
                for c in COLUMNS
            ]
            info = (
                f"{html.escape(s['folder'])}/{html.escape(Path(s['original']).name)} &middot; "
                f"{s['lux']:.0f} lux &middot; {s['exposure_us']} &micro;s &middot; "
                f"gain {s['gain']:.2f} &middot; {s['ct']:.0f} K"
                + (' &middot; <b class="held">held out</b>' if s["held_out"] else "")
                + f" &middot; isp minus camera {' '.join(f'{v:+.1f}' for v in r['signed'])}"
            )
            rows.append(
                f'<section class="shot"><p>{info}</p>'
                f'<div class="row">{"".join(cells)}</div></section>'
            )
        if rows:
            sections.append(f"<h2>{html.escape(title)} ({len(rows)})</h2>" + "".join(rows))

    push_rows = []
    for s, sh, p in pushed:
        what = (
            "deepest shadow, pushed +2 stops" if sh else "brightest unclipped area, pulled -2 stops"
        )
        info = (
            f"{html.escape(s['stem'])} &middot; {s['lux']:.1f} lux &middot; {what} &middot; "
            f"crop at ({p['left']}, {p['top']})"
            + (' &middot; <b class="held">held out</b>' if s["held_out"] else "")
        )
        cells = [
            figure(p["files"][k], f"from {k} &middot; {p['levels'][k]} green levels")
            for k in ("uint8", "float")
        ]
        push_rows.append(
            f'<section class="shot"><p>{info}</p>'
            f'<div class="row two">{"".join(cells)}</div></section>'
        )

    rows_html = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td>"
        f"<td>{' / '.join(f'{x:+.1f}' for x in v['signed_rgb'])}</td>"
        f"<td>{' / '.join(f'{x:.1f}' for x in v['absolute_rgb'])}</td>"
        f"<td>{v['centre_to_corner_isp_minus_camera']:+.3f}</td>"
        f"<td>{' / '.join(f'{x:+.3f}' for x in v['colour_shading_isp_minus_camera'])}"
        "</td></tr>"
        for k, v in validation.items()
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Raw base gallery</title>
<style>
body{{margin:0;padding:16px;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
h1{{font-size:20px}} h2{{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}}
.row.two{{grid-template-columns:repeat(2,minmax(0,600px))}}
figure{{margin:0}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#aaa;padding:2px 0 8px}}
.shot p{{margin:16px 0 6px;color:#bbb}} .held{{color:#e8b04a}}
table{{border-collapse:collapse;margin:8px 0}}
td,th{{padding:2px 10px;border-bottom:1px solid #333;text-align:left}}
@media (max-width:800px){{.row{{grid-template-columns:repeat(2,1fr)}}
.row.two{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>The DNG developed with the camera's tuning, beside the camera's JPEG</h1>
<p>Columns: <b>camera</b> (the camera's own <code>_original.jpg</code>), <b>isp</b> (the DNG
developed with the tuning file's lens shading, the shot's white balance and colour matrix, and the
ISP's gamma curve), <b>soft-1</b> and <b>soft-1.5</b> (the same with the shadows lifted 1 and 1.5
stops before the curve and the highlights rolled off). Nothing here is graded: no LUT, no grain.
Under each: the median linear luminance and the share of pixels with a channel at 255. Click a
picture for the larger file.</p>
<h2>isp against the camera's JPEG (levels, R / G / B)</h2>
<table><tr><th>photographs</th><th>n</th><th>signed mean</th><th>mean absolute</th>
<th>centre to corner, isp minus camera</th>
<th>corner minus centre log(R/G) / log(B/G), isp minus camera</th></tr>{rows_html}</table>
<h2>Does 8 bits hold what the raw has?</h2>
<p>The {PUSHED_PER_END} darkest and {PUSHED_PER_END} brightest scenes by Lux: a full-resolution
{CROP[1]} x {CROP[0]} crop of the <b>{CROP_TONE}</b> development, pushed after development, once
from the uint8 picture and once from the same development kept in float. Banding or lost texture
on the left that the right does not have means grading needs a float path. PNG, so no JPEG
artefacts.</p>
{"".join(push_rows)}
{"".join(sections)}
</body></html>"""
    (args.out / "index.html").write_text(page)
    summary = {
        "validation": validation,
        "pushed": [{"name": s["stem"], "lux": s["lux"], "shadow": sh, **p} for s, sh, p in pushed],
        "photographs": [
            {
                "name": s["stem"],
                "lux": s["lux"],
                "held_out": s["held_out"],
                "colour_temperature": s["ct"],
                **{
                    k: r[k]
                    for k in (
                        "stats",
                        "signed",
                        "absolute",
                        "shading",
                        "colour_shading",
                        "same_size",
                    )
                },
                "digital_gain": s["digital_gain"],
            }
            for s, r in results
        ],
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out / 'index.html'} ({len(results)} photographs) in {summary['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
