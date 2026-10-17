"""Gallery: halation on the colour look, at a few strengths and radii, for choosing by eye.

Halation is light that passes through the emulsion, reflects off the film base and comes
back through the red-sensitive layer (the deepest), so bright highlights grow a soft
red-orange glow. It happens at exposure, so every model here works on the normalised frame
in linear light, before the LUT (the look's own roll-off), with the glow's width in mm of a
36 mm frame, as the grain's is. Grain is added last, as in the pipeline, one seed per photo.

- A, threshold: blur only the highlights above a soft threshold, tint red-orange, add.
- B, global mix: mix a blurred copy of the whole frame (a sum of three Gaussians, about an
  exponential fall-off) into red and a little green. No threshold; flat areas unchanged.
- C, B with headroom: the camera JPEG has clipped its highlights, so before the mix they are
  expanded (x / (1 - c x)) and after it compressed back exactly (y / (1 + c y)).

Blurs run on a copy reduced by a whole factor and scaled back up; the glow is soft, so
this loses nothing visible and is what a Pi version would do.

Only photographs with some clipped highlights are used (between ``--min-clip`` and
``--max-clip`` of the pixels near white), at most ``--per-folder`` from each folder.

    .venv/bin/python scripts/experiments/halation_gallery.py \\
        --artifacts artifacts/imx477-2026-09-v4-dark-hp5-tile \\
        --out artifacts/halation-gallery \\
        data/test-scenes-imx477 data/source-imx477-v3-highlight data/pi-2026-09-30
"""

from __future__ import annotations

import argparse
import html
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pifilm._cv2 import require_cv2
from pifilm.artifacts import Artifacts
from pifilm.color import LUMA_709, linear_to_srgb, srgb_to_linear
from pifilm.grain import add_grain
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.normalize import normalize_u8

TINT = np.array([1.0, 0.28, 0.06], dtype=np.float32)  # red-orange, mostly the red layer
THRESHOLD = 0.70  # linear luminance where the glow starts (sRGB about 0.85)


@dataclass(frozen=True)
class Variant:
    tag: str
    caption: str
    model: str = "off"  # "off", "threshold", "mix"
    strength: float = 0.0
    sigma_mm: float = 0.0
    headroom: float = 0.0  # mix only: c in x / (1 - c x); 0 is none, 0.9 is 10x at white


VARIANTS = (
    Variant("off", "no halation"),
    Variant("mix", "B global mix: 0.15, 0.25 mm", "mix", 0.15, 0.25),
    Variant("mixhr2", "C headroom 2x: 0.15, 0.25 mm", "mix", 0.15, 0.25, 0.5),
    Variant("mixhr4", "C headroom 4x: 0.15, 0.25 mm", "mix", 0.15, 0.25, 0.75),
    Variant("mixhr4-narrow", "C headroom 4x, narrow: 0.15, 0.12 mm", "mix", 0.15, 0.12, 0.75),
    Variant("mixhr10-subtle", "C headroom 10x: 0.08, 0.25 mm (round 1)", "mix", 0.08, 0.25, 0.9),
)

MIX_WEIGHTS = np.array([1.0, 0.35, 0.0], dtype=np.float32)  # mostly red, some green
FALLOFF = ((1.0, 0.5), (2.0, 0.3), (4.0, 0.2))  # sigma multiples and weights: about exponential


def _blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur on a copy reduced by a whole factor, scaled back up."""
    cv2 = require_cv2()
    h, w = img.shape[:2]
    k = max(1, int(sigma_px // 4))  # keep at least four reduced pixels of sigma
    small = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma_px / k, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _falloff_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    return sum(wt * _blur(img, m * sigma_px) for m, wt in FALLOFF)


def glow_map(lin: np.ndarray, sigma_mm: float, frame_width_mm: float = 36.0) -> np.ndarray:
    """The blurred highlight mask, (H, W) float32, 0 to about 1."""
    y = lin @ LUMA_709
    t = np.clip((y - THRESHOLD) / (1.0 - THRESHOLD), 0.0, 1.0)
    mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep: no hard edge
    return _blur(mask, sigma_mm * mask.shape[1] / frame_width_mm)


def halate_threshold(lin: np.ndarray, v: Variant) -> np.ndarray:
    """Model A: blur the highlights above a soft threshold, tint, add."""
    return lin + v.strength * glow_map(lin, v.sigma_mm)[..., None] * TINT


def halate_mix(lin: np.ndarray, v: Variant) -> np.ndarray:
    """Model B/C: mix a blurred copy of the whole frame into red (and a little green).

    Linear and global, no threshold: a flat area equals its own blur, so it is unchanged
    and there is no white-balance shift. The camera JPEG has clipped its highlights, so a
    lamp is no brighter than white paper and its halo would be far weaker than on film;
    ``headroom`` undoes that roughly with x / (1 - c x) before the mix (white becomes
    1 / (1 - c)) and puts it back exactly with y / (1 + c y) after, the roll-off.
    """
    c = v.headroom
    x = lin / (1.0 - c * lin) if c else lin
    sigma_px = v.sigma_mm * lin.shape[1] / 36.0
    x = x + (v.strength * MIX_WEIGHTS) * (_falloff_blur(x, sigma_px) - x)
    return x / (1.0 + c * x) if c else x


def halate(rgb_u8: np.ndarray, v: Variant) -> np.ndarray:
    lin = srgb_to_linear(rgb_u8[..., :3] / 255.0)
    lin = halate_threshold(lin, v) if v.model == "threshold" else halate_mix(lin, v)
    return np.rint(linear_to_srgb(lin) * 255.0).astype(np.uint8)


def grade(art: Artifacts, lut_filter, rgb: np.ndarray, v: Variant, seed: int) -> np.ndarray:
    img, _ = normalize_u8(rgb, art.normalize)
    if v.model != "off":
        img = halate(img, v)
    img = art.lut.apply_pillow(img, lut_filter)[..., :3]
    if art.grain.enabled:
        img = add_grain(img, art.grain, np.random.default_rng(seed))
    return img


def clipped_share(rgb: np.ndarray) -> float:
    small = rgb[::8, ::8, :3].astype(np.float32) @ LUMA_709
    return float((small > 242).mean())


def _small(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    k = max(1, math.ceil(max(rgb.shape[:2]) / long_edge))
    return np.ascontiguousarray(rgb[::k, ::k, :3])


def _crop_box(rgb: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A full-size crop centred on where the medium glow is strongest."""
    lin = srgb_to_linear(rgb[..., :3] / 255.0)
    g = glow_map(lin, 0.6)
    y, x = np.unravel_index(int(np.argmax(g)), g.shape)
    cw, ch = min(size[0], rgb.shape[1]), min(size[1], rgb.shape[0])
    x0 = int(np.clip(x - cw // 2, 0, rgb.shape[1] - cw))
    y0 = int(np.clip(y - ch // 2, 0, rgb.shape[0] - ch))
    return x0, y0, cw, ch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--long-edge", type=int, default=1600)
    ap.add_argument("--per-folder", type=int, default=6)
    ap.add_argument("--min-clip", type=float, default=0.001)
    ap.add_argument("--max-clip", type=float, default=0.08)
    ap.add_argument("folders", nargs="+", type=Path)
    args = ap.parse_args()
    art = Artifacts.load(args.artifacts)
    lut_filter = art.lut.to_pillow()
    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for folder in args.folders:
        scored = []
        for p in sorted(folder.glob("*_original.jpg")):
            share = clipped_share(load_rgb(p)[0])
            if args.min_clip <= share <= args.max_clip:
                scored.append((share, p))
        picked = sorted(p for _, p in sorted(scored, reverse=True)[: args.per_folder])
        rows = []
        for p in picked:
            stem = f"{folder.name}_{p.name.removesuffix('_original.jpg')}"
            seed = zlib.crc32(stem.encode())
            rgb = load_rgb(p)[0]
            x0, y0, cw, ch = _crop_box(rgb, (720, 540))
            full, crops = [], []
            for v in VARIANTS:
                out = grade(art, lut_filter, rgb, v, seed)
                save_jpeg(_small(out, args.long_edge), args.out / f"{stem}_{v.tag}.jpg",
                          quality=90)
                crop = np.ascontiguousarray(out[y0:y0 + ch, x0:x0 + cw, :3])
                save_jpeg(crop, args.out / f"{stem}_{v.tag}_crop.jpg", quality=92)
                cap = html.escape(v.caption)
                full.append(f'<figure><a href="{stem}_{v.tag}.jpg"><img loading="lazy" '
                            f'src="{stem}_{v.tag}.jpg"></a><figcaption>{cap}</figcaption>'
                            "</figure>")
                crops.append(f'<figure><a href="{stem}_{v.tag}_crop.jpg"><img loading="lazy" '
                             f'src="{stem}_{v.tag}_crop.jpg"></a><figcaption>{cap}, 100% crop'
                             "</figcaption></figure>")
            rows.append(f'<section><p>{html.escape(stem)}</p><div class="row">{"".join(full)}'
                        f'</div><div class="row">{"".join(crops)}</div></section>')
            print(stem, flush=True)
        sections.append(f"<h2>{html.escape(folder.name)} ({len(picked)})</h2>" + "".join(rows))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Halation test</title>
<style>
body{{margin:0;padding:16px;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
h1{{font-size:20px}} h2{{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}}
.row{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:6px}}
figure{{margin:0}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#aaa;padding:2px 0 8px}} section p{{margin:24px 0 6px}}
@media (max-width:800px){{.row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Halation on the colour look</h1>
<p>Round 2. Each photograph six ways, all before the LUT, glow width in mm of a 36 mm
frame. <b>B</b>: a blurred copy of the whole frame (about an exponential fall-off) mixed into
red and a little green; flat areas are unchanged. <b>C</b>: B with the JPEG's clipped
highlights expanded before the mix and compressed back after it, so a clipped area counts
as 2x, 4x or 10x white. The same grain seed for every version of a photograph. The second
row is a 100% crop where the glow is strongest. Click a picture for the larger file.</p>
{"".join(sections)}
</body></html>"""
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
