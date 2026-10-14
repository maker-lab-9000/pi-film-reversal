"""Gallery: today's look on the camera's JPEG beside the raw look on the developed DNG.

Phase 2 of docs/superpowers/specs/2026-10-14-raw-look-design.md, the like-for-like
test: the same photographs, the same references and training settings, the same
normalisation and grain; only the input differs (the camera's JPEG against the
DNG developed by ``pifilm.raw`` with the ISP's tone curve). Both are graded with
the same grain seed per photograph, so the grain does not differ either.

    .venv/bin/python scripts/experiments/raw_look_gallery.py \\
        --today artifacts/imx477-2026-09-v4-dark-hp5-tile \\
        --raw artifacts/imx477-dng-isp-v1-dark-tile --out artifacts/raw-look-gallery \\
        data/source-imx477-v3-highlight:data/source-imx477-v3-highlight-dng-isp \\
        data/test-scenes-imx477:data/test-scenes-imx477-dng-isp:held-out
"""

from __future__ import annotations

import argparse
import html
import math
import zlib
from pathlib import Path

import numpy as np

from pifilm.artifacts import Artifacts
from pifilm.imageio import load_rgb, save_jpeg
from pifilm.pipeline import Pipeline


def _small(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    k = max(1, math.ceil(max(rgb.shape[:2]) / long_edge))
    return np.ascontiguousarray(rgb[::k, ::k, :3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--today", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--long-edge", type=int, default=1600)
    ap.add_argument("pairs", nargs="+", help="JPEG_DIR:DNG_DEVELOPED_DIR[:label]")
    args = ap.parse_args()
    today = Pipeline(Artifacts.load(args.today))
    raw = Pipeline(Artifacts.load(args.raw))
    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for spec in args.pairs:
        jpeg_dir, dng_dir, *label = spec.split(":")
        names = sorted(p.name for p in Path(jpeg_dir).glob("*_original.jpg"))
        rows = []
        for name in names:
            stem = name.removesuffix("_original.jpg")
            seed = zlib.crc32(stem.encode())
            cam = load_rgb(Path(jpeg_dir) / name)[0]
            dev = load_rgb(Path(dng_dir) / name)[0]
            a, _ = today.process(cam, rng=np.random.default_rng(seed))
            b, _ = raw.process(dev, rng=np.random.default_rng(seed))
            cells = []
            for tag, img, cap in (("camera", cam, "camera JPEG, ungraded"),
                                  ("today", a, "today's look (camera JPEG)"),
                                  ("raw", b, "raw look (developed DNG)")):
                save_jpeg(_small(img, args.long_edge), args.out / f"{stem}_{tag}.jpg", quality=90)
                cells.append(f'<figure><a href="{stem}_{tag}.jpg"><img loading="lazy" '
                             f'src="{stem}_{tag}.jpg"></a><figcaption>{cap}</figcaption></figure>')
            rows.append(f'<section><p>{html.escape(stem)}</p><div class="row">'
                        f'{"".join(cells)}</div></section>')
            print(stem, flush=True)
        title = f"{label[0]} ({len(names)})" if label else f"{Path(jpeg_dir).name} ({len(names)})"
        sections.append(f"<h2>{html.escape(title)}</h2>" + "".join(rows))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Raw look test</title>
<style>
body{{margin:0;padding:16px;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
h1{{font-size:20px}} h2{{margin-top:32px;border-bottom:1px solid #444;padding-bottom:4px}}
.row{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
figure{{margin:0}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#aaa;padding:2px 0 8px}} section p{{margin:16px 0 6px}}
@media (max-width:800px){{.row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Raw look against today's look</h1>
<p>Same photographs, references, training settings, normalisation and grain (same seed per
photograph). Only the input differs: the camera's JPEG, or the DNG developed with the
camera's own tone curve. Click a picture for the larger file.</p>
{"".join(sections)}
</body></html>"""
    (args.out / "index.html").write_text(page)
    print(f"wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
