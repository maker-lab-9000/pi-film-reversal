"""Measure film grain from scans, and synthesise grain with the measured statistics.

Offline experiment: it never changes ``pifilm/grain.py`` or any artifact. The
production grain is three hand-set numbers (strength, one blur, a fixed
``4Y(1-Y)`` envelope); this module asks what a real stock's grain looks like
in the same terms, so a later grain model can be fitted instead of guessed.

Grain is read from *flat* patches only. A scene's own texture (grass, bark,
brick) has the same spectral footprint as grain, so a patch is rejected when
its low-pass content varies (``structure``): only fog, sky, walls and
out-of-focus areas are measured. Within a patch, the high-pass residual
(patch minus a Gaussian blur of ``size / 16``) is the grain.

Two statistics come out:

* **Amplitude against brightness.** Residual standard deviation, binned by
  the patch's mean display luminance. This replaces the fixed envelope.
* **Clump size.** The residual's radially averaged power spectrum, fitted as
  white noise through a Gaussian filter, ``A exp(-4 pi^2 sigma^2 f^2) + C``.
  ``C`` absorbs the scanner's and JPEG's flat noise floor, which would
  otherwise pull ``sigma`` down. Sigma is measured in scan pixels and
  converted to millimetres on the film with the frame width (35 mm: 36 mm),
  which is what makes grain from a 2450 dpi and a 4000 dpi scan comparable
  and lets it be redrawn at the camera's resolution.

Limits: a scan only shows grain its optics resolve, so low-resolution scans
overstate clump size and understate amplitude; that is why the measurement is
reported per source.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._cv2 import require_cv2
from ..imageio import load_rgb

cv2 = require_cv2()

FILM_WIDTH_MM = 36.0  # 35 mm frame, long side
CAMERA_LONG_SIDE_PX = 4056  # IMX477 still


@dataclass(frozen=True)
class Patch:
    y0: int
    x0: int
    mean: float
    sd: float
    structure: float
    residual: np.ndarray


def display_luma(rgb_u8: np.ndarray) -> np.ndarray:
    """Rec.601 luma of sRGB-encoded pixels, 0..1: what grain.py perturbs."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def flat_patches(y: np.ndarray, size: int = 128, *, structure_ratio: float = 0.35,
                 lo: float = 0.05, hi: float = 0.95) -> list[Patch]:
    """Non-overlapping ``size`` tiles whose low-pass content is flat enough to read grain.

    A tile is kept when the standard deviation of its low-pass image is below
    ``structure_ratio`` times its residual's: grain alone leaves only a small
    low-pass ripple, while any scene detail or gradient dominates it.
    """
    sigma = size / 16
    out = []
    h, w = y.shape
    for y0 in range(0, h - size + 1, size):
        for x0 in range(0, w - size + 1, size):
            tile = y[y0:y0 + size, x0:x0 + size].astype(np.float32)
            mean = float(tile.mean())
            if not lo < mean < hi or np.mean((tile <= 0.002) | (tile >= 0.998)) > 0.01:
                continue
            low = cv2.GaussianBlur(tile, (0, 0), sigma)
            residual = tile - low
            sd = float(residual.std())
            structure = float(low.std())
            if sd <= 1e-4 or structure > structure_ratio * sd:
                continue
            out.append(Patch(y0, x0, mean, sd, structure, residual))
    return out


def radial_psd(residuals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean radially averaged power spectrum of equal-size square residuals."""
    n = residuals[0].shape[0]
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    radius = np.hypot(fy, fx)
    edges = np.linspace(0, 0.5, n // 2 + 1)
    idx = np.digitize(radius.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    acc = np.zeros(len(edges) - 1)
    for r in residuals:
        power = np.abs(np.fft.fft2((r - r.mean()) * win)) ** 2
        acc += np.bincount(idx[valid], power.ravel()[valid], minlength=len(acc))[: len(acc)]
    counts = np.bincount(idx[valid], minlength=len(acc))[: len(acc)]
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, acc / np.maximum(counts, 1) / len(residuals)


def fit_clump_sigma(f: np.ndarray, p: np.ndarray, f_lo: float = 0.05,
                    f_hi: float = 0.45) -> float:
    """Gaussian clump sigma in pixels from ``p = A exp(-4 pi^2 s^2 f^2) + C``."""
    from scipy.optimize import curve_fit

    sel = (f >= f_lo) & (f <= f_hi) & np.isfinite(p) & (p > 0)
    f, p = f[sel], p[sel]
    scale = p.max()
    p = p / scale

    def model(x, a, s, c):
        return a * np.exp(-4 * math.pi**2 * s**2 * x**2) + c

    # Seed sigma from the half-power frequency of the floor-free curve.
    half = f[np.argmax(p < (p[0] + p[-1]) / 2)] if np.any(p < (p[0] + p[-1]) / 2) else 0.2
    s0 = math.sqrt(math.log(2)) / (2 * math.pi * max(half, 1e-3))
    a0 = p[0] * math.exp(4 * math.pi**2 * s0**2 * f[0] ** 2)
    (a, s, c), _ = curve_fit(model, f, p, p0=(a0, s0, max(p[-1], 0.0)),
                             bounds=([0, 0.05, 0], [np.inf, 50, 1]), maxfev=20000)
    return float(abs(s))


def fit_clump_mix(f: np.ndarray, p: np.ndarray, f_lo: float = 0.05,
                  f_hi: float = 0.45) -> tuple[float, float, float]:
    """Fine and coarse clump sigmas (pixels) and the fine share of grain variance.

    HP5's measured spectrum falls off gradually, like fine plus coarse clumps;
    one Gaussian cannot follow it and pushes the fine part into the noise floor.
    A unit-variance field blurred by sigma has spectrum ``4 pi s^2 exp(-4 pi^2 s^2 f^2)``,
    so each fitted amplitude divided by its ``s^2`` is that component's variance.
    """
    from scipy.optimize import curve_fit

    sel = (f >= f_lo) & (f <= f_hi) & np.isfinite(p) & (p > 0)
    f, p = f[sel], p[sel] / p[sel].max()

    def g(x, s):
        return np.exp(-4 * math.pi**2 * s**2 * x**2)

    def model(x, a1, s1, a2, s2, c):
        return a1 * g(x, s1) + a2 * g(x, s2) + c

    (a1, s1, a2, s2, _c), _ = curve_fit(
        model, f, p, p0=(0.5, 0.5, 1.0, 2.5, 0.0),
        bounds=([0, 0.1, 0, 0.1, 0], [np.inf, 20, np.inf, 20, 0.05]), maxfev=50000)
    (a1, s1), (a2, s2) = sorted([(a1, abs(s1)), (a2, abs(s2))], key=lambda t: t[1])
    v1, v2 = a1 / s1**2, a2 / s2**2
    return float(s1), float(s2), float(v1 / (v1 + v2))


def amplitude_curve(means: np.ndarray, sds: np.ndarray, bins: np.ndarray,
                    min_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Median residual sd per brightness bin; NaN where too few patches."""
    centres = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(means, bins) - 1
    curve = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        sel = sds[idx == i]
        if len(sel) >= min_count:
            curve[i] = float(np.median(sel))
    return centres, curve


def synth_grain(lum: np.ndarray, sigma_px: float | tuple[float, ...], curve_x: np.ndarray,
                curve_sd: np.ndarray, rng: np.random.Generator,
                weights: tuple[float, ...] | None = None) -> np.ndarray:
    """Additive luminance grain: unit noise per clump scale, mixed by variance share,
    then scaled by the brightness-dependent sd."""
    sigmas = (sigma_px,) if np.isscalar(sigma_px) else tuple(sigma_px)
    weights = weights or (1.0,) * len(sigmas)
    total = np.zeros(lum.shape, np.float32)
    for s, w in zip(sigmas, weights, strict=True):
        noise = rng.standard_normal(lum.shape).astype(np.float32)
        if s > 0:
            noise = cv2.GaussianBlur(noise, (0, 0), s)
        total += math.sqrt(w / sum(weights)) * noise / max(float(noise.std()), 1e-6)
    total /= max(float(total.std()), 1e-6)
    return total * np.interp(lum, curve_x, curve_sd).astype(np.float32)


def measure_file(path: Path, size: int) -> dict:
    rgb, _ = load_rgb(path)
    y = display_luma(rgb)
    h, w = y.shape
    px_per_mm = max(h, w) / FILM_WIDTH_MM
    patches = flat_patches(y, size=size)
    rec = {"file": path.name, "width": w, "height": h, "px_per_mm": round(px_per_mm, 2),
           "patches": len(patches),
           "means": [round(p.mean, 4) for p in patches], "sds": [round(p.sd, 5) for p in patches]}
    if len(patches) >= 4:
        f, psd = radial_psd([p.residual for p in patches])
        sigma = fit_clump_sigma(f, psd)
        rec.update(sigma_px=round(sigma, 3), sigma_mm=round(sigma / px_per_mm, 5))
        s1, s2, w1 = fit_clump_mix(f, psd)
        rec.update(mix_sigma_mm=[round(s1 / px_per_mm, 5), round(s2 / px_per_mm, 5)],
                   mix_fine_share=round(w1, 3))
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure film grain from flat patches of scans.")
    parser.add_argument("scans", type=Path, help="folder with manifest.json (review: kept)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args(argv)
    manifest = json.loads((args.scans / "manifest.json").read_text())
    files = [f for f in manifest["files"] if f.get("review", "kept") == "kept"]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        rec = measure_file(args.scans / f["file"], args.size)
        rec["artist"] = f.get("artist", "")
        results.append(rec)
        print(f"{rec['file'][:48]:48} patches {rec['patches']:4} "
              f"sigma {rec.get('sigma_mm', float('nan')) * 1000:6.1f} um", flush=True)
    means = np.array([m for r in results for m in r["means"]])
    sds = np.array([s for r in results for s in r["sds"]])
    centres, curve = amplitude_curve(means, sds, np.linspace(0, 1, 11))
    sig = np.array([r["sigma_mm"] for r in results if "sigma_mm" in r])
    summary = {
        "files": len(results), "patches": int(len(means)),
        "amplitude_curve": {"luma": centres.round(3).tolist(),
                            "sd": [None if np.isnan(v) else round(float(v), 5) for v in curve]},
        "sigma_mm_median": float(np.median(sig)) if len(sig) else None,
        "sigma_mm_iqr": [float(np.percentile(sig, 25)), float(np.percentile(sig, 75))]
        if len(sig) else None,
        "camera_px_per_mm": CAMERA_LONG_SIDE_PX / FILM_WIDTH_MM,
    }
    (args.out / "grain-measurements.json").write_text(
        json.dumps({"summary": summary, "files": results}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
