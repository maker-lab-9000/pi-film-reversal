"""Several short frames of one scene merged into one photograph.

A multi-frame photograph (``Picamera2Camera.read_burst``) is a burst of frames
each shorter than the one long exposure it replaces, so each is sharper under
camera shake and noisier. This module aligns them to one frame, the reference,
and averages them in linear light, which brings the noise down by about
1/sqrt(n) without the blur of the long exposure. It is pure: arrays in, an
array and an info dict out, no camera. ``merge`` runs once on the Pi for the
photograph and again in ``pifilm-process`` to reproduce it, so it is
deterministic for the same inputs (no random state; the same OpenCV and NumPy
operations in the same order). Another machine's OpenCV may round the last bits
of ECC or the warp differently, so a Mac reproduction is close, not promised
byte for byte.

Why these choices
-----------------
* **The reference is the caller's.** The caller passes the sharpest frame (the
  Pi uses the highest ``FocusFoM``); the result has that frame's geometry and,
  where anything moved, that frame's content.
* **ECC with a Euclidean motion** (``cv2.findTransformECC``,
  ``MOTION_EUCLIDEAN``). Hand shake over a fraction of a second is a small
  shift plus a small roll about the lens axis; at these focal lengths the
  perspective part is far below a pixel. A Euclidean model has three
  parameters, so it cannot overfit to a moving subject the way a homography
  can, and ECC (enhanced correlation coefficient) is invariant to the small
  brightness differences between frames and refines to a fraction of a pixel.
* **Aligned on reduced luma.** ECC iterates over every pixel it is given; at
  12 MP that is seconds on a Pi 4. Luma reduced by a whole factor to about
  ``ALIGN_LONG_SIDE`` pixels on the long side keeps enough texture for a
  sub-pixel fit (the error scales up by the factor, and stays well under one
  full-size pixel in tests) at a few percent of the cost, and the reduction's
  area average removes most of the sensor noise ECC would otherwise chase.
  The reduction is by a whole factor over a crop to a multiple of it, so the
  scale between the two grids is exact and the fit maps to full size with
  pixel centres accounted for (``_full_size_warp``).
* **A frame that does not fit is dropped, not forced.** ECC that raises, ends
  with a correlation under ``MIN_CORRELATION`` (no convergence to the scene:
  the alignment would be noise), or finds a shift over ``MAX_SHIFT_FRACTION``
  of the width or a roll over ``MAX_ANGLE_DEG`` (no hand shake within a burst
  is that large; it is a wrong fit or a different picture) leaves the frame
  out. A photograph from fewer frames is noisier; a misaligned one is broken.
* **Warped once at full size** with ``cv2.warpAffine`` (bilinear, reflected
  border so no black edges enter the mean), kept as uint8: five aligned 12 MP
  frames are 185 MB, where float32 copies would be four times that.
* **Ghost weight.** Anything that moved between frames (a person, a car, a
  leaf) would appear several times, faintly. Each pixel of each other frame is
  weighted ``exp(-(d / GHOST_SCALE) ** 2)``, ``d`` its linear-light luma
  difference from the reference; the reference has weight 1. Where the frames
  agree up to noise the weights are near 1 and the mean averages the noise;
  where something moved the difference is large, the weight near 0, and the
  pixel comes from the reference alone (noisier there, but whole).
* **Linear light**, because the frames are light added up: averaging sRGB codes
  would darken edges and mid-tones. The sRGB <-> linear tables are the ones
  the double exposure uses (``color.SRGB8_LINEAR``, ``color.LINEAR16_SRGB8``).
* **Bands of rows** of about ``BAND_PIXELS`` (as ``halation.apply_cpu``): the
  float work for six 12 MP frames is a few MB per band, not several full-size
  float copies. Every pixel's arithmetic is its own, so bands change nothing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from ._cv2 import require_cv2
from .color import LINEAR16_SRGB8, LINEAR_LEVELS, LUMA_709, SRGB8_LINEAR

METHOD = "ecc-euclidean-ghost-weighted-mean"

GHOST_SCALE = 0.05
"""Linear-light luma difference at which another frame's weight falls to 1/e.

Two short, gained-up frames of a still scene differ by noise of the order of a
few hundredths of linear luma in the mid-tones (an estimate, to be checked on
the Pi's frames), so a pixel that only differs by noise keeps most of its
weight (0.7 at a difference of 0.03) and the mean still averages it. A moving
subject against its background differs by a tenth or more, where the weight
is under 0.02: the reference alone shows it. The difference is in luma only,
so a change of colour at the same brightness is not caught.
"""

ALIGN_LONG_SIDE = 1000
"""About this many pixels on the long side of the luma ECC aligns."""

ECC_ITERATIONS = 100
ECC_EPSILON = 1e-5
ECC_GAUSS_SIZE = 5

MIN_CORRELATION = 0.5
"""An ECC fit ending below this correlation has not converged on the scene."""

MAX_SHIFT_FRACTION = 0.05
MAX_ANGLE_DEG = 2.0

BAND_PIXELS = 1 << 19


def _check(frames: Sequence[np.ndarray], reference: int) -> None:
    if not frames:
        raise ValueError("merge() needs at least one frame")
    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"merge() expects RGB uint8 frames of shape (H, W, 3); "
                f"frame {i} is dtype {frame.dtype} shape {frame.shape}"
            )
        if frame.shape != frames[0].shape:
            raise ValueError(
                f"frames differ in size: frame 0 is {frames[0].shape}, frame {i} {frame.shape}"
            )
    if isinstance(reference, bool) or not 0 <= reference < len(frames):
        raise ValueError(f"reference {reference!r} is not a frame index (0 to {len(frames) - 1})")


def _factor(height: int, width: int) -> int:
    return max(1, round(max(height, width) / ALIGN_LONG_SIDE))


def _reduced_luma(frame: np.ndarray, factor: int) -> np.ndarray:
    cv2 = require_cv2()
    luma = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = luma.shape
    small_h, small_w = height // factor, width // factor
    crop = luma[: small_h * factor, : small_w * factor]
    if factor > 1:
        crop = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32)


def _full_size_warp(small: np.ndarray, factor: int) -> np.ndarray:
    """Map a 2x3 warp on the reduced grid to the full grid.

    A reduced pixel ``u`` covers full pixels whose centre is ``x = k u + c``,
    ``c = (k - 1) / 2``. Substituting, ``x' = R x + k t + (I - R) c``."""
    rot = small[:, :2].astype(np.float64)
    c = np.full(2, (factor - 1) / 2.0)
    full = np.empty((2, 3), dtype=np.float64)
    full[:, :2] = rot
    full[:, 2] = factor * small[:, 2] + c - rot @ c
    return full


def _align(reference: np.ndarray, frame: np.ndarray, factor: int) -> np.ndarray | None:
    """The full-size warp mapping reference pixels into ``frame``, or None (ECC failed)."""
    cv2 = require_cv2()
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ECC_ITERATIONS, ECC_EPSILON)
    try:
        rho, warp = cv2.findTransformECC(
            reference, frame, warp, cv2.MOTION_EUCLIDEAN, criteria, None, ECC_GAUSS_SIZE
        )
    except cv2.error:
        return None
    if not math.isfinite(rho) or rho < MIN_CORRELATION or not np.all(np.isfinite(warp)):
        return None
    return _full_size_warp(warp, factor)


def _motion(warp: np.ndarray, height: int, width: int) -> tuple[float, float, float]:
    """(dx, dy, angle in degrees): where the frame's content at the image centre
    sits relative to the reference, and its roll. The centre, not the corner, so a
    roll does not read as a shift."""
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    dx, dy = warp[:, :2] @ centre + warp[:, 2] - centre
    return float(dx), float(dy), math.degrees(math.atan2(warp[1, 0], warp[0, 0]))


def _entry(motion: tuple[float, float, float] | None, used: bool) -> dict:
    if motion is None:
        return {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "used": False}
    dx, dy, angle = motion
    return {
        "dx": round(dx, 3) + 0.0,  # + 0.0: no -0.0 in the record
        "dy": round(dy, 3) + 0.0,
        "angle_deg": round(angle, 4) + 0.0,
        "used": bool(used),
    }


def _within_limits(motion: tuple[float, float, float], width: int) -> bool:
    dx, dy, angle = motion
    return math.hypot(dx, dy) <= MAX_SHIFT_FRACTION * width and abs(angle) <= MAX_ANGLE_DEG


def merge_aligned(
    reference: np.ndarray, others: Sequence[np.ndarray], band_pixels: int = BAND_PIXELS
) -> np.ndarray:
    """Ghost-weighted mean in linear light of ``reference`` and already aligned
    ``others`` (all RGB uint8, one shape); returns sRGB uint8.

    The per-band arithmetic uses OpenCV where NumPy is several times slower on
    the Pi's sizes (decoding through a 256-entry table, the luma, spreading a
    weight over three channels and accumulating the product)."""
    if not others:
        return reference.copy()
    cv2 = require_cv2()
    height, width = reference.shape[:2]
    out = np.empty_like(reference)
    rows = max(1, band_pixels // max(width, 1))
    luma = LUMA_709.reshape(1, 3)
    inv_s = np.float32(1.0 / GHOST_SCALE)
    scale = np.float32(LINEAR_LEVELS - 1)
    for top in range(0, height, rows):
        band = slice(top, top + rows)
        total = cv2.LUT(reference[band], SRGB8_LINEAR)
        ref_luma = cv2.transform(total, luma)
        weights = np.ones(ref_luma.shape, dtype=np.float32)
        for other in others:
            lin = cv2.LUT(other[band], SRGB8_LINEAR)
            w = cv2.transform(lin, luma)
            w -= ref_luma                          # d
            w *= inv_s
            np.square(w, out=w)
            np.negative(w, out=w)
            np.exp(w, out=w)                       # exp(-(d / s)^2)
            weights += w
            cv2.accumulateProduct(lin, cv2.cvtColor(w, cv2.COLOR_GRAY2RGB), total)
        total = cv2.divide(total, cv2.cvtColor(weights, cv2.COLOR_GRAY2RGB))
        total *= scale
        total += np.float32(0.5)
        np.clip(total, 0.0, scale, out=total)
        out[band] = LINEAR16_SRGB8[total.astype(np.uint16)]
    return out


def merge(frames: Sequence[np.ndarray], *, reference: int) -> tuple[np.ndarray, dict]:
    """Align every frame to ``frames[reference]`` and merge them; see the module
    docstring. Returns the sRGB uint8 photograph and an info dict (JSON-ready):
    ``reference``, ``method``, ``frames`` (per input frame ``dx``, ``dy``: the
    shift at the image centre in full-size pixels, ``angle_deg``: the roll, ``used``;
    zeros for a frame ECC could not fit), ``align_ms`` and ``merge_ms``.
    One frame, or no other frame used, returns the reference frame itself."""
    _check(frames, reference)
    ref_frame = frames[reference]
    height, width = ref_frame.shape[:2]
    entries: list[dict] = [_entry(None, False) for _ in frames]
    entries[reference] = {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "used": True}

    start = time.perf_counter()
    aligned: list[np.ndarray] = []
    if len(frames) > 1:
        cv2 = require_cv2()
        factor = _factor(height, width)
        ref_small = _reduced_luma(ref_frame, factor)
        for i, frame in enumerate(frames):
            if i == reference:
                continue
            warp = _align(ref_small, _reduced_luma(frame, factor), factor)
            motion = None if warp is None else _motion(warp, height, width)
            used = motion is not None and _within_limits(motion, width)
            entries[i] = _entry(motion, used)
            if used:
                aligned.append(
                    cv2.warpAffine(
                        frame,
                        warp,
                        (width, height),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                )
    align_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    result = merge_aligned(ref_frame, aligned) if aligned else ref_frame
    merge_ms = (time.perf_counter() - start) * 1000.0

    info = {
        "reference": int(reference),
        "method": METHOD,
        "frames": entries,
        "align_ms": round(align_ms, 1),
        "merge_ms": round(merge_ms, 1),
    }
    return result, info
