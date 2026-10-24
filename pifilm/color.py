"""Colour-space conversions shared by the trainer and the Pi runtime.

Why these spaces
----------------
* **sRGB** is what the camera delivers and what JPEGs store. Its transfer
  curve is roughly gamma 2.2, so arithmetic on sRGB values does not model
  light. Every function here expects sRGB in ``[0, 1]``.
* **Linear RGB** is sRGB with the transfer curve removed. White balance and
  exposure are multiplications of light, so ``normalize.py`` works here.
* **Oklab** (Björn Ottosson, 2020) is a perceptual space: Euclidean distance
  approximates perceived difference and hue angles are far more uniform
  than in CIELAB, which bends visibly in the blues. The trainer computes
  hue histograms, distribution transport and metrics in Oklab so that
  "match the distribution" means "match what the eye sees".
* **Oklch** is Oklab in polar form: lightness L, chroma C, hue h (radians,
  from ``arctan2``, so in ``[-pi, pi]``).

All functions accept arrays of shape ``(..., 3)`` and return ``float32``.

Two lookup tables at the end serve the byte-level runtimes that add light
(``double.py``, ``stack.py``): ``SRGB8_LINEAR`` decodes a uint8 code exactly
(there are only 256), ``LINEAR16_SRGB8`` encodes linear light quantised to
``LINEAR_LEVELS`` levels, well under one output code.
"""

from __future__ import annotations

import numpy as np

# Oklab matrices from https://bottosson.github.io/posts/oklab/ (linear sRGB -> LMS -> Lab).
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)

# Rec. 709 / sRGB luminance weights for linear RGB.
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Remove the sRGB transfer curve. Input is clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer curve. Input is clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055
    ).astype(np.float32)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = np.cbrt(rgb @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    lms = (lab @ _M2_INV.T) ** 3
    return (lms @ _M1_INV.T).astype(np.float32)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def oklab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Polar Oklab: (L, chroma, hue in radians)."""
    lab = np.asarray(lab, dtype=np.float32)
    a, b = lab[..., 1], lab[..., 2]
    return np.stack([lab[..., 0], np.hypot(a, b), np.arctan2(b, a)], axis=-1).astype(np.float32)


def lch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float32)
    lum, chroma, hue = lch[..., 0], lch[..., 1], lch[..., 2]
    return np.stack([lum, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1).astype(np.float32)


def luminance(rgb_linear: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of linear RGB."""
    return np.asarray(rgb_linear, dtype=np.float32) @ LUMA_709


# uint8 sRGB code -> linear light, float32.
SRGB8_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
LINEAR_LEVELS = 65536
# Linear level index (linear light scaled to [0, LINEAR_LEVELS - 1]) -> sRGB uint8 code.
LINEAR16_SRGB8 = np.clip(
    np.round(
        linear_to_srgb(np.arange(LINEAR_LEVELS, dtype=np.float32) / (LINEAR_LEVELS - 1))
        * 255.0
    ),
    0,
    255,
).astype(np.uint8)
