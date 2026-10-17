"""Halation: the red-orange glow film grows round bright light, as a step of the look.

Light passes through the emulsion, reflects off the film base and comes back through
the red-sensitive layer, the deepest one, so a bright highlight grows a soft red glow.
It happens when the film is exposed, so the step runs after normalisation and before
the LUT, which is the look's roll-off: normalise, halation, LUT, grain.

The model (``"mix"``), chosen by eye from two galleries on 2026-10-17
(``scripts/experiments/halation_gallery.py``, ``docs/experiments/2026-10-17-halation.md``):

* **Global, linear, a mix.** A blurred copy of the whole frame is mixed into red and a
  little green, in linear light, with no threshold. A flat area equals its own blur, so
  it is unchanged and no colour shifts; the glow shows where the frame is uneven, most
  round bright light. A threshold model barely showed round a lamp and turned a sunlit
  wall cream.
* **Headroom.** The camera's JPEG has clipped its highlights, so a lamp is no brighter
  than white paper and a plain mix has nothing bright to spread (and pulls the red down
  inside the clipped area, turning a lamp blue-white). Each value is expanded with
  ``x / (1 - c x)`` before the mix and compressed back exactly with ``y / (1 + c y)``
  after; white counts as ``1 / (1 - c)``. At 10x a sunlit facade flooded its windows
  with orange; 2x (``c = 0.5``) is the chosen value.
* **The fall-off** is three Gaussians, ``sigma``, ``2 sigma`` and ``4 sigma``, weighted
  0.5, 0.3 and 0.2: about an exponential, softer than one Gaussian. ``sigma`` is in mm
  of a 36 mm frame, the grain's scale, so the glow is the same share of the picture at
  any size.
* **The blur runs on a reduced copy** (an area mean in expanded linear light, so a small
  lamp keeps its energy), by the whole factor that keeps four reduced pixels of sigma,
  and is read back with bilinear interpolation. The glow is soft, so this loses nothing
  visible, and it is what lets the Pi make it in tens of milliseconds.
* **The wider terms on smaller copies still.** On the Pi 4 (12 MP, a 579 x 434 reduced
  copy) the three blurs cost 28, 56 and 213 ms, most of the glow's time. The term with
  multiple m is therefore blurred on the reduced copy reduced by a further m (2 and 4),
  at the same width in the reduced copy's pixels, and read back bilinearly: about 20 ms
  for both. Such a term is smooth at that scale, so it is the same soft glow (within
  0.3 % of its peak of the direct blur on a lamp-like spot).

``prepare`` makes the glow for both paths, so the CPU and the GPU mix the same numbers:
the CPU here (``apply_cpu``), the GPU in the shader (``pifilm.gpu``). Changing the
fall-off, the reduction or the interpolation moves pixels: that is a new model name.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from ._cv2 import require_cv2
from .color import linear_to_srgb, srgb_to_linear

MODEL = "mix"
FALLOFF = ((1.0, 0.5), (2.0, 0.3), (4.0, 0.2))   # (sigma multiple, weight)
MIN_REDUCED_SIGMA = 4.0                          # reduced pixels of sigma kept


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"halation {name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"halation {name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class HalationParams:
    enabled: bool = False
    model: str = MODEL
    strength: float = 0.15
    sigma_mm: float = 0.25
    headroom: float = 0.5
    weights: tuple[float, float, float] = (1.0, 0.35, 0.0)
    frame_width_mm: float = 36.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"halation enabled must be true or false, got {self.enabled!r}")
        if self.model != MODEL:
            raise ValueError(f"halation model must be {MODEL!r}, got {self.model!r}")
        strength = _number("strength", self.strength)
        sigma = _number("sigma_mm", self.sigma_mm)
        headroom = _number("headroom", self.headroom)
        width = _number("frame_width_mm", self.frame_width_mm)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"halation strength must be 0 to 1, got {strength}")
        if not 0.0 < sigma <= 5.0:
            raise ValueError(f"halation sigma_mm must be above 0 and at most 5, got {sigma}")
        if not 0.0 <= headroom <= 0.9:
            raise ValueError(f"halation headroom must be 0 to 0.9, got {headroom}")
        if not width > 0.0:
            raise ValueError(f"halation frame_width_mm must be above 0, got {width}")
        weights = tuple(self.weights) if isinstance(self.weights, (list, tuple)) else ()
        if len(weights) != 3:
            raise ValueError(f"halation weights must be three numbers, got {self.weights!r}")
        weights = tuple(_number("weights", w) for w in weights)
        if not all(0.0 <= w <= 1.0 for w in weights):
            raise ValueError(f"halation weights must each be 0 to 1, got {list(weights)}")
        for name, value in (("strength", strength), ("sigma_mm", sigma),
                            ("headroom", headroom), ("frame_width_mm", width),
                            ("weights", weights)):
            object.__setattr__(self, name, value)

    @property
    def active(self) -> bool:
        """Whether the step changes a frame at all."""
        return self.enabled and self.strength > 0.0 and any(w > 0.0 for w in self.weights)

    @classmethod
    def from_dict(cls, d: dict) -> HalationParams:
        known = {"enabled", "model", "strength", "sigma_mm", "headroom", "weights",
                 "frame_width_mm"}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "model": self.model, "strength": self.strength,
                "sigma_mm": self.sigma_mm, "headroom": self.headroom,
                "weights": list(self.weights), "frame_width_mm": self.frame_width_mm}

    @property
    def sha1(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class HalationPass:
    """One photograph's halation, ready to mix: what both paths are given."""

    glow: np.ndarray        # (h, w, 3) float32, expanded linear, reduced and blurred
    headroom: float
    mix: np.ndarray = field(repr=False)   # (3,) float32: strength * weights


def sigma_px(shape: tuple[int, ...], params: HalationParams) -> float:
    return params.sigma_mm * max(shape[:2]) / params.frame_width_mm


def reduction(shape: tuple[int, ...], params: HalationParams) -> int:
    """The whole factor the glow is made at: four reduced pixels of sigma at least."""
    return max(1, int(sigma_px(shape, params) // MIN_REDUCED_SIGMA))


def expand_table(headroom: float) -> np.ndarray:
    """(256,) float32: an sRGB code to expanded linear light, ``lin / (1 - c lin)``."""
    lin = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float64)
    return (lin / (1.0 - headroom * lin)).astype(np.float32)


def _blur(cv2, small: np.ndarray, sigma: float, multiple: float) -> np.ndarray:
    """One term of the fall-off: ``small`` blurred by ``sigma * multiple`` reduced pixels.

    The term with multiple m is blurred on a copy reduced by a further m (an area mean)
    and read back bilinearly, the blur width given in that copy's own pixels (its actual
    scale, so a size m does not divide stays right). A frame too small for that is
    blurred directly."""
    height, width = small.shape[:2]
    m = int(multiple)
    if m <= 1 or width // m < 1 or height // m < 1:
        out = cv2.GaussianBlur(small, (0, 0), sigma * multiple, borderType=cv2.BORDER_REFLECT)
        return out.reshape(small.shape)
    size = (width // m, height // m)
    r = cv2.resize(small, size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], 3)
    blurred = cv2.GaussianBlur(r, (0, 0), sigma * multiple * size[0] / width,
                               borderType=cv2.BORDER_REFLECT).reshape(r.shape)
    out = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    return out.reshape(small.shape)


def prepare(normalised_u8: np.ndarray, params: HalationParams) -> HalationPass:
    """The glow of the frame the LUT is about to be given."""
    cv2 = require_cv2()
    height, width = normalised_u8.shape[:2]
    table = np.ascontiguousarray(
        np.repeat(expand_table(params.headroom)[:, None], 3, axis=1)).reshape(256, 1, 3)
    x = cv2.LUT(np.ascontiguousarray(normalised_u8[..., :3]), table)
    k = reduction(normalised_u8.shape, params)
    size = (max(1, width // k), max(1, height // k))
    small = cv2.resize(x, size, interpolation=cv2.INTER_AREA) if k > 1 else x
    small = np.ascontiguousarray(small, np.float32).reshape(size[1], size[0], 3)
    sigma = sigma_px(normalised_u8.shape, params) / k
    glow = np.zeros_like(small)
    for multiple, weight in FALLOFF:
        glow += np.float32(weight) * _blur(cv2, small, sigma, multiple)
    mix = (np.float32(params.strength) * np.asarray(params.weights, np.float32))
    return HalationPass(glow=glow, headroom=params.headroom, mix=mix)


def upsample(glow: np.ndarray, height: int, width: int) -> np.ndarray:
    """The glow at full size: bilinear, half-pixel centres, edges clamped."""
    cv2 = require_cv2()
    if glow.shape[:2] == (height, width):
        return glow
    up = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
    return up.reshape(height, width, glow.shape[2])


def apply_cpu(normalised_u8: np.ndarray, hp: HalationPass) -> np.ndarray:
    """Expand, mix with the glow, compress, back to bytes. A channel whose weight is
    0 is returned as it was."""
    height, width = normalised_u8.shape[:2]
    out = np.array(normalised_u8[..., :3], copy=True)
    table = expand_table(hp.headroom)
    up = upsample(hp.glow, height, width)
    c = np.float32(hp.headroom)
    for channel in range(3):
        weight = np.float32(hp.mix[channel])
        if weight == 0.0:
            continue
        x = table[normalised_u8[..., channel]]
        x += weight * (up[..., channel] - x)
        lin = x / (np.float32(1.0) + c * x)
        out[..., channel] = np.rint(linear_to_srgb(lin) * np.float32(255.0)).astype(np.uint8)
    return out
