"""Fine film grain, added after the LUT.

This project's low-ISO color-negative target uses subtle grain. The amount
is an aesthetic choice, not a calibrated emulation of a particular stock.
The model is deliberately simple:

* Noise goes on **luminance only**. Film grain is a density variation of the
  dye layers seen together; chroma noise reads as a digital sensor artefact.
* The noise field is Gaussian-blurred by ``blur_sigma`` and renormalised to
  unit variance. Pixel-independent noise looks like high-ISO noise;
  slightly correlated noise looks like grain clumps.
* An envelope ``4Y(1 - Y)`` scales it: zero at black and white, one at
  mid-grey, because real grain is least visible in deep shadow and in fully
  exposed highlights.

``strength`` is the noise standard deviation in luminance units at the
mid-grey peak; 0.025 is about six 8-bit levels.

A second model, ``two_scale``, is measured rather than chosen: Ilford HP5 Plus
400 from lossless 35 mm scans (``docs/experiments/2026-10-01-hp5-grain.md``).
A single blur could not follow its spectrum, which falls off like fine clumps
(4.3 um on the film, 76% of the variance) plus coarse ones (11.2 um). Sizes are
millimetres of a 36 mm frame, so the grain keeps its look at any resolution.
Its envelope is the measured strength against display luminance, strongest in
the upper mid-tones rather than the symmetric hump above, pinned to zero at
black and white. ``strength`` is again the sd at mid-grey. Artifacts using it
are written as params version 3 (``pifilm.artifacts``), because an older
build would read these fields as the Gaussian model at six times its strength.

Reproducibility: ``add_grain`` takes an explicit generator. The capture app
draws a seed per shot and records it, so a graded file can be regenerated
from its original (spec 7.2).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from ._cv2 import require_cv2

cv2 = require_cv2()


MODELS = ("gaussian", "two_scale")
GAUSSIAN_FIELDS = ("strength", "blur_sigma", "enabled")
# HP5 Plus 400, relative sd by display luminance (1.0 at mid-grey), measured
# 2026-10-01; the ends are pinned to zero so black and white stay untouched.
HP5_ENVELOPE_LUMA = (0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
HP5_ENVELOPE_REL = (0.0, 0.25, 0.35, 0.56, 0.63, 1.02, 0.98, 1.14, 1.04, 0.62, 0.36, 0.0)


@dataclass
class GrainParams:
    strength: float = 0.004
    blur_sigma: float = 0.7
    enabled: bool = True
    model: str = "gaussian"
    fine_sigma_mm: float = 0.0043
    coarse_sigma_mm: float = 0.0112
    fine_share: float = 0.76
    frame_width_mm: float = 36.0
    envelope_luma: tuple[float, ...] = HP5_ENVELOPE_LUMA
    envelope_rel: tuple[float, ...] = HP5_ENVELOPE_REL

    def __post_init__(self) -> None:
        self.envelope_luma = tuple(float(v) for v in self.envelope_luma)
        self.envelope_rel = tuple(float(v) for v in self.envelope_rel)
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        for name in ("strength", "blur_sigma", "fine_sigma_mm", "coarse_sigma_mm"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
        if not 0.0 <= self.fine_share <= 1.0:
            raise ValueError(f"fine_share must be in [0, 1], got {self.fine_share}")
        if not self.frame_width_mm > 0:
            raise ValueError(f"frame_width_mm must be positive, got {self.frame_width_mm}")
        lum = self.envelope_luma
        if len(lum) < 2 or any(b <= a for a, b in zip(lum, lum[1:], strict=False)) or not (
            0.0 <= lum[0] and lum[-1] <= 1.0
        ):
            raise ValueError("envelope_luma must be at least two increasing values in [0, 1]")
        if len(self.envelope_rel) != len(lum) or min(self.envelope_rel) < 0:
            raise ValueError("envelope_rel must match envelope_luma and not be negative")

    @classmethod
    def from_dict(cls, d: dict) -> GrainParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.model == "gaussian":
            # Exactly the pre-two_scale keys, so existing artifacts are unchanged.
            return {k: d[k] for k in GAUSSIAN_FIELDS}
        d["envelope_luma"] = list(self.envelope_luma)
        d["envelope_rel"] = list(self.envelope_rel)
        return d


def add_grain(
    rgb_u8: np.ndarray, params: GrainParams, rng: np.random.Generator | None = None
) -> np.ndarray:
    if not params.enabled or params.strength <= 0:
        return rgb_u8.copy()
    rng = rng if rng is not None else np.random.default_rng()

    ycc = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma = ycc[..., 0] / 255.0

    if params.model == "two_scale":
        noise = _two_scale_noise(luma.shape, params, rng)
        envelope = np.interp(luma, params.envelope_luma, params.envelope_rel).astype(np.float32)
        ycc[..., 0] = np.clip(luma + params.strength * envelope * noise, 0.0, 1.0) * 255.0
        return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8),
                            cv2.COLOR_YCrCb2RGB)

    noise = rng.standard_normal(luma.shape, dtype=np.float32)
    if params.blur_sigma > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), params.blur_sigma)
        noise /= max(float(noise.std()), 1e-6)

    envelope = 4.0 * luma * (1.0 - luma)
    ycc[..., 0] = np.clip(luma + params.strength * envelope * noise, 0.0, 1.0) * 255.0
    return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def _two_scale_noise(shape: tuple[int, int], params: GrainParams,
                     rng: np.random.Generator) -> np.ndarray:
    """Unit-variance mix of fine and coarse Gaussian clumps sized by the frame width."""
    px_per_mm = max(shape) / params.frame_width_mm
    total = np.zeros(shape, np.float32)
    for sigma_mm, share in ((params.fine_sigma_mm, params.fine_share),
                            (params.coarse_sigma_mm, 1.0 - params.fine_share)):
        if share <= 0:
            continue
        field = rng.standard_normal(shape, dtype=np.float32)
        sigma_px = sigma_mm * px_per_mm
        if sigma_px > 0:
            field = cv2.GaussianBlur(field, (0, 0), sigma_px)
        total += math.sqrt(share) * field / max(float(field.std()), 1e-6)
    return total / max(float(total.std()), 1e-6)
