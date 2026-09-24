"""The Parr pipeline: normalise, apply the LUT, add grain.

The order is fixed and matters. The LUT was fitted on normalised input, so
normalisation comes first; grain is a property of the developed film, so it
goes on last. The same ``Pipeline`` serves full-resolution captures, the
low-resolution live preview (``grain=False``) and batch reprocessing, which
is what guarantees the preview shows the grade the capture will get.

``info`` returns the gains that were applied, whether either gain hit its
clamp, the LUT's content hash and a hash of the normalisation parameters. The
capture app writes all of it to its log so a surprising frame can be explained
after the fact. Both hashes are needed: normalisation is a per-artifact
setting, so the same LUT under two different ``NormalizeParams`` produces two
different grades that ``lut_sha1`` alone cannot tell apart.

Exposure compensation (``ev``) has to survive normalisation. The camera
honours EV by exposing darker or brighter, but normalisation exists to bring
every frame to one exposure, so on its own it hands the LUT the same image at
-2 EV as at 0: the first shots with the viewfinder's EV buttons (2026-09-24)
came out 2 stops apart as originals and identical as graded files. So the
compensation is taken out of the frame (a ``2 ** -ev`` gain in linear light),
the frame is normalised as the plain auto exposure it would have been, and the
compensation is put back before the LUT. Undoing it first, rather than only
re-applying it afterwards, keeps the result exactly ``ev`` stops from the
uncompensated grade even where normalisation's clamps cannot fully restore a
very dark frame - re-applying alone darkened such a frame twice. Before the
LUT is where film takes exposure: the negative is exposed differently and
developed the same. The value is recorded as ``ev_comp`` so a regrade can
reproduce it.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .artifacts import Artifacts
from .color import linear_to_srgb, srgb_to_linear
from .grain import add_grain
from .normalize import normalize_u8


class Pipeline:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self._filter = artifacts.lut.to_pillow()
        self.normalize_sha1 = hashlib.sha1(
            json.dumps(artifacts.normalize.to_dict(), sort_keys=True).encode()
        ).hexdigest()

    def process(
        self,
        rgb_u8: np.ndarray,
        *,
        grain: bool = True,
        rng: np.random.Generator | None = None,
        ev: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError(
                f"process() expects an RGB uint8 array of shape (H, W, 3); "
                f"got dtype {rgb_u8.dtype} shape {rgb_u8.shape}"
            )
        ev = float(ev)
        if ev != 0.0:
            rgb_u8 = _ev_table(-ev)[rgb_u8]
        normalised, gains = normalize_u8(rgb_u8, self.artifacts.normalize)
        if ev != 0.0:
            normalised = _ev_table(ev)[normalised]
        graded = self.artifacts.lut.apply_pillow(normalised, self._filter)
        if grain and self.artifacts.grain.enabled:
            graded = add_grain(graded, self.artifacts.grain, rng)
        return graded, {
            "wb_gains": [round(float(g), 4) for g in gains.wb],
            "exposure_gain": round(float(gains.exposure), 4),
            "levels": (
                {k: (v if isinstance(v, str) else round(float(v), 4))
                 for k, v in gains.levels.items()}
                if gains.levels else None
            ),
            "clamped": dict(gains.clamped),
            "lut_sha1": self.artifacts.lut_sha1,
            "normalize_sha1": self.normalize_sha1,
            "ev_comp": ev,
        }


def _ev_table(ev: float) -> np.ndarray:
    """256-entry sRGB table for a ``2 ** ev`` exposure gain in linear light."""
    codes = np.arange(256, dtype=np.float32) / 255.0
    scaled = linear_to_srgb(np.clip(srgb_to_linear(codes) * 2.0 ** ev, 0.0, 1.0))
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
