"""Two exposures on one frame of colour negative film.

Film records exposure H (light x time) at each point, and a second exposure adds
to the first: H = H1 + H2. Darkness adds almost nothing, so where one frame is dark
the other shows through at full strength (the classic silhouette filled with
trees); where both are bright, detail separation is lost. Colours add as light,
not as paint: red + green gives yellow. The film's characteristic curve then acts
once, on the sum.

So the composite is made here, before any grading, and ``Pipeline.process`` runs
once on the result. Grading each frame first would apply the LUT's tone curve twice
and add values after the curve, which is a flat 50/50 blend, not added light; it
would also lay grain down twice and normalise each frame and then the mix, which
``pifilm.normalize`` does not support.

Each frame is taken one stop down (x0.5 in linear light) before they are added.
That is the usual film starting point for two equal exposures: where the frames
match, the total is one normal exposure. The camera auto-exposes every frame to a
full exposure, so a plain sum would sit a stop over and clip before the LUT's
shoulder could act; the mean never exceeds 1.0. Each frame goes in as shot, its EV
compensation included, so a frame shot darker contributes less light. The caller
grades the composite with ``ev=0``: normalisation then sets the pair's overall
exposure, and the ratio between the two exposures is what survives.
"""

from __future__ import annotations

import numpy as np

from .color import linear_to_srgb, srgb_to_linear

COMPOSITE_METHOD = "linear_mean"


def composite(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Expose two sRGB uint8 frames onto one "negative"; return sRGB uint8."""
    for name, frame in (("first", first), ("second", second)):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"composite() expects RGB uint8 frames of shape (H, W, 3); "
                f"{name} is dtype {frame.dtype} shape {frame.shape}"
            )
    if first.shape != second.shape:
        raise ValueError(
            f"double exposure frames differ in size: {first.shape} vs {second.shape}"
        )
    linear = 0.5 * (srgb_to_linear(first / 255.0) + srgb_to_linear(second / 255.0))
    return np.clip(np.round(linear_to_srgb(linear) * 255.0), 0, 255).astype(np.uint8)
