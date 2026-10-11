# pifilm/display/whitebalance.py
"""Manual white-balance steps for the viewfinder's WB pill. Pure.

The series is a short list of colour temperatures in kelvin, with ``None`` meaning
``A``: automatic white balance. The pill has room for one button, so a tap moves one
step up and the step after the last is back to ``A``. A higher temperature tells the
camera the light is bluer, so it renders warmer: 6500 or 7500 corrects a cold cast.

The values stay inside what the IMX477's calibration covers. Measured on the Pi 4
on 2026-10-11, a request below 2360 K is clamped to 2360 K and 9000 K is accepted
as asked. A value between steps moves to the next step up, never skipping one.

The labels are ASCII only: the viewfinder's default font has no glyphs beyond it.
"""

from __future__ import annotations

import math

from . import stepscale

WB_STEPS_K: tuple[int, ...] = (3200, 4000, 5000, 5600, 6500, 7500)


DEFAULT_START_K = 5600


def warmer(current_k: int | None, metered_k: float | None) -> int:
    """Higher kelvin: the camera renders warmer. From auto, the step nearest the
    metered temperature (log2), else 5600; stays on the top step."""
    return stepscale.up(WB_STEPS_K, current_k, metered_k, DEFAULT_START_K)


def cooler(current_k: int | None, metered_k: float | None) -> int | None:
    """Lower kelvin; one step below the lowest is back to auto (``None``)."""
    return stepscale.down(WB_STEPS_K, current_k, metered_k, DEFAULT_START_K)


def value_label(kelvin: float) -> str:
    """``5600K``; a metered value is rounded to the nearest 100 K."""
    return f"{int(round(kelvin / 100.0)) * 100}K"


def next_step(current_k: float | None) -> int | None:
    if current_k is None or not math.isfinite(current_k) or current_k <= 0:
        return WB_STEPS_K[0]
    for step in WB_STEPS_K:
        if step > current_k:
            return step
    return None


def label(current_k: int | None) -> str:
    return "WB A" if current_k is None else f"WB {int(current_k)}K"
