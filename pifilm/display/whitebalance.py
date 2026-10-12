# pifilm/display/whitebalance.py
"""Manual white-balance steps for the viewfinder's WB control. Pure.

The series is a short list of colour temperatures in kelvin, with ``None`` meaning
automatic white balance. A higher temperature tells the camera the light is bluer,
so it renders warmer: 6500 or 7500 corrects a cold cast. ``warmer`` and ``cooler``
move one step; one step below 3200 is back to auto, and ``warmer`` stays on 7500.
From auto the first press in either direction lands on the step nearest the colour
temperature auto white balance is using right now (see ``stepscale``); before any
metered frame the start is 5600.

The values stay inside what the IMX477's calibration covers. Measured on the Pi 4
on 2026-10-11, a request below 2360 K is clamped to 2360 K and 9000 K is accepted
as asked. A value between steps moves to the next step in that direction, never
skipping one.

The labels are ASCII only.
"""

from __future__ import annotations

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
