# pifilm/display/iso.py
"""Manual ISO steps for the viewfinder's ISO control. Pure.

The series is the full-stop scale 100 to 1600, with ``None`` meaning auto. ``higher``
and ``lower`` move one stop; one step below 100 is back to auto, and ``higher`` stays
on 1600. From auto the first press in either direction lands on the step nearest the
sensitivity auto-exposure is using right now, measured in log2 because a stop is what
the eye sees (see ``stepscale``); before any metered frame the start is 400.

A sensor may not reach the top of the series: ISO is analogue gain x 100, and the
camera refuses a gain above its limit. ``max_iso`` (the limit x 100) removes the
steps above it, so neither the walk up nor the first step from auto can land on a
value the camera would refuse; the scale then simply ends lower. A limit below
every step leaves the lowest one, and a meaningless limit is no limit.

The labels are ASCII only, the rule for every text the viewfinder draws.
"""

from __future__ import annotations

from . import stepscale

ISO_STEPS: tuple[int, ...] = (100, 200, 400, 800, 1600)
DEFAULT_START = 400


def _steps(max_iso: float | None) -> tuple[int, ...]:
    if not stepscale.is_usable(max_iso):
        return ISO_STEPS
    return tuple(step for step in ISO_STEPS if step <= max_iso) or ISO_STEPS[:1]


def higher(current: int | None, metered: float | None,
           max_iso: float | None = None) -> int:
    steps = _steps(max_iso)
    return stepscale.up(steps, current, metered, min(DEFAULT_START, steps[-1]))


def lower(current: int | None, metered: float | None,
          max_iso: float | None = None) -> int | None:
    steps = _steps(max_iso)
    return stepscale.down(steps, current, metered, min(DEFAULT_START, steps[-1]))


def label(iso: int) -> str:
    return str(int(iso))
