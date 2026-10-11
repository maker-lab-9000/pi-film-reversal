# pifilm/display/iso.py
"""Manual ISO steps for the viewfinder's ISO control. Pure.

The series is the full-stop scale 100 to 1600, with ``None`` meaning auto. ``higher``
and ``lower`` move one stop; one step below 100 is back to auto, and ``higher`` stays
on 1600. From auto the first press in either direction lands on the step nearest the
sensitivity auto-exposure is using right now, measured in log2 because a stop is what
the eye sees (see ``stepscale``); before any metered frame the start is 400.

The labels are ASCII only: the viewfinder's default font has no glyphs beyond it.
"""

from __future__ import annotations

from . import stepscale

ISO_STEPS: tuple[int, ...] = (100, 200, 400, 800, 1600)
DEFAULT_START = 400


def higher(current: int | None, metered: float | None) -> int:
    return stepscale.up(ISO_STEPS, current, metered, DEFAULT_START)


def lower(current: int | None, metered: float | None) -> int | None:
    return stepscale.down(ISO_STEPS, current, metered, DEFAULT_START)


def label(iso: int) -> str:
    return str(int(iso))
