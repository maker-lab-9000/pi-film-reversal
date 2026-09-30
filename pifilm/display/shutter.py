# pifilm/display/shutter.py
"""Shutter-priority steps for the viewfinder's shutter buttons. Pure.

The series is the standard 1/3-stop shutter scale from 1/2000 s to 1 s, as
(microseconds, label) pairs, with ``None`` meaning ``A``: full auto exposure. ``+``
is faster and ``-`` is slower; one step slower than 1 s is back to ``A``. A value
between steps moves to the next series value in that direction, never skipping one.

From ``A`` the first tap in either direction lands on the step nearest the exposure
auto-exposure is using right now, so switching into shutter priority never jumps the
picture; the next tap moves it. "Nearest" is measured in stops (log2), not
microseconds, because a stop is what the eye sees. Before any metered frame (the
first frame, or after a camera error) the start is 1/125, a safe hand-held speed.

The labels are ASCII only: the viewfinder's default font has no glyphs beyond it.
"""

from __future__ import annotations

import bisect
import math

SHUTTER_STEPS: tuple[tuple[int, str], ...] = (
    (500, "1/2000"), (625, "1/1600"), (800, "1/1250"), (1000, "1/1000"),
    (1250, "1/800"), (1563, "1/640"), (2000, "1/500"), (2500, "1/400"),
    (3125, "1/320"), (4000, "1/250"), (5000, "1/200"), (6250, "1/160"),
    (8000, "1/125"), (10_000, "1/100"), (12_500, "1/80"), (16_667, "1/60"),
    (20_000, "1/50"), (25_000, "1/40"), (33_333, "1/30"), (40_000, "1/25"),
    (50_000, "1/20"), (66_667, "1/15"), (76_923, "1/13"), (100_000, "1/10"),
    (125_000, "1/8"), (166_667, "1/6"), (200_000, "1/5"), (250_000, "1/4"),
    (300_000, "0.3s"), (400_000, "0.4s"), (500_000, "0.5s"), (600_000, "0.6s"),
    (800_000, "0.8s"), (1_000_000, "1s"),
)
SHUTTER_STEPS_US: tuple[int, ...] = tuple(us for us, _ in SHUTTER_STEPS)
DEFAULT_START_US = 8000  # 1/125
_LABELS = dict(SHUTTER_STEPS)


def _nearest_index(us: float) -> int:
    target = math.log2(us)
    return min(range(len(SHUTTER_STEPS_US)),
               key=lambda i: abs(math.log2(SHUTTER_STEPS_US[i]) - target))


def _start(metered_us: float | None) -> int:
    if metered_us is None or not metered_us > 0 or not math.isfinite(metered_us):
        return DEFAULT_START_US
    return SHUTTER_STEPS_US[_nearest_index(metered_us)]


def _is_auto(us: float | None) -> bool:
    return us is None or not math.isfinite(us) or us <= 0


def faster(current_us: int | None, metered_us: float | None) -> int | None:
    if _is_auto(current_us):
        return _start(metered_us)
    i = bisect.bisect_left(SHUTTER_STEPS_US, current_us)  # steps below current: [0, i)
    return SHUTTER_STEPS_US[max(0, i - 1)]


def slower(current_us: int | None, metered_us: float | None) -> int | None:
    if _is_auto(current_us):
        return _start(metered_us)
    i = bisect.bisect_right(SHUTTER_STEPS_US, current_us)  # first step above current
    return SHUTTER_STEPS_US[i] if i < len(SHUTTER_STEPS_US) else None


def label(us: int | None) -> str:
    """Label of the nearest step; ``A`` for ``None`` and for a non-positive or
    non-finite value (which ``faster``/``slower`` also treat as auto)."""
    if _is_auto(us):
        return "A"
    return _LABELS[SHUTTER_STEPS_US[_nearest_index(us)]]
