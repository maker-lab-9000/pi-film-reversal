# pifilm/display/shutter.py
"""Shutter-priority steps for the viewfinder's shutter buttons. Pure.

The series is the standard 1/3-stop shutter scale from 1/2000 s to 1 s, as
(microseconds, label) pairs, with ``None`` meaning ``A``: full auto exposure. ``+``
is faster and ``-`` is slower; one step slower than 1 s is back to ``A``.

From ``A`` the first tap in either direction lands on the step nearest the exposure
auto-exposure is using right now, so switching into shutter priority never jumps the
picture; the next tap moves it. "Nearest" is measured in stops (log2), not
microseconds, because a stop is what the eye sees. Before any metered frame (the
first frame, or after a camera error) the start is 1/125, a safe hand-held speed.

The labels are ASCII only: the viewfinder's default font has no glyphs beyond it.
"""

from __future__ import annotations

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


def faster(current_us: int | None, metered_us: float | None) -> int | None:
    if current_us is None:
        return _start(metered_us)
    i = _nearest_index(current_us)
    if SHUTTER_STEPS_US[i] > current_us:  # off-series value between steps: snap down
        return SHUTTER_STEPS_US[i - 1] if i > 0 else SHUTTER_STEPS_US[0]
    return SHUTTER_STEPS_US[max(0, i - 1)]


def slower(current_us: int | None, metered_us: float | None) -> int | None:
    if current_us is None:
        return _start(metered_us)
    i = _nearest_index(current_us)
    if SHUTTER_STEPS_US[i] < current_us:  # off-series value between steps: snap up
        i += 1
        return SHUTTER_STEPS_US[i] if i < len(SHUTTER_STEPS_US) else None
    return SHUTTER_STEPS_US[i + 1] if i + 1 < len(SHUTTER_STEPS_US) else None


def label(us: int | None) -> str:
    if us is None:
        return "A"
    if us in _LABELS:
        return _LABELS[us]
    return _LABELS[SHUTTER_STEPS_US[_nearest_index(us)]]
