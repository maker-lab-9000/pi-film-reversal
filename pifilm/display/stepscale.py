# pifilm/display/stepscale.py
"""Generic step scale with an automatic end, shared by the ISO and white-balance
controls. Pure, stdlib only.

``steps`` is an ascending series of positive numbers; ``None`` as a current value
means auto. From auto, the first press in either direction lands on the step nearest
the value auto is using right now (``metered``), so switching to manual never jumps
the picture; the next press moves it. "Nearest" is measured in log2, not linearly,
because a stop is what the eye sees (the same reasoning as ``shutter.py``). An exact
tie goes to the lower step. Without a usable metered value (none yet, or a
meaningless one) the start is ``default``, which is one of the steps.

From a step, ``up`` moves to the smallest step strictly greater than the current
value, and stays on the top step when there is none. ``down`` moves to the largest
step strictly less, and returns ``None`` (back to auto) when there is none. A value
between steps moves to the next step in that direction, never skipping one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def is_usable(value: object) -> bool:
    """A positive finite real number. ``bool`` is excluded although it is an ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def is_auto(value: float | None) -> bool:
    """``None``, non-finite, ``<= 0``, a ``bool`` or a non-number: all mean auto."""
    return not is_usable(value)


def nearest(steps: Sequence[int], value: float) -> int:
    target = math.log2(value)
    # min() keeps the first of equal keys, and steps are ascending: ties go lower.
    return min(steps, key=lambda s: abs(math.log2(s) - target))


def _start(steps: Sequence[int], metered: float | None, default: int) -> int:
    return nearest(steps, metered) if is_usable(metered) else default


def up(steps: Sequence[int], current: float | None, metered: float | None,
       default: int) -> int:
    if is_auto(current):
        return _start(steps, metered, default)
    for step in steps:
        if step > current:
            return step
    return steps[-1]


def down(steps: Sequence[int], current: float | None, metered: float | None,
         default: int) -> int | None:
    if is_auto(current):
        return _start(steps, metered, default)
    for step in reversed(steps):
        if step < current:
            return step
    return None
