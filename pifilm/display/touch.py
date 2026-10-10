"""Touch types shared by every panel: a point, a tap, and the tap detector.

They lived in ``cst3530.py`` while the 2.8" panel was the only one. The DSI panel
reads touch from kernel input events instead of I2C, and the viewfinder loop only
ever needs these three names, so they sit apart from either driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


class TapDetector:
    """Turn per-step point lists into taps: down then up within ``max_hold`` s,
    moving less than ``max_move`` px. Holds and drags produce nothing."""

    def __init__(self, max_hold: float = 0.6, max_move: float = 20.0) -> None:
        self._max_hold, self._max_move = max_hold, max_move
        self._down: tuple[TouchPoint, float] | None = None
        self._last: TouchPoint | None = None

    def feed(self, points: list[TouchPoint], now: float) -> Tap | None:
        if points:
            if self._down is None:
                self._down = (points[0], now)
            self._last = points[0]
            return None
        if self._down is None:
            return None
        first, t0 = self._down
        last = self._last or first
        self._down, self._last = None, None
        moved = math.hypot(last.x - first.x, last.y - first.y)
        if now - t0 <= self._max_hold and moved < self._max_move:
            return Tap(first.x, first.y)
        return None
