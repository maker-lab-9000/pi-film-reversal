"""Idle dimming for the LCD: full brightness, then dim, then off.

The panel is lit by a PWM backlight and the camera is often left on the X728
battery between shots, so an untouched screen first drops to half brightness
(still readable, still live) and later goes dark. Pure timing, no hardware: the
viewfinder loop feeds it activity and applies ``backlight`` to the panel.

A delay of 0 disables that step. An off delay shorter than the dim delay simply
skips dimming, since the screen is already off when dimming would start.
"""

from __future__ import annotations

from enum import Enum

DIM_FRACTION = 0.5


class Screen(Enum):
    FULL = "full"
    DIM = "dim"
    OFF = "off"


class IdleDimmer:
    def __init__(self, *, full: int, dim_after: float, off_after: float, now: float) -> None:
        if dim_after < 0 or off_after < 0:
            raise ValueError(f"idle delays must be >= 0, got {dim_after} and {off_after}")
        self._full = int(full)
        self._dim_after, self._off_after = float(dim_after), float(off_after)
        self._last_activity = float(now)

    def wake(self, now: float) -> None:
        """Record activity: a touch, a capture in progress or one just finished."""
        self._last_activity = float(now)

    def screen(self, now: float) -> Screen:
        idle = now - self._last_activity
        if self._off_after and idle >= self._off_after:
            return Screen.OFF
        if self._dim_after and idle >= self._dim_after:
            return Screen.DIM
        return Screen.FULL

    def backlight(self, now: float) -> int:
        """Backlight percent for the panel at ``now``."""
        screen = self.screen(now)
        if screen is Screen.OFF:
            return 0
        if screen is Screen.DIM:
            return int(round(self._full * DIM_FRACTION))
        return self._full
