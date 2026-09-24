"""Light-meter readout for the viewfinder, computed from a preview frame and its
libcamera metadata. Pure functions; nothing here touches hardware.

The needle (``deviation_ev``) is the scene's mean linear luminance against an
18 % mid-grey, in stops. With auto-exposure converged it rests near zero and only
swings when AE has run out of range or EV compensation is dialled in, which is
exactly what a camera's meter shows in auto mode.

``focus``/``FocusTracker`` serve the other manual control: the IMX477 has no
autofocus, so the ring is turned by hand and the screen has to say which way is
better. ``focus_score`` is a plain Laplacian-variance sharpness measure and its
absolute value means nothing — it moves with the scene's own contrast — so the
bar shows the score against a *decaying* peak instead. Racking past best focus
drops the bar immediately; the peak fades over a few seconds so that pointing
the camera at a new subject does not leave the mark stuck at an old scene's
contrast forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..color import luminance, srgb_to_linear

MID_GREY = 0.18
DEVIATION_CLAMP = 3.0
CLIP_LEVEL = 254
# The focus metric reads the middle half of the frame in each axis (a quarter of
# its area): the subject is what the user points the middle of the finder at, and
# a sharp background at the edges must not hold the bar up while the subject is soft.
FOCUS_REGION = 0.5
FOCUS_DECAY_PER_SECOND = 0.5
FOCUS_FLOOR = 1e-6


@dataclass(frozen=True)
class MeterReading:
    shutter: str | None
    iso: int | None
    ev_comp: float
    lux: float | None
    deviation_ev: float
    clip_pct: float
    battery_percent: int | None
    external_power: bool | None
    focus: float | None = None


def format_shutter(exposure_us: float) -> str:
    seconds = exposure_us / 1_000_000
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"1/{round(1.0 / seconds)}"


def format_ev(ev: float) -> str:
    if abs(ev) < 0.05:
        return "0"
    return f"{ev:+.1f}"


def _number(metadata: dict, key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def focus_score(preview_rgb: np.ndarray) -> float:
    """Laplacian variance over the frame's central region: higher is sharper.

    Deliberately plain arithmetic on the gamma-encoded bytes rather than
    linearised light: this is a contrast measure, not a photometric one, and
    sRGB encoding is what makes mid-tone detail visible to it. A 4-neighbour
    Laplacian by array slicing keeps ``pifilm/display/`` free of OpenCV (see the
    module note in ``viewfinder.py``) and costs well under a millisecond on a
    640x480 preview, which has to fit inside a 100 ms frame period.

    A frame with no detail in the centre scores 0.0, which the tracker reads as
    "nothing to say" rather than "out of focus".
    """
    h, w = preview_rgb.shape[:2]
    dh, dw = int(h * FOCUS_REGION / 2), int(w * FOCUS_REGION / 2)
    centre = preview_rgb[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw]
    if centre.shape[0] < 3 or centre.shape[1] < 3:
        return 0.0
    luma = (
        0.299 * centre[..., 0].astype(np.float32)
        + 0.587 * centre[..., 1].astype(np.float32)
        + 0.114 * centre[..., 2].astype(np.float32)
    )
    lap = (
        4.0 * luma[1:-1, 1:-1]
        - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    )
    return float(np.var(lap))


class FocusTracker:
    """Turns raw sharpness scores into a 0..1 bar level against a decaying peak.

    The peak is what the bar's mark sits at, and it must fade: without decay the
    first sharp frame of a high-contrast scene would pin the mark so high that
    every later subject reads as out of focus. With ``decay_per_second`` 0.5 the
    remembered peak halves each second, so a few seconds after pointing
    elsewhere the bar can reach the top again.
    """

    def __init__(
        self,
        decay_per_second: float = FOCUS_DECAY_PER_SECOND,
        floor: float = FOCUS_FLOOR,
    ) -> None:
        self._decay = decay_per_second
        self._floor = floor
        self._peak = 0.0
        self._last: float | None = None

    @property
    def peak(self) -> float:
        return self._peak

    def update(self, score: float, now: float) -> float:
        score = max(0.0, float(score))
        if self._last is not None:
            dt = max(0.0, now - self._last)
            self._peak *= self._decay ** dt
        self._last = now
        self._peak = max(self._peak, score)
        if self._peak <= self._floor:
            return 0.0
        return max(0.0, min(1.0, score / self._peak))


def compute_reading(
    metadata: dict | None, preview_rgb: np.ndarray, ev_comp: float, power: Any,
    *, focus: float | None = None,
) -> MeterReading:
    meta = metadata or {}
    exposure = _number(meta, "ExposureTime")
    shutter = format_shutter(exposure) if exposure and exposure > 0 else None
    analogue = _number(meta, "AnalogueGain")
    digital = _number(meta, "DigitalGain")
    iso = None
    if analogue is not None and analogue > 0:
        iso = int(round(analogue * (digital if digital and digital > 0 else 1.0) * 100 / 10) * 10)
    lux = _number(meta, "Lux")

    small = preview_rgb[::4, ::4].astype(np.float32) / 255.0
    mean_luma = float(np.mean(luminance(srgb_to_linear(small))))
    deviation = math.log2(max(mean_luma, 1e-4) / MID_GREY)
    deviation = max(-DEVIATION_CLAMP, min(DEVIATION_CLAMP, deviation))
    clip_pct = float(np.mean(np.any(preview_rgb >= CLIP_LEVEL, axis=-1)) * 100.0)

    battery = int(power.percent) if power is not None else None
    external = bool(power.external_power) if power is not None else None
    return MeterReading(
        shutter, iso, float(ev_comp), lux, deviation, clip_pct, battery, external, focus,
    )
