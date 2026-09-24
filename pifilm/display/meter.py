"""Light-meter readout for the viewfinder, computed from a preview frame and its
libcamera metadata. Pure functions; nothing here touches hardware.

The needle (``deviation_ev``) is the scene's mean linear luminance against an
18 % mid-grey, in stops. With auto-exposure converged it rests near zero and only
swings when AE has run out of range or EV compensation is dialled in, which is
exactly what a camera's meter shows in auto mode.
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


def compute_reading(
    metadata: dict | None, preview_rgb: np.ndarray, ev_comp: float, power: Any,
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
    return MeterReading(shutter, iso, float(ev_comp), lux, deviation, clip_pct, battery, external)
