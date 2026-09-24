"""Light-meter readout for the viewfinder, computed from a preview frame and its
libcamera metadata. Pure functions; nothing here touches hardware.

The needle (``deviation_ev``) is the scene's mean linear luminance against an
18 % mid-grey, in stops. With auto-exposure converged it rests near zero and only
swings when AE has run out of range or EV compensation is dialled in, which is
exactly what a camera's meter shows in auto mode.

``focus_score``/``FocusTracker`` serve the other manual control: the IMX477 has
no autofocus, so the ring is turned by hand and the screen has to say which way
is better. The score must be *absolute* - a blurred frame reads low on its own.
The first version showed a Laplacian variance against its own decaying peak, and
on the device a lens left out of focus read full: with no sharp frame since
switching on, the blurred frame *was* the peak. So the score is a blur ratio that
the scene's contrast cancels out of, gated against sensor noise, and the tracker
only places the "best seen" mark.
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
# Only edges whose re-blurred step is this many times the step noise alone makes
# there are scored: at 1/15 s and ISO 1580 the preview carries pixel noise that a
# sharpness measure would otherwise read as detail.
FOCUS_NOISE_GATE = 4.0
# Neighbour-step standard deviation of unit white noise after the binomial smoothing
# and the 9-tap re-blur in ``focus_score`` (measured): the noise estimate in step units.
_REBLURRED_NOISE_STEP = 0.043
# Step that the re-blur removes at an edge pixel from unit noise alone (measured on
# a noisy ramp): subtracted per scored pixel so noise does not read as sharpness.
_NOISE_LOST_STEP = 0.07
# The raw blur ratio a sharply focused preview reaches (0.55-0.65 on real 12 MP
# captures reduced to preview size, before noise), mapped to a full bar.
FOCUS_FULL_SCALE = 0.55
# The "best seen" mark halves in about three seconds: long enough to rack past
# best focus and come back to it, short enough to let go of an old subject.
FOCUS_DECAY_PER_SECOND = 0.8


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
    focus_peak: float | None = None


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


def _smooth(x: np.ndarray, axis: int, taps: np.ndarray) -> np.ndarray:
    """Convolve along one axis with edge padding, by summing shifted slices."""
    pad = len(taps) // 2
    widths = [(0, 0)] * x.ndim
    widths[axis] = (pad, pad)
    padded = np.pad(x, widths, mode="edge")
    n = x.shape[axis]
    out = np.zeros_like(x)
    for i, tap in enumerate(taps):
        window = [slice(None)] * x.ndim
        window[axis] = slice(i, i + n)
        out += tap * padded[tuple(window)]
    return out


_BINOMIAL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16   # a Gaussian, sigma ~1
_REBLUR = np.full(9, 1 / 9, dtype=np.float32)


def _noise_sigma(luma: np.ndarray) -> float:
    """Immerkaer's fast estimate: the [1,-2,1] x [1,-2,1] mask cancels smooth
    image structure and leaves mostly noise."""
    d2 = luma[:-2] - 2 * luma[1:-1] + luma[2:]
    r = d2[:, :-2] - 2 * d2[:, 1:-1] + d2[:, 2:]
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(r)) / 6)


def focus_score(preview_rgb: np.ndarray) -> float:
    """Sharpness of the frame's central region, 0.0 (no usable detail) to 1.0.

    Crete-Roffet's no-reference blur measure: blur the image again and see how
    much of its neighbour-to-neighbour contrast that removes. A sharp image
    loses a lot of it; an already blurred one barely changes. Being a ratio of
    two contrasts, the scene's own contrast cancels out, so a dim, flat subject
    in focus reads as high as a bright one - which a Laplacian variance does not.

    Noise would look like the finest detail of all, so the image is lightly
    smoothed first, and only edges count: pixels whose *re-blurred* step stands
    well clear of what noise alone would make there (``FOCUS_NOISE_GATE``). The
    re-blurred step is nearly noise-free and barely changes with focus, so
    choosing edges by it does not favour the sharp pixels - gating on the raw
    step does, and made a blurred low-contrast frame read sharp. What noise
    itself adds to the lost contrast at those pixels is known from the noise
    estimate and subtracted (``_NOISE_LOST_STEP``). A frame with no
    edges above the gate (a blank wall, or pure noise) scores 0.0: nothing to
    focus on is not the same as sharp.

    Plain gamma-encoded bytes, not linear light: this is a contrast measure and
    sRGB encoding is what makes mid-tone detail visible to it. Array slicing only,
    to keep ``pifilm/display/`` free of OpenCV (see ``viewfinder.py``).
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
    noise = _noise_sigma(luma)
    gate = FOCUS_NOISE_GATE * _REBLURRED_NOISE_STEP * noise
    luma = _smooth(_smooth(luma, 0, _BINOMIAL), 1, _BINOMIAL)
    lost = kept = 0.0
    for axis in (0, 1):
        step = np.abs(np.diff(luma, axis=axis))
        reblurred = np.abs(np.diff(_smooth(luma, axis, _REBLUR), axis=axis))
        edges = reblurred > max(gate, 1e-3)
        kept += float(step[edges].sum())
        lost += float(np.maximum(0.0, step - reblurred)[edges].sum())
        lost -= _NOISE_LOST_STEP * noise * int(edges.sum())
    if kept <= 0.0:
        return 0.0
    return max(0.0, min(1.0, lost / kept / FOCUS_FULL_SCALE))


class FocusTracker:
    """The "best seen" mark on the focus bar: the highest recent score, decaying.

    The bar itself is the absolute score; this only says where the best focus
    was, so the user can rack past it and come back. It must fade, or pointing
    the camera at a new, less detailed subject would leave the mark out of reach.
    """

    def __init__(self, decay_per_second: float = FOCUS_DECAY_PER_SECOND) -> None:
        self._decay = decay_per_second
        self._peak = 0.0
        self._last: float | None = None

    @property
    def peak(self) -> float:
        return self._peak

    def update(self, level: float, now: float) -> float:
        """Take this frame's score; return the mark's level."""
        level = max(0.0, min(1.0, float(level)))
        if self._last is not None:
            self._peak *= self._decay ** max(0.0, now - self._last)
        self._last = now
        self._peak = max(self._peak, level)
        return self._peak


def compute_reading(
    metadata: dict | None, preview_rgb: np.ndarray, ev_comp: float, power: Any,
    *, focus: float | None = None, focus_peak: float | None = None,
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
        shutter, iso, float(ev_comp), lux, deviation, clip_pct, battery, external,
        focus, focus_peak,
    )
