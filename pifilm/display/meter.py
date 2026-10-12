"""Light-meter readout for the viewfinder, computed from a preview frame and its
libcamera metadata. Pure functions; nothing here touches hardware.

The needle (``deviation_ev``) is the scene's mean linear luminance against an
18 % mid-grey, in stops. With auto-exposure converged it rests near zero and only
swings when AE has run out of range, or when the shutter and ISO are both fixed by
hand and the exposure is the user's, which is exactly what a camera's meter shows.

``focus_score``/``FocusTracker`` serve the other manual control: the IMX477 has
no autofocus, so the ring is turned by hand and the screen has to say which way
is better. The score must be *absolute* - a blurred frame reads low on its own.
The first version showed a Laplacian variance against its own decaying peak, and
on the device a lens left out of focus read full: with no sharp frame since
switching on, the blurred frame *was* the peak. So the score is a blur ratio that
the scene's contrast cancels out of, measured only at edges that stand clear of
the frame's own noise, and the tracker only places the "best seen" mark.

The second version chose those edges with a noise estimate calibrated on white
noise. On the Pi 4 at ISO 1600 a lens left completely out of focus read 0.43 to
0.67 of the bar, more than a slightly soft frame did: the ISP's noise reduction
leaves the preview's noise in soft blotches, about five times stronger at the
scale the edges are chosen at than white noise of the same fine grain, so most of
a featureless frame passed as edges and its noise was scored as detail. The edge
threshold is now taken from the frame's own re-blurred steps (see
``focus_score``), which needs no model of the noise.
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
# An edge is a pixel whose re-blurred step is this many robust standard deviations
# (from the median absolute deviation) of all the re-blurred steps in the region.
# In a frame with nothing sharp those steps are noise and slow shading, so nothing
# stands that far out; a real edge does. Measured on preview frames from the Pi 4
# at ISO 1600 (2026-10-13): at 6, fully defocused frames score 0.00 to 0.08 raw and
# sharp text 0.54, where the noise-model gate it replaces gave 0.24 to 0.37 and 0.55.
FOCUS_EDGE_SIGMAS = 6.0
# ... and at least this many 8-bit levels per pixel, so the one-level stairs of a
# smooth gradient in a noiseless frame are not edges either.
FOCUS_EDGE_FLOOR = 0.5
# Fewer edge pixels than this share of the region is nothing to focus on: a few
# hot pixels or one speck must not set the bar.
FOCUS_MIN_EDGE_FRACTION = 0.0005
# The raw blur ratio a sharply focused preview reaches (0.55-0.65 on real 12 MP
# captures reduced to preview size, before noise), mapped to a full bar.
FOCUS_FULL_SCALE = 0.55
# The "best seen" mark halves in about three seconds: long enough to rack past
# best focus and come back to it, short enough to let go of an old subject.
FOCUS_DECAY_PER_SECOND = 0.8
ISO_MAX_FRACTION = 0.98


@dataclass(frozen=True)
class MeterReading:
    shutter: str | None
    iso: int | None
    lux: float | None
    deviation_ev: float
    clip_pct: float
    battery_percent: int | None
    external_power: bool | None
    focus: float | None = None
    focus_peak: float | None = None
    # Analogue gain is at the sensor's maximum: a fixed shutter this fast cannot be
    # compensated and the frame will be dark.
    iso_max: bool = False


def format_shutter(exposure_us: float) -> str:
    seconds = exposure_us / 1_000_000
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"1/{round(1.0 / seconds)}"


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


# Mean of the positive half of a unit normal: what noise alone adds, per pixel, to
# a sum that keeps only the positive part of (step - re-blurred step).
_HALF_NORMAL_MEAN = 0.399
_SPREAD_STRIDE = 3


def _spread(steps: np.ndarray) -> float:
    """Robust standard deviation of a field of signed steps (median absolute
    deviation): the level of what is ordinary in the frame, be it noise or slow
    shading, which the few pixels of a real edge do not move. Every third pixel
    each way is plenty for a median and keeps the score near its old cost (about
    4 ms a frame on a laptop against 11 ms with every pixel)."""
    sample = steps[::_SPREAD_STRIDE, ::_SPREAD_STRIDE]
    return 1.4826 * float(np.median(np.abs(sample - np.median(sample))))


def focus_score(preview_rgb: np.ndarray) -> float:
    """Sharpness of the frame's central region, 0.0 (no usable detail) to 1.0.

    Crete-Roffet's no-reference blur measure: blur the image again and see how
    much of its neighbour-to-neighbour contrast that removes. A sharp image
    loses a lot of it; an already blurred one barely changes. Being a ratio of
    two contrasts, the scene's own contrast cancels out, so a dim, flat subject
    in focus reads as high as a bright one - which a Laplacian variance does not.

    Noise would look like the finest detail of all, so the image is lightly
    smoothed first, and only edges count: pixels whose *re-blurred* step stands
    far out from the rest of the region's (``FOCUS_EDGE_SIGMAS`` robust standard
    deviations, and at least ``FOCUS_EDGE_FLOOR``). The re-blurred step barely
    changes with focus, so choosing edges by it does not favour the sharp
    pixels - gating on the raw step does, and made a blurred low-contrast frame
    read sharp. Taking the threshold from the frame itself makes it right for
    whatever noise the frame has, blotchy or fine, without a model of it. What
    noise itself adds to the lost contrast at those pixels is estimated the
    same way, from the spread of the frame's own fine steps, and subtracted.
    A frame with too few edges (a blank wall, pure noise, a
    lens far out of focus) scores 0.0: nothing to focus on is not the same as
    sharp. The price is a region filled edge to edge with one even texture and
    nothing standing out of it, which also reads as nothing to focus on.

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
    luma = _smooth(_smooth(luma, 0, _BINOMIAL), 1, _BINOMIAL)
    lost = kept = 0.0
    found = total = 0
    for axis in (0, 1):
        fine = np.diff(luma, axis=axis)
        coarse = np.diff(_smooth(luma, axis, _REBLUR), axis=axis)
        step, reblurred = np.abs(fine), np.abs(coarse)
        edges = reblurred > max(FOCUS_EDGE_SIGMAS * _spread(coarse), FOCUS_EDGE_FLOOR)
        count = int(edges.sum())
        found += count
        total += edges.size
        kept += float(step[edges].sum())
        lost += float(np.maximum(0.0, step - reblurred)[edges].sum())
        lost -= _HALF_NORMAL_MEAN * _spread(fine) * count
    if kept <= 0.0 or found < FOCUS_MIN_EDGE_FRACTION * total:
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
    metadata: dict | None, preview_rgb: np.ndarray, power: Any,
    *, focus: float | None = None, focus_peak: float | None = None,
    max_gain: float | None = None,
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
    iso_max = bool(
        max_gain is not None and analogue is not None
        and analogue >= ISO_MAX_FRACTION * max_gain
    )
    return MeterReading(
        shutter, iso, lux, deviation, clip_pct, battery, external,
        focus, focus_peak, iso_max,
    )
