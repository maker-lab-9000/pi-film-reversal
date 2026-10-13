"""Per-image white balance and exposure normalisation.

This is the "dynamic" half of the Parr pipeline. The LUT fitted by the
trainer expects input at a neutral white point and a fixed exposure, so the
same normalisation runs on every capture: a tungsten-lit room and a cloudy
street both reach the LUT looking like the images it was fitted on. The
trainer applies this exact code to the corpora (``normalize_float``); the Pi
applies the same maths through three 256-entry lookup tables
(``normalize_u8``).

Why lookup tables suffice
-------------------------
White balance is a per-channel gain in linear light, baked into three
256-entry tables. The tone step -- a scalar gain (``levels=False``) or a
black/white-point stretch followed by a gamma (``levels=True``) -- is one
monotone curve. Applied per channel it would fit the same three tables, but
per-channel gamma widens the gaps between R, G and B, which is saturation:
skin came out +50% chroma in bright rooms. So by default the curve is
applied to Y of YCrCb through a fourth table and chroma is left alone
(``levels_tone="luma"``). Two ``cvtColor`` calls and three ``cv2.LUT``
calls: about 100 ms for 1080p on a Pi 400.

Levels, and why both sides get it
---------------------------------
``levels=True`` puts the 0.5th percentile of luminance at black and the
99.5th at white, then a gamma puts the median on the exposure target
without moving either end. It was added for scanned targets, which are
flat. Giving it to targets only was a mistake found on the first outdoor
captures: the target's white was pinned at 1.0 while the source's sat at
gain x white, so the target pool was 0.03-0.045 L lighter from the first
quartile up and the LUT learned to lift shadows, which read as haze. Both
sides are now normalised the same way; the runtime bakes it into the same
three tables.

Clipping-aware lift
-------------------
The levels gamma brightens any frame whose stretched median sits below the
exposure target, and it used to do so however much of the frame was already
at the ceiling. On the IMX708 pilot that fired on 53 of 108 outdoor-ish
shots: their highlights were already at 1.0, so the lift could only blow
them further while pulling the shadows up to grey -- the look readers
called "faded". ``levels_lift_highlight_ref`` scales the lift down in
proportion to the fraction of pixels at the ceiling, reaching no lift at
all at that fraction. The fraction is measured over every balanced pixel
and not over the statistics mask, because the mask excludes luminance above
``stats_lum_max`` -- that exclusion is precisely why such a frame reads as
"dark" to the median, so the mask can never see the clipping that caused
it. Darkening is never damped: a frame that is genuinely too bright should
still come down. ``None`` reproduces the old behaviour exactly; the
artifact is what turns it on.

Targets versus sources
----------------------
Reference photographs use ``white_balance=False`` and by default ``levels=False``:
their color balance is retained, with exposure matching but no levels stretch.

Normalising twice is not supported
----------------------------------
The statistics ignore highlights (``stats_lum_max``), so a pass that
brightens an image pushes pixels past the cutoff and the next pass measures
a different set. On the levels path the second pass is mild (gamma 0.94 on
a photo-like ramp) because black and white are pinned; on the legacy gain
path it is not (a second gain of 1.3), because a gain can push the whole
top of the range out of the statistic. Damping the lift leaves a frame
deliberately short of the target, so a second pass over a clipped frame
finds a little more residual lift than it otherwise would (gamma 0.88
rather than 0.94); it is still bounded and still milder than the first.
Nothing in the project normalises an image twice: the trainer normalises
corpora once and the Pi normalises a capture once, with the same code.
This is documented rather than engineered away, and tested as a bounded
second pass on the levels path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ._cv2 import require_cv2
from .color import LUMA_709, linear_to_srgb, srgb_to_linear

cv2 = require_cv2()

_EPS = 1e-6

# A linear luminance at or above this (about sRGB 254/255) is "at the ceiling".
# Not a parameter: it is the definition of clipped, not a tuning knob.
LEVELS_HIGHLIGHT_LIN = 0.97


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass
class NormalizeParams:
    white_balance: bool = True
    wb_gain_min: float = 0.6
    wb_gain_max: float = 1.6
    exposure_target_median: float = 0.25
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 3.0
    stats_lum_min: float = 0.02
    # 0.90, and not lower: excluding highlights from the statistic (0.60 was
    # tried) helped a face against a white wall but is content-dependent --
    # the K-14 corpus is 44% sky, so ignoring bright pixels re-exposed slides
    # for their ground and pushed skies toward white, while sky-light source
    # photos barely moved. The pools drifted apart and the held-out distance
    # went from 0.0122 to 0.0211, worse than identity.
    stats_lum_max: float = 0.90
    # Levels: black/white-point stretch plus a gamma to the median. Added for
    # scanned targets -- across 150 LoC Parrs the white point sat at a
    # median 0.72 linear luminance, and learning from them unstretched taught
    # the LUT to push white to 0.85 -- and then applied to sources as well,
    # because pinning only the target's white taught the LUT to lift shadows.
    # See the module docstring. Off by default; the trainer turns it on for
    # both corpora and records it in the artifact, which is what the Pi reads.
    levels: bool = False
    levels_low_pct: float = 0.5
    levels_high_pct: float = 99.5
    levels_max_stretch: float = 4.0
    levels_gamma_min: float = 0.5
    levels_gamma_max: float = 2.0
    # How the stretch+gamma is applied. "luma": to the Y channel of YCrCb with
    # chroma untouched. "channel": to each channel in linear light, which
    # widens the gaps between R, G and B -- that is saturation, and it made
    # skin +50% chroma in bright rooms while the LUT contributed 1%.
    levels_tone: str = "luma"
    # Where the levels gamma puts the median. None means the exposure target,
    # i.e. the same median as the source -- which assumes a well-exposed slide
    # has the same median as a well-exposed digital frame. It does not: slides
    # are exposed for highlights and are dense, and gamma-lifting them to the
    # camera's median taught the LUT to lift shadows, which read as haze on
    # outdoor shots. Set lower to keep the film's density in what is learned.
    levels_target_median: float | None = None
    # Clipping-aware lift. The levels gamma brightens a frame whose median is
    # below the exposure target regardless of how much of it is already at the
    # ceiling: on the IMX708 pilot 53 of 108 outdoor-ish shots were lifted while
    # their highlights sat at 1.0, which blew the highlights further and lifted
    # shadows to grey (read as "faded"). When set, the fraction of pixels at the
    # ceiling scales the lift down linearly, reaching zero lift at this fraction.
    # None keeps the old behaviour exactly; the artifact turns it on.
    levels_lift_highlight_ref: float | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name in ("white_balance", "levels", "levels_tone"):
                continue
            if getattr(self, f.name) is not None:
                _check_finite(f.name, getattr(self, f.name))
        if not 0.0 <= self.levels_low_pct < self.levels_high_pct <= 100.0:
            raise ValueError(
                f"levels_low_pct ({self.levels_low_pct}) must be below "
                f"levels_high_pct ({self.levels_high_pct}), both within [0, 100]"
            )
        if self.levels_tone not in ("luma", "channel"):
            raise ValueError(f"levels_tone must be 'luma' or 'channel', got {self.levels_tone!r}")
        if self.levels_max_stretch < 1.0:
            raise ValueError(
                f"levels_max_stretch must be at least 1, got {self.levels_max_stretch}"
            )
        if self.levels_target_median is not None and not 0.0 < self.levels_target_median < 1.0:
            raise ValueError(
                f"levels_target_median must be in (0, 1), got {self.levels_target_median}"
            )
        ref = self.levels_lift_highlight_ref
        if ref is not None and not 0.0 < ref <= 1.0:
            raise ValueError(f"levels_lift_highlight_ref must be in (0, 1], got {ref}")
        if not 0.0 < self.levels_gamma_min < self.levels_gamma_max:
            raise ValueError(
                f"levels_gamma_min ({self.levels_gamma_min}) must be positive and below "
                f"levels_gamma_max ({self.levels_gamma_max})"
            )
        if self.wb_gain_min <= 0:
            raise ValueError(f"wb_gain_min must be positive, got {self.wb_gain_min}")
        if self.exposure_gain_min <= 0:
            raise ValueError(f"exposure_gain_min must be positive, got {self.exposure_gain_min}")
        if self.wb_gain_min >= self.wb_gain_max:
            raise ValueError(
                f"wb_gain_min ({self.wb_gain_min}) must be below wb_gain_max ({self.wb_gain_max})"
            )
        if self.exposure_gain_min >= self.exposure_gain_max:
            raise ValueError(
                f"exposure_gain_min ({self.exposure_gain_min}) must be below "
                f"exposure_gain_max ({self.exposure_gain_max})"
            )
        if not 0.0 < self.exposure_target_median < 1.0:
            raise ValueError(
                f"exposure_target_median must be in (0, 1), got {self.exposure_target_median}"
            )
        if not 0.0 <= self.stats_lum_min < self.stats_lum_max <= 1.0:
            raise ValueError(
                f"stats_lum_min ({self.stats_lum_min}) must be below "
                f"stats_lum_max ({self.stats_lum_max}), both within [0, 1]"
            )

    @classmethod
    def from_dict(cls, d: dict) -> NormalizeParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gains:
    """Per-channel white balance gains and a scalar exposure gain, in linear light."""

    wb: np.ndarray
    exposure: float
    clamped: dict = field(default_factory=lambda: {"wb": False, "exposure": False})
    # Set only by the levels path: the black/white points found, the stretch
    # and gamma applied (after white balance). ``exposure`` stays 1.0 there,
    # because no scalar gain was applied.
    levels: dict | None = None

    @property
    def combined(self) -> np.ndarray:
        return (np.asarray(self.wb, dtype=np.float32) * np.float32(self.exposure)).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        d = {
            "wb": [round(float(g), 4) for g in self.wb],
            "exposure": round(float(self.exposure), 4),
            "clamped": dict(self.clamped),
        }
        if self.levels is not None:
            d["levels"] = {k: round(float(v), 4) for k, v in self.levels.items()}
        return d


def _require_float_image(rgb: np.ndarray, caller: str) -> np.ndarray:
    """Guard the float path against uint8 input, which it would destroy silently.

    ``srgb_to_linear`` clips to [0, 1], so every uint8 value from 1 to 255
    collapses to 1.0 and the image's entire content is lost with no error
    raised — a trainer handing over the result of ``load_rgb`` would get a
    flat frame and never know. Every other public entry point in this package
    validates its input; this one must too.
    """
    rgb = np.asarray(rgb)
    if not np.issubdtype(rgb.dtype, np.floating):
        raise ValueError(
            f"{caller} expects a float image in [0, 1], got dtype {rgb.dtype}. "
            "Divide a uint8 array by 255 first, or use normalize_u8."
        )
    return rgb


def _grey_world(sel_lin: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, bool]:
    means = np.maximum(sel_lin.mean(axis=0), _EPS)
    raw = float(means @ LUMA_709) / means
    wb = np.clip(raw, params.wb_gain_min, params.wb_gain_max).astype(np.float32)
    # Compare against the bounds, not against the clipped array: `raw` and
    # `wb` can differ by a float32 cast alone, which is not a clamp.
    clamped = bool(np.any(raw < params.wb_gain_min) or np.any(raw > params.wb_gain_max))
    return wb, clamped


def _stats_mask(lum: np.ndarray, params: NormalizeParams) -> np.ndarray:
    mask = (lum >= params.stats_lum_min) & (lum <= params.stats_lum_max)
    return mask if mask.mean() >= 0.01 else np.ones_like(mask)


def _damp_lift(raw_gamma: float, highlight_frac: float, ref: float | None) -> tuple[float, float]:
    """Scale a brightening lift down as the frame's highlights approach the ceiling.

    Returns ``(gamma, weight)``. ``weight`` is 1.0 when nothing was damped: ``ref``
    is None, the frame is being darkened (``raw_gamma >= 1``), or no pixel is at
    the ceiling. It falls linearly to 0.0 as ``highlight_frac`` reaches ``ref``,
    at which point the lift is removed entirely (gamma 1.0). Darkening is never
    touched: a frame that is too bright should still be brought down.
    """
    if ref is None or raw_gamma >= 1.0:
        return raw_gamma, 1.0
    weight = float(np.clip(1.0 - highlight_frac / ref, 0.0, 1.0))
    return 1.0 - (1.0 - raw_gamma) * weight, weight


def compute_gains(rgb: np.ndarray, params: NormalizeParams) -> Gains:
    """Grey-world white balance, then either a median-to-target gain or levels.

    Everything is in linear light. With ``levels`` the black and white points
    and the stretch are shared by all three channels and the gamma is a
    shared exponent, so a neutral input stays neutral; both are clamped and
    the clamp is recorded, never silent.
    """
    rgb = _require_float_image(rgb, "compute_gains")
    lin = srgb_to_linear(rgb).reshape(-1, 3)
    lum = lin @ LUMA_709
    sel = lin[_stats_mask(lum, params)]

    if params.white_balance:
        wb, wb_clamped = _grey_world(sel, params)
    else:
        wb, wb_clamped = np.ones(3, dtype=np.float32), False

    if not params.levels:
        median_lum = float(np.median((sel * wb) @ LUMA_709))
        raw_exposure = params.exposure_target_median / max(median_lum, _EPS)
        lo_g, hi_g = params.exposure_gain_min, params.exposure_gain_max
        exposure = float(np.clip(raw_exposure, lo_g, hi_g))
        return Gains(
            wb=wb,
            exposure=exposure,
            clamped={"wb": wb_clamped, "exposure": not lo_g <= raw_exposure <= hi_g},
        )

    balanced = lin * wb
    lum_b = balanced @ LUMA_709
    pct = np.percentile(lum_b, [params.levels_low_pct, params.levels_high_pct])
    lo, hi = float(pct[0]), float(pct[1])
    lo_c = np.percentile(balanced, params.levels_low_pct, axis=0)
    raw_stretch = 1.0 / max(hi - lo, _EPS)
    stretch = min(raw_stretch, params.levels_max_stretch)
    stretched = np.clip((balanced - lo) * stretch, 0.0, 1.0)
    lum_s = stretched @ LUMA_709
    median = float(np.clip(np.median(lum_s[_stats_mask(lum_s, params)]), _EPS, 1.0 - _EPS))
    target = params.levels_target_median
    if target is None:
        target = params.exposure_target_median
    raw_gamma = math.log(target) / math.log(median)
    # Over ALL balanced pixels, not the stats mask: the mask excludes luminance
    # above 0.90, so it can never see the ceiling -- that exclusion is why a
    # bright frame reads as "dark" to the median in the first place.
    highlight_frac = float(np.mean(lum_b >= LEVELS_HIGHLIGHT_LIN))
    damped, lift_weight = _damp_lift(raw_gamma, highlight_frac, params.levels_lift_highlight_ref)
    gamma = float(np.clip(damped, params.levels_gamma_min, params.levels_gamma_max))
    clamped = raw_stretch > params.levels_max_stretch or not (
        params.levels_gamma_min <= damped <= params.levels_gamma_max
    )
    return Gains(
        wb=wb,
        exposure=1.0,
        clamped={"wb": wb_clamped, "exposure": False, "levels": bool(clamped)},
        levels={"low": lo, "high": hi, "stretch": stretch, "gamma": gamma,
                "raw_gamma": float(raw_gamma), "highlight_frac": highlight_frac,
                "lift_weight": lift_weight,
                "black_rgb_spread": float(lo_c.max() - lo_c.min()), "tone": params.levels_tone},
    )


def _apply_levels_linear(lin: np.ndarray, levels: dict) -> np.ndarray:
    low, stretch = np.float32(levels["low"]), np.float32(levels["stretch"])
    out = np.clip((lin - low) * stretch, 0.0, 1.0)
    return np.power(out, np.float32(levels["gamma"]))


def _tone_curve(y_srgb: np.ndarray, levels: dict) -> np.ndarray:
    """The stretch+gamma as a curve on sRGB-encoded luma, for the luma path."""
    return linear_to_srgb(_apply_levels_linear(srgb_to_linear(y_srgb), levels)).astype(np.float32)


def _luma_tone(rgb_srgb: np.ndarray, levels: dict) -> np.ndarray:
    """Apply the tone curve to Y of YCrCb, leaving Cr and Cb as they are."""
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb_srgb, dtype=np.float32), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = _tone_curve(ycc[..., 0], levels)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0).astype(np.float32)


def tone_lut(gains: Gains) -> np.ndarray | None:
    """The 256-entry Y table for the luma path, or None when there is no such step."""
    if gains.levels is None or gains.levels.get("tone", "channel") != "luma":
        return None
    y = _tone_curve(np.arange(256, dtype=np.float32) / 255.0, gains.levels)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def apply_gains_float(rgb: np.ndarray, gains: Gains) -> np.ndarray:
    rgb = _require_float_image(rgb, "apply_gains_float")
    lin = srgb_to_linear(rgb) * gains.combined
    if gains.levels is None:
        return linear_to_srgb(np.clip(lin, 0.0, 1.0))
    if gains.levels.get("tone", "channel") == "luma":
        balanced = linear_to_srgb(np.clip(lin, 0.0, 1.0)).astype(np.float32)
        return _luma_tone(balanced, gains.levels)
    return linear_to_srgb(_apply_levels_linear(lin, gains.levels)).astype(np.float32)


def normalize_float(rgb: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, Gains]:
    """Reference path used by the trainer. ``rgb`` is float32 sRGB in [0, 1]."""
    gains = compute_gains(rgb, params)
    return apply_gains_float(rgb, gains), gains


def gains_to_luts(gains: Gains) -> np.ndarray:
    """Bake the per-channel part into three 256-entry uint8 tables.

    On the luma path that is white balance only; the tone curve lives in
    ``tone_lut`` and is applied to Y afterwards.
    """
    lin = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
    luts = np.empty((3, 256), dtype=np.uint8)
    per_channel_tone = gains.levels is not None and gains.levels.get("tone", "channel") != "luma"
    for c in range(3):
        x = lin * gains.combined[c]
        if per_channel_tone:
            x = _apply_levels_linear(x, gains.levels)
        else:
            x = np.clip(x, 0.0, 1.0)
        luts[c] = np.clip(np.round(linear_to_srgb(x) * 255.0), 0, 255).astype(np.uint8)
    return luts


def stats_gains(
    rgb_u8: np.ndarray, params: NormalizeParams, max_stats_pixels: int = 300_000
) -> Gains:
    """The gains ``normalize_u8`` computes, from the same strided subsample.

    On its own because the GPU pass (``pifilm.gpu``) applies the gains itself:
    the statistics stay on the CPU and stay this code, so a record's audit
    fields are the same whichever path graded the photograph."""
    h, w = rgb_u8.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_stats_pixels))))
    return compute_gains(rgb_u8[::step, ::step].astype(np.float32) / 255.0, params)


def normalize_u8(
    rgb_u8: np.ndarray, params: NormalizeParams, max_stats_pixels: int = 300_000
) -> tuple[np.ndarray, Gains]:
    """Fast path for the Pi: statistics from a strided subsample, applied with ``cv2.LUT``."""
    gains = stats_gains(rgb_u8, params, max_stats_pixels)
    table = np.ascontiguousarray(gains_to_luts(gains).T).reshape(256, 1, 3)
    out = cv2.LUT(np.ascontiguousarray(rgb_u8), table)
    ylut = tone_lut(gains)
    if ylut is not None:
        ycc = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        ycc[..., 0] = cv2.LUT(ycc[..., 0], ylut)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return out, gains
