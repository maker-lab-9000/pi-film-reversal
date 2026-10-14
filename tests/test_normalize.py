import numpy as np
import pytest

from pifilm import color
from pifilm.normalize import (
    Gains,
    NormalizeParams,
    apply_gains_float,
    compute_gains,
    gains_to_luts,
    normalize_float,
    normalize_u8,
)


def _gradient_image(h=48, w=64, cast=(1.0, 0.9, 0.75)):
    ramp = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    rows = np.linspace(0.8, 1.2, h, dtype=np.float32)[:, None, None]
    return np.clip(ramp * rows * np.array(cast, dtype=np.float32), 0, 1).astype(np.float32)


def test_params_from_dict_ignores_unknown_and_roundtrips():
    p = NormalizeParams.from_dict({"exposure_target_median": 0.2, "bogus": 1})
    assert p.exposure_target_median == 0.2
    assert p.white_balance is True
    assert NormalizeParams.from_dict(p.to_dict()) == p


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"wb_gain_min": 2.0}, "wb_gain_min"),           # min above max
        ({"exposure_gain_min": 0.0}, "exposure_gain_min"),  # not positive
        ({"exposure_target_median": 1.5}, "exposure_target_median"),
        ({"stats_lum_min": 0.95}, "stats_lum_min"),      # min above max
        ({"wb_gain_max": float("nan")}, "wb_gain_max"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        NormalizeParams(**kwargs)


def test_gains_combined_and_dict():
    g = Gains(
        wb=np.array([1.0, 1.1, 1.2], dtype=np.float32),
        exposure=2.0,
        clamped={"wb": False, "exposure": True},
    )
    assert np.allclose(g.combined, [2.0, 2.2, 2.4])
    assert g.to_dict() == {
        "wb": [1.0, 1.1, 1.2],
        "exposure": 2.0,
        "clamped": {"wb": False, "exposure": True},
    }


def test_grey_world_neutralises_a_mild_cast():
    # gains land near 0.83/1.04/1.33, inside the clamps, so the cast fully clears
    img = np.full((32, 32, 3), (0.5, 0.45, 0.4), dtype=np.float32)
    out, gains = normalize_float(img, NormalizeParams())
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    target = NormalizeParams().exposure_target_median
    assert np.median(color.luminance(color.srgb_to_linear(out))) == pytest.approx(target, abs=0.005)
    assert gains.wb[0] < 1.0 < gains.wb[2]
    assert gains.clamped == {"wb": False, "exposure": False}


def test_white_balance_can_be_disabled():
    img = np.full((8, 8, 3), (0.5, 0.4, 0.3), dtype=np.float32)
    gains = compute_gains(img, NormalizeParams(white_balance=False))
    assert np.array_equal(gains.wb, np.ones(3, dtype=np.float32))
    assert gains.exposure > 1.0


def test_gains_are_clamped_and_report_it():
    p = NormalizeParams()
    dark = np.full((8, 8, 3), 0.02, dtype=np.float32)
    g = compute_gains(dark, p)
    assert g.exposure == pytest.approx(p.exposure_gain_max)
    assert g.clamped["exposure"] is True
    bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
    assert compute_gains(bright, p).exposure == pytest.approx(p.exposure_gain_min)
    red = np.full((8, 8, 3), (0.9, 0.1, 0.1), dtype=np.float32)
    g = compute_gains(red, p)
    assert g.wb[0] == pytest.approx(p.wb_gain_min)
    assert g.wb[1] == pytest.approx(p.wb_gain_max)
    assert g.clamped["wb"] is True


def test_a_second_pass_on_the_levels_path_is_mild():
    """Normalising twice is not supported (see the module docstring), but on
    the path the Pi actually runs a second pass must stay mild: black and
    white are pinned, so only the gamma re-fires. Measured: gamma 0.889 then
    0.937, pixels within 6.6/255. The legacy gain path is much worse (a
    second gain of 1.3) and is not what captures use."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.1, 0.7, 480, dtype=np.float32)[None, :].repeat(270, 0)
    noise = rng.normal(0, 0.01, (270, 480, 3))
    img = np.clip(np.stack([base * 1.05, base, base * 0.95], -1) + noise, 0, 1).astype(np.float32)
    p = NormalizeParams(levels=True)
    once, _ = normalize_float(img, p)
    twice, g2 = normalize_float(once, p)
    assert abs(g2.levels["gamma"] - 1.0) < 0.15
    assert np.abs(twice - once).max() < 12 / 255

def test_float_and_u8_paths_agree():
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert out_u8.dtype == np.uint8 and out_u8.shape == img_u8.shape
    assert np.allclose(gains_f.combined, gains_u8.combined, atol=1e-6)
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 1


def test_u8_subsampling_branch_matches_the_reference():
    """At 1080p the Pi always subsamples for statistics, so prove that path.

    The 48x64 fixture above gives ``step == 1``, which quietly skips the
    strided branch entirely — the branch every real capture takes. A
    1920x1080 frame gives ``step == 3``, so the gains come from about a ninth
    of the pixels.

    Two claims, kept apart because they can fail independently: the table
    must apply exactly the gains it computed, and gains from a subsample must
    be close enough to full-image statistics that the result is
    indistinguishable at 8-bit precision. Measured: relative gain difference
    1.4e-4, table exact, fast path within one level.
    """
    img_u8 = (_gradient_image(1080, 1920) * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    reference, gains_full = normalize_float(img_u8.astype(np.float32) / 255.0, p)

    # The subsample really was taken: fewer pixels give slightly different gains.
    assert not np.array_equal(gains_u8.combined, gains_full.combined)
    assert np.allclose(gains_u8.combined, gains_full.combined, rtol=1e-3)

    # The table applies its own gains faithfully.
    own = apply_gains_float(img_u8.astype(np.float32) / 255.0, gains_u8)
    assert np.abs(out_u8.astype(int) - np.round(own * 255).astype(int)).max() <= 1

    # And the fast path is indistinguishable from the full-statistics path.
    assert np.abs(out_u8.astype(int) - np.round(reference * 255).astype(int)).max() <= 1


def test_the_float_path_refuses_uint8_input():
    """uint8 would be clipped to 1.0 by srgb_to_linear and silently destroyed."""
    u8 = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="float image"):
        normalize_float(u8, NormalizeParams())
    with pytest.raises(ValueError, match="float image"):
        compute_gains(u8, NormalizeParams())


def test_luts_are_monotone():
    luts = gains_to_luts(
        Gains(wb=np.array([0.8, 1.0, 1.5], dtype=np.float32), exposure=1.3,
              clamped={"wb": False, "exposure": False})
    )
    assert luts.shape == (3, 256) and luts.dtype == np.uint8
    assert np.all(np.diff(luts.astype(int), axis=1) >= 0)


# --- levels normalisation: the training-side step for scanned targets -------


def _low_contrast_grey(lo=0.05, hi=0.60, n=64, power=1.0):
    """A neutral ramp that never reaches black or white, like an archival scan.

    ``power`` > 1 piles the mass toward the dark end so the median after the
    stretch is low enough that no gamma clamp is hit."""
    ramp = np.linspace(0.0, 1.0, n * n, dtype=np.float32) ** power
    lin = (ramp * (hi - lo) + lo).reshape(n, n)
    rgb = color.linear_to_srgb(np.stack([lin, lin, lin], axis=-1))
    return rgb.astype(np.float32)


def test_levels_pins_black_white_and_median_and_keeps_neutrals_neutral():
    """The scans' whites sit at 0.72 linear luminance (measured, 150 scans).

    Learning from them unstretched taught the LUT to push white to 0.85.
    Levels puts p0.5 at black, p99.5 at white, then a gamma puts the median
    on the exposure target without moving either end.
    """
    p = NormalizeParams(white_balance=False, levels=True)
    out, gains = normalize_float(_low_contrast_grey(power=2), p)
    lum = color.luminance(color.srgb_to_linear(out))
    lo, hi = np.percentile(lum, [p.levels_low_pct, p.levels_high_pct])
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi == pytest.approx(1.0, abs=0.01)
    # The gamma targets the median on the statistics mask, exactly as the
    # source-side exposure gain does, so source and target agree on what
    # "median" means. Recompute it from the recorded black point and stretch
    # and check the exponent lands it on target. The whole-image median sits
    # a little above (0.29 here) because the mask drops the ramp's ends.
    lv = gains.levels
    lum_in = color.luminance(color.srgb_to_linear(_low_contrast_grey(power=2)))
    stretched = np.clip((lum_in - lv["low"]) * lv["stretch"], 0.0, 1.0)
    m = (stretched >= p.stats_lum_min) & (stretched <= p.stats_lum_max)
    landed = np.median(stretched[m]) ** lv["gamma"]
    assert landed == pytest.approx(p.exposure_target_median, abs=0.005)
    assert np.median(lum) == pytest.approx(p.exposure_target_median, abs=0.05)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    assert gains.levels is not None
    assert gains.exposure == 1.0 and np.allclose(gains.wb, 1.0)
    assert gains.clamped == {"wb": False, "exposure": False, "levels": False}


def test_levels_clamps_are_recorded_not_silent():
    p = NormalizeParams(white_balance=False, levels=True)
    # Nearly flat: the raw stretch would be ~20x, far past the 4x ceiling.
    flat = _low_contrast_grey(0.40, 0.45)
    _, g = normalize_float(flat, p)
    assert g.levels["stretch"] == pytest.approx(p.levels_max_stretch)
    assert g.clamped["levels"] is True
    # Mostly black with a small bright patch: median after stretch ~0.02, so
    # the gamma that would reach 0.25 is ~0.35, below the 0.5 floor.
    dark = _low_contrast_grey(0.0, 0.02)
    dark[:4, :4] = 1.0
    _, g = normalize_float(dark, p)
    assert g.levels["gamma"] == pytest.approx(p.levels_gamma_min)
    assert g.clamped["levels"] is True


def test_levels_off_is_exactly_the_old_path():
    img = _gradient_image()
    out, gains = normalize_float(img, NormalizeParams())
    ref = apply_gains_float(img, compute_gains(img, NormalizeParams()))
    assert np.array_equal(out, ref)
    assert gains.levels is None
    assert "levels" not in gains.clamped


def test_levels_with_white_balance_neutralises_a_cast_and_pins_white():
    """The camera path: grey-world first, then the stretch and gamma."""
    lin = np.linspace(0.05, 0.60, 48 * 48, dtype=np.float32).reshape(48, 48)
    cast = np.stack([lin * 1.15, lin, lin * 0.85], axis=-1)              # warm, flat
    srgb = color.linear_to_srgb(cast).astype(np.float32)
    out, gains = normalize_float(srgb, NormalizeParams(levels=True))
    assert gains.wb[0] < 1.0 < gains.wb[2] and gains.levels is not None
    assert np.allclose(out[..., 0], out[..., 1], atol=1.5 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1.5 / 255)
    lum = color.luminance(color.srgb_to_linear(out))
    assert np.percentile(lum, 99.5) == pytest.approx(1.0, abs=0.01)
    assert np.percentile(lum, 0.5) == pytest.approx(0.0, abs=0.01)


def test_levels_params_validate_and_round_trip():
    with pytest.raises(ValueError, match="levels_gamma"):
        NormalizeParams(
            white_balance=False, levels=True, levels_gamma_min=2.0, levels_gamma_max=1.0
        )
    with pytest.raises(ValueError, match="levels_low_pct"):
        NormalizeParams(
            white_balance=False, levels=True, levels_low_pct=99.0, levels_high_pct=1.0
        )
    p = NormalizeParams(white_balance=False, levels=True, levels_max_stretch=3.0)
    assert NormalizeParams.from_dict(p.to_dict()) == p
    assert "levels" in Gains(wb=np.ones(3), exposure=1.0, levels={"gamma": 0.7}).to_dict()


def test_float_and_u8_paths_agree_under_levels():
    """The Pi bakes WB, stretch and gamma into three tables; they must match the
    float reference the trainer used, or the LUT sees inputs it was not fit on."""
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams(levels=True)
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert gains_u8.levels is not None
    for key in ("low", "high", "stretch", "gamma"):
        assert gains_u8.levels[key] == pytest.approx(gains_f.levels[key], abs=1e-6), key
    # 4, not 1: on the luma path Y is quantised to 8 bits before the curve and
    # the two colour-space conversions round again. Measured max 4, mean 0.5.
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 4


def test_u8_subsampling_under_levels_stays_close_to_the_reference():
    """At 1080p statistics come from a strided subsample; percentiles are noisier
    than a median, so allow a little more than the exact-path test does."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.1, 0.7, 1920, dtype=np.float32)[None, :].repeat(1080, 0)
    noise = rng.normal(0, 0.01, (1080, 1920, 3))
    img = np.clip(np.stack([base * 1.05, base, base * 0.95], -1) + noise, 0, 1)
    img_u8 = (img * 255).round().astype(np.uint8)
    p = NormalizeParams(levels=True)
    out_f, gf = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gu = normalize_u8(img_u8, p)
    assert gu.levels["gamma"] == pytest.approx(gf.levels["gamma"], abs=0.03)
    diff = np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int))
    # Statistics from a stride-2 subsample plus the luma path's roundings:
    # measured mean 0.64, p99 3, max 7. Assert on all three, not the max alone.
    assert diff.mean() < 1.0 and np.percentile(diff, 99) <= 3 and diff.max() <= 8
def test_levels_reports_how_coloured_the_scan_black_is_without_correcting_it():
    """A per-channel black point was tried and made the fit worse than identity
    (see normalize.py). The spread is still reported so a coloured base shows
    up in the corpus report; the darkest tones keep their colour."""
    p = NormalizeParams(white_balance=False, levels=True)
    ramp = np.linspace(0.0, 0.6, 64 * 64, dtype=np.float32).reshape(64, 64)
    lin = np.stack([ramp + 0.010, ramp + 0.012, ramp + 0.030], axis=-1)   # blue-black base
    out, gains = normalize_float(color.linear_to_srgb(lin).astype(np.float32), p)
    assert gains.levels["black_rgb_spread"] == pytest.approx(0.020, abs=0.003)
    o = color.srgb_to_linear(out).reshape(-1, 3)
    dark = o[np.argsort(o @ np.array([0.2126, 0.7152, 0.0722]))[200:260]]
    assert dark[:, 2].mean() > dark[:, 0].mean(), "the shared black point must keep the base colour"


def test_levels_target_median_can_differ_from_the_exposure_target():
    """Slides are dense; gamma-lifting them to the camera's median taught the LUT
    to lift shadows. The target median is therefore its own parameter."""
    # 0.35 with the dark-weighted ramp keeps the gamma well inside the clamp,
    # so this exercises the target, not the clamp.
    p = NormalizeParams(white_balance=False, levels=True, levels_target_median=0.35)
    out, gains = normalize_float(_low_contrast_grey(power=2), p)
    lv = gains.levels
    assert gains.clamped["levels"] is False
    lum_in = color.luminance(color.srgb_to_linear(_low_contrast_grey(power=2)))
    stretched = np.clip((lum_in - lv["low"]) * lv["stretch"], 0.0, 1.0)
    m = (stretched >= p.stats_lum_min) & (stretched <= p.stats_lum_max)
    assert np.median(stretched[m]) ** lv["gamma"] == pytest.approx(0.35, abs=0.005)
    assert NormalizeParams.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="levels_target_median"):
        NormalizeParams(white_balance=False, levels=True, levels_target_median=1.5)


def test_luma_tone_keeps_chroma_where_per_channel_tone_would_not():
    """A skin-toned patch in a bright white room: the scene median asks for a
    strong gamma. Per channel, that gamma widens R-G-B gaps and saturates the
    skin; on luma only, chroma barely moves."""
    lin = np.full((64, 64, 3), 0.60, dtype=np.float32)              # bright walls
    lin[16:48, 16:48] = (0.35, 0.22, 0.17)                             # a face-sized patch
    lin[:6, :6] = 0.02                  # something dark, so the face is not the black point
    img = color.linear_to_srgb(lin).astype(np.float32)
    def patch_chroma(out):
        px = out[16:48, 16:48].reshape(-1, 3)
        return float(np.hypot(*color.srgb_to_oklab(px)[:, 1:].mean(0)))
    before = patch_chroma(img)
    out_l, g_l = normalize_float(img, NormalizeParams(white_balance=False, levels=True))
    out_c, g_c = normalize_float(img, NormalizeParams(white_balance=False, levels=True,
                                                      levels_tone="channel"))
    assert g_l.levels["gamma"] > 1.3 and g_l.levels["tone"] == "luma"
    assert abs(patch_chroma(out_l) / before - 1) < 0.10
    assert patch_chroma(out_c) / before > 1.20
    with pytest.raises(ValueError, match="levels_tone"):
        NormalizeParams(levels=True, levels_tone="magic")


# --- clipping-aware lift (Phase 6) --------------------------------------------

LIFT = NormalizeParams(white_balance=False, levels=True, levels_lift_highlight_ref=0.05)


def _scene(frac_ceiling: float, ceiling_value: float = 1.0, h: int = 40, w: int = 60,
           body_lo: float = 0.30, body_hi: float = 0.60):
    """A dark gradient body (sRGB 0.30-0.60) plus ``frac_ceiling`` of its pixels at
    ``ceiling_value``. With the body this dark the levels median lands below the
    exposure target, so raw_gamma < 1: a lift, the outdoor failure mode."""
    body = np.linspace(body_lo, body_hi, w, dtype=np.float32)
    img = np.repeat(body[None, :], h, axis=0)[..., None].repeat(3, axis=2).copy()
    n = int(round(frac_ceiling * h * w))
    flat = img.reshape(-1, 3)
    flat[:n] = ceiling_value
    return flat.reshape(h, w, 3)


def test_lift_ref_none_is_bit_for_bit_the_old_path_and_records_weight_one():
    img = _scene(0.10)
    old = compute_gains(img, NormalizeParams(white_balance=False, levels=True))
    assert old.levels["lift_weight"] == 1.0
    assert old.levels["gamma"] == pytest.approx(
        float(np.clip(old.levels["raw_gamma"], 0.5, 2.0))
    )
    for key in ("raw_gamma", "highlight_frac", "lift_weight"):
        assert key in old.levels


def test_a_dark_frame_with_nothing_at_the_ceiling_keeps_its_full_lift():
    # bright pixels at sRGB 0.977 (~0.95 linear) sit below the 0.97 ceiling line
    g = compute_gains(_scene(0.10, ceiling_value=0.977), LIFT)
    assert g.levels["raw_gamma"] < 1.0, "fixture must be a lift"
    assert g.levels["highlight_frac"] == 0.0
    assert g.levels["lift_weight"] == 1.0
    assert g.levels["gamma"] == pytest.approx(g.levels["raw_gamma"])


def test_a_frame_with_ref_or_more_at_the_ceiling_gets_no_lift():
    g = compute_gains(_scene(0.10), LIFT)  # 10 % at the ceiling, ref 5 %
    assert g.levels["raw_gamma"] < 1.0, "fixture must be a lift"
    assert g.levels["highlight_frac"] == pytest.approx(0.10, abs=0.005)
    assert g.levels["lift_weight"] == 0.0
    assert g.levels["gamma"] == pytest.approx(1.0)


def test_a_partial_ceiling_fraction_damps_the_lift_linearly():
    g = compute_gains(_scene(0.025), LIFT)  # half of ref
    assert g.levels["lift_weight"] == pytest.approx(0.5, abs=0.02)
    expected = 1.0 - (1.0 - g.levels["raw_gamma"]) * g.levels["lift_weight"]
    assert g.levels["gamma"] == pytest.approx(expected)


def test_darkening_is_never_damped():
    # A bright body (sRGB 0.75-0.95) puts the stretched median above the exposure
    # target, so raw_gamma > 1. A flat body would not work: the percentile stretch
    # maps a constant body to 0, collapsing the median into a lift instead.
    bright = _scene(0.10, body_lo=0.75, body_hi=0.95)
    g = compute_gains(bright, LIFT)
    assert g.levels["raw_gamma"] > 1.0, "fixture must be a darkening"
    assert g.levels["lift_weight"] == 1.0
    assert g.levels["gamma"] == pytest.approx(float(np.clip(g.levels["raw_gamma"], 0.5, 2.0)))


def test_highlight_fraction_counts_pixels_the_stats_mask_excludes():
    # The stats mask drops luminance > 0.90, so the ceiling pixels are invisible to
    # the median; highlight_frac must still see them or bright frames read as dark.
    g = compute_gains(_scene(0.10), LIFT)
    assert g.levels["highlight_frac"] == pytest.approx(0.10, abs=0.005)


def test_clamped_levels_reflects_the_damped_gamma():
    # Undamped raw_gamma is well below 1; damped to exactly 1.0 it is inside the
    # clamp, so the clamp flag must be False even though raw_gamma alone might clamp.
    g = compute_gains(_scene(0.10), NormalizeParams(
        white_balance=False, levels=True, levels_lift_highlight_ref=0.05,
        levels_gamma_min=0.95, levels_gamma_max=1.05,
    ))
    assert g.levels["gamma"] == pytest.approx(1.0)
    assert g.clamped["levels"] is False


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, float("nan")])
def test_lift_ref_validation_names_the_field(value):
    with pytest.raises(ValueError, match="levels_lift_highlight_ref"):
        NormalizeParams(levels_lift_highlight_ref=value)


def test_lift_ref_round_trips_and_is_optional():
    p = NormalizeParams(levels=True, levels_lift_highlight_ref=0.05)
    assert NormalizeParams.from_dict(p.to_dict()) == p
    assert NormalizeParams.from_dict({"levels": True}).levels_lift_highlight_ref is None


def test_a_second_pass_is_still_mild_with_the_lift_ref_set():
    # Damping deliberately stops short of the exposure target, so the frame the
    # second pass sees is further from target than an undamped one and the
    # residual gamma is correspondingly larger (0.88 here against 0.95 without
    # the lift ref). What must hold is that it stays bounded and strictly milder
    # than the first pass, i.e. the damping does not set up a runaway.
    img = _scene(0.02)
    once, once_gains = normalize_float(img, LIFT)
    twice_gains = compute_gains(once, LIFT)
    assert 0.85 <= twice_gains.levels["gamma"] <= 1.1
    assert abs(1.0 - twice_gains.levels["gamma"]) < abs(1.0 - once_gains.levels["gamma"])


def test_u8_and_float_paths_agree_on_the_damped_gamma():
    img = _scene(0.03)
    _, g_float = normalize_float(img, LIFT)
    _, g_u8 = normalize_u8((img * 255).round().astype(np.uint8), LIFT)
    assert g_u8.levels["gamma"] == pytest.approx(g_float.levels["gamma"], abs=0.03)
    assert g_u8.levels["lift_weight"] == pytest.approx(g_float.levels["lift_weight"], abs=0.05)


@pytest.mark.parametrize("kwargs", [
    {"levels": False},
    {"levels": True, "levels_tone": "luma"},
    {"levels": True, "levels_tone": "channel"},
    {"white_balance": False, "levels": True},
])
@pytest.mark.parametrize("shape", [(48, 64), (700, 900)])
def test_stats_gains_are_the_gains_inside_normalize_u8(kwargs, shape):
    """The GPU pass takes its tables from these gains, so they must be the ones
    the CPU path would have applied, from the same strided subsample."""
    from pifilm.normalize import stats_gains

    p = NormalizeParams(**kwargs)
    img_u8 = (_gradient_image(*shape) * 255).round().astype(np.uint8)
    out, inside = normalize_u8(img_u8, p)
    alone = stats_gains(img_u8, p)
    assert np.array_equal(alone.combined, inside.combined)
    assert np.array_equal(alone.wb, inside.wb) and alone.exposure == inside.exposure
    assert alone.levels == inside.levels and alone.clamped == inside.clamped

    # normalize_u8 as it was before the statistics were factored out.
    h, w = shape
    step = max(1, int(np.ceil(np.sqrt(h * w / 300_000))))
    old = compute_gains(img_u8[::step, ::step].astype(np.float32) / 255.0, p)
    assert np.array_equal(old.combined, inside.combined) and old.levels == inside.levels
    assert step == (1 if shape == (48, 64) else 2)


def test_stats_gains_honours_max_stats_pixels():
    from pifilm.normalize import stats_gains

    p = NormalizeParams(levels=True)
    img_u8 = np.random.default_rng(4).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    _, inside = normalize_u8(img_u8, p, max_stats_pixels=1000)
    assert stats_gains(img_u8, p, max_stats_pixels=1000).levels == inside.levels
    assert stats_gains(img_u8, p).levels != inside.levels
