import numpy as np
import pytest

from pifilm.grain import GrainParams, add_grain


def test_disabled_is_identity():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = add_grain(img, GrainParams(enabled=False))
    assert np.array_equal(out, img)
    assert out is not img


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"strength": -0.1}, "strength"),
        ({"blur_sigma": -1.0}, "blur_sigma"),
        ({"strength": float("inf")}, "strength"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        GrainParams(**kwargs)


def test_preserves_mean_luminance_and_adds_no_colour_bias():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    out = add_grain(img, GrainParams(strength=0.05), rng=np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), 128, atol=0.5)
    assert out.std() > 5


def test_strength_scales_noise():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    lo = add_grain(img, GrainParams(strength=0.02), rng=np.random.default_rng(2))
    hi = add_grain(img, GrainParams(strength=0.06), rng=np.random.default_rng(2))
    assert lo[..., 1].std() == pytest.approx(0.02 * 255, rel=0.25)
    assert hi[..., 1].std() == pytest.approx(0.06 * 255, rel=0.25)


def test_black_and_white_are_untouched():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[16:] = 255
    out = add_grain(img, GrainParams(strength=0.1), rng=np.random.default_rng(3))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_seeded_is_reproducible():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    a = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    b = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_params_dict_roundtrip():
    p = GrainParams(strength=0.03, blur_sigma=0.5, enabled=False)
    assert GrainParams.from_dict({**p.to_dict(), "extra": 1}) == p


# --- two-scale model (measured from HP5 Plus 400 scans) --------------------


def _two_scale(**kw):
    return GrainParams(model="two_scale", **{"strength": 0.025, **kw})


def _luma(rgb):
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def test_two_scale_strength_is_the_sd_at_mid_grey():
    img = np.full((1000, 1500, 3), 128, dtype=np.uint8)
    out = add_grain(img, _two_scale(), rng=np.random.default_rng(1))
    assert _luma(out).std() / 255 == pytest.approx(0.025, rel=0.12)


def test_two_scale_follows_the_measured_envelope():
    """HP5 grain is about a quarter as strong in deep shadow as at mid-grey."""
    rng = np.random.default_rng(2)
    shadow = add_grain(np.full((800, 1200, 3), 15, np.uint8), _two_scale(strength=0.05), rng)
    mid = add_grain(np.full((800, 1200, 3), 128, np.uint8), _two_scale(strength=0.05), rng)
    ratio = _luma(shadow).std() / _luma(mid).std()
    assert 0.15 < ratio < 0.45


def test_two_scale_leaves_black_and_white_untouched():
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[32:] = 255
    out = add_grain(img, _two_scale(strength=0.1), rng=np.random.default_rng(3))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_two_scale_clumps_scale_with_the_frame_not_the_pixel():
    """Sizes are millimetres of a 36 mm frame: a smaller frame gets finer pixels'
    worth of clump, so neighbouring pixels are less alike."""
    def lag1(width):
        img = np.full((width * 2 // 3, width, 3), 128, np.uint8)
        y = _luma(add_grain(img, _two_scale(strength=0.05), rng=np.random.default_rng(4)))
        y -= y.mean()
        return (y[:, 1:] * y[:, :-1]).mean() / (y * y).mean()
    assert lag1(4056) > lag1(1014) + 0.1


def test_two_scale_adds_no_colour():
    img = np.full((300, 450, 3), (150, 110, 90), dtype=np.uint8)
    out = add_grain(img, _two_scale(strength=0.04), rng=np.random.default_rng(5))
    import cv2
    a = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    b = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb).astype(int)
    assert np.abs(b[..., 1:] - a[..., 1:]).mean() < 1.0


def test_two_scale_is_reproducible_and_roundtrips():
    img = np.full((120, 180, 3), 100, dtype=np.uint8)
    p = _two_scale()
    a = add_grain(img, p, rng=np.random.default_rng(7))
    b = add_grain(img, GrainParams.from_dict(p.to_dict()), rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert GrainParams.from_dict(p.to_dict()) == p


def test_gaussian_to_dict_is_unchanged_so_old_artifacts_stay_byte_identical():
    assert GrainParams().to_dict() == {"strength": 0.004, "blur_sigma": 0.7, "enabled": True}


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"model": "speckle"}, "model"),
        ({"model": "two_scale", "fine_share": 1.5}, "fine_share"),
        ({"model": "two_scale", "fine_sigma_mm": -0.001}, "fine_sigma_mm"),
        ({"model": "two_scale", "envelope_luma": (0.5, 0.2), "envelope_rel": (1, 1)},
         "envelope_luma"),
        ({"model": "two_scale", "envelope_luma": (0.2, 0.5), "envelope_rel": (1,)},
         "envelope_rel"),
    ],
)
def test_two_scale_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        GrainParams(**kwargs)


def _reference_add_grain(rgb_u8, params, rng):
    """``add_grain`` as it was before the 2026-10-13 speed-up, kept as the
    reference the faster version must match byte for byte."""
    import cv2

    from pifilm.grain import _two_scale_noise

    ycc = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma = ycc[..., 0] / 255.0
    if params.model == "two_scale":
        noise = _two_scale_noise(luma.shape, params, rng)
        envelope = np.interp(luma, params.envelope_luma, params.envelope_rel).astype(np.float32)
    else:
        noise = rng.standard_normal(luma.shape, dtype=np.float32)
        if params.blur_sigma > 0:
            noise = cv2.GaussianBlur(noise, (0, 0), params.blur_sigma)
            noise /= max(float(noise.std()), 1e-6)
        envelope = 4.0 * luma * (1.0 - luma)
    ycc[..., 0] = np.clip(luma + params.strength * envelope * noise, 0.0, 1.0) * 255.0
    return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


@pytest.mark.parametrize("params", [
    GrainParams(),
    GrainParams(strength=0.05),
    GrainParams(strength=0.3, blur_sigma=0.0),
    GrainParams(model="two_scale", strength=0.02),
    GrainParams(model="two_scale", strength=0.4),
    GrainParams(model="two_scale", strength=0.05, fine_share=1.0),
])
def test_add_grain_matches_the_reference_byte_for_byte(params):
    """Old records regrade with the same seed, so the pixels must not move."""
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (301, 402, 3), dtype=np.uint8)
    img[:40] = 0
    img[40:80] = 255
    img[80:120] = np.linspace(0, 255, 402).astype(np.uint8)[None, :, None]
    for seed in (0, 1, 2):
        want = _reference_add_grain(img, params, np.random.default_rng(seed))
        got = add_grain(img, params, rng=np.random.default_rng(seed))
        assert np.array_equal(got, want)


def _tile_params(**kw):
    return GrainParams(model="two_scale_tile", **{"strength": 0.025, "tile_px": 256, **kw})


def test_tile_grain_is_the_tile_placed_by_the_seed():
    """The definition: round(y + 255 * strength * envelope(y) * tile), saturated."""
    import cv2

    from pifilm.grain import _oriented, grain_tile, place_tile

    params = _tile_params(strength=0.2)
    # Grey, so the result's luminance survives the trip back through RGB.
    grey = np.random.default_rng(3).integers(0, 256, (300, 610, 1), dtype=np.uint8)
    img = np.repeat(grey, 3, axis=2)
    got = add_grain(img, params, rng=np.random.default_rng(9))

    rng = np.random.default_rng(9)
    oy, ox = (int(v) for v in rng.integers(0, 256, 2))
    tile = _oriented(grain_tile(params, 610 / 36.0), int(rng.integers(0, 8)))
    noise = place_tile(tile, (300, 610), oy, ox)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0].astype(np.float64)
    envelope = np.interp(y / 255.0, params.envelope_luma, params.envelope_rel)
    want_y = np.clip(np.rint(y + 255.0 * 0.2 * envelope * noise), 0, 255)
    got_y = cv2.cvtColor(got, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float64)
    # Float32 against float64, and the trip back through RGB: one level.
    assert np.abs(got_y - want_y).max() <= 1
    assert np.mean(got_y != want_y) < 0.02


def test_tile_grain_is_reproducible_and_differs_between_seeds():
    img = np.full((200, 300, 3), 128, np.uint8)
    a = add_grain(img, _tile_params(), rng=np.random.default_rng(7))
    b = add_grain(img, _tile_params(), rng=np.random.default_rng(7))
    c = add_grain(img, _tile_params(), rng=np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_tile_grain_leaves_black_white_and_colour_alone():
    img = np.zeros((64, 64, 3), np.uint8)
    img[32:] = 255
    assert np.array_equal(add_grain(img, _tile_params(strength=0.3),
                                    rng=np.random.default_rng(1)), img)
    grey = add_grain(np.full((128, 128, 3), 128, np.uint8), _tile_params(strength=0.1),
                     rng=np.random.default_rng(1))
    assert np.abs(grey[..., 0].astype(int) - grey[..., 2]).max() <= 1


def test_tile_grain_has_the_strength_of_the_untiled_model():
    img = np.full((600, 900, 3), 128, np.uint8)
    tiled = _luma(add_grain(img, _tile_params(strength=0.05), rng=np.random.default_rng(2)))
    plain = _luma(add_grain(img, _two_scale(strength=0.05), rng=np.random.default_rng(2)))
    assert tiled.std() == pytest.approx(plain.std(), rel=0.05)


def test_the_tile_wraps_without_a_seam():
    """Neighbours across the wrap are as alike as neighbours inside the tile."""
    from pifilm.grain import grain_tile

    tile = grain_tile(_tile_params(), 4056 / 36.0)
    assert tile.shape == (256, 256) and not tile.flags.writeable
    assert float(tile.std()) == pytest.approx(1.0, abs=1e-3)
    inside = float(np.mean(tile[:, :-1] * tile[:, 1:]))
    for across in (np.mean(tile[:, -1] * tile[:, 0]), np.mean(tile[-1] * tile[0])):
        assert float(across) == pytest.approx(inside, abs=0.15)
    assert inside > 0.3


def test_place_tile_repeats_from_the_offset():
    from pifilm.grain import place_tile

    tile = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = place_tile(tile, (7, 9), 2, 3)
    rows, cols = np.indices((7, 9))
    assert np.array_equal(out, tile[(rows + 2) % 3, (cols + 3) % 4])


def test_all_eight_orientations_are_distinct():
    from pifilm.grain import _oriented

    tile = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert len({_oriented(tile, k).tobytes() for k in range(8)}) == 8


@pytest.mark.parametrize("bad", [0, 63, 8193, 2048.0, True])
def test_tile_px_is_validated(bad):
    with pytest.raises(ValueError, match="tile_px"):
        GrainParams(model="two_scale_tile", tile_px=bad)


def test_only_the_tile_model_writes_tile_px():
    assert "tile_px" not in _two_scale().to_dict()
    assert "tile_px" not in GrainParams().to_dict()
    p = _tile_params(tile_px=512)
    assert GrainParams.from_dict(p.to_dict()) == p
