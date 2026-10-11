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
