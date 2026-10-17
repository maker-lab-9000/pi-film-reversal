import math

import numpy as np
import pytest

from pifilm.halation import (
    FALLOFF,
    HalationParams,
    HalationPass,
    apply_cpu,
    expand_table,
    prepare,
    reduction,
)

ON = HalationParams(enabled=True)


def test_defaults_are_the_chosen_values_and_off():
    p = HalationParams()
    assert (p.enabled, p.model, p.strength, p.sigma_mm, p.headroom, p.weights,
            p.frame_width_mm) == (False, "mix", 0.15, 0.25, 0.5, (1.0, 0.35, 0.0), 36.0)
    assert not p.active and ON.active
    assert not HalationParams(enabled=True, strength=0.0).active


@pytest.mark.parametrize("bad", [
    {"strength": 1.5}, {"strength": -0.1}, {"sigma_mm": 0}, {"sigma_mm": 5.1},
    {"headroom": 0.95}, {"headroom": -0.1}, {"weights": [1, 0.35]}, {"weights": [1, 2, 0]},
    {"frame_width_mm": 0}, {"strength": float("nan")}, {"strength": True},
    {"model": "threshold"}, {"enabled": "yes"},
])
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError):
        HalationParams.from_dict({**ON.to_dict(), **bad})


def test_round_trip_and_sha1():
    d = ON.to_dict()
    assert d == {"enabled": True, "model": "mix", "strength": 0.15, "sigma_mm": 0.25,
                 "headroom": 0.5, "weights": [1.0, 0.35, 0.0], "frame_width_mm": 36.0}
    assert HalationParams.from_dict(d) == ON
    assert len(ON.sha1) == 40 and ON.sha1 != HalationParams(enabled=True, strength=0.2).sha1


def test_the_falloff_is_fixed():
    assert FALLOFF == ((1.0, 0.5), (2.0, 0.3), (4.0, 0.2))
    assert math.isclose(sum(w for _, w in FALLOFF), 1.0)


def test_reduction_factor():
    assert reduction((3040, 4056), ON) == 14         # 0.25 mm of 36 at 4056 px: 28.2 px
    assert reduction((760, 1014), ON) == 3           # the quick grade: 7.0 px
    assert reduction((2, 2), ON) == 1


def _spot(shape, size):
    frame = np.full((*shape, 3), 20, np.uint8)
    cy, cx = shape[0] // 2, shape[1] // 2
    frame[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = 255
    return frame


@pytest.mark.parametrize("shape, size", [((760, 1014), 4), ((760, 1014), 10),
                                         ((1520, 2028), 6)])
def test_the_reduced_glow_is_the_glow_made_at_full_size(shape, size):
    """Two reduced pixels of sigma (k 3 on the quick frame, 7 at half size; 14 at 12 MP,
    measured 4.9 to 5.2 %, too slow to blur directly here) against the fall-off as
    three direct Gaussians on the full frame (k 1, no further copies)."""
    from pifilm._cv2 import require_cv2
    from pifilm.halation import glow_at, sigma_px
    cv2 = require_cv2()
    frame = _spot(shape, size)
    x = expand_table(ON.headroom)[frame]
    s = sigma_px(frame.shape, ON)
    want = sum(np.float32(w) * cv2.GaussianBlur(x, (0, 0), s * m, borderType=cv2.BORDER_REFLECT)
               for m, w in FALLOFF)
    hp = prepare(frame, ON)
    assert reduction(frame.shape, ON) > 1 and hp.glow.shape[0] < shape[0]
    got = glow_at(hp.glow, 0, shape[0], shape[1], *shape).transpose(1, 2, 0)
    # Measured (2026-10-17): 4.9 to 5.7 % of the glow's peak above the background, at
    # the spot's edge, where the narrowest term is two reduced pixels wide; 7 % pins it.
    assert np.abs(got - want).max() <= 0.07 * (want - want.min()).max()
    # After the mix that is at most a level at the chosen strength, two at 0.5.
    for strength, levels in ((0.15, 1), (0.5, 2)):
        p = HalationParams(enabled=True, strength=strength)
        a = apply_cpu(frame, prepare(frame, p)).astype(int)
        direct = HalationPass(glow=want, headroom=p.headroom, mix=prepare(frame, p).mix)
        assert np.abs(a - apply_cpu(frame, direct).astype(int)).max() <= levels


def test_expand_table_round_trips_through_compress():
    for c in (0.0, 0.5, 0.9):
        x = expand_table(c)
        lin = x / (1.0 + c * x)
        codes = np.arange(256) / 255.0
        want = np.where(codes <= 0.04045, codes / 12.92, ((codes + 0.055) / 1.055) ** 2.4)
        assert np.allclose(lin, want, atol=1e-6)
    assert math.isclose(float(expand_table(0.5)[255]), 2.0, rel_tol=1e-6)


def test_a_flat_frame_is_unchanged():
    for value in (0, 40, 128, 230, 255):
        flat = np.full((60, 80, 3), value, np.uint8)
        assert np.array_equal(apply_cpu(flat, prepare(flat, ON)), flat)


def test_a_bright_spot_grows_a_red_halo_larger_than_green_and_none_in_blue():
    frame = np.full((200, 200, 3), 20, np.uint8)
    frame[95:105, 95:105] = 255
    out = apply_cpu(frame, prepare(frame, HalationParams(enabled=True, strength=0.5)))
    ring = out[100, 112:130].astype(int) - 20          # just outside the spot
    assert (ring[:, 0] > 0).any()
    assert ring[:, 0].sum() > ring[:, 1].sum() > 0
    assert (out[..., 2] == frame[..., 2]).all()          # blue weight is 0


def test_the_glow_is_the_same_share_of_the_frame_at_two_sizes():
    def spot(size):
        f = np.full((size, size, 3), 20, np.uint8)
        s = size // 20
        f[size // 2 - s:size // 2 + s, size // 2 - s:size // 2 + s] = 255
        return f
    small, big = spot(200), spot(800)
    a = apply_cpu(small, prepare(small, ON))[100, :, 0].astype(float)
    # Small pixel i covers big pixels 4i..4i+3: compare with their mean, as a single big
    # column is 1.5 big pixels off the small pixel's centre (8 levels on the glow's edge).
    b = apply_cpu(big, prepare(big, ON))[400:404, :, 0].astype(float)
    b = b.reshape(4, 200, 4).mean(axis=(0, 2))
    assert np.abs(a - b).max() <= 3


@pytest.mark.parametrize("shape", [(2, 2), (1, 50), (37, 53), (3040 // 8, 4056 // 8)])
def test_small_and_odd_frames(shape):
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    hp = prepare(frame, ON)
    assert hp.glow.dtype == np.float32 and hp.glow.ndim == 3 and hp.glow.shape[2] == 3
    assert np.isfinite(hp.glow).all()
    out = apply_cpu(frame, hp)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_odd_size_with_reduction_grades_the_last_row_and_column():
    frame = np.full((3041, 4057, 3), 30, np.uint8)
    frame[-6:, -6:] = 255                                  # bright corner
    out = apply_cpu(frame, prepare(frame, ON))
    assert out[-1, -20, 0] > 30 or out[-20, -1, 0] > 30   # the corner's halo reaches the edge
    assert (out[0, 0] == 30).all()                         # far away: unchanged


def test_clipped_pixels_at_the_largest_headroom():
    frame = np.full((64, 64, 3), 255, np.uint8)
    frame[:, :32] = 0
    p = HalationParams(enabled=True, strength=1.0, headroom=0.9)
    out = apply_cpu(frame, prepare(frame, p))
    assert out.dtype == np.uint8
    assert (out[..., 2] == frame[..., 2]).all()
    assert out[:, :28, 0].max() > 0                        # glow spills into the dark half


def _direct_glow(frame, params):
    """The fall-off as three direct Gaussians on the reduced copy: the reference."""
    from pifilm._cv2 import require_cv2
    from pifilm.halation import sigma_px
    cv2 = require_cv2()
    height, width = frame.shape[:2]
    x = expand_table(params.headroom)[frame]
    k = reduction(frame.shape, params)
    small = cv2.resize(x, (width // k, height // k), interpolation=cv2.INTER_AREA)
    sigma = sigma_px(frame.shape, params) / k
    return sum(np.float32(w) * cv2.GaussianBlur(small, (0, 0), sigma * m,
                                                borderType=cv2.BORDER_REFLECT)
               for m, w in FALLOFF)


def test_the_wide_blurs_on_smaller_copies_are_the_same_falloff():
    frame = np.full((600, 800, 3), 20, np.uint8)
    frame[295:305, 395:405] = 255
    p = HalationParams(enabled=True, sigma_mm=1.0)     # k 11, reduced 72 x 54, 4 sigma 8 px
    glow = prepare(frame, p).glow
    want = _direct_glow(frame, p)
    assert glow.shape == want.shape
    # Measured: 1.0 % of the glow's peak above the background at most (0.9 to 1.7 % on
    # the quick frame and at 12 MP; 0.3 % at four reduced pixels of sigma). A few percent
    # would still be invisible after the mix; 2 % pins that it is the same fall-off.
    excess = want - want.min()
    assert np.abs(glow - want).max() <= 0.02 * excess.max()


def test_the_srgb_encode_table_is_within_half_a_level_of_the_curve():
    """``apply_cpu`` encodes through a table rather than a ``pow`` per value. Read at
    the nearest of its entries it must stay within half a level of the curve
    everywhere, near 0 too, where the curve is steepest (3,295 levels per unit)."""
    from pifilm.color import linear_to_srgb
    from pifilm.halation import _ENCODE, _ENCODE_U8, _encode_index, encode_u8
    lin = np.concatenate([np.linspace(0.0, 1.0, 2_000_001, dtype=np.float32),
                          np.geomspace(1e-7, 0.01, 200_001).astype(np.float32),
                          np.float32([0.0031308, 0.04045 / 12.92, 1.0])])
    index = _encode_index(lin.copy())
    error = np.abs(_ENCODE[index] - linear_to_srgb(lin) * np.float32(255.0))
    assert error.max() <= 0.5
    assert error.max() < 0.05                           # measured 0.025: half a step
    assert np.array_equal(_ENCODE_U8, np.rint(_ENCODE).astype(np.uint8))
    assert np.array_equal(encode_u8(lin.copy()), _ENCODE_U8[index])
    # Just above 1 (float rounding in the compress) is white, not an overflow.
    assert encode_u8(np.float32([1.0000001, 1.01]))[0] == 255


@pytest.mark.parametrize("weights", [(1.0, 0.35, 0.0), (0.2, 0.0, 0.7)])
def test_apply_cpu_in_bands_is_the_whole_frame_at_once(weights):
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (101, 77, 3), dtype=np.uint8)
    frame[40:60, 30:50] = 255
    p = HalationParams(enabled=True, strength=0.6, sigma_mm=2.0, weights=weights)
    hp = prepare(frame, p)
    assert hp.glow.shape[:2] != frame.shape[:2]
    whole = apply_cpu(frame, hp, band_pixels=frame.shape[0] * frame.shape[1])
    for band_pixels in (77, 77 * 7, 1000, 77 * 50):   # 1, 7, 12 and 50 rows: odd bands
        assert apply_cpu(frame, hp, band_pixels=band_pixels).tobytes() == whole.tobytes()
    assert apply_cpu(frame, hp).tobytes() == whole.tobytes()


def test_apply_cpu_reads_the_glow_bilinearly_at_full_size():
    """The banded read and the encode table against ``cv2.resize``'s ``INTER_LINEAR``
    and ``linear_to_srgb``, the CPU's step before bands: a level apart at most, on a
    value at a rounding boundary (measured 0.15 % of the values)."""
    from pifilm._cv2 import require_cv2
    from pifilm.color import linear_to_srgb
    cv2 = require_cv2()
    frame = _spot((300, 400), 12)
    hp = prepare(frame, HalationParams(enabled=True, strength=0.5, sigma_mm=1.0))
    up = cv2.resize(hp.glow, (400, 300), interpolation=cv2.INTER_LINEAR)
    x = expand_table(hp.headroom)[frame]
    x += hp.mix * (up - x)
    want = np.rint(linear_to_srgb(x / (1.0 + np.float32(hp.headroom) * x)) * 255.0)
    diff = np.abs(apply_cpu(frame, hp).astype(int) - want.astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 0.005
