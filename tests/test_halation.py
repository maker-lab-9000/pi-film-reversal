import math

import numpy as np
import pytest

from pifilm.halation import (
    FALLOFF,
    HalationParams,
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
    assert reduction((3040, 4056), ON) == 7          # 0.25 mm of 36 at 4056 px: 28.2 px
    assert reduction((760, 1014), ON) == 1
    assert reduction((2, 2), ON) == 1


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
    p = HalationParams(enabled=True, sigma_mm=1.0)     # k 5, reduced 160 x 120, 4 sigma 18 px
    glow = prepare(frame, p).glow
    want = _direct_glow(frame, p)
    assert glow.shape == want.shape
    # Measured: 0.3 % of the glow's peak above the background at most. A few percent
    # would still be invisible after the mix; 1 % pins that it is the same fall-off.
    excess = want - want.min()
    assert np.abs(glow - want).max() <= 0.01 * excess.max()
