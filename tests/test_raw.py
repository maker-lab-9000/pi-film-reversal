import dataclasses

import numpy as np
import pytest

from pifilm._cv2 import require_cv2
from pifilm.raw import (
    TONE_CURVES,
    RawError,
    RawFrame,
    RawTuning,
    alsc_at,
    ccm_at,
    develop,
    read_dng,
    tuning_tables,
)
from tests.raw_fakes import LONG, RATIONAL, SHORT, make_dng, small_tuning

PATTERNS = ["BGGR", "RGGB", "GRBG", "GBRG"]


def _bayer(seed=0, shape=(6, 8), high=4096):
    return np.random.default_rng(seed).integers(0, high, size=shape, dtype=np.uint16)


# --- read_dng ------------------------------------------------------------


@pytest.mark.parametrize("bits", [12, 16])
@pytest.mark.parametrize("cfa", PATTERNS)
def test_round_trip(bits, cfa):
    bayer = _bayer()
    frame = read_dng(make_dng(bayer, bits=bits, cfa=cfa))
    assert isinstance(frame, RawFrame)
    assert frame.bayer.dtype == np.uint16
    assert frame.bayer.shape == bayer.shape
    np.testing.assert_array_equal(frame.bayer, bayer)
    assert frame.cfa == cfa
    assert frame.black == 256.0
    assert frame.white == 4095.0
    assert frame.as_shot_neutral == pytest.approx((0.3186, 1.0, 0.6277))


def test_without_as_shot_neutral():
    frame = read_dng(make_dng(_bayer(), as_shot_neutral=None))
    assert frame.as_shot_neutral is None


def test_twelve_bit_order():
    data = make_dng(np.zeros((1, 2)), pixel_bytes=bytes([0xAB, 0xCD, 0xEF]), white=4095)
    frame = read_dng(data)
    assert frame.bayer.tolist() == [[0xABC, 0xDEF]]


def test_one_black_level_value():
    frame = read_dng(make_dng(_bayer(), black=(64,)))
    assert frame.black == 64.0


def test_rational_black_level():
    frame = read_dng(make_dng(_bayer(), tags={50714: (RATIONAL, [256.0] * 4)}))
    assert frame.black == 256.0


def _refused(match, bayer=None, **kwargs):
    with pytest.raises(RawError, match=match):
        read_dng(make_dng(_bayer() if bayer is None else bayer, **kwargs))


def test_refuses_compressed():
    _refused("compression 7", tags={259: (SHORT, [7])})


def test_refuses_tiled():
    _refused("tile", tags={322: (LONG, [16]), 323: (LONG, [16])})


def test_refuses_several_strips():
    _refused("2 strips", strips=2)


def test_refuses_wrong_byte_count():
    _refused("71 bytes", tags={279: (LONG, [71])})


def test_refuses_big_endian():
    _refused("big-endian", big_endian=True)


def test_refuses_not_tiff():
    with pytest.raises(RawError, match="not a TIFF"):
        read_dng(b"\x89PNG\r\n\x1a\n" + bytes(64))


def test_refuses_truncated():
    data = make_dng(_bayer())
    with pytest.raises(RawError, match="IFD of .* entries runs past the end"):
        read_dng(data[:20])


def test_refuses_strip_past_end():
    _refused("strip at 1000000 .* runs past the end", tags={273: (LONG, [10**6])})


def test_refuses_tag_data_past_end():
    # The black levels and AsShotNeutral are stored after the strip, at the end.
    data = make_dng(_bayer())
    with pytest.raises(RawError, match="tag .* runs past the end"):
        read_dng(data[:-4])


def test_refuses_odd_width_at_12_bits():
    _refused("width 3 is odd", bayer=_bayer(shape=(2, 3)))


def test_refuses_nan_white_level():
    _refused("white level", tags={50717: (RATIONAL, [float("nan")])})


def test_refuses_black_at_or_above_white():
    _refused("black level 300 .* white level 200", black=(300,), white=200, bayer=_bayer(high=200))


@pytest.mark.parametrize("tag, name", [(33421, "CFARepeatPatternDim"), (33422, "CFAPattern")])
def test_refuses_missing_cfa_tags(tag, name):
    _refused(name, tags={tag: None})


def test_refuses_other_cfa_repeat():
    _refused("CFARepeatPatternDim", tags={33421: (SHORT, [4, 4])})


def test_refuses_unknown_cfa_pattern():
    _refused("CFA pattern", cfa="RGBG")


def test_refuses_unequal_black_levels():
    _refused("unequal black levels", black=(256, 256, 257, 256))


def test_refuses_value_above_white():
    bayer = _bayer()
    bayer[2, 3] = 4000
    _refused("above the white level", bayer=bayer, white=3000)


def test_refuses_other_bit_depth():
    _refused("14 bits", tags={258: (SHORT, [14])}, bits=16)


def test_refuses_not_cfa():
    _refused("photometric 34892", tags={262: (SHORT, [34892])})


def test_refuses_several_samples():
    _refused("samples per pixel 3", tags={277: (SHORT, [3])})


def test_refuses_missing_white_level():
    _refused("WhiteLevel", tags={50717: None})


# --- tuning_tables -------------------------------------------------------


def test_tuning_tables():
    t = tuning_tables(small_tuning())
    assert t.black_level == 4096.0
    assert [ct for ct, _ in t.ccms] == [3000.0, 5000.0]
    np.testing.assert_array_equal(t.ccms[0][1], np.eye(3))
    assert t.ccms[1][1].shape == (3, 3)
    assert t.ccms[1][1][0].tolist() == [2.0, -0.5, -0.5]
    assert [ct for ct, _ in t.alsc_cr] == [3000.0, 6000.0]
    assert t.alsc_cr[1][1].shape == (12, 16)
    assert t.alsc_cb[0][1][0, 0] == 3.0
    lut = np.array([1.0 + i / 191 for i in range(192)]).reshape(12, 16)
    np.testing.assert_allclose(t.luminance, 1 + (lut - 1) * 0.5)
    # Row-major, 16 columns: the second entry is row 0, column 1.
    assert t.luminance[0, 1] == pytest.approx(1 + 0.5 / 191)
    assert t.luminance[1, 0] == pytest.approx(1 + 0.5 * 16 / 191)
    assert t.gamma_curve.shape == (33, 2)
    assert t.gamma_curve[0].tolist() == [0.0, 0.0]
    assert t.gamma_curve[-1].tolist() == [1.0, 1.0]
    assert t.gamma_curve[1].tolist() == pytest.approx([2048 / 65535, 4096 / 65535])


def test_luminance_strength_defaults_to_one():
    alsc = small_tuning()["algorithms"]
    block = next(a["rpi.alsc"] for a in alsc if "rpi.alsc" in a)
    del block["luminance_strength"]
    tuning = {"version": 2.0, "algorithms": alsc}
    t = tuning_tables(tuning)
    np.testing.assert_allclose(t.luminance.ravel(), block["luminance_lut"])


def test_tuning_version_one_layout():
    v2 = small_tuning()
    v1 = {name: block for algo in v2["algorithms"] for name, block in algo.items()}
    t = tuning_tables(v1)
    assert t.black_level == 4096.0


@pytest.mark.parametrize("block", ["rpi.ccm", "rpi.alsc", "rpi.black_level", "rpi.contrast"])
def test_tuning_refuses_missing_block(block):
    with pytest.raises(RawError, match=block):
        tuning_tables(small_tuning({block: None}))


def _alsc(**changes):
    block = small_tuning()["algorithms"]
    alsc = dict(next(a["rpi.alsc"] for a in block if "rpi.alsc" in a))
    alsc.update(changes)
    return small_tuning({"rpi.alsc": alsc})


def test_tuning_refuses_short_alsc_table():
    bad = _alsc(calibrations_Cr=[{"ct": 3000, "table": [1.0] * 191}])
    with pytest.raises(RawError, match="191"):
        tuning_tables(bad)


def test_tuning_refuses_short_luminance_lut():
    with pytest.raises(RawError, match="luminance_lut"):
        tuning_tables(_alsc(luminance_lut=[1.0] * 12))


def test_tuning_refuses_short_ccm():
    bad = small_tuning({"rpi.ccm": {"ccms": [{"ct": 3000, "ccm": [1.0] * 8}]}})
    with pytest.raises(RawError, match="8"):
        tuning_tables(bad)


def test_tuning_refuses_empty_ccms():
    with pytest.raises(RawError, match="rpi.ccm"):
        tuning_tables(small_tuning({"rpi.ccm": {"ccms": []}}))


def test_tuning_refuses_repeated_ct():
    ccm = [1.0, 0, 0, 0, 1, 0, 0, 0, 1]
    bad = small_tuning({"rpi.ccm": {"ccms": [{"ct": 3000, "ccm": ccm}, {"ct": 3000, "ccm": ccm}]}})
    with pytest.raises(RawError, match="3000"):
        tuning_tables(bad)


def test_tuning_refuses_odd_gamma_curve():
    bad = small_tuning({"rpi.contrast": {"gamma_curve": [0, 0, 65535]}})
    with pytest.raises(RawError, match="gamma_curve"):
        tuning_tables(bad)


def test_tuning_refuses_unordered_gamma_curve():
    bad = small_tuning({"rpi.contrast": {"gamma_curve": [0, 0, 40000, 1, 30000, 2, 65535, 65535]}})
    with pytest.raises(RawError, match="gamma_curve"):
        tuning_tables(bad)


def test_tuning_tables_are_read_only():
    t = tuning_tables(small_tuning())
    with pytest.raises(ValueError):
        t.luminance[0, 0] = 5.0


# --- ccm_at, alsc_at -----------------------------------------------------


def test_ccm_at_calibrated_ct():
    t = tuning_tables(small_tuning())
    np.testing.assert_array_equal(ccm_at(t, 3000), np.eye(3))
    np.testing.assert_array_equal(ccm_at(t, 5000), t.ccms[1][1])


def test_ccm_at_midpoint():
    t = tuning_tables(small_tuning())
    np.testing.assert_allclose(ccm_at(t, 4000), (np.eye(3) + t.ccms[1][1]) / 2)


def test_ccm_at_clamps():
    t = tuning_tables(small_tuning())
    np.testing.assert_array_equal(ccm_at(t, 1500), np.eye(3))
    np.testing.assert_array_equal(ccm_at(t, 12000), t.ccms[1][1])


def test_ccm_at_returns_a_copy():
    t = tuning_tables(small_tuning())
    m = ccm_at(t, 3000)
    m[0, 0] = 9.0
    assert ccm_at(t, 3000)[0, 0] == 1.0


def test_alsc_at():
    t = tuning_tables(small_tuning())
    cr, cb = alsc_at(t, 3000)
    assert cr.shape == cb.shape == (12, 16)
    assert np.all(cr == 1.0) and np.all(cb == 3.0)
    cr, cb = alsc_at(t, 4500)
    np.testing.assert_allclose(cr, 1.5)
    np.testing.assert_allclose(cb, 2.0)
    cr, cb = alsc_at(t, 2000)
    assert np.all(cr == 1.0) and np.all(cb == 3.0)
    cr, cb = alsc_at(t, 9000)
    assert np.all(cr == 2.0) and np.all(cb == 1.0)


def test_single_calibration_is_used_everywhere():
    ccm = [1.0, 0, 0, 0, 2, 0, 0, 0, 3]
    t = tuning_tables(small_tuning({"rpi.ccm": {"ccms": [{"ct": 4000, "ccm": ccm}]}}))
    for ct in (1000, 4000, 9000):
        np.testing.assert_array_equal(ccm_at(t, ct), np.diag([1.0, 2.0, 3.0]))


def test_tuning_refuses_nested_list():
    ccm = [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]
    bad = small_tuning({"rpi.ccm": {"ccms": [{"ct": 3000, "ccm": ccm + [[0, 0, 0]] * 6}]}})
    with pytest.raises(RawError, match="not a flat list"):
        tuning_tables(bad)


@pytest.mark.parametrize("ct", [float("nan"), float("inf")])
def test_interpolation_refuses_bad_ct(ct):
    t = tuning_tables(small_tuning())
    with pytest.raises(RawError, match="colour temperature"):
        ccm_at(t, ct)
    with pytest.raises(RawError, match="colour temperature"):
        alsc_at(t, ct)


# --- develop -------------------------------------------------------------

BLACK, WHITE = 256.0, 4095.0
_SITES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _mosaic(rgb: np.ndarray, cfa: str) -> RawFrame:
    """A raw frame whose sites sample ``rgb`` (linear, 0..1) through ``cfa``."""
    h, w, _ = rgb.shape
    plane = np.empty((h, w))
    for (r, c), colour in zip(_SITES, cfa, strict=True):
        plane[r::2, c::2] = rgb[r::2, c::2, "RGB".index(colour)]
    bayer = np.rint(BLACK + np.clip(plane, 0, 1) * (WHITE - BLACK)).astype(np.uint16)
    return RawFrame(bayer=bayer, cfa=cfa, black=BLACK, white=WHITE, as_shot_neutral=None)


def _flat_tuning(ccm=None, luminance=None, curve=((0.0, 0.0), (1.0, 1.0))) -> RawTuning:
    """Identity lens shading and colour matrix, a linear tone curve."""
    ones = np.ones((12, 16))
    return RawTuning(
        black_level=4096.0,
        ccms=((4000.0, np.eye(3) if ccm is None else np.asarray(ccm, dtype=float)),),
        alsc_cr=((4000.0, ones),),
        alsc_cb=((4000.0, ones),),
        luminance=ones if luminance is None else luminance,
        gamma_curve=np.asarray(curve, dtype=float),
    )


PATCHES = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "grey": (0.5, 0.5, 0.5)}


def _patch_frame(size=32):
    """Four flat patches, two by two: red, green / blue, grey."""
    rgb = np.zeros((2 * size, 2 * size, 3))
    for i, colour in enumerate(PATCHES.values()):
        r, c = divmod(i, 2)
        rgb[r * size : (r + 1) * size, c * size : (c + 1) * size] = colour
    return rgb


def _interior(out, i, size=32, margin=6):
    r, c = divmod(i, 2)
    rows = slice(r * size + margin, (r + 1) * size - margin)
    cols = slice(c * size + margin, (c + 1) * size - margin)
    return out[rows, cols]


@pytest.mark.parametrize("cfa", PATTERNS)
def test_demosaic_code_for_each_cfa(cfa):
    out = develop(_mosaic(_patch_frame(), cfa), _flat_tuning(), colour_gains=(1.0, 1.0),
                  colour_temperature=4000)
    assert out.dtype == np.uint8 and out.shape == (64, 64, 3)
    for i, (name, colour) in enumerate(PATCHES.items()):
        want = np.rint(np.array(colour) * 255)
        got = _interior(out, i).reshape(-1, 3).astype(int)
        assert np.abs(got - want).max() <= 2, (cfa, name, got.min(0), got.max(0))


def test_white_balance_makes_grey_neutral():
    gains = (3.14, 1.59)
    raw = np.full((32, 32, 3), 0.4)
    raw[..., 0] /= gains[0]
    raw[..., 2] /= gains[1]
    out = develop(_mosaic(raw, "BGGR"), _flat_tuning(), colour_gains=gains, colour_temperature=4000)
    centre = out[8:-8, 8:-8].reshape(-1, 3).astype(int)
    assert np.abs(centre - 102).max() <= 1


def test_clipped_highlight_stays_white():
    ccm = [2.0, -0.5, -0.5, -0.5, 2.0, -0.5, -0.5, -0.5, 2.0]
    frame = _mosaic(np.ones((32, 32, 3)), "BGGR")
    assert np.all(frame.bayer == WHITE)
    out = develop(frame, _flat_tuning(ccm=np.reshape(ccm, (3, 3))), colour_gains=(3.1, 1.6),
                  colour_temperature=4000)
    assert out.min() > 250


def _corner_ratio(out):
    grey = out.astype(float).mean(axis=2)
    centre = grey[28:36, 40:48].mean()
    corner = grey[:4, :4].mean()
    return centre / corner


def test_lens_shading_flattens_vignetting():
    cv2 = require_cv2()

    yy, xx = np.mgrid[0:12, 0:16]
    radius = np.hypot((yy - 5.5) / 5.5, (xx - 7.5) / 7.5)
    lum = 1.0 + 0.6 * radius**2
    # A flat grey scene seen through the lens: darker towards the corners.
    falloff = cv2.resize(1.0 / lum, (88, 64), interpolation=cv2.INTER_LINEAR)
    frame = _mosaic(np.repeat((0.3 * falloff)[..., None], 3, axis=2), "BGGR")
    args = {"colour_gains": (1.0, 1.0), "colour_temperature": 4000}
    without = _corner_ratio(develop(frame, _flat_tuning(), **args))
    with_lut = _corner_ratio(develop(frame, _flat_tuning(luminance=lum), **args))
    assert without > 1.3
    assert abs(with_lut - 1.0) < 0.05
    assert abs(with_lut - 1.0) < abs(without - 1.0)


def _with_corner(value: float) -> np.ndarray:
    """A grid at ``value`` with its bottom-right cell at 1, the minimum."""
    grid = np.full((12, 16), value)
    grid[11, 15] = 1.0
    return grid


def test_lens_shading_colour_grids_apply_to_their_sites():
    flat = np.full((32, 32, 3), 0.25)
    tuning = dataclasses.replace(
        _flat_tuning(),
        alsc_cr=((4000.0, _with_corner(2.0)),),
        alsc_cb=((4000.0, _with_corner(3.0)),),
    )
    out = develop(_mosaic(flat, "RGGB"), tuning, colour_gains=(1.0, 1.0), colour_temperature=4000)
    centre = out[8:-8, 8:-8].reshape(-1, 3).astype(int)
    assert np.abs(centre - [128, 64, 191]).max() <= 1


def test_lens_shading_colour_grids_are_divided_by_their_minimum():
    # A flat Cr or Cb table is all colour and no shading, as libcamera reads it.
    flat = np.full((32, 32, 3), 0.25)
    tuning = dataclasses.replace(
        _flat_tuning(),
        alsc_cr=((4000.0, np.full((12, 16), 2.9)),),
        alsc_cb=((4000.0, np.full((12, 16), 1.5)),),
    )
    args = {"colour_gains": (1.0, 1.0), "colour_temperature": 4000}
    out = develop(_mosaic(flat, "BGGR"), tuning, **args)
    np.testing.assert_array_equal(out, develop(_mosaic(flat, "BGGR"), _flat_tuning(), **args))


def test_develop_uses_ccm_at():
    tuning = tuning_tables(small_tuning())
    frame = _mosaic(np.random.default_rng(3).uniform(0.05, 0.6, size=(32, 48, 3)), "BGGR")
    args = {"colour_gains": (1.5, 1.2), "tone": "isp"}
    got = develop(frame, tuning, colour_temperature=4000, **args)
    midpoint = (tuning.ccms[0][1] + tuning.ccms[1][1]) / 2
    by_hand = dataclasses.replace(tuning, ccms=((4000.0, midpoint),))
    want = develop(frame, by_hand, colour_temperature=4000, **args)
    np.testing.assert_array_equal(got, want)
    other = develop(frame, tuning, colour_temperature=3000, **args)
    assert not np.array_equal(got, other)


def _ramp_frame():
    ramp = np.linspace(0.0, 1.0, 256)
    rgb = np.repeat(np.repeat(ramp[None, :, None], 8, axis=0), 3, axis=2)
    return _mosaic(rgb, "BGGR")


def test_tone_curves():
    assert TONE_CURVES == ("isp", "soft-1", "soft-1.5")
    curve = tuning_tables(small_tuning()).gamma_curve
    tuning = dataclasses.replace(_flat_tuning(), gamma_curve=curve)
    args = {"colour_gains": (1.0, 1.0), "colour_temperature": 4000}
    frame = _ramp_frame()
    isp = develop(frame, tuning, tone="isp", **args)[4, :, 1].astype(int)
    soft = develop(frame, tuning, tone="soft-1", **args)[4, :, 1].astype(int)
    softer = develop(frame, tuning, tone="soft-1.5", **args)[4, :, 1].astype(int)
    for row in (isp, soft, softer):
        # The exact ends are tested on _tone: here the edge column's demosaic
        # borrows from its neighbour, which the soft curves lift.
        assert row[-1] >= 252
        assert np.all(np.diff(row[2:-2]) >= 0)
    assert np.all(soft[4:64] > isp[4:64])
    assert np.all(softer[4:64] >= soft[4:64])


def test_soft_tone_function():
    from pifilm.raw import _tone

    curve = np.array([[0.0, 0.0], [1.0, 1.0]])
    x = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(_tone(x, curve, "soft-1"), [0.0, 0.4, 2 / 3, 1.0], rtol=1e-6)
    np.testing.assert_allclose(_tone(x, curve, "isp"), x)
    gamma = tuning_tables(small_tuning()).gamma_curve
    ends = np.array([0.0, 1.0], dtype=np.float32)
    for tone in TONE_CURVES:
        assert _tone(ends, gamma, tone).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("name", ["linear", "soft-", "soft-x", "soft-nan", "soft--1", "SOFT-1"])
def test_unknown_tone_raises(name):
    with pytest.raises(ValueError, match="tone"):
        develop(_ramp_frame(), _flat_tuning(), colour_gains=(1.0, 1.0), colour_temperature=4000,
                tone=name)


@pytest.mark.parametrize("gains", [(0.0, 1.0), (1.0, float("nan")), (-1.0, 2.0)])
def test_bad_colour_gains_raise(gains):
    with pytest.raises(RawError, match="colour gains"):
        develop(_ramp_frame(), _flat_tuning(), colour_gains=gains, colour_temperature=4000)


def test_develop_is_deterministic():
    tuning = tuning_tables(small_tuning())
    frame = _mosaic(np.random.default_rng(5).uniform(0, 1, size=(40, 56, 3)), "GRBG")
    args = {"colour_gains": (2.0, 1.4), "colour_temperature": 4500, "tone": "soft-1"}
    assert develop(frame, tuning, **args).tobytes() == develop(frame, tuning, **args).tobytes()


@pytest.mark.parametrize("tone", TONE_CURVES)
def test_as_float_rounds_to_the_uint8_output(tone):
    tuning = tuning_tables(small_tuning())
    frame = _mosaic(np.random.default_rng(6).uniform(0, 1, size=(40, 56, 3)), "BGGR")
    args = {"colour_gains": (2.0, 1.4), "colour_temperature": 4500, "tone": tone}
    as_u8 = develop(frame, tuning, **args)
    as_f = develop(frame, tuning, as_float=True, **args)
    assert as_f.dtype == np.float32 and as_f.shape == as_u8.shape
    assert as_f.min() >= 0.0 and as_f.max() <= 1.0
    np.testing.assert_array_equal(np.rint(as_f * np.float32(255)).astype(np.uint8), as_u8)


def test_develop_does_not_change_its_input():
    frame = _mosaic(_patch_frame(), "BGGR")
    before = frame.bayer.copy()
    develop(frame, _flat_tuning(), colour_gains=(2.0, 2.0), colour_temperature=4000)
    np.testing.assert_array_equal(frame.bayer, before)
