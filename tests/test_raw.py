import numpy as np
import pytest

from pifilm.raw import RawError, RawFrame, alsc_at, ccm_at, read_dng, tuning_tables
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
