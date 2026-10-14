import numpy as np
import pytest

from pifilm.raw import RawError, RawFrame, read_dng
from tests.raw_fakes import LONG, RATIONAL, SHORT, make_dng

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
    with pytest.raises(RawError):
        read_dng(data[:20])


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
