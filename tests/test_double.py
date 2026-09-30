import numpy as np
import pytest

from pifilm.double import COMPOSITE_METHOD, composite


def _flat(value, shape=(4, 6)):
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_black_plus_black_stays_black():
    assert np.array_equal(composite(_flat(0), _flat(0)), _flat(0))


def test_black_lets_the_other_exposure_through_at_one_stop_under():
    """Darkness adds no light: white over black is white at half its linear light."""
    out = composite(_flat(0), _flat(255))
    assert np.all(out == 188)  # linear 0.5 -> sRGB 0.7354 -> 187.5 -> 188


def test_two_matching_exposures_make_one_normal_exposure():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    out = composite(frame, frame)
    assert np.abs(out.astype(int) - frame.astype(int)).max() <= 1


def test_colours_add_as_light_red_plus_green_is_yellow():
    red = _flat(0)
    red[..., 0] = 255
    green = _flat(0)
    green[..., 1] = 255
    out = composite(red, green)
    assert np.all(out[..., 0] == 188) and np.all(out[..., 1] == 188)
    assert np.all(out[..., 2] == 0)


def test_order_does_not_matter():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    assert np.array_equal(composite(a, b), composite(b, a))


def test_output_is_uint8_rgb_of_the_input_shape():
    out = composite(_flat(10, (3, 5)), _flat(200, (3, 5)))
    assert out.dtype == np.uint8 and out.shape == (3, 5, 3)


@pytest.mark.parametrize("second", [
    np.zeros((4, 7, 3), dtype=np.uint8),     # different size
    np.zeros((4, 6, 3), dtype=np.float32),   # wrong dtype
    np.zeros((4, 6), dtype=np.uint8),        # not RGB
])
def test_mismatched_or_malformed_frames_are_refused(second):
    with pytest.raises(ValueError):
        composite(_flat(0), second)


def test_method_name_is_recorded_verbatim():
    assert COMPOSITE_METHOD == "linear_mean"
