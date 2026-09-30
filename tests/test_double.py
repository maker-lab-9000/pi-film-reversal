import io

import numpy as np
import pytest
from PIL import Image

from pifilm.capture.thumbnail import (
    MAX_JPEG_BYTES,
    THUMBNAIL_SIZE,
    fitted_jpeg,
    render_exposure_card,
)
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


def test_exposure_card_is_a_stick_sized_jpeg(tmp_path):
    data = render_exposure_card(1, 2)
    assert len(data) <= MAX_JPEG_BYTES
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "JPEG" and img.size == THUMBNAIL_SIZE
    # The remote /image.jpg path serves it through fitted_jpeg from a file.
    path = tmp_path / "card.jpg"
    path.write_bytes(data)
    assert len(fitted_jpeg(path)) <= MAX_JPEG_BYTES


def test_exposure_card_text_depends_on_the_index():
    assert render_exposure_card(1, 2) != render_exposure_card(2, 2)


def _reference_composite(first, second):
    """Float64 sRGB -> linear -> mean -> sRGB, straight from the formulas."""
    def to_linear(x):
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    mean = 0.5 * (to_linear(first.astype(np.float64)) + to_linear(second.astype(np.float64)))
    srgb = np.where(mean <= 0.0031308, mean * 12.92, 1.055 * mean ** (1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


@pytest.mark.slow
def test_full_sensor_frame_matches_a_float64_reference_within_one_code():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=(3040, 4056, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(3040, 4056, 3), dtype=np.uint8)
    out = composite(a, b)
    ref = _reference_composite(a, b)
    assert out.shape == ref.shape and out.dtype == np.uint8
    assert np.abs(out.astype(np.int16) - ref.astype(np.int16)).max() <= 1
