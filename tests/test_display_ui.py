import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pifilm.display.ui import (
    DASH,
    HEIGHT,
    PROCESSING_BARS,
    PROCESSING_DIM,
    WIDTH,
    _letterbox,
    iso_label,
    render_message,
    render_processing,
    render_review,
    text_or_dash,
)


def test_render_processing_draws_the_same_colour_bars_as_the_opencv_screen():
    """The LCD shows the Stick's screen while a shot is graded, not the live view.

    ``pifilm.capture.app._capture_loading_screen`` lists the same seven bars in
    OpenCV's BGR order; the leftmost is grey and the rightmost blue on both.
    """
    img = render_processing()
    assert isinstance(img, Image.Image) and img.size == (WIDTH, HEIGHT) and img.mode == "RGB"
    grey, blue = img.getpixel((WIDTH // 14, 10)), img.getpixel((WIDTH * 13 // 14, 10))
    assert grey[0] == grey[1] == grey[2] > 0
    assert blue[0] == blue[1] == 0 and blue[2] > 0


def test_render_processing_bars_are_dimmed_on_the_lcd():
    """Full-strength bars glared on the panel in a dim room (asked 2026-09-24);
    they are drawn at PROCESSING_DIM of the shared colours, the label is not."""
    img = render_processing()
    for index, colour in enumerate(PROCESSING_BARS):
        x = (2 * index + 1) * WIDTH // 14
        expected = tuple(round(c * PROCESSING_DIM) for c in colour)
        assert img.getpixel((x, 10)) == expected
    assert 0.4 <= PROCESSING_DIM <= 0.7


def test_render_processing_is_black_below_the_bars_and_carries_a_label():
    img = render_processing()
    assert img.getpixel((5, HEIGHT - 5)) == (0, 0, 0)
    bottom = np.array(img)[HEIGHT * 3 // 4:]
    assert bottom.max() > 200  # the label is drawn in white in the bottom quarter


def test_render_review_and_message_sizes():
    graded = np.full((3040, 4056, 3), 90, dtype=np.uint8)[::8, ::8]
    assert render_review(graded, "1/250  ISO 100").size == (320, 240)
    assert render_message("Capture failed", "camera_error").size == (320, 240)


def test_the_review_caption_sits_on_a_darker_band_at_the_bottom():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = render_review(frame, "1/250  ISO 100")
    assert img.getpixel((160, 120)) == (200, 200, 200)
    assert sum(img.getpixel((160, HEIGHT - 3))) < sum(img.getpixel((160, 120)))


def test_missing_value_placeholder_is_ascii_dash_not_tofu_box():
    # ImageFont.load_default's bundled bitmap font has no glyph for U+2014 (em dash) or
    # U+2212 (minus sign): both rasterise as a hollow square "tofu" box under it, not a dash.
    # Missing-value placeholders must use ASCII "-" instead.
    assert text_or_dash(None) == DASH == "-"
    assert text_or_dash("1/250") == "1/250"
    assert iso_label(None) == "ISO -"
    assert iso_label(400) == "ISO 400"
    font = ImageFont.load_default(size=14)

    def render(ch):
        img = Image.new("L", (30, 30), 0)
        ImageDraw.Draw(img).text((15, 15), ch, fill=255, font=font, anchor="mm")
        return np.array(img)

    dash_pixels = np.count_nonzero(render("-"))
    box_pixels = np.count_nonzero(render("—"))  # previously used placeholder; renders as tofu
    assert dash_pixels < box_pixels


def test_letterbox_bands_a_16_9_frame_and_fills_a_4_3_one():
    """Shared with the live screen: the whole frame is shown, never cropped or stretched."""
    wide = _letterbox(np.full((360, 640, 3), 200, dtype=np.uint8))
    assert wide.size == (WIDTH, HEIGHT) and wide.mode == "RGB"
    assert wide.getpixel((5, 5)) == (0, 0, 0)            # band above the picture
    assert wide.getpixel((160, 120)) == (200, 200, 200)
    full = _letterbox(np.full((480, 640, 3), 200, dtype=np.uint8))
    assert full.getpixel((0, 0)) == full.getpixel((319, 239)) == (200, 200, 200)


def test_letterbox_fits_the_frame_to_a_640x480_panel():
    wide = _letterbox(np.full((360, 640, 3), 200, dtype=np.uint8), (640, 480))
    assert wide.size == (640, 480)
    assert wide.getpixel((320, 30)) == (0, 0, 0)          # band above the picture
    assert wide.getpixel((320, 240)) == (200, 200, 200)   # picture, not stretched


# -- scale ----------------------------------------------------------------------


def test_every_screen_renders_at_640x480_at_scale_2():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    images = [
        render_review(frame, "1/250  ISO 100", scale=2),
        render_processing(scale=2),
        render_message("Capture failed", "camera busy", scale=2),
    ]
    for img in images:
        assert img.size == (640, 480) and img.mode == "RGB"


def test_scale_1_is_the_default_and_unchanged():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert render_review(frame, "1/250  ISO 100", scale=1).tobytes() == render_review(
        frame, "1/250  ISO 100").tobytes()
    assert render_processing(scale=1).tobytes() == render_processing().tobytes()


def test_processing_bars_leave_no_gaps_at_scale_2():
    img = render_processing(scale=2)
    boundary = 2 * (WIDTH // 7)          # first column of the second bar
    for x in (boundary - 1, boundary):
        assert img.getpixel((x, 100)) != (0, 0, 0)
    assert img.getpixel((639, 100)) != (0, 0, 0)
