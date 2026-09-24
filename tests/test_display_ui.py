import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pifilm.display.meter import MeterReading
from pifilm.display.ui import (
    AMBER,
    BAR_TOP,
    DASH,
    FOCUS_BAR_X0,
    FOCUS_BAR_X1,
    FOCUS_BAR_Y0,
    FOCUS_BAR_Y1,
    HEIGHT,
    SHUTTER_CENTRE,
    SHUTTER_RADIUS,
    WIDTH,
    Action,
    hit,
    iso_label,
    render_live,
    render_message,
    render_processing,
    render_review,
    text_or_dash,
)


def _reading(**over):
    base = dict(shutter="1/250", iso=100, ev_comp=0.0, lux=640.0, deviation_ev=0.0,
                clip_pct=3.2, battery_percent=87, external_power=False)
    base.update(over)
    return MeterReading(**base)


def test_render_live_is_320x240_rgb_with_a_darker_bar():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading())
    assert isinstance(img, Image.Image) and img.size == (320, 240) and img.mode == "RGB"
    above = img.getpixel((160, BAR_TOP - 10))
    inside = img.getpixel((160, BAR_TOP + 4))
    assert sum(inside) < sum(above)


def test_render_live_letterboxes_a_16_9_frame():
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading())
    assert img.getpixel((5, 5)) == (0, 0, 0)  # top band
    assert img.getpixel((160, 120)) != (0, 0, 0)


def test_render_processing_draws_the_same_colour_bars_as_the_opencv_screen():
    """The LCD shows the Stick's screen while a shot is graded, not the live view.

    ``pifilm.capture.app._capture_loading_screen`` lists the same seven bars in
    OpenCV's BGR order; the leftmost is grey and the rightmost blue on both.
    """
    img = render_processing()
    assert isinstance(img, Image.Image) and img.size == (WIDTH, HEIGHT) and img.mode == "RGB"
    assert img.getpixel((WIDTH // 14, 10)) == (191, 191, 191)
    assert img.getpixel((WIDTH * 13 // 14, 10)) == (0, 0, 191)


def test_render_processing_is_black_below_the_bars_and_carries_a_label():
    img = render_processing()
    assert img.getpixel((5, HEIGHT - 5)) == (0, 0, 0)
    bottom = np.array(img)[HEIGHT * 3 // 4:]
    assert bottom.max() > 200  # the label is drawn in white in the bottom quarter


def test_render_live_handles_missing_fields():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    img = render_live(frame, _reading(shutter=None, iso=None, lux=None,
                                       battery_percent=None, external_power=None))
    assert img.size == (320, 240)


def test_render_review_and_message_sizes():
    graded = np.full((3040, 4056, 3), 90, dtype=np.uint8)[::8, ::8]
    assert render_review(graded, "1/250  ISO 100  0").size == (320, 240)
    assert render_message("Capture failed", "camera_error").size == (320, 240)


def test_missing_value_placeholder_is_ascii_dash_not_tofu_box():
    # ImageFont.load_default's bundled bitmap font has no glyph for U+2014 (em dash) or
    # U+2212 (minus sign): both rasterise as a hollow square "tofu" box under it, not a dash.
    # Missing-value placeholders and the EV-minus label must use ASCII "-" instead.
    assert text_or_dash(None) == DASH == "-"
    assert iso_label(None) == "ISO -"
    font = ImageFont.load_default(size=14)

    def render(ch):
        img = Image.new("L", (30, 30), 0)
        ImageDraw.Draw(img).text((15, 15), ch, fill=255, font=font, anchor="mm")
        return np.array(img)

    dash_pixels = np.count_nonzero(render("-"))
    box_pixels = np.count_nonzero(render("—"))  # previously used placeholder; renders as tofu
    assert dash_pixels < box_pixels


def test_hit_regions():
    cx, cy = SHUTTER_CENTRE
    assert hit(cx, cy) == Action.SHUTTER
    assert hit(cx + SHUTTER_RADIUS + 4, cy) == Action.SHUTTER   # 6 px margin
    assert hit(cx + SHUTTER_RADIUS + 20, cy) == Action.NONE
    assert hit(10, BAR_TOP + 10) == Action.EV_MINUS
    assert hit(310, BAR_TOP + 10) == Action.EV_PLUS
    assert hit(160, BAR_TOP + 10) == Action.NONE
    assert hit(160, 100) == Action.NONE


FOCUS_GREEN = (0, 220, 90)
_BAR_MID_X = (FOCUS_BAR_X0 + FOCUS_BAR_X1) // 2


def _live(focus):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    return render_live(frame, _reading(focus=focus))


def test_focus_bar_fills_to_the_top_when_the_image_is_at_its_sharpest():
    img = _live(1.0)
    assert img.getpixel((_BAR_MID_X, FOCUS_BAR_Y1 - 5)) == FOCUS_GREEN   # near the bottom
    assert img.getpixel((_BAR_MID_X, FOCUS_BAR_Y0 + 6)) == FOCUS_GREEN   # near the top


def test_focus_bar_only_fills_from_the_bottom_when_focus_is_low():
    img = _live(0.2)
    assert img.getpixel((_BAR_MID_X, FOCUS_BAR_Y1 - 5)) == FOCUS_GREEN
    assert img.getpixel((_BAR_MID_X, FOCUS_BAR_Y0 + 6)) != FOCUS_GREEN


def test_no_focus_bar_is_drawn_when_focus_is_not_computed():
    img = _live(None)
    column = [img.getpixel((_BAR_MID_X, y)) for y in range(FOCUS_BAR_Y0, FOCUS_BAR_Y1 + 1)]
    assert FOCUS_GREEN not in column


def test_focus_bar_carries_an_amber_peak_mark_at_the_top():
    """The mark is what the user turns the ring towards; it must be visible even
    at full fill, so it is drawn over the green."""
    img = _live(1.0)
    column = [img.getpixel((_BAR_MID_X, y)) for y in range(FOCUS_BAR_Y0, FOCUS_BAR_Y0 + 6)]
    assert AMBER in column


def test_focus_bar_clears_the_ev_minus_button_and_its_hit_region():
    assert FOCUS_BAR_Y1 <= BAR_TOP - 8
    for y in (FOCUS_BAR_Y0, (FOCUS_BAR_Y0 + FOCUS_BAR_Y1) // 2, FOCUS_BAR_Y1):
        assert hit(_BAR_MID_X, y) == Action.NONE


def test_focus_bar_does_not_disturb_the_rest_of_the_screen():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    with_bar = np.array(render_live(frame, _reading(focus=0.5)))
    without = np.array(render_live(frame, _reading(focus=None)))
    right_of_bar = np.s_[:, FOCUS_BAR_X1 + 6:]
    assert np.array_equal(with_bar[right_of_bar], without[right_of_bar])
