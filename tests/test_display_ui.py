import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pifilm.display.meter import MeterReading
from pifilm.display.ui import (
    BAR_TOP,
    SHUTTER_CENTRE,
    SHUTTER_RADIUS,
    Action,
    _text_or_dash,
    hit,
    render_live,
    render_message,
    render_review,
)


def _reading(**over):
    base = dict(shutter="1/250", iso=100, ev_comp=0.0, lux=640.0, deviation_ev=0.0,
                clip_pct=3.2, battery_percent=87, external_power=False)
    base.update(over)
    return MeterReading(**base)


def test_render_live_is_320x240_rgb_with_a_darker_bar():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(), busy=False)
    assert isinstance(img, Image.Image) and img.size == (320, 240) and img.mode == "RGB"
    above = img.getpixel((160, BAR_TOP - 10))
    inside = img.getpixel((160, BAR_TOP + 4))
    assert sum(inside) < sum(above)


def test_render_live_letterboxes_a_16_9_frame():
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(), busy=False)
    assert img.getpixel((5, 5)) == (0, 0, 0)  # top band
    assert img.getpixel((160, 120)) != (0, 0, 0)


def test_busy_marker_is_drawn_only_when_busy():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    idle = render_live(frame, _reading(), busy=False).getpixel((10, 10))
    busy = render_live(frame, _reading(), busy=True).getpixel((10, 10))
    assert idle != busy and busy[0] > 200


def test_render_live_handles_missing_fields():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    img = render_live(frame, _reading(shutter=None, iso=None, lux=None,
                                       battery_percent=None, external_power=None), busy=False)
    assert img.size == (320, 240)


def test_render_review_and_message_sizes():
    graded = np.full((3040, 4056, 3), 90, dtype=np.uint8)[::8, ::8]
    assert render_review(graded, "1/250  ISO 100  0").size == (320, 240)
    assert render_message("Capture failed", "camera_error").size == (320, 240)


def test_missing_value_placeholder_is_ascii_dash_not_tofu_box():
    # ImageFont.load_default's bundled bitmap font has no glyph for U+2014 (em dash) or
    # U+2212 (minus sign): both rasterise as a hollow square "tofu" box under it, not a dash.
    # Missing-value placeholders and the EV-minus label must use ASCII "-" instead.
    assert _text_or_dash(None) == "-"
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
