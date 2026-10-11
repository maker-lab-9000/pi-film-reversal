import math
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont

from pifilm.display import liveview
from pifilm.display.liveview import (
    AMBER,
    BAR_TOP,
    CELL_PITCH,
    CELL_VALUE_Y,
    CELL_X0,
    DIM_AMBER,
    DOUBLE_BOX,
    HEIGHT,
    KEY_RADIUS,
    METER_X0,
    METER_X1,
    METER_Y,
    MINUS_CENTRE,
    PLUS_CENTRE,
    SELECT_ACTIONS,
    SHUTTER_CENTRE,
    SHUTTER_RADIUS,
    WHITE,
    WIDTH,
    Action,
    Cell,
    LiveState,
    hit,
    render_live,
)
from pifilm.display.meter import MeterReading

FONT_FILES = ("BarlowCondensed-SemiBold.ttf", "BarlowCondensed-Medium.ttf")
# The hit radii and margins of the brief, restated here on purpose: the scan below
# checks ``hit`` against an independent statement of the regions.
SHUTTER_HIT_RADIUS = 34
KEY_HIT_RADIUS = 21
DOUBLE_HIT_MARGIN = 6
CELL_HIT_TOP = 204
CELL_HIT_X0 = 8
FRAME_GREY = (128, 128, 128)


def _reading(**over):
    base = dict(shutter="1/250", iso=100, ev_comp=0.0, lux=12.0, deviation_ev=0.0,
                clip_pct=0.0, battery_percent=88, external_power=True)
    base.update(over)
    return MeterReading(**base)


def _cells(**over):
    cells = [Cell("SHUTTER", "1/125"), Cell("ISO", "400"), Cell("WB", "5200K")]
    for index, cell in over.items():
        cells[int(index[1:])] = cell
    return tuple(cells)


def _state(**over):
    base = dict(cells=_cells(), selected=1, pressed=None, double=(False, 0))
    base.update(over)
    return LiveState(**base)


def _frame(value=128):
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _pixels(state=None, *, scale=1, frame=128, **reading):
    img = render_live(_frame(frame), _reading(**reading), state or _state(), scale=scale)
    return np.asarray(img).astype(int)


def _crop(pixels, box, scale=1):
    """The pixels of a box given in base units (x0, y0, x1, y1; far edges excluded)."""
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    return pixels[y0:y1, x0:x1]


def _near(pixels, colour, tol=40):
    return np.abs(pixels - np.array(colour)).max(axis=-1) <= tol


def _neutral_peak(pixels):
    """The brightest pixel with no colour cast: white or grey ink, never amber.

    Labels are 5 units tall, so at scale 1 their strokes are thinner than a pixel
    and never reach the ink's full value; the tests compare brightness, not colour.
    """
    neutral = pixels[(pixels.max(axis=-1) - pixels.min(axis=-1)) <= 14]
    return int(neutral.max())


# Brightest neutral pixel of a cell: a white AUTO tag is above the first, grey ink
# is between the two, and a cell with neither is below the second.
WHITE_INK, ANY_INK = 150, 60


def _cell_box(index):
    x0 = CELL_HIT_X0 + CELL_PITCH * index
    return (x0, BAR_TOP, x0 + CELL_PITCH, HEIGHT)


def _dot_x(j):
    return METER_X0 + j * (METER_X1 - METER_X0) / 12


def _dot_is_lit(pixels, j, scale=1):
    return bool(_near(pixels[METER_Y * scale, int(_dot_x(j) * scale)], AMBER))


def _regions():
    """Every hit region as a set of base pixels, from the brief's numbers."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]

    def circle(centre, radius):
        return np.hypot(xs - centre[0], ys - centre[1]) <= radius

    x0, y0, x1, y1 = DOUBLE_BOX
    regions = {
        Action.SHUTTER: circle(SHUTTER_CENTRE, SHUTTER_HIT_RADIUS),
        Action.PLUS: circle(PLUS_CENTRE, KEY_HIT_RADIUS),
        Action.MINUS: circle(MINUS_CENTRE, KEY_HIT_RADIUS),
        Action.DOUBLE_TOGGLE: ((xs >= x0 - DOUBLE_HIT_MARGIN) & (xs < x1 + DOUBLE_HIT_MARGIN)
                               & (ys >= y0 - DOUBLE_HIT_MARGIN) & (ys < y1 + DOUBLE_HIT_MARGIN)),
    }
    for index, action in enumerate(SELECT_ACTIONS):
        left = CELL_HIT_X0 + CELL_PITCH * index
        regions[action] = (ys >= CELL_HIT_TOP) & (xs >= left) & (xs < left + CELL_PITCH)
    return regions


@pytest.mark.parametrize("scale", [1, 2])
def test_render_live_is_an_rgb_image_of_the_panel_size(scale):
    img = render_live(_frame(), _reading(), _state(), scale=scale)
    assert isinstance(img, Image.Image)
    assert img.size == (320 * scale, 240 * scale) and img.mode == "RGB"


def test_the_geometry_constants_are_the_ones_the_viewfinder_loop_is_planned_against():
    assert (WIDTH, HEIGHT, BAR_TOP) == (320, 240, 208)
    assert (CELL_X0, CELL_PITCH) == (11, 61)
    assert (METER_X0, METER_X1, METER_Y) == (205, 306, 226)
    assert (SHUTTER_CENTRE, SHUTTER_RADIUS) == ((287, 39), 28)
    assert (PLUS_CENTRE, MINUS_CENTRE, KEY_RADIUS) == ((287, 125), (287, 168), 17)
    assert DOUBLE_BOX == (6, 5, 30, 16)
    assert SELECT_ACTIONS == (Action.SELECT_SHUTTER, Action.SELECT_ISO, Action.SELECT_WB)


@pytest.mark.parametrize("scale", [1, 2])
def test_every_control_is_hit_at_its_centre(scale):
    x0, y0, x1, y1 = DOUBLE_BOX
    centres = {
        Action.SHUTTER: SHUTTER_CENTRE,
        Action.PLUS: PLUS_CENTRE,
        Action.MINUS: MINUS_CENTRE,
        Action.DOUBLE_TOGGLE: ((x0 + x1) // 2, (y0 + y1) // 2),
    }
    for index, action in enumerate(SELECT_ACTIONS):
        centres[action] = (CELL_HIT_X0 + CELL_PITCH * index + CELL_PITCH // 2, 222)
    for action, (x, y) in centres.items():
        assert hit(x * scale, y * scale, scale) is action
        # The last panel pixel of the same base pixel belongs to it too.
        assert hit(x * scale + scale - 1, y * scale + scale - 1, scale) is action


def test_hit_regions_are_pairwise_disjoint_and_hit_agrees_with_each_of_them():
    regions = _regions()
    claimed = np.zeros((HEIGHT, WIDTH), dtype=int)
    for action, mask in regions.items():
        assert mask.any(), action
        claimed += mask
        for y, x in zip(*np.nonzero(mask), strict=True):
            assert hit(int(x), int(y)) is action, (action, x, y)
    assert claimed.max() == 1
    for y, x in zip(*np.nonzero(claimed == 0), strict=True):
        assert hit(int(x), int(y)) is Action.NONE, (x, y)


def test_hit_regions_are_the_same_on_the_dsi_panel_at_scale_2():
    for y in range(0, HEIGHT * 2, 3):
        for x in range(0, WIDTH * 2, 3):
            assert hit(x, y, 2) is hit(x // 2, y // 2), (x, y)


def test_the_shutter_is_at_least_20_units_from_the_plus_key():
    """A thumb working + and - must not be able to take a photo by slipping."""
    regions = _regions()
    shutter = np.argwhere(regions[Action.SHUTTER]).astype(float)
    plus = np.argwhere(regions[Action.PLUS]).astype(float)
    gaps = np.hypot(shutter[:, None, 0] - plus[None, :, 0], shutter[:, None, 1] - plus[None, :, 1])
    assert gaps.min() >= 20
    assert math.dist(SHUTTER_CENTRE, PLUS_CENTRE) - SHUTTER_HIT_RADIUS - KEY_HIT_RADIUS >= 20


@pytest.mark.parametrize("scale", [1, 2])
def test_the_image_and_the_meter_are_not_controls(scale):
    assert hit(160 * scale, 100 * scale, scale) is Action.NONE
    assert hit(255 * scale, METER_Y * scale, scale) is Action.NONE
    assert hit(255 * scale, 215 * scale, scale) is Action.NONE


@pytest.mark.parametrize("scale", [1, 2])
def test_the_strip_is_solid_black_where_nothing_is_drawn(scale):
    pixels = _pixels(scale=scale, frame=255)
    for x, y in ((2, 224), (195, 236), (150, BAR_TOP), (318, 239), (60, 239)):
        assert tuple(pixels[y * scale, x * scale]) == (0, 0, 0), (x, y)
    # ... and the image above it is the frame, right up to the strip's edge.
    assert tuple(pixels[BAR_TOP * scale - 1, 150 * scale]) == (255, 255, 255)


@pytest.mark.parametrize("scale", [1, 2])
def test_the_selected_cell_is_amber_and_underlined(scale):
    pixels = _pixels(_state(selected=1), scale=scale)
    cell = _crop(pixels, _cell_box(1), scale)
    assert _near(cell, AMBER).sum() > 20 * scale * scale
    assert _neutral_peak(cell) > WHITE_INK  # its AUTO tag
    underline = _crop(pixels, (CELL_X0 + CELL_PITCH, 237, CELL_X0 + CELL_PITCH + 46, 238), scale)
    assert underline.size and _near(underline, AMBER).all()


@pytest.mark.parametrize("scale", [1, 2])
def test_an_unselected_cell_is_dim_amber_with_no_underline(scale):
    pixels = _pixels(_state(selected=1), scale=scale)
    for index in (0, 2):
        cell = _crop(pixels, _cell_box(index), scale)
        assert not _near(cell, AMBER).any()
        assert _near(cell, DIM_AMBER, 25).any()
        assert ANY_INK < _neutral_peak(cell) < WHITE_INK  # label and AUTO, both grey
        left = CELL_X0 + CELL_PITCH * index
        assert _crop(pixels, (left, 237, left + 46, 239), scale).max() < 40


@pytest.mark.parametrize("scale", [1, 2])
def test_a_disabled_cell_is_grey_even_when_it_is_the_selected_index(scale):
    state = _state(cells=_cells(c2=Cell("WB", "-", auto=True, enabled=False)), selected=2)
    cell = _crop(_pixels(state, scale=scale), _cell_box(2), scale)
    assert _neutral_peak(cell) > ANY_INK
    label_rows = cell[: (CELL_VALUE_Y - 8 - BAR_TOP) * scale]
    assert ANY_INK < _neutral_peak(label_rows) < WHITE_INK  # AUTO is grey too, not white
    assert not _near(cell, AMBER).any()
    assert not _near(cell, DIM_AMBER, 25).any()
    # Grey means grey: no pixel of the cell carries a colour cast at all.
    assert (cell.max(axis=-1) - cell.min(axis=-1)).max() <= 12


def test_nothing_selected_draws_no_underline_anywhere():
    pixels = _pixels(_state(selected=None))
    assert _crop(pixels, (0, 237, 192, 240)).max() < 40
    assert not _near(_crop(pixels, (0, BAR_TOP, 192, HEIGHT)), AMBER).any()


def test_a_fixed_control_has_no_auto_tag():
    auto = _crop(_pixels(_state(cells=_cells(c1=Cell("ISO", "800", auto=True)))), _cell_box(1))
    fixed = _crop(_pixels(_state(cells=_cells(c1=Cell("ISO", "800", auto=False)))), _cell_box(1))
    assert _neutral_peak(auto) > WHITE_INK and _neutral_peak(fixed) < ANY_INK
    assert _near(fixed, AMBER).any()


def _key_centre(pixels, centre, scale, half=8):
    cx, cy = centre
    return _crop(pixels, (cx - half, cy - half, cx + half, cy + half), scale)


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("pressed", [Action.PLUS, Action.MINUS, Action.SHUTTER])
def test_a_pressed_key_is_filled_amber_and_the_others_are_unchanged(pressed, scale):
    centres = {Action.PLUS: PLUS_CENTRE, Action.MINUS: MINUS_CENTRE,
               Action.SHUTTER: SHUTTER_CENTRE}
    idle = _pixels(_state(pressed=None), scale=scale)
    lit = _pixels(_state(pressed=pressed), scale=scale)
    for action, centre in centres.items():
        before = _key_centre(idle, centre, scale)
        after = _key_centre(lit, centre, scale)
        assert _near(before, AMBER).mean() < 0.05
        if action is pressed:
            assert _near(after, AMBER).mean() > 0.75
        else:
            assert np.array_equal(before, after)


def test_the_keys_are_drawn_where_they_are_hit():
    idle = _pixels(frame=255)
    for cx, cy in (PLUS_CENTRE, MINUS_CENTRE):
        # Translucent black inside the ring, the untouched frame outside the hit circle.
        assert idle[cy - 10, cx - 10].max() < 200
        assert tuple(idle[cy, cx - KEY_HIT_RADIUS - 2]) == (255, 255, 255)
    cx, cy = SHUTTER_CENTRE
    assert idle[cy, cx].min() > 240  # the white disc
    assert _near(idle[cy, cx - SHUTTER_RADIUS:cx - SHUTTER_RADIUS + 3], AMBER, 90).any()


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize(
    "deviation, lit", [(-3.0, 0), (0.0, 6), (3.0, 12), (9.0, 12), (-9.0, 0), (1.0, 8),
                       (float("nan"), 6), (float("inf"), 12)],
)
def test_the_lit_meter_dot_follows_the_deviation(deviation, lit, scale):
    pixels = _pixels(scale=scale, deviation_ev=deviation)
    assert [j for j in range(13) if _dot_is_lit(pixels, j, scale)] == [lit]


def test_the_unlit_dots_and_the_scale_ends_are_drawn():
    pixels = _pixels(deviation_ev=0.0)
    for j in (0, 3, 12):
        assert pixels[METER_Y, int(_dot_x(j))].max() > 40
    assert _crop(pixels, (METER_X0 - 11, METER_Y - 4, METER_X0 - 3, METER_Y + 4)).any()
    assert _crop(pixels, (METER_X1 + 4, METER_Y - 4, METER_X1 + 11, METER_Y + 4)).any()


def test_the_lux_and_clip_text_is_drawn_and_survives_missing_values():
    with_lux = _crop(_pixels(lux=12.0, clip_pct=0.0), (205, 210, 306, 220))
    assert with_lux.any()
    without = _crop(_pixels(lux=None), (205, 210, 306, 220))
    assert without.any() and not np.array_equal(with_lux, without)
    assert liveview._meter_text(12.4, 0.0) == "12 LX   CLIP 0%"
    assert liveview._meter_text(None, 3.6) == "- LX   CLIP 4%"
    assert liveview._meter_text(float("nan"), float("nan")) == "- LX   CLIP -%"


@pytest.mark.parametrize("scale", [1, 2])
def test_the_double_exposure_toggle_is_hidden_outlined_or_amber(scale):
    hidden = _crop(_pixels(_state(double=None), scale=scale), DOUBLE_BOX, scale)
    assert (hidden == np.array(FRAME_GREY)).all()
    off = _crop(_pixels(_state(double=(False, 0)), scale=scale), DOUBLE_BOX, scale)
    assert not (off == np.array(FRAME_GREY)).all()
    assert _near(off, AMBER).mean() < 0.4 and _near(off, WHITE, 60).any()
    on = _crop(_pixels(_state(double=(True, 1)), scale=scale), DOUBLE_BOX, scale)
    assert _near(on, AMBER).mean() > 0.5
    first = _crop(_pixels(_state(double=(True, 0)), scale=scale), DOUBLE_BOX, scale)
    assert not np.array_equal(on, first)  # the progress count is part of the label


@pytest.mark.parametrize("scale", [1, 2])
def test_the_battery_tag_is_absent_without_a_battery(scale):
    box = (DOUBLE_BOX[2] + 4, 0, 120, 22)
    absent = _crop(_pixels(scale=scale, battery_percent=None, external_power=None), box, scale)
    assert (absent == np.array(FRAME_GREY)).all()
    on_ac = _crop(_pixels(scale=scale, battery_percent=88, external_power=True), box, scale)
    on_battery = _crop(_pixels(scale=scale, battery_percent=88, external_power=False), box, scale)
    assert _near(on_ac, WHITE, 60).any() and _near(on_ac, AMBER, 90).any()
    assert not np.array_equal(on_ac, on_battery)


def test_a_long_double_exposure_label_does_not_run_into_the_battery_tag():
    for scale in (1, 2):
        pixels = _pixels(_state(double=(True, 1)), scale=scale, frame=255, battery_percent=100)
        # The last base column before the tag (which starts at x 35) is still the frame.
        assert (_crop(pixels, (34, 0, 35, 22), scale) >= 200).all()


@pytest.mark.parametrize("scale", [1, 2])
def test_the_focus_gauge_is_absent_without_a_focus_score(scale):
    box = (0, 20, 14, 200)
    absent = _crop(_pixels(scale=scale, focus=None, focus_peak=0.7), box, scale)
    assert (absent == np.array(FRAME_GREY)).all()
    pixels = _pixels(scale=scale, focus=0.5, focus_peak=0.75)
    # Filled white from the bottom to half way, a darkened track above, an amber
    # tick at the peak, and nothing below the gauge.
    assert _near(pixels[150 * scale, 7 * scale], WHITE, 12)
    assert pixels[50 * scale, 7 * scale].max() < 128
    assert _near(_crop(pixels, (4, 64, 11, 68), scale), AMBER).any()
    assert not _near(_crop(pixels, (4, 22, 11, 62), scale), AMBER).any()
    assert (_crop(pixels, (0, 198, 14, 204), scale) == np.array(FRAME_GREY)).all()


@pytest.mark.parametrize("focus, peak", [(0.0, 0.0), (1.0, 1.0), (7.0, -2.0),
                                        (float("nan"), float("nan")), (0.3, None)])
def test_the_focus_gauge_stays_inside_its_track_for_any_score(focus, peak):
    without = _pixels(focus=None)
    pixels = _pixels(focus=focus, focus_peak=peak)
    changed = np.argwhere((pixels != without).any(axis=-1))
    assert changed.size
    assert changed[:, 0].min() >= 22 and changed[:, 0].max() < 198
    assert changed[:, 1].min() >= 4 and changed[:, 1].max() < 11


@pytest.mark.parametrize("scale", [1, 2])
def test_the_same_state_renders_the_same_image_and_only_the_meter_follows_the_needle(scale):
    state = _state(pressed=Action.PLUS, double=(True, 1))
    first = _pixels(state, scale=scale, deviation_ev=-2.0, focus=0.4, focus_peak=0.6)
    moved = _pixels(state, scale=scale, deviation_ev=2.5, focus=0.4, focus_peak=0.6)
    again = _pixels(state, scale=scale, deviation_ev=-2.0, focus=0.4, focus_peak=0.6)
    assert np.array_equal(first, again)
    changed = np.argwhere((first != moved).any(axis=-1)) / scale
    assert changed.size
    assert changed[:, 0].min() >= METER_Y - 4 and changed[:, 0].max() < METER_Y + 4
    assert changed[:, 1].min() >= METER_X0 - 4 and changed[:, 1].max() < METER_X1 + 4


def test_a_frame_with_new_meter_values_does_not_redraw_the_static_parts():
    """The static tiles are cached; a Pi 4 cannot repaint them ten times a second."""
    static = (liveview._strip_tile, liveview._cell_tile, liveview._controls_tile,
              liveview._tags_tile)
    state = _state()
    render_live(_frame(), _reading(), state, scale=2)
    before = [tile.cache_info().misses for tile in static]
    for step in range(20):
        render_live(_frame(), _reading(deviation_ev=-3 + 0.3 * step, lux=10.0 + step,
                                       focus=step / 20, focus_peak=0.9), state, scale=2)
    assert [tile.cache_info().misses for tile in static] == before
    # A metered value in one cell repaints that cell only.
    render_live(_frame(), _reading(), _state(cells=_cells(c0=Cell("SHUTTER", "1/60"))), scale=2)
    after = [tile.cache_info().misses for tile in static]
    assert [b - a for a, b in zip(before, after, strict=True)] == [0, 1, 0, 0]


def test_render_live_reads_only_the_fields_that_survive_the_ev_removal():
    reading = SimpleNamespace(deviation_ev=0.5, lux=12.0, clip_pct=1.0, focus=0.5,
                              focus_peak=0.6, battery_percent=50, external_power=False)
    img = render_live(_frame(), reading, _state())
    assert img.size == (WIDTH, HEIGHT)


def test_a_letterboxed_frame_and_odd_values_do_not_raise():
    wide = np.full((360, 640, 3), 200, dtype=np.uint8)
    state = _state(cells=_cells(c0=Cell("SHUTTER", "1/2000", auto=False), c1=Cell("ISO", "MAX"),
                                c2=Cell("WB", "-", enabled=False)), selected=None, double=None)
    for scale in (1, 2):
        img = render_live(wide, _reading(lux=None, clip_pct=100.0, battery_percent=None,
                                         external_power=None), state, scale=scale)
        assert img.size == (WIDTH * scale, HEIGHT * scale)


@pytest.mark.parametrize("name", FONT_FILES)
def test_the_bundled_fonts_are_packaged_and_load(name):
    path = resources.files("pifilm") / "data" / "fonts" / name
    assert path.is_file()
    assert (resources.files("pifilm") / "data" / "fonts" / "OFL.txt").is_file()
    font = liveview._font(name, 20)
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert font.getname()[0] == "Barlow Condensed"
    assert font.size == 20


def test_a_missing_font_file_falls_back_to_the_default_font():
    font = liveview._font("NoSuchFont.ttf", 14)
    assert font.getlength("400") > 0
