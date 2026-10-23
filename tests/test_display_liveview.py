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
    MENU_CENTRE,
    MENU_LABEL,
    MENU_SIZE,
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
# The MENU pill (287, 199), 40 x 12, under -: hit over x 264 to 310 and y 190 to 214
# (x0, y0 included, x1, y1 excluded), from one unit under -'s hit circle into the strip.
MENU_HIT_BOX = (264, 190, 311, 215)
FRAME_GREY = (128, 128, 128)


def _reading(**over):
    base = dict(shutter="1/250", iso=100, lux=12.0, deviation_ev=0.0,
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
    mx0, my0, mx1, my1 = MENU_HIT_BOX
    regions[Action.MENU] = (xs >= mx0) & (xs < mx1) & (ys >= my0) & (ys < my1)
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
    assert (MENU_CENTRE, MENU_SIZE, MENU_LABEL) == ((287, 199), (40, 12), "MENU")
    assert liveview.MENU_HIT_BOX == MENU_HIT_BOX
    assert SELECT_ACTIONS == (Action.SELECT_SHUTTER, Action.SELECT_ISO, Action.SELECT_WB)


@pytest.mark.parametrize("scale", [1, 2])
def test_every_control_is_hit_at_its_centre(scale):
    x0, y0, x1, y1 = DOUBLE_BOX
    centres = {
        Action.SHUTTER: SHUTTER_CENTRE,
        Action.PLUS: PLUS_CENTRE,
        Action.MINUS: MINUS_CENTRE,
        Action.DOUBLE_TOGGLE: ((x0 + x1) // 2, (y0 + y1) // 2),
        Action.MENU: MENU_CENTRE,
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


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("pressed", [Action.PLUS, Action.MINUS, Action.SHUTTER, Action.MENU])
def test_a_pressed_key_is_filled_amber_and_the_others_are_unchanged(pressed, scale):
    centres = {Action.PLUS: PLUS_CENTRE, Action.MINUS: MINUS_CENTRE,
               Action.SHUTTER: SHUTTER_CENTRE, Action.MENU: MENU_CENTRE}
    idle = _pixels(_state(pressed=None), scale=scale)
    lit = _pixels(_state(pressed=pressed), scale=scale)
    for action, centre in centres.items():
        cx, cy = centre
        box = (cx - 8, cy - 8, cx + 8, cy + 8)
        if action is Action.MENU:
            # The pill is 12 units tall and its label black when lit: look at its
            # left end, inside the outline and clear of the label.
            box = (cx - 17, cy - 4, cx - 10, cy + 4)
        before = _crop(idle, box, scale)
        after = _crop(lit, box, scale)
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


def _changed_box(before, after, scale):
    """The bounding box, in base units, of the pixels that differ between two renders.

    Differences of a few levels are left out: the Lanczos reduction of a tile rings
    faintly for some pixels around every edge, far below what can be seen.
    """
    rows, cols = np.nonzero((np.abs(before - after) > 12).any(axis=2))
    assert rows.size, "the two renders are identical"
    return (cols.min() / scale, rows.min() / scale,
            (cols.max() + 1) / scale, (rows.max() + 1) / scale)


def _apart(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


@pytest.mark.parametrize("scale", [1, 2])
def test_no_photographs_developing_draws_nothing(scale):
    for double in (None, (False, 0), (True, 1)):
        for battery in (None, 88):
            as_before = LiveState(cells=_cells(), selected=1, pressed=None, double=double)
            zero = LiveState(cells=_cells(), selected=1, pressed=None, double=double,
                             developing=0)
            assert np.array_equal(
                _pixels(as_before, scale=scale, battery_percent=battery),
                _pixels(zero, scale=scale, battery_percent=battery))


def test_developing_and_the_tuning_flag_are_the_last_fields_with_off_defaults():
    # New fields go on the end, so an older positional construction keeps its meaning.
    assert LiveState(cells=_cells()).developing == 0
    assert LiveState(cells=_cells()).tuning_mismatch is False
    assert list(LiveState.__dataclass_fields__)[-5:-3] == ["developing", "tuning_mismatch"]


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("developing", [3, 12, 250])
@pytest.mark.parametrize("battery, on_ac", [(5, False), (88, True), (100, True)])
@pytest.mark.parametrize("double", [(False, 0), (True, 1)])
def test_the_developing_count_sits_beside_the_battery_and_touches_nothing_else(
        scale, developing, battery, on_ac, double):
    reading = dict(battery_percent=battery, external_power=on_ac)
    plain = _pixels(_state(double=double), scale=scale, **reading)
    count = _changed_box(
        plain, _pixels(_state(double=double, developing=developing), scale=scale, **reading),
        scale)
    # Where the battery and the 2x mark are, found by taking each away.
    battery_box = _changed_box(
        plain, _pixels(_state(double=double), scale=scale, battery_percent=None), scale)
    double_box = _changed_box(
        plain, _pixels(_state(double=None), scale=scale, **reading), scale)
    assert _apart(count, battery_box) and _apart(count, double_box)
    # Just past the battery's right end, on its line and no taller than it.
    assert battery_box[2] <= count[0] <= battery_box[2] + 5
    assert count[2] - count[0] <= 16
    assert battery_box[1] <= count[1] and count[3] <= battery_box[3]
    middle, battery_middle = (count[1] + count[3]) / 2, (battery_box[1] + battery_box[3]) / 2
    assert abs(middle - battery_middle) <= 1.0
    # Clear of the keys on the right and of the focus gauge below.
    assert count[2] < 256 and count[3] <= 22


@pytest.mark.parametrize("scale", [1, 2])
def test_the_developing_count_is_amber_and_two_digits_are_wider_than_one(scale):
    plain = _pixels(scale=scale)
    one = _pixels(_state(developing=3), scale=scale)
    two = _pixels(_state(developing=12), scale=scale)
    box = _changed_box(plain, one, scale)
    assert _near(_crop(one, box, scale), AMBER, 60).any()
    wide = _changed_box(plain, two, scale)
    assert wide[2] - wide[0] > box[2] - box[0]


@pytest.mark.parametrize("scale", [1, 2])
def test_a_count_above_99_is_drawn_as_99(scale):
    assert np.array_equal(_pixels(_state(developing=250), scale=scale),
                          _pixels(_state(developing=99), scale=scale))
    assert not np.array_equal(_pixels(_state(developing=98), scale=scale),
                              _pixels(_state(developing=99), scale=scale))


@pytest.mark.parametrize("scale", [1, 2])
def test_the_developing_count_is_drawn_without_a_battery_or_a_2x_mark(scale):
    state = _state(double=None)
    plain = _pixels(state, scale=scale, battery_percent=None)
    assert (_crop(plain, (0, 0, 120, 22), scale) == np.array(FRAME_GREY)).all()
    count = _changed_box(
        plain, _pixels(_state(double=None, developing=2), scale=scale, battery_percent=None),
        scale)
    # Where the battery tag would start, clear of the place of the 2x mark.
    assert DOUBLE_BOX[2] < count[0] and count[2] < 60
    assert DOUBLE_BOX[1] <= count[1] and count[3] <= DOUBLE_BOX[3]


@pytest.mark.parametrize("scale", [1, 2])
def test_no_tuning_mismatch_draws_nothing(scale):
    for double in (None, (True, 1)):
        for battery in (None, 100):
            for developing in (0, 9):
                as_before = LiveState(cells=_cells(), selected=1, pressed=None, double=double,
                                      developing=developing)
                off = LiveState(cells=_cells(), selected=1, pressed=None, double=double,
                                developing=developing, tuning_mismatch=False)
                assert np.array_equal(
                    _pixels(as_before, scale=scale, battery_percent=battery),
                    _pixels(off, scale=scale, battery_percent=battery))


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("developing", [0, 9, 99])
@pytest.mark.parametrize("battery, on_ac", [(None, None), (5, False), (100, True)])
@pytest.mark.parametrize("double", [None, (True, 1)])
def test_the_tuning_tag_is_last_in_the_tag_row_and_inside_its_box(
        scale, developing, battery, on_ac, double):
    reading = dict(battery_percent=battery, external_power=on_ac)
    plain = _pixels(_state(double=double, developing=developing), scale=scale, **reading)
    tagged = _pixels(_state(double=double, developing=developing, tuning_mismatch=True),
                     scale=scale, **reading)
    tag = _changed_box(plain, tagged, scale)
    assert _near(_crop(tagged, tag, scale), AMBER, 60).any()
    # Past everything else in the row, which it leaves as it was.
    row = (_changed_box(_pixels(_state(double=None), scale=scale, battery_percent=None),
                        plain, scale)
           if double is not None or battery is not None or developing else None)
    if row is not None:
        assert row[2] <= tag[0] <= row[2] + 5
    else:
        assert DOUBLE_BOX[2] < tag[0]
    # On the battery tag's line and no taller.
    assert DOUBLE_BOX[1] <= tag[1] and tag[3] <= DOUBLE_BOX[3]
    # Drawn whole: it ends short of the tile's right edge, which would clip it.
    assert tag[2] < liveview._TAGS_BOX[2] - 1
    # Clear of every control. Without a battery tag it starts where the developing count
    # would, which the 2x toggle's grown hit region reaches by one unit, as it does the count.
    for x in range(math.floor(tag[0] * scale), math.ceil(tag[2] * scale)):
        for y in range(math.floor(tag[1] * scale), math.ceil(tag[3] * scale)):
            assert hit(x, y, scale) in (Action.NONE, Action.DOUBLE_TOGGLE)


def test_the_tag_box_clears_the_shutter_button_and_the_keys():
    x1 = liveview._TAGS_BOX[2]
    assert x1 < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    assert x1 <= liveview._CONTROLS_BOX[0]
    assert liveview._TAGS_BOX[3] <= liveview.FOCUS_TRACK[1]


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


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("focus", [1e-6, 1e-4, 0.001, 0.0019])
def test_a_focus_score_too_small_for_one_pixel_draws_an_empty_gauge(focus, scale):
    # Seen on the Pi 2026-10-15: a sharpness score just above zero made the fill
    # less than half a pixel tall, both edges rounded to the same row, the
    # inclusive far corner then sat above the near one and Pillow raised
    # "y1 must be greater than or equal to y0", which ended the service.
    tile = liveview._focus_tile(focus, focus, scale)
    empty = liveview._focus_tile(0.0, None, scale)
    assert tile.size == empty.size


@pytest.mark.parametrize("box", [(10, 10, 10, 20), (10, 10, 20, 10), (10, 10.1, 20, 10.2),
                                 (10.3, 10, 10.4, 20)])
def test_canvas_shapes_that_round_to_no_pixels_draw_nothing(box):
    canvas = liveview._Canvas((0, 0, 40, 40), 1)
    canvas.rect(box, (255, 255, 255, 255))
    canvas.pill(box, 2, (255, 255, 255, 255))
    assert canvas.finish().getbbox() is None


# The disk-space warning: amber PHOTOS / SD tags between the tag row and the shutter.

def test_the_disk_warning_is_a_late_field_and_empty_by_default():
    assert LiveState(cells=_cells()).disk_warning == ()
    assert list(LiveState.__dataclass_fields__)[-3] == "disk_warning"


@pytest.mark.parametrize("scale", [1, 2])
def test_no_disk_warning_draws_nothing(scale):
    for double in (None, (True, 1)):
        for battery in (None, 100):
            for developing, tuning in ((0, False), (9, True)):
                as_before = LiveState(cells=_cells(), selected=1, pressed=None, double=double,
                                      developing=developing, tuning_mismatch=tuning)
                empty = LiveState(cells=_cells(), selected=1, pressed=None, double=double,
                                  developing=developing, tuning_mismatch=tuning,
                                  disk_warning=())
                assert np.array_equal(
                    _pixels(as_before, scale=scale, battery_percent=battery),
                    _pixels(empty, scale=scale, battery_percent=battery))


def _assert_clear_of_every_control(box, scale):
    for x in range(math.floor(box[0] * scale), math.ceil(box[2] * scale)):
        for y in range(math.floor(box[1] * scale), math.ceil(box[3] * scale)):
            assert hit(x, y, scale) is Action.NONE


WIDEST_DISK = (("PHOTOS", 100), ("SD", 100))


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("warning", [(("PHOTOS", 92),), (("SD", 91),), WIDEST_DISK])
def test_the_disk_tag_sits_between_the_tag_row_and_the_shutter(scale, warning):
    # The fullest tag row there is, so the warning is shown beside it, not over it.
    full_row = dict(double=(True, 1), developing=99, tuning_mismatch=True)
    reading = dict(battery_percent=100, external_power=True)
    plain = _pixels(_state(**full_row), scale=scale, **reading)
    tagged = _pixels(_state(**full_row, disk_warning=warning), scale=scale, **reading)
    tag = _changed_box(plain, tagged, scale)
    assert _near(_crop(tagged, tag, scale), AMBER, 60).any()
    row = _changed_box(_pixels(_state(double=None), scale=scale, battery_percent=None),
                       plain, scale)
    assert row[2] < tag[0]
    assert liveview._TAGS_BOX[2] <= tag[0]
    # On the battery tag's line and no taller.
    assert DOUBLE_BOX[1] <= tag[1] and tag[3] <= DOUBLE_BOX[3]
    # Drawn whole, inside its own tile, and short of the shutter's hit circle.
    assert tag[2] < liveview._DISK_BOX[2] - 1
    assert tag[2] < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    _assert_clear_of_every_control(tag, scale)


@pytest.mark.parametrize("scale", [1, 2])
def test_two_disk_tags_are_photos_then_sd(scale):
    plain = _pixels(scale=scale)
    photos = _changed_box(plain, _pixels(_state(disk_warning=(("PHOTOS", 92),)), scale=scale),
                          scale)
    sd = _changed_box(plain, _pixels(_state(disk_warning=(("SD", 91),)), scale=scale), scale)
    both_pixels = _pixels(_state(disk_warning=(("PHOTOS", 92), ("SD", 91))), scale=scale)
    both = _changed_box(plain, both_pixels, scale)
    # PHOTOS where it is alone, SD after it; the two do not touch.
    assert both[0] == photos[0] and both[2] > photos[2] + (sd[2] - sd[0])
    first = _crop(both_pixels, (0, 0, math.ceil(photos[2]), 22), scale)
    assert np.array_equal(first, _crop(_pixels(_state(disk_warning=(("PHOTOS", 92),)),
                                               scale=scale), (0, 0, math.ceil(photos[2]), 22),
                                       scale))


@pytest.mark.parametrize("scale", [1, 2])
def test_the_disk_tag_shows_the_percent(scale):
    a = _pixels(_state(disk_warning=(("SD", 91),)), scale=scale)
    b = _pixels(_state(disk_warning=(("SD", 97),)), scale=scale)
    assert not np.array_equal(a, b)


def test_the_disk_box_clears_the_tag_row_the_shutter_and_the_keys():
    x0, y0, x1, y1 = liveview._DISK_BOX
    assert liveview._TAGS_BOX[2] <= x0
    assert x1 < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    assert x1 <= liveview._CONTROLS_BOX[0]
    assert y1 <= liveview._TAGS_BOX[3]


# The MENU key: a pill under -, hit over MENU_HIT_BOX (x0, y0 included, x1, y1 excluded).

def test_live_state_has_no_network_field():
    # The addresses moved to the menu; ``ControllerSnapshot.network`` stays.
    assert "network" not in LiveState.__dataclass_fields__
    assert not hasattr(liveview, "_network_tile") and not hasattr(liveview, "_NETWORK_BOX")


@pytest.mark.parametrize("scale", [1, 2])
def test_menu_is_hit_at_its_centre_and_at_the_corners_of_its_box(scale):
    x0, y0, x1, y1 = MENU_HIT_BOX
    for x, y in (MENU_CENTRE, (x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        assert hit(x * scale, y * scale, scale) is Action.MENU, (x, y)
        assert hit(x * scale + scale - 1, y * scale + scale - 1, scale) is Action.MENU, (x, y)
    # Just outside each side is not MENU.
    for x, y in ((x0 - 1, MENU_CENTRE[1]), (x1, MENU_CENTRE[1]),
                 (MENU_CENTRE[0], y0 - 1), (MENU_CENTRE[0], y1)):
        assert hit(x * scale, y * scale, scale) is not Action.MENU, (x, y)


def test_the_menu_box_is_under_minus_and_holds_the_pill_in_the_controls_tile():
    (cx, cy), (w, h) = MENU_CENTRE, MENU_SIZE
    x0, y0, x1, y1 = MENU_HIT_BOX
    # One unit under -'s hit circle (its bottom is y 189), into the strip's meter part.
    assert y0 == MINUS_CENTRE[1] + KEY_HIT_RADIUS + 1
    assert y1 > BAR_TOP and x0 >= CELL_HIT_X0 + CELL_PITCH * len(SELECT_ACTIONS)
    # The pill (y 193 to 205) lies inside the box, under y 208 and in the controls tile.
    assert x0 <= cx - w / 2 and cx + w / 2 <= x1 and y0 <= cy - h / 2 and cy + h / 2 <= y1
    assert cy + h / 2 < BAR_TOP
    tx0, ty0, tx1, ty1 = liveview._CONTROLS_BOX
    assert liveview._CONTROLS_BOX == (256, 8, 318, 207)
    assert tx0 <= cx - w / 2 and cx + w / 2 <= tx1 and ty0 <= cy - h / 2 and cy + h / 2 <= ty1
    assert ty1 <= BAR_TOP


def test_the_shutter_plus_and_minus_still_answer_everywhere_in_their_hit_circles():
    for centre, radius, action in ((SHUTTER_CENTRE, SHUTTER_HIT_RADIUS, Action.SHUTTER),
                                   (PLUS_CENTRE, KEY_HIT_RADIUS, Action.PLUS),
                                   (MINUS_CENTRE, KEY_HIT_RADIUS, Action.MINUS)):
        for y in range(centre[1] - radius, centre[1] + radius + 1):
            for x in range(centre[0] - radius, centre[0] + radius + 1):
                if math.hypot(x - centre[0], y - centre[1]) <= radius:
                    assert hit(x, y) is action, (x, y)


@pytest.mark.parametrize("scale", [1, 2])
def test_the_old_menu_place_is_nothing_again(scale):
    assert hit(287 * scale, 88 * scale, scale) is Action.NONE
    for y in range(78, 99):
        for x in range(264, 311):
            assert hit(x * scale, y * scale, scale) is Action.NONE, (x, y)


def test_no_point_outside_the_box_is_menu():
    x0, y0, x1, y1 = MENU_HIT_BOX
    for y in range(HEIGHT):
        for x in range(WIDTH):
            inside = x0 <= x < x1 and y0 <= y < y1
            assert (hit(x, y) is Action.MENU) is inside, (x, y)


def test_the_cells_hit_columns_in_the_strip_are_unchanged():
    for y in range(CELL_HIT_TOP, HEIGHT):
        for x in range(CELL_HIT_X0, CELL_HIT_X0 + CELL_PITCH * len(SELECT_ACTIONS)):
            assert hit(x, y) is SELECT_ACTIONS[(x - CELL_HIT_X0) // CELL_PITCH], (x, y)


def _without_the_menu_key(monkeypatch):
    """Render with the pill's drawing switched off; the tile cache is emptied both ways."""
    liveview._controls_tile.cache_clear()
    monkeypatch.setattr(liveview, "_menu_key", lambda canvas, lit: None)


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("pressed", [None, Action.PLUS, Action.MINUS, Action.SHUTTER,
                                     Action.MENU])
def test_the_menu_key_changes_nothing_outside_its_hit_box(scale, pressed, monkeypatch):
    states = [_state(pressed=pressed),
              LiveState(cells=_cells(), selected=None, pressed=pressed, double=(True, 1),
                        developing=9, tuning_mismatch=True, disk_warning=WIDEST_DISK)]
    readings = [dict(), dict(battery_percent=None, focus=0.5, focus_peak=0.7)]
    liveview._controls_tile.cache_clear()
    try:
        with_key = [_pixels(s, scale=scale, frame=f, **r)
                    for s in states for r in readings for f in (0, 255)]
        with monkeypatch.context() as m:
            _without_the_menu_key(m)
            without = [_pixels(s, scale=scale, frame=f, **r)
                       for s in states for r in readings for f in (0, 255)]
    finally:
        liveview._controls_tile.cache_clear()
    x0, y0, x1, y1 = MENU_HIT_BOX
    outside = np.ones((HEIGHT * scale, WIDTH * scale), dtype=bool)
    outside[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = False
    for a, b in zip(with_key, without, strict=True):
        assert not np.array_equal(a, b)  # the key is drawn at all
        assert np.array_equal(a[outside], b[outside])
        # Nothing in the strip, and nothing above y 190.
        assert np.array_equal(a[BAR_TOP * scale:], b[BAR_TOP * scale:])
        assert np.array_equal(a[:y0 * scale], b[:y0 * scale])


@pytest.mark.parametrize("scale", [1, 2])
def test_the_menu_key_is_an_outlined_pill_with_a_white_label(scale):
    pixels = _pixels(frame=255, scale=scale)
    (cx, cy), (w, h) = MENU_CENTRE, MENU_SIZE
    pill = _crop(pixels, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), scale)
    assert _near(pill, AMBER, 90).any()  # the outline
    inner = _crop(pixels, (cx - w / 2 + 3, cy - h / 2 + 2, cx + w / 2 - 3, cy + h / 2 - 2),
                  scale)
    assert inner.max(axis=-1).min() < 200  # the translucent fill
    # The label: white ink on the fill, over a black frame.
    dark = _crop(_pixels(frame=0, scale=scale),
                 (cx - w / 2 + 3, cy - h / 2 + 2, cx + w / 2 - 3, cy + h / 2 - 2), scale)
    assert _neutral_peak(dark) > WHITE_INK
    # Outside the pill, the untouched frame.
    assert tuple(pixels[cy * scale, (cx - w // 2 - 2) * scale]) == (255, 255, 255)


# The IMG reminder: an image control is not 1.0. First in the disk tags' tile.

def test_the_image_flag_is_a_late_field_and_off_by_default():
    assert LiveState(cells=_cells()).image_adjusted is False
    assert list(LiveState.__dataclass_fields__)[-2] == "image_adjusted"


@pytest.mark.parametrize("scale", [1, 2])
def test_no_image_flag_draws_nothing(scale):
    for warning in ((), WIDEST_DISK):
        assert np.array_equal(
            _pixels(_state(disk_warning=warning), scale=scale),
            _pixels(_state(disk_warning=warning, image_adjusted=False), scale=scale))


@pytest.mark.parametrize("scale", [1, 2])
def test_the_img_tag_sits_first_in_the_disk_tile(scale):
    full_row = dict(double=(True, 1), developing=99, tuning_mismatch=True)
    reading = dict(battery_percent=100, external_power=True)
    plain = _pixels(_state(**full_row), scale=scale, **reading)
    tagged = _pixels(_state(**full_row, image_adjusted=True), scale=scale, **reading)
    tag = _changed_box(plain, tagged, scale)
    assert _near(_crop(tagged, tag, scale), AMBER, 60).any()
    assert liveview._TAGS_BOX[2] <= tag[0]
    assert DOUBLE_BOX[1] <= tag[1] and tag[3] <= DOUBLE_BOX[3]
    _assert_clear_of_every_control(tag, scale)
    # With the widest disk warning beside it, all of it still fits its tile and
    # ends short of the shutter's hit circle; the disk tags move right past it.
    alone = _changed_box(plain, _pixels(_state(**full_row, disk_warning=WIDEST_DISK),
                                        scale=scale, **reading), scale)
    both = _changed_box(plain, _pixels(_state(**full_row, disk_warning=WIDEST_DISK,
                                              image_adjusted=True), scale=scale, **reading),
                        scale)
    assert both[0] == tag[0] and both[2] > alone[2]
    assert both[2] < liveview._DISK_BOX[2] - 1
    assert both[2] < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    _assert_clear_of_every_control(both, scale)


# The STAB reminder: a help against camera shake is on. Right after IMG.

def test_the_stabilise_flag_is_the_last_field_and_off_by_default():
    assert LiveState(cells=_cells()).stabilise_on is False
    assert list(LiveState.__dataclass_fields__)[-1] == "stabilise_on"


@pytest.mark.parametrize("scale", [1, 2])
def test_no_stabilise_flag_draws_nothing(scale):
    for image in (False, True):
        for warning in ((), WIDEST_DISK):
            assert np.array_equal(
                _pixels(_state(disk_warning=warning, image_adjusted=image), scale=scale),
                _pixels(_state(disk_warning=warning, image_adjusted=image,
                               stabilise_on=False), scale=scale))


@pytest.mark.parametrize("scale", [1, 2])
def test_the_stab_tag_sits_after_img_and_everything_still_fits(scale):
    full_row = dict(double=(True, 1), developing=99, tuning_mismatch=True)
    reading = dict(battery_percent=100, external_power=True)
    plain = _pixels(_state(**full_row), scale=scale, **reading)
    alone = _changed_box(plain, _pixels(_state(**full_row, stabilise_on=True), scale=scale,
                                        **reading), scale)
    img = _changed_box(plain, _pixels(_state(**full_row, image_adjusted=True), scale=scale,
                                      **reading), scale)
    stab = _pixels(_state(**full_row, stabilise_on=True), scale=scale, **reading)
    assert _near(_crop(stab, alone, scale), AMBER, 60).any()
    # Alone it is first in the tile, where IMG would be; with IMG it follows it.
    assert alone[0] == img[0]
    assert DOUBLE_BOX[1] <= alone[1] and alone[3] <= DOUBLE_BOX[3]
    _assert_clear_of_every_control(alone, scale)
    both = _changed_box(plain, _pixels(_state(**full_row, image_adjusted=True,
                                              stabilise_on=True), scale=scale, **reading),
                        scale)
    assert both[0] == img[0] and both[2] > img[2]
    # Everything at once: IMG, STAB and the widest disk warning end short of the
    # tile's edge and of the shutter's hit circle.
    every = _changed_box(plain, _pixels(_state(**full_row, image_adjusted=True,
                                               stabilise_on=True, disk_warning=WIDEST_DISK),
                                        scale=scale, **reading), scale)
    disk_and_img = _changed_box(plain, _pixels(_state(**full_row, image_adjusted=True,
                                                      disk_warning=WIDEST_DISK),
                                               scale=scale, **reading), scale)
    assert every[2] > disk_and_img[2]
    assert every[2] < liveview._DISK_BOX[2] - 1
    assert every[2] < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    _assert_clear_of_every_control(every, scale)
