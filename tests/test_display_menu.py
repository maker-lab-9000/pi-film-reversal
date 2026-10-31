import numpy as np
import pytest

from pifilm.display import menu
from pifilm.display.liveview import (
    AMBER,
    HEIGHT,
    SEMIBOLD,
    SHUTTER_CENTRE,
    WIDTH,
    _font,
)
from pifilm.display.menu import (
    DONE,
    ROW_HEIGHT,
    ROW_X0,
    ROW_X1,
    ROW_Y0,
    MenuRow,
    fit_text,
    menu_hit,
    render_menu,
)

# Restated on purpose, as in the live view's tests: DONE is hit like the shutter.
SHUTTER_HIT_RADIUS = 34

ROWS = (
    MenuRow("save_dng", "SAVE DNG", "ON", True),
    MenuRow("halation", "HALATION", "OFF", True),
    MenuRow("ae_constraint", "AE CONSTRAINT", "HIGHLIGHT", True),
    MenuRow("ae_metering", "AE METERING", "CENTRE", True),
    MenuRow("image", "IMAGE", ">", True),
    MenuRow("stabilise", "STABILISE", ">", True),
    MenuRow("upload", "UPLOAD", "TAP TO UPLOAD", True),
    MenuRow("eth", "ETH", "-", False),
    MenuRow("wlan", "WLAN", "192.168.178.56", False),
    MenuRow("look", "LOOK", "imx477-2026-09-v4-dark-hp5-tile-with-a-much-longer-name", False),
    MenuRow("reset", "RESET", "TAP TWICE", True),
)
LONG = "imx477-2026-09-v4-dark-hp5-tile-with-a-much-longer-name"


def _pixels(rows=ROWS, *, scale=1, done_pressed=False):
    return np.asarray(render_menu(rows, scale=scale, done_pressed=done_pressed)).astype(int)


@pytest.mark.parametrize("scale", [1, 2])
def test_every_row_is_hit_at_its_centre(scale):
    for index in range(len(ROWS)):
        x = (ROW_X0 + ROW_X1) // 2
        y = ROW_Y0 + index * ROW_HEIGHT + ROW_HEIGHT // 2
        assert menu_hit(x * scale, y * scale, len(ROWS), scale) == index


@pytest.mark.parametrize("scale", [1, 2])
def test_done_is_hit_at_the_shutters_centre_and_at_its_hit_edge(scale):
    cx, cy = SHUTTER_CENTRE
    assert menu_hit(cx * scale, cy * scale, len(ROWS), scale) == DONE
    assert menu_hit((cx - SHUTTER_HIT_RADIUS) * scale, cy * scale, len(ROWS), scale) == DONE
    assert menu_hit(cx * scale, (cy + SHUTTER_HIT_RADIUS) * scale, len(ROWS), scale) == DONE
    assert menu_hit((cx - SHUTTER_HIT_RADIUS - 1) * scale, cy * scale, len(ROWS),
                    scale) is None


@pytest.mark.parametrize("scale", [1, 2])
def test_the_gap_between_the_rows_and_done_is_nothing(scale):
    y = ROW_Y0 + ROW_HEIGHT // 2
    for x in range(ROW_X1, ROW_X1 + 5):  # 248 to 252
        assert menu_hit(x * scale, y * scale, len(ROWS), scale) is None


@pytest.mark.parametrize("scale", [1, 2])
def test_below_the_last_row_and_left_of_the_rows_is_nothing(scale):
    bottom = ROW_Y0 + len(ROWS) * ROW_HEIGHT
    assert menu_hit(100 * scale, bottom * scale, len(ROWS), scale) is None
    assert menu_hit(100 * scale, (bottom - 1) * scale, len(ROWS), scale) == len(ROWS) - 1
    assert menu_hit(100 * scale, (HEIGHT - 1) * scale, len(ROWS), scale) is None
    assert menu_hit((ROW_X0 - 1) * scale, (ROW_Y0 + 1) * scale, len(ROWS), scale) is None
    assert menu_hit(100 * scale, (ROW_Y0 - 1) * scale, len(ROWS), scale) is None


def test_fewer_rows_hit_fewer_rows():
    y = ROW_Y0 + 3 * ROW_HEIGHT + 1
    assert menu_hit(100, y, 4, 1) == 3
    assert menu_hit(100, y, 3, 1) is None


def test_fit_text_keeps_a_short_text_and_cuts_a_long_one_to_fit():
    font = _font(SEMIBOLD, 36)
    assert fit_text("ON", font, 200) == "ON"
    width = font.getlength(LONG) / 2
    cut = fit_text(LONG, font, width)
    assert cut.endswith("...")
    assert LONG.startswith(cut[:-3])
    assert len(cut) > 3
    assert font.getlength(cut) <= width


def test_fit_text_with_no_room_returns_what_fits_of_the_ellipsis():
    font = _font(SEMIBOLD, 36)
    assert font.getlength(fit_text(LONG, font, 1)) <= 1


@pytest.mark.parametrize("scale", [1, 2])
def test_render_is_an_rgb_page_of_the_panel_size(scale):
    img = render_menu(ROWS, scale=scale)
    assert img.mode == "RGB"
    assert img.size == (WIDTH * scale, HEIGHT * scale)


@pytest.mark.parametrize("scale", [1, 2])
def test_no_row_ink_falls_past_the_rows_right_edge(scale):
    pixels = _pixels(scale=scale)
    gap = pixels[:, 249 * scale:253 * scale]
    assert gap.max() == 0
    # The rows are drawn at all, and the values reach towards the right edge.
    rows = pixels[ROW_Y0 * scale:(ROW_Y0 + len(ROWS) * ROW_HEIGHT) * scale]
    assert rows[:, 200 * scale:248 * scale].max() > 150


@pytest.mark.parametrize("scale", [1, 2])
def test_the_page_is_black_away_from_the_rows_and_done(scale):
    pixels = _pixels(scale=scale)
    below = pixels[(ROW_Y0 + len(ROWS) * ROW_HEIGHT + 1) * scale:]
    assert below.max() == 0


@pytest.mark.parametrize("scale", [1, 2])
def test_inactive_labels_are_darker_than_active_ones(scale):
    rows = (MenuRow("eth", "ETH", "-", True), MenuRow("eth", "ETH", "-", False))
    pixels = _pixels(rows, scale=scale)

    def label_mean(index):
        y0 = (ROW_Y0 + index * ROW_HEIGHT) * scale
        return pixels[y0:y0 + (ROW_HEIGHT - 1) * scale, ROW_X0 * scale:40 * scale].mean()

    assert label_mean(1) < label_mean(0)
    assert label_mean(1) > 0


@pytest.mark.parametrize("scale", [1, 2])
def test_done_is_lit_amber_while_pressed(scale):
    cx, cy = (v * scale for v in SHUTTER_CENTRE)
    # A point inside the disc, clear of the DONE label.
    probe = (cy + 18 * scale, cx)
    idle, pressed = _pixels(scale=scale), _pixels(scale=scale, done_pressed=True)
    assert np.abs(pressed[probe] - np.array(AMBER)).max() <= 40
    assert idle[probe].max() < 40
    # Its ring is amber either way.
    ring = (cy, cx - 28 * scale)
    assert idle[ring][0] > 100 and idle[ring][2] < 60


def test_the_render_is_cached():
    assert render_menu(ROWS, scale=1) is render_menu(ROWS, scale=1)
    assert render_menu(ROWS, scale=1) is not render_menu(ROWS, scale=1, done_pressed=True)


def test_menu_row_is_frozen_and_hashable():
    row = MenuRow("eth", "ETH", "-", False)
    assert hash(row) == hash(MenuRow("eth", "ETH", "-", False))
    with pytest.raises(AttributeError):
        row.value = "x"  # type: ignore[misc]


def test_the_rows_stay_clear_of_dones_hit_circle():
    assert menu.ROW_X1 < SHUTTER_CENTRE[0] - SHUTTER_HIT_RADIUS
    assert menu.VALUE_RIGHT == ROW_X1 - 4


@pytest.mark.parametrize("scale", [1, 2])
def test_the_fullest_page_fits_on_the_panel(scale):
    """The page's capacity, eleven rows (more than any page of the viewfinder's
    tree has): the last one ends above the panel's foot, and every row is hit at
    its centre."""
    bottom = ROW_Y0 + len(ROWS) * ROW_HEIGHT
    assert len(ROWS) == 11 and bottom == 228 and bottom <= HEIGHT
    pixels = _pixels(scale=scale)
    # The last row (RESET) is drawn: its value has white ink.
    last = pixels[(bottom - ROW_HEIGHT) * scale:bottom * scale, 150 * scale:ROW_X1 * scale]
    assert last.max() > 150
    for index in range(len(ROWS)):
        y = ROW_Y0 + index * ROW_HEIGHT + ROW_HEIGHT // 2
        assert menu_hit(ROW_X0 * scale + 1, y * scale, len(ROWS), scale) == index


# -- the second page: a title, then its rows -------------------------------------

IMAGE_ROWS = (
    MenuRow("sharpness", "SHARPNESS", "1.0", True),
    MenuRow("contrast", "CONTRAST", "1.25", True),
    MenuRow("saturation", "SATURATION", "-", False),
    MenuRow("image_reset", "RESET", "TAP TWICE", True),
    MenuRow("back", "BACK", "", True),
)


@pytest.mark.parametrize("scale", [1, 2])
def test_a_titled_page_hits_its_rows_one_line_lower(scale):
    x = (ROW_X0 + ROW_X1) // 2
    for index in range(len(IMAGE_ROWS)):
        y = ROW_Y0 + (index + 1) * ROW_HEIGHT + ROW_HEIGHT // 2
        assert menu_hit(x * scale, y * scale, len(IMAGE_ROWS), scale, title=True) == index
    # The title is not a row, and nothing lies below the last one.
    title_y = ROW_Y0 + ROW_HEIGHT // 2
    assert menu_hit(x * scale, title_y * scale, len(IMAGE_ROWS), scale, title=True) is None
    below = ROW_Y0 + (len(IMAGE_ROWS) + 1) * ROW_HEIGHT + 1
    assert menu_hit(x * scale, below * scale, len(IMAGE_ROWS), scale, title=True) is None
    cx, cy = SHUTTER_CENTRE
    assert menu_hit(cx * scale, cy * scale, len(IMAGE_ROWS), scale, title=True) == DONE


@pytest.mark.parametrize("scale", [1, 2])
def test_a_titled_page_draws_the_title_amber_and_stays_clear_of_done(scale):
    pixels = np.asarray(render_menu(IMAGE_ROWS, scale=scale, title="IMAGE")).astype(int)
    title = pixels[ROW_Y0 * scale:(ROW_Y0 + ROW_HEIGHT) * scale, ROW_X0 * scale:80 * scale]
    reds = title[..., 0] - title[..., 2]
    assert reds.max() > 150  # amber ink: red well above blue
    assert pixels[:, 249 * scale:253 * scale].max() == 0
    bottom = ROW_Y0 + (len(IMAGE_ROWS) + 1) * ROW_HEIGHT
    assert bottom <= HEIGHT
    assert pixels[(bottom + 1) * scale:].max() == 0
    # The rows moved down a line: the last one (BACK) is drawn where it is hit.
    back = pixels[(bottom - ROW_HEIGHT) * scale:bottom * scale, ROW_X0 * scale:60 * scale]
    assert back.max() > 60


def test_a_titled_page_is_cached_apart_from_an_untitled_one():
    titled = render_menu(IMAGE_ROWS, scale=1, title="IMAGE")
    assert titled is render_menu(IMAGE_ROWS, scale=1, title="IMAGE")
    assert titled is not render_menu(IMAGE_ROWS, scale=1)


# -- the STABILISE page: the same kind of titled page ----------------------------

STABILISE_ROWS = (
    MenuRow("short_exposure", "SHORT EXPOSURE", "OFF", True),
    MenuRow("shutter_delay", "SHUTTER DELAY", "0.5 S", True),
    MenuRow("multi_frame", "MULTI-FRAME", "6", True),
    MenuRow("stabilise_reset", "RESET", "TAP AGAIN", True),
    MenuRow("back", "BACK", "", True),
)


@pytest.mark.parametrize("scale", [1, 2])
def test_the_stabilise_page_fits_and_stays_clear_of_done(scale, monkeypatch):
    fitted = []
    real = menu.fit_text

    def spy(text, font, width):
        fitted.append((text, real(text, font, width)))
        return fitted[-1][1]

    monkeypatch.setattr(menu, "fit_text", spy)
    menu._row_tile.cache_clear()
    menu._render.cache_clear()
    pixels = np.asarray(render_menu(STABILISE_ROWS, scale=scale, title="STABILISE")).astype(int)
    # Every value is drawn whole beside its label, the longest label included.
    assert fitted and all(text == shown for text, shown in fitted)
    assert pixels[:, 249 * scale:253 * scale].max() == 0
    bottom = ROW_Y0 + (len(STABILISE_ROWS) + 1) * ROW_HEIGHT
    assert pixels[(bottom + 1) * scale:].max() == 0
    x = (ROW_X0 + ROW_X1) // 2
    for index in range(len(STABILISE_ROWS)):
        y = ROW_Y0 + (index + 1) * ROW_HEIGHT + ROW_HEIGHT // 2
        assert menu_hit(x * scale, y * scale, len(STABILISE_ROWS), scale, title=True) == index


# -- the first page: an empty line before RESET ----------------------------------

FIRST_ROWS = (
    MenuRow("camera", "CAMERA", ">", True),
    MenuRow("pipeline", "PIPELINE", ">", True),
    MenuRow("network", "NETWORK", ">", True),
    MenuRow("look", "LOOK", "imx477-v4", False),
    MenuRow("reset", "RESET", "TAP TWICE", True),
)


@pytest.mark.parametrize("scale", [1, 2])
def test_a_gap_is_an_empty_line_that_is_not_a_row(scale):
    """The line before row ``gap`` hits nothing and the rows from ``gap`` on are
    hit one line lower; nothing lies below the last."""
    x = (ROW_X0 + ROW_X1) // 2
    gap = 4

    def at(line):
        return menu_hit(x * scale, (ROW_Y0 + line * ROW_HEIGHT + ROW_HEIGHT // 2) * scale,
                        len(FIRST_ROWS), scale, gap=gap)

    assert [at(line) for line in range(7)] == [0, 1, 2, 3, None, 4, None]
    # Every pixel row of the empty line, edges included, is nothing.
    for y in range((ROW_Y0 + gap * ROW_HEIGHT) * scale, (ROW_Y0 + (gap + 1) * ROW_HEIGHT) * scale):
        assert menu_hit(x * scale, y, len(FIRST_ROWS), scale, gap=gap) is None
    cx, cy = SHUTTER_CENTRE
    assert menu_hit(cx * scale, cy * scale, len(FIRST_ROWS), scale, gap=gap) == DONE


@pytest.mark.parametrize("scale", [1, 2])
def test_a_gap_is_drawn_empty_and_moves_the_rows_after_it_down(scale):
    pixels = np.asarray(render_menu(FIRST_ROWS, scale=scale, gap=4)).astype(int)
    gap_top, gap_bottom = ROW_Y0 + 4 * ROW_HEIGHT, ROW_Y0 + 5 * ROW_HEIGHT
    # The empty line has nothing but the LOOK row's hairline on its top edge.
    assert pixels[(gap_top + 1) * scale:gap_bottom * scale, :ROW_X1 * scale].max() == 0
    reset = pixels[gap_bottom * scale:(gap_bottom + ROW_HEIGHT) * scale,
                   ROW_X0 * scale:60 * scale]
    assert reset.max() > 60  # RESET's label, one line lower
    assert render_menu(FIRST_ROWS, scale=scale, gap=4) is not render_menu(
        FIRST_ROWS, scale=scale)
