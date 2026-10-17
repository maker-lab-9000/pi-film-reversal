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
    MenuRow("ae_constraint", "AE CONSTRAINT", "HIGHLIGHT", True),
    MenuRow("ae_metering", "AE METERING", "CENTRE", True),
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
