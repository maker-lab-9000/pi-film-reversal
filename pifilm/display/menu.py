"""The settings menu page and its hit regions. Pure.

Design intent:

- A black page. Nothing of the live picture lies behind it, so while the menu is
  open the viewfinder does not ask the frame thread for frames (as for a review)
  and the page is the same whatever the camera sees.
- ``DONE`` sits where the shutter button is, with the shutter's centre and hit
  circle. The thumb that pressed ``MENU`` just below the shutter finds the way
  back in the same place, and a tap that misses ``DONE`` lands on nothing rather
  than on a row: no setting changes by accident on the way out.
- One row per line, label left and value right, 20 units tall from y 8. Rows run
  from x 8 to 248 and ``DONE``'s hit circle (centre 287, radius 34) starts at
  x 253, so the two regions never meet and a gap of nothing lies between them.
  Each row is drawn on a tile that ends at x 248, so no long value can spill
  towards ``DONE`` either; the value is also cut with ``...`` to the room the
  label leaves it.

Drawing reuses the live view's canvas, fonts, colours and supersampling, so the
page reads as the same instrument. Geometry is in 320x240 base units at the
panel's integer ``scale``, as in ``liveview``.

Caching. Each row is a tile cached by what it shows, the ``DONE`` key is a tile
keyed by whether it is pressed, and the page assembled from them is cached by the
rows, the scale and the press. A value that changes every second (an upload's
elapsed time) then redraws one row, not the page. A cached page is shared between
calls and must never be drawn on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageFont

from .liveview import (
    AMBER,
    BLACK,
    GREY,
    HEIGHT,
    LABEL_SIZE,
    LABEL_TRACKING,
    MEDIUM,
    SEMIBOLD,
    SHUTTER_CENTRE,
    SHUTTER_FILL,
    SHUTTER_HIT_RADIUS,
    SHUTTER_RADIUS,
    SHUTTER_RING,
    SUPERSAMPLE,
    TAG_SIZE,
    UNLIT,
    WHITE,
    WIDTH,
    _Canvas,
)


@dataclass(frozen=True)
class MenuRow:
    """One line of the menu: ``value`` is already formatted for display."""

    key: str      # "save_dng", "ae_constraint", "ae_metering", "upload", "eth", ...
    label: str    # "SAVE DNG", ...
    value: str    # "ON", "UPLOADING 0:42", ...
    active: bool  # False: a dim label, and taps on it are ignored


DONE = "done"
ROW_X0, ROW_X1, ROW_Y0, ROW_HEIGHT = 8, 248, 8, 20
VALUE_RIGHT = ROW_X1 - 4
VALUE_SIZE = 9
LABEL_GAP = 8  # the least room between a row's label and its value
SEPARATOR = 0.5  # the hairline under each row, in base units
ELLIPSIS = "..."

_ROW_BOX = (ROW_X0, ROW_Y0, ROW_X1, ROW_Y0 + ROW_HEIGHT)  # row 0
_DONE_BOX = (SHUTTER_CENTRE[0] - SHUTTER_RADIUS - 1, SHUTTER_CENTRE[1] - SHUTTER_RADIUS - 1,
             SHUTTER_CENTRE[0] + SHUTTER_RADIUS + 1, SHUTTER_CENTRE[1] + SHUTTER_RADIUS + 1)


def fit_text(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
             width: float) -> str:
    """``text``, or its longest start followed by ``...`` that is at most ``width``
    wide, measured in ``font``'s own pixels. With too little room even for the
    ellipsis, as much of the ellipsis as fits (possibly nothing)."""
    if font.getlength(text) <= width:
        return text
    for end in range(len(text) - 1, -1, -1):
        cut = text[:end].rstrip() + ELLIPSIS
        if font.getlength(cut) <= width:
            return cut
    for end in range(len(ELLIPSIS) - 1, -1, -1):
        if font.getlength(ELLIPSIS[:end]) <= width:
            return ELLIPSIS[:end]
    return ""


@lru_cache(maxsize=64)
def _row_tile(row: MenuRow, scale: int) -> Image.Image:
    """One row, drawn as row 0; the caller pastes it at its own line."""
    canvas = _Canvas(_ROW_BOX, scale)
    middle = ROW_Y0 + ROW_HEIGHT / 2
    small = canvas.font(MEDIUM, LABEL_SIZE)
    canvas.text((ROW_X0, middle), row.label, small, GREY if row.active else UNLIT,
                tracking=LABEL_TRACKING)
    font = canvas.font(SEMIBOLD, VALUE_SIZE)
    room = VALUE_RIGHT - (ROW_X0 + canvas.text_width(row.label, small, LABEL_TRACKING)
                          + LABEL_GAP)
    value = fit_text(row.value, font, max(0.0, room) * scale * SUPERSAMPLE)
    if value:
        canvas.text((VALUE_RIGHT - canvas.text_width(value, font), middle), value, font,
                    WHITE)
    bottom = _ROW_BOX[3]
    canvas.rect((ROW_X0, bottom - SEPARATOR, ROW_X1, bottom), UNLIT + (255,))
    return canvas.finish()


@lru_cache(maxsize=4)
def _done_tile(pressed: bool, scale: int) -> Image.Image:
    """``DONE`` in the shutter button's ring style; filled amber while pressed."""
    canvas = _Canvas(_DONE_BOX, scale)
    if pressed:
        canvas.disc(SHUTTER_CENTRE, SHUTTER_RADIUS, fill=AMBER + (255,))
    else:
        canvas.disc(SHUTTER_CENTRE, SHUTTER_RADIUS, fill=SHUTTER_FILL,
                    outline=AMBER + (255,), width=SHUTTER_RING)
    canvas.text(SHUTTER_CENTRE, "DONE", canvas.font(SEMIBOLD, TAG_SIZE),
                BLACK if pressed else WHITE, anchor="mm")
    return canvas.finish()


@lru_cache(maxsize=16)
def _render(rows: tuple[MenuRow, ...], scale: int, done_pressed: bool) -> Image.Image:
    img = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BLACK)
    for index, row in enumerate(rows):
        tile = _row_tile(row, scale)
        img.paste(tile, (ROW_X0 * scale, (ROW_Y0 + index * ROW_HEIGHT) * scale), tile)
    tile = _done_tile(done_pressed, scale)
    img.paste(tile, (_DONE_BOX[0] * scale, _DONE_BOX[1] * scale), tile)
    return img


def render_menu(rows: Sequence[MenuRow], *, scale: int = 1,
                done_pressed: bool = False) -> Image.Image:
    """The menu page: RGB, ``(320 * scale, 240 * scale)``. Shared; do not draw on it."""
    return _render(tuple(rows), scale, done_pressed)


def menu_hit(x: int, y: int, row_count: int, scale: int = 1) -> int | str | None:
    """``DONE``, the index of the row under a panel pixel, or None."""
    x, y = x // scale, y // scale
    if math.hypot(x - SHUTTER_CENTRE[0], y - SHUTTER_CENTRE[1]) <= SHUTTER_HIT_RADIUS:
        return DONE
    if ROW_X0 <= x < ROW_X1 and ROW_Y0 <= y < ROW_Y0 + row_count * ROW_HEIGHT:
        return (y - ROW_Y0) // ROW_HEIGHT
    return None
