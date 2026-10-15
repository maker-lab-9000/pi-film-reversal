"""The live viewfinder screen in the style of an SLR finder, and its hit regions. Pure.

Design intent:

- The image fills the screen. Nothing translucent lies across it except the keys
  themselves, so what is framed is what is seen.
- All read-outs sit in one solid black strip along the bottom, amber on black like
  the display under an SLR's focusing screen: readable against any scene, and it
  never changes brightness with the picture.
- One select-and-adjust control set. The strip's three cells (SHUTTER, ISO, WB) are
  the selectors and a single ``+``/``-`` pair changes whichever is selected, instead
  of a pair of keys per control. A control the camera cannot change is drawn grey.
- The shutter button sits at the top right and ``+``/``-`` below it with a wide gap
  (31 units between the hit circles): a thumb that slips while stepping a value
  must not take a photograph. The hit regions are stated next to the drawn
  geometry, from the same constants, and never overlap.

Geometry is in 320x240 base units, as in ``ui.py``; every render takes the panel's
integer ``scale`` (1 for the 2.8" SPI panel, 2 for the 640x480 DSI panel). Unlike
``ui.py`` the coordinates here are continuous: a box ``(x0, y0, x1, y1)`` covers
the area between those edges, and a circle's centre is a point, not a pixel.
Fractional values (the 21.5 radius, a 1.25 wide ring) are multiplied by the drawing
scale and then rounded to whole pixels.

Anti-aliasing and caching. Thin rings and small condensed type look ragged when
Pillow draws them straight at panel resolution, so every piece is drawn at twice
the panel scale on a transparent layer and reduced with Lanczos. That is far too
slow to do for the whole screen ten times a second on a Pi 4, and almost none of
it changes between frames, so the screen is assembled from small tiles:

- static tiles, cached with ``functools.lru_cache`` and keyed by exactly what
  they show: the strip background, one tile per cell, the shutter and ``+``/``-``
  keys (keyed by which is pressed), and the ``2x`` and battery tags with the
  count of photographs still developing and the ``TUNING`` warning, and the
  ``PHOTOS``/``SD`` disk-space tags (a tile of their own, keyed by the warning),
  and the ``ETH``/``WLAN`` network addresses (a tile of their own at the bottom
  left, keyed by the addresses);
- dynamic tiles: the lit meter dot and the lux/clip text (cached too, they repeat)
  and the focus gauge (drawn every frame; it is 7 units wide).

A frame is then the letterboxed preview plus a handful of pastes. The cells are
separate tiles because an automatic control shows its live metered value: a
changing shutter read-out must repaint one 61x32 tile, not the keys. Cached tiles
are shared between frames and must never be drawn on.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meter import MeterReading
from .ui import _letterbox

WIDTH, HEIGHT = 320, 240
BAR_TOP = 208
# Cells: text starts at CELL_X0 + CELL_PITCH * i. The hit column starts 3 units
# further left and is one pitch wide, so the three columns tile the strip's left
# part without gaps or overlap.
CELL_X0, CELL_PITCH = 11, 61
CELL_HIT_X0 = 8
# Taps land a little above the strip as often as on it.
CELL_HIT_TOP = 204
# Pillow "middle" anchors (half way between the ascender and descender lines).
# The plan put the value at 229, measured in the mock-up's typeface; Barlow
# Condensed's figures hang lower from that line, so 229 stood them on the
# underline (and the slash of "1/2000" through it). 226 puts their feet at y 233,
# where the approved mock-up has them, four units clear of the underline.
CELL_LABEL_Y, CELL_VALUE_Y = 215, 226
CELL_UNDERLINE = (0, 237, 46, 238)  # relative to the cell's text origin in x
METER_X0, METER_X1, METER_Y = 205, 306, 226
METER_DOTS = 13  # -3.0 to +3.0 stops in half stops
METER_RANGE = 3.0
METER_END_OFFSET = 7  # the - and + glyphs, outside the first and last dot
METER_TEXT_CENTRE = (255.5, 215)
SHUTTER_CENTRE, SHUTTER_RADIUS = (287, 39), 28
SHUTTER_DISC_RADIUS = 21.5
SHUTTER_HIT_RADIUS = 34
PLUS_CENTRE, MINUS_CENTRE, KEY_RADIUS = (287, 125), (287, 168), 17
KEY_HIT_RADIUS = 21
DOUBLE_BOX = (6, 5, 30, 16)
DOUBLE_HIT_MARGIN = 6
TAG_RADIUS = 5.5
TAG_PADDING = 6  # each side of a tag's text
BATTERY_X0 = 35
# The amber 2x tag widens to its "2x 1/2" label but stops this far short of the
# battery tag, whatever font is in use.
TAG_GAP = 2
# The count of photographs still being graded: an amber number on a dark patch just
# past the battery tag's right end (or where that tag would start, without one).
# It cannot go on the tag's left, where the 2x tag ends two units away. The patch
# has no outline, so it does not read as a third key, and is darker than a key's
# fill: amber on a key's grey is hard to read against a bright sky. Two digits are
# the limit: the queue holds three, and only a backlog recovered at start-up goes
# higher.
DEVELOPING_GAP = 3
DEVELOPING_PADDING = 3  # each side of the number
DEVELOPING_MAX = 99
DEVELOPING_FILL = (0, 0, 0, 190)
# The camera's libcamera tuning is not the one the look was trained under: amber
# TUNING on the same dark patch as the count, last in the row. No outline either,
# since it is a warning and not a key, and the battery tag's padding, since it is a
# word like the battery's. It is the widest thing the row can end with, which is
# what sets the right edge of ``_TAGS_BOX``.
TUNING_LABEL = "TUNING"
# A filesystem at or above the disk warning's threshold (``pifilm.capture.disk``):
# amber "PHOTOS 92%" / "SD 91%" on the same dark patch as TUNING, in a tile of its
# own past the tag row and short of the shutter's hit circle, photos first. Its own
# tile, at a fixed place, so it never moves the row and the row never moves it.
DISK_X0 = 127
FOCUS_TRACK = (6, 22, 9, 198)
FOCUS_TICK_X = (4, 11)
FOCUS_TICK_HEIGHT = 1.5

AMBER = (255, 176, 0)
DIM_AMBER = (150, 106, 8)
WHITE = (245, 245, 240)
GREY = (125, 125, 120)
UNLIT = (95, 95, 90)
BLACK = (0, 0, 0)
KEY_FILL = (0, 0, 0, 110)
SHUTTER_FILL = (0, 0, 0, 70)
SHUTTER_DISC = (255, 255, 255, 215)
KEY_OUTLINE, SHUTTER_RING = 0.75, 1.25

SEMIBOLD, MEDIUM = "BarlowCondensed-SemiBold.ttf", "BarlowCondensed-Medium.ttf"
LABEL_SIZE, VALUE_SIZE, TAG_SIZE, KEY_GLYPH_SIZE, METER_GLYPH_SIZE = 5, 18, 6.5, 15, 7
LABEL_TRACKING = 0.8
# One placeholder for a missing read-out, as on the other screens: ASCII, because
# the fallback font has no dash glyphs beyond the hyphen.
DASH = "-"
# Each tile is drawn at this multiple of the panel scale, then reduced.
SUPERSAMPLE = 2

# Tile boxes in base units; whole numbers, so a tile lands on whole panel pixels.
_STRIP_BOX = (0, BAR_TOP, WIDTH, HEIGHT)
_CELL_BOX = (CELL_HIT_X0, BAR_TOP, CELL_HIT_X0 + CELL_PITCH, HEIGHT)  # cell 0
_CONTROLS_BOX = (256, 8, 318, 188)
# Wide enough for "AC 100%", a two-digit count and TUNING (they end at 115.5 units
# at scale 1, 114 at scale 2), and well short of the shutter's hit circle at 253.
_TAGS_BOX = (0, 0, 124, 22)
# From the tag box's end to the last whole unit before the shutter's hit circle
# (centre 287, radius 34: its leftmost point is x 253). "PHOTOS 100%" and "SD 100%"
# together end at about 207 units (see the tests), so there is room to spare.
_DISK_BOX = (_TAGS_BOX[2], 0, 252, _TAGS_BOX[3])
_METER_TEXT_BOX = (192, 209, WIDTH, 221)
# The Pi's IPv4 addresses (``pifilm.capture.network``): "ETH 192.168.1.20" and
# "WLAN 10.42.0.1" on the same dark patch, ETH first, in a tile of their own at the
# bottom left of the picture, just above the strip and right of the focus gauge.
# White, not amber: information, not a warning. Drawn only, never hit-tested: its
# lowest units lie in the band where taps on the cells are accepted, and a tap
# there still selects the cell under it.
#
# From just right of the focus gauge's ticks (x 4 to 11) to well short of the meter
# text (x 192), and from 196 to one unit above the strip (208): as tall as the
# battery tag's pill (11 units). "ETH 255.255.255.255" and "WLAN 255.255.255.255"
# together end at 148 units at scale 1 (140 at scale 2), so 168 leaves room for a
# plainer fallback font.
_NETWORK_BOX = (14, 196, 168, BAR_TOP - 1)
_FOCUS_BOX = (FOCUS_TICK_X[0], FOCUS_TRACK[1], FOCUS_TICK_X[1], FOCUS_TRACK[3])


class Action(Enum):
    NONE = "none"
    SHUTTER = "shutter"
    DOUBLE_TOGGLE = "double_toggle"
    SELECT_SHUTTER = "select_shutter"
    SELECT_ISO = "select_iso"
    SELECT_WB = "select_wb"
    PLUS = "plus"
    MINUS = "minus"


SELECT_ACTIONS = (Action.SELECT_SHUTTER, Action.SELECT_ISO, Action.SELECT_WB)  # index = cell
# The keys that light under a finger (``LiveState.pressed``). The cells and the 2x
# toggle show a tap by changing state instead. The viewfinder loop imports this.
PRESS_KEYS = (Action.PLUS, Action.MINUS, Action.SHUTTER)


@dataclass(frozen=True)
class Cell:
    label: str            # "SHUTTER", "ISO", "WB"
    value: str            # "1/125", "400", "5200K", "MAX", "-"
    auto: bool = True     # draws "AUTO" beside the label
    enabled: bool = True  # False: drawn grey, cannot be selected


@dataclass(frozen=True)
class LiveState:
    cells: tuple[Cell, Cell, Cell]
    selected: int | None = 0                 # index into cells; None when nothing is selectable
    pressed: Action | None = None            # PLUS, MINUS or SHUTTER while lit
    double: tuple[bool, int] | None = None   # (enabled, exposures taken); None hides the 2x toggle
    developing: int = 0                      # photographs still being graded; 0 draws nothing
    tuning_mismatch: bool = False            # camera tuning differs from the look's; draws TUNING
    disk_warning: tuple[tuple[str, int], ...] = ()  # (label, percent) per low filesystem
    network: tuple[tuple[str, str], ...] = ()  # (label, IPv4 address), ETH before WLAN


@lru_cache(maxsize=4)
def _font_bytes(name: str) -> bytes:
    return (resources.files("pifilm") / "data" / "fonts" / name).read_bytes()


@lru_cache(maxsize=32)
def _font(name: str, pixels: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A bundled font at a pixel size, or Pillow's default if it cannot be loaded.

    A viewfinder in a plainer typeface is still a viewfinder; one that raises on
    its first frame because a data file went missing from an install is not.
    Loaded from bytes so it also works from a zipped package.
    """
    try:
        return ImageFont.truetype(io.BytesIO(_font_bytes(name)), pixels)
    except OSError:
        return ImageFont.load_default(pixels)


def _half_up(value: float) -> int:
    # Python's round() rounds halves to even, which would make a 1.25 wide ring
    # thinner at scale 1 (2.5 -> 2) than its neighbours' rounding suggests.
    return int(math.floor(value + 0.5))


class _Canvas:
    """A transparent tile, drawn on in base units and reduced to panel pixels.

    ``box`` is the tile's place on the screen in whole base units; drawing calls
    take screen coordinates, so the layout constants are used as they stand.
    """

    def __init__(self, box: tuple[int, int, int, int], scale: int) -> None:
        self._x0, self._y0 = box[0], box[1]
        self._k = scale * SUPERSAMPLE
        self._size = ((box[2] - box[0]) * scale, (box[3] - box[1]) * scale)
        self._image = Image.new(
            "RGBA", (self._size[0] * SUPERSAMPLE, self._size[1] * SUPERSAMPLE), (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._image)

    def _x(self, x: float) -> int:
        return _half_up((x - self._x0) * self._k)

    def _y(self, y: float) -> int:
        return _half_up((y - self._y0) * self._k)

    def _box(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        # Pillow's far corner is inclusive; the boxes here are edge to edge.
        return (self._x(box[0]), self._y(box[1]), self._x(box[2]) - 1, self._y(box[3]) - 1)

    def _width(self, width: float) -> int:
        return max(1, _half_up(width * self._k))

    def font(self, name: str, size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return _font(name, _half_up(size * self._k))

    def _pixels(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
        """``_box``, or None when the box covers no whole pixel.

        A box thinner than half a pixel rounds both edges to the same row or
        column, and the inclusive far corner then lies before the near one, which
        Pillow refuses with a ValueError. Seen on the Pi on 2026-10-15: a sharpness
        score just above zero gave the focus gauge such a fill, and the error ended
        the service. Such a box has nothing to draw.
        """
        px = self._box(box)
        return px if px[2] >= px[0] and px[3] >= px[1] else None

    def rect(self, box: tuple[float, float, float, float], fill: tuple[int, ...]) -> None:
        if (px := self._pixels(box)) is not None:
            self._draw.rectangle(px, fill=fill)

    def pill(self, box: tuple[float, float, float, float], radius: float,
             fill: tuple[int, ...], outline: tuple[int, ...] | None = None,
             width: float = 0.0) -> None:
        if (px := self._pixels(box)) is not None:
            self._draw.rounded_rectangle(
                px, radius=_half_up(radius * self._k), fill=fill, outline=outline,
                width=self._width(width) if outline else 0)

    def disc(self, centre: tuple[float, float], radius: float,
             fill: tuple[int, ...] | None = None, outline: tuple[int, ...] | None = None,
             width: float = 0.0) -> None:
        cx, cy = centre
        if (px := self._pixels((cx - radius, cy - radius, cx + radius, cy + radius))) is not None:
            self._draw.ellipse(
                px, fill=fill, outline=outline, width=self._width(width) if outline else 0)

    def text_width(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
                   tracking: float = 0.0) -> float:
        """Advance width in base units, with ``tracking`` units between letters."""
        if not tracking:
            return font.getlength(text) / self._k
        return sum(font.getlength(ch) for ch in text) / self._k + tracking * (len(text) - 1)

    def text(self, xy: tuple[float, float], text: str,
             font: ImageFont.FreeTypeFont | ImageFont.ImageFont, fill: tuple[int, int, int],
             *, anchor: str = "lm", tracking: float = 0.0) -> None:
        """Text anchored at ``xy``: ``lm`` (left, middle) or ``mm`` (centred).

        The glyphs are drawn as a coverage mask and composited, not drawn straight
        onto the layer: ``ImageDraw`` blends text into a transparent RGBA image by
        interpolating all four channels, which darkens every anti-aliased edge.
        """
        mask = Image.new("L", self._image.size, 0)
        draw = ImageDraw.Draw(mask)
        x, y = (xy[0] - self._x0) * self._k, self._y(xy[1])
        if anchor[0] == "m":
            x -= self.text_width(text, font, tracking) * self._k / 2
        vertical = "l" + anchor[1]
        if tracking:
            for ch in text:
                draw.text((round(x), y), ch, font=font, fill=255, anchor=vertical)
                x += font.getlength(ch) + tracking * self._k
        else:
            draw.text((round(x), y), text, font=font, fill=255, anchor=vertical)
        self._composite(mask, fill)

    def glyph(self, centre: tuple[float, float], text: str,
              font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
              fill: tuple[int, int, int]) -> None:
        """One symbol with its ink centred on ``centre``.

        A font's "middle" is half way between its ascender and descender lines,
        and ``+`` and ``-`` sit visibly below that; in a round key the eye wants
        the strokes themselves in the middle.
        """
        left, top, right, bottom = font.getbbox(text, anchor="ls")
        mask = Image.new("L", self._image.size, 0)
        x = (centre[0] - self._x0) * self._k - (left + right) / 2
        y = (centre[1] - self._y0) * self._k - (top + bottom) / 2
        ImageDraw.Draw(mask).text((round(x), round(y)), text, font=font, fill=255, anchor="ls")
        self._composite(mask, fill)

    def _composite(self, mask: Image.Image, fill: tuple[int, int, int]) -> None:
        ink = Image.new("RGBA", self._image.size, fill + (0,))
        ink.putalpha(mask)
        self._image.alpha_composite(ink)

    def finish(self) -> Image.Image:
        """The tile at panel resolution (RGBA). Pillow premultiplies alpha for the
        reduction, so transparent pixels do not bleed their colour into the edges."""
        return self._image.resize(self._size, Image.LANCZOS)


def _dot_x(index: int) -> float:
    return METER_X0 + index * (METER_X1 - METER_X0) / (METER_DOTS - 1)


def _lit_dot(deviation_ev: float) -> int:
    """The dot nearest the deviation; the middle one when there is no reading."""
    if math.isnan(deviation_ev):
        return METER_DOTS // 2
    clamped = max(-METER_RANGE, min(METER_RANGE, deviation_ev))
    return _half_up((clamped + METER_RANGE) / (2 * METER_RANGE) * (METER_DOTS - 1))


def _whole(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return DASH
    return f"{value:.0f}"


def _meter_text(lux: float | None, clip_pct: float) -> str:
    return f"{_whole(lux)} LX   CLIP {_whole(clip_pct)}%"


def _unit(value: float) -> float:
    """A score clamped to 0..1; a NaN (no usable frame) counts as nothing."""
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


@lru_cache(maxsize=4)
def _strip_tile(scale: int) -> Image.Image:
    """The strip's background: solid black, the unlit meter scale and its ends. RGB.

    Composited onto black here, after the reduction, so the strip's own edge is a
    clean row of panel pixels and the strip is exactly black wherever it is empty.
    """
    canvas = _Canvas(_STRIP_BOX, scale)
    for index in range(METER_DOTS):
        canvas.disc((_dot_x(index), METER_Y), 1.5 if index % 2 == 0 else 1.0, fill=UNLIT)
    font = canvas.font(MEDIUM, METER_GLYPH_SIZE)
    canvas.glyph((METER_X0 - METER_END_OFFSET, METER_Y), "-", font, GREY)
    canvas.glyph((METER_X1 + METER_END_OFFSET, METER_Y), "+", font, GREY)
    marks = canvas.finish()
    tile = Image.new("RGB", marks.size, BLACK)
    tile.paste(marks, (0, 0), marks)
    return tile


@lru_cache(maxsize=32)
def _cell_tile(cell: Cell, selected: bool, scale: int) -> Image.Image:
    """One strip cell, drawn as cell 0; the caller pastes it at its own column."""
    if not cell.enabled:
        label_colour = auto_colour = value_colour = GREY
    elif selected:
        label_colour, auto_colour, value_colour = AMBER, WHITE, AMBER
    else:
        label_colour, auto_colour, value_colour = GREY, GREY, DIM_AMBER
    canvas = _Canvas(_CELL_BOX, scale)
    small = canvas.font(MEDIUM, LABEL_SIZE)
    canvas.text((CELL_X0, CELL_LABEL_Y), cell.label, small, label_colour,
                tracking=LABEL_TRACKING)
    if cell.auto:
        lead = canvas.text_width(cell.label + " ", small, LABEL_TRACKING) + LABEL_TRACKING
        canvas.text((CELL_X0 + lead, CELL_LABEL_Y), "AUTO", small, auto_colour,
                    tracking=LABEL_TRACKING)
    canvas.text((CELL_X0, CELL_VALUE_Y), cell.value, canvas.font(SEMIBOLD, VALUE_SIZE),
                value_colour)
    if selected and cell.enabled:
        x0, y0, x1, y1 = CELL_UNDERLINE
        canvas.rect((CELL_X0 + x0, y0, CELL_X0 + x1, y1), AMBER + (255,))
    return canvas.finish()


@lru_cache(maxsize=4)
def _controls_tile(pressed: Action | None, scale: int) -> Image.Image:
    """The shutter button and the +/- keys; the pressed one is filled amber."""
    canvas = _Canvas(_CONTROLS_BOX, scale)
    canvas.disc(SHUTTER_CENTRE, SHUTTER_RADIUS, fill=SHUTTER_FILL,
                outline=AMBER + (255,), width=SHUTTER_RING)
    canvas.disc(SHUTTER_CENTRE, SHUTTER_DISC_RADIUS,
                fill=AMBER + (255,) if pressed is Action.SHUTTER else SHUTTER_DISC)
    font = canvas.font(SEMIBOLD, KEY_GLYPH_SIZE)
    for centre, action, glyph in ((PLUS_CENTRE, Action.PLUS, "+"),
                                  (MINUS_CENTRE, Action.MINUS, "-")):
        if pressed is action:
            canvas.disc(centre, KEY_RADIUS, fill=AMBER + (255,))
            canvas.glyph(centre, glyph, font, BLACK)
        else:
            canvas.disc(centre, KEY_RADIUS, fill=KEY_FILL,
                        outline=AMBER + (255,), width=KEY_OUTLINE)
            canvas.glyph(centre, glyph, font, WHITE)
    return canvas.finish()


@lru_cache(maxsize=32)
def _tags_tile(double: tuple[bool, int] | None, battery: str | None, scale: int,
               developing: int = 0, tuning: bool = False) -> Image.Image:
    """The 2x double-exposure toggle, the battery tag, the developing count and the
    TUNING warning, top left.

    One tile for all of them because the count's place depends on the battery text's
    width, and the warning's on both. ``developing`` is already limited to
    ``DEVELOPING_MAX`` by the caller, so the cache (keyed by every argument) holds
    few of these.
    """
    canvas = _Canvas(_TAGS_BOX, scale)
    font = canvas.font(SEMIBOLD, TAG_SIZE)

    def tag(box: tuple[float, float, float, float], label: str, lit: bool) -> None:
        if lit:
            canvas.pill(box, TAG_RADIUS, fill=AMBER + (255,))
        else:
            canvas.pill(box, TAG_RADIUS, fill=KEY_FILL, outline=AMBER + (255,),
                        width=KEY_OUTLINE)
        canvas.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font,
                    BLACK if lit else WHITE, anchor="mm")

    if double is not None:
        x0, y0, x1, y1 = DOUBLE_BOX
        enabled, taken = double
        if enabled:
            label = f"2x {taken}/2"
            wanted = x0 + canvas.text_width(label, font) + 2 * TAG_PADDING
            tag((x0, y0, min(max(x1, wanted), BATTERY_X0 - TAG_GAP), y1), label, True)
        else:
            tag(DOUBLE_BOX, "2x", False)
    left = float(BATTERY_X0)
    if battery is not None:
        right = BATTERY_X0 + canvas.text_width(battery, font) + 2 * TAG_PADDING
        right = min(right, _TAGS_BOX[2] - 1)
        tag((BATTERY_X0, DOUBLE_BOX[1], right, DOUBLE_BOX[3]), battery, False)
        # From a whole unit, so the tag's soft edge and the patch never share a pixel.
        left = math.ceil(right) + DEVELOPING_GAP
    if developing > 0:
        label = str(developing)
        box = (left, DOUBLE_BOX[1],
               left + canvas.text_width(label, font) + 2 * DEVELOPING_PADDING, DOUBLE_BOX[3])
        canvas.pill(box, TAG_RADIUS, fill=DEVELOPING_FILL)
        canvas.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font, AMBER,
                    anchor="mm")
        left = math.ceil(box[2]) + DEVELOPING_GAP
    if tuning:
        box = (left, DOUBLE_BOX[1],
               left + canvas.text_width(TUNING_LABEL, font) + 2 * TAG_PADDING, DOUBLE_BOX[3])
        canvas.pill(box, TAG_RADIUS, fill=DEVELOPING_FILL)
        canvas.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), TUNING_LABEL, font,
                    AMBER, anchor="mm")
    return canvas.finish()


@lru_cache(maxsize=16)
def _disk_tile(warning: tuple[tuple[str, int], ...], scale: int) -> Image.Image:
    """One amber tag per filesystem low on space, left to right from ``DISK_X0``.

    Styled as the TUNING tag: amber text, the battery tag's padding, the count's
    dark patch, no outline (a warning, not a key). Keyed by the warning itself;
    a percentage changes at most once a minute, so the cache holds few of these.
    """
    canvas = _Canvas(_DISK_BOX, scale)
    font = canvas.font(SEMIBOLD, TAG_SIZE)
    left = float(DISK_X0)
    for label, percent in warning:
        text = f"{label} {percent}%"
        box = (left, DOUBLE_BOX[1],
               min(left + canvas.text_width(text, font) + 2 * TAG_PADDING, _DISK_BOX[2] - 1),
               DOUBLE_BOX[3])
        canvas.pill(box, TAG_RADIUS, fill=DEVELOPING_FILL)
        canvas.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font, AMBER,
                    anchor="mm")
        left = math.ceil(box[2]) + DEVELOPING_GAP
    return canvas.finish()


@lru_cache(maxsize=16)
def _network_tile(network: tuple[tuple[str, str], ...], scale: int) -> Image.Image:
    """One white tag per connected interface, ``ETH 192.168.1.20``, left to right.

    The disk tags' shape (the battery tag's padding, the count's dark patch, no
    outline) in white, filling the box's height. Keyed by the addresses, which
    change only when a cable or a network does.
    """
    canvas = _Canvas(_NETWORK_BOX, scale)
    font = canvas.font(SEMIBOLD, TAG_SIZE)
    top, bottom = _NETWORK_BOX[1], _NETWORK_BOX[3]
    left = float(_NETWORK_BOX[0])
    for label, address in network:
        text = f"{label} {address}"
        box = (left, top,
               min(left + canvas.text_width(text, font) + 2 * TAG_PADDING,
                   _NETWORK_BOX[2] - 1),
               bottom)
        canvas.pill(box, TAG_RADIUS, fill=DEVELOPING_FILL)
        canvas.text(((box[0] + box[2]) / 2, (top + bottom) / 2), text, font, WHITE,
                    anchor="mm")
        left = math.ceil(box[2]) + DEVELOPING_GAP
    return canvas.finish()


@lru_cache(maxsize=32)
def _lit_dot_tile(index: int, scale: int) -> tuple[Image.Image, tuple[int, int]]:
    """The amber dot for one meter position, and where its tile goes (base units).

    One tile per position, not one moved about: the dots are 8.4 units apart, so
    each sits at a different fraction of a pixel.
    """
    x = _dot_x(index)
    box = (math.floor(x) - 4, METER_Y - 4, math.floor(x) + 5, METER_Y + 4)
    canvas = _Canvas(box, scale)
    canvas.disc((x, METER_Y), 2.25, fill=AMBER + (255,))
    return canvas.finish(), box[:2]


@lru_cache(maxsize=32)
def _meter_text_tile(text: str, scale: int) -> Image.Image:
    canvas = _Canvas(_METER_TEXT_BOX, scale)
    canvas.text(METER_TEXT_CENTRE, text, canvas.font(MEDIUM, LABEL_SIZE), GREY,
                anchor="mm", tracking=LABEL_TRACKING)
    return canvas.finish()


def _focus_tile(focus: float, peak: float | None, scale: int) -> Image.Image:
    """The manual-focus gauge: a track filled from the bottom to the sharpness
    score, with an amber tick at the best level of the last few seconds.

    Not cached: the score changes with every frame. The tick is kept inside the
    track's height so the tile never has to reach past it.
    """
    canvas = _Canvas(_FOCUS_BOX, scale)
    x0, top, x1, bottom = FOCUS_TRACK
    span = bottom - top
    canvas.rect(FOCUS_TRACK, KEY_FILL)
    level = _unit(focus)
    if level > 0.0:
        canvas.rect((x0, bottom - level * span, x1, bottom), WHITE + (255,))
    if peak is not None and _unit(peak) > 0.0:
        centre = bottom - _unit(peak) * span
        tick_top = max(top, min(bottom - FOCUS_TICK_HEIGHT, centre - FOCUS_TICK_HEIGHT / 2))
        canvas.rect((FOCUS_TICK_X[0], tick_top, FOCUS_TICK_X[1], tick_top + FOCUS_TICK_HEIGHT),
                    AMBER + (255,))
    return canvas.finish()


def render_live(
    frame_rgb: np.ndarray, reading: MeterReading, state: LiveState, *, scale: int = 1,
) -> Image.Image:
    """The live screen: RGB, ``(320 * scale, 240 * scale)``.

    Reads from ``reading`` only ``deviation_ev``, ``lux``, ``clip_pct``, ``focus``,
    ``focus_peak``, ``battery_percent`` and ``external_power``; the shutter, ISO
    and white-balance read-outs arrive as text in ``state.cells``.
    """
    img = _letterbox(frame_rgb, (WIDTH * scale, HEIGHT * scale))

    def paste(tile: Image.Image, x: int, y: int) -> None:
        img.paste(tile, (x * scale, y * scale), tile)

    img.paste(_strip_tile(scale), (0, BAR_TOP * scale))
    for index, cell in enumerate(state.cells):
        paste(_cell_tile(cell, state.selected == index, scale),
              _CELL_BOX[0] + CELL_PITCH * index, _CELL_BOX[1])
    paste(_meter_text_tile(_meter_text(reading.lux, reading.clip_pct), scale),
          *_METER_TEXT_BOX[:2])
    dot, origin = _lit_dot_tile(_lit_dot(reading.deviation_ev), scale)
    paste(dot, *origin)

    pressed = state.pressed if state.pressed in PRESS_KEYS else None
    paste(_controls_tile(pressed, scale), *_CONTROLS_BOX[:2])
    battery = None
    if reading.battery_percent is not None:
        battery = f"{'AC ' if reading.external_power else ''}{reading.battery_percent}%"
    developing = max(0, min(DEVELOPING_MAX, state.developing))
    tuning = bool(state.tuning_mismatch)
    if state.double is not None or battery is not None or developing or tuning:
        paste(_tags_tile(state.double, battery, scale, developing, tuning), *_TAGS_BOX[:2])
    if state.disk_warning:
        disk = tuple((str(label), int(percent)) for label, percent in state.disk_warning)
        paste(_disk_tile(disk, scale), *_DISK_BOX[:2])
    if state.network:
        network = tuple((str(label), str(address)) for label, address in state.network)
        paste(_network_tile(network, scale), *_NETWORK_BOX[:2])
    if reading.focus is not None:
        paste(_focus_tile(reading.focus, reading.focus_peak, scale), *_FOCUS_BOX[:2])
    return img


def hit(x: int, y: int, scale: int = 1) -> Action:
    """The control under a panel pixel. Regions are in base units and disjoint."""
    x, y = x // scale, y // scale
    if math.hypot(x - SHUTTER_CENTRE[0], y - SHUTTER_CENTRE[1]) <= SHUTTER_HIT_RADIUS:
        return Action.SHUTTER
    if math.hypot(x - PLUS_CENTRE[0], y - PLUS_CENTRE[1]) <= KEY_HIT_RADIUS:
        return Action.PLUS
    if math.hypot(x - MINUS_CENTRE[0], y - MINUS_CENTRE[1]) <= KEY_HIT_RADIUS:
        return Action.MINUS
    x0, y0, x1, y1 = DOUBLE_BOX
    # Grown by the margin; the left and top sides end at the screen's edge.
    if (max(0, x0 - DOUBLE_HIT_MARGIN) <= x < x1 + DOUBLE_HIT_MARGIN
            and max(0, y0 - DOUBLE_HIT_MARGIN) <= y < y1 + DOUBLE_HIT_MARGIN):
        return Action.DOUBLE_TOGGLE
    if y >= CELL_HIT_TOP and x >= CELL_HIT_X0:
        index = (x - CELL_HIT_X0) // CELL_PITCH
        if index < len(SELECT_ACTIONS):
            return SELECT_ACTIONS[index]
    return Action.NONE
