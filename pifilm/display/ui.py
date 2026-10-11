"""Pillow rendering for the viewfinder's processing, review and message screens,
in 320x240 base units. Pure.

The live screen and its hit regions are in ``liveview.py``; these three screens
take no taps of their own (any tap ends a review or a notice), so there are no hit
regions here. ``_letterbox`` is shared with the live screen.

Every render takes ``scale``: 1 for the 2.8" SPI panel, 2 for the 640x480 DSI panel.
The constants below never change with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 320, 240
BAR_ALPHA = 153  # 60 %, the review caption's backing
# The seven TV colour bars, left to right in RGB. ``pifilm.capture.app``'s
# ``_capture_loading_screen`` draws the same seven for the OpenCV window, listed
# there in OpenCV's BGR order: keep the two visually identical.
PROCESSING_BARS = (
    (191, 191, 191), (191, 191, 0), (0, 191, 191), (0, 191, 0),
    (191, 0, 191), (191, 0, 0), (0, 0, 191),
)
# The LCD sits close to the eye and is often used in a dim room, where the bars
# at full strength glared; it draws them at this fraction. Only the LCD: the
# Stick and the OpenCV window keep PROCESSING_BARS as they are.
PROCESSING_DIM = 0.55
# The placeholder for a missing readout in the review caption: a plain hyphen,
# because the default font has no en dash glyph and renders one as a tofu box.
DASH = "-"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow >= 10.1 (the floor in pyproject.toml) ships a scalable default.
    return ImageFont.load_default(size=size)


class _ScaledDraw:
    """An ``ImageDraw`` that takes coordinates in 320x240 base units.

    The layout constants in this module are in base units so the 2.8" panel
    (scale 1) and the 3.5" DSI panel (scale 2) share one layout and one set of hit
    regions. Shapes with two corners (rectangles, ellipses) treat each base pixel
    as a ``scale`` x ``scale`` block, so the far corner maps to that block's last
    pixel and neighbouring shapes stay flush; at scale 1 that is the identity.
    Points (lines, polygons, text anchors) map to ``value * scale``. Widths,
    radii and font sizes are multiplied; ``textlength`` is returned in base units
    so callers keep doing their arithmetic there.
    """

    def __init__(self, draw: ImageDraw.ImageDraw, scale: int) -> None:
        self._draw, self._s = draw, scale

    def font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return _font(size * self._s)

    def _box(self, box: Any) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        s = self._s
        return (x0 * s, y0 * s, (x1 + 1) * s - 1, (y1 + 1) * s - 1)

    def _points(self, points: Any) -> list[tuple[float, float]]:
        flat = list(points)
        if flat and not isinstance(flat[0], (tuple, list)):
            flat = list(zip(flat[0::2], flat[1::2], strict=True))
        return [(x * self._s, y * self._s) for x, y in flat]

    def _kw(self, kw: dict[str, Any]) -> dict[str, Any]:
        # Pillow's outline width defaults to 1 for these shapes; scale that default
        # too, so an outline stays one base pixel thick.
        kw["width"] = kw.get("width", 1) * self._s
        return kw

    def rectangle(self, box: Any, **kw: Any) -> None:
        self._draw.rectangle(self._box(box), **self._kw(kw))

    def rounded_rectangle(self, box: Any, radius: int = 0, **kw: Any) -> None:
        self._draw.rounded_rectangle(self._box(box), radius=radius * self._s, **self._kw(kw))

    def ellipse(self, box: Any, **kw: Any) -> None:
        self._draw.ellipse(self._box(box), **self._kw(kw))

    def line(self, points: Any, **kw: Any) -> None:
        # An unspecified width is one base pixel. Always pass it: Pillow's own
        # default changed from 0 to 1, and from that release a width of 0 draws
        # nothing, while 1 is the same hairline on every version.
        self._draw.line(self._points(points), width=kw.pop("width", 1) * self._s, **kw)

    def polygon(self, points: Any, **kw: Any) -> None:
        self._draw.polygon(self._points(points), **kw)

    def text(self, xy: Any, text: str, **kw: Any) -> None:
        self._draw.text((xy[0] * self._s, xy[1] * self._s), text, **kw)

    def textlength(self, text: str, font: Any) -> float:
        return self._draw.textlength(text, font=font) / self._s


def _letterbox(rgb: np.ndarray, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    src = Image.fromarray(np.ascontiguousarray(rgb))
    scale = min(size[0] / src.width, size[1] / src.height)
    fitted = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))),
                        Image.BILINEAR)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def text_or_dash(value: str | None) -> str:
    return value if value else DASH


def iso_label(iso: int | None) -> str:
    return f"ISO {iso}" if iso is not None else f"ISO {DASH}"


def render_processing(label: str = "Processing photo...", *, scale: int = 1) -> Image.Image:
    """TV colour bars while a shot is graded: the Stick's screen, on the LCD.

    A grade takes about three seconds on the Pi 4, and during it the camera is
    the capture worker's, not the viewfinder's. Showing the live view with a
    small marker read as "nothing happened"; the bars are what the Stick and the
    Pi's own OpenCV window already show, so every screen says "working" the same
    way. Bar order and geometry mirror ``pifilm.capture.app``'s
    ``_capture_loading_screen``, dimmed to ``PROCESSING_DIM`` for the panel; the
    label stays full white and is ASCII because the default font has no glyphs
    beyond it.
    """
    img = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), (0, 0, 0))
    draw = _ScaledDraw(ImageDraw.Draw(img), scale)
    for index, colour in enumerate(PROCESSING_BARS):
        left, right = index * WIDTH // 7, (index + 1) * WIDTH // 7
        dimmed = tuple(round(c * PROCESSING_DIM) for c in colour)
        draw.rectangle((left, 0, right - 1, HEIGHT * 3 // 4 - 1), fill=dimmed)
    draw.text((WIDTH // 2, HEIGHT * 7 // 8), label,
              fill=(255, 255, 255), font=draw.font(14), anchor="mm")
    return img


def render_review(graded_rgb: np.ndarray, caption: str, *, scale: int = 1) -> Image.Image:
    size = (WIDTH * scale, HEIGHT * scale)
    base = _letterbox(graded_rgb, size).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = _ScaledDraw(ImageDraw.Draw(overlay), scale)
    draw.rectangle((0, HEIGHT - 22, WIDTH, HEIGHT), fill=(0, 0, 0, BAR_ALPHA))
    draw.text((6, HEIGHT - 19), caption, fill=(255, 255, 255, 255), font=draw.font(12))
    hint = "tap to continue"
    w = draw.textlength(hint, font=draw.font(11))
    draw.text((WIDTH - w - 6, HEIGHT - 18), hint, fill=(200, 200, 200, 255), font=draw.font(11))
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_message(title: str, detail: str, *, scale: int = 1) -> Image.Image:
    img = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), (20, 20, 20))
    draw = _ScaledDraw(ImageDraw.Draw(img), scale)
    draw.text((WIDTH // 2, HEIGHT // 2 - 16), title,
              fill=(255, 255, 255), font=draw.font(18), anchor="mm")
    draw.text((WIDTH // 2, HEIGHT // 2 + 14), detail[:60],
              fill=(200, 200, 200), font=draw.font(12), anchor="mm")
    return img
