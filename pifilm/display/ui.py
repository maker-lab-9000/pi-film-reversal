"""Pillow rendering for the 320x240 viewfinder and its hit regions. Pure.

Layout: the preview fills the screen letterboxed; a translucent bar along the
bottom carries the meter; a round shutter button sits at the right edge; EV
buttons occupy the bar's ends; optional shutter-priority +/- buttons
sit above and below the shutter button. ``hit`` mirrors the drawn regions plus a 6 px
margin so the two cannot drift apart: both read the same constants.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meter import MeterReading, format_ev

WIDTH, HEIGHT = 320, 240
BAR_TOP = 204
BAR_ALPHA = 153  # 60 %
SHUTTER_CENTRE = (288, 102)
SHUTTER_RADIUS = 28
EV_BUTTON_W = 40
HIT_MARGIN = 6
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
# One placeholder for every missing readout, on the live bar and the review
# caption alike: a plain hyphen, because the default font has no en dash glyph
# and renders one as a tofu box.
DASH = "-"
AMBER = (255, 176, 0)
NEEDLE_X0, NEEDLE_X1, NEEDLE_Y = 200, 272, 222
NEEDLE_RANGE = 3.0
# Focus bar: a vertical gauge down the left edge, clear of the shutter button on
# the right and stopping well above the meter bar so it never sits over the EV
# minus button or its hit margin (hit() starts the bar zone at BAR_TOP - HIT_MARGIN).
FOCUS_BAR_X0, FOCUS_BAR_X1 = 6, 14
FOCUS_BAR_Y0, FOCUS_BAR_Y1 = 40, 190
FOCUS_GREEN = (0, 220, 90)
FOCUS_LABEL_Y = 196
# Double-exposure toggle: a pill in the top-left corner. It sits above the focus
# bar's backing (y >= FOCUS_BAR_Y0 - 4) even with its hit margin, and away from the
# battery badge (top-right) and the shutter (right edge).
DOUBLE_BOX = (4, 4, 74, 24)
# Shutter-priority buttons, stacked above and below the shutter button. With
# HIT_MARGIN the + region ends at y 66 and the - region starts at y 138, both clear
# of the shutter button's hit circle (y 68-136), and the - region ends at y 184,
# above the bar's hit zone (BAR_TOP - HIT_MARGIN = 198).
SHUTTER_PLUS_BOX = (264, 26, 312, 60)
SHUTTER_MINUS_BOX = (264, 144, 312, 178)


class Action(Enum):
    NONE = "none"
    SHUTTER = "shutter"
    EV_MINUS = "ev_minus"
    EV_PLUS = "ev_plus"
    DOUBLE_TOGGLE = "double_toggle"
    SHUTTER_FASTER = "shutter_faster"
    SHUTTER_SLOWER = "shutter_slower"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow >= 10.1 (the floor in pyproject.toml) ships a scalable default.
    return ImageFont.load_default(size=size)


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


def render_live(
    frame_rgb: np.ndarray, reading: MeterReading, double: tuple[bool, int] | None = None,
    shutter_buttons: bool = False,
) -> Image.Image:
    base = _letterbox(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, BAR_TOP, WIDTH, HEIGHT), fill=(0, 0, 0, BAR_ALPHA))
    # EV buttons
    draw.rectangle((0, BAR_TOP, EV_BUTTON_W, HEIGHT), outline=(255, 255, 255, 200))
    draw.rectangle((WIDTH - EV_BUTTON_W, BAR_TOP, WIDTH, HEIGHT), outline=(255, 255, 255, 200))
    big = _font(18)
    draw.text((EV_BUTTON_W // 2, BAR_TOP + 18), "-",
              fill=(255, 255, 255, 255), font=big, anchor="mm")
    draw.text((WIDTH - EV_BUTTON_W // 2, BAR_TOP + 18), "+",
              fill=(255, 255, 255, 255), font=big, anchor="mm")
    # readout
    font = _font(14)
    small = _font(11)
    shutter_text = text_or_dash(reading.shutter)
    if reading.shutter_fixed:
        shutter_text = f"S {shutter_text}"
    x = EV_BUTTON_W + 6
    # "S 1/2000  ISO MAX  EV +1.7" at 14 pt ends near x 218 and would run under the needle
    # marker (its left edge can reach NEEDLE_X0 - 5). The fixed-shutter and ISO MAX
    # readouts are the longer ones, so only they drop to 12 pt (worst case ends at x 192,
    # under NEEDLE_X0 - 6); the ordinary readout keeps 14 pt exactly as before.
    if reading.shutter_fixed or reading.iso_max:
        font = _font(12)
    if reading.iso_max:
        # Drawn in three runs so only the ISO turns amber.
        for text, colour in ((f"{shutter_text}  ", (255, 255, 255, 255)),
                             ("ISO MAX", AMBER + (255,)),
                             (f"  EV {format_ev(reading.ev_comp)}", (255, 255, 255, 255))):
            draw.text((x, BAR_TOP + 3), text, fill=colour, font=font)
            x += draw.textlength(text, font=font)
    else:
        line1 = f"{shutter_text}  {iso_label(reading.iso)}  EV {format_ev(reading.ev_comp)}"
        draw.text((x, BAR_TOP + 3), line1, fill=(255, 255, 255, 255), font=font)
    lux = f"{reading.lux:.0f} lx" if reading.lux is not None else f"{DASH} lx"
    line2 = f"{lux}   clip {reading.clip_pct:.0f}%"
    draw.text((EV_BUTTON_W + 6, BAR_TOP + 21), line2, fill=(220, 220, 220, 255), font=small)
    # needle: scale -3..+3 stops
    draw.line((NEEDLE_X0, NEEDLE_Y, NEEDLE_X1, NEEDLE_Y), fill=(255, 255, 255, 200), width=1)
    for k in range(-3, 4):
        x = _needle_x(float(k))
        draw.line((x, NEEDLE_Y - (4 if k == 0 else 2), x, NEEDLE_Y + (4 if k == 0 else 2)),
                  fill=(255, 255, 255, 200))
    nx = _needle_x(reading.deviation_ev)
    draw.polygon([(nx, NEEDLE_Y - 8), (nx - 5, NEEDLE_Y - 15), (nx + 5, NEEDLE_Y - 15)],
                 fill=AMBER + (255,))
    if reading.focus is not None:
        _draw_focus_bar(draw, reading.focus, reading.focus_peak, small)
    # shutter button
    cx, cy = SHUTTER_CENTRE
    draw.ellipse(
        (cx - SHUTTER_RADIUS, cy - SHUTTER_RADIUS, cx + SHUTTER_RADIUS, cy + SHUTTER_RADIUS),
        fill=(255, 255, 255, 60), outline=(255, 255, 255, 255), width=2,
    )
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(255, 255, 255, 180))
    if shutter_buttons:
        for box, glyph in ((SHUTTER_PLUS_BOX, "+"), (SHUTTER_MINUS_BOX, "-")):
            draw.rectangle(box, fill=(0, 0, 0, BAR_ALPHA), outline=(255, 255, 255, 200))
            draw.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), glyph,
                      fill=(255, 255, 255, 255), font=big, anchor="mm")
    if double is not None:
        _draw_double_badge(draw, double[0], double[1], small)
    # battery badge
    if reading.battery_percent is not None:
        label = f"{'AC ' if reading.external_power else ''}{reading.battery_percent}%"
        w = draw.textlength(label, font=small)
        draw.rounded_rectangle(
            (WIDTH - w - 14, 4, WIDTH - 4, 20), radius=4, fill=(0, 0, 0, BAR_ALPHA),
        )
        draw.text((WIDTH - w - 9, 6), label, fill=(255, 255, 255, 255), font=small)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _draw_focus_bar(
    draw: ImageDraw.ImageDraw, focus: float, peak: float | None, font: Any,
) -> None:
    """The manual-focus gauge: fill height is the centre's absolute sharpness.

    The IMX477 has no autofocus, so this is the only feedback the user gets while
    turning the ring. It grows as the image sharpens and drops the moment focus
    is racked past, and the amber tick sits at the best level of the last few
    seconds — rack through focus, then turn back until the green reaches it.

    Drawn on the left because the right edge is the shutter button, over its own
    translucent backing so it stays readable against a bright scene, and ending
    at ``FOCUS_BAR_Y1`` so that neither the bar nor the label reaches the EV
    minus button's hit region (``BAR_TOP - HIT_MARGIN``).
    """
    focus = max(0.0, min(1.0, float(focus)))
    draw.rectangle(
        (FOCUS_BAR_X0 - 4, FOCUS_BAR_Y0 - 4, FOCUS_BAR_X1 + 4, FOCUS_LABEL_Y + 1),
        fill=(0, 0, 0, BAR_ALPHA),
    )
    draw.rectangle((FOCUS_BAR_X0, FOCUS_BAR_Y0, FOCUS_BAR_X1, FOCUS_BAR_Y1),
                   outline=(255, 255, 255, 255), width=1)
    inner_top, inner_bottom = FOCUS_BAR_Y0 + 1, FOCUS_BAR_Y1 - 1
    span = inner_bottom - inner_top + 1
    filled = int(round(focus * span))
    if filled > 0:
        draw.rectangle(
            (FOCUS_BAR_X0 + 1, inner_bottom - filled + 1, FOCUS_BAR_X1 - 1, inner_bottom),
            fill=FOCUS_GREEN + (255,),
        )
    # The peak mark, drawn last so the green does not hide it; two pixels tall,
    # its top edge at the peak's fill height.
    if peak is not None and peak > 0.0:
        peak = max(0.0, min(1.0, float(peak)))
        mark_top = min(inner_bottom - 1, inner_bottom - int(round(peak * span)) + 1)
        draw.rectangle((FOCUS_BAR_X0 + 1, mark_top, FOCUS_BAR_X1 - 1, mark_top + 1),
                       fill=AMBER + (255,))
    # Baseline-anchored: the glyph grows upwards from FOCUS_LABEL_Y, so it cannot
    # reach the EV minus button's hit region below it.
    draw.text(((FOCUS_BAR_X0 + FOCUS_BAR_X1) // 2, FOCUS_LABEL_Y), "F",
              fill=(255, 255, 255, 255), font=font, anchor="ms")


def _draw_double_badge(draw: ImageDraw.ImageDraw, enabled: bool, taken: int, font: Any) -> None:
    """The double-exposure toggle. Off: an outlined ``2x``. On: amber, with progress.

    ASCII only: the default font has no multiplication sign.
    """
    x0, y0, x1, y1 = DOUBLE_BOX
    if enabled:
        draw.rounded_rectangle(DOUBLE_BOX, radius=8, fill=AMBER + (255,))
        label, colour = f"2x {taken}/2", (0, 0, 0, 255)
    else:
        draw.rounded_rectangle(DOUBLE_BOX, radius=8, fill=(0, 0, 0, BAR_ALPHA),
                               outline=(255, 255, 255, 255), width=1)
        label, colour = "2x", (255, 255, 255, 255)
    draw.text(((x0 + x1) // 2, (y0 + y1) // 2), label, fill=colour, font=font, anchor="mm")


def render_processing(label: str = "Processing photo...") -> Image.Image:
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
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for index, colour in enumerate(PROCESSING_BARS):
        left, right = index * WIDTH // 7, (index + 1) * WIDTH // 7
        dimmed = tuple(round(c * PROCESSING_DIM) for c in colour)
        draw.rectangle((left, 0, right - 1, HEIGHT * 3 // 4 - 1), fill=dimmed)
    draw.text((WIDTH // 2, HEIGHT * 7 // 8), label,
              fill=(255, 255, 255), font=_font(14), anchor="mm")
    return img


def _needle_x(deviation: float) -> int:
    t = (max(-NEEDLE_RANGE, min(NEEDLE_RANGE, deviation)) + NEEDLE_RANGE) / (2 * NEEDLE_RANGE)
    return int(round(NEEDLE_X0 + t * (NEEDLE_X1 - NEEDLE_X0)))


def render_review(graded_rgb: np.ndarray, caption: str) -> Image.Image:
    base = _letterbox(graded_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, HEIGHT - 22, WIDTH, HEIGHT), fill=(0, 0, 0, BAR_ALPHA))
    draw.text((6, HEIGHT - 19), caption, fill=(255, 255, 255, 255), font=_font(12))
    hint = "tap to continue"
    w = draw.textlength(hint, font=_font(11))
    draw.text((WIDTH - w - 6, HEIGHT - 18), hint, fill=(200, 200, 200, 255), font=_font(11))
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_message(title: str, detail: str) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((WIDTH // 2, HEIGHT // 2 - 16), title,
              fill=(255, 255, 255), font=_font(18), anchor="mm")
    draw.text((WIDTH // 2, HEIGHT // 2 + 14), detail[:60],
              fill=(200, 200, 200), font=_font(12), anchor="mm")
    return img


def hit(x: int, y: int) -> Action:
    cx, cy = SHUTTER_CENTRE
    if math.hypot(x - cx, y - cy) <= SHUTTER_RADIUS + HIT_MARGIN:
        return Action.SHUTTER
    x0, y0, x1, y1 = DOUBLE_BOX
    if x0 - HIT_MARGIN <= x <= x1 + HIT_MARGIN and y0 - HIT_MARGIN <= y <= y1 + HIT_MARGIN:
        return Action.DOUBLE_TOGGLE
    for box, action in ((SHUTTER_PLUS_BOX, Action.SHUTTER_FASTER),
                        (SHUTTER_MINUS_BOX, Action.SHUTTER_SLOWER)):
        x0, y0, x1, y1 = box
        if x0 - HIT_MARGIN <= x <= x1 + HIT_MARGIN and y0 - HIT_MARGIN <= y <= y1 + HIT_MARGIN:
            return action
    if y >= BAR_TOP - HIT_MARGIN:
        if x <= EV_BUTTON_W + HIT_MARGIN:
            return Action.EV_MINUS
        if x >= WIDTH - EV_BUTTON_W - HIT_MARGIN:
            return Action.EV_PLUS
    return Action.NONE
