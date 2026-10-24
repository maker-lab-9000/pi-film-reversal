"""Render the LCD viewfinder's live view and every settings-menu page to PNGs.

    .venv/bin/python scripts/render_menu_preview.py OUT_DIR [--scale 1|2] [--frame JPEG]

For looking at the screens without a Pi: no camera, no panel, no ``.env``. The
pictures come from the real drawing code (``liveview.render_live`` and
``menu.render_menu``), and the menu's rows and titles from the viewfinder's own
row builders, driven with a fake camera and controller as a Picamera2 camera with
every setting row would drive them, so each PNG is what the panel shows.

Written into OUT_DIR:

- ``live.png`` and ``live-menu-pressed.png``: the live view over a sample frame
  (a synthetic seaside scene, or ``--frame``), the three cells with typical
  values, the meter, the IMG and STAB tags on, and the MENU key unlit and lit;
- ``menu-1-first.png`` .. ``menu-6-stabilise.png``: every page of the menu tree;
- ``contact-sheet.png``: all of them side by side, labelled.

The scale is 2 by default (640x480, the 3.5" DSI panel); ``--scale 1`` is the
2.8" SPI panel.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pifilm.capture.settings import Settings, SettingsControl
from pifilm.display.liveview import Action, Cell, LiveState, render_live
from pifilm.display.menu import render_menu
from pifilm.display.meter import MeterReading
from pifilm.display.viewfinder import MENU_MAIN, ViewfinderLoop

# Every optional row a Picamera2 camera shows, and a look with halation: the
# fullest tree, so every row of every page is drawn.
MENU_ROWS = ("save_dng", "halation", "ae_constraint", "ae_metering", "sharpness",
             "contrast", "saturation", "short_exposure", "shutter_delay", "multi_frame")
UNIT = Settings(save_dng=True, ae_constraint="normal", ae_metering="centre")
# One image control and one help against shake moved off their defaults, so the
# IMAGE and STABILISE pages agree with the IMG and STAB tags on the live view.
CHANGED = {"contrast": 1.25, "shutter_delay": 0.2}
LOOK = "imx477-2026-09-v4-dark"
NETWORK = (("ETH", "192.168.1.42"), ("WLAN", "10.42.0.1"))
UPLOAD_STATUS = "READY"
# (file stem, label, page): the first page, the categories, the settings pages.
PAGES = (
    ("menu-1-first", "MENU", MENU_MAIN),
    ("menu-2-camera", "CAMERA", "camera"),
    ("menu-3-pipeline", "PIPELINE", "pipeline"),
    ("menu-4-network", "NETWORK", "network"),
    ("menu-5-image", "CAMERA > IMAGE", "image"),
    ("menu-6-stabilise", "PIPELINE > STABILISE", "stabilise"),
)
# A fixed shutter, automatic ISO and white balance: one cell selected.
CELLS = (Cell("SHUTTER", "1/125", auto=False), Cell("ISO", "400"), Cell("WB", "5200K"))
READING = MeterReading(shutter="1/125", iso=400, lux=2400.0, deviation_ev=0.5,
                       clip_pct=1.0, battery_percent=78, external_power=False,
                       focus=0.62, focus_peak=0.8)


class _Upload:
    def start(self) -> bool:
        return True

    def status(self) -> str:
        return UPLOAD_STATUS


class _Clock:
    @staticmethod
    def monotonic() -> float:
        return 0.0

    @staticmethod
    def sleep(seconds: float) -> None:
        pass


class _Controller:
    """Only what the loop's constructor and the menu's rows read."""

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(finished_count=0, network=NETWORK)


def sample_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """A synthetic seaside scene, 4:3 like the preview: a sky graded from deep to
    pale blue, a low sun, a sea, striped beach huts and sand. RGB uint8."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    v = y / (height - 1)
    img = np.zeros((height, width, 3), np.float32)
    sky = v < 0.45
    t = (v / 0.45)[..., None]
    img[sky] = ((1 - t) * np.array([40, 110, 200]) + t * np.array([190, 220, 240]))[sky]
    sea = (v >= 0.45) & (v < 0.6)
    img[sea] = np.array([20, 120, 150])
    img[sea] += (8 * np.sin(x / 9 + y / 3))[sea][:, None]
    sand = v >= 0.6
    s = ((v - 0.6) / 0.4)[..., None]
    img[sand] = ((1 - s) * np.array([235, 205, 150]) + s * np.array([200, 160, 110]))[sand]
    sun = (x - 0.72 * width) ** 2 + (y - 0.18 * height) ** 2 < (0.06 * height) ** 2
    img[sun] = np.array([255, 245, 210])
    colours = np.array([[220, 40, 50], [250, 250, 245], [30, 140, 210],
                        [250, 200, 30], [40, 170, 90]], np.float32)
    hut_w = width // 10
    for i in range(6):
        x0 = int(width * 0.08) + i * (hut_w + width // 60)
        y0, y1 = int(height * 0.52), int(height * 0.78)
        stripes = colours[(np.arange(x0, x0 + hut_w) // max(1, hut_w // 6) + i) % len(colours)]
        img[y0:y1, x0:x0 + hut_w] = stripes[None, :, :]
        roof = np.array([90, 60, 50])
        img[y0 - height // 40:y0, x0 - 2:x0 + hut_w + 2] = roof
    return np.clip(img, 0, 255).astype(np.uint8)


def _menu_loop(state_dir: Path) -> ViewfinderLoop:
    settings = SettingsControl(state_dir / "settings.json", UNIT, {}, log=lambda *a: None)
    for key, value in CHANGED.items():
        settings.set(key, value)
    camera = SimpleNamespace(set_shutter=lambda *a: None, set_iso=lambda *a: None,
                             set_colour_temperature=lambda *a: None)
    display = SimpleNamespace(width=320, show=lambda image: None, close=lambda: None)
    return ViewfinderLoop(camera, _Controller(), display, SimpleNamespace(),
                          clock=_Clock(), log=lambda *a: None, settings=settings,
                          upload=_Upload(), look=LOOK, menu_rows=MENU_ROWS,
                          look_halation=True)


def render_pages(scale: int, frame: np.ndarray) -> list[tuple[str, str, Image.Image]]:
    """(file stem, label, image) for every screen, in the contact sheet's order."""
    state = LiveState(cells=CELLS, selected=0, image_adjusted=True, stabilise_on=True)
    images = [
        ("live", "LIVE", render_live(frame, READING, state, scale=scale)),
        ("live-menu-pressed", "LIVE, MENU PRESSED",
         render_live(frame, READING, LiveState(cells=CELLS, selected=0, pressed=Action.MENU,
                                               image_adjusted=True, stabilise_on=True),
                     scale=scale)),
    ]
    with tempfile.TemporaryDirectory() as state_dir:
        loop = _menu_loop(Path(state_dir))
        snap = loop._controller.snapshot()
        for stem, label, page in PAGES:
            loop._open_page(page)
            rows = loop._menu_rows(snap, 0.0)
            # A copy: render_menu's image is cached and shared.
            image = render_menu(rows, scale=scale, title=loop._menu_title()).copy()
            images.append((stem, label, image))
    return images


def contact_sheet(images: list[tuple[str, str, Image.Image]], columns: int = 4) -> Image.Image:
    """The screens in a grid on dark grey, each labelled underneath."""
    width, height = images[0][2].size
    gap, label_h = width // 20, max(16, height // 12)
    font = ImageFont.truetype(
        str(resources.files("pifilm") / "data" / "fonts" / "BarlowCondensed-SemiBold.ttf"),
        int(label_h * 0.7))
    rows = -(-len(images) // columns)
    sheet = Image.new("RGB", (gap + columns * (width + gap),
                              gap + rows * (height + label_h + gap)), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for index, (_, label, image) in enumerate(images):
        col, row = index % columns, index // columns
        x, y = gap + col * (width + gap), gap + row * (height + label_h + gap)
        sheet.paste(image, (x, y))
        draw.rectangle((x - 1, y - 1, x + width, y + height), outline=(90, 90, 90))
        draw.text((x + width // 2, y + height + label_h // 2), label, font=font,
                  fill=(235, 235, 230), anchor="mm")
    return sheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scale", type=int, choices=(1, 2), default=2,
                    help="1: the 2.8\" SPI panel (320x240); 2: the 3.5\" DSI panel (640x480)")
    ap.add_argument("--frame", type=Path, help="a JPEG to show in the live view")
    args = ap.parse_args(argv)
    if args.frame is not None:
        from pifilm.imageio import load_rgb
        try:
            frame, _ = load_rgb(args.frame)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read {args.frame}: {exc}", file=sys.stderr)
            return 2
    else:
        frame = sample_frame()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = render_pages(args.scale, frame)
    for stem, _, image in images:
        image.save(args.out_dir / f"{stem}.png")
    contact_sheet(images).save(args.out_dir / "contact-sheet.png")
    print(f"wrote {len(images) + 1} PNGs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
