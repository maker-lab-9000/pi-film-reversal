"""ST7789 driver for the Waveshare 2.8" Capacitive Touch LCD (V2).

Wiring (BCM GPIO → physical pin): MOSI 10 → 19, SCLK 11 → 23, CS 8 (SPI0 CE0) → 24,
DC 25 → 22, RST 27 → 13, BL 18 → 12, VCC 3.3 V → 1, GND → 6. MISO is not connected;
the panel is write-only. GPIO 5/6/12/16/20/26 belong to the X728 UPS and must not
be used.

The init sequence is the vendor's byte for byte. The frame push is not: the vendor
code converts the frame to a Python list and, in its landscape path, writes every
frame twice. Here the frame is packed to RGB565 with NumPy and written once in
``CHUNK``-byte pieces (spidev's transfer limit).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from . import DisplayError

WIDTH, HEIGHT = 320, 240
DC_PIN, RST_PIN, BL_PIN = 25, 27, 18
SPI_BUS, SPI_DEVICE, SPI_HZ = 0, 0, 40_000_000
CHUNK = 4096
MADCTL = {0: 0x70, 180: 0xB0}
BACKLIGHT_DEFAULT = 80

# (command, data bytes) — vendor ST7789_Init, then a 100 ms pause and DISPON (0x29).
INIT_SEQUENCE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (0x36, (0x00,)),
    (0x3A, (0x05,)),
    (0xB2, (0x0B, 0x0B, 0x00, 0x33, 0x35)),
    (0xB7, (0x11,)),
    (0xBB, (0x35,)),
    (0xC0, (0x2C,)),
    (0xC2, (0x01,)),
    (0xC3, (0x0D,)),
    (0xC4, (0x20,)),
    (0xC6, (0x13,)),
    (0xD0, (0xA4, 0xA1)),
    (0xD6, (0xA1,)),
    (0xE0, (0xF0, 0x06, 0x0B, 0x0A, 0x09, 0x26, 0x29, 0x33, 0x41, 0x18, 0x16, 0x15, 0x29, 0x2D)),
    (0xE1, (0xF0, 0x04, 0x08, 0x08, 0x07, 0x03, 0x28, 0x32, 0x40, 0x3B, 0x19, 0x18, 0x2A, 0x2E)),
    (0x21, ()),
    (0x11, ()),
)


def rgb565_bytes(rgb: np.ndarray) -> bytes:
    """Pack an (H, W, 3) uint8 array as big-endian RGB565, row-major."""
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    packed = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return packed.astype(">u2").tobytes()


class ST7789Display:
    def __init__(
        self, spi: Any, dc: Any, rst: Any, bl: Any, *, rotate: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rotate not in MADCTL:
            raise DisplayError(f"rotate must be one of {sorted(MADCTL)}, got {rotate}")
        self._spi, self._dc, self._rst, self._bl = spi, dc, rst, bl
        self._sleep = sleep
        self.width, self.height = WIDTH, HEIGHT
        self.backlight(BACKLIGHT_DEFAULT)
        self._reset()
        for command, data in INIT_SEQUENCE:
            self._command(command)
            if data:
                self._data(bytes(data))
        self._sleep(0.1)
        self._command(0x29)
        self._command(0x36)
        self._data(bytes([MADCTL[rotate]]))

    def _write(self, data: bytes) -> None:
        try:
            for i in range(0, len(data), CHUNK):
                self._spi.writebytes2(data[i:i + CHUNK])
        except OSError as exc:
            raise DisplayError(f"SPI write failed: {exc}") from exc

    def _command(self, command: int) -> None:
        self._dc.off()
        self._write(bytes([command]))

    def _data(self, data: bytes) -> None:
        self._dc.on()
        self._write(data)

    def _reset(self) -> None:
        self._rst.on()
        self._sleep(0.01)
        self._rst.off()
        self._sleep(0.01)
        self._rst.on()
        self._sleep(0.01)

    def _set_window(self) -> None:
        self._command(0x2A)
        self._data(bytes([0, 0, (WIDTH - 1) >> 8, (WIDTH - 1) & 0xFF]))
        self._command(0x2B)
        self._data(bytes([0, 0, (HEIGHT - 1) >> 8, (HEIGHT - 1) & 0xFF]))
        self._command(0x2C)

    def backlight(self, percent: int) -> None:
        self._bl.value = max(0, min(100, int(percent))) / 100

    def show(self, image: Image.Image) -> None:
        if image.size != (WIDTH, HEIGHT):
            raise DisplayError(
                f"frame must be {WIDTH}x{HEIGHT}, got {image.size[0]}x{image.size[1]}"
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        payload = rgb565_bytes(np.asarray(image))
        self._set_window()
        self._data(payload)

    def close(self) -> None:
        try:
            self._command(0x28)
            self._command(0x10)
        except DisplayError:
            pass
        finally:
            self.backlight(0)
            for obj in (self._spi, self._dc, self._rst, self._bl):
                close = getattr(obj, "close", None)
                if callable(close):
                    close()


def open_waveshare28(rotate: int = 0) -> ST7789Display:
    """Open the panel on SPI0 CE0 with gpiozero pins. Raises DisplayError with the cause."""
    try:
        import spidev
        from gpiozero import DigitalOutputDevice, PWMOutputDevice
    except ImportError as exc:
        raise DisplayError(
            "display libraries missing; install python3-spidev and python3-gpiozero from apt"
        ) from exc
    try:
        spi = spidev.SpiDev()
        spi.open(SPI_BUS, SPI_DEVICE)
        spi.max_speed_hz = SPI_HZ
        spi.mode = 0b00
    except PermissionError as exc:
        raise DisplayError(
            f"no permission for /dev/spidev{SPI_BUS}.{SPI_DEVICE}; add the service user to the "
            "spi group and log in again"
        ) from exc
    except OSError as exc:
        raise DisplayError(
            f"cannot open /dev/spidev{SPI_BUS}.{SPI_DEVICE}: {exc}; is dtparam=spi=on set?"
        ) from exc
    try:
        dc = DigitalOutputDevice(DC_PIN, active_high=True, initial_value=True)
        rst = DigitalOutputDevice(RST_PIN, active_high=True, initial_value=True)
        bl = PWMOutputDevice(BL_PIN, frequency=1000)
    except Exception as exc:  # gpiozero raises library-specific errors for a busy pin
        spi.close()
        raise DisplayError(
            f"cannot claim display GPIO {DC_PIN}/{RST_PIN}/{BL_PIN}: {exc}; remove any "
            "dtoverlay=fbtft line from config.txt and check the gpio group"
        ) from exc
    return ST7789Display(spi, dc, rst, bl, rotate=rotate)
