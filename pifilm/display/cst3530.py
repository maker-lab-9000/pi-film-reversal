"""CST3530 capacitive touch controller (Waveshare 2.8" V2) over I2C1.

Register protocol as in the vendor ``Touch_CST3530.py``: 32-bit register
addresses, the first byte sent as the SMBus "command" and the remaining three as
data; a report is 9 bytes at ``REG_DATA`` holding the point count in ``buf[3] &
0x0F`` and the first point, further points are 5 bytes each at ``REG_NEXT``, and
every read ends with a write to ``REG_END_READ``. Raw coordinates are in the
panel's 240x320 portrait space; ``to_display`` maps them to the 320x240 landscape
frame the viewfinder draws.

The bus (I2C1) is shared with the X728's fuel gauge (0x36) and RTC (0x68). Each
call here is one kernel ioctl and the controller's read pointer is not affected
by transactions to other addresses, so no cross-module locking is needed. The
driver polls instead of using the INT line: capacitive controllers pulse INT per
report rather than holding it, and a 10 Hz poll could miss the pulse.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from . import DisplayError

ADDRESS = 0x58
REG_DATA = 0xD0070000
REG_NEXT = 0xD0070900
REG_END_READ = 0xD00002AB
MAX_POINTS = 5
TP_RST_PIN = 17
PORTRAIT_W, PORTRAIT_H = 240, 320


@dataclass(frozen=True)
class RawPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


def decode_points(buf: bytes, count: int) -> list[RawPoint]:
    points = []
    for i in range(count):
        base = 4 + 5 * i
        hi = buf[base + 3]
        x = ((hi & 0x0F) << 8) | buf[base]
        y = ((hi & 0xF0) << 4) | buf[base + 1]
        points.append(RawPoint(x, y, buf[base + 2]))
    return points


def to_display(point: RawPoint, rotate: int) -> TouchPoint:
    if rotate == 0:
        return TouchPoint(point.y, PORTRAIT_W - 1 - point.x, point.strength)
    if rotate == 180:
        return TouchPoint(PORTRAIT_H - 1 - point.y, point.x, point.strength)
    raise DisplayError(f"rotate must be 0 or 180, got {rotate}")


def _split(reg: int) -> tuple[int, list[int]]:
    return (reg >> 24) & 0xFF, [(reg >> 16) & 0xFF, (reg >> 8) & 0xFF, reg & 0xFF]


class CST3530Touch:
    def __init__(self, i2c: Any, *, rotate: int = 0, address: int = ADDRESS) -> None:
        self._i2c = i2c
        self._rotate = rotate
        self._address = address

    def _read(self, reg: int, count: int) -> bytes:
        first, rest = _split(reg)
        self._i2c.write_i2c_block_data(self._address, first, rest)
        return bytes(self._i2c.read_byte(self._address) for _ in range(count))

    def _write(self, reg: int) -> None:
        first, rest = _split(reg)
        self._i2c.write_i2c_block_data(self._address, first, rest)

    def read(self) -> list[TouchPoint]:
        try:
            buf = self._read(REG_DATA, 9)
            count = buf[3] & 0x0F
            if count == 0 or count > MAX_POINTS or (buf[8] & 0xF0) == 0:
                self._write(REG_END_READ)
                return []
            extra = self._read(REG_NEXT, (count - 1) * 5) if count > 1 else b""
            self._write(REG_END_READ)
        except OSError as exc:
            raise DisplayError(f"touch I2C read failed: {exc}") from exc
        return [to_display(p, self._rotate) for p in decode_points(buf + extra, count)]

    def close(self) -> None:
        close = getattr(self._i2c, "close", None)
        if callable(close):
            close()


class TapDetector:
    """Turn per-step point lists into taps: down then up within ``max_hold`` s,
    moving less than ``max_move`` px. Holds and drags produce nothing."""

    def __init__(self, max_hold: float = 0.6, max_move: float = 20.0) -> None:
        self._max_hold, self._max_move = max_hold, max_move
        self._down: tuple[TouchPoint, float] | None = None
        self._last: TouchPoint | None = None

    def feed(self, points: list[TouchPoint], now: float) -> Tap | None:
        if points:
            if self._down is None:
                self._down = (points[0], now)
            self._last = points[0]
            return None
        if self._down is None:
            return None
        first, t0 = self._down
        last = self._last or first
        self._down, self._last = None, None
        moved = math.hypot(last.x - first.x, last.y - first.y)
        if now - t0 <= self._max_hold and moved < self._max_move:
            return Tap(first.x, first.y)
        return None


def open_waveshare28_touch(rotate: int = 0, sleep=time.sleep) -> CST3530Touch:
    try:
        from gpiozero import DigitalOutputDevice
        from smbus2 import SMBus
    except ImportError as exc:
        raise DisplayError(
            "touch libraries missing; install python3-smbus2 and python3-gpiozero from apt"
        ) from exc
    try:
        rst = DigitalOutputDevice(TP_RST_PIN, active_high=True, initial_value=True)
        rst.off()
        sleep(0.1)
        rst.on()
        sleep(0.5)
        rst.close()
    except Exception as exc:
        raise DisplayError(
            f"cannot reset the touch controller on GPIO {TP_RST_PIN}: {exc}"
        ) from exc
    try:
        bus = SMBus(1)
    except PermissionError as exc:
        raise DisplayError(
            "no permission for /dev/i2c-1; add the service user to the i2c group"
        ) from exc
    except OSError as exc:
        raise DisplayError(f"cannot open /dev/i2c-1: {exc}; is dtparam=i2c_arm=on set?") from exc
    return CST3530Touch(bus, rotate=rotate)
