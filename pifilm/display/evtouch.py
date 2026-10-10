"""Goodix GT911 touch for the Waveshare 3.5" DSI LCD (E), via kernel input events.

The DSI panel's touch controller is driven by the kernel's ``goodix_ts`` driver on
the DSI cable's own I2C bus, so there is nothing to poll over I2C and no reset
line: the kernel publishes multi-touch events on a ``/dev/input/eventN`` node.
This module reads that node with the standard library only (``python3-evdev`` is
not needed for one device and three event codes).

The node number is not stable (it was ``event4`` on 2026-10-10 and moves when a
USB device is added), so ``open_goodix_touch`` finds the device by its name.

The viewfinder loop polls ``read()`` and its ``TapDetector`` needs to see the finger
down and then up. Two things follow. A press and release that both arrive between
two polls would be invisible as state, so ``SlotTracker`` latches "pressed since the
last poll" and reports the point once. And a poll can land in the middle of a
report (the tracking id read, the coordinates not yet), so state is committed only
on ``SYN_REPORT``; otherwise a new press would briefly carry the previous touch's
coordinates and fire whatever control was last touched.

Only multi-touch slot 0 is followed: the UI has no gestures, and a palm or second
finger must not move the first one.

The event record is the 64-bit layout (two 8-byte time fields); the Pi runs a
64-bit userland.
"""

from __future__ import annotations

import fcntl
import os
import struct
from collections.abc import Callable, Iterable
from pathlib import Path

from . import DisplayError
from .touch import TouchPoint

DEVICE_NAME = "Goodix Capacitive TouchScreen"
RECORD = struct.Struct("<qqHHi")
EV_SYN, EV_ABS = 0x00, 0x03
SYN_REPORT = 0x00
ABS_MT_SLOT, ABS_MT_POSITION_X, ABS_MT_POSITION_Y, ABS_MT_TRACKING_ID = 0x2F, 0x35, 0x36, 0x39
READ_RECORDS = 64

Event = tuple[int, int, int]


def decode_events(data: bytes) -> list[Event]:
    """Split a read into (type, code, value) triples; a partial tail is dropped."""
    whole = len(data) - len(data) % RECORD.size
    return [(etype, code, value)
            for _, _, etype, code, value in RECORD.iter_unpack(data[:whole])]


class SlotTracker:
    """Follow multi-touch slot 0 through a stream of events."""

    def __init__(self) -> None:
        self._slot = 0
        self._pending_down: bool | None = None
        self._pending_x: int | None = None
        self._pending_y: int | None = None
        self._down = False
        self._x = self._y = 0
        self._pressed = False

    def feed(self, events: Iterable[Event]) -> None:
        for etype, code, value in events:
            if etype == EV_SYN and code == SYN_REPORT:
                self._commit()
            elif etype != EV_ABS:
                continue
            elif code == ABS_MT_SLOT:
                self._slot = value
            elif self._slot != 0:
                continue
            elif code == ABS_MT_TRACKING_ID:
                self._pending_down = value >= 0
            elif code == ABS_MT_POSITION_X:
                self._pending_x = value
            elif code == ABS_MT_POSITION_Y:
                self._pending_y = value

    def _commit(self) -> None:
        if self._pending_x is not None:
            self._x = self._pending_x
        if self._pending_y is not None:
            self._y = self._pending_y
        if self._pending_down is not None:
            self._down = self._pending_down
            if self._down:
                self._pressed = True
        self._pending_down = self._pending_x = self._pending_y = None

    def take(self) -> tuple[int, int] | None:
        """The raw position if the finger is down now, or was pressed since the last call."""
        seen = self._down or self._pressed
        self._pressed = False
        return (self._x, self._y) if seen else None


class EventTouch:
    def __init__(
        self, reader: Callable[[], bytes], x_range: tuple[int, int], y_range: tuple[int, int],
        size: tuple[int, int], *, rotate: int = 0, closer: Callable[[], None] | None = None,
    ) -> None:
        if rotate not in (0, 180):
            raise DisplayError(f"rotate must be 0 or 180, got {rotate}")
        if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
            raise DisplayError(f"touch device reports an empty range: x {x_range}, y {y_range}")
        self._reader, self._closer = reader, closer
        self._x_range, self._y_range = x_range, y_range
        self._width, self._height = size
        self._rotate = rotate
        self._tracker = SlotTracker()

    @staticmethod
    def _scale(value: int, lo: int, hi: int, pixels: int) -> int:
        mapped = round((value - lo) * (pixels - 1) / (hi - lo))
        return max(0, min(pixels - 1, mapped))

    def read(self) -> list[TouchPoint]:
        try:
            while True:
                data = self._reader()
                if not data:
                    break
                self._tracker.feed(decode_events(data))
        except OSError as exc:
            raise DisplayError(f"touch read failed: {exc}") from exc
        raw = self._tracker.take()
        if raw is None:
            return []
        x = self._scale(raw[0], *self._x_range, self._width)
        y = self._scale(raw[1], *self._y_range, self._height)
        if self._rotate == 180:
            x, y = self._width - 1 - x, self._height - 1 - y
        return [TouchPoint(x, y, 0)]

    def close(self) -> None:
        closer, self._closer = self._closer, None
        if closer is not None:
            closer()


def _ioc_read(number: int, size: int) -> int:
    """The ``_IOR('E', number, size)`` request code."""
    return (2 << 30) | (size << 16) | (ord("E") << 8) | number


def _device_name(fd: int) -> str:
    buf = bytearray(256)
    fcntl.ioctl(fd, _ioc_read(0x06, len(buf)), buf)
    return bytes(buf).split(b"\0", 1)[0].decode("utf-8", "replace")


def _abs_range(fd: int, axis: int) -> tuple[int, int]:
    buf = bytearray(24)  # struct input_absinfo: six 32-bit ints
    fcntl.ioctl(fd, _ioc_read(0x40 + axis, len(buf)), buf)
    _, low, high, _, _, _ = struct.unpack("<6i", buf)
    return low, high


def open_goodix_touch(
    size: tuple[int, int], rotate: int = 0, *, input_dir: Path = Path("/dev/input"),
) -> EventTouch:
    """Find the Goodix touchscreen by name and open it non-blocking."""
    denied = False
    for node in sorted(Path(input_dir).glob("event*")):
        try:
            fd = os.open(node, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            denied = True
            continue
        except OSError:
            continue
        try:
            if _device_name(fd) != DEVICE_NAME:
                os.close(fd)
                continue
            x_range = _abs_range(fd, ABS_MT_POSITION_X)
            y_range = _abs_range(fd, ABS_MT_POSITION_Y)
        except OSError:
            os.close(fd)
            continue

        def reader(fd: int = fd) -> bytes:
            try:
                return os.read(fd, RECORD.size * READ_RECORDS)
            except BlockingIOError:
                return b""

        try:
            return EventTouch(reader, x_range, y_range, size, rotate=rotate,
                              closer=lambda fd=fd: os.close(fd))
        except DisplayError:
            os.close(fd)
            raise
    if denied:
        raise DisplayError(
            f"no permission for {input_dir}/event*; add the service user to the input "
            "group and log in again"
        )
    raise DisplayError(
        f"touch device '{DEVICE_NAME}' not found under {input_dir}; check the DSI cable "
        "and the dtoverlay=waveshare_35DSI line in config.txt"
    )
