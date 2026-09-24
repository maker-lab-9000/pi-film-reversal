import sys
from types import SimpleNamespace

import pytest

from pifilm.display import DisplayError
from pifilm.display.cst3530 import (
    ADDRESS,
    CST3530Touch,
    RawPoint,
    Tap,
    TapDetector,
    TouchPoint,
    decode_points,
    open_waveshare28_touch,
    to_display,
)


def _frame(x, y, strength=40):
    # vendor layout: point i at buf[4+5i .. 8+5i]: x low, y low, strength, hi nibbles
    return bytes([x & 0xFF, y & 0xFF, strength, ((y >> 4) & 0xF0) | ((x >> 8) & 0x0F), 0x00])


def _buf(points):
    buf = bytearray(bytes([0, 0, 0, len(points)]) + b"".join(_frame(*p) for p in points))
    buf[8] = 0xA0  # the vendor treats buf[8] & 0xF0 == 0 as "no report"
    return bytes(buf)


def test_decode_single_point_matches_vendor_bit_layout():
    buf = _buf([(0x12C, 0x0F0)])  # x=300, y=240
    assert decode_points(buf, 1) == [RawPoint(300, 240, 40)]


def test_decode_two_points():
    buf = _buf([(10, 20), (200, 300)])
    assert decode_points(buf, 2) == [RawPoint(10, 20, 40), RawPoint(200, 300, 40)]


def test_to_display_rotations():
    raw = RawPoint(10, 50, 1)
    assert to_display(raw, 0) == TouchPoint(50, 229, 1)
    assert to_display(raw, 180) == TouchPoint(269, 10, 1)


class FakeI2C:
    def __init__(self, reports):
        self.reports = list(reports)  # each: (first9: bytes, extra: bytes)
        self.writes = []
        self.pending = b""
        self.closed = False

    def write_i2c_block_data(self, address, first, rest):
        assert address == ADDRESS
        reg = (first << 24) | (rest[0] << 16) | (rest[1] << 8) | rest[2]
        self.writes.append(reg)
        if reg == 0xD0070000:
            first9, extra = self.reports.pop(0) if self.reports else (bytes(9), b"")
            self.pending = first9
            self._extra = extra
        elif reg == 0xD0070900:
            self.pending = self._extra

    def read_byte(self, address):
        b, self.pending = self.pending[0], self.pending[1:]
        return b

    def close(self):
        self.closed = True


def test_read_returns_no_points_and_ends_the_read_on_an_empty_report():
    i2c = FakeI2C([(bytes(9), b"")])
    touch = CST3530Touch(i2c)
    assert touch.read() == []
    assert i2c.writes == [0xD0070000, 0xD00002AB]


def test_read_decodes_and_rotates_a_two_finger_report():
    buf = _buf([(10, 20), (200, 300)])
    i2c = FakeI2C([(buf[:9], buf[9:])])
    touch = CST3530Touch(i2c, rotate=0)
    assert touch.read() == [TouchPoint(20, 229, 40), TouchPoint(300, 39, 40)]
    assert i2c.writes == [0xD0070000, 0xD0070900, 0xD00002AB]


def test_close_releases_the_bus():
    i2c = FakeI2C([])
    CST3530Touch(i2c).close()
    assert i2c.closed


class FakeResetPin:
    def __init__(self, pin, active_high=True, initial_value=True):
        self.pin = pin
        self.states = [initial_value]
        self.closed = False

    def on(self):
        self.states.append(True)

    def off(self):
        self.states.append(False)

    def close(self):
        self.closed = True


class DeadI2C:
    """A bus with nothing at 0x58: every transfer NAKs, as it does with the
    panel's ribbon unplugged."""

    def __init__(self):
        self.closed = False

    def write_i2c_block_data(self, address, first, rest):
        raise OSError(121, "Remote I/O error")

    def read_byte(self, address):
        raise OSError(121, "Remote I/O error")

    def close(self):
        self.closed = True


def _install_touch_libraries(monkeypatch, bus):
    pins = []

    def make_pin(pin, active_high=True, initial_value=True):
        pins.append(FakeResetPin(pin, active_high, initial_value))
        return pins[-1]

    monkeypatch.setitem(
        sys.modules, "gpiozero", SimpleNamespace(DigitalOutputDevice=make_pin)
    )
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=lambda n: bus))
    return pins


def test_close_releases_the_bus_and_holds_the_reset_pin_until_then(monkeypatch):
    """TP_RST must stay an output for the panel's lifetime: closing the pin
    right after the reset pulse returns GPIO 17 to input and can leave the
    controller in an undefined reset state."""
    i2c = FakeI2C([(bytes(9), b"")])
    pins = _install_touch_libraries(monkeypatch, i2c)
    touch = open_waveshare28_touch(0, sleep=lambda _s: None)
    rst, = pins
    assert rst.closed is False
    touch.close()
    assert i2c.closed and rst.closed


def test_open_probes_the_controller_before_returning(monkeypatch):
    i2c = FakeI2C([(bytes(9), b"")])
    _install_touch_libraries(monkeypatch, i2c)
    open_waveshare28_touch(0, sleep=lambda _s: None)
    assert i2c.writes == [0xD0070000, 0xD00002AB]


def test_open_raises_when_the_controller_does_not_answer(monkeypatch):
    """Without a probe the factory succeeds against a panel that is not there,
    and every frame's read() then raises: a 10 Hz log flood forever."""
    bus = DeadI2C()
    pins = _install_touch_libraries(monkeypatch, bus)
    with pytest.raises(DisplayError) as excinfo:
        open_waveshare28_touch(0, sleep=lambda _s: None)
    assert "touch controller at 0x58 not answering on /dev/i2c-1" in str(excinfo.value)
    assert bus.closed and pins[0].closed


def test_tap_detector_emits_on_release_within_limits():
    det = TapDetector()
    assert det.feed([TouchPoint(100, 100, 5)], 0.0) is None
    assert det.feed([TouchPoint(104, 101, 5)], 0.1) is None
    assert det.feed([], 0.2) == Tap(100, 100)


def test_tap_detector_ignores_long_holds_and_drags():
    det = TapDetector()
    det.feed([TouchPoint(100, 100, 5)], 0.0)
    assert det.feed([], 1.0) is None
    det.feed([TouchPoint(100, 100, 5)], 2.0)
    det.feed([TouchPoint(150, 100, 5)], 2.1)
    assert det.feed([], 2.2) is None
