from pifilm.display.cst3530 import (
    ADDRESS,
    CST3530Touch,
    RawPoint,
    Tap,
    TapDetector,
    TouchPoint,
    decode_points,
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
