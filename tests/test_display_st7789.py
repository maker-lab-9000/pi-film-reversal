import numpy as np
import pytest
from PIL import Image

from pifilm.display import DisplayError
from pifilm.display.st7789 import (
    CHUNK,
    HEIGHT,
    INIT_SEQUENCE,
    MADCTL,
    WIDTH,
    ST7789Display,
    rgb565_bytes,
)


class FakePin:
    def __init__(self):
        self.level = 1
        self.history = []
        self.closed = False

    def on(self):
        self.level = 1
        self.history.append(1)

    def off(self):
        self.level = 0
        self.history.append(0)

    def close(self):
        self.closed = True


class FakeSpi:
    """Records each write together with the DC level in force, so command bytes
    (DC low) and data bytes (DC high) can be told apart afterwards."""

    def __init__(self, dc, fail=False):
        self.dc = dc
        self.writes = []      # bytes only
        self.tagged = []      # (dc_level, bytes)
        self.closed = False
        self.fail = fail

    def writebytes2(self, data):
        if self.fail:
            raise OSError(5, "spi")
        self.writes.append(bytes(data))
        self.tagged.append((self.dc.level, bytes(data)))

    def close(self):
        self.closed = True


class FakeBacklight:
    def __init__(self):
        self.value = 0.0
        self.closed = False

    def close(self):
        self.closed = True


def _display(rotate=0):
    dc, rst, bl = FakePin(), FakePin(), FakeBacklight()
    spi = FakeSpi(dc)
    disp = ST7789Display(spi, dc, rst, bl, rotate=rotate, sleep=lambda s: None)
    return disp, spi, dc, rst, bl


def test_rgb565_packs_big_endian_565():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    assert rgb565_bytes(rgb) == bytes([0xF8, 0x00, 0x07, 0xE0, 0x00, 0x1F, 0xFF, 0xFF])


def test_init_replays_the_vendor_sequence_then_sets_landscape_madctl():
    disp, spi, dc, rst, bl = _display()
    commands = [data[0] for level, data in spi.tagged if level == 0]
    vendor_commands = [cmd for cmd, _ in INIT_SEQUENCE] + [0x29]
    assert commands == vendor_commands + [0x36]  # MADCTL for rotation comes last
    data_after_madctl = spi.tagged[-1]
    assert data_after_madctl == (1, bytes([MADCTL[0]]))
    # the vendor data bytes travel with their commands, in order
    sent_data = [data for level, data in spi.tagged if level == 1]
    assert sent_data[0] == bytes([0x00]) and sent_data[1] == bytes([0x05])
    assert rst.history[:3] == [1, 0, 1]
    assert bl.value == pytest.approx(0.8)


def test_show_writes_one_frame_in_chunks():
    disp, spi, *_ = _display()
    spi.writes.clear()
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 0, 0))
    disp.show(image)
    data = b"".join(w for w in spi.writes if len(w) > 4)
    assert len(data) == WIDTH * HEIGHT * 2
    assert data[:2] == bytes([0xF8, 0x00])
    assert all(len(w) <= CHUNK for w in spi.writes)


def test_show_rejects_wrong_size():
    disp, *_ = _display()
    with pytest.raises(DisplayError, match="320x240"):
        disp.show(Image.new("RGB", (240, 320)))


def test_show_wraps_spi_errors():
    disp, spi, *_ = _display()
    spi.fail = True
    with pytest.raises(DisplayError):
        disp.show(Image.new("RGB", (WIDTH, HEIGHT)))


def test_rotate_180_uses_the_other_madctl():
    disp, spi, *_ = _display(rotate=180)
    assert spi.writes[-1] == bytes([MADCTL[180]])


def test_close_turns_the_panel_off_and_releases_everything():
    disp, spi, dc, rst, bl = _display()
    disp.close()
    commands = [w[0] for w in spi.writes[-4:]]
    assert 0x28 in commands and 0x10 in commands
    assert bl.value == 0
    assert spi.closed and dc.closed and rst.closed and bl.closed
