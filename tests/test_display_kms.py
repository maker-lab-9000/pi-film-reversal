import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pifilm.display import DisplayError
from pifilm.display.kms import KmsDisplay, open_waveshare35dsi, pack_xrgb8888


class FakeBackend:
    width, height = 640, 480

    def __init__(self):
        self.buffers = [None, None]
        self.flips = []
        self.power_calls = []
        self.closed = False
        self.fail_flip = None

    def write(self, index, pixels):
        self.buffers[index] = pixels.copy()

    def flip(self, index):
        if self.fail_flip is not None:
            raise self.fail_flip
        self.flips.append(index)

    def power(self, on):
        self.power_calls.append(on)

    def close(self):
        self.closed = True


def _image(colour=(10, 20, 30), size=(640, 480)):
    return Image.new("RGB", size, colour)


def test_pack_is_blue_green_red_then_opaque():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    out = pack_xrgb8888(rgb)
    assert out.shape == (2, 3, 4) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (3, 2, 1, 255)


def test_pack_rotates_180_by_flipping_both_axes():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    out = pack_xrgb8888(rgb, rotate=180)
    assert tuple(out[1, 2]) == (3, 2, 1, 255)
    assert tuple(out[0, 0]) == (0, 0, 0, 255)


def test_show_draws_into_the_hidden_buffer_and_alternates():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.show(_image((10, 20, 30)))
    display.show(_image((40, 50, 60)))
    display.show(_image((70, 80, 90)))
    assert backend.flips == [1, 0, 1]
    assert tuple(backend.buffers[0][0, 0]) == (60, 50, 40, 255)
    assert tuple(backend.buffers[1][0, 0]) == (90, 80, 70, 255)


def test_size_comes_from_the_backend_and_the_panel_is_not_dimmable():
    display = KmsDisplay(FakeBackend())
    assert (display.width, display.height) == (640, 480)
    assert display.dimmable is False


def test_show_rejects_a_wrong_sized_frame_and_converts_other_modes():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    with pytest.raises(DisplayError, match="640x480"):
        display.show(_image(size=(320, 240)))
    display.show(Image.new("L", (640, 480), 128))
    assert tuple(backend.buffers[1][0, 0]) == (128, 128, 128, 255)


@pytest.mark.parametrize("error", [OSError(16, "busy"), RuntimeError("commit failed")])
def test_a_failed_swap_is_a_display_error_and_keeps_the_front_buffer(error):
    backend = FakeBackend()
    display = KmsDisplay(backend)
    backend.fail_flip = error
    with pytest.raises(DisplayError, match="swap failed"):
        display.show(_image())
    backend.fail_flip = None
    display.show(_image())
    assert backend.flips == [1]     # retried on the same hidden buffer


def test_backlight_only_switches_power_and_only_on_a_change():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(80)
    display.backlight(40)
    display.backlight(0)
    display.backlight(0)
    display.backlight(80)
    assert backend.power_calls == [False, True]


def test_a_power_failure_is_a_display_error():
    backend = FakeBackend()
    backend.power = lambda on: (_ for _ in ()).throw(OSError(22, "invalid"))
    with pytest.raises(DisplayError, match="power"):
        KmsDisplay(backend).backlight(0)


def test_close_restores_power_and_closes_even_if_power_fails():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(0)
    display.close()
    assert backend.power_calls == [False, True] and backend.closed

    broken = FakeBackend()
    display = KmsDisplay(broken)
    display.backlight(0)
    broken.power = lambda on: (_ for _ in ()).throw(OSError(22, "invalid"))
    display.close()
    assert broken.closed


def test_bad_rotation_is_rejected():
    with pytest.raises(DisplayError, match="rotate"):
        KmsDisplay(FakeBackend(), rotate=90)


def test_open_without_pykms_names_the_apt_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pykms", None)   # makes `import pykms` raise
    with pytest.raises(DisplayError, match="python3-kms\\+\\+"):
        open_waveshare35dsi()


@pytest.mark.parametrize("error", [
    RuntimeError("searching for connector DSI-1: not found"),
    PermissionError(13, "Permission denied"),
    OSError(16, "Device or resource busy"),
    ValueError("cannot reshape array"),
])
def test_open_maps_any_backend_failure_to_display_error(monkeypatch, error):
    def card():
        raise error

    monkeypatch.setitem(sys.modules, "pykms", SimpleNamespace(Card=card))
    with pytest.raises(DisplayError, match="DSI display"):
        open_waveshare35dsi()
