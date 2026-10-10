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
        self.calls = []             # every call in order, for the tests that need it
        self.closed = False
        self.closes = 0
        self.fail_flip = None
        self.fail_power = None

    def write(self, index, pixels):
        self.calls.append("write")
        self.buffers[index] = pixels.copy()

    def flip(self, index):
        if self.fail_flip is not None:
            raise self.fail_flip
        self.calls.append("flip")
        self.flips.append(index)

    def power(self, on):
        if self.fail_power is not None:
            raise self.fail_power
        self.calls.append(("power", on))
        self.power_calls.append(on)

    def close(self):
        self.closed = True
        self.closes += 1


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


@pytest.mark.parametrize("error", [
    OSError(16, "busy"),
    RuntimeError("commit failed"),
    ValueError("bad buffer"),       # pybind's mapping of a C++ invalid_argument
])
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


def test_a_power_failure_of_any_type_is_a_display_error():
    backend = FakeBackend()
    backend.fail_power = ValueError("bad buffer")
    with pytest.raises(DisplayError, match="power off failed"):
        KmsDisplay(backend).backlight(0)


def test_close_contains_a_power_failure_of_any_type_and_still_closes():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(0)
    backend.fail_power = ValueError("bad buffer")
    display.close()
    assert backend.closed


def test_show_powers_a_dark_panel_on_before_the_swap():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(0)
    display.show(_image())
    assert backend.calls == [("power", False), ("power", True), "write", "flip"]
    display.show(_image())
    assert backend.power_calls == [False, True]     # lit now: no further power calls


def test_show_retries_a_power_on_that_failed():
    """The viewfinder loop sets the level once and does not retry, so a power-on
    that failed must not leave the panel dark while frames are drawn to it."""
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(0)
    backend.fail_power = OSError(16, "busy")
    with pytest.raises(DisplayError, match="power on failed"):
        display.backlight(80)
    with pytest.raises(DisplayError, match="power on failed"):
        display.show(_image())
    assert backend.flips == []
    backend.fail_power = None
    display.show(_image())
    assert backend.calls == [("power", False), ("power", True), "write", "flip"]


def test_close_twice_closes_the_backend_once():
    backend = FakeBackend()
    display = KmsDisplay(backend)
    display.backlight(0)
    display.close()
    display.close()
    assert backend.closes == 1
    assert backend.power_calls == [False, True]


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


# -- _PykmsBackend against a fake pykms -------------------------------------------


def _fake_pykms(set_mode_result):
    """A stand-in for the binding with exactly the calls ``_PykmsBackend.__init__``
    makes. ``seen`` records them, so a test can check what reached the mode-set."""
    seen = {"connector": None, "framebuffers": [], "set_mode": None}

    class Card:
        pass

    class Crtc:
        primary_plane = object()

        def set_mode(self, conn, fb, mode):
            seen["set_mode"] = (conn, fb, mode)
            return set_mode_result

    class Connector:
        def get_default_mode(self):
            return SimpleNamespace(hdisplay=640, vdisplay=480)

    class ResourceManager:
        def __init__(self, card):
            assert isinstance(card, Card)

        def reserve_connector(self, name):
            seen["connector"] = name
            return Connector()

        def reserve_crtc(self, conn):
            assert isinstance(conn, Connector)
            return Crtc()

    class DumbFramebuffer:
        def __init__(self, card, width, height, fmt):
            assert isinstance(card, Card)
            self._width, self._height = width, height
            self.id = 100 + len(seen["framebuffers"])
            seen["framebuffers"].append((width, height, fmt))

        def stride(self, plane):
            return self._width * 4

        def map(self, plane):
            return bytearray(self._height * self._width * 4)

    module = SimpleNamespace(
        Card=Card, ResourceManager=ResourceManager, DumbFramebuffer=DumbFramebuffer,
    )
    return module, seen


def test_open_fails_when_the_mode_set_is_refused(monkeypatch):
    """``Crtc.set_mode`` reports a refusal (no DRM master: a desktop session, or a
    second copy of the program) as a return value, not an exception."""
    module, seen = _fake_pykms(set_mode_result=-13)
    monkeypatch.setitem(sys.modules, "pykms", module)
    with pytest.raises(DisplayError, match="boots to the console") as raised:
        open_waveshare35dsi()
    assert "set_mode returned -13" in str(raised.value)
    assert seen["set_mode"] is not None


@pytest.mark.parametrize("result", [0, None])
def test_open_succeeds_when_the_mode_set_is_accepted(monkeypatch, result):
    module, seen = _fake_pykms(set_mode_result=result)
    monkeypatch.setitem(sys.modules, "pykms", module)
    display = open_waveshare35dsi()
    assert (display.width, display.height) == (640, 480)
    assert seen["connector"] == "DSI-1"
    assert seen["framebuffers"] == [(640, 480, "XR24")] * 2
    _, first_fb, mode = seen["set_mode"]
    assert first_fb.id == 100 and (mode.hdisplay, mode.vdisplay) == (640, 480)
