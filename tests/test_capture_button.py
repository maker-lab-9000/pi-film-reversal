import sys

import pytest

from pifilm.capture.button import (
    MAX_PIN,
    MIN_PIN,
    WAVESHARE28_PINS,
    X728_PINS,
    ButtonError,
    ShutterButton,
    open_shutter_button,
)


class FakeDevice:
    """Stands in for gpiozero.Button: a ``when_pressed`` slot and ``close``."""

    def __init__(self):
        self.when_pressed = None
        self.closed = False

    def press(self):
        if self.when_pressed is not None:
            self.when_pressed()

    def close(self):
        self.closed = True


def test_a_press_calls_the_callback_once():
    device = FakeDevice()
    button = open_shutter_button(21, open_device=lambda pin: device)
    presses = []
    button.on_press(lambda: presses.append(1))
    device.press()
    assert presses == [1]


def test_the_pin_number_reaches_the_device_factory():
    seen = []

    def factory(pin):
        seen.append(pin)
        return FakeDevice()

    open_shutter_button(21, open_device=factory)
    assert seen == [21]


def test_presses_before_a_callback_is_attached_do_nothing():
    device = FakeDevice()
    open_shutter_button(21, open_device=lambda pin: device)
    device.press()   # must not raise


def test_a_failing_callback_is_logged_and_the_next_press_still_works(capsys):
    device = FakeDevice()
    button = open_shutter_button(21, open_device=lambda pin: device)
    calls = []

    def callback():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("controller exploded")

    button.on_press(callback)
    device.press()
    device.press()
    assert calls == [1, 1]
    assert "button: controller exploded" in capsys.readouterr().err


def test_close_detaches_the_callback_and_releases_the_pin_once():
    device = FakeDevice()
    button = open_shutter_button(21, open_device=lambda pin: device)
    presses = []
    button.on_press(lambda: presses.append(1))
    button.close()
    button.close()
    assert device.closed and device.when_pressed is None
    device.press()
    assert presses == []


def test_a_missing_library_is_a_button_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", None)   # makes `import gpiozero` raise
    with pytest.raises(ButtonError, match="python3-gpiozero"):
        open_shutter_button(21)


@pytest.mark.parametrize("error", [
    RuntimeError("GPIO busy"), PermissionError(13, "Permission denied"), OSError(16, "busy"),
])
def test_a_pin_that_cannot_be_claimed_is_a_button_error(error):
    def factory(pin):
        raise error

    with pytest.raises(ButtonError, match="cannot claim BCM 21"):
        open_shutter_button(21, open_device=factory)


@pytest.mark.parametrize("pin", [-1, 0, 3, 28, 40])
def test_a_pin_outside_the_usable_range_is_refused(pin):
    with pytest.raises(ButtonError, match="between 4 and 27"):
        open_shutter_button(pin, open_device=lambda pin: FakeDevice())


def test_the_reserved_pin_sets_match_the_documented_wiring():
    assert X728_PINS == {5, 6, 12, 16, 20, 26}
    assert WAVESHARE28_PINS == {8, 10, 11, 17, 18, 25, 27}
    assert (MIN_PIN, MAX_PIN) == (4, 27)
    assert 21 not in X728_PINS | WAVESHARE28_PINS


def test_shutter_button_is_constructible_from_a_device():
    assert isinstance(ShutterButton(FakeDevice()), ShutterButton)
