"""A physical shutter button on one GPIO pin.

The button is a momentary switch between a GPIO pin and ground. The pin is held
high by the Pi's internal pull-up and a press pulls it low, so the switch needs
no resistor and no power wire: two jumper leads. The documented wiring is BCM 21
(physical pin 40) and the ground beside it (physical pin 39), the last two pins
of the header, which neither the X728 UPS nor either LCD uses.

A press is delivered to one callback. ``pifilm-capture`` makes that callback a
``CaptureController.submit``, the same call the LCD tap makes, so the button is
one more trigger and never a second owner of the camera; a press while a capture
is running is answered ``busy`` by the controller and dropped.

gpiozero calls ``when_pressed`` on its own thread. An exception raised there
would be printed by gpiozero and, depending on its version, can stop further
callbacks, so the callback is wrapped: a failure is logged as ``button: ...`` and
the next press still works.

gpiozero is imported only inside ``_open_device`` so this module imports on a Mac
and in the tests, which pass ``open_device`` instead.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

BOUNCE_SECONDS = 0.05
# BCM 0-3 are the ID EEPROM and the I2C bus the X728 gauge and RTC sit on; the
# header stops at 27.
MIN_PIN, MAX_PIN = 4, 27
# Pins that belong to other hardware this project supports (docs/x728-ups.md
# section 3, docs/lcd-viewfinder.md section 2). The command line refuses them
# when that hardware is selected.
X728_PINS = frozenset({5, 6, 12, 16, 20, 26})
WAVESHARE28_PINS = frozenset({8, 10, 11, 17, 18, 25, 27})


class ButtonError(Exception):
    """The shutter button's pin could not be opened."""


class ShutterButton:
    def __init__(self, device: Any) -> None:
        self._device = device
        self._closed = False

    def on_press(self, callback: Callable[[], None]) -> None:
        def guarded() -> None:
            try:
                callback()
            except Exception as exc:  # must not escape into gpiozero's thread
                print(f"button: {exc}", file=sys.stderr)

        self._device.when_pressed = guarded

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._device.when_pressed = None
        self._device.close()


def _open_device(pin: int) -> Any:
    from gpiozero import Button  # lazy: only present on the Pi

    return Button(pin, pull_up=True, bounce_time=BOUNCE_SECONDS)


def open_shutter_button(
    pin: int, *, open_device: Callable[[int], Any] | None = None,
) -> ShutterButton:
    """Claim ``pin`` (BCM numbering) as the shutter button. Raises ButtonError."""
    if not MIN_PIN <= pin <= MAX_PIN:
        raise ButtonError(f"BCM {pin} is not usable; choose a pin between {MIN_PIN} and {MAX_PIN}")
    try:
        device = (open_device or _open_device)(pin)
    except ImportError as exc:
        raise ButtonError(
            "python3-gpiozero is not installed; install it from apt"
        ) from exc
    except Exception as exc:  # gpiozero raises library-specific errors for a busy pin
        raise ButtonError(
            f"cannot claim BCM {pin}: {exc}; is another process or a dtoverlay using it, "
            "and is the service user in the gpio group?"
        ) from exc
    return ShutterButton(device)
