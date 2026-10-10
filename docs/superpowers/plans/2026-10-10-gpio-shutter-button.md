# GPIO Shutter Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A physical push button wired to the Pi's GPIO header takes a photo, exactly as the on-screen shutter or the Stick does.

**Architecture:** A new `pifilm/capture/button.py` claims one GPIO pin through gpiozero (lazy import) and calls a callback on each press. `pifilm-capture --shutter-gpio N` opens it and attaches the callback to the shared `CaptureController`, so the button is a third trigger beside the LCD and the Stick and needs no change to the viewfinder, the remote API or the capture path.

**Tech Stack:** Python 3.11+, gpiozero (`python3-gpiozero` from apt, Pi only, lazy import), pytest, ruff.

**Spec:** none; this is a bounded change. The design agreed in discussion on 2026-10-10 is the next section and is the authority for this plan.

## Design

| Topic | Decision |
| --- | --- |
| Action | One action: a press takes a photo. No long press, no half press. |
| Button | Momentary, normally open, two wires: one to a GPIO pin, one to ground. The Pi's internal pull-up is used; no resistor. |
| Pin | Chosen with `--shutter-gpio N` (BCM number). The documented wiring is BCM 21 (physical pin 40) and ground on physical pin 39: the last two pins of the header, opposite each other. No default: without the flag no pin is claimed. |
| Reserved pins | BCM 4 to 27 are accepted. With `--ups x728`, the shield's pins (5, 6, 12, 16, 20, 26) are refused; with `--display waveshare28`, that panel's pins (8, 10, 11, 17, 18, 25, 27) are refused. Both are argument errors (exit 2), before any hardware is touched. |
| Trigger path | `CaptureController.submit(new uuid)`, the same call the LCD tap makes. A press while a capture is running is ignored (the controller answers `busy`); presses are never queued. |
| Dark screen | A press with the screen off takes the photo, and the screen wakes into the review, as a Stick shot does. This needs no new code: the viewfinder loop already wakes on any finished job. |
| Modes | Works with or without the LCD and with or without the remote API. With the button and neither of those, the app uses the same shared-controller loops the remote API uses. |
| Failure | If the pin cannot be claimed, or gpiozero is missing, one line `warning: shutter button unavailable (<message>); continuing without it` and the service carries on. |
| Debounce | 50 ms, in gpiozero. A bounce that gets through is harmless: the second submit is answered `busy`. |
| Out of scope | Long-press actions, a second button, LED feedback, the shutter half-press, waking the screen without shooting. |

## Global Constraints

- Always use `.venv/bin/...`. Tests: `.venv/bin/pytest -q -m 'not slow'`. Lint: `.venv/bin/ruff check .` (rules E, F, I, B, UP; line length 100).
- `gpiozero` is imported only inside the device factory, so `pifilm.capture.button` and `pifilm.capture.app` import on a Mac.
- No new pip dependency.
- The flag is exactly `--shutter-gpio`. The warning text is exactly `warning: shutter button unavailable (<message>); continuing without it`.
- A button failure never exits the process and never takes the remote API or the viewfinder down.
- Existing behaviour without `--shutter-gpio` does not change: every existing test passes unmodified.
- Module docstrings carry the design rationale.
- Never put the token, Wi-Fi password or SSH password in a command, file or commit. Do not read or echo `.env` values.
- Work on branch `feat/gpio-shutter-button`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Task 4 runs on the Pi and needs the user.

## Review Focus

1. **Press while a capture is running.** The press must be ignored with one log line, not queued and not an error. Pinned in Task 2 (`test_a_press_while_busy_is_ignored_and_logged`).
2. **An exception inside the press callback.** gpiozero runs the callback on its own thread; an exception there must be logged and must not stop later presses. Pinned in Task 1 (`test_a_failing_callback_is_logged_and_the_next_press_still_works`).
3. **A reserved pin.** `--shutter-gpio 6 --ups x728` would fight the shield's power-loss line. Pinned in Task 2 (`test_shutter_gpio_refuses_reserved_pins`).
4. **A press after shutdown has begun.** The controller answers `closed`; the callback must not raise. Pinned in Task 2 (`test_a_press_after_close_is_ignored`).
5. **The pin is already claimed.** A leftover overlay or a second copy holds the pin: one warning, the service continues. Pinned in Task 2 (`test_main_warns_and_continues_when_the_button_cannot_open`).

## File Structure

| File | Responsibility |
| --- | --- |
| `pifilm/capture/button.py` (new) | Claim the pin, deliver presses to one callback, release the pin |
| `pifilm/capture/app.py` (modify) | `--shutter-gpio`, pin checks, attach the button to the controller |
| `deploy/pifilm-capture.service.example` (modify) | Passes `--shutter-gpio 21` |
| `tests/test_capture_button.py` (new), `tests/test_app.py` (modify) | Unit tests |
| `docs/setup.md`, `docs/x728-ups.md`, `README.md`, `CLAUDE.md` (modify) | Documentation |

---

### Task 1: The button module

**Files:**
- Create: `pifilm/capture/button.py`
- Test: `tests/test_capture_button.py`

**Interfaces:**
- Produces: `open_shutter_button(pin: int, *, open_device: Callable[[int], Any] | None = None) -> ShutterButton` (raises `ButtonError`); `ShutterButton.on_press(callback: Callable[[], None]) -> None`; `ShutterButton.close() -> None`; `ButtonError(Exception)`; `X728_PINS: frozenset[int]`, `WAVESHARE28_PINS: frozenset[int]`, `MIN_PIN = 4`, `MAX_PIN = 27`.

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull --ff-only && git checkout -b feat/gpio-shutter-button
git add docs/superpowers/plans/2026-10-10-gpio-shutter-button.md
git commit -m "docs: GPIO shutter button plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_capture_button.py`:

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_capture_button.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pifilm.capture.button'`.

- [ ] **Step 4: Create `pifilm/capture/button.py`**

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest -q tests/test_capture_button.py && .venv/bin/ruff check pifilm/capture/button.py tests/test_capture_button.py`
Expected: 16 passed, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pifilm/capture/button.py tests/test_capture_button.py
git commit -m "capture: a shutter button on one GPIO pin

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire the button into `pifilm-capture`

**Files:**
- Modify: `pifilm/capture/app.py` (imports, a `_gpio_pin` argument type, `_button_capture`, the `--shutter-gpio` argument, pin checks in `main`, the controller/loop block in `main`, the `finally` block)
- Modify: `deploy/pifilm-capture.service.example`
- Test: `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `open_shutter_button(pin)`, `ShutterButton.on_press(callback)`, `.close()`, `ButtonError`, `X728_PINS`, `WAVESHARE28_PINS`, `MIN_PIN`, `MAX_PIN` from Task 1; `CaptureController.submit(request_id: str) -> JobSnapshot` (fields `state`, `error_code`).
- Produces: `pifilm-capture --shutter-gpio N`; `app._button_capture(controller) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app.py`. `camera_cli` is the existing fixture (it stubs the pipeline, reports no desktop display, and gives a terminal that reads `q`); `argparse`, `SimpleNamespace` and `pytest` are already imported at the top of the file.

```python
# -- the GPIO shutter button --------------------------------------------------------


class _ButtonDouble:
    def __init__(self):
        self.callback = None
        self.closed = False

    def on_press(self, callback):
        self.callback = callback

    def close(self):
        self.closed = True


class _ControllerDouble:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def submit(self, request_id):
        self.requests.append(request_id)
        return self.answer


def test_a_press_submits_one_new_request(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="queued", error_code=None))
    app._button_capture(controller)
    app._button_capture(controller)
    assert len(controller.requests) == 2
    assert controller.requests[0] != controller.requests[1]
    assert "button: capture requested" in capsys.readouterr().out


def test_a_press_while_busy_is_ignored_and_logged(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="failed", error_code="busy"))
    app._button_capture(controller)
    assert "button: press ignored (busy)" in capsys.readouterr().out


def test_a_press_after_close_is_ignored(capsys):
    from pifilm.capture import app

    controller = _ControllerDouble(SimpleNamespace(state="failed", error_code="closed"))
    app._button_capture(controller)   # must not raise
    assert "button: press ignored (closed)" in capsys.readouterr().out


def test_main_opens_the_button_attaches_it_and_closes_it(camera_cli, monkeypatch, capsys):
    button = _ButtonDouble()
    opened = []

    def open_button(pin):
        opened.append(pin)
        return button

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    assert opened == [21]
    assert callable(button.callback)
    assert button.closed
    assert "Shutter button on BCM 21." in capsys.readouterr().out


def test_main_warns_and_continues_when_the_button_cannot_open(camera_cli, monkeypatch, capsys):
    from pifilm.capture.button import ButtonError

    def open_button(pin):
        raise ButtonError("cannot claim BCM 21: GPIO busy")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "21"]) == 0
    assert (
        "warning: shutter button unavailable (cannot claim BCM 21: GPIO busy); "
        "continuing without it"
    ) in capsys.readouterr().err


def test_main_without_the_flag_never_opens_a_button(camera_cli, monkeypatch):
    def open_button(pin):
        raise AssertionError("the button must not be opened without --shutter-gpio")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    assert camera_cli.main(["--fake", "--no-preview"]) == 0


@pytest.mark.parametrize("argv", [
    ["--fake", "--no-preview", "--shutter-gpio", "6", "--ups", "x728"],
    ["--fake", "--no-preview", "--shutter-gpio", "17", "--display", "waveshare28"],
    ["--fake", "--no-preview", "--shutter-gpio", "3"],
    ["--fake", "--no-preview", "--shutter-gpio", "28"],
    ["--fake", "--no-preview", "--shutter-gpio", "twenty"],
])
def test_shutter_gpio_refuses_reserved_pins(camera_cli, monkeypatch, argv):
    def open_button(pin):
        raise AssertionError("an argument error must come before any hardware is touched")

    monkeypatch.setattr(camera_cli, "open_shutter_button", open_button)
    monkeypatch.setattr(camera_cli, "build_ups", lambda name: None)
    with pytest.raises(SystemExit) as excinfo:
        camera_cli.main(argv)
    assert excinfo.value.code == 2


def test_a_reserved_pin_is_allowed_when_its_hardware_is_not_selected(camera_cli, monkeypatch):
    monkeypatch.setattr(camera_cli, "open_shutter_button", lambda pin: _ButtonDouble())
    assert camera_cli.main(["--fake", "--no-preview", "--shutter-gpio", "17"]) == 0
```

Read the existing tests around the `camera_cli` fixture first. If `main(["--fake", "--no-preview"])` does not return 0 under that fixture as it stands (for example because another stub is needed), adapt the new tests' set-up the way the neighbouring tests do it, keep every assertion, and note the adaptation in your report.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_app.py -k "press or button or shutter_gpio or reserved_pin"`
Expected: FAIL: `AttributeError` for `app._button_capture` and `app.open_shutter_button`, and `SystemExit: 2` (unrecognised argument) where 0 is expected.

- [ ] **Step 3: Add the imports, the argument type and the press handler**

In `pifilm/capture/app.py`, add beside the other `from .` imports (keep ruff's import order):

```python
from .button import (
    MAX_PIN,
    MIN_PIN,
    WAVESHARE28_PINS,
    X728_PINS,
    ButtonError,
    ShutterButton,
    open_shutter_button,
)
```

If `uuid` and `functools` are not already imported at the top of the file, add `import functools` and `import uuid` to the standard-library imports.

Add these two functions directly above `def _idle_seconds`:

```python
def _gpio_pin(text: str) -> int:
    """An argparse type: a BCM pin number the shutter button may use."""
    try:
        pin = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a BCM pin number") from None
    if not MIN_PIN <= pin <= MAX_PIN:
        raise argparse.ArgumentTypeError(
            f"BCM {pin} is not usable; choose a pin between {MIN_PIN} and {MAX_PIN}"
        )
    return pin


def _button_capture(controller: CaptureController) -> None:
    """One press of the GPIO shutter button: the same request an LCD tap makes.

    Runs on gpiozero's callback thread. ``submit`` only takes the controller's
    lock and queues the job, so the press returns at once; a capture already in
    progress (or a controller that is shutting down) answers with a failed
    snapshot, and the press is dropped, never queued.
    """
    job = controller.submit(str(uuid.uuid4()))
    if job.state == "failed":
        print(f"button: press ignored ({job.error_code})")
    else:
        print("button: capture requested")
```

- [ ] **Step 4: Add the argument and the pin checks**

In `main`, add after the `--display-off-after` argument:

```python
    parser.add_argument(
        "--shutter-gpio", type=_gpio_pin, default=None, metavar="BCM",
        help="take a photo when a push button wired between this BCM pin and ground is "
             "pressed (for example 21: physical pin 40, with ground on pin 39)",
    )
```

Directly after `args = parser.parse_args(argv)` (before any hardware is opened, including `build_ups`), add:

```python
    if args.shutter_gpio is not None:
        if args.ups == "x728" and args.shutter_gpio in X728_PINS:
            parser.error(f"--shutter-gpio {args.shutter_gpio} is used by the X728 UPS")
        if args.display == "waveshare28" and args.shutter_gpio in WAVESHARE28_PINS:
            parser.error(f"--shutter-gpio {args.shutter_gpio} is used by the 2.8 inch LCD")
```

If `args = parser.parse_args(argv)` is followed by other argument validation, put this block with it; the requirement is that it runs before `build_ups`, `_open_display` and the camera.

- [ ] **Step 5: Attach the button to the controller in `main`**

The block to change starts at `controller: CaptureController | None = None` and ends at the `except KeyboardInterrupt: return 0` that closes the `if args.remote_listen:` branch. Today the shared-controller loops run only inside `if args.remote_listen:`. Restructure it so they run when the remote API **or** the button is in use. The loops themselves do not change; only their guard and indentation do.

Replace that block with:

```python
    controller: CaptureController | None = None
    remote: RemoteCaptureServer | None = None
    button: ShutterButton | None = None
    if display_pair is not None:
        controller = CaptureController(session)
    try:
        if power is not None:
            power.start()
            print(f"Reading the {args.ups} UPS every 10 s.")
        if args.shutter_gpio is not None:
            try:
                button = open_shutter_button(args.shutter_gpio)
            except ButtonError as exc:
                print(
                    f"warning: shutter button unavailable ({exc}); continuing without it",
                    file=sys.stderr,
                )
        # The remote API and the button are both triggers that arrive on another
        # thread, so either one needs the shared controller and the loops that
        # tolerate it. A button that failed to open leaves the mode as it was.
        shared = bool(args.remote_listen) or button is not None
        if shared:
            controller = controller or CaptureController(session)
        if button is not None:
            assert controller is not None
            button.on_press(functools.partial(_button_capture, controller))
            print(f"Shutter button on BCM {args.shutter_gpio}.")
        if args.remote_listen:
            assert controller is not None
            try:
                remote = RemoteCaptureServer(
                    controller, remote_token, args.remote_listen,
                    power=power.snapshot if power else None,
                )
                remote.start()
            except OSError as exc:
                print(f"error: could not start remote listener: {exc}", file=sys.stderr)
                return 2
            host, port = args.remote_listen
            print(f"Remote capture API listening on {host}:{port}.")
        if shared:
            try:
                if display_pair is not None:
                    assert controller is not None
                    return _run_viewfinder(camera, controller, display_pair, power, args)
                if has_display() and (args.show_captures or not args.no_preview):
                    if sys.stdin.isatty():
                        with TerminalKeys() as keys:
                            displayed = run_preview_loop(
                                session, CAPTURE_WINDOW_NAME, captures_only=True,
                                read_key=lambda: keys.read(timeout=0), controller=controller,
                            )
                    else:
                        displayed = run_preview_loop(
                            session, CAPTURE_WINDOW_NAME, captures_only=True, controller=controller,
                        )
                    if displayed:
                        return 0
                    print("Capture display unavailable; remote API remains active.")
                if sys.stdin.isatty():
                    with TerminalKeys() as keys:
                        run_headless_loop(session, keys.read, controller=controller)
                    return 0
                return _serve_until_interrupt()
            except KeyboardInterrupt:
                return 0
```

Compare it line by line with the code it replaces: the body of the loops, the remote server's construction and every message must be identical to what is there now. If the current code differs from what this block assumes (a renamed variable, an extra argument), keep the current code's version of that line and apply only the structural change: the `button` open, the `shared` flag, the `on_press` attach, and the loops moving from under `if args.remote_listen:` to under `if shared:`.

In the `finally:` block at the end of `main`, add as its first statement:

```python
        if button is not None:
            button.close()
```

so the pin is released, and no further press can arrive, before the controller is closed.

- [ ] **Step 6: Update the service unit example**

In `deploy/pifilm-capture.service.example`, add these comment lines directly above the `ExecStart=` line:

```ini
# --shutter-gpio 21 takes a photo when a push button wired between BCM 21 (physical
# pin 40) and ground (pin 39) is pressed. With no button wired the pin just idles
# high, so the flag is harmless; if the pin cannot be claimed the service logs one
# warning line and carries on.
```

and append ` --shutter-gpio 21` to the end of the `ExecStart=` line.

- [ ] **Step 7: Run the whole suite and lint**

Run: `.venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: all pass (every pre-existing test unmodified), no lint errors.

- [ ] **Step 8: Commit**

```bash
git add pifilm/capture/app.py deploy/pifilm-capture.service.example tests/test_app.py
git commit -m "pifilm-capture: --shutter-gpio, a physical shutter button

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Documentation

**Files:**
- Modify: `docs/setup.md`, `docs/x728-ups.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: the flag, messages and pin sets from Tasks 1 and 2. Check each statement below against the committed code before writing it; where they differ, the code wins and the difference goes in the report.

- [ ] **Step 1: `docs/setup.md`: a new optional section**

Add a section after the LCD viewfinder section (4.8), numbered to follow it (`### 4.9 Optional: shutter button`; if 4.9 exists, renumber the later optional sections and fix any links to them):

````markdown
### 4.9 Optional: shutter button

A push button on the GPIO header takes a photo, like the on-screen shutter or
the Stick. Any momentary (normally open) button works; it needs two wires and no
resistor, because the Pi's internal pull-up holds the pin high until the button
connects it to ground.

**Wiring**, with the Pi powered off:

| Button lead | Header pin | |
| --- | --- | --- |
| One lead | Physical pin 40 (BCM 21) | The last pin of the outer row |
| The other lead | Physical pin 39 (ground) | The last pin of the inner row, directly opposite |

Which lead goes where does not matter. These two pins are free on this build:
the X728 UPS uses BCM 5, 6, 12, 16, 20 and 26, and the DSI panel uses no header
pins. Do not connect the button to a 3.3 V or 5 V pin.

**Service.** The example unit already passes `--shutter-gpio 21`. On an existing
install, add it to the `ExecStart=` line of
`/etc/systemd/system/pifilm-capture.service`, then
`sudo systemctl daemon-reload && sudo systemctl restart pifilm-capture`. The
journal shows `Shutter button on BCM 21.` at start-up, and `button: capture
requested` for each press.

**Behaviour.** One press takes one photo. A press while a photo is still being
processed is ignored (`button: press ignored (busy)`), never queued. A press
with the screen dark from idle takes the photo and wakes the screen into the
review. To use another pin, change the number (BCM 4 to 27); pins that belong
to the X728 or to the 2.8" LCD are refused when that hardware is selected.

**If it does not work.** `warning: shutter button unavailable (...)` in the
journal names the cause: `python3-gpiozero` missing, the service user not in the
`gpio` group, or the pin claimed by something else. To test the wiring without
the app, stop the service and run
`python3 -c "from gpiozero import Button; b = Button(21); b.wait_for_press(); print('pressed')"`,
then press the button.
````

- [ ] **Step 2: `docs/x728-ups.md`**

Directly after the pin table in section 3 (the table whose header is `| BCM pin | Direction | Meaning | Used by |`), add:

```markdown
The optional shutter button uses BCM 21 (physical pin 40), which is not in this
table; `pifilm-capture` refuses `--shutter-gpio` on any pin that is, when
`--ups x728` is given. See [setup](setup.md), "Optional: shutter button".
```

- [ ] **Step 3: `README.md` (two lines only; the rest is the owner's wording)**

- In the parts list, add one line after the Waveshare LCD line: `- A momentary push button on two jumper wires (physical shutter button, GPIO 21)`
- In the flag table, add a row after the `--display` row: `` | `--shutter-gpio 21` | Physical shutter button between BCM 21 and ground ([setup](docs/setup.md)) | ``

- [ ] **Step 4: `CLAUDE.md`**

In the "Capture" list, add a bullet after the `controller.py` bullet:

```markdown
- `button.py`: optional physical shutter button (`--shutter-gpio BCM`, documented on BCM 21 with
  ground on physical pin 39). gpiozero `Button` with the internal pull-up, lazy import; a press
  calls `CaptureController.submit` like an LCD tap, so a press during a capture is dropped as
  `busy`. Pins of the X728 and of the 2.8" LCD are refused when that hardware is selected; a pin
  that cannot be claimed is one warning line, not a failure to start.
```

In the `controller.py` bullet, change `Shared by the local SPACE key and the remote API.` to `Shared by the local SPACE key, the LCD, the GPIO shutter button and the remote API.`

- [ ] **Step 5: Check and commit**

Run: `grep -n "shutter-gpio" README.md CLAUDE.md docs/setup.md docs/x728-ups.md deploy/pifilm-capture.service.example`
Expected: at least one hit in each file.

```bash
git add docs/setup.md docs/x728-ups.md README.md CLAUDE.md
git commit -m "docs: the GPIO shutter button

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Hardware acceptance (on the Pi, with the user)

**Files:**
- Modify: `docs/setup.md` (a dated "verified" line in the shutter button section)

- [ ] **Step 1: Push and ask the user to wire and deploy**

```bash
git push -u origin feat/gpio-shutter-button
```

Ask the user to shut the Pi down, connect the button's two leads to physical pins 39 and 40, power up, and run:

```bash
cd /home/george/repos/pi-film-reversal
git fetch origin && git checkout feat/gpio-shutter-button && git pull
sudo sed -i '/^ExecStart=/ s/$/ --shutter-gpio 21/' /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload && sudo systemctl restart pifilm-capture
journalctl -u pifilm-capture -f
```

- [ ] **Step 2: Walk the checks with the user**

1. The journal shows `Shutter button on BCM 21.` at start-up.
2. One press: colour bars, then the review on the LCD; the journal shows `button: capture requested`; the day folder gains one photo.
3. Press again while the bars are showing: `button: press ignored (busy)` and no extra photo.
4. With double exposure on (`2x`), two presses make one composite.
5. Leave the screen to go dark, then press: the photo is taken and the screen wakes into the review.
6. A Stick shot and an LCD tap still work.
7. Ten presses in a row, waiting for each to finish: exactly ten photos (no double fires from bounce, no missed presses).
8. Leave the camera idle for a few minutes without touching the button: no photo is taken by itself (no false triggers from the wires).

- [ ] **Step 3: Handle a failed check**

| Failure | Action |
| --- | --- |
| No start-up line, a `warning: shutter button unavailable` line instead | Read the message; the usual causes are the `gpio` group or a leftover overlay claiming the pin |
| A press does nothing and nothing is logged | Wiring: run the `gpiozero` one-liner from the setup section with the service stopped |
| Photos taken with no press (check 8) | Report it: the fix is a longer debounce or a minimum hold time in `button.py`, planned separately |
| Occasional missed presses (check 7) | Report how many of ten; the first suspect is the 50 ms debounce against this particular switch |

- [ ] **Step 4: Record and open the pull request**

Add a line at the end of the shutter button section in `docs/setup.md`: `Verified on the Pi 4 on <date>: <one sentence with the outcome of the eight checks>.` Commit, push, then:

```bash
gh pr create --repo maker-lab-9000/pi-film-reversal --head feat/gpio-shutter-button --base main --title "GPIO shutter button" --body "$(cat <<'EOF'
Adds `--shutter-gpio BCM`: a push button between a GPIO pin and ground takes a photo through the shared capture controller, like an LCD tap or a Stick press.

- `pifilm/capture/button.py`: gpiozero `Button`, internal pull-up, 50 ms debounce, lazy import.
- `pifilm/capture/app.py`: the flag, reserved-pin checks, and the shared-controller loops now also run for the button.
- Example unit and docs: BCM 21 (physical pin 40) with ground on pin 39.

Plan: docs/superpowers/plans/2026-10-10-gpio-shutter-button.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
