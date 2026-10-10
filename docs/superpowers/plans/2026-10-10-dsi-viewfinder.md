# DSI Viewfinder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing LCD viewfinder full screen on the Waveshare 3.5" DSI LCD (E) through `--display waveshare35dsi`, keeping the 2.8" SPI panel as a fallback.

**Architecture:** Two new hardware modules sit behind the duck-typed interfaces the viewfinder loop already uses: `kms.py` (DRM/KMS through `pykms`, two buffers swapped per frame) for `show`/`backlight`/`close`, and `evtouch.py` (kernel input events from the Goodix controller) for `read`/`close`. `ui.py` gains a `scale` factor so the 320×240 layout draws natively at 640×480. Each hardware module is split into pure logic, unit-tested on the Mac, and a thin device layer proven on the Pi.

**Tech Stack:** Python 3.11+, NumPy, Pillow, `pykms` (`python3-kms++` from apt, Pi only, lazy import), Linux evdev read with the standard library (`os`, `fcntl`, `struct`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-10-dsi-viewfinder-design.md`

## Global Constraints

- Always use `.venv/bin/...`. Tests: `.venv/bin/pytest -q -m 'not slow'`. Lint: `.venv/bin/ruff check .` (rules E, F, I, B, UP; line length 100).
- Nothing under `pifilm/display/` may import OpenCV, and hardware libraries (`pykms`) are imported only inside `open_*` factories so the package imports on a Mac.
- No new pip dependency. `pykms` comes from apt (`python3-kms++`); touch uses the standard library only.
- `--display waveshare28`, `st7789.py` and `cst3530.py` keep working; at `scale=1` every `ui.py` render is pixel-identical to today.
- New flag value is exactly `waveshare35dsi`. DRM connector name is exactly `DSI-1`. Touch device name is exactly `Goodix Capacitive TouchScreen`.
- A display or touch failure at open is one `warning: display unavailable (<message>); continuing without it` line and a headless service, never an exit.
- Module docstrings carry the design rationale; write them for each new module.
- Never put the token, Wi-Fi password or SSH password in a command, file or commit. Do not read or echo `.env` values.
- Work on branch `feat/dsi-viewfinder`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Tasks 1 and 9 run on the Pi and need the user. Tasks 2 to 5 and 7 to 8 do not depend on Task 1's result; Task 6 does.

## Review Focus

1. **Touch events split across two reads.** A press whose tracking id arrives in one read and whose coordinates arrive in the next must not produce a tap at the previous touch's position. Pinned in Task 3 (`test_state_is_committed_only_on_syn_report`).
2. **A second finger on the glass.** A palm or second finger (slot 1) must not move the first finger's position or fire a control. Pinned in Task 3 (`test_other_slots_are_ignored`).
3. **Taps at the glass edge.** A raw coordinate at the device's maximum, or outside its range, must map inside 0..639 × 0..479. Pinned in Task 3 (`test_coordinates_scale_and_clamp_to_the_screen`).
4. **A 16:9 sensor.** The README's IMX708 gives a 640×360 preview; at `scale=2` it must be letterboxed in 640×480, not stretched. Pinned in Task 4 (`test_render_live_letterboxes_a_16_9_frame_at_scale_2`).
5. **The display is busy or half-configured.** A running desktop session, a missing overlay or an unexpected buffer layout makes `pykms` raise arbitrary exception types; every one must become `DisplayError`. Pinned in Task 6 (`test_open_maps_any_backend_failure_to_display_error`).

## File Structure

| File | Responsibility |
| --- | --- |
| `pifilm/display/touch.py` (new) | `TouchPoint`, `Tap`, `TapDetector`: panel-independent touch types |
| `pifilm/display/cst3530.py` (modify) | 2.8" touch driver; imports the shared types and re-exports them |
| `pifilm/display/evtouch.py` (new) | Goodix touch through kernel input events |
| `pifilm/display/kms.py` (new) | DSI display through `pykms` |
| `pifilm/display/ui.py` (modify) | `scale` parameter on every render and on `hit` |
| `pifilm/display/viewfinder.py` (modify) | Derives `scale` from the display; honours `dimmable` |
| `pifilm/display/__init__.py` (modify) | Docstring names both panels |
| `pifilm/capture/app.py` (modify) | `--display waveshare35dsi` wiring |
| `deploy/pifilm-capture.service.example` (modify) | Uses the new panel |
| `tests/test_display_touch.py`, `tests/test_display_evtouch.py`, `tests/test_display_kms.py` (new) | Unit tests |
| `tests/test_display_ui.py`, `tests/test_viewfinder.py`, `tests/test_app.py` (modify) | Scale, dimmable and wiring tests |
| `docs/lcd-viewfinder.md`, `docs/setup.md`, `README.md`, `CLAUDE.md` (modify) | Documentation |

---

### Task 1: Hardware spike: per-frame swap and screen-off (on the Pi, with the user)

Proves the three `pykms` calls that `kms.py` depends on and that have not been run on this panel. It writes no repository code. Its result selects between the primary code and the named alternatives in Task 6.

**Files:** none.

**Interfaces:**
- Produces: three facts for Task 6: `SWAP` is `atomic` or `setmode`; `POWER` is `dpms`, `active` or `black`; `STRIDE` is the printed number. Done: see "Spike result" in Task 6.

- [ ] **Step 1: Create the branch and commit the spec and this plan**

```bash
git checkout -b feat/dsi-viewfinder
git add docs/superpowers/specs/2026-10-10-dsi-viewfinder-design.md docs/superpowers/plans/2026-10-10-dsi-viewfinder.md
git commit -m "docs: DSI viewfinder spec and plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [x] **Step 2: Ask the user to run the spike on the Pi**

Give the user this block to paste into an SSH session as `george` (no `sudo`, no leading spaces). It shows a grey screen with a white stripe moving right for about 10 seconds, then turns the screen off for 3 seconds and back on.

```bash
python3 - <<'EOF'
import time
import numpy as np
import pykms
card = pykms.Card()
res = pykms.ResourceManager(card)
conn = res.reserve_connector("DSI-1")
crtc = res.reserve_crtc(conn)
mode = conn.get_default_mode()
w, h = mode.hdisplay, mode.vdisplay
fbs = [pykms.DumbFramebuffer(card, w, h, "XR24") for _ in range(2)]
print("STRIDE", fbs[0].stride(0), "expected", w * 4)
maps = [np.frombuffer(fb.map(0), dtype=np.uint8)[: h * w * 4].reshape(h, w, 4) for fb in fbs]
print("writable", maps[0].flags.writeable)
crtc.set_mode(conn, fbs[0], mode)
plane = crtc.primary_plane
print("primary plane", plane.id)
n, worst = 0, 0.0
for i in range(100):
    back = i % 2
    maps[back][:] = 60
    x = (i * 6) % (w - 20)
    maps[back][:, x:x + 20] = 255
    start = time.monotonic()
    req = pykms.AtomicReq(card)
    req.add(plane, "FB_ID", fbs[back].id)
    result = req.commit_sync()
    worst = max(worst, time.monotonic() - start)
    if result:
        print("commit_sync returned", result)
        break
    n += 1
    time.sleep(0.1)
print(f"SWAP atomic flips={n} worst_ms={worst * 1000:.1f}")
print("POWER dpms: screen should go dark for 3 s")
conn.set_prop("DPMS", 3)
time.sleep(3)
conn.set_prop("DPMS", 0)
print("POWER dpms: screen should be back")
time.sleep(2)
EOF
```

Ask the user for the full output and three observations: did the stripe move smoothly without tearing, did the screen go dark, did it come back.

- [x] **Step 3: Record the result**

Decide and write the three facts into the Task 6 section of this plan file (replace the line `Spike result: pending`):

| Observation | Fact |
| --- | --- |
| `SWAP atomic flips=100`, stripe smooth | `SWAP = atomic` |
| Exception or non-zero `commit_sync` at the swap | `SWAP = setmode`; re-run with the three `req` lines replaced by `crtc.set_mode(conn, fbs[back], mode)` to confirm it shows the stripe |
| Screen went dark and came back | `POWER = dpms` |
| `set_prop` raised, or screen stayed lit | Re-run the last block with `req = pykms.AtomicReq(card); req.add(crtc, "ACTIVE", 0); req.commit_sync(allow_modeset=True)` and `"ACTIVE", 1` to restore. Dark and back: `POWER = active`. Otherwise `POWER = black` |
| `STRIDE` differs from `expected` | Stop and report: `kms.py`'s buffer view assumes they are equal |

- [ ] **Step 4: Commit the recorded result**

```bash
git add docs/superpowers/plans/2026-10-10-dsi-viewfinder.md
git commit -m "docs: record DSI spike result

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Shared touch types

**Files:**
- Create: `pifilm/display/touch.py`
- Modify: `pifilm/display/cst3530.py` (remove the three definitions, import them)
- Modify: `pifilm/display/viewfinder.py:50`
- Test: `tests/test_display_touch.py`

**Interfaces:**
- Produces: `pifilm.display.touch.TouchPoint(x: int, y: int, strength: int)`, `Tap(x: int, y: int)`, `TapDetector(max_hold: float = 0.6, max_move: float = 20.0)` with `feed(points: list[TouchPoint], now: float) -> Tap | None`. The same names stay importable from `pifilm.display.cst3530`.

- [ ] **Step 1: Write the failing test**

`tests/test_display_touch.py`:

```python
from pifilm.display import cst3530, touch
from pifilm.display.touch import Tap, TapDetector, TouchPoint


def test_a_quick_press_and_release_is_a_tap_at_the_press_position():
    detector = TapDetector()
    assert detector.feed([TouchPoint(100, 50, 0)], 0.0) is None
    assert detector.feed([TouchPoint(104, 52, 0)], 0.1) is None
    assert detector.feed([], 0.2) == Tap(100, 50)


def test_a_hold_or_a_drag_is_not_a_tap():
    detector = TapDetector()
    detector.feed([TouchPoint(100, 50, 0)], 0.0)
    assert detector.feed([], 1.0) is None
    detector.feed([TouchPoint(100, 50, 0)], 2.0)
    detector.feed([TouchPoint(160, 50, 0)], 2.1)
    assert detector.feed([], 2.2) is None


def test_max_move_is_configurable_for_a_denser_panel():
    detector = TapDetector(max_move=40.0)
    detector.feed([TouchPoint(100, 50, 0)], 0.0)
    detector.feed([TouchPoint(130, 50, 0)], 0.1)
    assert detector.feed([], 0.2) == Tap(100, 50)


def test_the_2_8_inch_driver_still_exports_the_shared_types():
    assert cst3530.TouchPoint is touch.TouchPoint
    assert cst3530.Tap is touch.Tap
    assert cst3530.TapDetector is touch.TapDetector
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest -q tests/test_display_touch.py`
Expected: FAIL with `ImportError: cannot import name 'touch'`.

- [ ] **Step 3: Create `pifilm/display/touch.py`**

```python
"""Touch types shared by every panel: a point, a tap, and the tap detector.

They lived in ``cst3530.py`` while the 2.8" panel was the only one. The DSI panel
reads touch from kernel input events instead of I2C, and the viewfinder loop only
ever needs these three names, so they sit apart from either driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    strength: int


@dataclass(frozen=True)
class Tap:
    x: int
    y: int


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
```

- [ ] **Step 4: Point `cst3530.py` and `viewfinder.py` at it**

In `pifilm/display/cst3530.py`: delete the `TouchPoint` dataclass, the `Tap` dataclass and the whole `TapDetector` class; delete `import math` (it is used only by `TapDetector`); keep `RawPoint`. Replace the line `from . import DisplayError` with:

```python
from . import DisplayError
from .touch import Tap, TapDetector, TouchPoint

__all__ = ["CST3530Touch", "RawPoint", "Tap", "TapDetector", "TouchPoint",
           "decode_points", "open_waveshare28_touch", "to_display"]
```

In `pifilm/display/viewfinder.py` replace `from .cst3530 import Tap, TapDetector` with `from .touch import Tap, TapDetector`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest -q tests/test_display_touch.py tests/test_display_cst3530.py tests/test_viewfinder.py && .venv/bin/ruff check pifilm/display tests/test_display_touch.py`
Expected: all pass, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pifilm/display/touch.py pifilm/display/cst3530.py pifilm/display/viewfinder.py tests/test_display_touch.py
git commit -m "display: move the shared touch types out of the CST3530 driver

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Goodix touch through kernel input events

**Files:**
- Create: `pifilm/display/evtouch.py`
- Test: `tests/test_display_evtouch.py`

**Interfaces:**
- Consumes: `pifilm.display.touch.TouchPoint`, `pifilm.display.DisplayError`.
- Produces: `open_goodix_touch(size: tuple[int, int], rotate: int = 0, *, input_dir: Path = Path("/dev/input")) -> EventTouch`; `EventTouch.read() -> list[TouchPoint]`; `EventTouch.close() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_display_evtouch.py`:

```python
import struct

import pytest

from pifilm.display import DisplayError
from pifilm.display.evtouch import (
    ABS_MT_POSITION_X,
    ABS_MT_POSITION_Y,
    ABS_MT_SLOT,
    ABS_MT_TRACKING_ID,
    EV_ABS,
    EV_SYN,
    EventTouch,
    SlotTracker,
    decode_events,
    open_goodix_touch,
)
from pifilm.display.touch import TapDetector, TouchPoint

SYN = (EV_SYN, 0, 0)


def _pack(*events):
    return b"".join(struct.pack("<qqHHi", 0, 0, *event) for event in events)


def _press(x, y, slot=0, tracking_id=7):
    return [(EV_ABS, ABS_MT_SLOT, slot), (EV_ABS, ABS_MT_TRACKING_ID, tracking_id),
            (EV_ABS, ABS_MT_POSITION_X, x), (EV_ABS, ABS_MT_POSITION_Y, y), SYN]


def _release(slot=0):
    return [(EV_ABS, ABS_MT_SLOT, slot), (EV_ABS, ABS_MT_TRACKING_ID, -1), SYN]


class Feed:
    """A scripted device: one chunk per ``EventTouch.read``.

    ``read`` keeps calling until it gets nothing, so each non-empty chunk is
    followed by one empty answer. An empty chunk stands for a poll with no events.
    """

    def __init__(self, *chunks):
        self.chunks = list(chunks)
        self._drained = False

    def __call__(self):
        if self._drained:
            self._drained = False
            return b""
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        self._drained = bool(chunk)
        return chunk


def _touch(feed, rotate=0):
    return EventTouch(feed, (0, 639), (0, 479), (640, 480), rotate=rotate)


def test_decode_events_reads_whole_records_and_drops_a_partial_tail():
    data = _pack((EV_ABS, ABS_MT_POSITION_X, 300), SYN) + b"\x01\x02\x03"
    assert decode_events(data) == [(EV_ABS, ABS_MT_POSITION_X, 300), SYN]


def test_finger_down_reports_one_point_and_release_reports_none():
    feed = Feed(_pack(*_press(320, 240)), b"", _pack(*_release()))
    touch = _touch(feed)
    assert touch.read() == [TouchPoint(320, 240, 0)]
    assert touch.read() == [TouchPoint(320, 240, 0)]   # still held, nothing new
    assert touch.read() == []


def test_a_press_and_release_inside_one_read_is_still_seen_once():
    touch = _touch(Feed(_pack(*_press(100, 80), *_release())))
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == []


def test_a_fast_tap_reaches_the_tap_detector():
    touch = _touch(Feed(_pack(*_press(100, 80), *_release())))
    detector = TapDetector()
    assert detector.feed(touch.read(), 0.00) is None
    tap = detector.feed(touch.read(), 0.02)
    assert (tap.x, tap.y) == (100, 80)


def test_state_is_committed_only_on_syn_report():
    """A read that ends mid-report must not expose the previous touch's position."""
    first = _pack(*_press(600, 400), *_release())
    half = _pack((EV_ABS, ABS_MT_SLOT, 0), (EV_ABS, ABS_MT_TRACKING_ID, 8))
    rest = _pack((EV_ABS, ABS_MT_POSITION_X, 50), (EV_ABS, ABS_MT_POSITION_Y, 60), SYN)
    touch = _touch(Feed(first, half, rest))
    assert touch.read() == [TouchPoint(600, 400, 0)]
    assert touch.read() == []                           # press not yet complete
    assert touch.read() == [TouchPoint(50, 60, 0)]      # never (600, 400)


def test_other_slots_are_ignored():
    touch = _touch(Feed(_pack(*_press(100, 80)), _pack(*_press(500, 300, slot=1, tracking_id=9)),
                        _pack(*_release(slot=1))))
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == [TouchPoint(100, 80, 0)]     # slot 1 lifting is not slot 0 lifting


def test_coordinates_scale_and_clamp_to_the_screen():
    feed = Feed(_pack(*_press(1023, 767)), _pack(*_release()), _pack(*_press(5000, -40)))
    touch = EventTouch(feed, (0, 1023), (0, 767), (640, 480))
    assert touch.read() == [TouchPoint(639, 479, 0)]
    assert touch.read() == []
    assert touch.read() == [TouchPoint(639, 0, 0)]


def test_rotate_180_mirrors_both_axes():
    touch = _touch(Feed(_pack(*_press(0, 0))), rotate=180)
    assert touch.read() == [TouchPoint(639, 479, 0)]


def test_a_read_error_is_a_display_error():
    def broken():
        raise OSError(19, "No such device")

    with pytest.raises(DisplayError, match="touch read failed"):
        _touch(broken).read()


def test_bad_rotation_and_degenerate_range_are_rejected():
    with pytest.raises(DisplayError, match="rotate"):
        _touch(Feed(), rotate=90)
    with pytest.raises(DisplayError, match="range"):
        EventTouch(Feed(), (0, 0), (0, 479), (640, 480))


def test_slot_tracker_take_clears_the_pressed_latch():
    tracker = SlotTracker()
    tracker.feed(_press(10, 20) + _release())
    assert tracker.take() == (10, 20)
    assert tracker.take() is None


def test_open_reports_a_missing_device(tmp_path):
    with pytest.raises(DisplayError, match="Goodix Capacitive TouchScreen"):
        open_goodix_touch((640, 480), input_dir=tmp_path)


def test_close_calls_the_closer_once():
    closed = []
    touch = EventTouch(Feed(), (0, 639), (0, 479), (640, 480), closer=lambda: closed.append(1))
    touch.close()
    touch.close()
    assert closed == [1]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_evtouch.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pifilm.display.evtouch'`.

- [ ] **Step 3: Create `pifilm/display/evtouch.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q tests/test_display_evtouch.py && .venv/bin/ruff check pifilm/display/evtouch.py tests/test_display_evtouch.py`
Expected: 13 passed, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/evtouch.py tests/test_display_evtouch.py
git commit -m "display: Goodix touch through kernel input events

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: UI scale factor

**Files:**
- Modify: `pifilm/display/ui.py`
- Test: `tests/test_display_ui.py` (append)

**Interfaces:**
- Produces: `render_live(frame_rgb, reading, double=None, shutter_buttons=False, *, scale: int = 1)`, `render_review(graded_rgb, caption, *, scale: int = 1)`, `render_processing(label="Processing photo...", *, scale: int = 1)`, `render_message(title, detail, *, scale: int = 1)`, each returning a `(320*scale, 240*scale)` RGB image; `hit(x: int, y: int, scale: int = 1) -> Action`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_display_ui.py`:

```python
# -- scale ----------------------------------------------------------------------


def test_every_screen_renders_at_640x480_at_scale_2():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    images = [
        render_live(frame, _reading(focus=0.5, focus_peak=0.7), (True, 1),
                    shutter_buttons=True, scale=2),
        render_review(frame, "1/250  ISO 100  EV 0", scale=2),
        render_processing(scale=2),
        render_message("Capture failed", "camera busy", scale=2),
    ]
    for img in images:
        assert img.size == (640, 480) and img.mode == "RGB"


def test_scale_1_is_the_default_and_unchanged():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert render_live(frame, _reading(), scale=1).tobytes() == render_live(
        frame, _reading()).tobytes()


def test_render_live_letterboxes_a_16_9_frame_at_scale_2():
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(battery_percent=None), scale=2)
    assert img.getpixel((320, 30)) == (0, 0, 0)          # band above the picture
    assert img.getpixel((320, 240)) == (200, 200, 200)   # picture, not stretched


def test_the_bar_is_in_the_same_place_at_scale_2():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = render_live(frame, _reading(), scale=2)
    above = img.getpixel((320, 2 * (BAR_TOP - 10)))
    inside = img.getpixel((320, 2 * (BAR_TOP + 4)))
    assert sum(inside) < sum(above)


def test_processing_bars_leave_no_gaps_at_scale_2():
    img = render_processing(scale=2)
    boundary = 2 * (WIDTH // 7)          # first column of the second bar
    for x in (boundary - 1, boundary):
        assert img.getpixel((x, 100)) != (0, 0, 0)
    assert img.getpixel((639, 100)) != (0, 0, 0)


def test_hit_regions_scale_with_the_screen():
    cx, cy = SHUTTER_CENTRE
    points = [
        (cx, cy), (10, BAR_TOP + 10), (310, BAR_TOP + 10), (160, 100),
        ((DOUBLE_BOX[0] + DOUBLE_BOX[2]) // 2, (DOUBLE_BOX[1] + DOUBLE_BOX[3]) // 2),
        ((SHUTTER_PLUS_BOX[0] + SHUTTER_PLUS_BOX[2]) // 2, SHUTTER_PLUS_BOX[1] + 5),
        ((SHUTTER_MINUS_BOX[0] + SHUTTER_MINUS_BOX[2]) // 2, SHUTTER_MINUS_BOX[1] + 5),
    ]
    for x, y in points:
        assert hit(2 * x, 2 * y, scale=2) == hit(x, y)
        assert hit(2 * x + 1, 2 * y + 1, scale=2) == hit(x, y)
    assert hit(639, 479, scale=2) == Action.EV_PLUS
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_ui.py -k "scale"`
Expected: FAIL with `TypeError: render_live() got an unexpected keyword argument 'scale'`.

- [ ] **Step 3: Add the scaled drawing wrapper to `ui.py`**

Insert after the `_font` function:

```python
class _ScaledDraw:
    """An ``ImageDraw`` that takes coordinates in 320x240 base units.

    The layout constants in this module are in base units so the 2.8" panel
    (scale 1) and the 3.5" DSI panel (scale 2) share one layout and one set of hit
    regions. Shapes with two corners (rectangles, ellipses) treat each base pixel
    as a ``scale`` x ``scale`` block, so the far corner maps to that block's last
    pixel and neighbouring shapes stay flush; at scale 1 that is the identity.
    Points (lines, polygons, text anchors) map to ``value * scale``. Widths,
    radii and font sizes are multiplied; ``textlength`` is returned in base units
    so callers keep doing their arithmetic there.
    """

    def __init__(self, draw: ImageDraw.ImageDraw, scale: int) -> None:
        self._draw, self._s = draw, scale

    def font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return _font(size * self._s)

    def _box(self, box: Any) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        s = self._s
        return (x0 * s, y0 * s, (x1 + 1) * s - 1, (y1 + 1) * s - 1)

    def _points(self, points: Any) -> list[tuple[float, float]]:
        flat = list(points)
        if flat and not isinstance(flat[0], (tuple, list)):
            flat = list(zip(flat[0::2], flat[1::2], strict=True))
        return [(x * self._s, y * self._s) for x, y in flat]

    def _kw(self, kw: dict[str, Any]) -> dict[str, Any]:
        # Pillow's outline width defaults to 1 for these shapes; scale that default
        # too, so an outline stays one base pixel thick.
        kw["width"] = kw.get("width", 1) * self._s
        return kw

    def rectangle(self, box: Any, **kw: Any) -> None:
        self._draw.rectangle(self._box(box), **self._kw(kw))

    def rounded_rectangle(self, box: Any, radius: int = 0, **kw: Any) -> None:
        self._draw.rounded_rectangle(self._box(box), radius=radius * self._s, **self._kw(kw))

    def ellipse(self, box: Any, **kw: Any) -> None:
        self._draw.ellipse(self._box(box), **self._kw(kw))

    def line(self, points: Any, **kw: Any) -> None:
        # Pillow's line width defaults to 0 (a hairline); keep that at scale 1 so
        # the 2.8" panel's output does not change by a pixel.
        width = kw.pop("width", 0)
        scaled = width * self._s if width else (self._s if self._s > 1 else 0)
        self._draw.line(self._points(points), width=scaled, **kw)

    def polygon(self, points: Any, **kw: Any) -> None:
        self._draw.polygon(self._points(points), **kw)

    def text(self, xy: Any, text: str, **kw: Any) -> None:
        self._draw.text((xy[0] * self._s, xy[1] * self._s), text, **kw)

    def textlength(self, text: str, font: Any) -> float:
        return self._draw.textlength(text, font=font) / self._s
```

- [ ] **Step 4: Thread `scale` through the render functions and `hit`**

Make exactly these edits in `ui.py`. Nothing else in the function bodies changes.

`render_live`:
- Signature becomes `def render_live(frame_rgb: np.ndarray, reading: MeterReading, double: tuple[bool, int] | None = None, shutter_buttons: bool = False, *, scale: int = 1) -> Image.Image:`
- `base = _letterbox(frame_rgb).convert("RGBA")` → `size = (WIDTH * scale, HEIGHT * scale)` then `base = _letterbox(frame_rgb, size).convert("RGBA")`
- `overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))` → `overlay = Image.new("RGBA", size, (0, 0, 0, 0))`
- `draw = ImageDraw.Draw(overlay)` → `draw = _ScaledDraw(ImageDraw.Draw(overlay), scale)`
- `big = _font(18)` → `big = draw.font(18)`; `font = _font(14)` → `font = draw.font(14)`; `small = _font(11)` → `small = draw.font(11)`; `font = _font(12)` → `font = draw.font(12)`
- `draw.line((NEEDLE_X0, NEEDLE_Y, NEEDLE_X1, NEEDLE_Y), fill=(255, 255, 255, 200), width=1)` stays as written (the wrapper accepts the flat four-number form).
- The tick line `draw.line((x, NEEDLE_Y - ..., x, NEEDLE_Y + ...), fill=(255, 255, 255, 200))` stays as written.

`_draw_focus_bar` and `_draw_double_badge`: change the `draw` parameter's annotation from `ImageDraw.ImageDraw` to `_ScaledDraw`. Bodies unchanged.

`render_processing`:
- Signature becomes `def render_processing(label: str = "Processing photo...", *, scale: int = 1) -> Image.Image:`
- `img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))` → `img = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), (0, 0, 0))`
- `draw = ImageDraw.Draw(img)` → `draw = _ScaledDraw(ImageDraw.Draw(img), scale)`
- `font=_font(14)` → `font=draw.font(14)`

`render_review`:
- Signature becomes `def render_review(graded_rgb: np.ndarray, caption: str, *, scale: int = 1) -> Image.Image:`
- `base = _letterbox(graded_rgb).convert("RGBA")` → `size = (WIDTH * scale, HEIGHT * scale)` then `base = _letterbox(graded_rgb, size).convert("RGBA")`
- `overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))` → `overlay = Image.new("RGBA", size, (0, 0, 0, 0))`
- `draw = ImageDraw.Draw(overlay)` → `draw = _ScaledDraw(ImageDraw.Draw(overlay), scale)`
- Each `_font(12)` → `draw.font(12)` and each `_font(11)` → `draw.font(11)` (three places).

`render_message`:
- Signature becomes `def render_message(title: str, detail: str, *, scale: int = 1) -> Image.Image:`
- `img = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))` → `img = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), (20, 20, 20))`
- `draw = ImageDraw.Draw(img)` → `draw = _ScaledDraw(ImageDraw.Draw(img), scale)`
- `_font(18)` → `draw.font(18)`; `_font(12)` → `draw.font(12)`

`hit`:
- Signature becomes `def hit(x: int, y: int, scale: int = 1) -> Action:`
- First line of the body: `x, y = x // scale, y // scale`

Module docstring, first line: `"""Pillow rendering for the viewfinder and its hit regions, in 320x240 base units. Pure.` and add a closing paragraph: `Every render takes ``scale``: 1 for the 2.8" SPI panel, 2 for the 640x480 DSI panel. The constants below never change with it.`

- [ ] **Step 5: Run the UI tests**

Run: `.venv/bin/pytest -q tests/test_display_ui.py && .venv/bin/ruff check pifilm/display/ui.py tests/test_display_ui.py`
Expected: all pass (the pre-existing tests prove scale 1 is unchanged), no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pifilm/display/ui.py tests/test_display_ui.py
git commit -m "display: draw the viewfinder at any integer scale

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Viewfinder loop: scale and non-dimmable displays

**Files:**
- Modify: `pifilm/display/viewfinder.py`
- Test: `tests/test_viewfinder.py` (append)

**Interfaces:**
- Consumes: Task 4's `scale` keyword on `render_*` and `hit`; `TapDetector(max_move=...)` from Task 2.
- Produces: `ViewfinderLoop` reads two optional display attributes: `width: int` (scale is `width // 320`, default 1) and `dimmable: bool` (`False` disables the dim step only).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viewfinder.py`:

```python
# -- the 640x480 DSI panel --------------------------------------------------------


class BigDisplay(FakeDisplay):
    width, height = 640, 480
    dimmable = False


def test_a_640_wide_display_is_drawn_at_640x480(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    loop.step()
    assert display.images[-1].size == (640, 480)


def test_taps_are_hit_tested_at_the_display_scale(controller, monkeypatch):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    submitted = []
    monkeypatch.setattr(ctl, "submit", lambda rid: submitted.append(rid))
    cx, cy = SHUTTER_CENTRE
    _tap(loop, touch, clock, (2 * cx, 2 * cy))
    assert len(submitted) == 1
    _tap(loop, touch, clock, (cx, cy))        # the 320x240 position is not the button here
    assert len(submitted) == 1


def test_review_and_processing_screens_use_the_display_scale(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl, display=BigDisplay())
    ctl.submit("one")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "REVIEW"
    assert display.images[-1].size == (640, 480)


def test_a_non_dimmable_display_skips_dimming_but_still_turns_off(controller):
    camera, ctl = controller
    loop, touch, display, clock = _idle_loop(camera, ctl, display=BigDisplay())
    loop.step()
    clock.t = 60.0
    loop.step()
    assert display.levels == [80]
    clock.t = 300.0
    loop.step()
    assert display.levels[-1] == 0


def test_a_display_without_a_width_is_drawn_at_320x240(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert display.images[-1].size == (320, 240)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py -k "640 or scale or non_dimmable or without_a_width"`
Expected: the first four FAIL (image size `(320, 240)`, tap not submitted, `levels` contains 40); the last passes.

- [ ] **Step 3: Implement**

In `ViewfinderLoop.__init__`, replace `self._taps = TapDetector()` with:

```python
        # The layout is in 320x240 base units; a 640-wide panel draws it at 2x and
        # reports taps in its own pixels. The tap detector's drag limit is in panel
        # pixels too, so it grows with the scale.
        self._scale = max(1, int(getattr(display, "width", 320)) // 320)
        self._taps = TapDetector(max_move=20.0 * self._scale)
```

In the same method, directly after the existing block

```python
        if not callable(getattr(display, "backlight", None)):
            dim_after = off_after = 0.0
```

add:

```python
        # A panel that can only be on or off (the DSI one has no brightness control)
        # skips the dim step and keeps the off step.
        if getattr(display, "dimmable", True) is False:
            dim_after = 0.0
```

Add `scale=self._scale` to each render call and to `hit`:

- `_review_image`: `render_message("Capture failed", job.error_message or job.error_code or "", scale=self._scale)`, `render_review(rgb, caption, scale=self._scale)`, `render_message("Review unavailable", str(exc)[:60], scale=self._scale)`
- `step`: `render_message("Exposure 1/2", "frame the second exposure", scale=self._scale)`, `hit(tap.x, tap.y, self._scale)`, `render_processing(self._processing_label(snap), scale=self._scale)`, and `render_live(frame.rgb, reading, (snap.double_exposure, snap.exposures_taken), shutter_buttons=self._shutter_ok, scale=self._scale)`

In the module docstring, replace the sentence beginning `The CST3530 is polled, not latched,` with `Touch is polled, not latched, on either panel,` (the rest of the sentence stays).

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py && .venv/bin/ruff check pifilm/display/viewfinder.py tests/test_viewfinder.py`
Expected: all pass. If a pre-existing test spies on `render_processing` or `render_live` with a positional-only stub, it still passes because the spies forward `*args, **kwargs`.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/viewfinder.py tests/test_viewfinder.py
git commit -m "viewfinder: follow the display's scale and skip dimming where there is none

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: KMS display driver

Spike result (run on the Pi 2026-10-10): `STRIDE = 2560` (as expected), `SWAP = atomic` (100 swaps, worst 20.3 ms, no tearing), `POWER = active`. Setting the connector's `DPMS` property printed `commit failed` twice and left the screen lit, without raising; an atomic commit of the CRTC's `ACTIVE` property with `allow_modeset=True` returned 0 both ways, and the panel went dark and came back.

The code below is written for that result. Hardware acceptance (Task 9) can still overturn it; if so, change only the named method of `_PykmsBackend`:

| Symptom at acceptance | Replace the body of | With |
| --- | --- | --- |
| Tearing or swap errors | `flip` | `self._crtc.set_mode(self._conn, self._fbs[index], self._mode)` |
| Screen does not go dark, or does not come back | `power` | `if not on:` fill both buffers with zero (`self._maps[0][:] = 0; self._maps[1][:] = 0`); nothing when `on`. Also add the row "the backlight stays lit when the screen is off" to the docs in Task 8 |

**Files:**
- Create: `pifilm/display/kms.py`
- Modify: `pifilm/display/__init__.py` (docstring)
- Test: `tests/test_display_kms.py`

**Interfaces:**
- Consumes: `pifilm.display.DisplayError`.
- Produces: `open_waveshare35dsi(rotate: int = 0) -> KmsDisplay`; `KmsDisplay.width`, `.height`, `.dimmable = False`, `.show(image: PIL.Image.Image) -> None`, `.backlight(percent: int) -> None`, `.close() -> None`; `pack_xrgb8888(rgb: np.ndarray, rotate: int = 0) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/test_display_kms.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_kms.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pifilm.display.kms'`.

- [ ] **Step 3: Create `pifilm/display/kms.py`**

```python
"""Waveshare 3.5" DSI LCD (E), 640x480, driven through DRM/KMS with ``pykms``.

Unlike the 2.8" SPI panel, this one is a kernel display: the ``waveshare_35DSI``
overlay registers it as connector ``DSI-1`` and the app only has to hand the
kernel finished frames. ``pykms`` (apt ``python3-kms++``) is already on the Pi as
a Picamera2 dependency, and it is imported only inside ``open_waveshare35dsi`` so
this module imports on a Mac.

Frames go into one of two dumb XRGB8888 buffers and the buffers are swapped, so
the panel never shows a half-drawn frame. The alternative, writing into
``/dev/fb0``, was measured to work (16 bpp, 2026-10-10) and is the documented
fallback, but it tears and it shares the screen with the login console. While
this process holds the display the kernel console is kept off the panel, and it
returns when the card is closed.

The service must own the screen, so the Pi boots to the console: under a desktop
session the compositor is the DRM master and opening the output here fails. That
failure, like every other at open, is a ``DisplayError`` and costs the screen
only; ``pifilm-capture`` carries on headless for the Stick.

The panel has no brightness control (``/sys/class/backlight`` is empty), so
``backlight`` is on/off: 0 powers the output down and anything else powers it up.
``dimmable = False`` tells the viewfinder loop to skip its half-brightness step.

``KmsDisplay`` holds the logic and is tested against a fake backend;
``_PykmsBackend`` is the only code that touches ``pykms`` and is proven on the Pi
(see the acceptance checklist in docs/lcd-viewfinder.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from . import DisplayError

CONNECTOR = "DSI-1"
PIXEL_FORMAT = "XR24"  # XRGB8888: bytes in memory are B, G, R, X


def pack_xrgb8888(rgb: np.ndarray, rotate: int = 0) -> np.ndarray:
    """An (H, W, 3) RGB array as (H, W, 4) B, G, R, X bytes, optionally turned 180."""
    if rotate == 180:
        rgb = rgb[::-1, ::-1]
    out = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., 0] = rgb[..., 2]
    out[..., 1] = rgb[..., 1]
    out[..., 2] = rgb[..., 0]
    out[..., 3] = 255
    return out


class KmsDisplay:
    dimmable = False

    def __init__(self, backend: Any, *, rotate: int = 0) -> None:
        if rotate not in (0, 180):
            raise DisplayError(f"rotate must be 0 or 180, got {rotate}")
        self._backend, self._rotate = backend, rotate
        self.width, self.height = int(backend.width), int(backend.height)
        self._front = 0
        self._on = True

    def show(self, image: Image.Image) -> None:
        if image.size != (self.width, self.height):
            raise DisplayError(
                f"frame must be {self.width}x{self.height}, "
                f"got {image.size[0]}x{image.size[1]}"
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        back = 1 - self._front
        try:
            self._backend.write(back, pack_xrgb8888(np.asarray(image), self._rotate))
            self._backend.flip(back)
        except (OSError, RuntimeError) as exc:
            raise DisplayError(f"display swap failed: {exc}") from exc
        self._front = back

    def backlight(self, percent: int) -> None:
        on = int(percent) > 0
        if on == self._on:
            return
        try:
            self._backend.power(on)
        except (OSError, RuntimeError) as exc:
            raise DisplayError(f"display power {'on' if on else 'off'} failed: {exc}") from exc
        self._on = on

    def close(self) -> None:
        try:
            if not self._on:
                self._backend.power(True)
        except (OSError, RuntimeError):
            pass
        finally:
            self._backend.close()


class _PykmsBackend:
    """The pykms calls. No logic beyond what the hardware needs."""

    def __init__(self, pykms: Any) -> None:
        self._pykms = pykms
        self._card = pykms.Card()
        resources = pykms.ResourceManager(self._card)
        self._conn = resources.reserve_connector(CONNECTOR)
        self._crtc = resources.reserve_crtc(self._conn)
        self._mode = self._conn.get_default_mode()
        self.width, self.height = self._mode.hdisplay, self._mode.vdisplay
        self._fbs = [
            pykms.DumbFramebuffer(self._card, self.width, self.height, PIXEL_FORMAT)
            for _ in range(2)
        ]
        self._maps = []
        for fb in self._fbs:
            if fb.stride(0) != self.width * 4:
                raise RuntimeError(
                    f"unexpected buffer stride {fb.stride(0)} for width {self.width}"
                )
            flat = np.frombuffer(fb.map(0), dtype=np.uint8)
            self._maps.append(
                flat[: self.height * self.width * 4].reshape(self.height, self.width, 4)
            )
        self._crtc.set_mode(self._conn, self._fbs[0], self._mode)
        self._plane = self._crtc.primary_plane

    def write(self, index: int, pixels: np.ndarray) -> None:
        np.copyto(self._maps[index], pixels)

    def flip(self, index: int) -> None:
        request = self._pykms.AtomicReq(self._card)
        request.add(self._plane, "FB_ID", self._fbs[index].id)
        result = request.commit_sync()
        if result:
            raise OSError(f"atomic commit returned {result}")

    def power(self, on: bool) -> None:
        # The connector's DPMS property is refused on this panel ("commit failed",
        # measured 2026-10-10); switching the CRTC's ACTIVE state is what works.
        request = self._pykms.AtomicReq(self._card)
        request.add(self._crtc, "ACTIVE", 1 if on else 0)
        result = request.commit_sync(allow_modeset=True)
        if result:
            raise OSError(f"ACTIVE commit returned {result}")

    def close(self) -> None:
        # Dropping the last references closes the card, which hands the screen back
        # to the kernel console.
        self._maps.clear()
        self._fbs.clear()
        self._plane = self._crtc = self._conn = self._card = None


def open_waveshare35dsi(rotate: int = 0) -> KmsDisplay:
    """Open the DSI panel. Raises DisplayError with the cause and the fix."""
    if rotate not in (0, 180):
        raise DisplayError(f"rotate must be 0 or 180, got {rotate}")
    try:
        import pykms
    except ImportError as exc:
        raise DisplayError(
            "display libraries missing; install python3-kms++ from apt"
        ) from exc
    try:
        backend = _PykmsBackend(pykms)
    except PermissionError as exc:
        raise DisplayError(
            f"no permission for the DSI display: {exc}; add the service user to the "
            "video and render groups and log in again"
        ) from exc
    except Exception as exc:  # pykms raises plain RuntimeError/ValueError for most faults
        raise DisplayError(
            f"cannot open the DSI display on {CONNECTOR}: {exc}; check the "
            "dtoverlay=waveshare_35DSI line in config.txt and the cable, and that the Pi "
            "boots to the console (a desktop session owns the screen)"
        ) from exc
    return KmsDisplay(backend, rotate=rotate)
```

- [ ] **Step 4: Update the package docstring**

Replace the docstring of `pifilm/display/__init__.py` with:

```python
"""Optional LCD viewfinder for the Pi: the 3.5" DSI panel, or the 2.8" SPI panel.

Hardware libraries are imported only inside the ``open_*`` factories so that
the package imports cleanly on a Mac and in the test suite.
"""
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest -q tests/test_display_kms.py && .venv/bin/ruff check pifilm/display tests/test_display_kms.py`
Expected: 16 passed, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pifilm/display/kms.py pifilm/display/__init__.py tests/test_display_kms.py
git commit -m "display: DSI panel driver through DRM/KMS

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Application wiring and the service unit

**Files:**
- Modify: `pifilm/capture/app.py` (`_open_display`, the `--display` and `--display-dim-after` arguments, `_drop_display_on_v4l2` docstring)
- Modify: `deploy/pifilm-capture.service.example`
- Test: `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `pifilm.display.kms.open_waveshare35dsi(rotate)`, `pifilm.display.evtouch.open_goodix_touch(size, rotate)`.
- Produces: `pifilm-capture --display waveshare35dsi`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app.py` (it already has the `camera_cli` fixture used by the `--display waveshare28` tests near the end of the file; reuse it as those tests do):

```python
# -- the DSI panel ----------------------------------------------------------------


def test_open_display_opens_the_dsi_panel_and_its_touch(monkeypatch):
    import argparse

    from pifilm.capture import app
    from pifilm.display import evtouch, kms

    calls = {}
    display = SimpleNamespace(width=640, height=480, close=lambda: None)

    def open_display(rotate):
        calls["display"] = rotate
        return display

    def open_touch(size, rotate):
        calls["touch"] = (size, rotate)
        return "touch"

    monkeypatch.setattr(kms, "open_waveshare35dsi", open_display)
    monkeypatch.setattr(evtouch, "open_goodix_touch", open_touch)
    args = argparse.Namespace(display="waveshare35dsi", display_rotate=180)
    assert app._open_display(args, Path("/unused")) == (display, "touch")
    assert calls == {"display": 180, "touch": ((640, 480), 180)}


def test_open_display_closes_the_dsi_panel_when_touch_fails(monkeypatch):
    import argparse

    from pifilm.capture import app
    from pifilm.display import DisplayError, evtouch, kms

    closed = []
    display = SimpleNamespace(width=640, height=480, close=lambda: closed.append(1))

    def no_touch(size, rotate):
        raise DisplayError("touch device not found")

    monkeypatch.setattr(kms, "open_waveshare35dsi", lambda rotate: display)
    monkeypatch.setattr(evtouch, "open_goodix_touch", no_touch)
    args = argparse.Namespace(display="waveshare35dsi", display_rotate=0)
    with pytest.raises(DisplayError, match="touch device not found"):
        app._open_display(args, Path("/unused"))
    assert closed == [1]


def test_dsi_display_warns_and_continues_on_v4l2(camera_cli, monkeypatch, capsys):
    assert camera_cli.main(
        ["--camera", "v4l2", "--display", "waveshare35dsi", "--no-preview"]) == 0
    assert "warning: display unavailable" in capsys.readouterr().err
```

If `SimpleNamespace`, `Path` or `pytest` is not already imported at the top of `tests/test_app.py`, add `from types import SimpleNamespace`, `from pathlib import Path` and `import pytest` there.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/test_app.py -k "dsi"`
Expected: the first two FAIL (`_open_display` opens the 2.8" panel instead); the third FAILS with `SystemExit: 2` (invalid choice).

- [ ] **Step 3: Implement the wiring**

In `pifilm/capture/app.py`, in `_open_display`, insert this block directly after the `if args.display == "fake":` block and before the `from ..display.cst3530 import ...` line:

```python
    if args.display == "waveshare35dsi":
        from ..display import evtouch, kms
        display = kms.open_waveshare35dsi(args.display_rotate)
        try:
            touch = evtouch.open_goodix_touch(
                (display.width, display.height), args.display_rotate,
            )
        except DisplayError:
            display.close()
            raise
        return display, touch
```

Replace the `--display` argument definition with:

```python
    parser.add_argument(
        "--display", choices=("none", "waveshare35dsi", "waveshare28", "fake"), default="none",
        help="LCD viewfinder with exposure meter (Picamera2 or --fake only): "
             "'waveshare35dsi' is the 3.5 inch DSI panel, 'waveshare28' the 2.8 inch SPI "
             "panel, 'fake' writes OUT/viewfinder-last.png instead of driving a panel",
    )
```

In the `--display-dim-after` argument, change the help text to `"halve the LCD backlight after this long untouched; 0 disables (no effect on the DSI panel, which has no brightness control)"`.

In `_drop_display_on_v4l2`'s docstring, change `passes ``--display waveshare28``` to `passes ``--display waveshare35dsi```.

In the comment above `display_pair = None` in `main` that reads `# claims SPI, GPIO and I2C and pulses the touch reset line.`, change it to `# claims the panel (SPI, GPIO and I2C, or the DRM output) and may pulse a reset line.`

- [ ] **Step 4: Update the service unit example**

In `deploy/pifilm-capture.service.example`:

- Replace the three comment lines starting `# --display waveshare28 runs the LCD viewfinder.` with:

```ini
# --display waveshare35dsi runs the viewfinder on the Waveshare 3.5" DSI panel (use
# waveshare28 for the older 2.8" SPI panel). The Pi must boot to the console so the
# service owns the screen. On a Pi without the panel, or on one whose camera is USB
# (the V4L2 backend has no preview mode), the service logs one warning line and
# continues headless for the Stick rather than failing to start.
```

- In the `ExecStart=` line, replace `--display waveshare28` with `--display waveshare35dsi`.
- In the comment line ending `clipped over 1% of pixels at the sensor (docs/retraining-imx477.md).`, delete ` (docs/retraining-imx477.md)` so it ends `at the sensor.` (the document was deleted in `6ee14db`).

- [ ] **Step 5: Run the whole suite and lint**

Run: `.venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 6: Check the fake display still writes a frame**

Run: `.venv/bin/pifilm-capture --fake --display fake --no-preview --out "$TMPDIR/pifilm-vf" & pid=$!; sleep 4; kill $pid; .venv/bin/python -c "from PIL import Image; import os; print(Image.open(os.path.join(os.environ['TMPDIR'], 'pifilm-vf', 'viewfinder-last.png')).size)"`
Expected: `(320, 240)`.

- [ ] **Step 7: Commit**

```bash
git add pifilm/capture/app.py deploy/pifilm-capture.service.example tests/test_app.py
git commit -m "pifilm-capture: --display waveshare35dsi

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/lcd-viewfinder.md`, `docs/setup.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: the flag name, group names and messages from Tasks 3, 6 and 7; the spike facts from Task 1.

- [ ] **Step 1: `docs/lcd-viewfinder.md`: intro and a DSI section**

Replace the opening paragraph (the three lines under `# LCD viewfinder`, ending `see the [checklist](#7-hardware-acceptance-checklist).`) with:

```markdown
A touch LCD on the Pi, run by `pifilm-capture --display ...` as a live viewfinder
with a light-meter readout and an on-screen shutter. Two panels are supported:

| Panel | Flag | Connection | Status |
| --- | --- | --- | --- |
| Waveshare 3.5" DSI LCD (E), 640×480, Goodix touch | `--display waveshare35dsi` | DSI ribbon cable only | The default. Setup in [section 9](#9-the-35-dsi-panel) |
| Waveshare 2.8" Capacitive Touch LCD (V2), 320×240, ST7789 + CST3530 | `--display waveshare28` | SPI, I2C and four GPIO pins on the header | Fallback. Sections 2, 3 and 8 describe it |

Sections 5 and 6 (the screen, the meter) apply to both: the DSI panel shows the
same layout at twice the size. Where the two differ, section 9 says how.
```

Append this section at the end of the file:

````markdown
## 9. The 3.5" DSI panel

The Waveshare 3.5" DSI LCD (E) is a kernel display, not an SPI device: an overlay
registers it with the Pi's display system and `pifilm-capture` hands it finished
frames. It uses no pins on the GPIO header, and its touch controller is on the DSI
cable's own I2C bus, not the one shared with the X728.

### 9.1 Pi setup

1. **Cable.** Connect the panel to the Pi 4's 15-pin display (DSI) port with the
   supplied ribbon cable. Nothing else is wired.

2. **Overlay file.** The overlay is Waveshare's, not part of Raspberry Pi OS
   (installed 2026-10-10 from the address below; a kernel upgrade may need a
   newer file from the same wiki page):

   ```sh
   wget -O /tmp/Waveshare_35DSI.dtbo https://files.waveshare.com/wiki/common/Waveshare_35DSI.dtbo
   sudo cp /tmp/Waveshare_35DSI.dtbo /boot/firmware/overlays/
   ```

3. **`config.txt`** (`/boot/firmware/config.txt`). Keep the existing
   `dtoverlay=vc4-kms-v3d` line and add:

   ```text
   dtoverlay=waveshare_35DSI,35E,dsi1
   ```

4. **Boot to the console.** The service must own the screen; under a desktop
   session the compositor owns it and the viewfinder cannot open it. On the
   Desktop image:

   ```sh
   sudo raspi-config nonint do_boot_behaviour B1
   ```

   (`B4` restores the desktop with autologin, which two-screen mode needs.)

5. **Package and groups.** `python3-kms++` comes with `python3-picamera2`; install
   it if `python3 -c "import pykms"` fails. The service user needs `video`,
   `render` and `input`:

   ```sh
   sudo apt install python3-kms++
   sudo usermod -aG video,render,input george
   ```

   Reboot.

6. **Verify.** After the reboot the panel shows boot text and a login prompt, and:

   ```sh
   cat /sys/class/drm/card*-DSI-1/status          # connected
   cat /sys/class/drm/card*-DSI-1/modes           # 640x480
   grep -i goodix /proc/bus/input/devices         # Goodix Capacitive TouchScreen
   ```

### 9.2 Running

```sh
# Terminal test: live view on the panel, Ctrl-C to stop
.venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --no-preview --out ~/Pictures/pifilm-lcd

# Upside down for how the panel is mounted
.venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --display-rotate 180 --no-preview --out ~/Pictures/pifilm-lcd

# Print mapped touch coordinates for each tap
.venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --touch-debug --no-preview --out ~/Pictures/pifilm-lcd
```

`deploy/pifilm-capture.service.example` passes `--display waveshare35dsi`, so with
the unit installed the viewfinder appears at boot without logging in. While the
service runs it replaces the console on the panel; `sudo systemctl stop
pifilm-capture` brings the login prompt back.

### 9.3 What differs from the 2.8" panel

| | 2.8" SPI | 3.5" DSI |
| --- | --- | --- |
| Resolution | 320×240 | 640×480 (same layout, drawn at 2×) |
| Idle | Dims at 1 minute, off at 5 | No brightness control: no dim step; off at 5 minutes (`--display-off-after`). `--display-dim-after` has no effect |
| Touch | Polled over I2C1, shared with the X728 | Kernel input events; nothing on I2C1 |
| GPIO header | SPI0 and BCM 17, 18, 25, 27 | None |
| Needs | `dtparam=spi=on`, `dtparam=i2c_arm=on` | The overlay, console boot |

### 9.4 Measured on the Pi 4 (2026-10-10)

| Item | Value |
| --- | --- |
| DRM connector | `DSI-1`, one mode, 640×480 |
| Framebuffer | `/dev/fb0`, `vc4drmfb`, 16 bpp (not used; the driver uses DRM buffers) |
| Backlight | none under `/sys/class/backlight/` |
| Touch | `Goodix Capacitive TouchScreen`, I2C bus 10 address `0x5d`; the `/dev/input/eventN` number varies, so it is found by name |

### 9.5 Troubleshooting

Printed as `warning: display unavailable (<message>); continuing without it`.

| Message contains | Cause | Fix |
| --- | --- | --- |
| `install python3-kms++` | `pykms` is not importable | `sudo apt install python3-kms++`; the venv must see system packages, as for Picamera2 |
| `cannot open the DSI display on DSI-1` | No such connector, the cable is out, or a desktop session owns the screen | Check the `dtoverlay=waveshare_35DSI,35E,dsi1` line and the overlay file, reseat the cable, boot to the console (9.1 step 4) |
| `no permission for the DSI display` | Service user not in `video`/`render` | `sudo usermod -aG video,render george`; reboot |
| `touch device 'Goodix Capacitive TouchScreen' not found` | The touch driver did not load | Same overlay and cable checks; `grep -i goodix /proc/bus/input/devices` |
| `no permission for /dev/input/event*` | Service user not in `input` | `sudo usermod -aG input george`; reboot |
| Panel blank after boot, no boot text | Overlay not applied | `dmesg \| grep -i "dsi\|panel\|goodix"`; check steps 2 and 3 |

### 9.6 Hardware acceptance checklist

Not yet run. Record results here.

1. Cold boot: the live view appears on the panel without logging in. Note the time
   from power-on.
2. The journal's `viewfinder: N fps` line reads 8 or more; panning shows no tearing.
3. Each control responds where it is drawn, including in the corners;
   `--touch-debug` prints coordinates within 0..639 × 0..479.
4. Shutter tap, EV, shutter priority, `2x` and review behave as in items 3, 5, 11
   and 12 of [section 7](#7-hardware-acceptance-checklist).
5. A Stick shot appears on the panel, which returns to live after 30 s untouched.
6. Idle: no dimming at one minute; dark at five; a tap wakes it without taking a
   photo (no new `captures.jsonl` line); a Stick shot while dark wakes it into the
   review.
7. `sudo systemctl stop pifilm-capture`: the login prompt returns to the panel.
8. Unplug the DSI cable and restart the service: one `warning: display unavailable`
   line in the journal, and the Stick still captures.
9. `--display-rotate 180`: the image is inverted and taps still land on the controls.
10. Optional, only if the 2.8" panel is still wired: `--display waveshare28` works.
````

Also in the same file: in section 4, after the sentence ending `It is not an error there, though:`, no change; in the paragraph beginning `` `deploy/pifilm-capture.service.example`'s `ExecStart` already includes ``, replace `` `--display waveshare28` `` with `` `--display waveshare35dsi` `` and replace `five consecutive SPI failures` with `five consecutive display failures`.

- [ ] **Step 2: `docs/setup.md`**

Read section 4.8 (`### 4.8 Optional: LCD viewfinder`) and replace its body with:

```markdown
For a live viewfinder with a light-meter readout and an on-screen shutter, fit
the Waveshare 3.5" DSI LCD (E) and run the service with `--display
waveshare35dsi`, which the example unit already does. It needs Waveshare's
overlay file, one `config.txt` line, the Pi booting to the console, and the
service user in `video`, `render` and `input`; the steps, the screen layout and
the hardware acceptance checklist are in
[docs/lcd-viewfinder.md](lcd-viewfinder.md#9-the-35-dsi-panel). The older
Waveshare 2.8" SPI panel is still supported with `--display waveshare28`
(sections 2 and 3 of the same guide).
```

In the paragraph near line 234 that quotes the unit's arguments, replace `--display waveshare28` with `--display waveshare35dsi` and `[Waveshare panel](lcd-viewfinder.md)` with `[Waveshare DSI panel](lcd-viewfinder.md#9-the-35-dsi-panel)`.

- [ ] **Step 3: `README.md` (three lines only; the rest is the user's wording)**

- Parts list line `- Waveshare 2.8 inch LCD Display Module and touchscreen (light meter, focus meter, trigger, exposure controls, battery meter)` → `- Waveshare 3.5 inch DSI LCD (E) touchscreen (light meter, focus meter, trigger, exposure controls, battery meter)`
- In the bullet beginning `- **LCD too**`, replace `(`--display waveshare28`)` with `(`--display waveshare35dsi`)`.
- Flag table row: `` | `--display waveshare28` | LCD viewfinder with exposure meter and touch shutter ([guide](docs/lcd-viewfinder.md)) | `` → `` | `--display waveshare35dsi` | LCD viewfinder with exposure meter and touch shutter; `waveshare28` for the older 2.8" SPI panel ([guide](docs/lcd-viewfinder.md)) | ``

- [ ] **Step 4: `CLAUDE.md`**

- In "What this is", replace `` `docs/lcd-viewfinder.md` the optional Waveshare LCD viewfinder (wiring, setup, screen layout, meter, hardware acceptance) `` with `` `docs/lcd-viewfinder.md` the optional Waveshare LCD viewfinder (the 3.5" DSI panel and the fallback 2.8" SPI panel: setup, screen layout, meter, hardware acceptance) ``.
- In "Capture", `app.py` bullet: replace `` `--display waveshare28` runs the LCD viewfinder instead of any OpenCV window `` with `` `--display waveshare35dsi` (or `waveshare28`) runs the LCD viewfinder instead of any OpenCV window ``.
- Replace the whole "Display" paragraph with:

```markdown
Optional LCD viewfinder, wired up by `--display waveshare35dsi` (Waveshare 3.5" DSI LCD (E),
640×480) or `--display waveshare28` (the fallback 2.8" SPI panel, 320×240). `kms.py` drives the
DSI panel through DRM/KMS (`pykms`, two swapped buffers; the Pi must boot to the console so the
service owns the screen) and `evtouch.py` reads its Goodix touch from kernel input events, found
by device name; `st7789.py` and `cst3530.py` drive the SPI panel and its touch controller.
`touch.py` holds the shared `TouchPoint`/`Tap`/`TapDetector`. `meter.py` computes the light-meter
readout (pure, from preview metadata + pixels); `shutter.py` is the shutter-priority step scale
(pure); `ui.py` renders the live/review/message screens and hit-tests taps in 320×240 base units
at an integer `scale` (pure); `viewfinder.py`'s `ViewfinderLoop` is the LIVE/REVIEW state machine
and takes the scale from the display's width. The DSI panel has no brightness control, so it
skips the idle dim step and only turns off. All hardware imports (`pykms`, `spidev`, `gpiozero`,
`smbus2`) are lazy, inside the `open_*` factories, so the package imports on a Mac and in tests.
`fake.py` (`--display fake`) writes frames to a PNG instead. The loop shares the
`CaptureController` with the Stick's remote server, so an LCD tap and a Stick request are the
same kind of job.
```

- [ ] **Step 5: Check the links and commit**

Run: `grep -n "waveshare28\|waveshare35dsi" README.md CLAUDE.md docs/setup.md docs/lcd-viewfinder.md deploy/pifilm-capture.service.example`
Expected: every place that names the default panel says `waveshare35dsi`; `waveshare28` appears only where the fallback panel is described.

```bash
git add docs/lcd-viewfinder.md docs/setup.md README.md CLAUDE.md
git commit -m "docs: the 3.5 inch DSI panel is the default viewfinder

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Hardware acceptance (on the Pi, with the user)

**Files:**
- Modify: `docs/lcd-viewfinder.md` (section 9.6, results)

**Interfaces:**
- Consumes: everything above, deployed to the Pi.

- [ ] **Step 1: Push the branch and ask the user to deploy it**

```bash
git push -u origin feat/dsi-viewfinder
```

Ask the user to run on the Pi (the project directory is the unit's `WorkingDirectory`, `/home/george/repos/pi-film-reversal`):

```bash
cd /home/george/repos/pi-film-reversal
git fetch origin && git checkout feat/dsi-viewfinder && git pull
sudo systemctl stop pifilm-capture
.venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --touch-debug --no-preview --out ~/Pictures/pifilm-lcd
```

Expected on the panel: the live view with the meter bar, at full screen. Expected in the terminal: `LCD viewfinder running.` and, every 10 s, `viewfinder: N fps`.

- [ ] **Step 2: Walk the checklist**

Go through items 2 to 6 and 9 of `docs/lcd-viewfinder.md` section 9.6 with the user in that terminal session, collecting the fps figure, any coordinates that look wrong, and what the screen did at the idle times (use `--display-off-after 20` to avoid waiting five minutes).

- [ ] **Step 3: Switch the installed service to the new panel**

Ask the user to edit the installed unit and restart:

```bash
sudo sed -i 's/--display waveshare28/--display waveshare35dsi/' /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload && sudo systemctl restart pifilm-capture
journalctl -u pifilm-capture -n 20 --no-pager
sudo reboot
```

Then items 1, 7 and 8 of section 9.6 (cold boot, stop returns the console, cable unplugged).

- [ ] **Step 4: Handle a failed item**

| Failure | Action |
| --- | --- |
| Below 8 fps | Report the figure. The remedy from the spec is to pre-render the static overlay once per state change; that is a follow-up change to `ui.py`, planned separately |
| Screen does not go dark at the off time, or stays dark on wake | Apply the `power` row of Task 6's table, re-run Task 6's tests, redeploy |
| Tearing or swap errors in the journal | Apply the `flip` row of Task 6's table, re-run Task 6's tests, redeploy |
| Taps land mirrored or offset | Capture the `--touch-debug` output for the four corners and stop: the coordinate mapping in `evtouch.py` needs the device's real axis orientation |

- [ ] **Step 5: Record the results and commit**

Replace `Not yet run. Record results here.` in section 9.6 with a dated line and the outcome of each item (pass, or what was seen), then:

```bash
git add docs/lcd-viewfinder.md
git commit -m "docs: DSI viewfinder hardware acceptance results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push
```

- [ ] **Step 6: Open the pull request**

```bash
gh pr create --title "DSI viewfinder: Waveshare 3.5\" DSI LCD (E)" --body "$(cat <<'EOF'
Adds `--display waveshare35dsi`: the viewfinder on the Waveshare 3.5" DSI LCD (E), 640x480, through DRM/KMS and kernel touch events. The 2.8" SPI panel stays as a fallback.

- Spec: docs/superpowers/specs/2026-10-10-dsi-viewfinder-design.md
- Plan: docs/superpowers/plans/2026-10-10-dsi-viewfinder.md
- Hardware acceptance: docs/lcd-viewfinder.md section 9.6

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
