# DSI viewfinder (Waveshare 3.5" DSI LCD (E)) — design

**Date:** 2026-10-10
**Status:** approved in discussion; awaiting spec review.
**Depends on:** the LCD viewfinder (`docs/lcd-viewfinder.md`) and the Picamera2 backend.
Builds on `main` at `f30c85b`.

## 1. Goal

The camera's viewfinder moves from the Waveshare 2.8" SPI panel to the Waveshare
3.5" DSI LCD (E), connected by the DSI ribbon cable. After boot, `pifilm-capture`
comes up full screen on the new panel with the same screens and controls as today:
live view, meter, focus bar, touch shutter, EV, shutter priority, `2x`, processing
bars and review. The 2.8" panel stays supported as a fallback and is otherwise
untouched.

## 2. Decisions taken in brainstorming

| Topic | Decision |
| --- | --- |
| Boot mode | Console boot on the existing Desktop image (`raspi-config nonint do_boot_behaviour B1`). No reflash. `pifilm-capture` stays a system service and owns the screen. |
| Old panel | Kept. `--display waveshare28` and its two drivers remain, tested, as a fallback. |
| New flag value | `--display waveshare35dsi`. The example service unit uses it. |
| Display path | DRM/KMS through `pykms` (`python3-kms++`, already installed with Picamera2). Two buffers, swapped per frame. |
| Fallback display path | The kernel framebuffer `/dev/fb0`. Not built unless the KMS path fails on hardware (section 8). |
| Touch path | Kernel input events from the Goodix device, decoded with the standard library. No new dependency. |
| Resolution | The UI is drawn natively at 640×480 by scaling the existing 320×240 layout by 2. No upscaled bitmap. |
| Idle | No brightness control exists on this panel, so the dim step is skipped. Screen-off after `--display-off-after` stays, by powering the output down. |
| Rotation | `--display-rotate 0\|180`, done in software for both image and touch. The panel is upright at 0 in the current mounting. |
| Out of scope | Brightness control, running under the desktop compositor, removing the 2.8" drivers, hiding boot text, two-screen mode on the DSI panel. |

## 3. Facts measured on the Pi (2026-10-10)

Raspberry Pi 4, Raspberry Pi OS Trixie Desktop image booted to the console, with
`dtoverlay=waveshare_35DSI,35E,dsi1` in `/boot/firmware/config.txt` and
`Waveshare_35DSI.dtbo` (from Waveshare's wiki) in `/boot/firmware/overlays/`.

| Item | Value |
| --- | --- |
| DRM connector | `DSI-1` on `card0`, connected, one mode: 640×480 |
| Framebuffer | `/dev/fb0`, `vc4drmfb`, 640×480, 16 bpp, stride 1280, `root:video` mode 660 |
| Backlight | `/sys/class/backlight/` is empty |
| Touch | `Goodix Capacitive TouchScreen`, `/dev/input/event4`, I2C bus 10 address `0x5d`, multi-touch (slots, tracking ids, `ABS_MT_POSITION_X/Y`) |
| Service user | `george` is in `video`, `render` and `input` |
| `pykms` | Imports. A dumb `XR24` framebuffer set on `DSI-1` showed a clean, upright test pattern, both as root and as `george` without `sudo` |

The event node number (`event4`) is not stable across boots or hardware changes, so
the driver finds the device by name.

## 4. Components

### 4.1 Shared touch types (`pifilm/display/touch.py`, new, pure)

`TouchPoint`, `Tap` and `TapDetector` move here unchanged from `cst3530.py`, because
both panels and the viewfinder loop use them. `cst3530.py` imports them from the new
module and keeps re-exporting the names, so existing imports and tests keep working.
`viewfinder.py` imports from `touch.py`.

### 4.2 KMS display (`pifilm/display/kms.py`, new)

```python
class KmsDisplay:
    width: int            # 640
    height: int           # 480
    def show(self, image: PIL.Image.Image) -> None
    def backlight(self, percent: int) -> None     # 0 powers the output off, >0 on
    dimmable = False
    def close(self) -> None

def open_waveshare35dsi(rotate: int = 0) -> KmsDisplay
```

- `open_waveshare35dsi` imports `pykms` lazily, opens the card, reserves the `DSI-1`
  connector and a CRTC, takes the connector's default mode and allocates two dumb
  `XR24` framebuffers of that size. It sets the mode with the first buffer.
- Every failure raises `DisplayError` with the cause: `pykms` missing (install
  `python3-kms++` from apt), no `DSI-1` connector or not connected (overlay line or
  cable), cannot become DRM master (a desktop session is running; boot to console),
  permission denied (service user not in `video`/`render`).
- `show` rejects a frame that is not `width`×`height`, converts RGB to the buffer's
  byte order (B, G, R, X) with NumPy, applies the 180° rotation as an array flip when
  asked, copies into the buffer that is not on screen, and swaps buffers. The swap
  follows the atomic-commit pattern of Picamera2's own `DrmPreview`. An `OSError`
  or `RuntimeError` from the commit becomes `DisplayError`, so the loop's existing
  five-failure breaker applies.
- `pack_xrgb8888(rgb: np.ndarray, rotate: int) -> np.ndarray` is a pure module
  function, tested on the Mac.
- `backlight(0)` powers the output off by an atomic commit that sets the CRTC's
  `ACTIVE` property to 0 (the connector's DPMS property is refused on this panel,
  measured 2026-10-10); any other value sets it back to 1. The level itself is ignored. This lets the existing
  `IdleDimmer` and `_apply_backlight` drive screen-off without knowing the panel.
- `close` powers the output on, releases the buffers and drops the card, which hands
  the screen back to the kernel console.

While the process holds the display, the kernel console is not drawn on the panel.
Boot messages before the service starts, and the login prompt after it stops, are
left as they are.

### 4.3 Event touch (`pifilm/display/evtouch.py`, new)

```python
class EventTouch:
    def read(self) -> list[TouchPoint]
    def close(self) -> None

def open_goodix_touch(size: tuple[int, int], rotate: int = 0) -> EventTouch
```

- `open_goodix_touch` scans `/dev/input/event*`, reads each device's name with the
  `EVIOCGNAME` ioctl and picks the one named `Goodix Capacitive TouchScreen`. It reads
  the X and Y ranges with `EVIOCGABS` and opens the node non-blocking. Not found,
  or permission denied, raises `DisplayError` naming the fix (overlay or cable;
  `input` group).
- `read` drains every pending `input_event` record and updates the state of
  multi-touch slot 0 (tracking id, X, Y). It returns one `TouchPoint` while the
  finger is down and an empty list otherwise. Coordinates are scaled from the
  device's range to `size` and mirrored on both axes when `rotate` is 180.
- A press and release that both arrive inside one `read` would otherwise be invisible
  to `TapDetector`. `read` therefore reports the point once for a press seen since the
  previous call, even if the finger is already up, and the following call reports
  none.
- Record decoding is a pure function, `decode_events(data: bytes) -> list[tuple[int,
  int, int]]` (type, code, value), and the slot tracking is a small pure class, so
  both are tested on the Mac with hand-built byte strings. The record layout is the
  64-bit one (24 bytes); the Pi runs a 64-bit userland.
- The device is not grabbed. On a console boot nothing else acts on touch input.
- An `OSError` during `read` becomes `DisplayError`; the loop already treats that as
  a logged glitch, not a failure.

### 4.4 UI scale (`pifilm/display/ui.py`)

The layout constants stay as they are, in 320×240 base units. Each public function
gains a `scale: int = 1` keyword:

- `render_live`, `render_review`, `render_processing`, `render_message` return an
  image of `(320*scale, 240*scale)`. Every coordinate, line width, radius and font
  size is multiplied by `scale`, so text and outlines are drawn sharp at the target
  size.
- `hit(x, y, scale=1)` divides the tap coordinates by `scale` before testing the
  existing regions.

At `scale=1` the output is pixel-identical to today's, which the existing tests pin.

The Picamera2 preview stream is already 640 pixels wide (`PREVIEW_WIDTH`), so at
`scale=2` the live image is shown at its own resolution on a 4:3 sensor and
letterboxed, unscaled in width, on a 16:9 one. The camera backend does not change.

### 4.5 Viewfinder loop (`pifilm/display/viewfinder.py`)

- `ViewfinderLoop` derives `scale = display.width // 320` when the display has a
  `width` (default 1, which covers `FileDisplay`) and passes it to every `render_*`
  call and to `hit`.
- A display whose `dimmable` attribute is `False` gets `dim_after = 0`; `off_after`
  is kept. Displays without the attribute behave as today.
- Nothing else changes. Touch polling, the tap queue, the states and the display
  breaker are as they are.

### 4.6 Application wiring (`pifilm/capture/app.py`)

- `--display` gains the choice `waveshare35dsi`. Its help text names both panels.
- `_open_display` opens `open_waveshare35dsi(rotate)` then
  `open_goodix_touch((display.width, display.height), rotate)`, closing the display
  if touch fails, exactly as the 2.8" branch does.
- The warning-and-continue path for `DisplayError`, the V4L2 downgrade and the
  closing order are shared with the existing panel and unchanged.
- `--display-dim-after` has no effect on this panel; its help text says so.

### 4.7 Deployment and docs

- `deploy/pifilm-capture.service.example`: `--display waveshare35dsi`, and its
  comment names the panel. The dangling reference to the deleted
  `docs/retraining-imx477.md` in the same file is removed while the line is touched.
- `docs/lcd-viewfinder.md`: a section for the DSI panel covering the cable, the
  overlay file and `config.txt` line, console boot, groups (`video`, `render`,
  `input`), the apt package (`python3-kms++`), the measured facts in section 3, how
  to run it, what differs from the 2.8" (no dim step, no GPIO pins used), new
  troubleshooting rows, and the acceptance checklist in section 7. The 2.8" material
  stays, marked as the fallback panel.
- `README.md`, `docs/setup.md` and `CLAUDE.md`: name the DSI panel as the default
  viewfinder and the 2.8" as the fallback. README wording is the user's own; edits
  there are limited to the parts list line, the `--display` table row and the LCD
  bullet.

## 5. Data flow

```
camera.read(full=False) ─► meter/focus ─► ui.render_live(scale=2) ─► KmsDisplay.show
                                                                        │ pack XRGB, copy, swap
Goodix ─► kernel ─► /dev/input/eventN ─► EventTouch.read ─► TapDetector ─► ui.hit(scale=2)
```

## 6. Error handling

| Situation | Behaviour |
| --- | --- |
| Panel absent, overlay missing, `pykms` missing, no permission | One `warning: display unavailable (...)` line; service continues headless for the Stick |
| A desktop session holds the display | Same warning, with the message naming console boot |
| Touch device absent | Display is closed again; same warning; headless |
| Five consecutive failed swaps | Existing breaker: panel closed, remote API keeps serving |
| Touch read error | Logged, rate-limited, treated as no finger |
| Process exits or is killed | Output is powered on and the console returns to the panel |

## 7. Testing

Unit tests, run on the Mac with fakes:

- `pack_xrgb8888`: byte order, shape, 0° and 180°.
- `KmsDisplay` against fake card/framebuffer objects: alternates buffers, rejects a
  wrong-sized frame, maps commit errors to `DisplayError`, `backlight(0)`/`(80)`
  toggle power, `close` restores power.
- `open_waveshare35dsi` with `pykms` absent raises `DisplayError`.
- `decode_events` and slot tracking: down, move, up; press and release in one read;
  scaling to 640×480; 180° mirroring; events for other slots ignored.
- `ui`: at `scale=2` each render returns 640×480; `hit` at doubled coordinates returns
  the same action as at base coordinates for every region; `scale=1` output unchanged.
- `ViewfinderLoop` with a 640-wide fake display: renders at 640×480, a tap at doubled
  shutter coordinates submits a capture, and a non-dimmable display never dims but
  still turns off.
- `app`: `--display waveshare35dsi` parses; an open failure prints the warning and
  continues.

Hardware acceptance on the Pi, recorded in `docs/lcd-viewfinder.md`:

1. Cold boot: live view on the panel without logging in; time from power-on noted.
2. Frame rate line in the journal at or above 8 fps; no visible tearing when panning.
3. Tap accuracy: each button responds at its drawn position, including the corners;
   `--touch-debug` prints mapped coordinates within the 640×480 range.
4. Shutter tap, EV, shutter priority, `2x` and review behave as on the 2.8" checklist.
5. A Stick shot appears on the panel.
6. Idle: no dimming at one minute; dark at five; a tap wakes without firing; a Stick
   shot wakes into the review.
7. `sudo systemctl stop pifilm-capture`: the login prompt returns to the panel.
8. Panel cable unplugged, then restart: one warning line, the Stick still captures.
9. `--display-rotate 180`: image and touch both inverted and still aligned.
10. `--display waveshare28` with the old panel attached still works (fallback check;
    may be skipped if the panel is no longer wired).

## 8. Risks

- **Buffer swap from Python.** Proven on the panel on 2026-10-10: 100 atomic swaps
  at 10 fps, worst 20.3 ms, no tearing. It has not yet run for hours inside the
  service. If it proves unreliable there, `kms.py` is replaced by an
  `/dev/fb0` writer behind the same interface (16 bpp RGB565, which `st7789.py`
  already packs), with blanking through the framebuffer's blank control and a
  documented step to keep the console login off the panel.
- **Output power-off.** Proven on the panel on 2026-10-10 through the CRTC's `ACTIVE`
  property, once each way. If it misbehaves in service, screen-off falls back to
  showing a black frame, and the docs say the backlight stays lit.
- **Rendering cost at 4× the pixels.** Pillow compositing at 640×480 on the Pi 4 is
  expected to hold 8 fps but is unmeasured. Acceptance item 2 is the gate; the
  remedy, if needed, is to pre-render the static overlay once per state change.
- **Overlay provenance.** The panel depends on a binary overlay downloaded from
  Waveshare, not shipped with the OS. A kernel upgrade could break it; the docs
  record the file's source and the date it was installed.
