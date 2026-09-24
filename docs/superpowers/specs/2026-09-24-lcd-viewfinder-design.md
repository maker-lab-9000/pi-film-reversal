# LCD viewfinder with exposure meter — design

**Date:** 2026-09-24
**Status:** approved in discussion; spec for review
**Hardware:** Waveshare 2.8" Capacitive Touch LCD V2 (ST7789 over SPI0, CST3530 touch over
I2C1) wired to the Pi 4's GPIO through the Geekworm X728 pass-through header. First sensor
under test: Sony IMX477 (HQ camera, fixed lens). The IMX708 must keep working.

## 1. Goal

When the Pi boots, the LCD shows a live viewfinder from the camera with a light-meter-style
exposure readout. A shutter button on the screen takes a photo through the same capture
path the Stick uses. After a shot the graded result is shown until tapped or for 30 s. EV
compensation is adjustable from the screen. The Stick keeps working unchanged, and a shot
from either trigger shows up on both.

## 2. Decisions taken in brainstorming

| Topic | Decision |
| --- | --- |
| Meter | Light-meter readout from camera metadata: shutter, ISO, EV compensation, lux, a needle for stops from mid-grey; plus clipped-highlight percentage and the X728 battery badge. No histogram, no zebra. |
| Feed | Ungraded ISP preview. The look is shown only on the result screen. |
| Trigger | On-screen shutter button. The Stick and the SPACE key remain. |
| After a shot | Graded result held until a tap, or 30 s, whichever first. Stick-triggered shots are shown too. |
| Process model | Inside `pifilm-capture`, sharing the existing `CaptureController`. Not a separate process, not an fbtft framebuffer. |
| Out of scope | Focus peaking, manual shutter/gain, backlight dimming, menus, video. |

## 3. Architecture

```
pifilm-capture --display waveshare28 [--remote-listen ...] [--ups x728]

  Picamera2Camera (preview mode) ──read(full=False)──▶ ViewfinderLoop ──▶ ST7789Display
        │  lock                                             ▲   │  ▲
        └──read(full=True)◀── CaptureController ◀── submit ─┘   │  └── CST3530Touch
                 ▲                      ▲                        └── PowerMonitor.snapshot
                 │                      └─────── RemoteCaptureServer (Stick)
             CaptureSession
```

Everything new lives in a `pifilm/display/` package; `pifilm/capture/` changes are limited to
the Picamera2 backend (preview mode, multi-sensor), the controller (a completion counter),
and `app.py` (flag and wiring). The OpenCV desktop loops are untouched.

### 3.1 `pifilm/display/st7789.py` — display driver

- `class ST7789Display` with `show(image: PIL.Image)`, `backlight(percent: int)`, `close()`.
  `width, height = 320, 240` (landscape). The constructor takes `spi`, `dc`, `rst`, `bl`
  objects (duck-typed: `spi.writebytes2(bytes)`, `pin.on()/off()`, `bl.value`) or builds them
  from `spidev`/`gpiozero` when not given. `open_waveshare28(rotate: int)` is the factory
  that imports the hardware libraries lazily and raises `DisplayError` with the actionable
  cause (missing library, `/dev/spidev0.0` permission, GPIO busy).
- Init sequence: the vendor `ST7789_Init` command/data bytes verbatim (MADCTL, COLMOD 0x05,
  porch, gate, VCOM, gamma tables, INVON, SLPOUT, DISPON). MADCTL for landscape is `0x70`
  at `rotate=0` and `0xB0` at `rotate=180`; the flag exists because the correct orientation
  for the mounted cable can only be confirmed on the device.
- Frame push: `show()` requires a 320×240 RGB image, converts to RGB565 big-endian with
  NumPy (`(r & 0xF8) << 8 | (g & 0xFC) << 3 | b >> 3`), sets the full window once, and
  writes the 153 600-byte buffer with `spi.writebytes2` in ≤4096-byte chunks (the spidev
  transfer limit). No Python list conversion, no double write. Chunk size is a constant.
- Failure policy: an `OSError` from SPI during `show()` is wrapped as `DisplayError`; the
  caller decides (the loop counts consecutive failures and gives up after 5).
- `close()` writes DISPOFF and SLPIN, sets the backlight to 0, releases SPI and GPIO.

Pins (BCM): DC 25, RST 27, BL 18, SPI0 CE0. Constants at module top, with the physical-pin
table from the hardware handover in the docstring.

### 3.2 `pifilm/display/cst3530.py` — touch driver

- `class CST3530Touch` with `read() -> list[TouchPoint]` (`TouchPoint(x, y, strength)` in
  display coordinates) and `close()`. Constructor takes an `i2c` object (duck-typed on
  `write_i2c_block_data` and `read_byte`, as `smbus2.SMBus` provides), an `int_pin` object
  with `is_active`, and `rotate`. `open_waveshare28_touch(rotate)` builds them.
- Protocol: the vendor `Touch_CST3530.read_data` register sequence (32-bit register
  address, data at `0xD0070000`, next-coordinates at `0xD0070900`, end-read at
  `0xD00002AB`, address `0x58`) reimplemented as a pure function
  `decode_points(buf: bytes, extra: bytes) -> list[RawPoint]` so it is unit-tested with the
  vendor byte layout, plus the two I2C transactions around it. Reset on open: RST low 100 ms,
  high 500 ms, as the vendor does.
- Polling, not interrupt callbacks: the loop calls `read()` each frame, which always queries
  the bus. Deviation from the original design (which gated on an `int_pin.is_active` check
  before touching the bus): capacitive controllers pulse `INT` per report rather than holding
  it, and a 10 Hz poll could miss the pulse, so gating on it risked dropping taps. Reading the
  bus every step instead costs two short I2C transactions per frame, negligible on the shared
  bus (X728 gauge at 0x36, RTC at 0x68); no `INT` wiring is required.
- Rotation: raw coordinates are in 240×320 portrait space. `to_display(raw, rotate)` maps
  them to 320×240; at `rotate=0`, `x = raw.y`, `y = 239 - raw.x`; at 180, `x = 319 - raw.y`,
  `y = raw.x`. Both mappings unit-tested; which one is right is a hardware acceptance item.
- Debounce: `TapDetector` turns the per-frame point lists into `Tap(x, y)` events on release
  (finger down then up, ≤600 ms, movement <20 px). Pure, tested.

### 3.3 `pifilm/display/meter.py` — exposure readout (pure)

`compute_reading(metadata: dict, preview_rgb: np.ndarray, ev_comp: float,
power: PowerStatus | None) -> MeterReading`:

| Field | Source | Rule |
| --- | --- | --- |
| `shutter` | `ExposureTime` µs | ≥1 s: `"1.3s"`, else `"1/250"` (nearest integer denominator) |
| `iso` | `AnalogueGain × DigitalGain × 100` | rounded to the nearest 10; `DigitalGain` defaults to 1 |
| `ev_comp` | argument | formatted `"+0.7"` / `"0"` / `"-1.3"` |
| `lux` | `Lux` | `"640 lx"`; `"—"` when absent |
| `deviation_ev` | preview frame | `log2(mean_linear_luma / 0.18)`, luma BT.709 on sRGB-linearised pixels of the frame downsampled 4×; clamped to ±3 |
| `clip_pct` | preview frame | % of pixels with any channel ≥ 254 |
| `battery` | `power.percent`, `power.external_power` | `None` when no UPS |

Missing metadata keys give `None` fields, rendered as `"—"`; the function never raises on
an incomplete dict. Tested with synthetic frames (uniform grey 18 % → deviation 0 ± 0.05;
half-white frame → `clip_pct` 50).

### 3.4 `pifilm/display/ui.py` — rendering and hit-testing (pure)

- Layout constants for 320×240: image area full screen; meter bar `y ∈ [204, 240)` filled
  black at 60 % alpha; shutter button a 56 px circle centred at (288, 102) with a 2 px white
  ring; EV buttons 40×36 at the bar's left (`−`) and right (`+`) ends; busy marker a 10 px
  amber dot at (10, 10); battery badge at top-right; text via `ImageFont.load_default(size=14)` (Pillow ≥ 10.1 ships a TrueType default; the
  `Pillow` floor in `pyproject.toml` moves from 10.0 to 10.1), 11 px for labels.
- `render_live(frame_rgb, reading, busy: bool) -> PIL.Image`: fits the whole preview frame
  into 320×240 by letterboxing, never cropping (a 4:3 IMX477 preview fills exactly; a 16:9
  IMX708 preview gets 30 px black bands top and bottom, the lower one under the meter bar),
  then draws the bar, needle, buttons and badge.
- `render_review(graded_rgb, caption: str) -> PIL.Image`: letterboxed result with a one-line
  caption (`"1/250  ISO 100  +0.3"`), "tap to continue" hint.
- `render_message(title, detail) -> PIL.Image`: for errors and start-up.
- `hit(x, y) -> Action`: `Action.SHUTTER`, `EV_MINUS`, `EV_PLUS`, or `NONE`; in review mode any
  tap is `DISMISS` (handled in the loop, not here). Hit regions are exactly the drawn regions
  plus a 6 px margin.
- Tests: image dimensions and mode; pixels sampled inside the bar are darker than the same
  frame pixel; `hit()` at each region centre and just outside.

### 3.5 `pifilm/display/viewfinder.py` — the loop (state machine)

`class ViewfinderLoop(camera, controller, display, touch, clock, power_snapshot, *,
frame_period=0.1, review_timeout=30.0)`; `run(stop: threading.Event)` and `step()` for tests.

States and transitions:

| State | Each step | Transitions |
| --- | --- | --- |
| `LIVE` | read preview (`camera.read(full=False)`), poll touch, render live with `busy = controller has active job`, `display.show` | `SHUTTER` tap → `controller.submit(uuid4)` → stays `LIVE` (busy). `EV_±` tap → `camera.set_ev(ev ± 1/3)` clamped ±2. Controller's `completed_count` increased since last seen → load graded result → `REVIEW`. |
| `REVIEW` | render review once; poll touch | any tap or `review_timeout` elapsed → `LIVE`. |
| `ERROR` | render message | camera or display errors: after 5 consecutive `show()` failures the loop exits with `DisplayError`; a `CameraError` on a preview read shows "camera busy" and retries next step (the existing headless promise that a dropped frame never ends the session holds here too). |

- Pacing: `clock.sleep(max(0, frame_period − elapsed))`; `frame_period` 0.1 s (10 fps
  target). Touch is polled every step even while a read is slow. Every 10 s the loop prints
  one line with the achieved frame rate (`viewfinder: 9.6 fps`).
- Result loading uses `JobSnapshot.result.pifilm` (the graded JPEG path) through
  `pifilm.imageio.load_rgb`, downscaled with Pillow to fit 320×240. A failed job shows
  `render_message("Capture failed", error_message)` in `REVIEW` with the same dismissal.
- The loop never calls `session.capture()` directly; all shots go through the controller,
  so the Stick's 409-when-busy contract is unchanged.
- Fully tested with fakes: `FakeDisplay` records images, `FakeTouch` scripts point lists,
  `FakeClock`, the real `CaptureController` over a `FakeCamera` session with a temp dir.

### 3.6 Picamera2 backend changes (`pifilm/capture/picamera.py`)

1. **Preview mode.** `Picamera2Camera(..., preview: tuple[int, int] | None = None)`. When
   set, the camera is configured with `create_preview_configuration(main={"size":
   preview, "format": "RGB888"}, raw={"size": binned})` where `binned` is the largest sensor
   mode with both dimensions ≤ half the native size (IMX477: 2028×1520; IMX708:
   2304×1296), and `read(full=False)` returns that frame with its metadata.
   `read(full=True)` calls `switch_mode_and_capture_request(self._still_config)` and then
   processes the request exactly as today (array check, RGB conversion, metadata, DNG),
   after which Picamera2 returns to the preview configuration. When `preview` is `None`,
   behaviour is byte-for-byte today's (still configuration only), so the headless Stick
   service is unaffected.
2. **Lock.** A `threading.Lock` around every `capture_request`/`switch_mode_and_capture_request`
   and `set_controls` call. The controller worker and the viewfinder thread share the camera.
3. **`set_ev(value: float)`** applies `{"ExposureValue": value}` at runtime; validated against
   `_EV_RANGE`. `FakeCamera` gets the same method (records the value).
4. **Multi-sensor.** `_NATIVE_SIZE` becomes `camera.sensor_resolution` read after opening;
   `_stream_info` and the array shape check use it. `DEFAULT_TUNING_FILE` becomes `None`,
   meaning libcamera's automatic tuning for the detected sensor; `--tuning-file` still
   overrides; the recorded `tuning_file` is `"auto:<Model>"` from `camera_properties["Model"]`
   when automatic. Autofocus controls are applied only when `"AfMode" in
   camera.camera_controls`; on a fixed-lens sensor the record says `"autofocus": "none"` and
   `--autofocus` other than the default is rejected with a clear error.
5. **Fake.** `tests/test_picamera.py`'s installer gains `sensor_resolution`,
   `camera_controls`, `camera_properties`, `create_preview_configuration`,
   `switch_mode_and_capture_request`, and a `fixed_lens=True` option.

### 3.7 Controller (`pifilm/capture/controller.py`)

`ControllerSnapshot` gains `completed_count: int`, incremented on every finished job
(complete or failed). The viewfinder compares it with the last value it saw; this is how a
Stick shot reaches the LCD without the controller knowing about displays.

### 3.8 `app.py`

- `--display {none,waveshare28,fake}` (default `none`), `--display-rotate {0,180}` (default 0),
  `--touch-debug` (print raw and mapped touch coordinates to stdout, for the orientation check).
  `--display` is Picamera2-only for now (the V4L2 backend has no preview-mode split and the
  meter needs libcamera metadata); it is added to `_reject_picamera2_only_flags` except for
  `--fake`, which is allowed so the loop can be exercised without hardware using
  `FakeCamera`; `--display fake` renders each frame to `OUT/viewfinder-last.png` instead
  of SPI and reads no touch. `--display waveshare28` sets `preview=(640, 480)`
  on the backend.
- Wiring: when a display is requested, a `CaptureController` is always created (shared with
  the remote server when `--remote-listen` is also given), the display and touch are opened,
  and `ViewfinderLoop.run()` runs on the main thread until SIGTERM/Ctrl-C. If opening the
  display or touch raises `DisplayError`, the error is printed and the process continues in
  the mode it would have used without `--display` (headless remote, or terminal), so the
  systemd service never crash-loops on a display fault.
- `--no-preview` continues to mean "no OpenCV window"; it does not disable the LCD.

### 3.9 Service and setup

- `deploy/pifilm-capture.service.example`: `ExecStart` gains `--display waveshare28`,
  with a comment that it is harmless on a Pi without the LCD (falls back), and `After=`
  is unchanged.
- `docs/lcd-viewfinder.md`: the wiring and reserved-pin tables from the hardware handover,
  `config.txt` requirements (`dtparam=spi=on`, `dtparam=i2c_arm=on`, no `fbtft` overlay),
  the `spi`/`i2c`/`gpio` groups for the service user, apt packages (`python3-spidev`,
  `python3-smbus2` or `python3-smbus`, `python3-gpiozero`, `python3-lgpio`; none via pip),
  the screen layout, how to read the meter, and the hardware acceptance checklist below.
  `docs/setup.md` gets a step pointing to it; README gets one row in the capture-modes table
  and one line under "From button press to displayed photo"; CLAUDE.md gets the package and
  the doc.
- IMX477 note in `docs/picamera2-bringup.md`: fixed lens, 4056×3040, 12-bit raw, tuning
  automatic, and that the DNG acceptance test applies unchanged: its red-is-red check is
  exactly what catches a wrong colour-filter-order tag on a new sensor.

## 4. Hardware acceptance checklist (user-run, recorded in the progress log)

1. `pifilm-capture --camera picamera2 --display waveshare28 --no-preview --out ~/Pictures/pifilm-lcd`
   from a terminal: live view within 5 s of start; frame rate ≥8 fps (the loop prints a
   one-line rate every 10 s to stdout).
2. `--display-rotate 0` vs `180`: image upright for the mounted cable; touch lands where
   the finger is (tap the four corners; the loop prints mapped coordinates with `--touch-debug`).
3. Shutter tap → busy dot → result → tap returns to live. `captures.jsonl` gains a record
   with `sensor_mode 4056x3040`, `tuning_file auto:imx477`, `autofocus none`.
4. Stick shot while the LCD is live: appears on both; LCD returns to live after 30 s untouched.
5. EV `+` three times: readout shows `+1.0`, live view brightens, and a shot's
   `camera_metadata.ExposureTime` is longer than at `0`.
6. Meter sanity: cover the lens → needle hard left, lux near 0; point at a lamp → `clip %`
   rises.
7. Service: `systemctl restart pifilm-capture` with the LCD → live view at boot;
   unplug the LCD's DC wire, restart → journal shows the `DisplayError` line and the Stick
   still captures.
8. IMX477 DNG opens per the existing DNG acceptance test.

## 5. Testing summary

- Unit (no hardware): RGB565 conversion, MADCTL per rotation, init byte sequence equals the
  vendor list; CST3530 decoding against vendor byte layouts, rotation mapping, tap
  detection; meter fields on synthetic frames and partial metadata; renderer dimensions
  and hit regions; loop transitions (shutter, EV clamp, review by tap, review by timeout,
  Stick completion via `completed_count`, failed job message, 5 display failures exit,
  camera error keeps going); backend preview mode (config choice for both sensors, lock held
  during switch, `set_ev` range, fixed-lens AF rejection, auto tuning recorded); app wiring
  (`--display` rejected on V4L2, fallback on `DisplayError`, `--display fake` writes a PNG).
- Existing suite unchanged and green; ruff clean.
- Hardware: checklist above.

## 6. Risks

- **Mode-switch cost.** `switch_mode_and_capture_request` costs roughly 0.3–0.6 s per shot on
  top of today's ~3.3 s grade. Acceptable; the alternative (a lores stream on the 12 MP still
  configuration) caps the viewfinder at the full-resolution sensor rate, 10 fps on the
  IMX477, and loads the ISP continuously.
- **Touch orientation** is unknown until tested; the `--display-rotate` flag and the
  `--touch-debug` printout exist for that.
- **Shared I2C bus.** The gauge poll (every 10 s) and touch reads use separate `SMBus`
  handles; each vendor transaction is a single kernel ioctl, and the CST3530's internal
  pointer is not disturbed by transactions to other addresses. Touch is polled every frame
  (see §3.2) rather than gated on `INT`, but each poll is only two short transactions, so
  bus traffic stays low.
- **GPIO permissions under systemd.** The service user must be in `spi`, `i2c`, `gpio`; the
  driver's error message names the group when it sees `PermissionError`.
- **CPU during grade.** The viewfinder keeps drawing during the 3 s grade; the frame period
  will stretch. Acceptable and visible as the busy dot.
