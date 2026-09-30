# LCD viewfinder

A Waveshare 2.8" Capacitive Touch LCD (V2, ST7789 + CST3530) wired to the Pi's
GPIO header, run by `pifilm-capture --display waveshare28` as a live viewfinder
with a light-meter readout and an on-screen shutter. Hardware acceptance on the
real panel has not been run yet; see the [checklist](#7-hardware-acceptance-checklist).

## 1. What it is

The LCD shows a live, ungraded feed from the camera with a light-meter-style
exposure readout, an on-screen shutter button and EV controls, sharing the same
`CaptureController` as the Stick — a shot from either trigger shows up on both.
The SPACE key is not available while the viewfinder runs: the loop owns the
process's main thread, so there is no terminal key reader.

There is no screenshot in this repo; run
`pifilm-capture --fake --display fake` (below) and open the PNG it writes to
see the exact layout without any hardware attached.

## 2. Wiring

Pins as used by `pifilm/display/st7789.py` and `pifilm/display/cst3530.py`.
**Waveshare's own diagrams label pins with BCM GPIO numbers, not physical
header position** — check the physical-pin column below before connecting
jumper wires by hand.

| Signal | BCM GPIO | Physical pin | Notes |
| --- | --- | --- | --- |
| MOSI | 10 | 19 | SPI0 |
| SCLK | 11 | 23 | SPI0 |
| CS (CE0) | 8 | 24 | SPI0 |
| MISO | — | — | not connected; the panel is write-only |
| DC | 25 | 22 | data/command select |
| RST (display) | 27 | 13 | display reset |
| BL | 18 | 12 | backlight, PWM |
| VCC | — | 1 | 3.3 V |
| GND | — | 6 | |
| TP_RST (touch) | 17 | 11 | touch controller reset |
| I2C1 SDA | 2 | 3 | shared with the X728 fuel gauge (`0x36`) and RTC (`0x68`) |
| I2C1 SCL | 3 | 5 | shared with the X728 fuel gauge (`0x36`) and RTC (`0x68`) |

The touch controller answers on I2C1 at address `0x58`. It is polled each
frame rather than gated on its `INT` line (see [Troubleshooting](#8-troubleshooting)
and the design note in the spec, §3.2) — no `INT` wiring is required.

If the Geekworm X728 UPS is fitted, its GPIO pins are reserved and must not be
reused by anything else on this board, reproduced from
[docs/x728-ups.md](x728-ups.md) section 3:

| BCM pin | Direction | Meaning | Used by |
| --- | --- | --- | --- |
| 5 | Pi input | Shutdown request from the shield. High pulse of 200 to 600 ms after a 1 to 2 s button hold means reboot; longer means power off | `xPWR.sh` |
| 12 | Pi output | "OS is up". Driven high at boot; the shield cuts power when it falls after halt | `xPWR.sh` |
| 26 | Pi output | Software shutdown request to the shield: pulse high for about 2 s, then low. The shield answers by raising pin 5, so the normal `poweroff` path runs | `xSoft.sh`, the low-battery script |
| 6 | Pi input | Power-loss detection. 0 = external power present, 1 = running on the cells | the capture service, optional alarm scripts |
| 20 | Pi output | Buzzer (v2.1 and later) | optional |
| 16 | Pi output | Battery charge enable (v2.5 and later, "advanced users only") | leave alone |

None of the LCD/touch pins above collide with this list.

## 3. Pi setup

1. **`config.txt`** (`/boot/firmware/config.txt`) needs both interfaces on, and
   must **not** carry a `dtoverlay=fbtft` line — this project talks to the
   panel directly over SPI from Python, not through a kernel framebuffer
   driver, and `fbtft` would fight over the same GPIO:

   ```text
   dtparam=spi=on
   dtparam=i2c_arm=on
   ```

   Reboot after editing.

2. **Packages, from apt only, never pip:**

   ```sh
   sudo apt install python3-spidev python3-smbus2 python3-gpiozero python3-lgpio
   ```

   If `python3-smbus2` is not packaged for your OS release, install
   `python3-smbus` instead. The import name differs: `python3-smbus2` provides
   `import smbus2` (what `pifilm/display/cst3530.py` imports), `python3-smbus`
   provides `import smbus`.

3. **Groups**, so the service user can open `/dev/spidev0.0` and `/dev/i2c-1`
   without root:

   ```sh
   sudo usermod -aG spi,i2c,gpio george
   ```

   Log out and back in (or reboot) — group membership only takes effect in a
   new login session.

4. **Verify:**

   ```sh
   ls -l /dev/spidev0.0 /dev/i2c-1
   sudo i2cdetect -y 1        # expect 58 (touch); 36 and 68 too if the X728 is fitted
   ```

## 4. Running

```sh
# Terminal test: live view, no window, Ctrl-C to stop
.venv/bin/pifilm-capture --camera picamera2 --display waveshare28 --no-preview --out ~/Pictures/pifilm-lcd

# If the image is upside down for how the cable is mounted
.venv/bin/pifilm-capture --camera picamera2 --display waveshare28 --display-rotate 180 --no-preview --out ~/Pictures/pifilm-lcd

# Orientation check: prints raw and mapped touch coordinates for each tap
.venv/bin/pifilm-capture --camera picamera2 --display waveshare28 --touch-debug --no-preview --out ~/Pictures/pifilm-lcd

# Idle dimming: halve the backlight after 30 s, never turn it off (defaults: 60 s and 300 s)
.venv/bin/pifilm-capture --camera picamera2 --display waveshare28 --display-dim-after 30 --display-off-after 0 --no-preview --out ~/Pictures/pifilm-lcd

# No hardware: writes each frame to OUT/viewfinder-last.png instead of driving SPI
.venv/bin/pifilm-capture --fake --display fake --no-preview --out /tmp/pifilm-viewfinder
open /tmp/pifilm-viewfinder/viewfinder-last.png   # see the layout without a panel
```

`--display` only works with the Picamera2 backend (or `--fake`); it needs the
preview-mode split and libcamera metadata that the V4L2 backend does not have.
It is not an error there, though: with `--camera v4l2`, with `--device`, or on
a Pi where Picamera2 is not installed, `pifilm-capture` prints

```text
warning: display unavailable (the V4L2 backend has no preview mode); continuing without it
```

and runs exactly as it would without the flag.

`deploy/pifilm-capture.service.example`'s `ExecStart` already includes
`--display waveshare28`; it is harmless on a Pi without the panel and on a Pi
with a USB camera — the service logs one `warning: display unavailable (...)`
line and keeps serving the Stick instead of crash-looping. The same is true
once the viewfinder is running: five consecutive SPI failures close the panel
and leave the remote API serving the Stick, rather than exiting for systemd to
restart.

## 5. The screen

| Region | Shows | Tap does |
| --- | --- | --- |
| Live image | Full-screen letterboxed, ungraded ISP preview | — |
| Meter bar (bottom strip, translucent black) | Shutter, ISO, EV compensation, lux, clip % | — |
| Needle | Deviation from mid-grey in stops, −3 to +3, amber marker | — |
| Focus bar (vertical gauge, left edge, labelled `F`) | Absolute sharpness of the central quarter of the frame, 0 (far out of focus, or nothing to focus on) to full; amber tick marks the best level of the last few seconds | — |
| Shutter button (circle, right edge) | White ring, filled centre | Submits a capture through the shared controller; the processing screen holds until it finishes |
| EV `−` / EV `+` buttons (bar's left/right ends) | `-` / `+` labels | Adjusts exposure compensation by 1/3 stop, clamped to ±2; resets to `0` every time `pifilm-capture` restarts. The graded file keeps it too (normalisation used to cancel it until 2026-09-24), and it is logged as `ev_comp` in `captures.jsonl` |
| Shutter `+` / `−` buttons (right edge, above and below the shutter button) | `+` and `-` | Shutter priority: `+` one 1/3 stop faster, `−` slower, 1/2000 to 1 s; slower than 1 s returns to auto (`A`). The first tap from auto starts at the speed auto-exposure is using. ISO stays automatic; EV still works. Picamera2 only; resets to auto on restart |
| Shutter readout `S 1/250` | `S` prefix when the shutter is fixed; otherwise the metered speed | — |
| `ISO MAX` (amber, in the readout) | Gain is at the sensor's maximum: the fixed shutter is too fast for the light and the photo will be dark | — |
| Processing screen (TV colour bars, `Processing photo...`) | Replaces the live view while any capture — from this screen or the Stick — is being graded, roughly 3 s on the Pi 4; the same bars the Stick and the OpenCV window show, dimmed to 55 % on the panel (`PROCESSING_DIM`) so they do not glare in a dark room. The camera is not read during it | — |
| Battery badge (top-right) | `NN%` or `AC NN%` from the X728 gauge; absent without `--ups x728` | — |
| `2x` toggle (pill, top-left) | Outlined `2x` when off; amber `2x 0/2` / `2x 1/2` when on | Turns double exposure on or off. Off at `1/2` discards the pending first exposure (its original stays on disk) |
| Review screen | Full-screen graded result, one-line caption (`shutter  ISO NNN  EV ±N.N`), "tap to continue" hint | Any tap, or 30 s untouched, returns to live view |

### Idle dimming

Left untouched, the panel saves itself and the battery:

| Untouched for | Screen | A tap |
| --- | --- | --- |
| under 1 minute | full brightness (backlight 80 %) | acts normally |
| 1 minute (`--display-dim-after`) | half brightness, live view still running | restores full brightness **and** acts (the buttons are visible) |
| 5 minutes (`--display-off-after`) | backlight off; the camera preview is neither read nor drawn | **only wakes** the screen; it never fires the shutter, since you cannot see what you touch |

Any capture counts as activity: while one is being graded the screen stays
lit, and a finished shot, from the LCD or the Stick, wakes it into the review.
`0` disables either step. `--display fake` has no backlight and is never dimmed.

A tap that lands during the roughly one second of still acquisition is
**dropped, not queued**: the camera lock is held for the whole capture request,
so the loop is not polling touch at all during it, and the controller is busy
anyway (it runs one job at a time). Wait for the colour bars to clear.

### Double exposure

With `2x` on, every two shots become one picture, the way two exposures on one frame
of colour negative film do. Light adds: a dark area in one frame lets the other show
through, and bright areas stack. Any trigger counts: the shutter button, the Stick,
or both mixed.

1. **Exposure 1:** colour bars labelled `Exposure 1 of 2...`, then `Exposure 1/2` for
   about a second, then the live view with the badge at `1/2`. Only the original is
   saved; the Stick shows an "Exposure 1/2" card.
2. **Exposure 2:** colour bars labelled `Developing double exposure...`, then the
   review screen with the composite (caption starts `2x` and ends with both
   exposures' EVs, `EV e1/e2`). The badge returns to `0/2` and the mode stays on.

Files: each exposure keeps its own `HHMMSS_original.jpg` (and `.dng`). The composite
is `HHMMSS_double_graded.jpg`, named after exposure 2. Its `captures.jsonl` line has a
`double` block naming both originals (as paths relative to the output folder, such as
`2026-09-30/120000_original.jpg`, since a pair can straddle midnight), their EVs and the
method (`linear_mean`). Exposure compensation per frame is the film shooter's control:
shoot the frame you want to recede at −1 EV. Do not under-expose both frames as you
would on film: the one-stop reduction is already applied, and normalisation cancels an
offset common to both, so only the difference between the two EVs matters. A tap ends
the `Exposure 1/2` notice early. The mode is not remembered across restarts. A camera
error on exposure 2 keeps `1/2`, so just shoot again. See [how it works](how-it-works.md#double-exposure)
for why the frames are added before grading.

### Shutter priority

Tap the shutter `+` / `−` buttons to fix the shutter speed; the camera keeps choosing
the ISO to expose correctly, and EV compensation still shifts the result. Use a fast
speed (1/500 and up) to freeze motion and a slow one (1/15 and down) to blur it; the
readout shows `S` in front of a fixed speed. Keep tapping `−` past 1 s to return to
full auto.

- At long speeds the live view slows to match: at 1/4 s it shows about 4 frames a
  second. That is the exposure, not a fault.
- `ISO MAX` in amber means the camera has run out of gain for this speed. The photo
  will come out dark and grainy (it is graded like any dark frame); pick a slower
  speed or add light.
- Each photo's `captures.jsonl` line records `shutter_us` (null on auto); the real
  exposure time and gain are in `camera_metadata`.
- At start-up the journal names how the shutter is fixed:
  `picamera2: shutter priority via ExposureTimeMode` on current libcamera, or
  `... ExposureTime (legacy libcamera)` on older stacks.

## 6. Reading the meter

- **Needle near the centre mark** — auto-exposure has converged on this scene.
- **Needle hard left or hard right** — auto-exposure has run out of range (more
  than 3 stops from mid-grey); dial in EV compensation or expect the shot to be
  under/overexposed.
- **`clip %`** — the fraction of pixels with any channel at or above 254; this
  is the same clipped-highlight number Phase 6's normalisation cares about
  ([docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md](superpowers/plans/2026-09-13-rpi4-imx708-progress.md)).
  Aim low outdoors, where clipping is easy to get.
- **`lux`** — libcamera's own scene-brightness estimate from the sensor, not an
  independently measured value.

### Focusing by hand

The IMX477 has no autofocus, so the lens ring is the only focus control and the
**focus bar** down the left edge is how the screen says which way to turn it.
It measures how much fine edge contrast there is over the central quarter of
the frame (the middle half of the width and of the height), as a share of all
the edge contrast there. Because it is a ratio, the scene's own brightness and
contrast cancel out: a blurred frame reads low on its own, and a dim, flat
subject in focus reads about as high as a bright one.

- **A full bar means sharp; a short bar means soft**, with no need to have seen
  a sharp frame first. Far out of focus reads close to zero.
- **The amber tick is the best level of the last few seconds.** Rack through
  focus, then turn back until the green reaches the tick again.
- **The tick fades** (it halves in about three seconds), so it lets go of an
  old subject once the camera points elsewhere.
- **At high ISO in a dim room the bar tops out lower**, typically two-thirds to
  three-quarters at best focus: the noise allowance takes some of the reading.
  The peak is still the peak, so focus to the tick.
- **It needs edges in the middle of the frame.** A blank wall, clear sky or an
  evenly lit sheet of paper gives it nothing to measure and the bar sits at
  zero however well focused the lens is. Aim the centre at an edge, print or
  fabric.

## 7. Hardware acceptance checklist

Not yet run on real hardware. Record results in the progress log
([docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md](superpowers/plans/2026-09-13-rpi4-imx708-progress.md)),
per spec §4:

1. `pifilm-capture --camera picamera2 --display waveshare28 --no-preview --out ~/Pictures/pifilm-lcd`
   from a terminal: live view within 5 s of start; frame rate ≥8 fps (the loop
   prints one line every 10 s to stdout).
2. `--display-rotate 0` vs `180`: image upright for the mounted cable; touch
   lands where the finger is (tap the four corners; `--touch-debug` prints the
   mapped coordinates).
3. Shutter tap → colour bars → result → tap returns to live. `captures.jsonl`
   gains a record with `sensor_mode 4056x3040`, `tuning_file auto:imx477`,
   `autofocus none`.
4. Stick shot while the LCD is live: appears on both; the LCD returns to live
   after 30 s untouched.
5. EV `+` three times: readout shows `+1.0`, the live view brightens, and a
   shot's `camera_metadata.ExposureTime` is longer than at `0`.
6. Meter sanity: cover the lens → needle hard left, lux near 0; point at a
   lamp → `clip %` rises.
7. Service: `systemctl restart pifilm-capture` with the LCD → live view at
   boot. Then provoke a fault that the *open* path can actually detect, and
   restart again: the journal must show one `warning: display unavailable (...)`
   line and the Stick must still capture. Either
   - unplug the panel's ribbon cable entirely, so the touch probe at `0x58`
     gets no answer (`touch controller at 0x58 not answering on /dev/i2c-1`), or
   - `sudo gpasswd -d george spi` and reboot, so `/dev/spidev0.0` cannot be
     opened (`no permission for /dev/spidev0.0`); put the group back afterwards.

   Unplugging only the DC wire does **not** test this: SPI writes are not
   acknowledged, so the driver cannot tell a blank panel from a working one and
   the process runs on happily, drawing into the void.
8. IMX477 DNG opens per the existing [DNG acceptance test](picamera2-bringup.md#dng-acceptance-test).
9. Focus bar: with the IMX477, turn the focus ring slowly through best focus on a
   textured subject in the middle of the frame. From a lens left far out of focus
   the bar must start near zero (the first version read full here). At best focus
   it should be well above half; it must fall away on **both** sides of the peak,
   with the amber tick left at the best level.
10. Idle dimming: leave the screen untouched: at 1 minute it halves in brightness
    and the live view keeps moving; at 5 minutes it goes dark. A tap on the dark
    screen lights it without taking a photo (no new `captures.jsonl` line); a
    Stick shot while dark wakes it into the review.
11. Double exposure: tap `2x` → badge `2x 0/2` in amber. Shoot a dark silhouette
    against a bright window, then a textured subject: `Exposure 1/2` shows briefly and
    the badge reads `1/2`; the second shot reviews a composite in which the texture
    fills the silhouette. The day folder gains two originals and one
    `_double_graded.jpg`, and no `_graded.jpg` for exposure 1. Repeat with the Stick
    as the trigger: it shows the `Exposure 1/2` card, then the composite. Tap `2x` at
    `1/2`: the badge goes to the outlined `2x`. `systemctl restart pifilm-capture`:
    the mode comes back off. Exposure 2 on the IMX477 finishes within a few seconds of
    a single shot (colour bars not held noticeably longer).

12. Shutter priority: in a dim room tap `+` up to 1/1000: the readout shows
    `S 1/1000` and amber `ISO MAX`, and the shot is darker. Tap `−` down to 1/4 and pan
    across a scene: the live view drops to about 4 fps and a shot shows motion blur.
    Each shot's `camera_metadata.ExposureTime` is within a few percent of the chosen
    speed, and its `shutter_us` matches. After taking a shot at a fixed speed, the live
    view keeps it (the readout still shows `S`, and the slow live view or blur persists).
    Keep tapping `−` past 1 s: the readout loses the `S` and the live view returns to
    full rate. After returning to `A`, a shot's `camera_metadata.ExposureTime` changes
    with the scene again (point at a lamp, then a dark corner). The journal line at
    start-up names the control path.

## 8. Troubleshooting

Every line below is printed by `pifilm-capture` as
`warning: display unavailable (<message>); continuing without it`, and the
process falls back to the mode it would have run in without `--display`.

| Message contains | Cause | Fix |
| --- | --- | --- |
| `display libraries missing` | `python3-spidev` or `python3-gpiozero` not installed | Install the apt packages in [Pi setup](#3-pi-setup) |
| `touch libraries missing` | `python3-smbus2` or `python3-gpiozero` not installed | Same |
| `no permission for /dev/spidev0.0` | Service user not in the `spi` group | `sudo usermod -aG spi george`; log out and in |
| `no permission for /dev/i2c-1` | Service user not in the `i2c` group | `sudo usermod -aG i2c george`; log out and in |
| `cannot open /dev/spidev0.0` (asks `is dtparam=spi=on set?`) | SPI not enabled | Add `dtparam=spi=on` to `config.txt`, reboot |
| `cannot open /dev/i2c-1` (asks `is dtparam=i2c_arm=on set?`) | I2C not enabled | Add `dtparam=i2c_arm=on` to `config.txt`, reboot |
| `cannot claim display GPIO 25/27/18` | GPIO already held — usually a leftover `dtoverlay=fbtft`, or another process | Remove any `dtoverlay=fbtft` line from `config.txt`; check the `gpio` group |
| `cannot reset the touch controller on GPIO 17` | GPIO 17 busy, or a wiring fault on `TP_RST` | Check the `TP_RST` connection; confirm nothing else claims GPIO 17 |
| `touch controller at 0x58 not answering on /dev/i2c-1` | Nothing responded to the probe read at open — usually the ribbon cable | Check the ribbon cable and `TP_RST` wiring, and that the panel has power; confirm with `sudo i2cdetect -y 1` showing `58` |
| `the V4L2 backend has no preview mode` | The camera is a USB/UVC one; the viewfinder needs Picamera2's preview-mode split and libcamera metadata | Nothing to fix unless a Pi camera is intended: check `--camera`/`--device` and that `python3-picamera2` is installed |
