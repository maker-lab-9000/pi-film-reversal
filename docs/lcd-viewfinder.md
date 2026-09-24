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
| Focus bar (vertical gauge, left edge, labelled `F`) | Sharpness of the central quarter of the frame, as a fraction of the recently seen best; amber tick at the top is the remembered peak | — |
| Shutter button (circle, right edge) | White ring, filled centre | Submits a capture through the shared controller; the processing screen holds until it finishes |
| EV `−` / EV `+` buttons (bar's left/right ends) | `-` / `+` labels | Adjusts exposure compensation by 1/3 stop, clamped to ±2; resets to `0` every time `pifilm-capture` restarts |
| Processing screen (TV colour bars, `Processing photo...`) | Replaces the live view while any capture — from this screen or the Stick — is being graded, roughly 3 s on the Pi 4; the same bars the Stick and the OpenCV window show. The camera is not read during it | — |
| Battery badge (top-right) | `NN%` or `AC NN%` from the X728 gauge; absent without `--ups x728` | — |
| Review screen | Full-screen graded result, one-line caption (`shutter  ISO NNN  EV ±N.N`), "tap to continue" hint | Any tap, or 30 s untouched, returns to live view |

A tap that lands during the roughly one second of still acquisition is
**dropped, not queued**: the camera lock is held for the whole capture request,
so the loop is not polling touch at all during it, and the controller is busy
anyway (it runs one job at a time). Wait for the colour bars to clear.

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
It measures contrast (Laplacian variance) over the central quarter of the frame
— the middle half of the width and of the height — and shows it as a fraction
of the sharpest thing seen in the last few seconds.

- **Turn the ring until the green reaches the amber tick at the top.** The tick
  is the best focus the bar remembers.
- **Racking past best focus drops the bar**, immediately and on both sides of
  it, which is what makes the peak findable: go past, come back.
- **The mark decays** — the remembered peak halves every second — so pointing
  the camera at a new subject lets the bar reach the top again within a few
  seconds rather than leaving it pinned to an old, higher-contrast scene.
- **It is a contrast measure, so it needs texture in the middle of the frame.**
  A blank wall, clear sky or an evenly lit sheet of paper gives it nothing to
  work with and the bar sits at zero however well focused the lens is. Aim the
  centre at an edge, print or fabric.

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
   textured subject in the middle of the frame. The bar must rise to the amber tick
   at the peak and fall away on **both** sides of it, and settle back to full within
   a few seconds of stopping.

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
