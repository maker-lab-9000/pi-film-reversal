# LCD viewfinder

A touch LCD on the Pi, run by `pifilm-capture --display ...` as a live viewfinder
with a light-meter readout and an on-screen shutter. Two panels are supported:

| Panel | Flag | Connection | Status |
| --- | --- | --- | --- |
| Waveshare 3.5" DSI LCD (E), 640×480, Goodix touch | `--display waveshare35dsi` | DSI ribbon cable only | The default. Setup in [section 9](#9-the-35-dsi-panel) |
| Waveshare 2.8" Capacitive Touch LCD (V2), 320×240, ST7789 + CST3530 | `--display waveshare28` | SPI, I2C and four GPIO pins on the header | Fallback. Sections 2, 3 and 8 describe it |

Sections 5 and 6 (the screen, the meter) apply to both: the DSI panel shows the
same layout at twice the size. Where the two differ, section 9 says how. The one
exception inside section 5 is [Idle dimming](#idle-dimming): the DSI panel has no
brightness control, so it has no one-minute dim step, only the five-minute off
step (see [9.3](#93-what-differs-from-the-28-panel)).
The commands in section 4 and the checklist in section 7 are written for the
2.8" panel; sections 9.2 and 9.6 are their equivalents for the DSI panel.

## 1. What it is

The LCD shows a live, ungraded feed from the camera with a light-meter-style
exposure meter, an on-screen shutter button and three manual controls (shutter,
ISO, white balance), sharing the same
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
`--display waveshare35dsi`; it is harmless on a Pi without the panel and on a Pi
with a USB camera — the service logs one `warning: display unavailable (...)`
line and keeps serving the Stick instead of crash-looping. The same is true
once the viewfinder is running: five consecutive display failures close the panel
and leave the remote API serving the Stick, rather than exiting for systemd to
restart.

## 5. The screen

The live screen is drawn in the style of an SLR finder: the image fills the screen,
and every read-out sits in a solid black strip along the bottom, amber on black.
There is no screenshot of a real panel in the repo; the approved design mock-up is
[2026-10-11-manual-controls-ui.png](superpowers/mockups/2026-10-11-manual-controls-ui.png)
(left: all automatic; right: full manual, `+` pressed). The layout is in 320×240
base units, drawn at 1× on the 2.8" panel and 2× on the DSI panel. The
type is Barlow Condensed (SIL Open Font Licence 1.1; the licence text is
`pifilm/data/fonts/OFL.txt`) on the live screen only; the review, processing and
message screens keep the default font.

| Region | Shows | Tap does |
| --- | --- | --- |
| Live image | Full-screen letterboxed, ungraded ISP preview | — |
| Strip cells `SHUTTER`, `ISO`, `WB` (bottom left, three columns) | Label, then the value. The selected cell is amber (label and value, with a white `AUTO` tag if automatic); the others are grey with a dim amber value. An automatic control shows `AUTO` beside its name and the live metered value (`1/125`, `400`, `5200K`); a fixed one shows only the value. A control the camera cannot change is drawn all grey | Selects the cell. `SHUTTER` is selected at start-up and the selection stays until another cell is tapped |
| `MENU` key (right edge, under `-`, just above the strip) | An outlined pill in the keys' style; filled while a finger is on it. Absent while the colour bars show | Opens the [settings menu](#settings-menu). Its hit box (x 264 to 310, y 190 to 214) starts one unit under `-`'s hit circle and reaches into the strip's meter part, which has no tap action, so no tap that worked before changes meaning; the right edge above it holds only the shutter and `+`/`-` |
| `+` and `-` keys (right edge, below the shutter button) | Outlined amber circles; filled amber while a finger is on them | Steps the selected control; see [Manual controls](#manual-controls) |
| Shutter button (top right, ring with a white disc) | Filled amber while a finger is on it | Submits a capture through the shared controller; the processing screen holds until it finishes. It is at the top and `+`/`-` are low on the same edge, 31 units apart between their hit circles, so a slip while stepping a value does not take a photo |
| Exposure meter (strip, right) | 13 dots from −3 to +3 stops in half stops; the dot nearest the measured deviation from mid-grey is lit amber. Small grey text above it: `NNN LX   CLIP NN%` | — |
| Focus gauge (vertical bar, left edge) | Absolute sharpness of the central quarter of the frame, 0 (far out of focus, or nothing to focus on) to full; an amber tick marks the best level of the last few seconds | — |
| `2x` toggle (outlined tag, top left) | Outlined `2x` when off; amber `2x 0/2` / `2x 1/2` when on | Turns double exposure on or off. Off at `1/2` discards the pending first exposure (its original stays on disk) |
| Battery tag (outlined, top left, right of `2x`) | `NN%` or `AC NN%` from the X728 gauge; absent without `--ups x728` | — |
| Developing count (amber number on a dark patch, right of the battery tag) | How many photographs are still being graded behind the shutter; see [Developing count](#developing-count). Absent at zero and with `--grade inline` | — |
| `TUNING` tag (amber word on a dark patch, last in the top-left row) | The camera's libcamera tuning differs from the one the look was trained under; see [Tuning warning](#tuning-warning). Absent otherwise | — |
| `PHOTOS NN%` / `SD NN%` tags (amber on a dark patch, top, between the top-left row and the shutter button) | The photo volume or the system SD card is at or above `--disk-warn-percent` (90 by default); see [Disk-space warning](#disk-space-warning). Absent otherwise | — |
| `IMG` tag (amber on a dark patch, first in the same place as the disk tags) | Sharpness, contrast or saturation is not 1.0; see [Image page](#settings-menu). Absent otherwise | — |
| Processing screen (TV colour bars, `Processing photo...`) | Replaces the live view while any capture — from this screen or the Stick — holds the camera: until the original is saved and the quick grade is made, or, with `--grade inline`, until the full grade is saved (roughly 3 s on the Pi 4). It stays longer when three grades are already waiting; the same bars the Stick and the OpenCV window show, dimmed to 55 % on the panel (`PROCESSING_DIM`) so they do not glare in a dark room. The camera is not read during it | — |
| Review screen | Full-screen quick grade (the frame reduced by a whole factor to about 1,000 pixels on the long side, four times each way for the IMX477, graded without grain; the graded file itself with `--grade inline`), one-line caption (`1/250  ISO 400`, starting `2x` for a double exposure), "tap to continue" hint. The caption's ISO is analogue × digital gain as the camera measured it, so a shot at a fixed ISO 100 can read slightly higher (for example `ISO 120`) | Any tap, or 30 s untouched, returns to live view |

Press feedback: `+`, `-` and the shutter button light while a finger is on them, not
only on release, and a tap shorter than one frame still lights its key for 0.15 s.
The action itself fires on release. A press of up to 1.5 s counts (a lit key
invites a longer press than a blind tap); one held longer than that lights the key
but changes nothing, so a resting thumb never steps a value. `+` and `-` light even
when they will do nothing (the top of a scale).

While the processing screen is showing, taps on the cells, on `+`/`-` and on `2x`
are ignored: the colour bars hide them, so nothing is selected or stepped. A tap on
the shutter button's place still reaches the controller, which answers busy.

### Developing count

The full-resolution grade of a photograph runs behind the shutter, on a queue
(`--grade background`, the default). While any grade is unfinished, an amber number
beside the battery tag says how many photographs are still developing. It counts the
grade that is running and the ones waiting. It disappears at zero.

- At zero nothing is being graded and nothing is waiting, so it is safe to switch off.
  Every graded file and its record are then on disk, unless a grade failed: a failed
  grade is reported in the log and its entry is kept for the next start, or set aside
  as `.failed` after two failures.
- The queue holds three waiting grades. When it is full the next capture waits for
  a place, with the processing screen showing. That is why the shutter may wait
  although the last picture has already been reviewed.
- Just after a restart the number can be higher than the queue allows: grades left
  unfinished by a power cut are counted too, and are finished first. With many of
  them, three new photographs can be taken at once and a fourth waits until the
  recovered grades are done.
- Numbers above 99 are drawn as `99`.

The review screen and the Stick show the quick grade, not the final file: it is
reduced and has no grain, and its tone can differ slightly from the graded file.

### Tuning warning

A trained look is fitted to the colours of one libcamera tuning file. At start-up
the service compares the fingerprint of the tuning the camera opened with against
the one stamped in the look. When they differ, an amber `TUNING` tag shows at the
end of the top-left row, after the battery tag and the developing count. It stays
for the whole run, because the tuning cannot change while the service runs.

- The camera still works. Photographs are taken and graded as usual, but their
  colours may be off, and their records say `tuning_matches: false`.
- The journal has the line `tuning: DIFFERS from the look: ...`, naming the
  tuning file in use.
- The usual cause is an OS or libcamera update, or a `--tuning-file` that is not
  the one the look was trained under. What to do is in
  [known issues](known-issues.md#a-libcamera-update-could-silently-retune-a-deployed-trained-lut).

There is no tag when they match, when the look has no fingerprint (the bundled
starter, or a trained look not yet stamped with `pifilm-tuning --stamp`), or with
a USB camera.

### Disk-space warning

The service checks two filesystems once a minute (and once at start-up): the photo
folder (`--out`, normally the [photo volume](photo-volume.md)) and the system SD
card at `/`. When either is at or above `--disk-warn-percent` (90 by default, 1 to
100), an amber tag shows at the top of the screen, after the top-left row and short
of the shutter button: `PHOTOS 92%`, `SD 91%`, or both, photos first.

- The percentage is the one `df -h` prints as Use%, rounded up the same way, so
  `df -h ~/Pictures/pifilm /` over SSH shows the same numbers.
- Photographs are still taken. The tag is a reminder to move photographs off the
  Pi or to grow the volume ([Check and grow](photo-volume.md#check-and-grow)).
- The journal has one line when a filesystem crosses the threshold, for example
  `disk: photos 92% full (2.4 GB free)`, and one when it drops below it again,
  `disk: photos back to 85% full (4.5 GB free)`. GB here is df's G (2^30 bytes).
- When the photo folder is on the root filesystem (no separate volume), there is
  one filesystem and one tag, `PHOTOS`.
- A path that cannot be measured is left out after one `disk: cannot measure ...`
  line, and tried again at the next check.

The check runs with every display mode, headless too, for the journal lines.

### Settings menu

A tap on `MENU` opens a black page in the live view's fonts and colours, one setting
per row: a grey label on the left, the value in white on the right. A tap anywhere
on a row steps it to its next value, wrapping round; a dimmer label marks a row that
only shows something and ignores taps. `DONE`, drawn where the shutter button is,
returns to the live view. Nothing in it needs sudo.

| Row | Values | Shown when | Effect |
| --- | --- | --- | --- |
| `SAVE DNG` | `ON` / `OFF` | Picamera2 | from the next photograph |
| `HALATION` | `ON` / `OFF`; a dim `-` when the look has no halation | Picamera2 | from the next photograph |
| `AE CONSTRAINT` | `NORMAL` / `HIGHLIGHT` / `SHADOWS` | the camera reports `AeConstraintMode`, and no `--ae-lock` | at once |
| `AE METERING` | `CENTRE` / `SPOT` / `MATRIX` | the camera reports `AeMeteringMode`, and no `--ae-lock` | at once |
| `IMAGE` | `>`; a dim `-` when the camera can change none of the three | a settings file (always on Picamera2; dim under `--fake`) | opens the image page |
| `UPLOAD` | see below | `--nextcloud-env` given | starts a Nextcloud sync |
| `ETH` | the Ethernet IPv4 address, or `-` | always | read-only |
| `WLAN` | the Wi-Fi IPv4 address (the hotspot's own `10.42.0.1` included), or `-` | always | read-only |
| `LOOK` | the `--artifacts` folder's name (for example `imx477-2026-09-v4-dark-hp5-tile`), cut with `...` if too wide; `starter (bundled)` for the bundled look | always | read-only |
| `RESET` | `TAP TWICE` | a camera with setting rows (not `--fake`) | see below |

The Pi's libcamera does not report the two AE modes in its frame metadata, so a
record's `ae_constraint` / `ae_metering` are what pifilm sent, and each change in the
menu logs one line such as `settings: ae-metering matrix`. A dark scene at the
exposure limits (the longest shutter and the top gain) looks the same in every mode.

A row the camera cannot use is left out and the rows below move up. A USB camera
(V4L2) runs without the display, so it has no menu at all; under `--fake` only a dim
`IMAGE`, `UPLOAD`, the addresses and `LOOK` are there.

**Image page.** `IMAGE` opens a second page titled IMAGE with the camera's own
image controls, which libcamera applies before pifilm's look:

| Row | Values | Effect |
| --- | --- | --- |
| `SHARPNESS` | `0.5` / `0.75` / `1.0` / `1.25` / `1.5` | at once, live view and photographs |
| `CONTRAST` | the same | at once |
| `SATURATION` | the same | at once |
| `RESET` | `TAP TWICE` | puts only these three back to 1.0 |
| `BACK` | | back to the first page |

A tap steps to the next value; `1.5` wraps to `0.5`. 1.0 is the camera's normal
rendering. `DONE` closes the menu from this page too, and the menu always opens on
the first page. A control the camera does not report is dim with `-`. Each
photograph's record has `sharpness`, `contrast` and `saturation` right after
`ae_metering`, the values it was taken with. While any of the three is not 1.0 the
live view shows an amber `IMG` tag.

Every look so far was trained on photographs taken at 1.0. At another value the
look grades a picture it was not trained on, so the result can differ from the
look's examples. Put them back to 1.0 for photographs you want the look to grade as
trained, and for photographs meant as training sources.

**Halation.** `OFF` leaves the look's halation out of the photographs that follow,
`ON` puts it back. A photograph keeps the value it was taken with, even if its
grade runs later or after a power cut. For a double exposure the second shutter
decides. The live view never shows halation either way. The record of a photograph
taken with it off says `"halation": "off"` where `halation_sha1` would be, and
`pifilm-process` then regrades it without halation too. With a look that has no
halation the row shows `-` and ignores taps, and records are as before.

**Remembered.** The seven settings are written to `settings.json` beside the
pending-grades directory (`~/.local/state/pifilm/settings.json` by default, the
parent of `--state-dir`) after every change, atomically, and read at the next
start before the camera opens. A saved value wins over the unit's flag
(`--no-dng`, `--ae-constraint`, `--ae-metering`; halation has no flag and the unit
says on, and sharpness, contrast and saturation have none and the unit says 1.0), and the journal then says so in one line, for example
`settings: ae-constraint shadows (saved in the menu; the unit says ae-constraint highlight)`.
A file that cannot be read, is not JSON or has another version is ignored with one
line and left in place; an unknown or invalid key is ignored with one line. Neither
stops the service. An older build ignores the three image keys with one line each and
shoots at 1.0. Only these seven settings are stored; every other flag, the
screen's rotation included, still comes from the unit alone.

**Reset.** The first tap on `RESET` changes it to `TAP AGAIN` for 3 s. A second tap
in that time puts every setting back to what the unit's flags say (applied at once),
deletes `settings.json` and logs `settings: reset to the unit's flags`.

**Upload.** With `--nextcloud-env PATH` the `UPLOAD` row runs the
[Nextcloud sync](nextcloud-sync.md#from-the-viewfinders-menu) once, as a child
process; the service never reads that file itself. Each line the sync prints is in
the journal as `nextcloud: <line>`. A tap while a sync runs does nothing.

| State | Row value |
| --- | --- |
| No upload yet in this run | `TAP TO UPLOAD` |
| Running | `UPLOADING 0:42` (elapsed time) |
| Finished, files copied | `DONE 14:05` |
| Finished, nothing done | `SKIPPED: NO NETWORK`, `SKIPPED: NO NEXTCLOUD`, `SKIPPED: BUSY` (another sync, such as the timer's, was running), or `SKIPPED` |
| The sync failed | `FAILED (EXIT 4)` |

A finished state stays until the next tap or restart. Stopping the service during
an upload stops the sync and its rclone (SIGTERM to the sync's process group, then
SIGKILL after 5 s); rclone copies each file whole, so the next run carries on.

**While the menu is open** the camera is not read, as for a review. The GPIO button
and the Stick still take photographs; a shot finished then does not open the review,
and the menu stays. The developing count and the warnings return with the live view.
Idle dimming and screen-off apply as in the live view, and a tap in the menu counts
as activity.

**The addresses.** The `ETH` and `WLAN` rows are how to find the Pi for SSH without
a router's client list. They are read at start-up and every 10 s, so a cable plugged
in shows within about 10 s; only `eth0` and `wlan0`, IPv4, an interface counting
while the kernel reports it up and it has an address. The journal has one line when
an address appears or changes, `network: wlan0 192.168.178.56`, and one when it
goes, `network: eth0 gone`, in every display mode, headless too. The live view no
longer shows them.

### Idle dimming

Left untouched, the panel saves itself and the battery:

| Untouched for | Screen | A tap |
| --- | --- | --- |
| under 1 minute | full brightness (backlight 80 %) | acts normally |
| 1 minute (`--display-dim-after`) | half brightness, live view still running | restores full brightness **and** acts (the buttons are visible) |
| 5 minutes (`--display-off-after`) | backlight off; the camera preview is neither read nor drawn | **only wakes** the screen; it never fires the shutter, since you cannot see what you touch |

The 3.5" DSI panel has no dim step: it stays at full brightness until the
five-minute row, where its output is powered down (see
[9.3](#93-what-differs-from-the-28-panel)).

Any capture counts as activity: while one is being graded the screen stays
lit, and a finished shot, from the LCD or the Stick, wakes it into the review.
`0` disables either step. `--display fake` has no backlight and is never dimmed.

Touch is polled on its own thread, about 50 times a second, and finished taps are
queued, so a tap is never missed because the screen was busy. The camera's preview
frames are read on a second thread, so the loop never waits for the camera: about ten
times a second it takes one tap off the queue, acts on it and redraws the screen with
the most recent frame, whatever the shutter speed. The colour bars therefore appear on
the loop's next turn after a shutter tap, without waiting for the still. While they
are up a shutter tap is answered busy by the controller (it runs one job at a time)
and takes no second photo, and taps on the cells, `+`/`-` and `2x` are ignored as
described above. Wait for the colour bars to clear.

### Double exposure

With `2x` on, every two shots become one picture, the way two exposures on one frame
of colour negative film do. Light adds: a dark area in one frame lets the other show
through, and bright areas stack. Any trigger counts: the shutter button, the Stick,
or both mixed.

1. **Exposure 1:** colour bars labelled `Exposure 1 of 2...`, then `Exposure 1/2` for
   about a second, then the live view with the badge at `1/2`. Only the original is
   saved; the Stick shows an "Exposure 1/2" card.
2. **Exposure 2:** colour bars labelled `Developing double exposure...`, then the
   review screen with the composite (caption `2x  shutter  ISO NNN`, from the second
   exposure's metadata). The badge returns to `0/2` and the mode stays on.

Files: each exposure keeps its own `HHMMSS_original.jpg` (and `.dng`). The composite
is `HHMMSS_double_graded.jpg`, named after exposure 2. Its `captures.jsonl` line has a
`double` block naming both originals (as paths relative to the output folder, such as
`2026-09-30/120000_original.jpg`, since a pair can straddle midnight), their `ev_comp`
values (always 0 since the EV buttons were removed on 2026-10-11) and the method
(`linear_mean`). Do not under-expose both frames as you would on film: the one-stop
reduction is already applied, and normalisation cancels an offset common to both. To
make one frame recede, fix both the shutter and the ISO (full manual), then step one
of them down for that frame: a faster shutter or a lower ISO. Fixing only one does
nothing, because auto-exposure compensates with the other. A tap ends
the `Exposure 1/2` notice early. The mode is not remembered across restarts. A camera
error on exposure 2 keeps `1/2`, so just shoot again. See [how it works](how-it-works.md#double-exposure)
for why the frames are added before grading.

### Manual controls

Three controls, each of which can be automatic or fixed: shutter speed, ISO and white
balance. Tap a strip cell to select it; one `+`/`-` pair then changes whichever cell
is selected, and only that one.

| Control | Steps | `+` | `-` |
| --- | --- | --- | --- |
| `SHUTTER` | 1/3-stop steps, 1/2000 s to 1 s (`pifilm/display/shutter.py`) | Faster | Slower; slower than 1 s returns to auto |
| `ISO` | 100, 200, 400, 800, 1600 (`iso.py`), less any step above the sensor's gain limit | Higher; stays on the top step | Lower; below 100 returns to auto |
| `WB` | 3200, 4000, 5000, 5600, 6500, 7500 K (`whitebalance.py`) | Warmer (a higher number); stays on 7500 | Cooler; below 3200 returns to auto |

- **Auto and the first step.** Every control starts automatic and resets to automatic
  on restart (white balance to `--white-balance` if given). The first step out of auto,
  in either direction, lands on the step nearest, measured in stops (log2), to what the
  camera is using now, so the picture does not jump. Without a metered value yet the
  start is 1/125, ISO 400 or 5600 K. The scales are in `stepscale.py`.
- **Full manual.** With shutter and ISO both fixed there is no auto-exposure left: the
  picture follows the settings, and the exposure meter is the only guide. Stepping
  either one darkens or lightens the picture, and the meter dot follows.
- **`MAX`.** On automatic ISO, when the gain is at or near the sensor's maximum (98 %
  or more), the `ISO` cell reads `MAX`. On a fixed shutter that means the speed is too
  fast for the light and the photo will be dark: pick a slower speed or add light. It
  can also appear on pure auto in very dim light.
- **The sensor's gain limit.** ISO is analogue gain × 100, and the camera refuses a
  gain above the sensor's maximum. Steps above that limit are not offered: the scale
  ends lower, and the first step from auto lands on the nearest step the sensor can
  reach. The IMX477 (maximum gain 22.26) reaches all five.
- **Unavailable controls.** A cell is drawn grey and cannot be selected when the camera
  has no setter for it; `SHUTTER` and `ISO` when auto-exposure was locked with
  `--ae-lock`; `WB` when white balance was already fixed with `--awb-lock` or
  `--colour-gains`, or on a libcamera without a settable `ColourTemperature`. A grey
  cell still shows the metered value but no `AUTO` tag, because the control is fixed,
  only not from this screen. At start-up the selection is `SHUTTER`, or the first cell
  that can be changed. If no cell is selectable, `+` and `-` do nothing. This is
  decided once, at start-up.
- **`--white-balance K`** (2360 to 9000) starts the camera at a fixed value, for
  instance while a lens that fools auto is mounted. A restart returns to the flag's
  value, or to auto without the flag. It cannot be combined with `--awb-lock` or
  `--colour-gains`. There is no `--iso` or shutter flag. A start value above the top
  step (8000 or 9000 K) stays where it is when `+` is pressed; `-` brings it onto
  7500 K and the scale from there.
- **Records.** Each photo's `captures.jsonl` line has `shutter_us`, `iso` and
  `colour_temperature_k`, each null on auto. The values the camera actually used are in
  `camera_metadata` (`ExposureTime`, `AnalogueGain`, `ColourGains`).
- **No exposure compensation.** The EV buttons, the `--ev` flag and the camera `set_ev`
  were removed on 2026-10-11. `ev_comp` stays in every record as `0` (the camera still
  sends `ExposureValue: 0.0`), so old records regrade exactly as before;
  `pifilm-process` and `Pipeline.process(ev=...)` are unchanged.
- The deployed look does not correct white balance itself (`white_balance: false` in its
  `params.json`), so a fixed white balance goes straight through to the graded file. A
  higher number renders warmer: 5600 K is daylight, 3200 K is tungsten lamps, and 6500
  or 7500 K corrects a picture that auto leaves cold.
- Returning `WB` to `AUTO` is slow, and that is libcamera's auto white balance, not
  the viewfinder: it eases from the old value toward its own answer in small steps so
  the colour does not flicker. Measured on the Pi 4 + IMX477 on 2026-10-12 in dim room
  light: a fixed value was reported by the camera 0.6 s after it was set (1.7 s after
  a photo), while a return from 7500 K to auto did nothing for 2 s and was still
  drifting after 8 s; by eye it settles in about 12 s. A photo taken meanwhile uses
  whatever auto has reached. Nothing was changed for this: the damping is part of the
  sensor tuning.
- **The controls are checked against the camera.** On the Pi 4 a control sent to the
  running camera does not always take effect, and the cause is not known (see
  [known issues](known-issues.md#live-white-balance-changes-are-ignored-after-a-photo-pi-4-imx477)).
  What you see: before the first photo after start-up a white-balance change shows
  within about 2 s, with no blink. After a photo it shows in the old colour for about
  a second and a half, then the live view blinks once and shows the new one. Shutter
  and ISO change live either way.
  How: every new camera frame's metadata is compared with what is fixed:
  `ColourTemperature` within 100 K of a fixed white balance, `ExposureTime` within 5 %
  of a fixed shutter, `AnalogueGain` within 5 % of a fixed ISO, and, after `WB` has
  been returned to `AUTO`, a `ColourTemperature` that has left the old fixed value. A
  frame without the value cannot tell and is never a fault.
  - *White balance* that disagrees for 1.5 s and 6 camera frames (4 s for a return to
    auto, because libcamera's own damping holds the old value for about 2 s) restarts
    the live view at once, which reconfigures the camera as a photo does:
    `camera: white balance 6500K not in the live view after 1.6 s (camera reports 5600K); restarting the live view; picture R/G 1.03 B/G 0.96`,
    or `camera: auto white balance has not moved from 7500K after 4.1 s; restarting the live view; picture ...`.
    It is never resent: on the Pi that cured 2 cases of about 7, and a restart cured
    3 of 3.
  - *Shutter and ISO* that disagree for 3 s and 6 camera frames are sent again:
    `camera: shutter 1/250 not in the live view after 3.2 s (camera reports 1/30); resending; picture ...`
    (likewise `ISO 800`). If two resends change nothing the live view is restarted:
    `camera: shutter 1/250 still not in the live view after 2 resends; restarting the live view; picture ...`.
  The first frame that agrees again logs `camera: ... applied after resend` or
  `... applied after restarting the live view`. There is one restart per control per
  selection: if the disagreement outlives it, or goes away and comes back,
  `camera: ... not applied even after restarting the live view` is logged once and the
  live view is not restarted again for that value (shutter and ISO go on being resent
  every 3 s, with the `resending` line written at most once in 30 s and a count of the
  ones held back; white balance is left alone). A step that changes that control, a
  photo, a review or a dark screen starts afresh and may restart again, but two
  restarts are never less than 5 s apart. A restart for an auto white balance that has
  not moved also ends the watch on auto until a white balance is next fixed, so a
  capped lens or a dark scene, where auto reports the old value for good, costs one
  blink and no more. The lines end with the picture's colour,
  the means of the preview frame's red and blue channels over its green one. They are
  the evidence of the camera not taking a live control:
  `journalctl -u pifilm-capture | grep -E 'camera:|control:'`.
- **Every step leaves a trace.** A `+`/`-` step that changes a control is followed,
  on the first new camera frame at least 2 s later, by one journal line with what was
  selected, what the camera reports and what the picture looks like:
  `control: WB 7500K selected; camera reports 7500K gains (3.35, 1.36), exposure 1/60, ISO 400; picture R/G 2.42 B/G 0.00`
  (`WB AUTO selected` for a return to auto; `-` for a value the frame does not carry).
  All three reported values are written whichever control was stepped. A second step
  before the line is written replaces it, and a photo, a review or a dark screen
  cancels it. The camera alone applies a live white balance correctly, even after a
  photo (tried on the Pi), so the fault above is somewhere in how the app runs it;
  this line is how a case would be caught where the camera reports the selected value
  and the picture does not show it, which the check above cannot see. A step the camera
  rejects, or `+` on a top step, changes nothing and is not traced.

Shutter notes:

- At long speeds the live view slows to match: at 1/4 s it shows about 4 frames a
  second. That is the exposure, not a fault.
- At long speeds the controls stay quick although the picture is slow: a tap on a
  cell, `+` or `-` changes the value on screen and lights the key on the loop's next
  redraw, about a tenth of a second later, not on the next camera frame. The picture
  itself changes when the camera delivers the next frame, so at 1 s the new exposure
  shows up to a second or two after the tap. There is no need to hold the tap.
- The photo waits for the preview exposure in progress: the camera takes one frame
  at a time, so at 1 s the still can start up to a second after the shutter tap.
  The colour bars appear at once all the same.
- The `viewfinder: N fps` line in the journal counts camera frames, not redraws, so
  at 1 s it reads about 1 while the screen is redrawn ten times a second.
- At start-up the journal names how each control is fixed:
  `picamera2: shutter priority via ExposureTimeMode` on current libcamera, or
  `... ExposureTime (legacy libcamera)` on older stacks; and on the next line
  `picamera2: ISO via AnalogueGainMode`, or `... AnalogueGain (legacy libcamera)`.
- Under `--ae-lock` the `SHUTTER` and `ISO` cells are drawn grey and cannot be
  selected: auto-exposure is off, so there is nothing for a fixed shutter or ISO to
  work against. The camera refuses both controls in that mode.

Measured on the Pi 4 with the IMX477 (libcamera 0.7.2, Picamera2 0.3.37) on 2026-10-11,
preview configuration, dim room:

| Set | ExposureTime | AnalogueGain | DigitalGain |
| --- | --- | --- | --- |
| auto | 66656 | 16.0 | 1.0 |
| AnalogueGain 1.0 | 66656 | 1.0 | 1.17 |
| AnalogueGain 4.0 | 65443 | 4.0 | 1.0 |
| AnalogueGain 16.0 | 65434 | 16.0 | 1.0 |
| AnalogueGain 4.0 + ExposureTime 8000 | 7996 | 4.0 | 1.0 |
| AnalogueGain 0 + ExposureTime 0 | 66656 | 15.75 | 1.0 |

`camera_controls["AnalogueGain"]` is `(1.0, 22.26, 1.0)` and `AnalogueGainMode` exists. A
fixed gain holds while the exposure time stays automatic, both can be fixed, and zero
returns both to auto. ISO 100 is gain 1.0, the convention the meter already uses. The
room was too dark to show the exposure time compensating for a fixed gain (it was
already at the frame limit); that is step 3 of the acceptance list in section 7.

White balance, measured on the IMX477 on 2026-10-11 (libcamera 0.7.2): the red and
blue gains were (2.155, 2.052) at 3200 K, (3.102, 1.428) at 5600 K and (3.355, 1.360)
at 7500 K. A request below 2360 K is clamped to 2360 K and 9000 K is accepted as asked.

## 6. Reading the meter

- **Lit dot near the middle** — the exposure is near mid-grey: auto-exposure has
  converged on this scene, or your fixed settings suit it.
- **Dot at either end** — the exposure is 3 stops or more from mid-grey. On auto the
  camera has run out of range; in manual, change the shutter or ISO or expect the
  shot to be under/overexposed.
- **`CLIP NN%`** — the fraction of pixels with any channel at or above 254; this
  is the same clipped-highlight number Phase 6's normalisation cares about
  ([docs/superpowers/plans/2026-09-13-rpi4-imx708-progress.md](superpowers/plans/2026-09-13-rpi4-imx708-progress.md)).
  Aim low outdoors, where clipping is easy to get.
- **`NNN LX`** — libcamera's own scene-brightness estimate from the sensor, not an
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
- **Nothing to focus on also reads zero.** The bar only scores edges that stand
  clearly out of the frame's own noise and shading, so a plain surface (a wall,
  skin close up) leaves it empty however the ring is set. Aim the middle of the
  frame at something with an edge in it. A surface covered edge to edge in one
  even texture, such as grass filling the frame, reads lower than its sharpness
  deserves and changes little until it is well out of focus.
- **Measured on the Pi 4 + IMX477 at ISO 1600, 1/15 s (2026-10-13)**, on saved
  preview frames. Before that date the edge threshold assumed fine-grained
  noise, and the ISP's blotchy noise passed as detail, so a lens left
  completely out of focus read higher than a slightly soft one:

  | Frame | Old bar | Bar now |
  | --- | --- | --- |
  | Monitor text, sharp | 1.00 | 1.00 |
  | Monitor text, slightly soft | 0.33 to 0.40 | 0.07 to 0.10 |
  | Monitor text, completely out | 0.43 to 0.45 | 0.05 to 0.14 |
  | Room scene, sharp | 0.78 to 0.81 | 0.71 to 0.74 |
  | Room scene, slightly soft | 0.28 | 0.11 to 0.18 |
  | Room scene, completely out | 0.43 to 0.45 | 0.00 |
  | Plain surface, any focus | 0.50 to 0.67 | 0.00 to 0.06 |

  Run live on the Pi 4 on 2026-10-13: the bar was near empty fully defocused,
  climbed toward focus and peaked where the picture looked sharpest.
- **The amber tick is the best level of the last few seconds.** Rack through
  focus, then turn back until the white fill reaches the tick again.
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
5. (Removed 2026-10-11 with EV compensation.)
6. Meter sanity: cover the lens → the leftmost meter dot is lit, `LX` near 0; point
   at a lamp → `CLIP` rises.
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

12. (Replaced 2026-10-11: shutter priority is now part of the manual controls; see
    "Manual controls and the SLR finder" below.)

13. (Replaced 2026-10-11 likewise. Recorded before the replacement, on the Pi 4 with
    the DSI panel on 2026-10-11: the old `WB` pill's steps, a shot at a fixed value,
    the return to auto, the record field and a tele lens all passed. The
    `--white-balance` start-up flag was not run on hardware.)

#### Manual controls and the SLR finder

Not yet run on hardware. The branch is `feat/manual-controls-ui`.

| Step | Check | Result |
| --- | --- | --- |
| 1 | Start-up: full-screen image, strip with three cells all `AUTO`, `SHUTTER` selected. The journal names the shutter and ISO control paths | Not yet run |
| 2 | Select each cell; `+`/`-` change only the selected one; `-` below the bottom returns it to `AUTO`; the first step from auto does not jump the picture | Not yet run |
| 3 | ISO fixed, shutter auto, in good light: the shutter read-out changes to compensate as ISO is stepped, and the meter stays near the centre | Not yet run |
| 4 | Full manual: fix shutter and ISO; the picture gets darker and lighter with the steps and the meter dot follows | Not yet run |
| 5 | `+`/`-` light while the finger is down, and on a quick tap | Not yet run |
| 6 | A shot at fixed shutter + ISO + WB: the record has `shutter_us`, `iso`, `colour_temperature_k`, and `camera_metadata` agrees. Also `--white-balance 6500`: the `WB` cell reads `6500K` from the first frame | Record part passed 2026-10-12: a shot with shutter 1/125, ISO 1600 and 7500 K fixed logged `shutter_us` 8000, `iso` 1600, `colour_temperature_k` 7500, and `camera_metadata` reported 7986 us, gain 16.0 and 7500 K; ten shots at a fixed white balance all matched their metadata. The `--white-balance` start-up flag: not yet run |
| 7 | The shutter button is not hit while working `+`/`-` | Not yet run |
| 8 | `2x`, focus gauge, battery, review, idle screen-off, Stick and GPIO button work as before | Not yet run |
| 9 | About 10 fps is kept (`viewfinder: N fps` in the journal) | Not yet run |

#### Background grading

Not yet run on hardware. The branch is `feat/background-grade`; the design is
[2026-10-13-background-grading-design.md](superpowers/specs/2026-10-13-background-grading-design.md).

| Step | Check | Result |
| --- | --- | --- |
| 1 | A single shot: the review appears and the camera is free in about 1.6 s; the graded file appears a few seconds later; the record has `shutter_to_ready_ms` | |
| 2 | Five shots as fast as the camera allows: all five graded files and records are present, in order; the developing count rises and returns to zero | |
| 3 | A double exposure: the card, then the composite's review; one graded file | |
| 4 | Stick: the picture arrives after each shot, as before | |
| 5 | Power cut with the count above zero (pull the supply with the UPS off, or `kill -9`): after the restart the missing graded files appear, and their records are marked `recovered` | |
| 6 | `--grade inline`: behaves as before this change | |
| 7 | Ten minutes of ordinary use: no preview stall, and the live view is acceptable | |

#### GPU grading

Not yet run on hardware. The branch is `feat/gpu-grade`; the design is
[2026-10-13-gpu-grading-design.md](superpowers/specs/2026-10-13-gpu-grading-design.md).
Use a look with the tile grain (`two_scale_tile`): the bundled starter has Gaussian
grain and is always graded on the CPU. Run `pip install -e .` again in the Pi's venv
first, so that `pifilm-gpu-check` exists.

| Step | Check | Result |
| --- | --- | --- |
| 1 | The service starts with the display on. The journal has `gpu grading on (..., self-test passed)`. This is the first time the GPU context runs beside the display in one process. If the start hangs here, the next start logs `gpu grading off: the last start did not finish opening the GPU; delete <path> to try again` and grades on the CPU: note it, and delete the file only to try again | |
| 2 | `pifilm-gpu-check --artifacts DIR --image FILE` on a real photograph: within 1 level of the reference, and timings near the spike's (upload 142 ms, shader 239 ms, read-back 180 ms) | |
| 3 | Ten photographs: every record has `pipeline_backend: gpu-2`; `stage_ms` shows the grade near 0.6 s without halation, about 1.0 s with (`normalize` plus `gpu`); the graded files look right at 100 %, grain included | |
| 4 | A double exposure and a Stick shot work as before | |
| 5 | `--grade-backend cpu`: records have `pipeline_backend: cpu`, and the timings are as before | |
| 6 | `pifilm-process` on today's folder: it runs, and prints the note about the records graded on the GPU | |
| 7 | Ten minutes of ordinary use: no preview stall, and the live view is no worse than now | |

#### Tuning check

Passed on the Pi 4 with the IMX477 on 2026-10-14 (PR #63). The design is
[2026-10-14-tuning-identity-design.md](superpowers/specs/2026-10-14-tuning-identity-design.md).
Run `pip install -e .` again in the Pi's venv first, so that `pifilm-tuning` exists.

| Step | Check | Result |
| --- | --- | --- |
| 1 | `pifilm-tuning --tuning-file imx477.json` prints a fingerprint and the file | Pass |
| 2 | `pifilm-tuning --tuning-file imx477.json --stamp <the service's artifacts>`, run as `george` (not with `sudo`), then restart the service with `sudo systemctl restart pifilm-capture.service`: the journal says `tuning: matches the look`; records carry `tuning_sha1` and `tuning_matches: true`; no tag on the screen | Pass |
| 3 | Stamp the look with a wrong fingerprint (`pifilm-tuning --sha1 0000000000000000000000000000000000000000 --stamp <dir>`) and restart: the journal says `DIFFERS`, the amber `TUNING` tag shows, records say `tuning_matches: false`, and photographs are still taken and graded. (An edited copy of `imx477.json` passed to `--tuning-file` by its absolute path tests the same thing from the camera's side.) | Pass |
| 4 | Stamp the right fingerprint again and restart: the tag is gone | Pass |

#### Disk-space warning

Not yet run on hardware. With the photo volume about 50 % full and the SD card
about 70 % (`df -h ~/Pictures/pifilm /` says what they are), stop the service with
`sudo systemctl stop pifilm-capture` and run by hand from the project directory:

```sh
.venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --no-preview \
    --out ~/Pictures/pifilm --disk-warn-percent 40
```

| Step | Check | Result |
| --- | --- | --- |
| 1 | Within a second of the live view, amber `PHOTOS 50%` and `SD 70%` tags show between the top-left row and the shutter button (the numbers are whatever `df -h` says) | |
| 2 | The terminal has `disk: photos 50% full (... GB free)` and `disk: sd 70% full (... GB free)`, once each, and no more over two minutes | |
| 3 | A photograph is taken as usual with the tags showing, and a tap on the tags does nothing | |
| 4 | Ctrl-C, then run again without `--disk-warn-percent` (90): no tag and no `disk:` line. `sudo systemctl start pifilm-capture` afterwards | |

#### Settings menu

Not yet run on hardware. The design is
[2026-10-15-settings-menu-design.md](superpowers/specs/2026-10-15-settings-menu-design.md).
With the service running with `--nextcloud-env` and the panel in live view:

| Step | Check | Result |
| --- | --- | --- |
| 1 | `MENU` opens the menu; `DONE` returns to the live view; the shutter, `+`, `-` and the cells work as before | Not yet run |
| 2 | `SAVE DNG` off: the next photograph has no `.dng`; on: it has one. Restart the service: the setting is kept and the journal has the `settings:` line | Not yet run |
| 3 | `HALATION` off (with a look that has halation): the next photograph has no glow round bright lights, and its record has `"halation": "off"` and no `halation_sha1`; on again: both back as before. Restart the service: the setting is kept and the journal has the `settings: halation off ...` line | Not yet run |
| 4 | `AE CONSTRAINT` and `AE METERING`: after a change in the menu the journal has `settings: ae-metering <mode>` (or `ae-constraint`), the next photograph's record has `ae_metering` / `ae_constraint` with the new mode, and after a restart the journal has the `settings: ... (saved in the menu; the unit says ...)` line | Not yet run |
| 5 | `UPLOAD` on home Wi-Fi: `UPLOADING`, then `DONE`; the files are on Nextcloud; `journalctl -u pifilm-capture \| grep nextcloud:` shows rclone's lines. In hotspot mode: `SKIPPED: NO NEXTCLOUD`. Started while the timer's run is going: `SKIPPED: BUSY` | Not yet run |
| 6 | `ETH`/`WLAN` show the addresses (unplug the cable: `ETH` turns to `-` within about 10 s), and the live view no longer shows them. `LOOK` shows the deployed look's folder name, or `starter (bundled)` | Not yet run |
| 7 | `RESET`, tapped twice: the unit's flags are back and `settings.json` is gone | Not yet run |
| 8 | A GPIO button press while the menu is open takes a photograph, and the menu stays open | Not yet run |
| 9 | `IMAGE` opens the image page. Step `SATURATION` to `0.5`: the live view loses colour at once and the amber `IMG` tag shows. Do the same with `CONTRAST` (flatter at `0.5`) and `SHARPNESS` (softer at `0.5`, crisper at `1.5`; easiest seen at `2x`). Take a photograph: it shows the same change, and its record has `sharpness`, `contrast` and `saturation` after `ae_metering` with the values shown. The journal has a `settings: saturation 0.5` line per change | Not yet run |
| 10 | The image page's `RESET`, tapped twice: the three are back to `1.0`, the `IMG` tag goes, and the other settings are unchanged. `BACK` returns to the first page; `DONE` from the image page returns to the live view. Restart with a value saved: the journal has the `settings: ... (saved in the menu; the unit says ...)` line and the live view shows it | Not yet run |

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
| `not in the live view after ... resending`, `restarting the live view`, `applied after resend` | The camera did not take a control sent to it while running (seen on the Pi 4, cause not yet known). For white balance the viewfinder restarted the live view; for shutter and ISO it sent the control again and restarted the live view after two resends | Nothing to do when an `applied after ...` line follows. `not applied even after restarting the live view` means that did not help: step the control again or take a photo (both allow another restart or reconfigure the camera) and keep the journal lines, see [Manual controls](#manual-controls) and [known issues](known-issues.md#live-white-balance-changes-are-ignored-after-a-photo-pi-4-imx477) |
| `the V4L2 backend has no preview mode` | The camera is a USB/UVC one; the viewfinder needs Picamera2's preview-mode split and libcamera metadata | Nothing to fix unless a Pi camera is intended: check `--camera`/`--device` and that `python3-picamera2` is installed |

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

   (`B4` restores the desktop with autologin, which two-screen mode,
   `--show-captures`, needs.)

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

# Print mapped touch coordinates while a finger is on the panel
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
| Idle | Dims at 1 minute, backlight off at 5 | No brightness control, so no dim step and `--display-dim-after` has no effect. At 5 minutes (`--display-off-after`) the display output is powered down and a tap powers it back up (measured on the panel 2026-10-10: the screen goes fully dark and comes back) |
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

Each message below is printed by `pifilm-capture` as
`warning: display unavailable (<message>); continuing without it`, and the
service carries on without the screen, still serving the Stick.

| Message contains | Cause | Fix |
| --- | --- | --- |
| `install python3-kms++` | `pykms` is not importable | `sudo apt install python3-kms++`; the venv must see system packages, as for Picamera2 |
| `cannot open the DSI display on DSI-1` | No such connector, the cable is out, or a desktop session owns the screen | Check the `dtoverlay=waveshare_35DSI,35E,dsi1` line and the overlay file, reseat the cable, boot to the console (9.1 step 4) |
| `cannot open the DSI display on DSI-1`, ending `set_mode returned ...`, on a Pi that does boot to the console | Something else already holds the display: usually a second copy of `pifilm-capture` (the service is running and you started another from a terminal) | Stop the other one first: `sudo systemctl stop pifilm-capture` before a terminal test, and `sudo systemctl start pifilm-capture` afterwards |
| `no permission for the DSI display` | Service user not in `video`/`render` | `sudo usermod -aG video,render george`; log out and in, or reboot |
| `touch device 'Goodix Capacitive TouchScreen' not found` | The touch driver did not load | Same overlay and cable checks; `grep -i goodix /proc/bus/input/devices` |
| `no permission for /dev/input/event*` | Service user not in `input` | `sudo usermod -aG input george`; log out and in, or reboot |
| Panel blank after boot, no boot text | Overlay not applied | `dmesg \| grep -i "dsi\|panel\|goodix"`; check steps 2 and 3 |
| Live image freezes for about 20 seconds, then the service restarts by itself; the journal has `camera: no preview frame within` | Picamera2 stopped handing over frames | Nothing to do: the restart reopens the camera. See [known issues](known-issues.md#picamera2-stops-handing-over-preview-frames-on-the-raspberry-pi-4-viewfinder-freezes) |

### 9.6 Hardware acceptance checklist

Run on 2026-10-10 on the Pi 4 with the IMX477, Raspberry Pi OS Trixie booted to the
console, branch `feat/dsi-viewfinder` at `2d2b6fc`. Idle was checked with
`--display-off-after 20` rather than the five-minute default.

| Item | Result |
| --- | --- |
| 1 Cold boot | Pass. Live view about 28 s after power-on, without logging in |
| 2 Frame rate and tearing | Pass. 9.1 and 10.0 fps in the journal lines; no tearing when panning |
| 3 Touch accuracy | Pass. Every control responds where it is drawn, corners included |
| 4 Controls | Pass, for the screen before 2026-10-11 (shutter, review, EV buttons, shutter priority and `2x`). The redesigned screen is checked by the manual-controls table in section 7 |
| 5 Stick shot | Pass. Appears on the panel |
| 6 Idle | Pass. No dimming; dark at the off time; a tap wakes it without taking a photo |
| 7 Stick shot while dark | Pass. Wakes into the review, no `display:` error lines |
| 8 Service stop | Pass. The login prompt returns to the panel |
| 9 Cable unplugged | Pass. One warning line; the Stick still captures |
| 10 Rotation 180 | Not run. The panel is upright at 0 in the current mounting |
| 11 2.8" fallback | Not run. The panel is no longer wired |
| 12 Open-time check with the desktop running | Not confirmed. The warning line was not observed; repeat before relying on it |


1. Cold boot: the live view appears on the panel without logging in. Note the time
   from power-on.
2. The journal's `viewfinder: N fps` line reads 8 or more; panning shows no tearing.
3. Each control responds where it is drawn, including in the corners;
   `--touch-debug` prints coordinates within 0..639 × 0..479.
4. Shutter tap, the manual controls, `2x` and review behave as in items 3 and 11 and
   the manual-controls table of [section 7](#7-hardware-acceptance-checklist).
5. A Stick shot appears on the panel, which returns to live after 30 s untouched.
6. Idle: no dimming at one minute; dark at five; a tap wakes it without taking a
   photo (no new `captures.jsonl` line).
7. With the screen dark from idle, take a shot from the Stick: the panel wakes
   into the review, and the journal (`journalctl -u pifilm-capture`) shows no
   `display:` error lines. This checks drawing straight after the output is
   powered back on.
8. `sudo systemctl stop pifilm-capture`: the login prompt returns to the panel.
9. Unplug the DSI cable and restart the service: one `warning: display unavailable`
   line in the journal, and the Stick still captures.
10. `--display-rotate 180`: the image is inverted and taps still land on the controls.
11. Optional, only if the 2.8" panel is still wired: `--display waveshare28` works.
12. The open-time check, with the desktop running. This is the only way to prove on
    hardware that a refused mode-set is caught at open. Set the boot behaviour to
    the desktop (`sudo raspi-config nonint do_boot_behaviour B4`) and reboot. Over
    SSH, run:

    ```sh
    .venv/bin/pifilm-capture --camera picamera2 --display waveshare35dsi --no-preview --out ~/Pictures/pifilm-lcd
    ```

    Expect exactly one
    `warning: display unavailable (cannot open the DSI display on DSI-1 ...)` line,
    naming console boot, and the process to carry on without the screen. Then
    restore the console (`sudo raspi-config nonint do_boot_behaviour B1`) and reboot.
