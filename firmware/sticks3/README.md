# StickS3 remote capture firmware

This PlatformIO project turns an M5StickS3 into the authenticated remote
capture button for the local `pifilm` API. It displays local TV colour bars
while ready, then a state screen and elapsed timer while a request is active.
The most recent successfully decoded 240×135 JPEG remains on screen until a
later complete image replaces it.

## Prerequisites

Install PlatformIO Core and connect an M5StickS3 over USB-C. The project uses
M5Stack's documented StickS3 base: `esp32-s3-devkitc-1`, 8 MB partition table,
QIO/OPI PSRAM, USB CDC, M5Unified, M5GFX, and M5PM1.

The pre-build script reads the repository-root `.env`; see the
[deployment guide](../../docs/sticks3-remote.md). Missing configuration leaves
safe empty defaults and the device cannot join a network. Only Wi-Fi settings,
API URL and API token enter the firmware; the SSH password stays on the host.
In the deployed topology the Pi runs its own hotspot, so `WIFI_SSID` and
`WIFI_PASSWORD` are that hotspot's credentials and `PIFILM_REMOTE_URL` is
`http://10.42.0.1:8765`, the hotspot address.
Build outputs contain device credentials. Keep them private and avoid verbose
compiler logs when using real settings.

The server must expose the bearer-authenticated API contract at `/v1/status`
and `/v1/captures`. The device enters Ready only after `/v1/status` says the
Pi is ready and no other capture is active. It records that endpoint's
`instance_id`, submits `{"request_id":"UUID"}`, polls the same ID every
500 ms, and fetches `/v1/captures/<UUID>/image.jpg` only on completion. A
changed server instance or an unknown job interrupts the active request and
requires a fresh deliberate press; it never automatically resubmits it.

## Build, flash, and serial monitor

```sh
pio run -d firmware/sticks3
pio run -d firmware/sticks3 -t upload
pio device monitor -d firmware/sticks3 -b 115200
```

Run the hardware-independent state machine tests with:

```sh
pio test -d firmware/sticks3 -e native
```

On first flash, verify physical hardware before enabling network credentials:

1. Serial prints `StickS3 remote capture boot` and identifies the expected USB device.
2. The boot smoke screen appears, the short speaker tone is audible, and colour bars follow.
3. Press and hold the primary button: it should yield only one shutter tone and one request.

## Buttons

- **M5 button (BtnA)**, the large front button: the shutter. One deliberate
  press is one capture; holding it yields one request, not a stream.
- **Side button (BtnB)**, the small button on the edge: turns the screen off
  and back on to save battery, which on this board is mostly the backlight.
  A click is a press and release, so holding the button does nothing until it
  is let go. While the screen is dark the Stick still connects, submits
  captures, downloads and stores the thumbnail, and sounds the shutter tone -
  that tone is the only feedback in the dark, because a new photo deliberately
  does not wake the screen. Turning it back on repaints the current screen,
  including the most recent photo and both battery badges.
- The **power button** is untouched; it still powers the Stick off and on as
  the hardware defines.

The toggle lives in `DisplayPower` (`include/display_power.h`), an
Arduino-free module covered by the native tests, with the same 30 ms debounce
as the shutter path.

## Battery indicator

The bottom-right corner of every screen shows the Stick's own battery as a
small badge prefixed `S3`: `S3 87%` on battery, `S3 87%+` while external power
is attached over USB, `S3 --%` when the power chip cannot report a level. It turns red at 15% or
below on battery. The READY caption is centred in the space left of the badge.

The level comes from M5Unified's `M5.Power.getBatteryLevel()`, which on the
StickS3 derives it from the PM1 power chip's battery voltage. The voltage sags
while the Wi-Fi radio transmits, so raw readings swing by five or six points
between polls (81, 75, 80, 75 was measured). Readings are averaged with a time
constant of about four polls, and the shown level then holds until the average
differs from it by at least 3 points. A sustained change of a few points shows
within a minute or so; a steady drain is tracked with a lag of a couple of
points. The `+` mark means external power is present, read from the PM1's
power-source register. It deliberately does not follow M5Unified's
`isCharging()`, which on the StickS3 reads the charger's CHG_STAT pin: with the
cable attached that pin cycles on and off for about 20 seconds at a time as the
charger regulates, which made the mark blink. If the power-source read fails
the charger pin is used instead, with a two-poll debounce before the mark is
dropped. An unknown level shows `--%` at once; an unknown power state leaves
the mark as it was. Expect the level to move in steps rather than smoothly, and
to read high while a charger is attached. The chip is polled every 10 seconds
by `BatteryMonitor` (`include/battery_status.h`), which is Arduino-free and
covered by the native tests; the display repaints the badge only when the text
changes. Each change is also logged on the serial port, with the PM1 source
bitmap (bit 0 VIN, bit 1 VIN/OUT, bit 2 battery):

```text
[1203] battery 87%+ (level 87, external power, power sources 0x05, 4012 mV)
```

The bottom-left corner shows a second badge for the Pi's own UPS battery, as
published by the Pi in `GET /v1/status`'s `pi_battery` object: `Pi 87%+` while
the Pi is on external power, `Pi --%` when the Pi has no UPS attached or the
published value is more than a minute old. It turns red at 15% or below while
the Pi is on battery. This badge is fed purely from the status poll, not from
the Stick's own PM1, and is driven by its own `BatteryMonitor` instance so a
one-point wobble in the Pi's reading does not repaint the badge. Each change
is logged, for example:

```text
[1556] pi battery 87%+ (known=1, external power)
```

## Reading the serial debug log

The firmware logs every hop of a capture on the serial port, each line
prefixed with the uptime in milliseconds. A healthy capture looks like this:

```text
[12034] status: HTTP 200 ready=1 active_capture=0
[12040] state CONNECTING -> READY
[12041] [display] render READY (pi ready=1, stored photo=0 bytes)
[30212] submit 3f9c1a2b: HTTP 202 (accepted)
[30215] state READY -> REQUESTING
[30731] poll 3f9c1a2b: HTTP 200 job state 'processing'
[30733] state REQUESTING -> PROCESSING
[33245] poll 3f9c1a2b: HTTP 200 job state 'complete'
[33247] state PROCESSING -> DOWNLOADING
[33802] download 3f9c1a2b: HTTP 200, content-length 18342
[33951] download ok: 18342 bytes, valid JPEG markers, handed to UI loop for decode
[33970] [display] JPEGDEC validation ok; M5GFX drawJpg 18342 bytes at 0,0 240x135 -> ok
[33971] ui: 18342 byte JPEG decoded and drawn; state -> PHOTO
[33972] state DOWNLOADING -> PHOTO
[33973] [display] render PHOTO (pi ready=1, stored photo=18342 bytes)
[33990] [display] drawPhoto: M5GFX drawJpg 18342 bytes at 0,0 240x135 on 240x135 screen -> ok
[41022] ui: display off (BtnB)
```

Where the sequence stops tells you which side to look at:

- No `status: HTTP 200` line: Wi-Fi or the API URL/token. Check the Pi's
  `pifilm-capture` service and the `.env` values baked into this build.
- `submit` returns 409: the Pi is already busy with a capture.
- `download` returns 409: the job is not complete yet; the state machine retries.
- `download rejected`: the JPEG arrived truncated or exceeded the 64 KiB buffer.
- `JPEGDEC validation FAILED`: the bytes are not a decodable JPEG.
- `drawJpg ... -> FAILED`: JPEGDEC accepted the file but M5GFX's decoder did not
  draw it. The stored photo is kept; the screen shows black behind the overlay.
- `render PHOTO` followed by `drawPhoto ... -> ok` with nothing visible: check
  the panel itself (brightness, rotation), not the network path - and check for
  an earlier `ui: display off (BtnB)` line, which means the screen was
  deliberately turned off with the side button.

Status and poll results are logged only when they change, so an idle Stick is
quiet apart from Wi-Fi events.

### Confirming the photo is the graded one

The log proves a JPEG arrived and was drawn, not which file it came from. The
server builds the thumbnail from the job's `*_graded.jpg` and the thumbnail
generator is deterministic on one Pillow build, so regenerating thumbnails on
the Pi from both saved files and hashing them settles it. Fetch what the Stick
received (the last completed job ID comes from `/v1/status`), then on the Pi:

```sh
cd ~/repos/pi-film-reversal
.venv/bin/python - <<'PY'
import hashlib, json
from datetime import date
from pathlib import Path
from pifilm.capture.thumbnail import fitted_jpeg
day = Path.home() / 'Pictures' / 'pifilm' / date.today().isoformat()
rec = json.loads((day / 'captures.jsonl').read_text().splitlines()[-1])
for label in ('pifilm', 'original'):
    data = fitted_jpeg(day / rec[label])
    print(label, len(data), hashlib.sha256(data).hexdigest()[:16])
print('lut_sha1 used:', rec['lut_sha1'])
PY
```

Exactly one hash matches `sha256sum` of the downloaded image; it must be the
`pifilm` one. The `lut_sha1` line tells you which look graded it: compare it with
`lut_sha1` in the `params.json` of the artifact you intended the service to
load. The bundled starter and a trained artifact have different hashes.

Network traffic runs in a FreeRTOS worker. The UI loop remains responsive; Wi-Fi
reconnects back off from one to ten seconds. The shutter tone is queued only
after a 2xx capture acknowledgement. A job still unresolved after 120 seconds
stays attached to its original UUID and continues lookup without submitting a
replacement capture.
