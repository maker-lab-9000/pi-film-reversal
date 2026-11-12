# Setup from zero: Mac and Raspberry Pi

This is the ordered install guide. Each section depends on the one before it,
so do them in order. Training is optional and comes last: the camera works
with the bundled starter look without any training data.

Verified 2026-09-08 on: macOS with Python 3.12; Raspberry Pi 3B on Raspberry
Pi OS Trixie (Debian 13, NetworkManager 1.52 with netplan).

## 0. What runs where, and why

| Machine | Runs | Why here |
|---|---|---|
| Mac | Development, tests, training, remote administration of the Pi | Training needs SciPy and a fast CPU. There is one `.env` file, and it lives here. |
| Raspberry Pi 3B | `pifilm-capture` as a systemd service: camera, grading, storage, the LCD viewfinder and the GPIO shutter button, and its own Wi-Fi hotspot | The camera is plugged into it. It must boot unattended on battery. Its OpenCV comes from apt because the pip wheel has no GUI support for two-screen mode. |

At home the Pi's Ethernet port is how you administer it and pull photos. In the
field there is no home network: the Pi is its own access point, and a phone or
laptop joins it to SSH in. The README describes what happens at capture time;
this guide is about getting there.

## 1. Mac workstation

Everything else is driven from here, including the Pi's install over Ethernet.

```bash
git clone <repo-url> pi-film-reversal
cd pi-film-reversal
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[train,dev,deploy]'
```

The extras: `train` adds SciPy and requests for `pifilm-train`, `dev` adds
pytest, ruff and build, `deploy` adds Paramiko for `scripts/deploy_remote.py`.

Verify before going on:

```bash
.venv/bin/pytest -q -m 'not slow'          # Python suite, about 4 minutes
```

## 2. The `.env` file: one file, every consumer

Copy the example and fill it in. It is gitignored; never paste a value from it
into a command, a source file, a service unit, or a commit.

```bash
cp .env.example .env
chmod 600 .env
```

| Key | Read by | Must match |
|---|---|---|
| `PI_HOST`, `PI_USER`, `PI_PROJECT_DIR`, `PI_SSH_PASSWORD` | `scripts/deploy_remote.py` on the Mac | The Pi's hostname (`parr.local` over Ethernet), account, and checkout path |
| `WIFI_SSID`, `WIFI_PASSWORD` | `scripts/pi_hotspot.py` on the Pi (creates the hotspot) | What you type on the phone or laptop that joins it. WPA2 needs 8 to 63 printable ASCII characters. |
| `PI_HOTSPOT_ADDRESS` (optional) | `pi_hotspot.py` (the hotspot's own address, as a /24) | A plain IPv4 address; `10.42.0.1` when left out |
| `PI_ARTIFACT_DIR` | `deploy_remote.py` inspection; mirrors the unit's `--artifacts` | A directory on the Pi holding `params.json` and `pifilm.cube` |
| `NEXTCLOUD_*` (optional) | The Nextcloud sync on the Pi | See [docs/nextcloud-sync.md](nextcloud-sync.md) |

Why one file: every script reads the same values, so there is one place to
change them. Change a hotspot key in `.env`, and you re-run the hotspot script
on the Pi.

## 3. Raspberry Pi: OS and network

Order matters here more than anywhere. Once the Pi becomes an access point it
leaves your home Wi-Fi, so Ethernet access must work first.

### 3.1 Flash and first boot

Use Raspberry Pi Imager with Raspberry Pi OS Trixie. Lite is enough for the
handheld; a desktop is only needed for the optional two-screen mode. In the
Imager settings: user `george`, enable SSH, and your home Wi-Fi for the first
boot. Boot, log in, update:

```sh
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git python3-venv python3-opencv python3-numpy python3-pil
```

### 3.2 Ethernet administration and hostname

Plug the Pi into your router or straight into the Mac. Then:

```sh
sudo raspi-config nonint do_hostname pifilm
nmcli -t -f NAME,TYPE connection show | grep ethernet      # netplan-eth0 on Trixie
sudo nmcli connection modify netplan-eth0 ipv4.method auto ipv4.link-local enabled
sudo raspi-config nonint do_wifi_country DE                # your country code
sudo reboot
```

From the Mac, `ssh george@parr.local` must now work over the cable. Behind a
router the Pi has a DHCP address; on a direct cable it has a link-local one
and mDNS resolves it either way. Do not continue until this works: it is your
way back in after the next step.

Why link-local: with no router between Mac and Pi there is no DHCP server, and
without this setting the wired port would have no IPv4 address at all.

### 3.3 Checkout and `.env` on the Pi

```sh
mkdir -p ~/repos && cd ~/repos
git clone <repo-url> pi-film-reversal
```

From the Mac, copy the completed `.env`:

```sh
scp .env george@parr.local:repos/pi-film-reversal/.env
ssh george@parr.local chmod 600 repos/pi-film-reversal/.env
```

### 3.4 Hotspot

On the Pi, over the Ethernet session:

```sh
cd ~/repos/pi-film-reversal
.venv/bin/python scripts/pi_hotspot.py --env .env          # dry run, passphrase redacted
sudo .venv/bin/python scripts/pi_hotspot.py --env .env --apply
nmcli connection show --active                              # pifilm-ap on wlan0
nmcli -t -f NAME,TYPE connection show | grep wireless       # find the home profile
sudo nmcli connection modify netplan-wlan0-<SSID> connection.autoconnect no
```

(The script needs the venv from section 4.1. If you prefer to do the network
first, run it with `python3` instead of `.venv/bin/python`; it has no third-party
dependencies.)

The script writes a root-only NetworkManager keyfile: access point on `wlan0`,
WPA2, DHCP for clients on `10.42.0.0/24` (or the /24 around
`PI_HOTSPOT_ADDRESS`), autoconnect at boot. It also disables Protected
Management Frames, without which the Pi 3B's radio cannot start an access point
at all, and Wi-Fi power save, which made clients drop. Details, the netplan
notes and the recovery commands are in
[the network and service guide](pi-service.md).

Why the Pi is the access point: the camera must work with no infrastructure,
and you still want to reach it over SSH away from home.

## 4. Raspberry Pi: application and service

### 4.1 Install the package

```sh
cd ~/repos/pi-film-reversal
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/pifilm-capture --fake --no-preview                  # smoke test; Q to quit
```

`--system-site-packages` is deliberate: it lets the venv use the apt OpenCV,
which is built with GTK. The pip wheel would work headless but not for the
two-screen mode. On a Pi with a native camera (Camera Module 3 / IMX708) it also
exposes the apt `python3-picamera2` and libcamera, which must never be installed
from pip; select that backend with `--camera picamera2`, and follow
[the Picamera2 bring-up checklist](picamera2-bringup.md)
for first-time hardware acceptance. The tuning file is libcamera's automatic choice for
the detected sensor by default; `--tuning-file imx708_wide.json` pins it, which is only
needed for a non-default module variant. A Pi still on the USB camera should pass
`--camera v4l2` explicitly once this code is installed, because an installed
Picamera2 changes the default backend.

On Picamera2, each capture saves three files: `_original.jpg` (the ISP's own
rendering), `<stem>.dng` (a raw sidecar from the same capture request, on by
default; `--no-dng` disables it and saves storage — about 28 MB per shot
otherwise, and only omits the `.dng`; the original still stays `_original.jpg`),
and `_graded.jpg` (the graded output). The example unit passes `--no-dng` (since 2026-10-14): a look trained on developed DNGs showed no visible gain ([write-up](experiments/2026-10-14-raw-look.md)), and leaving the DNG out frees the camera about 0.5 s sooner. `captures.jsonl` gains
`camera_metadata` (a filtered subset of the request's metadata, such as
`ExposureTime`, `AnalogueGain`, `Lux`, `LensPosition`) and `dng` (the sidecar's
filename) when present. Autofocus defaults to continuous
(`--autofocus {continuous,auto,manual}`, `--af-range {normal,macro,full}`);
since there is no flash, exposure and white balance stay auto by default too,
with `--ae-lock`, `--awb-lock` and `--colour-gains R,B` available for
controlled, reference-matching shoots.

### 4.2 Choose the look

The service loads one artifact directory. Two choices:

- `pifilm/data` inside the checkout. The bundled, untrained starter preset. It
  exists in every clone and is the default in `.env.example` and the unit.
- A trained artifact. Copy its directory, containing `params.json` and
  `pifilm.cube`, from the Mac. Artifacts are gitignored, so they never arrive by
  `git pull`:

  ```sh
  rsync -av artifacts/<name>/ george@parr.local:repos/pi-film-reversal/artifacts/<name>/
  ```

Whichever you choose, use the same absolute path for `PI_ARTIFACT_DIR` in
`.env` and for `--artifacts` in the unit below. Every capture record in
`captures.jsonl` carries the `lut_sha1` of the look that graded it, so you can
always check which one was in use.

### 4.3 Install and start the service

Open `deploy/pifilm-capture.service.example` and check `User`,
`WorkingDirectory`, the venv path and `--artifacts` match your Pi. Then:

```sh
sudo install -m 644 deploy/pifilm-capture.service.example /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-capture.service
sleep 25 && systemctl status pifilm-capture.service --no-pager
journalctl -u pifilm-capture.service -n 30 --no-pager      # the start-up lines
```

The unit runs `pifilm-capture --no-preview --artifacts <dir> --ups x728
--display waveshare35dsi --shutter-gpio 21` (plus the settings flags it
explains). It runs the LCD viewfinder when the
[Waveshare DSI panel](lcd-viewfinder.md#9-the-35-dsi-panel) is fitted and falls
back to a headless service, where the GPIO shutter button still takes photos,
when it is not, or when the camera is USB;
`--no-preview` because it suppresses the OpenCV window, which needs a desktop
session the service does not have (it does not disable the LCD); `--ups x728`
reads the Geekworm fuel gauge for the LCD's battery tag and is dropped
on a Pi without the shield; `Restart=on-failure` with `StartLimitIntervalSec=0`
because the camera can enumerate after the service starts and the retries
must never give up. Start-up takes 15 to 25 seconds on a Pi 3B: OpenCV import,
LUT load, camera warm-up.

For the optional two-screen mode, where a monitor on the Pi also shows each
photo, the same command with `--show-captures` must be launched from the Pi's
desktop session, not over SSH. See [the network and service guide](pi-service.md).

### 4.4 Passwordless restart for the deploy script (optional)

`scripts/deploy_remote.py --restart` on the Mac stops and starts the service
through `sudo -n`, which needs a rule. Two short lines, because a long rule
that soft-wraps on paste becomes a sudoers syntax error:

```sh
sudo tee /etc/sudoers.d/pifilm-capture >/dev/null <<'EOF'
george ALL=(root) NOPASSWD: /usr/bin/systemctl start pifilm-capture.service
george ALL=(root) NOPASSWD: /usr/bin/systemctl stop pifilm-capture.service
EOF
sudo chmod 440 /etc/sudoers.d/pifilm-capture
sudo visudo -c
```

A Pi set up before the `parr` → `pifilm` rename may still have
`/etc/sudoers.d/parr-capture`, naming `parr-capture.service`. It matches nothing
any more, so `sudo -n` asks for a password; `sudo -n -l` shows which rule is
installed. Write the file above and `sudo rm /etc/sudoers.d/parr-capture`.

The script refuses to restart while another process holds a video node. A
desktop session's PipeWire and WirePlumber always do, only to monitor the
devices, and are allowed; anything else (`rpicam-still`, a browser) blocks it.

### 4.5 Reboot test

```sh
sudo reboot
```

About a minute later, from the Mac:

```sh
ssh george@parr.local 'nmcli -t -f NAME,DEVICE connection show --active; systemctl is-active pifilm-capture.service'
```

You want `pifilm-ap:wlan0` and `active`. Everything from here on assumes the Pi
comes back on its own after a power cycle, because in the field that is the
only way it will ever start.

### 4.6 UPS (optional)

For a battery-powered handheld, fit the Geekworm X728 UPS shield. `--ups
x728` on the `pifilm-capture` service (§4.3) is the only project-side switch;
reading the gauge needs the header I2C bus enabled, which is disabled by
default ([docs/x728-ups.md](x728-ups.md) section 4). Geekworm's installer
provides the power-button service, which is unrelated to reading the gauge.
The guide, [docs/x728-ups.md](x728-ups.md), covers that bus setup, the pin
and address map, Geekworm's power-button service, the low-battery shutdown
policy suitable for a camera, the DS1307 real-time clock that finally gives
offline captures the right date, and a verification checklist. With `--ups
x728`, the LCD viewfinder shows the Pi's battery level in its top strip, and
`pifilm-battery` prints it.

### 4.7 Optional: LCD viewfinder

For a live viewfinder with a light-meter readout and an on-screen shutter, fit
the Waveshare 3.5" DSI LCD (E) and run the service with `--display
waveshare35dsi`, which the example unit already does. It needs Waveshare's
overlay file, one `config.txt` line, the Pi booting to the console, and the
service user in `video`, `render` and `input`; the steps, the screen layout and
the hardware acceptance checklist are in
[docs/lcd-viewfinder.md](lcd-viewfinder.md#9-the-35-dsi-panel). The older
Waveshare 2.8" SPI panel is still supported with `--display waveshare28`
(sections 2 and 3 of the same guide).

### 4.8 Optional: shutter button

A push button on the GPIO header takes a photo, like the on-screen shutter. Any
momentary (normally open) button works; it needs two wires and no
resistor, because the Pi's internal pull-up holds the pin high until the button
connects it to ground.

**Wiring**, with the Pi powered off:

| Button lead | Header pin | |
| --- | --- | --- |
| One lead | Physical pin 40 (BCM 21) | The last pin of the outer row |
| The other lead | Physical pin 39 (ground) | The last pin of the inner row, directly opposite |

Which lead goes where does not matter. These two pins are free on this build:
the X728 UPS uses BCM 5, 6, 12, 16, 20 and 26, and the DSI panel uses no header
pins. The two header pins next to these, physical 37 and 38 (BCM 26 and BCM
20), belong to the X728 UPS, so count from the end of the header: pins 39 and 40
are the very last pair. Do not connect the button to a 3.3 V or 5 V pin.

**Service.** The example unit already passes `--shutter-gpio 21`. On an existing
install, add it to the `ExecStart=` line of
`/etc/systemd/system/pifilm-capture.service`, then
`sudo systemctl daemon-reload && sudo systemctl restart pifilm-capture`. The
journal shows `Shutter button on BCM 21.` at start-up, and `button: capture
requested` for each press.

**Behaviour.** One press takes one photo. A press while a photo is still being
processed is ignored (`button: press ignored (busy)`), never queued: a press is
only accepted once the previous photo has finished processing (the colour bars
have cleared), which with double exposure on means waiting for the live view to
return before the second press. A press
with the screen dark from idle takes the photo and wakes the screen to show it.
To use another pin, change the number (BCM 4 to 27); pins that belong to the
X728 are refused with `--ups x728`, and pins that belong to the 2.8" LCD with
`--display waveshare28`. The button also works without an LCD, but it then puts
the app in its capture-only modes: run from the Pi's desktop session, the live
preview window is replaced by a window that shows only each captured photo; with
no screen and no terminal (the headless service) the journal says `Shutter button
active without a screen or terminal controls.` and the process waits for presses.
If the LCD viewfinder fails repeatedly while running, the service exits for
systemd to restart it, which brings the button back too.

**If it does not work.** `warning: shutter button unavailable (...); continuing
without it` in the journal names the cause, and the service carries on without
the button. There are two: `python3-gpiozero is not installed` (install it with
`sudo apt install python3-gpiozero python3-lgpio`), or `cannot claim BCM 21:`
followed by the underlying error, which covers a pin held by another process or
a `dtoverlay`, a service user who is not in the `gpio` group, and a missing GPIO
backend (`python3-lgpio`). To test the wiring without the app, run

```sh
python3 -c "from gpiozero import Button; b = Button(21, bounce_time=0.05); print('waiting'); b.wait_for_press(); print('pressed'); b.wait_for_release(); b.close()"
```

It prints `waiting`, then `pressed` when the button is pressed, and exits when
the button is let go. An earlier form of this test, without the debounce and the
wait for release, printed `pressed` and then aborted with "Fatal Python error
... at interpreter shutdown" on this Pi (2026-10-10); that abort came from the
test exiting while the GPIO library was still delivering edges, and did not
indicate a wiring fault. Once the service
has been given `--shutter-gpio 21` it holds the pin, so stop it first
(`sudo systemctl stop pifilm-capture`) and start it again afterwards.

**Verified on the Pi 4 on 2026-10-10** (button on physical pins 39 and 40, X728
fitted, DSI viewfinder running): the start-up line appears; one press takes one
photo; presses during processing are dropped as `busy`; ten deliberate presses
in five seconds produced no false triggers and no double fires; with `2x` on,
two presses with a wait between them make one composite; a few idle minutes
produced no photo; and stopping the service while the button was being tapped
left no "Fatal Python error" in the journal. Not checked separately: a press
with the screen dark from idle, and a run of ten spaced presses counted against
ten files.

### 4.9 Nextcloud photo sync (optional)

To keep an off-device archive of every capture, push `~/Pictures/pifilm` to a
Nextcloud folder over WebDAV. It runs from the Pi only when it is on the home
network (wired or Wi-Fi), copies (never deletes) so an SD-card cleanup cannot
erase the archive, and keeps the Nextcloud password out of every command line. Fill the `NEXTCLOUD_*`
keys in `.env`, `sudo apt install rclone`, and enable the timer. Full steps,
including the app-password setup and how to test with a dry run, are in
[docs/nextcloud-sync.md](nextcloud-sync.md).

To also upload on demand from the LCD viewfinder's settings menu (the `UPLOAD`
row on its NETWORK page), the example unit passes the same file to the capture service with
`--nextcloud-env /home/george/repos/pi-film-reversal/.env`. The service never reads
it; it only hands the path to the sync it starts. Drop the flag from the unit if
Nextcloud is not set up, and the row is left out. After changing the unit:
`sudo systemctl daemon-reload && sudo systemctl restart pifilm-capture`.

## 5. Training a look (optional)

Skip this section entirely unless you want to replace the bundled starter. The
camera does not need it. This section is the outline; the full step-by-step
procedure, from an empty `data/` folder to a verified deployment, with the
report explained gate by gate, is [the training guide](training.md).

### 5.1 What training does and does not do

`pifilm-train` fits a 33³ colour lookup table so that the colour distribution of
your camera's frames moves toward the colour distribution of a folder of
reference photographs. It learns global colour and tone. It cannot learn flash
lighting, composition, or subjects, and it cannot add colour to a scene that
has none. The current roadmap for better results is in `todo.md`, section 1;
the short version is: real camera captures as the source, a coherent reference
set, and a flash.

### 5.2 Data layout

Both folders are gitignored. Nothing in them is ever committed.

| Folder | Contents | Minimum | Notes |
|---|---|---|---|
| `data/source/` | Ungraded frames from the Innomaker camera | 30 images, 50+ better | Use the `*_ungraded.jpg` or `*_original.jpg` files from `~/Pictures/pifilm/` on the Pi, across many scenes and lighting conditions. Not web photos. |
| `data/references/` | Finished reference photographs in the target look | 200 images | One coherent period and lighting style. Only files you have permission to use. See [reference sources](reference-sources.md). |

Pull camera frames from the Pi with:

```sh
rsync -av --include='*/' --include='*_ungraded.jpg' --include='*_original.jpg' --exclude='*' \
  george@parr.local:Pictures/pifilm/ data/source-raw/
```

then select and copy the frames you want into `data/source/`.

### 5.3 Train, inspect, test

```bash
.venv/bin/pifilm-train --source data/source --target data/references --out artifacts/pifilm-v1
cat artifacts/pifilm-v1/report/summary.txt
open artifacts/pifilm-v1/report/contact_sheet.png
.venv/bin/pifilm-process /path/to/test-frames /path/to/results --artifacts artifacts/pifilm-v1
```

Exit codes: 0 all gates passed; 1 the corpus or arguments were unusable; 3 an
artifact was written but a quality gate failed, and the report names it. Do not
deploy a 3. Useful flags: `--strength 0.8` weakens the learned grade,
`--grain-strength 0` removes grain, `--allow-small` permits an exploratory fit
below the minimums and marks the held-out evaluation as unavailable,
`--proxy-source` records that the source frames are stand-ins rather than
camera captures. The three dated `docs/training-*.md` reports show what past
runs produced and why two of them were rejected.

### 5.4 Deploy a trained look to the Pi

```sh
rsync -av artifacts/pifilm-v1/ george@parr.local:repos/pi-film-reversal/artifacts/pifilm-v1/
```

On the Pi, set `--artifacts` in `/etc/systemd/system/pifilm-capture.service` to
that directory, `daemon-reload`, restart, and take a capture. Its record in
`captures.jsonl` must show the new `lut_sha1`. Update `PI_ARTIFACT_DIR` in
`.env` to match.

## 6. Day to day

- **Power on.** Hotspot up in 10 to 20 s, the service ready (viewfinder live) in
  about 30 s.
- **Pull photos.** Plug in the cable: `rsync -av george@parr.local:Pictures/pifilm/ ~/Pictures/pifilm-pi/`
- **Inspect or restart remotely.** `.venv/bin/python scripts/deploy_remote.py --env .env`
  inspects; add `--restart` for the guarded restart (needs 4.4).
- **Logs.** On the Pi: `journalctl -u pifilm-capture.service -n 100`.
- **Change the hotspot passphrase.** Edit `.env`, then re-run
  `pi_hotspot.py --apply` over Ethernet.
- **Back on home Wi-Fi temporarily.** Over Ethernet: `sudo nmcli connection down pifilm-ap`
  then `sudo nmcli connection up netplan-wlan0-<SSID>`; reverse to return.
- **Stop.** `sudo systemctl stop pifilm-capture.service`.

## 7. The order, on one page

1. Mac: clone, venv, `pip install -e '.[train,dev,deploy]'`, tests pass.
2. `.env`: copy from example, fill in, `chmod 600`.
3. Pi OS: flash Trixie, first boot on home Wi-Fi, update.
4. Pi network: hostname `pifilm`, wired link-local, Wi-Fi country, reboot, confirm `ssh parr.local` over the cable.
5. Pi checkout: clone, copy `.env`.
6. Pi app: venv with system packages, `pip install -e .`, `--fake` smoke test.
7. Pi hotspot: dry run, `--apply`, disable the home profile.
8. Pi look: `pifilm/data` or rsync a trained artifact; paths agree in `.env` and the unit.
9. Pi service: install unit, enable, check its start-up lines in the journal.
10. Pi sudoers rule (optional), then reboot test: hotspot and service return unattended.
11. Later, optionally: collect camera frames, curate references, train, inspect, deploy, verify `lut_sha1`.
