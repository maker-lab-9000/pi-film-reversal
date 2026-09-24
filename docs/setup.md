# Setup from zero: Mac, Raspberry Pi, StickS3

This is the ordered install guide. Each section depends on the one before it,
so do them in order. Training is optional and comes last: the camera works
with the bundled starter look without any training data.

Verified 2026-09-08 on: macOS with Python 3.12 and PlatformIO 6.1; Raspberry
Pi 3B on Raspberry Pi OS Trixie (Debian 13, NetworkManager 1.52 with netplan);
M5StickS3 with M5Unified.

## 0. What runs where, and why

| Machine | Runs | Why here |
|---|---|---|
| Mac | Development, tests, training, firmware build and flash, remote administration of the Pi | Training needs SciPy and a fast CPU. The PlatformIO toolchain lives here. There is one `.env` file, and it lives here. |
| Raspberry Pi 3B | `pifilm-capture` as a systemd service: camera, grading, storage, the HTTP API, and its own Wi-Fi hotspot | The camera is plugged into it. It must boot unattended on battery. Its OpenCV comes from apt because the pip wheel has no GUI support for two-screen mode. |
| M5StickS3 | Firmware: shutter button, status screen, last-photo thumbnail, battery badge | No compute of its own. It only talks to the Pi over the Pi's hotspot. Its Wi-Fi and token are baked in at build time. |

In the field there is no home network. The Pi is the access point, the Stick is
its only client, and the Pi's Ethernet port is how you administer it and pull
photos. The README's "From button press to displayed photo" section describes
what happens at capture time; this guide is about getting there.

## 1. Mac workstation

Everything else is driven from here, including the Pi's install over Ethernet
and the Stick's flash over USB.

```bash
git clone <repo-url> pi-film-reversal
cd pi-film-reversal
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[train,dev,deploy]'
brew install platformio          # or: pip install platformio
```

The extras: `train` adds SciPy and requests for `pifilm-train`, `dev` adds
pytest, ruff and build, `deploy` adds Paramiko for `scripts/deploy_remote.py`.

Verify before going on:

```bash
.venv/bin/pytest -q -m 'not slow'          # Python suite, about 40 s
pio test -d firmware/sticks3 -e native      # firmware state machine, host-side
```

## 2. The `.env` file: one file, three consumers

Copy the example and fill it in. It is gitignored; never paste a value from it
into a command, a source file, a service unit, or a commit.

```bash
cp .env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # a token value
```

| Key | Read by | Must match |
|---|---|---|
| `PI_HOST`, `PI_USER`, `PI_PROJECT_DIR`, `PI_SSH_PASSWORD` | `scripts/deploy_remote.py` on the Mac | The Pi's hostname (`parr.local` over Ethernet), account, and checkout path |
| `WIFI_SSID`, `WIFI_PASSWORD` | `scripts/pi_hotspot.py` on the Pi (creates the hotspot) and the firmware build (joins it) | Each other, by construction. WPA2 needs 8 to 63 printable ASCII characters. |
| `PIFILM_REMOTE_URL` | Firmware build (the address the Stick calls) and `pi_hotspot.py` (the hotspot's own address) | A plain IPv4 host, `http://10.42.0.1:8765` |
| `PIFILM_LISTEN` | `deploy_remote.py`; mirrors the unit's `--remote-listen` | `0.0.0.0:8765` |
| `PIFILM_REMOTE_TOKEN` | Firmware build, and the Pi's `/etc/pifilm-capture.env` | Identical on both sides, or every request gets 401 |
| `PI_ARTIFACT_DIR` | `deploy_remote.py` inspection; mirrors the unit's `--artifacts` | A directory on the Pi holding `params.json` and `pifilm.cube` |

Why one file: the hotspot the Pi creates, the network the Stick joins, the
address the Stick calls, and the address the Pi owns must all agree. They do
because they come from one place. Change `.env`, and you re-run the hotspot
script on the Pi and rebuild the Stick.

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
WPA2, DHCP for clients on `10.42.0.0/24`, autoconnect at boot. It also disables
Protected Management Frames, without which the Pi 3B's radio cannot start an
access point at all. Details, the netplan notes and the recovery commands are in
[the network and deployment guide](sticks3-remote.md).

Why the Pi is the access point: the camera must work with no infrastructure,
and the Stick is 2.4 GHz only, which the Pi 3B radio provides.

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
and `_graded.jpg` (the graded output). `captures.jsonl` gains
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

### 4.3 The token file

The service must not have the token on its command line, so it reads it from a
root-only environment file. The key must be spelled exactly, with no `export`,
quotes or Windows line endings; a misspelling puts the service in a restart
loop with `error: --remote-listen requires PIFILM_REMOTE_TOKEN`.

```sh
sudo install -m 600 /dev/null /etc/pifilm-capture.env
sudoedit /etc/pifilm-capture.env          # one line: PIFILM_REMOTE_TOKEN=<same value as .env>
sudo sed -E 's/=.*/=<redacted>/' /etc/pifilm-capture.env | cat -A   # check the shape
```

### 4.4 Install and start the service

Open `deploy/pifilm-capture.service.example` and check `User`,
`WorkingDirectory`, the venv path and `--artifacts` match your Pi. Then:

```sh
sudo install -m 644 deploy/pifilm-capture.service.example /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-capture.service
sleep 25 && systemctl status pifilm-capture.service --no-pager
sudo ss -ltnp | grep 8765                                   # 0.0.0.0:8765 by pifilm-capture
curl -s -o /dev/null -w '%{http_code}\n' http://10.42.0.1:8765/v1/status   # 401 without token
```

The unit runs `pifilm-capture --no-preview --remote-listen 0.0.0.0:8765
--artifacts <dir> --ups x728 --display waveshare28`. It runs the LCD viewfinder
when the [Waveshare panel](lcd-viewfinder.md) is fitted and falls back to
headless Stick-only service when it is not, or when the camera is USB;
`--no-preview` because it suppresses the OpenCV window, which needs a desktop
session the service does not have (it does not disable the LCD);
`0.0.0.0` because the hotspot address may not exist yet at the moment the
service starts, and binding to all interfaces avoids that race; `--ups x728`
reads the Geekworm fuel gauge for the Stick's Pi-battery badge and is dropped
on a Pi without the shield; `Restart=on-failure` with `StartLimitIntervalSec=0`
because the camera can enumerate after the service starts and the retries
must never give up. Start-up takes 15 to 25 seconds on a Pi 3B: OpenCV import,
LUT load, camera warm-up.

For the optional two-screen mode, where a monitor on the Pi also shows each
photo, the same command with `--show-captures` must be launched from the Pi's
desktop session, not over SSH. See the deployment guide.

### 4.5 Passwordless restart for the deploy script (optional)

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

### 4.6 Reboot test

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

### 4.7 UPS (optional)

For a battery-powered handheld, fit the Geekworm X728 UPS shield. `--ups
x728` on the `pifilm-capture` service (§4.4) is the only project-side switch;
reading the gauge needs the header I2C bus enabled, which is disabled by
default ([docs/x728-ups.md](x728-ups.md) section 4). Geekworm's installer
provides the power-button service, which is unrelated to reading the gauge.
The guide, [docs/x728-ups.md](x728-ups.md), covers that bus setup, the pin
and address map, Geekworm's power-button service, the low-battery shutdown
policy suitable for a camera, the DS1307 real-time clock that finally gives
offline captures the right date, and a verification checklist. With `--ups
x728`, the service publishes `pi_battery` and the Stick shows it bottom-left
as `Pi 77%`, `Pi 77%+` on external power, and `Pi --%` when unknown.

### 4.8 Optional: LCD viewfinder

For a live viewfinder with a light-meter readout and an on-screen shutter,
wire up the Waveshare 2.8" Capacitive Touch LCD and add `--display waveshare28`
to the service (already in the example unit). It needs `sudo usermod -aG
spi,i2c,gpio george` and `sudo apt install python3-spidev python3-smbus2
python3-gpiozero python3-lgpio`; full wiring, `config.txt` requirements and the
hardware acceptance checklist are in [docs/lcd-viewfinder.md](lcd-viewfinder.md).

### 4.9 Nextcloud photo sync (optional)

To keep an off-device archive of every capture, push `~/Pictures/pifilm` to a
Nextcloud folder over WebDAV. It runs from the Pi only when the wired LAN is
up, copies (never deletes) so an SD-card cleanup cannot erase the archive, and
keeps the Nextcloud password out of every command line. Fill the `NEXTCLOUD_*`
keys in `.env`, `sudo apt install rclone`, and enable the timer. Full steps,
including the app-password setup and how to test with a dry run, are in
[docs/nextcloud-sync.md](nextcloud-sync.md).

## 5. StickS3

Last, because it has nothing to show until the Pi is up, and because its
network credentials and token are compiled in from `.env`.

Connect the Stick over USB-C to the Mac:

```sh
python3 firmware/sticks3/scripts/generate_config.py --env .env    # validates, prints no secrets
pio run -d firmware/sticks3 -t upload
pio device monitor -d firmware/sticks3 -b 115200
```

On the serial monitor expect `StickS3 remote capture boot`, a battery line,
then `status: HTTP 200 ready=1` and `state CONNECTING -> READY`. The screen
shows colour bars with READY and the battery badge in the corner. Press the
button once: one shutter tone, colour bars while waiting, then the graded
thumbnail. The full healthy log, what each early stop means, and a hash recipe
that proves the thumbnail came from the graded file are in
[the firmware README](../firmware/sticks3/README.md).

Rebuild and reflash after any change to `WIFI_SSID`, `WIFI_PASSWORD`,
`PIFILM_REMOTE_URL` or `PIFILM_REMOTE_TOKEN`. Build outputs contain those values;
keep them off shared machines.

## 6. Training a look (optional)

Skip this section entirely unless you want to replace the bundled starter. The
camera does not need it. This section is the outline; the full step-by-step
procedure, from an empty `data/` folder to a verified deployment, with the
report explained gate by gate, is [the training guide](training.md).

### 6.1 What training does and does not do

`pifilm-train` fits a 33³ colour lookup table so that the colour distribution of
your camera's frames moves toward the colour distribution of a folder of
reference photographs. It learns global colour and tone. It cannot learn flash
lighting, composition, or subjects, and it cannot add colour to a scene that
has none. The current roadmap for better results is in `todo.md`, section 1;
the short version is: real camera captures as the source, a coherent reference
set, and a flash.

### 6.2 Data layout

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

### 6.3 Train, inspect, test

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

### 6.4 Deploy a trained look to the Pi

```sh
rsync -av artifacts/pifilm-v1/ george@parr.local:repos/pi-film-reversal/artifacts/pifilm-v1/
```

On the Pi, set `--artifacts` in `/etc/systemd/system/pifilm-capture.service` to
that directory, `daemon-reload`, restart, and take a capture. Its record in
`captures.jsonl` must show the new `lut_sha1`. Update `PI_ARTIFACT_DIR` in
`.env` to match.

## 7. Day to day

- **Power on.** Hotspot up in 10 to 20 s, service answering in about 30 s, Stick
  READY shortly after. The Stick retries on its own.
- **Pull photos.** Plug in the cable: `rsync -av george@parr.local:Pictures/pifilm/ ~/Pictures/pifilm-pi/`
- **Inspect or restart remotely.** `.venv/bin/python scripts/deploy_remote.py --env .env`
  inspects; add `--restart` for the guarded restart (needs 4.5).
- **Logs.** On the Pi: `journalctl -u pifilm-capture.service -n 100`. On the Stick:
  the serial monitor.
- **Change the token or passphrase.** Edit `.env`; update `/etc/pifilm-capture.env`
  and restart the service (token), or re-run `pi_hotspot.py --apply` over
  Ethernet (passphrase); rebuild and reflash the Stick.
- **Back on home Wi-Fi temporarily.** Over Ethernet: `sudo nmcli connection down pifilm-ap`
  then `sudo nmcli connection up netplan-wlan0-<SSID>`; reverse to return.
- **Stop.** `sudo systemctl stop pifilm-capture.service`.

## 8. The order, on one page

1. Mac: clone, venv, `pip install -e '.[train,dev,deploy]'`, PlatformIO, tests pass.
2. `.env`: copy from example, fill in, `chmod 600`. Generate the token here.
3. Pi OS: flash Trixie, first boot on home Wi-Fi, update.
4. Pi network: hostname `pifilm`, wired link-local, Wi-Fi country, reboot, confirm `ssh parr.local` over the cable.
5. Pi checkout: clone, copy `.env`.
6. Pi app: venv with system packages, `pip install -e .`, `--fake` smoke test.
7. Pi hotspot: dry run, `--apply`, disable the home profile.
8. Pi look: `pifilm/data` or rsync a trained artifact; paths agree in `.env` and the unit.
9. Pi service: token file, install unit, enable, check port 8765 and a 401 without token.
10. Pi sudoers rule (optional), then reboot test: hotspot and service return unattended.
11. Stick: validate `.env`, build, flash, watch the log, take a photo, verify it is graded.
12. Later, optionally: collect camera frames, curate references, train, inspect, deploy, verify `lut_sha1`.
