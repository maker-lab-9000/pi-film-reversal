# StickS3 remote capture and Pi deployment

This guide connects the M5StickS3 button to the Pi's authenticated capture API.
The Pi runs its own Wi-Fi hotspot and the Stick joins it directly; no home
network sits between them. Administration (SSH, photo transfer, updates) goes
over an Ethernet cable, either straight into a laptop or into a router.

The examples use account `george`, hostname `pifilm`, project checkout
`/home/george/repos/pi-film-reversal`, hotspot `pifilm-cam` at `10.42.0.1`.
Keep credentials in the ignored local `.env`; never paste them into a shell
command, firmware source, service unit, terminal transcript, or Git commit.

## Local configuration

On the Mac or Pi checkout, make the local file with restrictive permissions:

```sh
cp .env.example .env
chmod 600 .env
```

Set these values in `.env`:

```dotenv
PI_HOST=parr.local
PI_USER=george
PI_PROJECT_DIR=/home/george/repos/pi-film-reversal
PI_SSH_PASSWORD=replace-locally
WIFI_SSID=pifilm-cam
WIFI_PASSWORD=replace-locally-8-to-63-ascii-characters
PIFILM_REMOTE_URL=http://10.42.0.1:8765
PIFILM_LISTEN=0.0.0.0:8765
PIFILM_REMOTE_TOKEN=replace-locally-with-a-long-random-token
PI_ARTIFACT_DIR=/home/george/repos/pi-film-reversal/pifilm/data
```

What each group means in this topology:

- `PI_HOST` is where SSH goes. Over Ethernet with avahi running on the Pi that
  is the mDNS name `parr.local`; a plain IP address also works.
- `WIFI_SSID` and `WIFI_PASSWORD` are the **Pi's hotspot** credentials. The
  Stick joins this network. WPA2 requires 8 to 63 printable ASCII characters.
- `PIFILM_REMOTE_URL` is the address the Stick calls. Its host must be a plain
  IPv4 address, and `scripts/pi_hotspot.py` gives the hotspot exactly that
  address, so the two can never disagree.
- `PIFILM_LISTEN` is where `pifilm-capture` binds on the Pi. `0.0.0.0:8765`
  listens on both the hotspot and Ethernet and avoids a boot-order race with
  NetworkManager bringing the hotspot address up.
- `PI_ARTIFACT_DIR` is the look the service loads. `pifilm/data` is the bundled
  untrained starter and exists in every checkout. A trained model under
  `artifacts/` is gitignored and must be copied to the Pi first. Whatever you
  choose, use the same path in the service unit's `--artifacts` and verify it
  on the Pi before starting capture: it must contain both `params.json` and
  `pifilm.cube`. Each capture's `captures.jsonl` record carries the `lut_sha1` of
  the look that graded it, so you can always tell which one was in use.

The scripts parse this file as `KEY=VALUE` data. They do not source it, expand
variables, run command substitutions, or execute any of its contents. The Pi
SSH password is used only for the SSH connection. The firmware generator passes
only Wi-Fi SSID, Wi-Fi password, API URL, and remote token to PlatformIO; it
never writes a credentials file and never passes `PI_SSH_PASSWORD` to firmware.
The hotspot script writes a root-only NetworkManager keyfile rather than passing
the passphrase on a command line.

Install the optional, cross-platform SSH dependency before using deployment:

```sh
.venv/bin/python --version  # requires Python 3.11 or newer
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[deploy]'
```

If the environment uses an older Python, preserve it and recreate it first
(on macOS with Python 3.12 installed):

```sh
mv .venv .venv-old
python3.12 -m venv .venv
```

Then run the installation commands above. Pip 21.2.4 cannot install this
project in editable mode; upgrading pip fixes that error only when the Python
version also meets the project requirement.

## Network topology: Pi hotspot and Ethernet administration

The Pi 3B's onboard radio (2.4 GHz b/g/n) supports access-point mode with the
standard `brcmfmac` driver. The ESP32-S3 in the Stick is 2.4 GHz only, and the
firmware joins with `WiFi.begin(ssid, password)`, which is plain WPA2-PSK. No
firmware code changes are needed; only a rebuild with the hotspot's SSID,
passphrase and API URL.

One radio cannot reliably be a client of your home Wi-Fi and an access point at
the same time. **Activating the hotspot disconnects the Pi from home Wi-Fi.**
Do the steps below with an Ethernet cable connected, in this order.

Verified 2026-09-08 on a Pi 3B running Raspberry Pi OS Trixie (Debian 13),
NetworkManager 1.52.1 with netplan-generated profiles, brcmfmac firmware
7.45.98. Hotspot and capture service both return unattended after a reboot.

### 1. Ethernet administration (do this first)

Plug the Pi into your router, or directly into the laptop (the Pi 3B port is
auto-MDI-X, so any patch cable works). Log in and give the Pi a memorable name:

```sh
sudo raspi-config nonint do_hostname pifilm
```

Make the wired connection usable both behind a router (DHCP) and on a direct
cable with no DHCP server (IPv4 link-local, `169.254.x.y`):

```sh
nmcli -t -f NAME,TYPE connection show | grep ethernet   # find the wired profile name
sudo nmcli connection modify <wired-profile> ipv4.method auto ipv4.link-local enabled
nmcli -f ipv4.method,ipv4.link-local connection show <wired-profile>
```

The profile name depends on the OS release: Bookworm images call it
`Wired connection 1`; Trixie images, where netplan generates the
NetworkManager profiles, call it `netplan-eth0`. On Trixie, check that the
change was persisted and not only applied to the generated copy in `/run`:
a new `/etc/netplan/90-NM-<uuid>.yaml` (or a keyfile under
`/etc/NetworkManager/system-connections/`) should appear, and the property
must still read `enabled` after a reboot.

`ipv4.link-local` needs NetworkManager 1.40 or newer (`nmcli --version`);
Bookworm ships 1.42 and Trixie 1.52. If the property is missing, add a second
static profile on `192.168.7.2/24` and set the Mac's Ethernet adapter to
`192.168.7.1` instead.

avahi is installed by default on Raspberry Pi OS, and macOS resolves `.local`
names natively, so after a reboot both of these work with a direct cable:

```sh
ssh george@parr.local
ping parr.local
```

To give the Pi internet through the laptop for `apt` and `pip`, enable macOS
Internet Sharing from Wi-Fi to the Ethernet port; the Pi then receives a
`192.168.2.x` address over DHCP.

### 2. Wi-Fi country

Without a regulatory domain the radio stays soft-blocked and the hotspot will
not come up. Use your two-letter country code:

```sh
sudo raspi-config nonint do_wifi_country DE
```

### 3. Create the hotspot from `.env`

Copy the completed `.env` to the Pi checkout (mode 600), then preview the
NetworkManager keyfile the script will write. The passphrase is redacted in the
preview and nothing is written:

```sh
cd /home/george/repos/pi-film-reversal
.venv/bin/python scripts/pi_hotspot.py --env .env
```

With the Ethernet session confirmed working, apply it:

```sh
sudo .venv/bin/python scripts/pi_hotspot.py --env .env --apply
nmcli connection show --active
```

The script writes `/etc/NetworkManager/system-connections/pifilm-ap.nmconnection`
with mode 0600, reloads NetworkManager and activates `pifilm-ap`. The profile is
an access point on `wlan0`, WPA2-PSK only, `ipv4.method shared` (NetworkManager
runs a DHCP server for the Stick on the `/24` around the hotspot address and
NATs to Ethernet when that is up), `autoconnect` with priority 10. Re-running
with the same SSID produces the same connection UUID, so applying twice
replaces rather than duplicates.

The profile also sets `pmf=1`, which disables Protected Management Frames.
NetworkManager's default is "optional", and that makes hostapd install an
AES-CMAC group key when the access point starts. The Pi 3B and Zero W radio
(BCM43430, firmware 7.45.98) has no management-frame protection, `iw list`
shows no CMAC cipher, and the kernel rejects the key. The visible symptom is
`nmcli` failing with "802.1X supplicant took too long to authenticate" and
`journalctl -u wpa_supplicant` showing `key setting validation failed` followed
by `Failed to initialize AP interface`. The Pi 3B+ and Pi 4 radios do support
it, which is why generic hotspot recipes work there without this line. The
Stick does not use PMF, so nothing is lost.

If the Pi previously joined your home Wi-Fi, stop that profile from competing
at boot. The Imager-created profile is named `preconfigured` on Bookworm and
`netplan-wlan0-<SSID>` on Trixie; take the name from the listing:

```sh
nmcli -t -f NAME,TYPE connection show | grep wireless
sudo nmcli connection modify <home-wifi-profile> connection.autoconnect no
```

Expect 10 to 20 seconds after Pi boot before the hotspot is up. The firmware
already backs off reconnects between 1 and 10 seconds, so the Stick settles
into READY on its own.

### 4. Point the service and the firmware at the hotspot

The systemd example already listens on `0.0.0.0:8765`. Install or update it as
described in the next section, then rebuild and flash the Stick as described in
"Generate and flash StickS3 configuration" so it carries the hotspot SSID,
passphrase and `http://10.42.0.1:8765`.

The API is plain HTTP with a bearer token. On a private WPA2 hotspot with a
strong passphrase that is acceptable; keep the token anyway so a guest on the
hotspot cannot fire the shutter. The Ethernet side also sees port 8765 when
listening on `0.0.0.0`; the same token protects it.

### 5. Verify

From a phone or laptop joined to `pifilm-cam` (or from the Pi itself):

```sh
curl -s -H "Authorization: Bearer $PIFILM_REMOTE_TOKEN" http://10.42.0.1:8765/v1/status
```

Expect `"ready":true`. `"pi_battery"` is always present; it is `null` unless
the service runs with `--ups x728`. Allow up to 30 seconds after a boot
before the port answers: the service imports OpenCV, loads the LUT and warms
up the camera first, and on a Pi 3B that takes 15 to 25 seconds. Then power
the Stick: colour
bars, READY, one capture per deliberate press, graded thumbnail displayed. The
Stick's serial log shows each step; `firmware/sticks3/README.md` explains how to
read it and how to prove the displayed image is the graded file. The small side
button (BtnB) turns the Stick's screen off and back on to save battery; the
shutter keeps working while the screen is dark, and a new photo does not wake
it. Pull photos
over Ethernet with:

```sh
rsync -av george@parr.local:Pictures/pifilm/ ~/Pictures/pifilm-pi/
```

## Start the Pi capture process

Add the Pi host key before deployment, by checking its fingerprint in person
and accepting it once with normal OpenSSH:

```sh
ssh george@parr.local
```

`scripts/deploy_remote.py` deliberately uses Paramiko with your system known
hosts and a reject-unknown-host policy. It will not silently trust a changed or
unknown host key.

First check the Pi's existing state without restarting anything:

```sh
.venv/bin/python scripts/deploy_remote.py --env .env
```

It checks the project directory, selected artifact, existing `pifilm-capture`
processes, `/dev/video*` owners, and logged-in desktop sessions before a restart
is considered. Resolve a reported camera owner first; this prevents two capture
processes from competing for the same camera.

For **both-screen mode** (Pi screen plus Stick), run the following command from
the actual logged-in Pi desktop session or that session's autostart entry:

```sh
.venv/bin/pifilm-capture --no-preview --show-captures --remote-listen 0.0.0.0:8765 --artifacts <verified-pi-artifact-directory>
```

Replace the placeholder with the verified `PI_ARTIFACT_DIR`. The token must be
present in that desktop session's environment as `PIFILM_REMOTE_TOKEN`, but do not
put the token literal in the command. An SSH login does not inherit `DISPLAY`,
`WAYLAND_DISPLAY`, `XAUTHORITY`, or the desktop display authorization, so do not
launch `--show-captures` through SSH and expect it to appear on the Pi screen.
Use a desktop autostart entry or launch it manually after logging into that
desktop session.

For **Stick-only mode**, omit `--show-captures` and use the supplied headless
systemd example. On the Pi, create `/etc/pifilm-capture.env` as root-owned mode
`600` containing only `PIFILM_REMOTE_TOKEN=...`, then install and enable the unit:

```sh
sudo install -m 644 deploy/pifilm-capture.service.example /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-capture.service
sudo systemctl status pifilm-capture.service
```

The service is intentionally headless and has `StandardInput=null`; it needs no
terminal controls. If `journalctl -u pifilm-capture.service` shows
`error: --remote-listen requires PIFILM_REMOTE_TOKEN` in a restart loop, the
environment file is not delivering the variable. Check its shape without
printing the secret, and confirm the key is spelled exactly `PIFILM_REMOTE_TOKEN`
with no `export`, quotes, or Windows line endings:

```sh
sudo sed -E 's/=.*/=<redacted>/' /etc/pifilm-capture.env | cat -A
```

The guarded remote restart below runs `sudo -n systemctl`, which needs a
passwordless rule for exactly those two commands. The default Raspberry Pi OS
user does not have one, so add it once on the Pi. Two short rules rather than
one long line: sudoers has no implicit line continuation, and a long rule that
soft-wraps on paste becomes a syntax error that disables the rule.

```sh
sudo tee /etc/sudoers.d/pifilm-capture >/dev/null <<'EOF'
george ALL=(root) NOPASSWD: /usr/bin/systemctl start pifilm-capture.service
george ALL=(root) NOPASSWD: /usr/bin/systemctl stop pifilm-capture.service
EOF
sudo chmod 440 /etc/sudoers.d/pifilm-capture
sudo visudo -c
sudo -k && sudo -n systemctl start pifilm-capture.service && echo passwordless-ok
```

If `visudo -c` reports a syntax error, fix the file with
`sudo visudo -f /etc/sudoers.d/pifilm-capture`. Should `sudo` itself refuse to
run after a sudoers mistake, `pkexec rm /etc/sudoers.d/pifilm-capture` removes
the file through polkit instead.

With that in place, a guarded remote restart is available:

```sh
.venv/bin/python scripts/deploy_remote.py --env .env --restart
```

The script refuses to stop the service when its initial inspection finds a
foreign camera/capture owner. It permits only the service's own known process,
then stops it, verifies that no process or camera owner remains, and only then
starts it. For a service change, run the inspection command again afterward and use
`journalctl -u pifilm-capture.service -n 100` on the Pi for diagnostics.

## Generate and flash StickS3 configuration

Validate the local configuration without printing or writing secrets:

```sh
python3 firmware/sticks3/scripts/generate_config.py --env .env
pio run -d firmware/sticks3
```

The PlatformIO pre-build script reads `.env` and appends the four device defines
while retaining the base environment build flags. `PIFILM_REMOTE_URL` becomes the
API base the Stick calls; when it is absent the generator falls back to
`http://<PI_HOST>:8765`, which is wrong in the hotspot topology, so keep
`PIFILM_REMOTE_URL` set. On a clean clone with no `.env`, it adds no defines and
the firmware keeps its safe empty defaults; the explicit validation command
above still requires a complete env file. First USB flash and serial
diagnostics are:

```sh
pio run -d firmware/sticks3 -t upload
pio device monitor -d firmware/sticks3 -b 115200
```

Confirm `StickS3 remote capture boot`, the boot screen/tone, and colour bars.
Then confirm the device joins `pifilm-cam`, reaches the Pi, and produces one
capture per deliberate button press. See `firmware/sticks3/README.md` for the
expected screen and button behaviour.

## Stop, rollback, and replace a token

For desktop mode, stop the process in its desktop session with `Q` or Escape.
For headless mode, use:

```sh
sudo systemctl stop pifilm-capture.service
```

To roll back, first stop the current process/service and verify `/dev/video*`
has no owner. Then start the previously recorded working capture command from
the same desktop context, or restore the prior service unit and run `daemon-reload`
before starting it. Do not start the old command until the new process has
stopped.

To return the Pi to your home Wi-Fi temporarily, connect over Ethernet and run
`sudo nmcli connection down pifilm-ap` followed by
`sudo nmcli connection up <home-wifi-profile>`. Re-enable the hotspot with
`sudo nmcli connection up pifilm-ap`.

To replace a compromised or expired token, generate a new long random token,
replace the local `PIFILM_REMOTE_TOKEN` in `.env`, update the secured Pi service
environment (or desktop session environment), restart the Pi process only after
inspection, then rebuild and flash the StickS3. The old token stops working as
soon as the Pi process has been restarted with the new value. To change the
hotspot passphrase, edit `WIFI_PASSWORD` in `.env`, re-run the hotspot script
with `--apply` over Ethernet, then rebuild and flash the Stick.
