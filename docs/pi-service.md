# Pi network and capture service

This guide covers how the Pi is reached and how its capture service is
installed and restarted. Administration (SSH, photo transfer, updates) goes
over an Ethernet cable at home, either straight into a laptop or into a router.
In the field, where there is no cable, the Pi runs its own Wi-Fi hotspot and a
phone or laptop joins it to SSH in.

The examples use account `george`, hostname `pifilm`, project checkout
`/home/george/repos/pi-film-reversal`, hotspot `pifilm-cam` at `10.42.0.1`.
Keep credentials in the ignored local `.env`; never paste them into a shell
command, source file, service unit, terminal transcript, or Git commit.

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
# PI_HOTSPOT_ADDRESS=10.42.0.1
PI_ARTIFACT_DIR=/home/george/repos/pi-film-reversal/pifilm/data
```

What each group means:

- `PI_HOST` is where SSH goes. Over Ethernet with avahi running on the Pi that
  is the mDNS name `parr.local`; a plain IP address also works (in the field,
  the hotspot address).
- `WIFI_SSID` and `WIFI_PASSWORD` are the **Pi's hotspot** credentials. A phone
  or laptop joins this network. WPA2 requires 8 to 63 printable ASCII characters.
- `PI_HOTSPOT_ADDRESS` (optional, default `10.42.0.1`) is the address the Pi
  owns on its hotspot. It must be a plain IPv4 address; the hotspot serves the
  `/24` around it.
- `PI_ARTIFACT_DIR` is the look the service loads. `pifilm/data` is the bundled
  untrained starter and exists in every checkout. A trained model under
  `artifacts/` is gitignored and must be copied to the Pi first. Whatever you
  choose, use the same path in the service unit's `--artifacts` and verify it
  on the Pi before starting capture: it must contain both `params.json` and
  `pifilm.cube`. Each capture's `captures.jsonl` record carries the `lut_sha1` of
  the look that graded it, so you can always tell which one was in use.

The scripts parse this file as `KEY=VALUE` data (`pifilm/envfile.py`). They do
not source it, expand variables, run command substitutions, or execute any of
its contents. The Pi SSH password is used only for the SSH connection. The
hotspot script writes a root-only NetworkManager keyfile rather than passing
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

## Network topology: Ethernet administration and a field hotspot

The Pi's onboard radio (2.4 GHz b/g/n on the Pi 3B) supports access-point mode
with the standard `brcmfmac` driver. Phones and laptops join with plain
WPA2-PSK.

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
runs a DHCP server for clients on the `/24` around `PI_HOTSPOT_ADDRESS` and
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
it, which is why generic hotspot recipes work there without this line. Clients
connect without PMF, so nothing is lost.

It sets `powersave=2` as well, which disables Wi-Fi power save on the access
point. The brcmfmac driver otherwise turns power management on for `wlan0`
(seen when Ethernet's carrier drops), and the hotspot then drops clients: a
client associates and gets a lease, then re-associates every few minutes and
its connections stall. Measured on a Pi 4, 2026-09-19.

If the Pi previously joined your home Wi-Fi, stop that profile from competing
at boot. The Imager-created profile is named `preconfigured` on Bookworm and
`netplan-wlan0-<SSID>` on Trixie; take the name from the listing:

```sh
nmcli -t -f NAME,TYPE connection show | grep wireless
sudo nmcli connection modify <home-wifi-profile> connection.autoconnect no
```

Expect 10 to 20 seconds after Pi boot before the hotspot is up.

### 4. Verify

From a phone or laptop joined to `pifilm-cam`:

```sh
ssh george@10.42.0.1
```

The LCD viewfinder's settings menu shows the Pi's addresses under NETWORK
(`WLAN 10.42.0.1` while the hotspot is up). Pull photos over Ethernet with:

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

For **two-screen mode** (the graded photo on a monitor attached to the Pi), run
the following command from the actual logged-in Pi desktop session or that
session's autostart entry:

```sh
.venv/bin/pifilm-capture --no-preview --show-captures --artifacts <verified-pi-artifact-directory>
```

Replace the placeholder with the verified `PI_ARTIFACT_DIR`. An SSH login does
not inherit `DISPLAY`, `WAYLAND_DISPLAY`, `XAUTHORITY`, or the desktop display
authorization, so do not launch `--show-captures` through SSH and expect it to
appear on the Pi screen. Use a desktop autostart entry or launch it manually
after logging into that desktop session.

For the usual **service mode**, use the supplied systemd example. It runs the
LCD viewfinder when a panel is present; without one (or with a USB camera) it
logs one warning line and carries on headless, where the GPIO shutter button
(`--shutter-gpio 21`) still takes photographs. Install and enable the unit:

```sh
sudo install -m 644 deploy/pifilm-capture.service.example /etc/systemd/system/pifilm-capture.service
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-capture.service
sudo systemctl status pifilm-capture.service
```

The service has `StandardInput=null`; it needs no terminal controls. Check its
start-up lines with `journalctl -u pifilm-capture.service -n 100`.

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

## Stop, rollback, and change the hotspot passphrase

For desktop mode, stop the process in its desktop session with `Q` or Escape.
For service mode, use:

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

To change the hotspot passphrase (or its address), edit `WIFI_PASSWORD` (or
`PI_HOTSPOT_ADDRESS`) in `.env` and re-run the hotspot script with `--apply`
over Ethernet; then rejoin from the phone or laptop with the new passphrase.
