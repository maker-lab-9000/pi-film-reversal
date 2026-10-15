# Nextcloud photo sync

Push every capture under `~/Pictures/pifilm` to a folder on your Nextcloud over
WebDAV, from the Pi, whenever it is on the home network, wired or Wi-Fi. This is an optional
off-device archive; it changes nothing about capture or grading.

## What it does, and what it does not

`scripts/nextcloud_sync.py` runs `rclone copy` from the Pi's photo folder to one
Nextcloud folder. Design choices, each deliberate:

- **Copy, never mirror.** It uses `rclone copy`, which only adds or updates
  files on Nextcloud. Deleting a photo on the Pi (an SD-card cleanup) never
  deletes the cloud copy. There is no flag to make it delete.
- **A quiet no-op off the home network.** Each run first checks `eth0`, then
  `wlan0`, for a routable IPv4 (not a `169.254.x` link-local address) and
  takes the first that has one; then the Nextcloud host must answer a TCP
  connect. If either fails it logs one line and exits 0. A systemd timer can
  therefore fire it every few minutes and it does nothing until the Pi is on
  the home network, by cable or by Wi-Fi. When it goes ahead it logs which
  interface it chose, e.g. `Nextcloud reachable over wlan0 (192.168.178.56)`.
  Only the first interface with an address is checked; the connect follows
  the routing table either way. `--iface` overrides the list and can be
  repeated (`--iface wlan0 --iface eth0`); one given replaces the default.
- **Hotspot mode skips on its own.** When the Pi runs its own hotspot for the
  Stick, `wlan0` holds `10.42.0.1`, which passes the address check, but
  Nextcloud is not reachable from the hotspot, so the run logs the
  "not reachable" line and exits 0. No special case is needed.
- **It can upload while you shoot.** On Wi-Fi the Pi may be shooting while a
  sync runs. rclone's `--bwlimit` would be the knob if an upload ever slowed
  capture; it is not set today.
- **The password never touches a command line.** It is read from `.env`,
  obscured with `rclone obscure` (plaintext on stdin), and handed to rclone
  through `RCLONE_CONFIG_*` environment variables. It never appears in an argv,
  a process listing, or an rclone config file on disk.
- **Whole folder, incrementally.** Everything under `~/Pictures/pifilm` is
  synced — the `_original`/`_ungraded` JPEGs, the graded `_graded.jpg`, the `.dng`
  raws, and `captures.jsonl`. rclone skips files already uploaded, so repeated
  runs only transfer what is new. `--min-age 30s` avoids grabbing a file that a
  capture is still writing.

Why not plain `rsync`? A Nextcloud server speaks HTTP/WebDAV, not the rsync
protocol, unless you have shell access to its data directory (and writing there
directly bypasses Nextcloud's file index). rclone's WebDAV backend is the
rsync-equivalent that talks to Nextcloud correctly.

## Prerequisites

1. **rclone on the Pi:**

   ```bash
   sudo apt install rclone
   ```

   The script prints a clear error and does nothing if rclone is missing.

2. **A Nextcloud app password.** In Nextcloud, go to **Settings → Security →
   Devices & sessions → Create new app password**. Use that, not your login
   password: it is scoped to this device and can be revoked without changing
   your account password.

## Configure `.env`

Add these keys to the project `.env` on the Pi (they are documented in
`.env.example`). `.env` is gitignored; never commit real values.

```ini
NEXTCLOUD_URL=http://192.168.178.241:8083/
NEXTCLOUD_USER=george
NEXTCLOUD_PASSWORD=the-app-password-from-above
NEXTCLOUD_TARGET_DIR=MartinParr
NEXTCLOUD_SOURCE_DIR=/home/george/Pictures/pifilm
```

- `NEXTCLOUD_URL` is the base URL; the script appends
  `/remote.php/dav/files/<user>/` itself.
- `NEXTCLOUD_TARGET_DIR` is the folder created on Nextcloud; the `YYYY-MM-DD/`
  day folders are mirrored under it (e.g. `MartinParr/2026-09-14/181541_graded.jpg`).
- `NEXTCLOUD_SOURCE_DIR` defaults to `~/Pictures/pifilm` if left blank.

## Test with a dry run

Run it by hand first. Without `--apply` it does everything except transfer
files, so you can confirm the gate passes and the file list looks right:

```bash
cd ~/repos/pi-film-reversal
.venv/bin/python scripts/nextcloud_sync.py --env .env          # dry run
.venv/bin/python scripts/nextcloud_sync.py --env .env --apply  # real copy
```

If the Pi is not on the home network, both print a "skipping" line and exit —
plug the Ethernet cable into the home router or join the home Wi-Fi first (the
address must be on the same `192.168.178.x` network as Nextcloud, not the
link-local cable to the Mac or the Pi's own hotspot).

## Enable the timer

Install the example units (they drop the `.example` suffix), then enable the
timer. The service is `oneshot` and is triggered by the timer, not enabled on
its own.

```bash
sudo cp deploy/pifilm-nextcloud-sync.service.example /etc/systemd/system/pifilm-nextcloud-sync.service
sudo cp deploy/pifilm-nextcloud-sync.timer.example   /etc/systemd/system/pifilm-nextcloud-sync.timer
sudo systemctl daemon-reload
sudo systemctl enable --now pifilm-nextcloud-sync.timer
```

Verify and watch it work:

```bash
systemctl list-timers pifilm-nextcloud-sync.timer     # next/last run
systemctl start pifilm-nextcloud-sync.service         # force one run now
journalctl -u pifilm-nextcloud-sync.service -n 30     # the run's summary line
```

The service runs as `george` with the repo as its working directory, so it
reads the same `.env`. It fires a few minutes after boot and every 15 minutes
after that; each run is a no-op unless the Pi is on the home network, so the
frequent schedule is cheap.

## Troubleshooting

- **Every run says "skipping".** `no network on eth0 or wlan0` means neither
  interface has a routable IPv4 (`ip -4 addr show dev eth0`,
  `ip -4 addr show dev wlan0`). `not reachable over ...` names the interface
  and address it chose: Nextcloud cannot be reached from there. Confirm the
  Ethernet cable is in the home router or the Pi is on the home Wi-Fi (not its
  hotspot, `10.42.0.1`) and `ping 192.168.178.241` works.
- **401 / authentication errors.** Regenerate the app password and update
  `NEXTCLOUD_PASSWORD`; a login password with 2FA enabled will not work over
  WebDAV.
- **Files not appearing.** Check `NEXTCLOUD_TARGET_DIR` exists or can be created
  by the account, and run with `--apply` (a dry run transfers nothing).
