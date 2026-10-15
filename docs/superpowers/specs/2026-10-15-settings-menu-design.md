# Settings menu on the viewfinder: design

Amended 2026-10-15, during implementation, at the owner's request: the `ROTATE` row
is dropped ("leave the display flip option out, not important"). The menu remembers
three settings, not four: `save_dng`, `ae_constraint` and `ae_metering`. There is no
`set_rotate` on the displays or touch readers, `settings.json` has no `rotate` key,
and rotation stays `--display-rotate` in the unit. Read every mention of rotation
below as removed (the `ROTATE` row, "Applying each setting: Rotation", the
`set_rotate` code and tests, and acceptance step 2).

Date: 2026-10-15. Branch `feat/settings-menu`, pull request into `main`.
The design was approved in conversation on 2026-10-15: a `MENU` key on the live
view, Wi-Fi switching left out of this round, and the look's name shown.

Depends on PR #73 (Nextcloud sync over `eth0` or `wlan0`). Implementation starts
from a `main` that contains it.

## Goal

Today, changing the screen's rotation, DNG saving or the auto-exposure modes means
editing the systemd unit and restarting the service. This adds a menu on the LCD
for those four settings, with immediate effect and remembered across restarts. The
same menu holds:

- an "upload to Nextcloud now" button;
- the network addresses, which move off the live view;
- the name of the look in use.

Nothing here needs sudo.

## Opening and closing

- **The `MENU` key.** It sits on the live view, in the right column, between the
  shutter button and `+`. It is a pill in the keys' style (outline, `KEY_FILL`,
  label `MENU` in the tag font), centred at (287, 88) in 320 × 240 base units, about
  40 × 14.
  - Its hit box is the pill grown by 3 units: y 78 to 98.
  - That box stays clear of the shutter's hit circle (bottom at y 73) and of `+`'s
    hit circle (top at y 104), so no tap that works today changes meaning.
  - It is drawn and hit-tested only in LIVE, and only while no colour bars are
    shown (like the `2x` toggle).
- **The menu screen.** A tap on `MENU` enters a new viewfinder state, MENU: an
  opaque black page drawn in the live view's fonts and colours. While it is open
  the frame thread is not asked for frames, as for a review.
- **`DONE`.** A `DONE` key is drawn where the shutter button is, with the same
  centre and hit circle. It returns to LIVE.
- **Other ways to shoot.** The shared `CaptureController` is unchanged, so the
  GPIO button and the Stick still take photographs while the menu is open. A shot
  finished while the menu is open does not open REVIEW; the menu stays. The
  developing count and the warnings return with the live view.
- **Idle handling.** The idle dim and screen-off apply in MENU as in LIVE, and a
  tap in the menu counts as activity.

## Rows

One row per line, in 320 × 240 base units:

- Rows run from x 8 to 248, so they stay clear of `DONE`'s hit circle, which starts
  at x 253.
- Rows are 20 units tall, starting at y 8.
- A grey small-caps label sits on the left (the cells' label style) and the value
  on the right in white.
- A tap anywhere on an active row steps it to its next value, wrapping round.
- Read-only rows have a dimmer label and ignore taps.
- Rows the camera cannot use are left out, and the rows below move up.

| Row | Values | Shown when | Effect |
|---|---|---|---|
| `ROTATE` | `0°` / `180°` | always | at once: screen and touch together |
| `SAVE DNG` | `ON` / `OFF` | the backend supplies raw frames (Picamera2) | from the next photograph |
| `AE CONSTRAINT` | `NORMAL` / `HIGHLIGHT` / `SHADOWS` | the camera reports `AeConstraintMode` | at once |
| `AE METERING` | `CENTRE` / `SPOT` / `MATRIX` | the camera reports `AeMeteringMode` | at once |
| `UPLOAD` | see below | `--nextcloud-env` given | starts an upload |
| `ETH` | address, or `-` | always | read-only |
| `WLAN` | address, or `-` | always | read-only |
| `LOOK` | artifact folder name, or `starter (bundled)` | always | read-only |
| `RESET` | `TAP TWICE` | always | see below |

- **Look name.** `LOOK` shows the name of the `--artifacts` directory (for example
  `imx477-2026-09-v4-dark-hp5-tile`). If it is too wide, it is cut with a trailing
  `...` to fit the row.
- **Reset.** The first tap on `RESET` changes its value to `TAP AGAIN` for 3 s. A
  second tap within that time:
  - sets the four settings back to what the unit's flags say, applying each at
    once;
  - deletes the settings file;
  - logs `settings: reset to the unit's flags`.

## Applying each setting

- **Rotation.**
  - `set_rotate(rotate)` is added to each display (`kms.KmsDisplay`, `st7789`,
    `fake`) and each touch reader (`evtouch.EventTouch`, `cst3530`). Each one checks
    the value (0 or 180, `DisplayError` otherwise) as its constructor does.
  - KMS changes the pack rotation. ST7789 rewrites its `MADCTL` register. Touch
    changes its mapping.
  - The viewfinder calls both setters together, from the loop thread, between two
    frames. While the touch thread runs it is the only code that touches the
    touch device, so the touch setter only sets a field that the thread reads on
    its next poll.
- **DNG.** A `CaptureSession.save_dng` property with a setter. The capture side
  reads it once per shot.
- **AE modes.** `Picamera2Camera.set_ae_constraint(name)` and
  `set_ae_metering(name)`, under the configuration lock and written into both
  configurations, exactly as `set_iso` is. The names are validated against
  `_AE_CONSTRAINTS` and `_AE_METERING`. Whether the camera supports each mode is
  decided once at start-up, as for the strip's cells. `--ae-constraint` /
  `--ae-metering` stay as they are; they are now the unit's starting value.

## Remembered settings

- **The file.** `settings.json` sits beside the pending-grades directory, in the
  parent of `--state-dir` (default `~/.local/state/pifilm/settings.json`), where
  the `gpu-opening` marker also lives.
- **Contents.** A JSON object with `version: 1` and any of `rotate` (0 or 180),
  `save_dng` (bool), `ae_constraint` and `ae_metering` (the names above).
- **Writing.** It is written after every change in the menu, through a temporary
  file in the same directory, fsync and `os.replace`. A failure to write is one
  `settings: cannot save (...)` line; the change still applies for this run.
- **At start-up.** Each key present and valid overrides the unit's flag for that
  setting. When anything differs from the flags, one line names what came from the
  menu and what the unit says, for example
  `settings: rotate 180, ae-constraint shadows (saved in the menu; the unit says rotate 0, ae-constraint highlight)`.
- **Bad values.**
  - An unknown or invalid key is ignored with one line.
  - A file that is not valid JSON, or not version 1, is ignored with one line and
    left in place.
  - None of these is ever fatal.
- **Scope.** Only these four settings are stored. Every other flag still comes
  from the unit alone.

## Upload to Nextcloud from the menu

- **The sync moves into the package.**
  - The sync logic moves from `scripts/nextcloud_sync.py` into
    `pifilm/nextcloud.py`, unchanged in behaviour.
  - The `.env` parser it needs moves into `pifilm/envfile.py`.
    `firmware/sticks3/scripts/generate_config.py` keeps its own copy, because it
    runs without the package installed. Each copy's docstring names the other.
  - `scripts/nextcloud_sync.py` becomes a wrapper that calls
    `pifilm.nextcloud.main`, so the installed timer unit keeps working unchanged.
  - `tests/test_nextcloud_sync.py` moves to `tests/test_nextcloud.py`. A small
    test keeps the wrapper honest.
- **One sync at a time.** `pifilm.nextcloud` takes an exclusive, non-blocking
  `fcntl.flock` on `nextcloud-sync.lock` in the same state directory before the
  network checks.
  - If another run holds the lock, it logs
    `another Nextcloud sync is running; skipping` and exits 0.
  - The lock path can be passed in (`--lock PATH`) for tests. The default is
    `~/.local/state/pifilm/nextcloud-sync.lock`.
- **The flag.** `pifilm-capture --nextcloud-env PATH` enables the `UPLOAD` row.
  The capture service never reads that file itself.
- **Running it.** A tap on `UPLOAD` while no upload is running starts
  `[sys.executable, "-m", "pifilm.nextcloud", "--apply", "--env", PATH]` on a
  `pifilm-upload` thread.
  - The child's environment is the service's own, minus `PIFILM_REMOTE_TOKEN`.
  - Each line of the child's output goes to the service's log as
    `nextcloud: <line>`.
  - A tap while an upload is running does nothing.
- **What the row shows.**

  | State | Row value |
  |---|---|
  | No upload yet in this run | `TAP TO UPLOAD` |
  | Running | `UPLOADING 0:42` (elapsed time) |
  | Exit 0, last line starts with `rclone copied` | `DONE 14:05` |
  | Exit 0 otherwise | `SKIPPED: NO NETWORK`, `SKIPPED: NO NEXTCLOUD` or `SKIPPED: BUSY`, chosen from the last line's start; `SKIPPED` alone otherwise |
  | Any other exit | `FAILED (EXIT 4)` |

  A finished state stays until the next tap or restart.
- **Stopping the service.** If the service stops during an upload, the child gets
  SIGTERM, then SIGKILL after 5 s. Copied files are complete, because rclone uploads
  each file whole, so the next run carries on.

## The live view without the network tags

The `ETH`/`WLAN` tile leaves the live view: `_NETWORK_BOX`, `_network_tile` and
`LiveState.network` go, and the area returns to the picture. The `NetworkMonitor`,
its `network:` log lines and `ControllerSnapshot.network` stay; the menu reads the
addresses from there.

## Code

- `pifilm/display/menu.py` (new, pure, in 320 × 240 units at an integer scale like
  `liveview.py`):
  - `MenuRow`, the row list built from what the camera supports;
  - `render_menu(rows, state, *, scale)`;
  - `hit(x, y, rows, scale)` returning a row index, `DONE` or nothing;
  - cached tiles.
- `pifilm/display/liveview.py`: the `MENU` key (drawn and hit-tested, a new
  `Action.MENU`); the network tile removed.
- `pifilm/display/viewfinder.py`:
  - the MENU state and its taps;
  - the reset's double tap;
  - the upload thread's status, read each frame;
  - applying a setting through the session, camera, display and touch.
- `pifilm/capture/settings.py` (new): `Settings` (the four values), `load(path,
  defaults) -> (Settings, list[str] log lines)`, `save(path, settings)`.
- `pifilm/capture/upload.py` (new): `Upload`, which starts the child, keeps its
  state, logs its lines and stops it on close. No display code.
- `pifilm/capture/app.py`: `--nextcloud-env`; the settings loaded at start-up and
  applied before the camera opens; `save_dng` as a property; the upload passed to
  the viewfinder.
- `pifilm/capture/picamera.py`: `set_ae_constraint`, `set_ae_metering`.
- `pifilm/display/kms.py`, `st7789.py`, `fake.py`, `evtouch.py`, `cst3530.py`:
  `set_rotate`.
- `pifilm/nextcloud.py`, `pifilm/envfile.py` (moved), `scripts/nextcloud_sync.py`
  (wrapper).
- Docs:
  - `docs/lcd-viewfinder.md`: the menu, the layout table, the network tags'
    removal from the live view, and acceptance.
  - `docs/nextcloud-sync.md`: the menu button and the lock.
  - `docs/setup.md`: `--nextcloud-env` in the unit.
  - `deploy/pifilm-capture.service.example`: `--nextcloud-env`.
  - `CLAUDE.md`.
  - No README change without an approved draft.

## Tests

- **Settings:** load with no file, a valid file, partial keys, bad values, bad
  JSON and the wrong version; save is atomic (a temporary file and `os.replace`);
  the override log line.
- **Menu:**
  - rows hidden and shown by capability;
  - hit-testing of every row and `DONE`;
  - a long look name is cut to fit at scales 1 and 2;
  - the reset's double tap and its 3 s window, on an injected clock.
- **Live view:**
  - the `MENU` key's hit box does not overlap the shutter or `+`;
  - every existing hit point keeps its action;
  - with no network field, renders differ from before only in the `MENU` key's
    tile (pixels outside its box unchanged).
- **Viewfinder:**
  - LIVE → MENU → LIVE;
  - a GPIO or Stick shot while in MENU stays in MENU;
  - each row's tap calls the right setter and saves the file;
  - rotation calls display and touch together.
- **Upload:**
  - with a fake child: each row state, the token stripped from the environment,
    and lines logged;
  - stopping the service terminates, then kills, the child.
- **Nextcloud:** the moved tests pass unchanged; the lock skips a second run; the
  wrapper calls `main`.
- **Camera:** the AE setters write both configurations and reject unknown names.
- **Rotation:** `set_rotate` on each display and touch class, with bad values
  refused.

## Acceptance (hardware, on the Pi)

1. `MENU` opens the menu; `DONE` returns to the live view; the shutter, `+`, `-`
   and cells work as before.
2. `ROTATE` flips screen and touch together; a restart keeps it; the log has the
   `settings:` line.
3. `SAVE DNG` off: the next photograph has no `.dng`; on: it has one.
4. `AE CONSTRAINT` and `AE METERING`: the record's camera metadata shows the new
   mode after a change.
5. `UPLOAD` on home Wi-Fi: `UPLOADING`, then `DONE`; the files are on Nextcloud;
   `journalctl -u pifilm-capture | grep nextcloud:` shows rclone's lines. In
   hotspot mode: `SKIPPED: NO NEXTCLOUD`. Started while the timer's run is going:
   `SKIPPED: BUSY`.
6. `ETH`/`WLAN` show the addresses, and the live view no longer shows them.
   `LOOK` shows `imx477-2026-09-v4-dark-hp5-tile`.
7. `RESET`, tapped twice: the unit's flags are back and the settings file is gone.
8. A GPIO button press while the menu is open takes a photograph, and the menu
   stays open.

## Not in this round

- Wi-Fi switching. It would need a polkit rule installed once with sudo, because
  NetworkManager only lets a user with an active local session change networks.
- Joining new networks with an on-screen keyboard.
- Changing the look from the menu.
- Storing any setting beyond the four above.
- Showing the menu on the Stick.
