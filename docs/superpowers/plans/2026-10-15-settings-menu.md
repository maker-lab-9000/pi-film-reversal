# Settings Menu Implementation Plan

> **Amended 2026-10-15 (owner): rotation is out.** See the spec's amendment note.
> - Task 2 is amended by Task 2b below: `rotate`, `ROTATIONS` and every rotation
>   test leave `pifilm/capture/settings.py`; `KEYS` is
>   `("save_dng", "ae_constraint", "ae_metering")` and `Settings` has those three
>   fields.
> - Task 5 is cancelled (no `set_rotate` anywhere).
> - Task 7: no `rotate` row key. Task 8: no `rotate` row, no rotation applier, no
>   forced redraw after rotation. Task 9: no `rotate` default, no
>   `args.display_rotate` override, no rotate applier in `_run_viewfinder`; the
>   tests about rotation are dropped. The summary log line's example becomes
>   `settings: dng off, ae-constraint shadows (saved in the menu; the unit says dng on, ae-constraint highlight)`.
>
> ### Task 2b: rotation leaves the settings
> **Files:** `pifilm/capture/settings.py`, `tests/test_settings.py`.
> Remove `rotate` from `Settings`, `KEYS`, validation, the summary line and the
> docstring; remove `ROTATIONS`; update or remove every test that uses `rotate`,
> keeping equivalent coverage with the remaining keys (for example, a wrong-type
> value is now `save_dng: 1` or `ae_constraint: 3`). A `rotate` key in an existing
> file is an unknown key: one `settings: ignoring rotate in <path>` line. Commit
> `capture: the settings without rotation`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `MENU` key on the LCD live view opens a settings page. On that page you
can:

- change rotation, DNG saving and the two auto-exposure modes, with immediate
  effect and the choice remembered across restarts;
- start a Nextcloud upload;
- read the network addresses and the look in use.

**Architecture:**

- **Pure, tested units.** `capture/settings.py` (the remembered values and their
  appliers), `capture/upload.py` (the upload child process) and
  `display/menu.py` (drawing and hit-testing).
- **Glue.** The viewfinder gains a MENU state that uses those units.
- **Wiring.** `app.py` loads the settings before the panel and camera open,
  registers an applier per setting, and passes the control, the upload and the
  look's name to the viewfinder.
- **Nextcloud.** The sync moves into the package, so the service can run it with
  `python -m`.

**Tech Stack:** Python 3.11+, Pillow, NumPy, Picamera2/libcamera (lazy imports),
pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-15-settings-menu-design.md`. Read it
first; it is the authority on behaviour.

## Global Constraints

- **Always use the venv.** Run tools as `.venv/bin/pytest`, `.venv/bin/ruff`,
  `.venv/bin/python`. The full suite is `.venv/bin/pytest -q -m 'not slow'`; lint
  is `.venv/bin/ruff check .` (line length 100). Both must pass at the end of
  every task.
- **The package must import on a Mac.** Hardware imports (`libcamera`, `pykms`,
  `spidev`, `gpiozero`, `smbus2`) stay lazy, inside functions.
- **Secrets.** `.env` holds secrets. Parse it as `KEY=VALUE` data only: never
  source it and never print its values. The token, Wi-Fi password and Nextcloud
  password must never appear on a command line, in a source file, in a unit or in
  a commit.
- **The upload child's environment** is `os.environ` minus `PIFILM_REMOTE_TOKEN`.
- **Commits.** One or more per task, in the repo's style (`capture: ...`,
  `viewfinder: ...`, `sync: ...`, `docs: ...`). Every message ends with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Staging.** Stage by explicit path only. Never touch the untracked directory
  named `" "` (a single space) at the repo root.
- **No push.** No subagents dispatched by implementers.
- **Read docstrings first.** Module docstrings carry design rationale. Read them
  before changing a module, and add rationale to new modules in the same style.
- **Layout units.** All layout is in 320 × 240 base units, drawn at an integer
  `scale` (2 on the 640 × 480 DSI panel).
- **Labels and values on screen** are exactly as written in the spec's tables.

## Review Focus

1. **A finished shot while the menu is open must not leave the menu.** Today
   `step()` switches to REVIEW, or to the processing colour bars, whenever a job is
   active or has finished. Expected: the menu stays, and `_seen_finished` still
   advances, so no stale review appears later. This is tested in Task 8.
2. **A saved AE mode on a V4L2 camera, or under `--fake`, must not stop the
   service from starting.** `_reject_picamera2_only_flags` refuses
   `args.ae_constraint is not None`. Expected: saved AE values are applied to the
   start-up arguments only when the backend is Picamera2. This is tested in Task 9.
3. **A damaged or foreign `settings.json` must never be fatal.** This covers bad
   JSON, the wrong version, a value of the wrong type, `rotate: 90`, and a file
   that is a directory. Expected: one log line, then the unit's flags. This is
   tested in Task 2.
4. **An applier that raises.** For example, the camera rejects an AE mode.
   Expected: one `settings: cannot apply ...` line, the value is not saved, and the
   menu shows the unchanged value. This is tested in Task 2 and Task 8.
5. **The service stops during an upload.** Expected: the child gets SIGTERM, then
   SIGKILL after 5 s, the thread is joined, and nothing hangs. This is tested in
   Task 3.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `pifilm/envfile.py` | new | `parse_env_file`, a copy of the firmware script's parser. |
| `pifilm/nextcloud.py` | new, moved | The sync logic from `scripts/nextcloud_sync.py`, plus the lock. |
| `scripts/nextcloud_sync.py` | rewritten | A thin wrapper that calls `pifilm.nextcloud.main`. |
| `tests/test_nextcloud.py` | moved | From `tests/test_nextcloud_sync.py`, plus lock and wrapper tests. |
| `pifilm/capture/settings.py` | new | `Settings`, `load`, `save`, `SettingsControl`. |
| `pifilm/capture/upload.py` | new | `Upload`: the child process, its state and its row text. |
| `pifilm/capture/picamera.py` | modified | `set_ae_constraint`, `set_ae_metering`, `set_save_dng`, AE support flags. |
| `pifilm/capture/app.py` | modified | `CaptureSession.save_dng` property, `--nextcloud-env`, settings at start-up, appliers, upload, look name. |
| `pifilm/display/{kms,st7789,fake,evtouch,cst3530}.py` | modified | `set_rotate`. |
| `pifilm/display/liveview.py` | modified | The `MENU` key and `Action.MENU`; the network tile removed. |
| `pifilm/display/menu.py` | new | `MenuRow`, `render_menu`, `menu_hit`. |
| `pifilm/display/viewfinder.py` | modified | The MENU state. |
| Docs | modified | `docs/lcd-viewfinder.md`, `docs/nextcloud-sync.md`, `docs/setup.md`, `deploy/pifilm-capture.service.example`, `CLAUDE.md`. |

---

### Task 1: The Nextcloud sync moves into the package, with a lock

**Files:**
- Create: `pifilm/envfile.py`, `pifilm/nextcloud.py`
- Rewrite: `scripts/nextcloud_sync.py`
- Move: `tests/test_nextcloud_sync.py` → `tests/test_nextcloud.py` (`git mv`)

**Interfaces:**

Produces:

- `pifilm.nextcloud.main(argv) -> int`, the same command line as today plus
  `--lock PATH`.
- `DEFAULT_LOCK = <state>/pifilm/nextcloud-sync.lock`. `<state>` is
  `$XDG_STATE_HOME`, else `~/.local/state`, resolved as `default_state_dir` in
  `pifilm/capture/gradequeue.py` does.
- Skip lines that Task 3 matches by their start:
  - `no network on ...; skipping Nextcloud sync`
  - `Nextcloud <host>:<port> not reachable over ...`
  - `another Nextcloud sync is running; skipping` (new)
- Success line: `rclone copied ...`

Steps:

- [ ] **Step 1: Move the code and the tests.**
  - `git mv tests/test_nextcloud_sync.py tests/test_nextcloud.py`.
  - Create `pifilm/nextcloud.py` with the whole body of `scripts/nextcloud_sync.py`:
    the docstring, constants, functions and `main`.
  - Drop the `sys.path.insert` line and the `firmware...` import, and use
    `from .envfile import parse_env_file` instead.
  - Create `pifilm/envfile.py` with `_ENV_KEY` and `parse_env_file`, copied
    verbatim from `firmware/sticks3/scripts/generate_config.py`. Its docstring
    says it is a copy kept in step with that file, which runs without the package
    installed. Add one line to the firmware function's docstring naming
    `pifilm/envfile.py`.
  - In the test file, change `from scripts.nextcloud_sync import (` to
    `from pifilm.nextcloud import (`. Fix any `monkeypatch` targets that name the
    old module.
  - Run `.venv/bin/pytest -q tests/test_nextcloud.py`. Expected: everything passes
    unchanged.

- [ ] **Step 2: Rewrite the wrapper.** `scripts/nextcloud_sync.py` becomes:

```python
"""Push captured photos to Nextcloud: a wrapper kept for the installed timer unit.

The sync lives in ``pifilm.nextcloud`` so the capture service can run it with
``python -m pifilm.nextcloud`` from its menu; the unit
``deploy/pifilm-nextcloud-sync.service.example`` still runs this path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifilm.nextcloud import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

  Add a test that `scripts.nextcloud_sync.main is pifilm.nextcloud.main`.

- [ ] **Step 3: Write the failing lock tests.**
  - `main([... "--lock", str(tmp/"l")])` while another open file descriptor holds
    `fcntl.flock(LOCK_EX)` on the same path:
    - it prints `another Nextcloud sync is running; skipping`;
    - it returns 0;
    - `run_sync` is never called (monkeypatch `pifilm.nextcloud.run_sync` to
      record).
  - With the lock free, `run_sync` is called, and afterwards the lock is free
    again (a second `flock(LOCK_EX | LOCK_NB)` succeeds).
  - The lock's parent directories are created when missing.
  - Use a temp env file, as the existing `main` tests do.

- [ ] **Step 4: Implement the lock in `main`**, after the config is read and before
  `run_sync`:

```python
def _default_lock() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "pifilm" / "nextcloud-sync.lock"


@contextlib.contextmanager
def _sync_lock(path: Path):
    """Yield True while this process holds the lock, False if another run does."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
```

  - The argument is `--lock`, `type=Path`, default `None`, which means
    `_default_lock()`.
  - In `main`:

```python
with _sync_lock(args.lock or _default_lock()) as held:
    if not held:
        print("another Nextcloud sync is running; skipping")
        return 0
    return run_sync(...)
```

  - Add a paragraph to the module docstring on why the lock exists: the menu's
    upload and the timer's run must not run rclone twice over the same files.

- [ ] **Step 5:** Run `.venv/bin/pytest -q tests/test_nextcloud.py` and
  `.venv/bin/ruff check .`. Then run
  `.venv/bin/python scripts/nextcloud_sync.py --help`, which prints usage.

- [ ] **Step 6: Commit** `sync: the Nextcloud sync in the package, one run at a time`.

---

### Task 2: Remembered settings

**Files:**
- Create: `pifilm/capture/settings.py`, `tests/test_settings.py`

**Interfaces:**

Produces:

```python
SETTINGS_FILE = "settings.json"
VERSION = 1
ROTATIONS = (0, 180)
AE_CONSTRAINTS = ("normal", "highlight", "shadows")
AE_METERINGS = ("centre", "spot", "matrix")
KEYS = ("rotate", "save_dng", "ae_constraint", "ae_metering")


@dataclass(frozen=True)
class Settings:
    rotate: int
    save_dng: bool
    ae_constraint: str
    ae_metering: str


def load(path: Path, defaults: Settings) -> tuple[Settings, dict[str, object], list[str]]:
    """(effective settings, the valid overrides read, log lines). Never raises."""


def save(path: Path, overrides: Mapping[str, object]) -> None:
    """Write {"version": 1, **overrides} atomically. Raises OSError."""


class SettingsControl:
    def __init__(self, path: Path, defaults: Settings, overrides: Mapping[str, object],
                 *, log: Callable[[str], None] = print) -> None: ...
    @property
    def current(self) -> Settings: ...
    def register(self, key: str, applier: Callable[[object], None]) -> None: ...
    def set(self, key: str, value: object) -> bool:  # True when applied
    def reset(self) -> None: ...
```

Behaviour:

- **Validity rules.**
  - `rotate` must be an `int` (not a `bool`) in `ROTATIONS`.
  - `save_dng` must be a `bool`.
  - `ae_constraint` must be in `AE_CONSTRAINTS`, and `ae_metering` in
    `AE_METERINGS`.
- **`load` log lines.**
  - No file: no lines.
  - Unreadable, not JSON, not an object, or `version != 1`: one line,
    `settings: ignoring <path> (<reason>)`. The file is left in place.
  - An invalid or unknown key: `settings: ignoring <key> in <path>`.
  - When any override differs from its default, one summary line:
    `settings: rotate 180, ae-constraint shadows (saved in the menu; the unit says rotate 0, ae-constraint highlight)`.
    - Keys are printed with `-` for `_`.
    - Booleans print as `on`/`off`, with the key `dng`: `dng off`.
    - Only differing keys are listed, in `KEYS` order.
- **`save`.** Writes the file in the same directory through a temporary file:
  write, flush, `os.fsync`, then `os.replace`. It creates the parent directory if
  needed.
- **`SettingsControl.set(key, value)`.**
  - Calls the registered applier first, if there is one. If it raises, log
    `settings: cannot apply <key> <value> (<exc>)` and return False; nothing
    changes.
  - Otherwise, update `current`. The override is `value`, or is removed when
    `value` equals the default.
  - Then `save`. On `OSError`, log `settings: cannot save (<exc>)` and keep the
    change. Return True.
  - An invalid key or value raises `ValueError`; that is a programming error.
- **`reset()`.**
  - For each key whose current value differs from the default, call its applier
    with the default. One that raises is logged as above and the key keeps its
    value.
  - Clear the overrides of the keys that were reset.
  - Delete the file; a missing file is fine, and another `OSError` is logged.
  - Log `settings: reset to the unit's flags`.

- [ ] **Step 1: Write the failing tests** in `tests/test_settings.py`, using
  `tmp_path` and a list as `log`:
  - **Loading:**
    - No file: the defaults, `{}`, `[]`.
    - A valid file with all four keys: the effective values and the summary line
      (check the exact text above with `rotate 180, ae-constraint shadows`).
    - Partial keys.
    - Each invalid value, including `rotate: 90`, `rotate: true`,
      `save_dng: "yes"` and `ae_metering: "average"`: ignored with its line, and
      the other keys still applied.
    - An unknown key.
    - Bad JSON, a JSON list, `version: 2`, and a directory at the path: each gives
      one ignore line and the defaults.
    - An override equal to its default gives no summary line.
  - **Saving:** atomic (monkeypatch `os.replace` to raise: the old file is
    unchanged and no temporary file is left behind); the parent directory is
    created; the content round-trips through `load`.
  - **The control:**
    - `set` calls the applier, then saves.
    - An applier that raises: False, the line is logged, and `current` and the
      file are unchanged.
    - A save failure: True, the line is logged, and `current` is changed.
    - Setting a value back to its default removes the key from the file.
    - `reset` calls appliers only for keys that differ, deletes the file and logs
      the line.
    - `reset` with no file present does not raise.
    - `set("rotate", 90)` raises `ValueError`.
  - **Run** `.venv/bin/pytest -q tests/test_settings.py`. Expected: they fail with
    an `ImportError`.
- [ ] **Step 2: Implement `pifilm/capture/settings.py`** to the interface above. Its
  module docstring covers:
  - why only these four settings;
  - why saved values win over the unit's flags, and why there is a log line and a
    reset;
  - why a bad file is never fatal;
  - why the write is atomic.
- [ ] **Step 3:** Run the tests and ruff. Expected: they pass.
- [ ] **Step 4: Commit** `capture: settings the menu changes, remembered in settings.json`.

---

### Task 3: The upload child

**Files:**
- Create: `pifilm/capture/upload.py`, `tests/test_upload.py`

**Interfaces:**

Consumes the start of each line that Task 1 prints.

Produces:

```python
class Upload:
    def __init__(self, env_path: Path, *, python: str = sys.executable,
                 popen: Callable[..., Any] = subprocess.Popen, clock: Any = time,
                 log: Callable[[str], None] = print,
                 environ: Mapping[str, str] = os.environ) -> None: ...
    def start(self) -> bool: ...           # False while one is running
    @property
    def running(self) -> bool: ...
    def status(self) -> str: ...           # the UPLOAD row's value, upper case
    def close(self, timeout: float = 5.0) -> None: ...
```

Behaviour:

- **`start`.**
  - Launches `[python, "-m", "pifilm.nextcloud", "--apply", "--env", str(env_path)]`
    with `stdout=PIPE`, `stderr=STDOUT`, `text=True`, `bufsize=1`, and
    `env = {k: v for k, v in environ.items() if k != "PIFILM_REMOTE_TOKEN"}`.
  - A `pifilm-upload` daemon thread reads the lines. Each line, stripped, is
    logged as `nextcloud: <line>` and kept as the last line.
  - The thread then calls `wait()` and records the exit code and the finish wall
    time (`clock.time()`).
  - A `popen` that raises `OSError` is logged and recorded as exit 127.
- **`status()`.**

  | State | Text |
  |---|---|
  | Never started | `TAP TO UPLOAD` |
  | Running | `UPLOADING m:ss`, from `clock.monotonic()` since the start |
  | Exit 0, last line starts with `rclone copied` | `DONE HH:MM`, from `time.localtime(finish)` |
  | Exit 0, last line starts with `no network on` | `SKIPPED: NO NETWORK` |
  | Exit 0, last line starts with `Nextcloud ` and contains `not reachable` | `SKIPPED: NO NEXTCLOUD` |
  | Exit 0, last line starts with `another Nextcloud sync is running` | `SKIPPED: BUSY` |
  | Exit 0, anything else | `SKIPPED` |
  | Any other exit | `FAILED (EXIT n)` |

- **`close(timeout)`.** While a child is running: `terminate()`, then
  `wait(timeout)`; on `subprocess.TimeoutExpired`, `kill()` and `wait()`. Then join
  the thread for up to 2 s. It is safe to call twice and before any start.
- **Locking.** All state is behind one lock.

- [ ] **Step 1: Write the failing tests** with a fake `popen`. The fake returns an
  object with:
  - `stdout`, an `io.StringIO` of scripted lines;
  - `wait(timeout=None)`, which returns a scripted code, or raises
    `TimeoutExpired` once for the kill test;
  - `terminate()` and `kill()`, which record that they were called;
  - `poll()`.

  Use a fake clock with `time()` and `monotonic()`. Test:
  - the exact argv;
  - that the environment lacks `PIFILM_REMOTE_TOKEN` and keeps the other keys;
  - each line logged with the `nextcloud: ` prefix;
  - every row of the status table;
  - `start` returns False while running (hold the fake's `wait` on a
    `threading.Event`);
  - `close` terminates, then kills on timeout, and joins;
  - `close` with nothing running does nothing;
  - a `popen` `OSError` gives `FAILED (EXIT 127)`.

  Run them. Expected: they fail with an `ImportError`.
- [ ] **Step 2: Implement `pifilm/capture/upload.py`.** Its docstring covers:
  - why a child process (rclone and the `.env` stay out of the capture process,
    and a hang cannot block capture);
  - why the token is removed from the child's environment;
  - why the status is read from the last line.
- [ ] **Step 3:** Run the tests and ruff. Expected: they pass.
- [ ] **Step 4: Commit** `capture: an upload to Nextcloud run as a child process`.

---

### Task 4: Live setters on the camera and the session

**Files:**
- Modify: `pifilm/capture/picamera.py`, `pifilm/capture/app.py` (`CaptureSession`
  only)
- Test: `tests/test_picamera.py`, or the existing Picamera2 test file (find it with
  `grep -l "set_iso" tests/`); `tests/test_app.py`

**Interfaces:**

Produces:

- On `Picamera2Camera`:
  - the attributes `ae_constraint: str`, `ae_metering: str`,
    `ae_constraint_supported: bool` and `ae_metering_supported: bool`;
  - `set_ae_constraint(name: str) -> None`;
  - `set_ae_metering(name: str) -> None`;
  - `set_save_dng(on: bool) -> None`.
- On `CaptureSession`: the property `save_dng: bool` with a setter.

Behaviour:

- **Support flags.** At open, `ae_constraint_supported` is True when
  `"AeConstraintMode" in camera.camera_controls` and the libcamera enum exists;
  likewise for metering. Factor the two enum lookups out of `_camera_controls` into
  `_ae_constraint_value(name)` and `_ae_metering_value(name)`. Each raises
  `AttributeError` when the enum is missing, and `_camera_controls` keeps its
  current behaviour, including the warning.
- **`set_ae_constraint` / `set_ae_metering`** follow `set_iso` exactly:
  - validate against `_AE_CONSTRAINTS` / `_AE_METERING`, else `CameraError`;
  - a closed camera, or an unsupported mode, is a `CameraError`;
  - under `self._config_lock`: build the control inside a `try`, call
    `camera.set_controls`, and write it into both configurations with
    `_set_config_control`; any exception becomes a `CameraError` and changes
    nothing;
  - on success, update the attribute.
- **`set_save_dng(on)`** sets `self._save_dng = bool(on)`. It is read once per
  still at line ~466, which already happens under the request lock.
- **`CaptureSession.save_dng`.**
  - The getter returns `self._save_dng`.
  - The setter sets it, then calls `self.camera.set_save_dng(on)` when the camera
    has that method.
  - Find the session's camera attribute name in `__init__`.

- [ ] **Step 1: Write the failing tests.** Reuse the existing fake Picamera2 from
  the `set_iso` tests:
  - each setter sends the control, writes both configurations and updates the
    attribute;
  - an unknown name raises `CameraError`;
  - a camera that raises on `set_controls` leaves the configurations and the
    attribute unchanged;
  - an unsupported mode raises `CameraError`;
  - the support flags at open, with and without the control in `camera_controls`;
  - `set_save_dng(False)` means the next still writes no DNG (use whatever the
    existing DNG test asserts);
  - on the session, setting `save_dng` reaches the camera, and a camera without
    `set_save_dng` is tolerated.

  Run them. Expected: they fail.
- [ ] **Step 2: Implement.**
- [ ] **Step 3:** Run the full suite and ruff. Expected: they pass.
- [ ] **Step 4: Commit** `capture: auto-exposure modes and DNG saving changed while the camera runs`.

---

### Task 5: Rotation changed at run time

**Files:**
- Modify: `pifilm/display/kms.py`, `st7789.py`, `fake.py`, `evtouch.py`,
  `cst3530.py`
- Test: their existing test files (`tests/test_display_*.py`; find them with
  `grep -l "rotate" tests/`)

**Interfaces:**

Produces `set_rotate(rotate: int) -> None` on `KmsDisplay`, `ST7789Display`,
`FileDisplay`, `NoTouch`, `EventTouch` and `CST3530Touch`. Each one refuses
anything other than 0 or 180 with `DisplayError`, using the constructor's message.

Behaviour:

- **KMS:** sets `self._rotate`. The next `show` packs with it.
- **ST7789:** sends command `0x36` with data `bytes([MADCTL[rotate]])`, as `__init__`
  does, then sets `self._rotate`. Store `self._rotate` in `__init__` if it is not
  stored yet.
- **evtouch and cst3530:** set `self._rotate`, one attribute assignment. The touch
  thread reads it on its next poll.
- **Fake:** `FileDisplay` stores `rotate` and rotates the saved PNG by 180° when it
  is set. `NoTouch.set_rotate` checks the value and does nothing else.

- [ ] **Step 1: Write the failing tests.** For each class:
  - a valid change takes effect (KMS packs rotated; ST7789's fake SPI sees `0x36`
    and the MADCTL byte; touch maps a point to the opposite corner after
    `set_rotate(180)`);
  - `set_rotate(90)` raises `DisplayError`.

  Run them. Expected: they fail.
- [ ] **Step 2: Implement.**
- [ ] **Step 3:** Run the full suite and ruff.
- [ ] **Step 4: Commit** `display: rotation changed while the panel runs`.

---

### Task 6: The `MENU` key on the live view; the network tile leaves it

**Files:**
- Modify: `pifilm/display/liveview.py`, `pifilm/display/viewfinder.py` (only the
  line that passes `network=` into `LiveState`)
- Test: `tests/test_display_liveview.py`, `tests/test_viewfinder.py`

**Interfaces:**

Produces:

- `Action.MENU`;
- constants `MENU_CENTRE = (287, 88)`, `MENU_SIZE = (40, 14)`,
  `MENU_HIT_MARGIN = 3`, `MENU_LABEL = "MENU"`;
- `PRESS_KEYS` gains `Action.MENU`, so the key lights while pressed, like the
  others.

Removes `LiveState.network`, `_NETWORK_BOX` and `_network_tile`, and their
comment block.

Behaviour:

- **Drawing.** The key is drawn inside `_controls_tile`, whose box
  `(256, 8, 318, 188)` contains it. It is a pill the size of `MENU_SIZE`, centred
  at `MENU_CENTRE`, in the `+`/`-` keys' style:
  - outline `KEY_OUTLINE` in the keys' outline colour;
  - fill `KEY_FILL`, or the lit fill when pressed;
  - the label in the tag font (`SEMIBOLD`, `TAG_SIZE`), white.
  The comment states the gap: the shutter's hit circle ends at y 73, `+`'s starts
  at y 104, and the hit box is y 78 to 98.
- **Hit-testing.** In `hit()`, after the shutter, `+` and `-` checks: inside the
  pill grown by `MENU_HIT_MARGIN` on every side → `Action.MENU`.

- [ ] **Step 1: Write the failing tests.**
  - `hit` at the key's centre and at its four grown corners is `MENU`, at scales 1
    and 2.
  - Every base-unit point inside the shutter's and `+`'s hit circles still returns
    `SHUTTER` / `PLUS` (loop over the circles).
  - No point returns `MENU` outside the grown box.
  - A render with and without the key differs only inside the key's box. Compare
    with a copy of the old tile: monkeypatch the drawing off, or compare against
    the pixels outside the box, which must be identical to a render with the
    pill's drawing disabled.
  - Pressed `MENU` draws lit.
  - The network tests are deleted, and `LiveState` no longer has `network`.

  Run them. Expected: they fail.
- [ ] **Step 2: Implement.** Remove the network tile and its tests in
  `test_display_liveview.py`, and the `network=` passthrough and its test in
  `viewfinder.py` / `test_viewfinder.py`. `ControllerSnapshot.network` stays.
- [ ] **Step 3:** Run the full suite and ruff.
- [ ] **Step 4: Commit** `viewfinder: a MENU key on the live view; the addresses leave it`.

---

### Task 7: The menu page

**Files:**
- Create: `pifilm/display/menu.py`, `tests/test_display_menu.py`

**Interfaces:**

Produces:

```python
@dataclass(frozen=True)
class MenuRow:
    key: str      # "rotate", "save_dng", "ae_constraint", "ae_metering", "upload",
                  # "eth", "wlan", "look", "reset"
    label: str    # "ROTATE", "SAVE DNG", ...
    value: str    # already formatted, e.g. "180°", "ON", "UPLOADING 0:42"
    active: bool  # False: dim label, taps ignored

DONE = "done"
ROW_X0, ROW_X1, ROW_Y0, ROW_HEIGHT = 8, 248, 8, 20
VALUE_RIGHT = ROW_X1 - 4

def render_menu(rows: Sequence[MenuRow], *, scale: int = 1, done_pressed: bool = False) -> Image.Image
def menu_hit(x: int, y: int, row_count: int, scale: int = 1) -> int | str | None
def fit_text(text: str, font, width: float) -> str   # cut with "..." to fit
```

Behaviour:

- **The page.** Black, `(320 * scale, 240 * scale)`. Each row is drawn from
  `ROW_Y0 + i * ROW_HEIGHT`:
  - the label is left-aligned at `ROW_X0` in the cells' label style (`MEDIUM`,
    `LABEL_SIZE`, `LABEL_TRACKING`), grey, or `UNLIT` when the row is inactive;
  - the value is right-aligned at `VALUE_RIGHT` in `SEMIBOLD`, size 9, white, cut
    with `fit_text` to the room left after the label plus 8 units;
  - a hairline separator in `UNLIT` runs under each row.
- **`DONE`.** A disc drawn at `SHUTTER_CENTRE` with `SHUTTER_RADIUS`, in the
  shutter button's ring style, with the label `DONE` in `SEMIBOLD` `TAG_SIZE`
  centred on it. It is lit while pressed.
- **Font and supersampling** are reused from `liveview` (import its `_Canvas`,
  fonts and colour constants rather than redefining them), with the same
  supersampling.
- **`menu_hit`.**
  - Convert to base units.
  - Inside `SHUTTER_HIT_RADIUS` of `SHUTTER_CENTRE` → `DONE`.
  - Else, if `ROW_X0 <= x < ROW_X1` and
    `ROW_Y0 <= y < ROW_Y0 + row_count * ROW_HEIGHT`, the row index.
  - Else `None`.
- **Caching.** Cache the render with `functools.lru_cache` keyed on
  `(tuple(rows), scale, done_pressed)`; `MenuRow` is frozen and hashable.

- [ ] **Step 1: Write the failing tests.**
  - `menu_hit`:
    - every row's centre at scales 1 and 2;
    - `DONE` at the shutter's centre and at its hit edge;
    - a point between rows' right edge (x 248) and `DONE` (x 253) → `None`;
    - below the last row → `None`.
  - `fit_text` keeps a short text and cuts
    `imx477-2026-09-v4-dark-hp5-tile-with-a-much-longer-name` with `...`, so it
    fits the given width.
  - A render with nine rows at scales 1 and 2 has the right size, and no row's
    ink falls past x 248 (crop and check that the area from x 249 to 252 is
    black).
  - Inactive labels are drawn darker than active ones (mean pixel value in the
    label area).

  Run them. Expected: they fail.
- [ ] **Step 2: Implement.** Write the menu's rationale in the module docstring:
  - a black page (no live picture behind it, so the frame thread rests);
  - why `DONE` sits where the shutter is;
  - the row geometry against `DONE`'s hit circle.
- [ ] **Step 3:** Run the tests and ruff, then save a render at scale 2 with
  realistic rows to `.superpowers/sdd/<this plan's workspace>/menu.png` for the
  controller to look at.
- [ ] **Step 4: Commit** `viewfinder: the settings menu page`.

---

### Task 8: The viewfinder's MENU state

**Files:**
- Modify: `pifilm/display/viewfinder.py`
- Test: `tests/test_viewfinder.py`

**Interfaces:**

Consumes:

- `SettingsControl` (Task 2);
- `Upload` (Task 3), duck-typed: `start()`, `status()`;
- `MenuRow`, `render_menu`, `menu_hit` and `DONE` (Task 7);
- `Action.MENU` (Task 6);
- `ControllerSnapshot.network`.

Produces new `ViewfinderLoop.__init__` keyword arguments:

- `settings: SettingsControl | None = None`
- `upload: Any | None = None`
- `look: str = ""`
- `menu_rows: tuple[str, ...] | None = None`: which optional setting rows the
  camera supports, any of `"save_dng"`, `"ae_constraint"`, `"ae_metering"`. `None`
  means none of them.

Behaviour:

- **Entering the menu.** In LIVE, a tap whose `hit` is `Action.MENU` (and no
  active job, as for the other keys) sets `self.state = "MENU"`. It also clears any
  `TAP AGAIN` arming and shows the menu.
- **Building the rows** each MENU step, in the spec's order:
  - `rotate`: `f"{v}°"`;
  - `save_dng`: `ON`/`OFF`;
  - `ae_constraint` and `ae_metering`: the value upper-cased;
  - `upload`: `upload.status()`;
  - `eth` and `wlan`: from `snap.network`, a dict of label to address, else `-`;
    inactive;
  - `look`: `look or "starter (bundled)"`; inactive;
  - `reset`: `TAP AGAIN` while armed, else `TAP TWICE`.

  The setting rows appear only when `settings` is not `None`, and the three
  optional ones only when they are named in `menu_rows`. The `upload` row appears
  only when `upload` is not `None`.
- **Showing it.** `self._show(render_menu(rows, scale=..., done_pressed=...))`, but
  only when the rows or the pressed state changed since the last show. This keeps
  the SPI bus quiet, and the upload's elapsed time changes once a second.
- **Taps in MENU** go through `menu_hit`:
  - `DONE`: back to LIVE, `self._restart_rate_window()`, and clear the arming.
  - The `rotate`, `save_dng`, `ae_constraint` or `ae_metering` row:
    `settings.set(key, next_value)`, cycling through `ROTATIONS`, `(True, False)`,
    `AE_CONSTRAINTS` and `AE_METERINGS` from the current value.
  - `upload`: `upload.start()`.
  - `reset`: the first tap arms it until `now + 3.0`; a tap while armed calls
    `settings.reset()` and disarms.
  - Inactive rows and `None`: nothing.
- **Finished jobs while in MENU.** The `finished_count` branch at the top of
  `step()` advances `self._seen_finished` and returns to the menu without entering
  REVIEW or NOTICE. The active-job colour bars are not shown in MENU either.
  Restructure so that `if self.state == "MENU": ...; return` comes right after the
  finished-count bookkeeping and before the NOTICE/REVIEW/active-job branches.
- **No frames in MENU.** No frames are requested (`ask` stays cleared, and
  `_was_live` is False).
- **Idle handling** is unchanged: the screen-off check runs before the menu
  handling, and a dark screen draws nothing.
- **Rotation** is applied by the applier that Task 9 registers, so the viewfinder
  only calls `settings.set`. After a rotation change, force the next `_show`, since
  the same rows must be redrawn the other way up: reset the cached last rows.

- [ ] **Step 1: Write the failing tests**, using the existing fake
  controller/display/touch harness in `tests/test_viewfinder.py`:
  - a tap on `MENU` enters MENU and shows the menu image; a tap on `DONE` returns
    to LIVE;
  - a job that finishes while in MENU stays in MENU, and after `DONE` no review
    appears;
  - an active job while in MENU shows no colour bars;
  - each setting row's tap calls `settings.set` with the next value (use a real
    `SettingsControl` on `tmp_path` with recording appliers), and the file is
    written;
  - an applier that raises leaves the row's value unchanged;
  - an `upload` row tap calls `start()`; `status()` text appears in the rows
    (inspect them through a monkeypatched `render_menu` that records its
    arguments);
  - reset: one tap shows `TAP AGAIN`; a second tap within 3 s on the fake clock
    calls `reset()`; a second tap after 3 s only re-arms;
  - rows are hidden when `menu_rows` is `None`, when `upload` is `None` and when
    `settings` is `None`;
  - the network rows show addresses from the snapshot, and `-` when there are
    none;
  - no frame reads happen while in MENU (the fake camera counts reads);
  - the screen being off in MENU draws nothing.

  Run them. Expected: they fail.
- [ ] **Step 2: Implement.** Extend the module docstring's state list with MENU and
  its rules: no review while open, and photos still taken.
- [ ] **Step 3:** Run the full suite and ruff.
- [ ] **Step 4: Commit** `viewfinder: the MENU state`.

---

### Task 9: Wiring in `pifilm-capture`, and the docs

**Files:**
- Modify: `pifilm/capture/app.py`, `deploy/pifilm-capture.service.example`,
  `docs/lcd-viewfinder.md`, `docs/nextcloud-sync.md`, `docs/setup.md`, `CLAUDE.md`
- Test: `tests/test_app.py`

**Interfaces:**

Consumes Tasks 1 to 8.

Produces the flag `--nextcloud-env PATH` (`type=Path`, default `None`).

Behaviour, in `main`:

- **Loading the settings.**
  - Path: `settings_path = (args.state_dir or default_state_dir()).parent / SETTINGS_FILE`.
  - Defaults: `Settings(rotate=args.display_rotate, save_dng=not args.no_dng,
    ae_constraint=args.ae_constraint or "normal", ae_metering=args.ae_metering or
    "centre")`.
  - Call `load(settings_path, defaults)` after the backend is resolved and before
    `_open_display`. Print each returned line.
- **Applying them to the start-up arguments.**
  - Always: `args.display_rotate = eff.rotate`.
  - When `backend == "picamera2"` (not `--fake`, not V4L2): `args.no_dng = not
    eff.save_dng`, `args.ae_constraint = eff.ae_constraint`,
    `args.ae_metering = eff.ae_metering`.
  - Otherwise leave those three alone, so `_reject_picamera2_only_flags` behaves
    as today.
- **Creating the control.**
  `control = SettingsControl(settings_path, defaults, overrides, log=print)`.
  - On the Picamera2 backend, register:
    - `save_dng` → `lambda v: setattr(session, "save_dng", v)`;
    - `ae_constraint` → `camera.set_ae_constraint`;
    - `ae_metering` → `camera.set_ae_metering`.
  - `menu_rows` is
    `("save_dng",) + (("ae_constraint",) if camera.ae_constraint_supported else ()) + (("ae_metering",) if camera.ae_metering_supported else ())`
    on Picamera2, and `None` otherwise.
- **The upload.** It exists when `args.nextcloud_env` is set and a display is
  open: `Upload(args.nextcloud_env.expanduser(), log=print)`. Close it in `main`'s
  `finally`, next to the network monitor.
- **The look's name.** `look = Path(args.artifacts).name if args.artifacts else ""`.
- **`_run_viewfinder`** gains `settings`, `upload`, `look` and `menu_rows`, passed
  to `ViewfinderLoop`. Before the loop starts, it registers `rotate` →
  `lambda r: (display.set_rotate(r), touch.set_rotate(r))`.
- **Docstring.** Add a section to `app.py`'s module docstring, "The settings
  menu", in the style of "The disk-space warning".

Docs:

- **`docs/lcd-viewfinder.md`:**
  - the `MENU` key in the layout table;
  - a "Settings menu" section with the rows table, what is remembered, the reset,
    and the upload states;
  - remove the network tags from the live-view description (they are in the menu
    now);
  - an acceptance table copied from the spec's eight steps, marked "Not yet run".
- **`docs/nextcloud-sync.md`:**
  - the menu's upload button and `--nextcloud-env`;
  - the lock and its `SKIPPED: BUSY` / `another Nextcloud sync is running` line;
  - that the script now lives in `pifilm/nextcloud.py`.
- **`docs/setup.md`:** `--nextcloud-env` in the unit, in the Nextcloud section.
- **`deploy/pifilm-capture.service.example`:** add
  `--nextcloud-env /home/george/repos/pi-film-reversal/.env`, with a comment that
  the service never reads it and only hands the path to the upload child.
- **`CLAUDE.md`:**
  - capture: one sentence each for `settings.py` and `upload.py`;
  - display: `menu.py`, and the `MENU` key in the liveview paragraph; remove the
    network tags from the liveview sentence;
  - processing core: `nextcloud.py` and `envfile.py`;
  - the commands block: `python -m pifilm.nextcloud`.
- **README:** no change. If it needs one, the controller drafts it for the owner.

- [ ] **Step 1: Write the failing tests** in `tests/test_app.py`, following the
  existing `main` tests with fakes:
  - a saved `rotate: 180` reaches `_open_display` (check the
    `args.display_rotate` it receives) and prints the `settings:` line;
  - saved AE values with `--fake` do not trip `_reject_picamera2_only_flags`, and
    the service starts;
  - on a fake Picamera2 backend, the saved AE values reach the camera's
    constructor;
  - `--nextcloud-env` with a display creates an `Upload` that is closed on exit;
    without the flag, there is none;
  - `_run_viewfinder` registers the `rotate` applier, and calling it rotates the
    display and the touch.

  Run them. Expected: they fail.
- [ ] **Step 2: Implement the wiring.**
- [ ] **Step 3: Write the docs.**
- [ ] **Step 4:** Run the full suite and ruff. Then run
  `.venv/bin/pifilm-capture --fake --display fake --no-preview` for a few seconds
  (Ctrl-C), and check that the `viewfinder-last.png` it writes shows the `MENU`
  key.
- [ ] **Step 5: Commit**, in two commits:
  `capture: the settings menu wired into pifilm-capture` and
  `docs: the settings menu`.
