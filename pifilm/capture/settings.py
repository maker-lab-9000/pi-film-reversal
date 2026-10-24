"""The settings the LCD menu changes, remembered in ``settings.json``.

Why only these seven (DNG saving, the look's halation, the auto-exposure
constraint and metering mode, and the camera's sharpness, contrast and
saturation): they are the ones a photographer wants to change in the field, away
from a keyboard, and each can be applied to a running session at once. Every
other flag (the look, the camera, the network, where photographs go, the
display's rotation) is set up once and stays with the systemd unit alone; a
``rotate`` key left in an older file is an unknown key, ignored with one line.

``halation`` has no flag: its unit value is True, "as the look says", and False
leaves the glow out of the photographs of a look that has one (a look without
halation has nothing to leave out). It was added without a new ``VERSION``: an
older build reads a file with it as an unknown key, ignored with one line, and
keeps the look's halation, which is what it always did.

``sharpness``, ``contrast`` and ``saturation`` have no flags either: each is
1.0 (the ISP's normal rendering) unless the menu moved it to another of
``IMAGE_STEPS``. They change the picture before pifilm's look, and every look
so far was trained on captures taken at 1.0, so a value away from 1.0 grades a
picture the look never saw; the live view's amber ``IMG`` tag is there so it is
not forgotten. They too came without a new ``VERSION``: an older build ignores
each as an unknown key with one line and shoots at 1.0.

Why a saved value wins over the unit's flag: a setting changed in the menu has to
survive a restart, a power cut or a reboot, or the menu is a trap that undoes
itself. The unit's flags become the starting values, and the file holds only the
keys the menu has moved away from them. Because the effective setting is then no
longer what the unit says, start-up logs one line naming what came from the menu
and what the unit says, so someone reading the journal is not misled by the unit
file; and the menu's reset puts every setting back to the flags and deletes the
file, so there is always a way back without a shell.

Why a bad file is never fatal: the camera has to start whatever is on disk. A
file that cannot be read, is not JSON, is not an object or has another version
is ignored as a whole with one line and left in place for a person to look at
(a later build may understand a newer version). A single unknown or invalid key
is ignored with one line and the valid keys still apply.

Why the write is atomic: the file is written after every change in the menu,
and a power cut can come at any moment on a battery-powered camera. It is written
to a temporary file in the same directory, flushed, fsynced and moved into place
with ``os.replace``, so the next start sees the old file or the new one, never
half of one.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

SETTINGS_FILE = "settings.json"
VERSION = 1
AE_CONSTRAINTS = ("normal", "highlight", "shadows")
AE_METERINGS = ("centre", "spot", "matrix")
# The camera's own image controls (libcamera's Sharpness, Contrast, Saturation;
# 1.0 is normal for each) and the values the menu steps through.
IMAGE_KEYS = ("sharpness", "contrast", "saturation")
IMAGE_STEPS = (0.5, 0.75, 1.0, 1.25, 1.5)
IMAGE_NORMAL = 1.0
KEYS = ("save_dng", "halation", "ae_constraint", "ae_metering") + IMAGE_KEYS


@dataclass(frozen=True)
class Settings:
    """The seven values the menu can change. ``halation`` defaults to True, the
    look's own halation, and the image controls to 1.0, because no flag sets
    them."""

    save_dng: bool
    ae_constraint: str
    ae_metering: str
    halation: bool = True
    sharpness: float = IMAGE_NORMAL
    contrast: float = IMAGE_NORMAL
    saturation: float = IMAGE_NORMAL


def format_step(value: float) -> str:
    """``1.0``, ``0.75``, ``1.25``, ``1.5``: two decimals at most, one at least."""
    text = f"{float(value):.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _valid(key: str, value: object) -> bool:
    if key in ("save_dng", "halation"):
        return isinstance(value, bool)
    if key in IMAGE_KEYS:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and value in IMAGE_STEPS)
    if key == "ae_constraint":
        return isinstance(value, str) and value in AE_CONSTRAINTS
    if key == "ae_metering":
        return isinstance(value, str) and value in AE_METERINGS
    return False


def _describe(key: str, value: object) -> str:
    """``ae-constraint shadows``, ``dng off``, ``halation off``, ``contrast 1.25``."""
    if key == "save_dng":
        return f"dng {'on' if value else 'off'}"
    if key == "halation":
        return f"halation {'on' if value else 'off'}"
    if key in IMAGE_KEYS:
        return f"{key} {format_step(value)}"
    return f"{key.replace('_', '-')} {value}"


def _read(path: Path) -> tuple[dict | None, str | None]:
    """(the parsed object, None) or (None, the reason it is ignored).

    A missing file is ``(None, None)``. Any other error reading it is a reason,
    not an exception: an unreadable state directory (EACCES) must not stop the
    camera starting, and neither may a file nested deeply enough to exhaust
    the JSON parser's recursion.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"cannot read: {exc}"
    except UnicodeDecodeError:
        return None, "not JSON"
    try:
        data = json.loads(text)
    except ValueError:
        return None, "not JSON"
    except RecursionError:
        return None, "nested too deeply"
    if not isinstance(data, dict):
        return None, "not an object"
    version = data.get("version")
    if isinstance(version, bool) or version != VERSION:
        return None, f"version {version!r}, expected {VERSION}"
    return data, None


def load(path: Path, defaults: Settings) -> tuple[Settings, dict[str, object], list[str]]:
    """(effective settings, the valid overrides read, log lines). Never raises."""
    data, reason = _read(path)
    if data is None and reason is None:
        return defaults, {}, []
    if data is None:
        return defaults, {}, [f"settings: ignoring {path} ({reason})"]
    lines: list[str] = []
    overrides: dict[str, object] = {}
    for key, value in data.items():
        if key == "version":
            continue
        if key in KEYS and _valid(key, value):
            overrides[key] = float(value) if key in IMAGE_KEYS else value
        else:
            lines.append(f"settings: ignoring {key} in {path}")
    current = replace(defaults, **overrides)
    differing = [key for key in KEYS if getattr(current, key) != getattr(defaults, key)]
    if differing:
        menu = ", ".join(_describe(key, getattr(current, key)) for key in differing)
        unit = ", ".join(_describe(key, getattr(defaults, key)) for key in differing)
        lines.append(f"settings: {menu} (saved in the menu; the unit says {unit})")
    return current, overrides, lines


def save(path: Path, overrides: Mapping[str, object]) -> None:
    """Write {"version": 1, **overrides} atomically. Raises OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": VERSION, **overrides}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SettingsControl:
    """The menu's view of the seven settings: apply, remember, reset.

    An applier puts a value into effect in the running session (send the camera
    a control, start or stop saving DNGs or applying halation). It runs before
    anything is remembered, so a value the session refused is never saved for
    the next start.
    """

    def __init__(
        self,
        path: Path,
        defaults: Settings,
        overrides: Mapping[str, object],
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        self._path = path
        self._defaults = defaults
        self._overrides: dict[str, object] = dict(overrides)
        self._current = replace(defaults, **self._overrides)
        self._log = log
        self._appliers: dict[str, Callable[[object], None]] = {}

    @property
    def current(self) -> Settings:
        return self._current

    def register(self, key: str, applier: Callable[[object], None]) -> None:
        if key not in KEYS:
            raise ValueError(f"unknown setting {key!r}")
        self._appliers[key] = applier

    def _apply(self, key: str, value: object) -> bool:
        applier = self._appliers.get(key)
        if applier is None:
            return True
        try:
            applier(value)
        except Exception as exc:
            self._log(f"settings: cannot apply {key} {value} ({exc})")
            return False
        return True

    def set(self, key: str, value: object) -> bool:
        """Apply ``value``, log it and remember it. True when applied."""
        if key not in KEYS:
            raise ValueError(f"unknown setting {key!r}")
        if not _valid(key, value):
            raise ValueError(f"invalid value {value!r} for {key}")
        if key in IMAGE_KEYS:
            value = float(value)
        if not self._apply(key, value):
            return False
        # One line per change, so the journal shows the menu took effect (the
        # Pi's frame metadata does not report the AE modes); before the save,
        # whose failure keeps its own line.
        self._log(f"settings: {_describe(key, value)}")
        self._current = replace(self._current, **{key: value})
        if value == getattr(self._defaults, key):
            self._overrides.pop(key, None)
        else:
            self._overrides[key] = value
        try:
            save(self._path, self._overrides)
        except OSError as exc:
            self._log(f"settings: cannot save ({exc})")
        return True

    def _restore(self, keys: tuple[str, ...]) -> None:
        """Put ``keys`` back to their defaults; a key whose applier raises stays."""
        for key in keys:
            default = getattr(self._defaults, key)
            if getattr(self._current, key) == default:
                self._overrides.pop(key, None)
                continue
            if self._apply(key, default):
                self._current = replace(self._current, **{key: default})
                self._overrides.pop(key, None)

    def reset_image(self) -> None:
        """Put only sharpness, contrast and saturation back to 1.0 and save.

        The image page's own reset: the other settings and their saved values
        stay as they are.
        """
        self._restore(IMAGE_KEYS)
        self._log("settings: sharpness, contrast, saturation reset to 1.0")
        try:
            save(self._path, self._overrides)
        except OSError as exc:
            self._log(f"settings: cannot save ({exc})")

    def reset(self) -> None:
        """Put every setting back to the unit's flags and delete the file."""
        self._restore(KEYS)
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._log(f"settings: cannot delete {self._path} ({exc})")
        self._log("settings: reset to the unit's flags")
