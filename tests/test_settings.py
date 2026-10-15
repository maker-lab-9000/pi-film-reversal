"""Tests for the settings the LCD menu changes, remembered in settings.json."""

from __future__ import annotations

import json
import os

import pytest

from pifilm.capture import settings as settings_mod
from pifilm.capture.settings import (
    SETTINGS_FILE,
    VERSION,
    Settings,
    SettingsControl,
    load,
    save,
)

DEFAULTS = Settings(rotate=0, save_dng=True, ae_constraint="highlight", ae_metering="centre")


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- loading -----------------------------------------------------------------


def test_load_without_a_file_gives_the_defaults(tmp_path):
    assert load(tmp_path / SETTINGS_FILE, DEFAULTS) == (DEFAULTS, {}, [])


def test_load_a_valid_file_with_all_four_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(
        path,
        {
            "version": 1,
            "rotate": 180,
            "save_dng": False,
            "ae_constraint": "shadows",
            "ae_metering": "spot",
        },
    )
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(
        rotate=180, save_dng=False, ae_constraint="shadows", ae_metering="spot"
    )
    assert overrides == {
        "rotate": 180,
        "save_dng": False,
        "ae_constraint": "shadows",
        "ae_metering": "spot",
    }
    assert lines == [
        "settings: rotate 180, dng off, ae-constraint shadows, ae-metering spot "
        "(saved in the menu; the unit says rotate 0, dng on, ae-constraint highlight, "
        "ae-metering centre)"
    ]


def test_load_summary_line_names_only_the_differing_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "rotate": 180, "ae_constraint": "shadows"})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(
        rotate=180, save_dng=True, ae_constraint="shadows", ae_metering="centre"
    )
    assert overrides == {"rotate": 180, "ae_constraint": "shadows"}
    assert lines == [
        "settings: rotate 180, ae-constraint shadows (saved in the menu; the unit says "
        "rotate 0, ae-constraint highlight)"
    ]


def test_load_partial_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "ae_metering": "matrix"})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(
        rotate=0, save_dng=True, ae_constraint="highlight", ae_metering="matrix"
    )
    assert overrides == {"ae_metering": "matrix"}
    assert lines == [
        "settings: ae-metering matrix (saved in the menu; the unit says ae-metering centre)"
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rotate", 90),
        ("rotate", True),
        ("rotate", "180"),
        ("rotate", 180.0),
        ("save_dng", "yes"),
        ("save_dng", 0),
        ("ae_constraint", "bright"),
        ("ae_metering", "average"),
        ("ae_metering", None),
    ],
)
def test_load_ignores_an_invalid_value_and_keeps_the_others(tmp_path, key, value):
    path = tmp_path / SETTINGS_FILE
    other = "ae_constraint" if key != "ae_constraint" else "ae_metering"
    other_value = "shadows" if other == "ae_constraint" else "spot"
    _write(path, {"version": 1, key: value, other: other_value})
    current, overrides, lines = load(path, DEFAULTS)
    assert overrides == {other: other_value}
    assert getattr(current, key) == getattr(DEFAULTS, key)
    assert getattr(current, other) == other_value
    assert lines[0] == f"settings: ignoring {key} in {path}"
    assert len(lines) == 2  # the ignore line, then the summary


def test_load_ignores_an_unknown_key(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "brightness": 3, "rotate": 180})
    current, overrides, lines = load(path, DEFAULTS)
    assert current.rotate == 180
    assert overrides == {"rotate": 180}
    assert lines[0] == f"settings: ignoring brightness in {path}"
    assert len(lines) == 2


@pytest.mark.parametrize(
    ("content", "reason_part"),
    [
        ("{not json", "not JSON"),
        ("[1, 2]", "not an object"),
        (json.dumps({"version": 2, "rotate": 180}), "version"),
        (json.dumps({"rotate": 180}), "version"),
        (json.dumps({"version": True, "rotate": 180}), "version"),
    ],
)
def test_load_ignores_a_bad_file_and_leaves_it(tmp_path, content, reason_part):
    path = tmp_path / SETTINGS_FILE
    path.write_text(content, encoding="utf-8")
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert len(lines) == 1
    assert lines[0].startswith(f"settings: ignoring {path} (")
    assert reason_part in lines[0]
    assert path.read_text(encoding="utf-8") == content


def test_load_ignores_a_directory_at_the_path(tmp_path):
    path = tmp_path / SETTINGS_FILE
    path.mkdir()
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert len(lines) == 1
    assert lines[0].startswith(f"settings: ignoring {path} (")
    assert path.is_dir()


def test_load_an_override_equal_to_its_default_gives_no_summary(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "rotate": 0, "save_dng": True})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == DEFAULTS
    assert overrides == {"rotate": 0, "save_dng": True}
    assert lines == []


# --- saving ------------------------------------------------------------------


def test_save_round_trips_through_load(tmp_path):
    path = tmp_path / SETTINGS_FILE
    save(path, {"rotate": 180, "save_dng": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": VERSION,
        "rotate": 180,
        "save_dng": False,
    }
    current, overrides, _ = load(path, DEFAULTS)
    assert overrides == {"rotate": 180, "save_dng": False}
    assert current.rotate == 180 and current.save_dng is False


def test_save_creates_the_parent_directory(tmp_path):
    path = tmp_path / "state" / "pifilm" / SETTINGS_FILE
    save(path, {"ae_metering": "spot"})
    assert load(path, DEFAULTS)[1] == {"ae_metering": "spot"}


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / SETTINGS_FILE
    save(path, {"rotate": 180})
    before = path.read_bytes()

    def broken_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(settings_mod.os, "replace", broken_replace)
    with pytest.raises(OSError):
        save(path, {"rotate": 0})
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [SETTINGS_FILE]


# --- the control -------------------------------------------------------------


def _control(tmp_path, overrides=None):
    lines: list[str] = []
    path = tmp_path / SETTINGS_FILE
    control = SettingsControl(path, DEFAULTS, overrides or {}, log=lines.append)
    return control, path, lines


def test_control_current_reflects_the_overrides(tmp_path):
    control, _, _ = _control(tmp_path, {"rotate": 180})
    assert control.current == Settings(
        rotate=180, save_dng=True, ae_constraint="highlight", ae_metering="centre"
    )


def test_set_calls_the_applier_then_saves(tmp_path):
    control, path, lines = _control(tmp_path)
    calls = []

    def applier(value):
        calls.append((value, path.exists()))

    control.register("rotate", applier)
    assert control.set("rotate", 180) is True
    assert calls == [(180, False)]
    assert control.current.rotate == 180
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "rotate": 180}
    assert lines == []


def test_set_without_an_applier_still_saves(tmp_path):
    control, path, _ = _control(tmp_path)
    assert control.set("ae_metering", "spot") is True
    assert load(path, DEFAULTS)[1] == {"ae_metering": "spot"}


def test_set_with_an_applier_that_raises_changes_nothing(tmp_path):
    control, path, lines = _control(tmp_path)
    save(path, {"save_dng": False})
    before = path.read_bytes()
    control = SettingsControl(path, DEFAULTS, {"save_dng": False}, log=lines.append)

    def applier(value):
        raise RuntimeError("camera busy")

    control.register("ae_constraint", applier)
    assert control.set("ae_constraint", "shadows") is False
    assert lines == ["settings: cannot apply ae_constraint shadows (camera busy)"]
    assert control.current.ae_constraint == "highlight"
    assert path.read_bytes() == before


def test_set_with_a_save_failure_keeps_the_change(tmp_path, monkeypatch):
    control, path, lines = _control(tmp_path)

    def broken_save(path, overrides):
        raise OSError("disk full")

    monkeypatch.setattr(settings_mod, "save", broken_save)
    assert control.set("rotate", 180) is True
    assert control.current.rotate == 180
    assert lines == ["settings: cannot save (disk full)"]
    assert not path.exists()


def test_set_back_to_the_default_removes_the_key(tmp_path):
    control, path, _ = _control(tmp_path)
    control.set("rotate", 180)
    control.set("save_dng", False)
    control.set("rotate", 0)
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "save_dng": False}
    assert control.current.rotate == 0


def test_set_the_same_value_applies_and_saves_again(tmp_path):
    control, path, _ = _control(tmp_path)
    calls = []
    control.register("ae_metering", calls.append)
    control.set("ae_metering", "spot")
    assert control.set("ae_metering", "spot") is True
    assert calls == ["spot", "spot"]
    assert load(path, DEFAULTS)[1] == {"ae_metering": "spot"}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rotate", 90),
        ("rotate", True),
        ("save_dng", "yes"),
        ("ae_metering", "average"),
        ("brightness", 3),
    ],
)
def test_set_an_invalid_key_or_value_raises(tmp_path, key, value):
    control, path, _ = _control(tmp_path)
    with pytest.raises(ValueError):
        control.set(key, value)
    assert control.current == DEFAULTS
    assert not path.exists()


def test_register_an_unknown_key_raises(tmp_path):
    control, _, _ = _control(tmp_path)
    with pytest.raises(ValueError):
        control.register("brightness", lambda value: None)


def test_reset_applies_only_the_differing_keys_and_deletes_the_file(tmp_path):
    path = tmp_path / SETTINGS_FILE
    overrides = {"rotate": 180, "ae_metering": "centre"}
    save(path, overrides)
    lines: list[str] = []
    control = SettingsControl(path, DEFAULTS, overrides, log=lines.append)
    calls = []
    for key in settings_mod.KEYS:
        control.register(key, lambda value, key=key: calls.append((key, value)))
    control.reset()
    assert calls == [("rotate", 0)]
    assert control.current == DEFAULTS
    assert not path.exists()
    assert lines == ["settings: reset to the unit's flags"]


def test_reset_keeps_a_key_whose_applier_raises(tmp_path):
    path = tmp_path / SETTINGS_FILE
    overrides = {"rotate": 180, "save_dng": False}
    save(path, overrides)
    lines: list[str] = []
    control = SettingsControl(path, DEFAULTS, overrides, log=lines.append)

    def broken(value):
        raise RuntimeError("no display")

    control.register("rotate", broken)
    control.reset()
    assert control.current.rotate == 180
    assert control.current.save_dng is True
    assert lines == [
        "settings: cannot apply rotate 0 (no display)",
        "settings: reset to the unit's flags",
    ]
    assert not path.exists()


def test_reset_without_a_file_does_not_raise(tmp_path):
    control, path, lines = _control(tmp_path)
    control.reset()
    assert not path.exists()
    assert lines == ["settings: reset to the unit's flags"]


def test_reset_logs_a_file_that_cannot_be_deleted(tmp_path):
    lines: list[str] = []
    path = tmp_path / SETTINGS_FILE
    path.mkdir()  # unlink of a directory raises an OSError other than not-found
    control = SettingsControl(path, DEFAULTS, {"rotate": 180}, log=lines.append)
    control.reset()
    assert control.current == DEFAULTS
    assert len(lines) == 2
    assert lines[0].startswith("settings: cannot delete ")
    assert lines[1] == "settings: reset to the unit's flags"


def test_settings_file_name_is_settings_json():
    assert SETTINGS_FILE == "settings.json"
    assert os.path.basename(SETTINGS_FILE) == SETTINGS_FILE
