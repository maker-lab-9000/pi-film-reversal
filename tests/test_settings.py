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

DEFAULTS = Settings(save_dng=True, ae_constraint="highlight", ae_metering="centre")


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- loading -----------------------------------------------------------------


def test_load_without_a_file_gives_the_defaults(tmp_path):
    assert load(tmp_path / SETTINGS_FILE, DEFAULTS) == (DEFAULTS, {}, [])


def test_load_a_valid_file_with_all_three_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(
        path,
        {"version": 1, "save_dng": False, "ae_constraint": "shadows", "ae_metering": "spot"},
    )
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(save_dng=False, ae_constraint="shadows", ae_metering="spot")
    assert overrides == {"save_dng": False, "ae_constraint": "shadows", "ae_metering": "spot"}
    assert lines == [
        "settings: dng off, ae-constraint shadows, ae-metering spot "
        "(saved in the menu; the unit says dng on, ae-constraint highlight, "
        "ae-metering centre)"
    ]


def test_load_summary_line_names_only_the_differing_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "save_dng": False, "ae_constraint": "shadows"})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(save_dng=False, ae_constraint="shadows", ae_metering="centre")
    assert overrides == {"save_dng": False, "ae_constraint": "shadows"}
    assert lines == [
        "settings: dng off, ae-constraint shadows (saved in the menu; the unit says "
        "dng on, ae-constraint highlight)"
    ]


def test_load_partial_keys(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "ae_metering": "matrix"})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == Settings(save_dng=True, ae_constraint="highlight", ae_metering="matrix")
    assert overrides == {"ae_metering": "matrix"}
    assert lines == [
        "settings: ae-metering matrix (saved in the menu; the unit says ae-metering centre)"
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("save_dng", "yes"),
        ("save_dng", 0),
        ("save_dng", 1),
        ("save_dng", None),
        ("halation", "off"),
        ("halation", 0),
        ("halation", None),
        ("ae_constraint", "bright"),
        ("ae_constraint", 3),
        ("ae_constraint", True),
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


def test_halation_defaults_to_the_looks_own():
    assert DEFAULTS.halation is True
    assert settings_mod.KEYS.index("halation") == settings_mod.KEYS.index("save_dng") + 1


def test_load_a_saved_halation_off(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "halation": False})
    current, overrides, lines = load(path, DEFAULTS)
    assert current.halation is False and current.save_dng is True
    assert overrides == {"halation": False}
    assert lines == [
        "settings: halation off (saved in the menu; the unit says halation on)"
    ]


def test_set_halation_off_saves_and_back_on_removes_the_key(tmp_path):
    control, path, _ = _control(tmp_path)
    seen = []
    control.register("halation", seen.append)
    assert control.set("halation", False) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "halation": False}
    assert load(path, DEFAULTS)[0].halation is False
    assert control.set("halation", True) is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert seen == [False, True]


def test_reset_puts_halation_back_on(tmp_path):
    path = tmp_path / SETTINGS_FILE
    save(path, {"halation": False})
    control = SettingsControl(path, DEFAULTS, {"halation": False}, log=lambda _: None)
    calls = []
    control.register("halation", calls.append)
    control.reset()
    assert calls == [True]
    assert control.current.halation is True
    assert not path.exists()


def test_load_ignores_an_unknown_key(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "brightness": 3, "save_dng": False})
    current, overrides, lines = load(path, DEFAULTS)
    assert current.save_dng is False
    assert overrides == {"save_dng": False}
    assert lines[0] == f"settings: ignoring brightness in {path}"
    assert len(lines) == 2


def test_load_ignores_a_rotate_key_left_by_an_older_build(tmp_path):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "rotate": 180})
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert lines == [f"settings: ignoring rotate in {path}"]


@pytest.mark.parametrize(
    ("content", "reason_part"),
    [
        ("{not json", "not JSON"),
        ("[1, 2]", "not an object"),
        (json.dumps({"version": 2, "save_dng": False}), "version"),
        (json.dumps({"save_dng": False}), "version"),
        (json.dumps({"version": True, "save_dng": False}), "version"),
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
    _write(path, {"version": 1, "ae_metering": "centre", "save_dng": True})
    current, overrides, lines = load(path, DEFAULTS)
    assert current == DEFAULTS
    assert overrides == {"ae_metering": "centre", "save_dng": True}
    assert lines == []


# --- saving ------------------------------------------------------------------


def test_save_round_trips_through_load(tmp_path):
    path = tmp_path / SETTINGS_FILE
    save(path, {"ae_constraint": "shadows", "save_dng": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": VERSION,
        "ae_constraint": "shadows",
        "save_dng": False,
    }
    current, overrides, _ = load(path, DEFAULTS)
    assert overrides == {"ae_constraint": "shadows", "save_dng": False}
    assert current.ae_constraint == "shadows" and current.save_dng is False


def test_save_creates_the_parent_directory(tmp_path):
    path = tmp_path / "state" / "pifilm" / SETTINGS_FILE
    save(path, {"ae_metering": "spot"})
    assert load(path, DEFAULTS)[1] == {"ae_metering": "spot"}


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / SETTINGS_FILE
    save(path, {"save_dng": False})
    before = path.read_bytes()

    def broken_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(settings_mod.os, "replace", broken_replace)
    with pytest.raises(OSError):
        save(path, {"save_dng": True})
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [SETTINGS_FILE]


# --- the control -------------------------------------------------------------


def _control(tmp_path, overrides=None):
    lines: list[str] = []
    path = tmp_path / SETTINGS_FILE
    control = SettingsControl(path, DEFAULTS, overrides or {}, log=lines.append)
    return control, path, lines


def test_control_current_reflects_the_overrides(tmp_path):
    control, _, _ = _control(tmp_path, {"ae_metering": "spot"})
    assert control.current == Settings(save_dng=True, ae_constraint="highlight", ae_metering="spot")


def test_set_calls_the_applier_then_saves(tmp_path):
    control, path, lines = _control(tmp_path)
    calls = []

    def applier(value):
        calls.append((value, path.exists()))

    control.register("save_dng", applier)
    assert control.set("save_dng", False) is True
    assert calls == [(False, False)]
    assert control.current.save_dng is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "save_dng": False}
    assert lines == ["settings: dng off"]


@pytest.mark.parametrize(
    ("key", "value", "line"),
    [
        ("ae_metering", "matrix", "settings: ae-metering matrix"),
        ("ae_constraint", "shadows", "settings: ae-constraint shadows"),
        ("save_dng", False, "settings: dng off"),
        ("save_dng", True, "settings: dng on"),
        ("halation", False, "settings: halation off"),
        ("halation", True, "settings: halation on"),
    ],
)
def test_set_logs_one_line_for_the_change(tmp_path, key, value, line):
    control, _, lines = _control(tmp_path)
    control.register(key, lambda value: None)
    assert control.set(key, value) is True
    assert lines == [line]


def test_set_logs_the_change_after_the_applier_ran(tmp_path):
    control, _, lines = _control(tmp_path)
    seen = []
    control.register("ae_metering", lambda value: seen.append(list(lines)))
    control.set("ae_metering", "spot")
    assert seen == [[]]
    assert lines == ["settings: ae-metering spot"]


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
    assert control.set("ae_constraint", "normal") is True
    assert control.current.ae_constraint == "normal"
    assert lines == ["settings: ae-constraint normal", "settings: cannot save (disk full)"]
    assert not path.exists()


def test_set_back_to_the_default_removes_the_key(tmp_path):
    control, path, _ = _control(tmp_path)
    control.set("ae_metering", "matrix")
    control.set("save_dng", False)
    control.set("ae_metering", "centre")
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "save_dng": False}
    assert control.current.ae_metering == "centre"


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
        ("save_dng", "yes"),
        ("save_dng", 1),
        ("halation", "on"),
        ("halation", 1),
        ("ae_constraint", 3),
        ("ae_metering", "average"),
        ("brightness", 3),
        ("rotate", 180),
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
    overrides = {"ae_constraint": "shadows", "ae_metering": "centre"}
    save(path, overrides)
    lines: list[str] = []
    control = SettingsControl(path, DEFAULTS, overrides, log=lines.append)
    calls = []
    for key in settings_mod.KEYS:
        control.register(key, lambda value, key=key: calls.append((key, value)))
    control.reset()
    assert calls == [("ae_constraint", "highlight")]
    assert control.current == DEFAULTS
    assert not path.exists()
    assert lines == ["settings: reset to the unit's flags"]


def test_reset_keeps_a_key_whose_applier_raises(tmp_path):
    path = tmp_path / SETTINGS_FILE
    overrides = {"ae_metering": "spot", "save_dng": False}
    save(path, overrides)
    lines: list[str] = []
    control = SettingsControl(path, DEFAULTS, overrides, log=lines.append)

    def broken(value):
        raise RuntimeError("camera busy")

    control.register("ae_metering", broken)
    control.reset()
    assert control.current.ae_metering == "spot"
    assert control.current.save_dng is True
    assert lines == [
        "settings: cannot apply ae_metering centre (camera busy)",
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
    control = SettingsControl(path, DEFAULTS, {"save_dng": False}, log=lines.append)
    control.reset()
    assert control.current == DEFAULTS
    assert len(lines) == 2
    assert lines[0].startswith("settings: cannot delete ")
    assert lines[1] == "settings: reset to the unit's flags"


def test_settings_file_name_is_settings_json():
    assert SETTINGS_FILE == "settings.json"
    assert os.path.basename(SETTINGS_FILE) == SETTINGS_FILE


def test_load_survives_an_unreadable_state_directory(tmp_path, monkeypatch):
    path = tmp_path / SETTINGS_FILE
    _write(path, {"version": 1, "save_dng": False})

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(settings_mod.Path, "exists", denied)
    monkeypatch.setattr(settings_mod.Path, "is_symlink", denied)
    monkeypatch.setattr(settings_mod.Path, "read_text", denied)
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert len(lines) == 1
    assert lines[0].startswith(f"settings: ignoring {path} (cannot read: ")
    assert "Permission denied" in lines[0]


def test_load_survives_deeply_nested_json(tmp_path):
    path = tmp_path / SETTINGS_FILE
    content = "[" * 100000 + "]" * 100000
    path.write_text(content, encoding="utf-8")
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert lines == [f"settings: ignoring {path} (nested too deeply)"]
    assert path.read_text(encoding="utf-8") == content


def test_load_survives_a_file_where_the_state_directory_should_be(tmp_path):
    parent = tmp_path / "pifilm"
    parent.write_text("not a directory", encoding="utf-8")
    path = parent / SETTINGS_FILE
    current, overrides, lines = load(path, DEFAULTS)
    assert (current, overrides) == (DEFAULTS, {})
    assert len(lines) == 1
    assert lines[0].startswith(f"settings: ignoring {path} (cannot read: ")
