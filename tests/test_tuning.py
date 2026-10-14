import hashlib
import json

import pytest

from pifilm import tuning
from pifilm.tuning import ENV_VAR, Resolved, check, fingerprint, resolve, short

TUNING = {"version": 2.0, "algorithms": [{"rpi.ccm": {"ccms": [{"ct": 2850, "ccm": [1, 2, 3]}]}}]}


def _formula(obj):
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _no_load(name):
    raise AssertionError(f"load must not be called, got {name!r}")


# fingerprint


def test_fingerprint_is_the_formula():
    assert fingerprint(TUNING) == _formula(TUNING)
    assert len(fingerprint(TUNING)) == 40


def test_key_order_and_whitespace_of_the_source_json_do_not_change_it():
    compact = '{"a":1,"b":{"c":[1,2],"d":"x"}}'
    loose = '{\n  "b": {"d": "x",   "c": [1, 2]},\n  "a": 1\n}\n'
    assert fingerprint(json.loads(compact)) == fingerprint(json.loads(loose))


def test_one_changed_nested_value_changes_it():
    changed = json.loads(json.dumps(TUNING))
    changed["algorithms"][0]["rpi.ccm"]["ccms"][0]["ccm"][1] = 2.0001
    assert fingerprint(changed) != fingerprint(TUNING)


def test_short_is_twelve_hex_digits():
    assert short("0123456789abcdef" * 2 + "01234567") == "0123456789ab"


# resolve, given a name


def test_a_given_name_fingerprints_the_loaded_dict_without_loading_again():
    got = resolve("imx477.json", "imx477", load=_no_load, environ={}, loaded=TUNING)
    assert got == Resolved(label="imx477.json", source="imx477.json",
                           sha1=fingerprint(TUNING), reason=None)


def test_a_given_name_without_a_loaded_dict_is_loaded_by_name():
    calls = []

    def load(name):
        calls.append(name)
        return TUNING

    got = resolve("custom.json", "imx477", load=load, environ={ENV_VAR: "/ignored.json"})
    assert calls == ["custom.json"]
    assert got.label == "custom.json"
    assert got.sha1 == fingerprint(TUNING)


def test_a_given_name_whose_load_fails_raises():
    def load(name):
        raise FileNotFoundError(name)

    with pytest.raises(FileNotFoundError):
        resolve("missing.json", "imx477", load=load, environ={})


# resolve, automatic


def test_auto_reads_the_environment_variable_file(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(TUNING, indent=4))
    got = resolve(None, "imx477", load=_no_load, environ={ENV_VAR: str(path)})
    assert got.label == "auto:imx477"
    assert got.source == str(path)
    assert got.sha1 == fingerprint(TUNING)
    assert got.reason is None


def test_auto_with_the_environment_variable_naming_a_missing_file(tmp_path):
    path = tmp_path / "missing.json"
    got = resolve(None, "imx477", load=_no_load, environ={ENV_VAR: str(path)})
    assert got.sha1 is None
    assert got.reason and str(path) in got.reason
    assert got.label == "auto:imx477"


def test_auto_with_the_environment_variable_naming_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    got = resolve(None, "imx477", load=_no_load, environ={ENV_VAR: str(path)})
    assert got.sha1 is None
    assert got.reason


def test_auto_without_the_environment_variable_loads_the_sensor_default():
    calls = []

    def load(name):
        calls.append(name)
        return TUNING

    got = resolve(None, "IMX477", load=load, environ={})
    assert calls == ["imx477.json"]
    assert got.label == "auto:IMX477"
    assert got.source == "imx477.json"
    assert got.sha1 == fingerprint(TUNING)


def test_auto_whose_load_fails_gives_no_fingerprint():
    def load(name):
        raise RuntimeError("no such tuning file")

    got = resolve(None, "imx477", load=load, environ={})
    assert got.sha1 is None
    assert "no such tuning file" in got.reason


# check

LOOK = "a" * 40
CAMERA = Resolved(label="imx477.json", source="imx477.json", sha1="b" * 40, reason=None)


def test_a_look_without_a_fingerprint_is_not_checked():
    assert check(None, CAMERA) == (None, None)


def test_no_camera_is_not_checked():
    assert check(LOOK, None) == (None, None)


def test_a_camera_that_could_not_fingerprint_cannot_check():
    camera = Resolved(label="auto:imx477", source=None, sha1=None, reason="file missing")
    assert check(LOOK, camera) == (
        None, "tuning: cannot check the look's tuning (file missing)")


def test_an_equal_fingerprint_matches():
    camera = Resolved(label="imx477.json", source="imx477.json", sha1=LOOK, reason=None)
    assert check(LOOK, camera) == (True, "tuning: matches the look (aaaaaaaaaaaa)")


def test_a_different_fingerprint_differs():
    assert check(LOOK, CAMERA) == (
        False,
        "tuning: DIFFERS from the look: camera bbbbbbbbbbbb (imx477.json), "
        "look aaaaaaaaaaaa; colours may be off",
    )


def test_the_module_imports_no_picamera2():
    assert "picamera2" not in vars(tuning)
