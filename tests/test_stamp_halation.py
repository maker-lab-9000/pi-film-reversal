import json
import os

import pytest

from pifilm.artifacts import Artifacts, ArtifactsError, stamp_halation, write_artifact
from pifilm.halation import HalationParams
from scripts import stamp_halation as script


@pytest.fixture
def look(tmp_path):
    base = Artifacts.default()
    path = write_artifact(tmp_path / "look", base.lut, base.normalize, base.grain,
                          training={"note": "kept"},
                          tuning_sha1="5404134af8256b674e5a8f93fe1bb616a1e0407e")
    os.chmod(path / "params.json", 0o644)
    return path


def test_stamp_turns_it_on_keeps_the_rest_and_raises_the_version(look):
    before = json.loads((look / "params.json").read_text())
    stamp_halation(look, HalationParams(enabled=True))
    after = json.loads((look / "params.json").read_text())
    assert after["version"] == 5 and after["halation"]["enabled"] is True
    for key in ("lut_sha1", "normalize", "grain", "training", "tuning_sha1"):
        assert after[key] == before[key]
    assert [k for k in after if k != "halation"] == list(before)   # order kept
    assert oct((look / "params.json").stat().st_mode & 0o777) == "0o644"
    assert Artifacts.load(look).halation.active


def test_stamp_off_never_lowers_the_version(look):
    stamp_halation(look, HalationParams(enabled=True))
    stamp_halation(look, HalationParams(enabled=False))
    raw = json.loads((look / "params.json").read_text())
    assert raw["version"] == 5 and raw["halation"]["enabled"] is False


def test_a_broken_look_is_refused_untouched(look):
    raw = json.loads((look / "params.json").read_text())
    raw["lut_sha1"] = "0" * 40
    (look / "params.json").write_text(json.dumps(raw))
    text = (look / "params.json").read_text()
    with pytest.raises(ArtifactsError):
        stamp_halation(look, HalationParams(enabled=True))
    assert (look / "params.json").read_text() == text


def test_the_script_stamps_the_chosen_values(look, capsys):
    assert script.main(["--artifacts", str(look)]) == 0
    art = Artifacts.load(look)
    assert art.halation == HalationParams(enabled=True)
    assert "stamped" in capsys.readouterr().out


def test_the_script_refuses_a_bad_value_and_leaves_the_file(look, capsys):
    before = (look / "params.json").read_bytes()
    assert script.main(["--artifacts", str(look), "--headroom", "2"]) == 2
    assert (look / "params.json").read_bytes() == before
    assert "headroom" in capsys.readouterr().err
