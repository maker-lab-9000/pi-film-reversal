import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from pifilm.artifacts import (
    PARAMS_VERSION,
    Artifacts,
    ArtifactsError,
    publish,
    write_artifact,
)
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D, sha1_hex, write_cube
from pifilm.normalize import NormalizeParams


@pytest.fixture
def staged(tmp_path):
    d = tmp_path / "staging"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams(), training={"note": "t"})
    return d


def test_write_then_load(staged):
    data = json.loads((staged / "params.json").read_text())
    # Gaussian grain keeps version 2, so a Pi on older code still reads it.
    assert data["version"] == 2
    assert data["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    art = Artifacts.load(staged)
    assert art.lut.size == 9
    assert art.normalize == NormalizeParams()
    assert art.grain == GrainParams()
    assert art.training == {"note": "t"}
    assert art.lut_sha1 == data["lut_sha1"]


def test_missing_params_is_clear(tmp_path):
    with pytest.raises(ArtifactsError, match="params.json"):
        Artifacts.load(tmp_path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{not json", "is not valid JSON"),
        ('{"version": 99}', "unsupported params version"),
        ('{"lut_file": "k.cube"}', "unsupported params version"),
        ("[1, 2, 3]", "top level must be a JSON object"),
        ('{"version": 2, "normalize": 5}', "'normalize' must be a JSON object"),
        ('{"version": 2, "grain": "x"}', "'grain' must be a JSON object"),
    ],
)
def test_schema_rejections(tmp_path, payload, message):
    """Match whole phrases, not tokens.

    These assertions read `tmp_path`, whose name pytest derives from the test
    id — which for a parametrised case embeds the expected token. Bare tokens
    like "version" survive today only because pytest truncates the directory
    name before reaching them; that is an implementation detail, not a
    guarantee. Phrases do not depend on it.
    """
    (tmp_path / "params.json").write_text(payload)
    with pytest.raises(ArtifactsError, match=re.escape(message)):
        Artifacts.load(tmp_path)


def test_missing_cube_is_clear(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"version": 2, "lut_file": "k.cube"}))
    with pytest.raises(ArtifactsError, match="k.cube"):
        Artifacts.load(tmp_path)


def test_lut_hash_mismatch_is_refused(staged):
    data = json.loads((staged / "params.json").read_text())
    data["lut_sha1"] = "0" * 40
    (staged / "params.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactsError, match="sha1"):
        Artifacts.load(staged)


def test_packaged_default_loads_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    art = Artifacts.default()
    assert art.lut.size >= 2
    assert art.lut_sha1 == sha1_hex(art.lut)


def test_the_default_artifacts_path_still_exists(tmp_path, monkeypatch):
    """`path` is provenance: it must name a directory that is still there.

    Resolving the packaged data as its own package yields a MultiplexedPath,
    which `as_file` materialises into a temp directory deleted before the call
    returns — so `path` pointed at nothing and every startup copied 950 KB.
    """
    monkeypatch.chdir(tmp_path)
    art = Artifacts.default()
    assert art.path.is_dir(), f"{art.path} does not exist after default() returned"
    assert (art.path / "params.json").is_file()
    assert Artifacts.default().path == art.path


@pytest.mark.parametrize(
    "lut_file, message",
    [
        (5, "'lut_file' must be a non-empty string"),
        (None, "'lut_file' must be a non-empty string"),
        ("", "'lut_file' must be a non-empty string"),
        ("../escape.cube", "must be a plain name"),
        ("/etc/passwd.cube", "must be a plain name"),
    ],
)
def test_lut_file_is_validated(tmp_path, lut_file, message):
    """A bad lut_file must not escape as TypeError, nor read outside the directory."""
    (tmp_path / "params.json").write_text(json.dumps({"version": 2, "lut_file": lut_file}))
    with pytest.raises(ArtifactsError, match=re.escape(message)):
        Artifacts.load(tmp_path)


@pytest.mark.parametrize("version", [True, 0, -1, PARAMS_VERSION + 1])
def test_bad_version_values_are_refused(tmp_path, version):
    """`True` is an int subclass, so isinstance would have let it through."""
    (tmp_path / "params.json").write_text(json.dumps({"version": version}))
    with pytest.raises(ArtifactsError, match=re.escape("unsupported params version")):
        Artifacts.load(tmp_path)


def test_resolve_prefers_the_override(staged, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Artifacts.resolve(staged).lut.size == 9
    assert Artifacts.resolve(None).path != staged


def test_publish_moves_only_after_validation(staged, tmp_path):
    dest = tmp_path / "live"
    published = publish(staged, dest)
    assert published == dest
    assert (dest / "pifilm.cube").exists() and (dest / "params.json").exists()
    assert Artifacts.load(dest).lut.size == 9
    assert not staged.exists()


def test_publish_refuses_an_invalid_staging_dir_and_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    before = (dest / "params.json").read_text()

    bad = tmp_path / "bad"
    bad.mkdir()
    write_cube(LUT3D.identity(9), bad / "pifilm.cube")
    (bad / "params.json").write_text(json.dumps({"version": 2, "lut_file": "pifilm.cube",
                                                 "lut_sha1": "0" * 40}))
    with pytest.raises(ArtifactsError):
        publish(bad, dest)
    assert (dest / "params.json").read_text() == before
    assert Artifacts.load(dest).lut.size == 5


def test_publish_replaces_an_existing_artifact(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    staging = tmp_path / "new"
    write_artifact(staging, LUT3D.identity(9), NormalizeParams(), GrainParams())
    publish(staging, dest)
    assert Artifacts.load(dest).lut.size == 9


def test_a_missing_lut_sha1_is_refused(tmp_path):
    """Omitting the field must not silently disable the integrity check.

    Demonstrated before this test existed: deleting `lut_sha1` and swapping in
    a completely different table loaded without complaint.
    """
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    params = tmp_path / "params.json"
    data = json.loads(params.read_text())
    del data["lut_sha1"]
    params.write_text(json.dumps(data))
    write_cube(LUT3D(np.clip(LUT3D.identity(9).table**1.7, 0, 1)), tmp_path / "pifilm.cube")

    # Same hazard as above: this test's name contains "lut_sha1", so it appears
    # in tmp_path. Match the sentence, not the token.
    with pytest.raises(ArtifactsError, match=re.escape("'lut_sha1' is required")):
        Artifacts.load(tmp_path)


def test_a_field_level_type_error_is_wrapped(tmp_path):
    """A bad value inside a well-formed section must not escape as TypeError."""
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    params = tmp_path / "params.json"
    data = json.loads(params.read_text())
    data["normalize"]["wb_gain_min"] = "oops"
    params.write_text(json.dumps(data))

    with pytest.raises(ArtifactsError):
        Artifacts.load(tmp_path)


def test_a_malformed_training_section_is_named(tmp_path):
    """Reported as a bad training section, not as a missing LUT file."""
    (tmp_path / "params.json").write_text(
        json.dumps({"version": 2, "training": "not-an-object"})
    )
    # Match the reason, not the word. `tmp_path` is named after the test
    # function, so a bare match="training" also matches the directory in a
    # "LUT file not found" message and passes for the wrong reason.
    with pytest.raises(ArtifactsError, match=re.escape("'training' must be a JSON object")):
        Artifacts.load(tmp_path)


def test_publish_reports_failure_as_an_artifacts_error(tmp_path, monkeypatch):
    """Every failure out of this module is an ArtifactsError naming the paths."""
    staging = tmp_path / "staging"
    write_artifact(staging, LUT3D.identity(9), NormalizeParams(), GrainParams())
    dest = tmp_path / "live"

    real_replace = os.replace

    def failing_replace(src, dst):
        if Path(src) == staging:
            raise OSError("simulated failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ArtifactsError, match="could not publish"):
        publish(staging, dest)


def test_a_non_identity_artifact_survives_a_write_load_cycle(tmp_path):
    """The integrity check must not fire on an artifact this code just wrote.

    Every other test here uses an identity LUT, whose values are exact at the
    six decimals a .cube stores. A fitted LUT is not: it loses about 5e-7 per
    value on the way to disk. If the recorded hash came from the in-memory
    table rather than the persisted file, this test would fail while all the
    identity-based ones passed — and every trained artifact would be
    unloadable.
    """
    lut = LUT3D(np.clip(LUT3D.identity(17).table ** 1.4, 0.0, 1.0))
    write_artifact(tmp_path, lut, NormalizeParams(), GrainParams())

    art = Artifacts.load(tmp_path)  # raises ArtifactsError on a hash mismatch
    assert art.lut.size == 17
    assert np.abs(art.lut.table - lut.table).max() < 1e-5
    assert art.lut_sha1 == json.loads((tmp_path / "params.json").read_text())["lut_sha1"]


def test_publish_accepts_a_non_identity_artifact(tmp_path):
    """The same hazard, reached through the trainer's actual path."""
    staging = tmp_path / "staging"
    write_artifact(
        staging,
        LUT3D(np.clip(LUT3D.identity(9).table ** 0.8, 0.0, 1.0)),
        NormalizeParams(),
        GrainParams(),
    )
    dest = publish(staging, tmp_path / "live")
    assert Artifacts.load(dest).lut.size == 9


def test_committed_default_is_loadable(repo_root):
    art = Artifacts.load(repo_root / "pifilm" / "data")
    assert art.lut.size in (2, 9, 33)
    assert np.isfinite(art.lut.table).all()


def test_two_scale_grain_writes_version_3_and_loads(tmp_path):
    """Older builds read grain they do not know as the Gaussian model, six times too
    strong; version 3 makes them refuse the artifact instead."""
    grain = GrainParams(model="two_scale", strength=0.025)
    write_artifact(tmp_path / "a", LUT3D.identity(9), NormalizeParams(), grain)
    assert json.loads((tmp_path / "a" / "params.json").read_text())["version"] == 3
    assert Artifacts.load(tmp_path / "a").grain == grain


def test_two_scale_grain_in_a_version_2_file_is_refused(tmp_path):
    write_artifact(tmp_path / "a", LUT3D.identity(9), NormalizeParams(),
                   GrainParams(model="two_scale"))
    p = tmp_path / "a" / "params.json"
    data = json.loads(p.read_text())
    data["version"] = 2
    p.write_text(json.dumps(data))
    with pytest.raises(ArtifactsError, match="version 3"):
        Artifacts.load(tmp_path / "a")
