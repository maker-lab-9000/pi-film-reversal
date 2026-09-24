import numpy as np
import pytest

from pifilm.artifacts import Artifacts
from pifilm.color import srgb_to_oklab
from pifilm.highlight import protect_highlights
from pifilm.lut import read_cube, write_cube
from pifilm.preset import main as preset_main
from pifilm.preset import starter_lut, write_starter
from pifilm.train.fit import build_parser


def test_starter_keeps_neutrals_neutral_and_tone_monotonic():
    lut = starter_lut()
    ramp = np.repeat(np.linspace(0, 1, 256)[:, None], 3, axis=1).astype(np.float32)
    graded = lut.apply_numpy(ramp)
    assert np.max(np.ptp(graded, axis=1)) < 0.001
    assert np.all(np.diff(graded, axis=0) >= -1e-6)
    assert np.allclose(graded[0], 0, atol=1e-5)
    assert np.allclose(graded[-1], 1, atol=1e-5)


def test_starter_boosts_midtone_chroma_without_invalid_values():
    lut = starter_lut()
    colours = np.array([[0.65, 0.35, 0.35], [0.35, 0.55, 0.65]], dtype=np.float32)
    before = np.linalg.norm(srgb_to_oklab(colours)[:, 1:], axis=1)
    after = np.linalg.norm(srgb_to_oklab(lut.apply_numpy(colours))[:, 1:], axis=1)
    assert np.all(after > before * 1.2)
    assert np.isfinite(lut.table).all()
    assert lut.table.min() >= 0 and lut.table.max() <= 1


def test_starter_artifact_records_untrained_provenance(tmp_path):
    art = Artifacts.load(write_starter(tmp_path / "starter"))
    assert art.training["trained"] is False
    assert art.training["kind"] == "handcrafted-preset"
    assert art.grain.strength == 0.004


def test_write_starter_applies_highlight_protection_and_records_it(tmp_path):
    path = write_starter(tmp_path / "protected", highlights=0.6)
    art = Artifacts.load(path)
    expected_path = tmp_path / "expected.cube"
    write_cube(protect_highlights(starter_lut(), 0.6), expected_path)

    assert art.training["highlights"] == 0.6
    assert np.array_equal(art.lut.table, read_cube(expected_path).table)


def test_write_starter_default_preserves_the_unprotected_cube_exactly(tmp_path):
    plain_path = write_starter(tmp_path / "plain")
    explicit_path = write_starter(tmp_path / "explicit", highlights=0.0)
    expected_path = tmp_path / "expected.cube"
    write_cube(starter_lut(), expected_path)

    plain = Artifacts.load(plain_path)
    assert plain.training["highlights"] == 0.0
    assert (plain_path / "pifilm.cube").read_bytes() == expected_path.read_bytes()
    assert (plain_path / "pifilm.cube").read_bytes() == (explicit_path / "pifilm.cube").read_bytes()


def test_preset_cli_accepts_highlights(tmp_path):
    path = tmp_path / "cli"
    code = preset_main(["--out", str(path), "--highlights", "0.5"])

    assert code == 0
    assert Artifacts.load(path).training["highlights"] == 0.5


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_preset_cli_rejects_invalid_highlights_without_an_artifact(tmp_path, capsys, value):
    path = tmp_path / "invalid"

    with pytest.raises(SystemExit, match="2"):
        preset_main(["--out", str(path), "--highlights", value])

    assert "highlights must be finite and in [0, 1]" in capsys.readouterr().err
    assert not path.exists()


def test_training_skips_reference_levels_stretch_by_default():
    args = build_parser().parse_args(["--source", "source"])
    assert args.target_levels is False
    assert args.grain_strength == 0.004
    assert str(args.target) == "data/references"


REF = 0.02


def test_starter_normalisation_trusts_the_isp_and_damps_the_lift(tmp_path):
    out = write_starter(tmp_path / "s")
    normalize = Artifacts.load(out).normalize
    assert normalize.white_balance is False
    assert normalize.levels is True
    assert normalize.levels_lift_highlight_ref == REF  # literal chosen value


def test_bundled_params_match_write_starter_and_the_cube_is_unchanged(tmp_path):
    # exp/kodachrome-look: pifilm/data holds the Kodachrome experiment and the
    # starter lives beside it under looks/starter.
    out = write_starter(tmp_path / "s")
    bundled = Artifacts.load(Artifacts.default().path / "looks" / "starter")
    assert Artifacts.load(out).normalize == bundled.normalize
    assert (out / "pifilm.cube").read_bytes() == (bundled.path / "pifilm.cube").read_bytes()
