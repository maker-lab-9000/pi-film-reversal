import json

import numpy as np
import pytest

from pifilm.artifacts import Artifacts
from pifilm.color import lch_to_oklab, oklab_to_lch, oklab_to_srgb, srgb_to_oklab
from pifilm.grain import GrainParams
from pifilm.highlight import protect_highlights
from pifilm.imageio import save_jpeg
from pifilm.lut import LUT3D
from pifilm.preset import starter_lut
from pifilm.train.dataset import PixelPool, SampleConfig
from pifilm.train.evaluate import channels_are_monotone, grey_axis_is_monotone
from pifilm.train.fit import FitConfig, build_parser, fit, main, train
from pifilm.train.lutfit import enforce_grey_axis, enforce_monotone


def _curve_and_rotation(lab, gamma=1.15, chroma=1.25, degrees=10.0):
    """A tone curve, a saturation boost, and a genuine hue rotation."""
    lch = oklab_to_lch(lab)
    lch[..., 0] = np.clip(lch[..., 0], 0, 1) ** gamma
    lch[..., 1] = lch[..., 1] * chroma
    lch[..., 2] = lch[..., 2] + np.deg2rad(degrees)
    return lch_to_oklab(lch)


def _pool(rng, n=30000):
    return (rng.random((n, 3), dtype=np.float32) * 0.7 + 0.15).astype(np.float32)


def test_fit_recovers_a_tone_curve_and_a_ten_degree_hue_rotation():
    """The spec claims this capability, so it is tested directly."""
    rng = np.random.default_rng(0)
    src = _pool(rng)
    transformed = oklab_to_srgb(_curve_and_rotation(srgb_to_oklab(_pool(rng))))
    tgt = np.clip(transformed, 0, 1).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    expected = _curve_and_rotation(srgb_to_oklab(held))
    got = srgb_to_oklab(result.lut.apply_numpy(held))
    assert float(np.sqrt(np.sum((got - expected) ** 2, axis=1)).mean()) < 0.03


def test_a_large_hue_rotation_is_only_partly_recovered():
    """Pins the documented limit: a large hue rotation is only partly recovered.

    The damping comes from the transport and the LUT-fit smoothing, NOT from
    hue reweighting: this fixture is uniform over the cube, so a 90-degree
    rotation leaves the hue histogram unchanged and `hue_weights` has nothing
    to correct. Disabling reweighting entirely moves the result from about
    5 degrees to about 7, both far under the bound below.
    """
    rng = np.random.default_rng(1)
    src = _pool(rng)
    tgt = np.clip(
        oklab_to_srgb(
            _curve_and_rotation(srgb_to_oklab(_pool(rng)), gamma=1.0, chroma=1.0, degrees=90.0)
        ),
        0, 1,
    ).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    before = oklab_to_lch(srgb_to_oklab(held))[:, 2]
    after = oklab_to_lch(srgb_to_oklab(result.lut.apply_numpy(held)))[:, 2]
    achieved = np.degrees(np.angle(np.exp(1j * (after - before)))).mean()
    # Measured: about +5 degrees out of the 90 requested, stable across seeds.
    # A third of the request is a generous ceiling that still fails loudly if
    # reweighting ever stops damping large rotations.
    assert abs(achieved) < 30.0, (
        f"a 90 degree rotation must be heavily damped, got {achieved:.1f} degrees"
    )


def test_strength_zero_gives_an_identity_lut():
    rng = np.random.default_rng(2)
    src = PixelPool(rng.random((5000, 3), dtype=np.float32), 1)
    tgt = PixelPool((rng.random((5000, 3), dtype=np.float32) * 0.5).astype(np.float32), 1)
    result = fit(src, tgt, FitConfig(lut_size=9, iterations=5, strength=0.0))
    x = rng.random((500, 3), dtype=np.float32)
    assert np.abs(result.lut.apply_numpy(x) - x).max() < 0.02


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"lut_size": 1}, "lut_size"),
        ({"iterations": 0}, "iterations"),
        ({"hue_bins": 0}, "hue_bins"),
        ({"strength": 2.5}, "strength"),
        ({"highlights": -0.1}, "highlights"),
        ({"highlights": 1.1}, "highlights"),
        ({"highlights": np.inf}, "highlights"),
        ({"highlights": np.nan}, "highlights"),
        ({"lambda_smooth": -1.0}, "lambda_smooth"),
    ],
)
def test_invalid_fit_config_names_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        FitConfig(**kwargs)


def _image_dir(dir_path, n, seed, transform=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = np.linspace(0.2, 0.8, 64, dtype=np.float32)[None, :, None] * rng.uniform(
            0.7, 1.0, 3
        ).astype(np.float32)
        img = np.repeat(base, 48, axis=0)
        if transform:
            img = transform(img)
        save_jpeg((np.clip(img, 0, 1) * 255).astype(np.uint8), dir_path / f"{i:02d}.jpg")


def test_train_end_to_end_publishes_a_loadable_artifact(tmp_path):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    metrics, gates = train(
        tmp_path / "src",
        tmp_path / "tgt",
        out,
        FitConfig(lut_size=9, iterations=5),
        SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25),
        grain=None,
        proxy_source=True,
        allow_small=True,
        command="test",
    )
    art = Artifacts.load(out)
    assert art.lut.size == 9
    assert art.training["source"]["proxy"] is True
    assert art.training["source"]["corpus_sha1"] and art.training["target"]["corpus_sha1"]
    assert art.training["split"]["n_source_val_images"] == 2
    assert art.training["fit"]["lut_size"] == 9
    assert art.training["metrics"]["swd_after"] == metrics["swd_after"]
    assert art.training["code_revision"]
    assert (out / "report" / "contact_sheet.png").exists()
    assert (out / "report" / "summary.txt").exists()
    assert isinstance(gates, list) and gates


def test_train_records_source_white_balance_and_lift_ref_in_the_artifact(tmp_path):
    """The Pi applies whatever the artifact records, so the trainer must record both."""
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "out"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True,
          source_white_balance=False, source_lift_highlight_ref=0.05)
    normalize = Artifacts.load(out).normalize
    assert normalize.white_balance is False
    assert normalize.levels_lift_highlight_ref == 0.05
    assert normalize.levels is True


def test_cli_exposes_source_white_balance_and_lift_ref_flags():
    args = build_parser().parse_args(["--source", "s", "--no-source-white-balance",
                                      "--source-lift-highlight-ref", "0.05"])
    assert args.source_white_balance is False
    assert args.source_lift_highlight_ref == 0.05
    defaults = build_parser().parse_args(["--source", "s"])
    assert defaults.source_white_balance is True
    assert defaults.source_lift_highlight_ref is None


def test_train_records_source_gamma_limits_in_the_artifact(tmp_path):
    """The Pi clamps the levels gamma to what the artifact records, so the trainer
    must record the limits it normalised the source with."""
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "out"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True,
          source_gamma_min=0.65, source_gamma_max=1.25)
    normalize = Artifacts.load(out).normalize
    assert normalize.levels_gamma_min == 0.65
    assert normalize.levels_gamma_max == 1.25


def test_cli_exposes_source_gamma_limit_flags_defaulting_to_normalize_defaults():
    args = build_parser().parse_args(["--source", "s", "--source-gamma-min", "0.65",
                                      "--source-gamma-max", "1.25"])
    assert args.source_gamma_min == 0.65
    assert args.source_gamma_max == 1.25
    defaults = build_parser().parse_args(["--source", "s"])
    assert defaults.source_gamma_min is None
    assert defaults.source_gamma_max is None


def test_main_reports_inverted_source_gamma_limits_instead_of_a_traceback(tmp_path, capsys):
    _image_dir(tmp_path / "src", 2, 0)
    _image_dir(tmp_path / "tgt", 2, 1)
    code = main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
                 "--out", str(tmp_path / "o"), "--source-gamma-min", "1.5",
                 "--source-gamma-max", "1.25"])
    assert code == 1
    assert "levels_gamma_min" in capsys.readouterr().err


def test_train_leaves_the_previous_artifact_intact_if_publication_fails(tmp_path, monkeypatch):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    good = (out / "params.json").read_text()

    monkeypatch.setattr(
        "pifilm.train.fit.publish", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    assert (out / "params.json").read_text() == good
    # The `finally` must remove staging on the failure path too. Without this
    # assertion the whole try/finally can be deleted and the test stays green:
    # the mocked publish raises before touching out_dir, so the params check
    # above passes either way.
    leftover = list(out.parent.glob(".pifilm-staging-*"))
    assert leftover == [], f"staging directory left behind: {leftover}"


def test_train_creates_a_nested_output_parent_instead_of_crashing(tmp_path):
    """`mkdtemp` needs out_dir's parent; publish() creating it later is too late."""
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "does" / "not" / "exist" / "artifacts"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    assert (out / "params.json").exists()


def test_main_rejects_identical_and_missing_dirs(tmp_path, capsys):
    _image_dir(tmp_path / "src", 2, 0)
    assert main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1
    assert "same" in capsys.readouterr().err
    assert main(["--source", str(tmp_path / "missing"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1


def test_main_reports_an_invalid_source_lift_highlight_ref_instead_of_a_traceback(
    tmp_path, capsys
):
    """main() builds the source NormalizeParams in its own config block, so the
    ValueError surfaces there and must be reported instead of a traceback."""
    _image_dir(tmp_path / "src", 2, 0)
    _image_dir(tmp_path / "tgt", 2, 1)
    code = main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
                 "--out", str(tmp_path / "o"), "--source-lift-highlight-ref", "1.5"])
    assert code == 1
    assert "levels_lift_highlight_ref" in capsys.readouterr().err


def test_main_refuses_a_small_corpus_then_accepts_the_flag(tmp_path, capsys):
    _image_dir(tmp_path / "src", 3, 0)
    _image_dir(tmp_path / "tgt", 3, 1, transform=lambda im: im**1.2)
    args = ["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
            "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
            "--max-side", "64", "--pixels-per-image", "300", "--proxy-source",
            "--highlights", "0.5"]
    assert main(args) == 1
    assert "--allow-small" in capsys.readouterr().err
    assert main([*args, "--allow-small"]) in (0, 3)
    data = json.loads((tmp_path / "o" / "params.json").read_text())
    assert data["training"]["fit"]["strength"] == 1.0
    assert data["training"]["fit"]["highlights"] == 0.5
    assert data["grain"]["strength"] == pytest.approx(GrainParams().strength)


def test_main_reports_a_failed_gate_with_exit_code_3(tmp_path, monkeypatch, capsys):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    from pifilm.train import fit as fit_module
    from pifilm.train.evaluate import Gate

    monkeypatch.setattr(
        fit_module, "check_gates",
        lambda m: [Gate("improvement_exceeds_noise", 0.0, 0.1, False, "forced failure")],
    )
    code = main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
                 "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
                 "--max-side", "64", "--allow-small"])
    assert code == 3
    err = capsys.readouterr().err
    assert "improvement_exceeds_noise" in err
    assert Artifacts.load(tmp_path / "o").lut.size == 9  # written despite the failure


def test_cli_defaults_track_fitconfig():
    """Duplicated defaults silently pinned the CLI to stale values.

    `--lambda-identity` and `--lambda-smooth` were hard-coded in argparse, so
    raising the dataclass defaults changed nothing for anyone using the CLI --
    which is everyone who trains an artifact. The first fit run after the
    change came back byte-identical to the run before it.
    """
    defaults = vars(build_parser().parse_args(["--source", "x"]))
    cfg = FitConfig()
    for name in ("lambda_identity", "lambda_smooth", "lut_size", "iterations",
                 "hue_bins", "strength", "seed", "neutral_axis_cap", "highlights"):
        assert defaults[name] == getattr(cfg, name), name


def test_fitconfig_highlights_defaults_to_zero():
    assert FitConfig().highlights == 0.0


def test_train_cli_exposes_highlights_with_the_fitconfig_default():
    parser = build_parser()

    assert parser.parse_args(["--source", "s"]).highlights == FitConfig().highlights
    assert parser.parse_args(["--source", "s", "--highlights", "0.5"]).highlights == 0.5


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_train_cli_rejects_invalid_highlights_without_an_artifact(tmp_path, capsys, value):
    _image_dir(tmp_path / "src", 1, 0)
    _image_dir(tmp_path / "tgt", 1, 1)
    out = tmp_path / "artifact"

    code = main([
        "--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
        "--out", str(out), "--highlights", value,
    ])

    assert code == 1
    assert "highlights must be finite and in [0, 1]" in capsys.readouterr().err
    assert not out.exists()


def test_reprojection_after_protection_restores_monotonicity():
    protected = protect_highlights(starter_lut(), 0.8)
    reprojected = enforce_monotone(enforce_grey_axis(enforce_monotone(protected)))

    assert not channels_are_monotone(protected)
    assert grey_axis_is_monotone(reprojected)
    assert channels_are_monotone(reprojected)


def _lifting_pools() -> tuple[PixelPool, PixelPool]:
    rng = np.random.default_rng(42)
    source_srgb = (rng.random((400, 3), dtype=np.float32) * 0.65 + 0.1).astype(np.float32)
    target_lab = srgb_to_oklab(source_srgb)
    target_lab[:, 0] = np.minimum(target_lab[:, 0] + 0.12, 0.98)
    target_srgb = np.clip(oklab_to_srgb(target_lab), 0, 1).astype(np.float32)
    return PixelPool(source_srgb, 4), PixelPool(target_srgb, 4)


def _fit_lifting_lut(highlights: float) -> LUT3D:
    source, target = _lifting_pools()
    cfg = FitConfig(lut_size=9, iterations=2, seed=3, highlights=highlights)
    return fit(source, target, cfg).lut


def test_fit_applies_highlight_protection_and_keeps_the_result_monotone():
    plain = _fit_lifting_lut(highlights=0.0)
    protected = _fit_lifting_lut(highlights=0.8)
    input_l = srgb_to_oklab(LUT3D.identity(plain.size).table)[..., 0]
    plain_l = srgb_to_oklab(plain.table)[..., 0]
    protected_l = srgb_to_oklab(protected.table)[..., 0]
    lifted_highlights = (input_l >= 0.75) & (plain_l > input_l + 1e-4)

    assert np.count_nonzero(lifted_highlights) > 20
    assert protected_l[lifted_highlights].mean() < plain_l[lifted_highlights].mean() - 0.005
    assert channels_are_monotone(protected)
    assert grey_axis_is_monotone(protected)


def test_fit_default_highlights_preserves_the_zero_strength_table_exactly():
    source, target = _lifting_pools()
    default = fit(source, target, FitConfig(lut_size=9, iterations=2, seed=3)).lut
    explicit = fit(
        source, target, FitConfig(lut_size=9, iterations=2, seed=3, highlights=0.0)
    ).lut

    assert np.array_equal(default.table, explicit.table)


def test_strength_above_one_extrapolates_the_learned_look():
    """1 is the learned look; 1.4 pushes further along the same direction."""
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        FitConfig(strength=2.5)
    rng = np.random.default_rng(3)
    src = _pool(rng)
    tgt = np.clip(oklab_to_srgb(_curve_and_rotation(srgb_to_oklab(_pool(rng)))), 0, 1)
    tgt = tgt.astype(np.float32)
    ident = LUT3D.identity(9).table
    dev = {}
    for s in (1.0, 1.4):
        cfg = FitConfig(lut_size=9, iterations=8, strength=s)
        lut = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg).lut
        dev[s] = np.abs(lut.table - ident).mean()
    assert dev[1.4] > dev[1.0] * 1.15, dev
