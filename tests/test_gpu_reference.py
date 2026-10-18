"""The NumPy reference of the GPU pass, against the CPU pipeline it replaces.

The shader itself cannot run here, so these tests pin the maths it is written
from: the reference must stay within the measured distance of the CPU grade,
and must place the grain where the CPU does for the same seed.
"""

import dataclasses

import numpy as np
import pytest

from pifilm.artifacts import Artifacts
from pifilm.capture.camera import synthetic_frame
from pifilm.gpu import shader
from pifilm.gpu.reference import PassTables, pass_tables, reference_pass, supports
from pifilm.grain import GrainParams, _oriented, grain_tile, place_tile, tile_placement
from pifilm.halation import HalationParams, apply_cpu, prepare
from pifilm.lut import LUT3D
from pifilm.normalize import Gains, apply_gains_u8, stats_gains
from pifilm.pipeline import Pipeline

TILE_GRAIN = GrainParams(model="two_scale_tile", strength=0.025, tile_px=256)
H, W = 400, 600


def _artifacts(**normalize) -> Artifacts:
    base = Artifacts.default()
    return dataclasses.replace(
        base, grain=TILE_GRAIN, normalize=dataclasses.replace(base.normalize, **normalize)
    )


def _scene() -> np.ndarray:
    """Soft gradients in every channel with a cast, like an ordinary photograph."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 0.5 + 0.45 * np.sin(x / 90.0) * np.cos(y / 70.0)
    g = 0.45 + 0.4 * np.cos(x / 130.0 + 1.0)
    b = 0.4 + 0.35 * np.sin((x + y) / 110.0)
    rgb = np.stack([r, g, b], axis=-1) * (0.35 + 0.65 * (y / H))[..., None]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _extremes() -> np.ndarray:
    """Saturated primaries and secondaries, near-black and near-white bands, a
    hue sweep at full saturation and random texture: the hard cases."""
    frame = np.empty((H, W, 3), np.uint8)
    ramp = np.linspace(0, 255, W).astype(np.uint8)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    band = H // 10
    for k, colour in enumerate(colours):
        # Each primary from black up to full strength.
        frame[k * band:(k + 1) * band] = (
            ramp[:, None].astype(np.float32) * np.array(colour, np.float32) / 255.0
        ).astype(np.uint8)
    frame[6 * band:7 * band] = np.linspace(0, 12, W).astype(np.uint8)[:, None]
    frame[7 * band:8 * band] = np.linspace(243, 255, W).astype(np.uint8)[:, None]
    hue = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    sweep = np.stack([np.cos(hue), np.cos(hue - 2.094), np.cos(hue + 2.094)], axis=-1)
    frame[8 * band:9 * band] = np.clip(128 + 200 * sweep, 0, 255).astype(np.uint8)
    frame[9 * band:] = np.random.default_rng(21).integers(
        0, 256, (H - 9 * band, W, 3), dtype=np.uint8)
    return frame


def _muted() -> np.ndarray:
    """Dark to bright with a little colour: the tones of an ordinary photograph."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = (0.03 + 0.95 * (x / W) ** 1.8) * (0.75 + 0.25 * np.cos(y / 55.0))
    rgb = np.stack([base * (1 + 0.12 * np.sin(x / 70.0 + y / 90.0)),
                    base * (1 + 0.12 * np.cos(x / 110.0)),
                    base * (1 + 0.12 * np.sin(y / 60.0 + 2.0))], axis=-1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


FRAMES = {
    "patches": lambda: synthetic_frame(H, W),
    "scene": _scene,
    "extremes": _extremes,
    "muted": _muted,
}
NORMALISATIONS = {
    "luma": {"levels": True, "levels_tone": "luma"},
    "channel": {"levels": True, "levels_tone": "channel"},
    "no-levels": {"levels": False, "white_balance": True},
}


def _reference(artifacts: Artifacts, frame: np.ndarray, seed: int | None) -> np.ndarray:
    grain = artifacts.grain
    tables = pass_tables(stats_gains(frame, artifacts.normalize), grain)
    if seed is None:
        return reference_pass(frame, tables, artifacts.lut.table, None, None)
    tile = grain_tile(grain, max(frame.shape[:2]) / grain.frame_width_mm)
    placement = tile_placement(grain, np.random.default_rng(seed))
    return reference_pass(frame, tables, artifacts.lut.table, tile, placement)


def _difference(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.abs(got.astype(np.int16) - want.astype(np.int16))


# (mean below, 99.9th percentile at most, maximum at most), in 8-bit levels.
# The first two rows are the design's numbers, from real photographs. The luma
# rows are wider, and measured here (2026-10-13: mean up to 0.82 and 1.00,
# 99.9th percentile 11, maximum 20), because these frames are harder than a
# photograph in two ways the CPU's rounding is sensitive to. Its luma tone step
# goes through a byte YCrCb round trip, which moves a saturated colour by a
# level or two before the LUT, and the starter LUT is steep at the edge of the
# gamut: (0, 231, 0) becomes (1, 233, 0) and grades 18 levels away. And a dark
# frame gets a tone curve several times steeper than 1 near black, applied to a
# Y the CPU has already rounded. The float-path test below shows which side of
# the difference the reference is on.
DESIGN = {False: (0.5, 4, 16), True: (1.0, 5, 16)}
LUMA_HARD = {False: (1.0, 12, 24), True: (1.25, 12, 24)}


def _within(diff: np.ndarray, limits: tuple[float, int, int]) -> None:
    mean, tail, most = limits
    assert diff.mean() < mean
    assert np.percentile(diff, 99.9) <= tail
    assert diff.max() <= most


@pytest.mark.parametrize("grain", [False, True])
@pytest.mark.parametrize("normalisation", ["channel", "no-levels"])
@pytest.mark.parametrize("name", FRAMES)
def test_the_reference_is_the_cpu_grade_within_the_designs_tolerances(
        name, normalisation, grain):
    artifacts = _artifacts(**NORMALISATIONS[normalisation])
    frame = FRAMES[name]()
    cpu, _ = Pipeline(artifacts).process(frame, grain=grain, rng=np.random.default_rng(1234))
    got = _reference(artifacts, frame, 1234 if grain else None)
    assert got.shape == frame.shape and got.dtype == np.uint8
    _within(_difference(got, cpu), DESIGN[grain])


@pytest.mark.parametrize("grain", [False, True])
@pytest.mark.parametrize("name", FRAMES)
def test_with_the_luma_tone_curve_the_reference_is_the_cpu_grade_more_loosely(name, grain):
    artifacts = _artifacts(**NORMALISATIONS["luma"])
    frame = FRAMES[name]()
    cpu, _ = Pipeline(artifacts).process(frame, grain=grain, rng=np.random.default_rng(1234))
    diff = _difference(_reference(artifacts, frame, 1234 if grain else None), cpu)
    _within(diff, LUMA_HARD[grain])
    if name == "muted":
        # Without saturated colours the tails are the design's; only the mean
        # is above it, from the tone curve on a rounded Y.
        _within(diff, (LUMA_HARD[grain][0], *DESIGN[grain][1:]))


@pytest.mark.parametrize("normalisation", NORMALISATIONS)
@pytest.mark.parametrize("name", FRAMES)
def test_the_reference_is_the_float_path_and_the_cpu_is_what_differs(name, normalisation):
    """The trainer's float normalisation and the float LUT never round until the
    end. The reference matches them almost everywhere, and the CPU pipeline is as
    far from them as it is from the reference: the difference is the CPU's
    rounding, not the pass."""
    from pifilm.normalize import apply_gains_float

    artifacts = _artifacts(**NORMALISATIONS[normalisation])
    frame = FRAMES[name]()
    gains = stats_gains(frame, artifacts.normalize)
    normalised = apply_gains_float(frame.astype(np.float32) / 255.0, gains)
    exact = np.rint(np.clip(artifacts.lut.apply_numpy(normalised), 0, 1) * 255).astype(np.uint8)

    diff = _difference(_reference(artifacts, frame, None), exact)
    assert diff.mean() < 0.05
    # The tone curve is a 256-entry table read with linear interpolation, in the
    # shader as here. With a gamma below 1 the curve leaves zero at the black
    # point with an unbounded slope, and within a level of that point the table
    # is up to 8 levels from the curve (the dark "scene" frame: 0.3 % of its
    # pixels, 6 levels after the LUT). Everywhere else the two agree.
    tail, most = (4, 10) if normalisation == "luma" else (1, 1)
    assert np.percentile(diff, 99.9) <= tail
    assert diff.max() <= most

    cpu, _ = Pipeline(artifacts).process(frame, grain=False)
    from_exact = _difference(cpu, exact).mean()
    from_reference = _difference(cpu, _reference(artifacts, frame, None)).mean()
    assert abs(from_exact - from_reference) < 0.05


@pytest.mark.parametrize("name", ["scene", "muted"])
def test_the_seed_places_the_grain_where_the_cpu_places_it(name):
    """The grain each side added to its own clean grade: the same field for the
    same seed, an unrelated one for another seed."""
    artifacts = _artifacts()
    frame = FRAMES[name]()
    pipeline = Pipeline(artifacts)
    clean, _ = pipeline.process(frame, grain=False)
    cpu, _ = pipeline.process(frame, rng=np.random.default_rng(7))
    cpu_grain = (cpu.astype(np.float32) - clean).mean(axis=2).ravel()
    assert cpu_grain.std() > 1.0
    reference_clean = _reference(artifacts, frame, None).astype(np.float32)

    def correlation(seed: int) -> float:
        added = _reference(artifacts, frame, seed).astype(np.float32) - reference_clean
        return float(np.corrcoef(added.mean(axis=2).ravel(), cpu_grain)[0, 1])

    assert correlation(7) > 0.95
    assert abs(correlation(8)) < 0.2


def test_the_grain_step_alone_is_the_cpus_within_two_levels():
    """From the same graded bytes, so only the grain differs: the CPU adds it to
    a Y byte and comes back through byte chroma, the pass adds it to R, G and B."""
    from pifilm.grain import add_grain

    artifacts = _artifacts()
    grain = artifacts.grain
    tables = dataclasses.replace(_identity_tables(), gain=pass_tables(
        Gains(wb=np.ones(3, np.float32), exposure=1.0), grain).gain)
    tile = grain_tile(grain, W / grain.frame_width_mm)
    for name in FRAMES:
        graded, _ = Pipeline(artifacts).process(FRAMES[name](), grain=False)
        cpu = add_grain(graded, grain, np.random.default_rng(5))
        got = reference_pass(graded, tables, LUT3D.identity(33).table, tile,
                             tile_placement(grain, np.random.default_rng(5)))
        diff = _difference(got, cpu)
        assert diff.max() <= 2 and diff.mean() < 0.5


def _identity_tables(gain: float = 0.0) -> PassTables:
    levels = np.arange(256, dtype=np.float32) / 255.0
    return PassTables(chan=np.repeat(levels[:, None], 3, axis=1), tone=levels, use_tone=False,
                      gain=np.full(256, gain, np.float32))


@pytest.mark.parametrize("orientation", range(8))
def test_each_orientation_and_offset_is_the_cpus(orientation):
    """The shader turns the tile by index arithmetic; the CPU by flipping and
    transposing the array. A frame larger than the tile, so both wrap."""
    params = GrainParams(model="two_scale_tile", strength=0.05, tile_px=64)
    shape = (100, 150)
    tile = grain_tile(params, 150 / params.frame_width_mm)
    oy, ox = 37, 5 + orientation
    frame = np.full((*shape, 3), 128, np.uint8)
    got = reference_pass(frame, _identity_tables(gain=0.05), LUT3D.identity(17).table,
                         tile, (oy, ox, orientation))

    noise = place_tile(_oriented(tile, orientation), shape, oy, ox)
    want = np.rint(np.clip(np.float32(128 / 255) + np.float32(0.05) * noise, 0, 1) * 255.0)
    assert (got[..., 0] == got[..., 1]).all() and (got[..., 0] == got[..., 2]).all()
    # An identity LUT interpolates to mid-grey within float32 rounding.
    assert np.abs(got[..., 0].astype(np.int16) - want.astype(np.int16)).max() <= 1
    assert np.mean(got[..., 0] != want) < 0.001
    assert got[..., 0].std() > 5


def test_the_orientations_are_eight_different_placements():
    params = GrainParams(model="two_scale_tile", strength=0.05, tile_px=64)
    tile = grain_tile(params, 150 / params.frame_width_mm)
    frame = np.full((100, 150, 3), 128, np.uint8)
    lut = LUT3D.identity(17).table
    results = {reference_pass(frame, _identity_tables(0.05), lut, tile, (3, 9, k)).tobytes()
               for k in range(8)}
    assert len(results) == 8


def test_without_grain_identity_tables_and_lut_leave_black_and_white_alone():
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[:, 30:] = 255
    out = reference_pass(frame, _identity_tables(), LUT3D.identity(33).table, None, None)
    assert np.array_equal(out, frame)
    # A gain table alone adds nothing: grain needs a tile and a placement.
    out = reference_pass(frame, _identity_tables(0.5), LUT3D.identity(33).table, None, None)
    assert np.array_equal(out, frame)


def test_identity_tables_and_lut_are_the_identity_on_every_byte():
    frame = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = reference_pass(frame, _identity_tables(), LUT3D.identity(33).table, None, None)
    assert np.array_equal(out, frame)


def test_tile_and_placement_come_together():
    frame = np.zeros((8, 8, 3), np.uint8)
    lut = LUT3D.identity(17).table
    tile = np.zeros((64, 64), np.float32)
    with pytest.raises(ValueError, match="placement"):
        reference_pass(frame, _identity_tables(), lut, tile, None)
    with pytest.raises(ValueError, match="placement"):
        reference_pass(frame, _identity_tables(), lut, None, (0, 0, 0))


def test_the_pass_rejects_what_is_not_a_frame():
    lut = LUT3D.identity(17).table
    for bad in (np.zeros((8, 8, 4), np.uint8), np.zeros((8, 8, 3), np.float32)):
        with pytest.raises(ValueError, match="RGB uint8"):
            reference_pass(bad, _identity_tables(), lut, None, None)


def test_bands_do_not_show(monkeypatch):
    """A tall frame is worked in row bands; the result is the same as in one."""
    from pifilm.gpu import reference

    artifacts = _artifacts()
    frame = _scene()
    whole = _reference(artifacts, frame, 5)
    monkeypatch.setattr(reference, "BAND_PIXELS", 7 * W)
    assert np.array_equal(_reference(artifacts, frame, 5), whole)


def test_pass_tables_are_the_normalisation_without_its_rounding():
    from pifilm.normalize import gains_to_luts, tone_lut

    frame = _scene()
    for normalisation in NORMALISATIONS.values():
        params = dataclasses.replace(Artifacts.default().normalize, **normalisation)
        gains = stats_gains(frame, params)
        tables = pass_tables(gains, TILE_GRAIN)
        assert tables.chan.shape == (256, 3) and tables.chan.dtype == np.float32
        assert tables.tone.shape == (256,) and tables.tone.dtype == np.float32
        assert tables.gain.shape == (256,) and tables.gain.dtype == np.float32
        assert np.array_equal(np.rint(tables.chan * 255.0).astype(np.uint8), gains_to_luts(gains).T)
        ylut = tone_lut(gains)
        assert tables.use_tone == (ylut is not None)
        if ylut is None:
            assert np.array_equal(tables.tone, np.arange(256, dtype=np.float32) / 255.0)
        else:
            assert np.array_equal(np.rint(tables.tone * 255.0).astype(np.uint8), ylut)


def test_the_gain_table_is_strength_times_the_envelope():
    gains = Gains(wb=np.ones(3, np.float32), exposure=1.0)
    levels = np.arange(256, dtype=np.float32) / 255.0
    want = 0.025 * np.interp(levels, TILE_GRAIN.envelope_luma, TILE_GRAIN.envelope_rel)
    assert np.allclose(pass_tables(gains, TILE_GRAIN).gain, want, atol=1e-7)
    for off in (dataclasses.replace(TILE_GRAIN, enabled=False),
                dataclasses.replace(TILE_GRAIN, strength=0.0),
                GrainParams(enabled=False)):
        assert not pass_tables(gains, off).gain.any()


@pytest.mark.parametrize("model", ["gaussian", "two_scale"])
def test_pass_tables_refuse_a_grain_the_pass_cannot_make(model):
    gains = Gains(wb=np.ones(3, np.float32), exposure=1.0)
    with pytest.raises(ValueError, match=model):
        pass_tables(gains, GrainParams(model=model, strength=0.02))


def test_supports_the_tile_grain_and_no_grain_only():
    assert supports(TILE_GRAIN)
    assert supports(GrainParams(enabled=False))
    assert supports(GrainParams(model="two_scale", enabled=False, strength=0.02))
    assert supports(GrainParams(model="gaussian", strength=0.0))
    assert supports(GrainParams(model="two_scale", strength=0.0))
    assert not supports(GrainParams(model="gaussian", strength=0.004))
    assert not supports(GrainParams(model="two_scale", strength=0.02))
    assert not supports(Artifacts.default().grain)


def test_the_shader_source_and_its_version():
    for source in (shader.VERTEX, shader.FRAGMENT):
        assert isinstance(source, bytes) and source.startswith(b"#version 300 es\n")
    assert shader.SHADER_VERSION == 2
    assert shader.BACKEND == "gpu-2"
    # The uniforms the device binds, by the names the reference uses.
    for name in (b"img", b"chan", b"tone", b"lut", b"gain", b"tile", b"offset", b"orient",
                 b"use_tone", b"use_grain", b"glow", b"use_halation", b"hr", b"hw"):
        assert b" " + name + b";" in shader.FRAGMENT



# --- halation ------------------------------------------------------------------


def _halation_case(params: HalationParams):
    """A random frame with a bright square, the bundled look without grain, and
    what each path is given: the pass's tables and the glow ``prepare`` made.

    Mid-tones (64 to 191), not the full range: on a full-range random frame the
    starter LUT's steep gamut edge puts the CPU up to 22 levels from the pass on
    a saturated pixel without any halation (19 with it), the documented
    difference of the CPU's rounding, which would hide what this checks."""
    rng = np.random.default_rng(5)
    frame = rng.integers(64, 192, (90, 130, 3), dtype=np.uint8)
    frame[40:50, 60:70] = 255
    art = Artifacts.default()
    gains = stats_gains(frame, art.normalize)
    normalised = apply_gains_u8(frame, gains)
    hp = prepare(normalised, params)
    tables = pass_tables(gains, dataclasses.replace(art.grain, enabled=False))
    return frame, art, normalised, hp, tables


def test_reference_with_halation_matches_the_cpu_step():
    frame, art, normalised, hp, tables = _halation_case(
        HalationParams(enabled=True, strength=0.4))
    want = art.lut.apply_pillow(apply_cpu(normalised, hp), art.lut.to_pillow())[..., :3]
    got = reference_pass(frame, tables, art.lut.table, None, None, halation=hp)
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.mean() < 1.0 and diff.max() <= 12   # the CPU's byte rounding, as without halation
    # And the halation shows: the pass with it is not the pass without it.
    plain = reference_pass(frame, tables, art.lut.table, None, None)
    assert np.abs(got.astype(int) - plain.astype(int)).max() > 5


def test_reference_without_halation_is_unchanged():
    frame, art, _, _, tables = _halation_case(HalationParams(enabled=True, strength=0.4))
    tile = grain_tile(TILE_GRAIN, max(frame.shape[:2]) / TILE_GRAIN.frame_width_mm)
    placement = tile_placement(TILE_GRAIN, np.random.default_rng(3))
    for args in ((None, None), (tile, placement)):
        assert np.array_equal(
            reference_pass(frame, tables, art.lut.table, *args, halation=None),
            reference_pass(frame, tables, art.lut.table, *args))


def test_the_glow_read_is_the_cpus_bilinear_upsampling():
    """The shader reads the glow by hand at half-pixel centres with clamped
    edges; that is what ``cv2.resize(..., INTER_LINEAR)`` does on the CPU."""
    from pifilm._cv2 import require_cv2
    from pifilm.halation import glow_at

    glow = np.random.default_rng(9).random((7, 11, 3)).astype(np.float32)
    want = require_cv2().resize(glow, (70, 45), interpolation=require_cv2().INTER_LINEAR)
    got = np.concatenate([glow_at(glow, top, rows, 70, 45, 70)
                          for top, rows in ((0, 20), (20, 25))], axis=1).transpose(1, 2, 0)
    assert got.dtype == np.float32 and got.shape == want.shape
    assert np.abs(got - want).max() < 1e-3
    # A glow at full size reads back as itself.
    assert np.allclose(glow_at(glow, 0, 7, 11, 7, 11), glow.transpose(2, 0, 1), atol=1e-6)


def test_halation_off_in_the_mix_leaves_a_channel_as_the_pass_without_it():
    """With a zero weight a channel goes through expand and compress only, which
    is the identity to well below a level."""
    frame, art, normalised, _, tables = _halation_case(HalationParams())
    hp = prepare(normalised, HalationParams(enabled=True, weights=(0.0, 0.0, 0.0)))
    identity = LUT3D.identity(33).table
    got = reference_pass(frame, tables, identity, None, None, halation=hp)
    want = reference_pass(frame, tables, identity, None, None)
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


def test_bands_do_not_show_with_halation(monkeypatch):
    from pifilm.gpu import reference

    frame, art, _, hp, tables = _halation_case(HalationParams(enabled=True, strength=0.4))
    whole = reference_pass(frame, tables, art.lut.table, None, None, halation=hp)
    monkeypatch.setattr(reference, "BAND_PIXELS", 7 * frame.shape[1])
    assert np.array_equal(
        reference_pass(frame, tables, art.lut.table, None, None, halation=hp), whole)
