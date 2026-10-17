import re

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.camera import synthetic_frame
from pifilm.gpu.grader import GpuError, GpuGrader
from pifilm.gpu.shader import BACKEND
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D, sha1_hex
from pifilm.normalize import NormalizeParams, stats_gains
from pifilm.pipeline import Pipeline
from tests.gpu_fakes import FakeDevice, tile_artifacts


@pytest.fixture
def pipeline(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(tmp_path))


def test_process_shapes_and_info(pipeline):
    frame = np.random.default_rng(0).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out, info = pipeline.process(frame, rng=np.random.default_rng(0))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert set(info) == {"wb_gains", "exposure_gain", "levels", "clamped", "lut_sha1",
                         "normalize_sha1", "ev_comp"}
    assert info["ev_comp"] == 0.0
    assert len(info["wb_gains"]) == 3
    assert info["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    assert set(info["clamped"]) == {"wb", "exposure"}


def test_normalize_sha1_identifies_the_normalisation(tmp_path):
    """The LUT hash alone no longer pins the grade: normalisation is per artifact.

    A capture record has to say which normalisation produced it, or two records
    with the same ``lut_sha1`` and different ``levels_lift_highlight_ref``
    cannot be told apart (see docs/known-issues.md).
    """
    def _pipeline(sub, params):
        write_artifact(tmp_path / sub, LUT3D.identity(9), params, GrainParams())
        return Pipeline(Artifacts.load(tmp_path / sub))

    frame = np.random.default_rng(3).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    base = _pipeline("a", NormalizeParams())
    same = _pipeline("b", NormalizeParams())
    damped = _pipeline("c", NormalizeParams(levels_lift_highlight_ref=0.02))

    digest = base.process(frame, grain=False)[1]["normalize_sha1"]
    assert re.fullmatch(r"[0-9a-f]{40}", digest)
    assert same.process(frame, grain=False)[1]["normalize_sha1"] == digest
    assert damped.process(frame, grain=False)[1]["normalize_sha1"] != digest


def test_grain_can_be_skipped(pipeline):
    frame = np.full((36, 64, 3), 128, dtype=np.uint8)
    no_grain, _ = pipeline.process(frame, grain=False)
    with_grain, _ = pipeline.process(frame, grain=True, rng=np.random.default_rng(0))
    assert no_grain.std() < with_grain.std()


def test_same_seed_reproduces_the_output(pipeline):
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    a, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    b, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_process_returns_a_writeable_array_either_way(pipeline):
    """Writability must not depend on whether grain happened to run."""
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    for grain in (False, True):
        out, _ = pipeline.process(frame, grain=grain, rng=np.random.default_rng(0))
        assert out.flags.writeable, f"grain={grain} returned a read-only array"
        out[0, 0, 0] = 7


def test_process_rejects_wrong_input(pipeline):
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4), dtype=np.uint8))


def test_disabled_grain_in_artifact_wins(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams(enabled=False))
    pipe = Pipeline(Artifacts.load(tmp_path))
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    out, _ = pipe.process(frame, grain=True, rng=np.random.default_rng(0))
    assert out.std() < 1.0


def _mean_linear(rgb):
    from pifilm.color import luminance, srgb_to_linear
    return float(luminance(srgb_to_linear(rgb.astype(np.float32) / 255.0)).mean())


def _scene(seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(10, 200, (48, 64, 3), dtype=np.uint8)


@pytest.fixture
def starter():
    """The deployed look: levels normalisation with its clamps, and a real LUT."""
    return Pipeline(Artifacts.default())


def _exposed(frame, ev):
    """The frame the camera would deliver at ``ev``: a linear-light gain."""
    from pifilm.color import linear_to_srgb, srgb_to_linear
    lin = np.clip(srgb_to_linear(frame / 255.0) * 2.0 ** ev, 0.0, 1.0)
    return np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("ev", [-2.0, -1.0, -1 / 3])
def test_a_compensated_exposure_grades_that_many_stops_from_the_plain_one(starter, ev):
    """Normalisation brings every frame to the same exposure, which on the device
    (2026-09-24) turned a -2 EV shot into the same graded file as a 0 EV one.
    Before the LUT the compensated frame must sit ``ev`` stops from the plain
    one; the frame exposed at ``ev`` and graded with ``ev`` shows it."""
    from pifilm.normalize import normalize_u8

    frame = _scene()
    starter.process(frame, grain=False)  # warm path; the LUT is checked below
    params = starter.artifacts.normalize
    plain = _mean_linear(normalize_u8(frame, params)[0])
    captured, info = _exposed(frame, ev), None
    # The pre-LUT image: grade with an identity check by undoing nothing else.
    from pifilm.pipeline import _ev_table
    pre_lut = _ev_table(ev)[normalize_u8(_ev_table(-ev)[captured], params)[0]]
    assert np.log2(_mean_linear(pre_lut) / plain) == pytest.approx(ev, abs=0.15)
    _, info = starter.process(captured, grain=False, ev=ev)
    assert info["ev_comp"] == ev


@pytest.mark.parametrize("ev", [-2.0, -1.0])
def test_minus_ev_grades_darker_through_the_real_look(starter, ev):
    frame = _scene()
    plain = _mean_linear(starter.process(frame, grain=False)[0])
    darker = _mean_linear(starter.process(_exposed(frame, ev), grain=False, ev=ev)[0])
    # The LUT's tone curve compresses the difference a little; it must not erase it.
    assert np.log2(darker / plain) == pytest.approx(ev, abs=0.5)


def test_plus_ev_grades_brighter(starter):
    frame = _scene()
    plain = _mean_linear(starter.process(frame, grain=False)[0])
    brighter = _mean_linear(starter.process(_exposed(frame, 1.0), grain=False, ev=1.0)[0])
    assert brighter > plain * 1.3


def test_ev_zero_is_the_unchanged_grade(pipeline):
    frame = _scene()
    a = pipeline.process(frame, grain=False)[0]
    b = pipeline.process(frame, grain=False, ev=0.0)[0]
    assert np.array_equal(a, b)


def test_process_reports_how_long_each_stage_took(pipeline):
    """Step 1 of the pipeline roadmap: decisions about speed rest on per-stage
    numbers from the Pi, so the pipeline has to hand them over."""
    rgb = np.full((24, 32, 3), 120, np.uint8)
    timings = {}
    _, info = pipeline.process(rgb, rng=np.random.default_rng(0), timings=timings)
    assert set(timings) == {"normalize", "lut", "grain"}
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())
    # The audit fields are what a regrade is reproduced from: timings stay out.
    assert not any("ms" in key or key == "timings" for key in info)


def test_a_skipped_grain_stage_is_not_timed(pipeline):
    timings = {}
    pipeline.process(np.full((24, 32, 3), 120, np.uint8), grain=False, timings=timings)
    assert set(timings) == {"normalize", "lut"}


def test_timings_do_not_change_the_pixels(pipeline):
    rgb = np.random.default_rng(3).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    plain, _ = pipeline.process(rgb, rng=np.random.default_rng(1))
    timed, _ = pipeline.process(rgb, rng=np.random.default_rng(1), timings={})
    assert np.array_equal(plain, timed)


# --- the GPU grader ----------------------------------------------------------


class _Grader:
    """A grader as ``Pipeline.process`` sees one: no GL, a fixed picture."""

    def __init__(self, *, can=True, fail=False):
        self.can, self.fail = can, fail
        self.asked, self.graded = [], []

    def can_grade(self, shape, *, ev, grain):
        self.asked.append((tuple(shape), ev, grain))
        return self.can

    def grade(self, rgb_u8, gains, rng, *, grain=True, timings=None):
        self.graded.append((gains, grain))
        if grain and rng is not None:
            rng.integers(0, 2048, 2)  # the placement draw, as the real grader makes it
            rng.integers(0, 8)
        if timings is not None:
            timings.update(gpu_upload=1.0, gpu_shader=2.0, gpu_readback=3.0)
        if self.fail:
            raise GpuError("GL error after grading: 0x505")
        return np.full((*rgb_u8.shape[:2], 4), 77, np.uint8)


def test_a_grader_that_can_grade_supplies_the_picture(pipeline):
    frame = _scene()
    _, cpu_info = pipeline.process(frame, rng=np.random.default_rng(4))
    grader, used, timings = _Grader(), {}, {}
    out, info = pipeline.process(frame, rng=np.random.default_rng(4), timings=timings,
                                 grader=grader, used=used)
    assert out.shape == (48, 64, 4) and out.dtype == np.uint8 and (out == 77).all()
    assert used == {"backend": BACKEND} and BACKEND == "gpu-2"
    assert info == cpu_info
    assert set(timings) == {"normalize", "gpu", "gpu_upload", "gpu_shader", "gpu_readback"}
    assert grader.asked == [((48, 64, 3), 0.0, True)]
    ((gains, grain),) = grader.graded
    assert grain is True
    want = stats_gains(frame, pipeline.artifacts.normalize)
    assert np.array_equal(gains.wb, want.wb) and gains.exposure == want.exposure
    assert gains.levels == want.levels and gains.clamped == want.clamped


def test_the_grader_is_told_whether_grain_will_run(pipeline, tmp_path):
    grader = _Grader()
    pipeline.process(_scene(), grain=False, grader=grader)
    assert grader.asked[-1][2] is False and grader.graded[-1][1] is False
    write_artifact(tmp_path / "off", LUT3D.identity(9), NormalizeParams(),
                   GrainParams(enabled=False))
    Pipeline(Artifacts.load(tmp_path / "off")).process(_scene(), grader=grader)
    assert grader.asked[-1][2] is False and grader.graded[-1][1] is False


@pytest.mark.parametrize("ev", [0.0, -1.0])
def test_a_grader_that_cannot_grade_changes_nothing(starter, ev):
    frame = _scene()
    plain, plain_info = starter.process(frame, rng=np.random.default_rng(9), ev=ev)
    grader, used, timings = _Grader(can=False), {}, {}
    out, info = starter.process(frame, rng=np.random.default_rng(9), ev=ev, timings=timings,
                                grader=grader, used=used)
    assert out.shape == frame.shape and out.tobytes() == plain.tobytes()
    assert info == plain_info
    assert used == {"backend": "cpu"}
    assert grader.graded == []
    assert set(timings) == {"normalize", "lut", "grain"}


def test_a_failed_gpu_grade_is_the_cpu_grade_of_the_same_seed(starter):
    """The grader draws the grain's placement before the device fails; the CPU
    must then draw the same numbers, or the photograph's seed no longer
    reproduces its grain."""
    frame = _scene()
    plain, plain_info = starter.process(frame, rng=np.random.default_rng(21))
    grader, used, timings = _Grader(fail=True), {}, {}
    out, info = starter.process(frame, rng=np.random.default_rng(21), timings=timings,
                                grader=grader, used=used)
    assert len(grader.graded) == 1
    assert out.tobytes() == plain.tobytes() and info == plain_info
    assert used["backend"] == "cpu"
    assert "0x505" in used["gpu_error"]
    # What the failed attempt timed is not this photograph's grade.
    assert set(timings) == {"normalize", "lut", "grain"}


def test_a_failed_gpu_grade_leaves_the_generator_where_the_cpu_does(starter):
    frame = _scene()
    cpu, gpu = np.random.default_rng(5), np.random.default_rng(5)
    starter.process(frame, rng=cpu)
    starter.process(frame, rng=gpu, grader=_Grader(fail=True))
    assert cpu.bit_generator.state == gpu.bit_generator.state


def test_a_real_grader_on_a_fake_device_grades_within_the_pass_tolerance():
    """The whole chain without GL: the real grader, a device built on the
    reference, and the same seed placing the grain on both paths."""
    pipe = Pipeline(tile_artifacts())
    frame = synthetic_frame(96, 160)
    cpu, cpu_info = pipe.process(frame, rng=np.random.default_rng(3))
    device = FakeDevice()
    grader = GpuGrader(pipe.artifacts, device, log=lambda _: None)
    used = {}
    out, info = pipe.process(frame, rng=np.random.default_rng(3), grader=grader, used=used)
    assert used == {"backend": BACKEND} and info == cpu_info
    assert out.shape == (96, 160, 4)
    diff = np.abs(out[..., :3].astype(int) - cpu.astype(int))
    assert diff.mean() < 1.25 and diff.max() <= 24  # the reference's measured bounds on luma
    # And when the device fails, the CPU's own bytes.
    device.fail_next = 1
    used = {}
    out, _ = pipe.process(frame, rng=np.random.default_rng(3), grader=grader, used=used)
    assert out.tobytes() == cpu.tobytes() and used["backend"] == "cpu"


def test_used_is_filled_without_a_grader(pipeline):
    used = {}
    pipeline.process(_scene(), grain=False, used=used)
    assert used == {"backend": "cpu"}
