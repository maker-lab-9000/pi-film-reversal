"""The GPU grader, driven through a fake device built on the NumPy reference.

No graphics library runs here. What is pinned: the grader hands the device
what the reference needs to reproduce the CPU's grain placement, it refuses
what the pass does not cover without touching the device, and every way the
device can fail ends in a ``GpuError`` (so the caller grades on the CPU), a
closed device or a switched-off grader, never a hang or a stray exception.
"""

import dataclasses
import threading

import numpy as np
import pytest

from pifilm.artifacts import Artifacts
from pifilm.gpu import grader as grader_module
from pifilm.gpu.grader import GpuError, GpuGrader, self_test_frame
from pifilm.gpu.reference import pass_tables, reference_pass
from pifilm.grain import GrainParams, grain_tile, tile_placement
from pifilm.normalize import stats_gains
from tests.gpu_fakes import TILE_GRAIN, FakeDevice, tile_artifacts


def _frame(height: int = 64, width: int = 96, seed: int = 5) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    rgb = np.stack([40 + 170 * x / width, 30 + 190 * y / height,
                    220 - 150 * (x + y) / (width + height)], axis=-1)
    rgb += np.random.default_rng(seed).normal(0, 4, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _expected(artifacts: Artifacts, frame: np.ndarray, seed: int | None) -> np.ndarray:
    grain = artifacts.grain
    gains = stats_gains(frame, artifacts.normalize)
    if seed is None:
        tables = pass_tables(gains, dataclasses.replace(grain, enabled=False))
        return reference_pass(frame, tables, artifacts.lut.table, None, None)
    tile = grain_tile(grain, max(frame.shape[:2]) / grain.frame_width_mm)
    placement = tile_placement(grain, np.random.default_rng(seed))
    return reference_pass(frame, pass_tables(gains, grain), artifacts.lut.table, tile, placement)


def _grade(grader: GpuGrader, artifacts: Artifacts, frame: np.ndarray, seed: int = 7, **kwargs):
    return grader.grade(frame, stats_gains(frame, artifacts.normalize),
                        np.random.default_rng(seed), **kwargs)


def _open(artifacts=None, **device_kwargs):
    artifacts = artifacts or tile_artifacts()
    device, lines = FakeDevice(**device_kwargs), []
    return GpuGrader(artifacts, device, log=lines.append), device, lines, artifacts


# --- grading -----------------------------------------------------------------


def test_grade_is_the_reference_with_the_placement_the_seed_gives_the_cpu():
    grader, device, _, artifacts = _open()
    frame = _frame()
    out = _grade(grader, artifacts, frame, seed=1234)
    assert out.shape == (64, 96, 4) and out.dtype == np.uint8
    assert np.array_equal(out[..., :3], _expected(artifacts, frame, 1234))
    assert device.runs[-1]["placement"] == tile_placement(
        artifacts.grain, np.random.default_rng(1234))


def test_grade_draws_from_the_generator_exactly_what_the_cpu_draws():
    grader, _, _, artifacts = _open()
    rng, twin = np.random.default_rng(99), np.random.default_rng(99)
    grader.grade(_frame(), stats_gains(_frame(), artifacts.normalize), rng)
    tile_placement(artifacts.grain, twin)
    assert rng.bit_generator.state == twin.bit_generator.state


def test_grade_without_a_generator_still_grades():
    grader, device, _, artifacts = _open()
    out = grader.grade(_frame(), stats_gains(_frame(), artifacts.normalize), None)
    assert out.shape == (64, 96, 4)
    assert device.runs[-1]["placement"] is not None


def test_a_frame_of_another_size_is_graded_correctly_each_time():
    grader, device, _, artifacts = _open()
    runs_before = len(device.runs)
    for height, width in ((64, 96), (120, 200), (64, 96)):
        frame = _frame(height, width)
        out = _grade(grader, artifacts, frame, seed=3)
        assert np.array_equal(out[..., :3], _expected(artifacts, frame, 3))
    assert [run["shape"] for run in device.runs[runs_before:]] == [
        (64, 96, 3), (120, 200, 3), (64, 96, 3)]


def test_the_tile_is_loaded_once_per_scale():
    grader, device, _, artifacts = _open()
    loads = len(device.tile_loads)
    _grade(grader, artifacts, _frame(64, 96))
    _grade(grader, artifacts, _frame(64, 96, seed=6))
    assert len(device.tile_loads) == loads + 1
    _grade(grader, artifacts, _frame(96, 64))       # same longer side: same scale
    assert len(device.tile_loads) == loads + 1
    _grade(grader, artifacts, _frame(120, 200))
    assert len(device.tile_loads) == loads + 2
    assert device.tile_loads[-1] == (256, 256)


def test_grain_off_uses_no_tile_a_zero_gain_table_and_no_draw():
    grader, device, _, artifacts = _open()
    loads = len(device.tile_loads)
    frame = _frame()
    rng = np.random.default_rng(4)
    before = rng.bit_generator.state
    out = grader.grade(frame, stats_gains(frame, artifacts.normalize), rng, grain=False)
    assert rng.bit_generator.state == before
    assert device.runs[-1]["placement"] is None
    assert not device.runs[-1]["gain"].any()
    assert len(device.tile_loads) == loads
    assert np.array_equal(out[..., :3], _expected(artifacts, frame, None))


@pytest.mark.parametrize("grain", [
    dataclasses.replace(TILE_GRAIN, enabled=False),
    dataclasses.replace(TILE_GRAIN, strength=0.0),
    GrainParams(model="gaussian", strength=0.0),
])
def test_an_artifact_without_grain_grades_without_a_tile_or_a_draw(grain):
    grader, device, _, artifacts = _open(tile_artifacts(grain=grain))
    frame = _frame()
    rng = np.random.default_rng(4)
    before = rng.bit_generator.state
    out = grader.grade(frame, stats_gains(frame, artifacts.normalize), rng)
    assert rng.bit_generator.state == before
    assert device.tile_loads == [None]
    assert np.array_equal(out[..., :3], _expected(artifacts, frame, None))


def test_timings_receive_the_devices_numbers():
    grader, _, _, artifacts = _open()
    timings: dict = {"normalize": 12.0}
    _grade(grader, artifacts, _frame(), timings=timings)
    assert timings == {"normalize": 12.0, "gpu_upload": 1.5, "gpu_shader": 2.5,
                       "gpu_readback": 3.5}


# --- the self-test -----------------------------------------------------------


def test_the_self_test_frame_is_fixed_and_has_mid_tones():
    frame = self_test_frame()
    assert frame.shape == (64, 96, 3) and frame.dtype == np.uint8
    assert np.array_equal(frame, self_test_frame())
    assert frame.min() == 0 and frame.max() == 255
    mid = ((frame > 64) & (frame < 192)).mean()
    assert mid > 0.4


def test_the_self_test_runs_once_at_set_up_on_the_fixed_frame():
    _, device, _, _ = _open()
    assert [run["shape"] for run in device.runs] == [(64, 96, 3)]
    assert device.runs[0]["placement"] == grader_module.SELF_TEST_PLACEMENT
    assert device.runs[0]["use_tone"] is True          # the bundled look's luma tone curve
    assert device.lut is not None


def test_the_self_test_passes_a_device_off_by_one_level():
    grader, device, _, _ = _open(off_by=1)
    assert grader.enabled and device.closed == 0


def test_the_self_test_refuses_a_device_off_by_two_levels_and_closes_it():
    device = FakeDevice(off_by=2)
    with pytest.raises(GpuError, match="self-test: .*2 levels"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1


@pytest.mark.parametrize("fault", ["skip_grain", "skip_tone"])
def test_the_self_test_notices_a_pass_that_leaves_a_step_out(fault):
    device = FakeDevice(**{fault: True})
    with pytest.raises(GpuError, match="self-test"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1


def test_the_self_test_notices_a_wrong_lut():
    class WrongLut(FakeDevice):
        def load_lut(self, table):
            super().load_lut(np.ascontiguousarray(table.transpose(2, 1, 0, 3)))

    device = WrongLut()
    with pytest.raises(GpuError, match="self-test"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)


def test_a_graphics_error_in_the_self_test_is_a_self_test_failure():
    device = FakeDevice()
    device.fail_next = 1
    with pytest.raises(GpuError, match="self-test: GL error after grading: 0x505"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1


def test_any_other_exception_from_the_device_is_a_gpu_error():
    device = FakeDevice()
    device.fail_next = 1
    device.error = OSError("exception: access violation")
    with pytest.raises(GpuError, match="self-test: OSError: exception: access violation"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1


def test_a_device_returning_the_wrong_shape_fails_the_self_test():
    class ThreeBytes(FakeDevice):
        def run(self, rgb_u8, **kwargs):
            return super().run(rgb_u8, **kwargs)[..., :3]

    with pytest.raises(GpuError, match="self-test"):
        GpuGrader(tile_artifacts(), ThreeBytes(), log=lambda _line: None)


def test_the_self_test_of_an_artifact_without_grain_runs_without_grain():
    _, device, _, _ = _open(tile_artifacts(grain=dataclasses.replace(TILE_GRAIN, enabled=False)))
    assert device.runs[0]["placement"] is None


def test_unsupported_grain_raises_naming_the_model_and_closes_the_device():
    device = FakeDevice()
    with pytest.raises(GpuError, match="gaussian"):
        GpuGrader(Artifacts.default(), device, log=lambda _line: None)
    assert device.closed == 1 and device.runs == []


def test_a_tile_larger_than_the_largest_texture_is_refused():
    device = FakeDevice(max_texture=128)
    with pytest.raises(GpuError, match="256"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1 and device.runs == []


# --- open --------------------------------------------------------------------


def test_open_returns_a_grader_and_logs_one_line():
    lines: list[str] = []
    grader = GpuGrader.open(tile_artifacts(), log=lines.append, device_factory=FakeDevice)
    assert isinstance(grader, GpuGrader) and grader.enabled
    assert lines == ["gpu grading on (Fake GPU, self-test passed)"]


def test_open_logs_no_renderer_when_the_device_has_none():
    class Anonymous(FakeDevice):
        renderer = None

    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append, device_factory=Anonymous)
    assert lines == ["gpu grading on (self-test passed)"]


@pytest.mark.parametrize("error", [
    GpuError("libEGL.so.1: cannot open"), OSError("libEGL.so.1 not found"),
    AttributeError("glTexStorage3D"), RuntimeError("anything else"),
])
def test_open_returns_none_and_logs_once_when_the_factory_raises(error):
    def factory():
        raise error

    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append, device_factory=factory) is None
    assert len(lines) == 1 and lines[0].startswith("gpu grading off: ")
    assert str(error) in lines[0]


def test_open_returns_none_and_logs_once_when_the_self_test_fails():
    devices: list[FakeDevice] = []

    def factory():
        devices.append(FakeDevice(off_by=3))
        return devices[-1]

    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append, device_factory=factory) is None
    assert len(lines) == 1 and lines[0].startswith("gpu grading off: self-test: ")
    assert devices[0].closed == 1


def test_open_with_unsupported_grain_names_the_model_and_builds_no_device():
    made: list[FakeDevice] = []

    def factory():
        made.append(FakeDevice())
        return made[-1]

    lines: list[str] = []
    assert GpuGrader.open(Artifacts.default(), log=lines.append, device_factory=factory) is None
    assert len(lines) == 1 and lines[0].startswith("gpu grading off: ")
    assert "gaussian" in lines[0]
    assert made == []


def test_open_survives_a_log_that_raises():
    def log(_line):
        raise OSError("stdout closed")

    assert GpuGrader.open(tile_artifacts(), log=log, device_factory=FakeDevice) is not None
    assert GpuGrader.open(Artifacts.default(), log=log, device_factory=FakeDevice) is None


# --- can_grade ---------------------------------------------------------------


def test_can_grade_an_ordinary_frame():
    grader, _, _, _ = _open()
    assert grader.can_grade((3040, 4056, 3), ev=0.0, grain=True)
    assert grader.can_grade((3040, 4056, 3), ev=0.0, grain=False)
    assert grader.can_grade((4096, 4096, 3), ev=0, grain=True)


@pytest.mark.parametrize("shape, ev", [
    ((3040, 4056, 3), 0.5),
    ((3040, 4056, 3), -1.0),
    ((2592, 4608, 3), 0.0),          # the IMX708: wider than the largest texture
    ((4608, 2592, 3), 0.0),
    ((3040, 4056), 0.0),
    ((3040, 4056, 4), 0.0),
    ((0, 4056, 3), 0.0),
    ((), 0.0),
])
def test_cannot_grade_what_the_pass_does_not_cover(shape, ev):
    grader, device, lines, _ = _open()
    runs = len(device.runs)
    assert not grader.can_grade(shape, ev=ev, grain=True)
    assert lines == [] and len(device.runs) == runs


def test_cannot_grade_from_another_thread_and_grade_there_touches_nothing():
    grader, device, lines, artifacts = _open()
    seen: dict = {}

    def other():
        seen["can"] = grader.can_grade((64, 96, 3), ev=0.0, grain=True)
        rng = np.random.default_rng(1)
        before = rng.bit_generator.state
        try:
            grader.grade(_frame(), stats_gains(_frame(), artifacts.normalize), rng)
        except GpuError as exc:
            seen["error"] = str(exc)
        seen["drew"] = rng.bit_generator.state != before
        grader.close()

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert seen["can"] is False and "thread" in seen["error"] and seen["drew"] is False
    # Nothing on the device was called from that thread, close included.
    assert device.threads == {threading.get_ident()}
    assert device.closed == 0
    assert not grader.enabled          # closed all the same
    assert lines == []


def test_cannot_grade_after_close_and_close_is_safe_twice():
    grader, device, _, artifacts = _open()
    grader.close()
    grader.close()
    assert device.closed == 1
    assert not grader.enabled
    assert not grader.can_grade((64, 96, 3), ev=0.0, grain=True)
    with pytest.raises(GpuError):
        _grade(grader, artifacts, _frame())


def test_close_survives_a_device_whose_close_raises():
    class Stuck(FakeDevice):
        def close(self):
            super().close()
            raise OSError("no display")

    grader = GpuGrader(tile_artifacts(), Stuck(), log=lambda _line: None)
    grader.close()
    assert not grader.enabled


# --- failures ----------------------------------------------------------------


def test_one_failed_run_is_raised_and_leaves_the_grader_on():
    grader, device, lines, artifacts = _open()
    device.fail_next = 1
    with pytest.raises(GpuError, match="0x505"):
        _grade(grader, artifacts, _frame())
    assert grader.enabled and grader.can_grade((64, 96, 3), ev=0.0, grain=True)
    assert lines == []
    frame = _frame()
    assert np.array_equal(_grade(grader, artifacts, frame, seed=8)[..., :3],
                          _expected(artifacts, frame, 8))


def test_a_success_in_between_resets_the_count():
    grader, device, lines, artifacts = _open()
    for _ in range(4):
        device.fail_next = 2
        for _ in range(2):
            with pytest.raises(GpuError):
                _grade(grader, artifacts, _frame())
        _grade(grader, artifacts, _frame())
    assert grader.enabled and lines == []


def test_three_failures_in_a_row_switch_it_off_and_log_once():
    grader, device, lines, artifacts = _open()
    device.fail_next = 5
    for _ in range(3):
        with pytest.raises(GpuError, match="0x505"):
            _grade(grader, artifacts, _frame())
    assert not grader.enabled
    assert lines == ["gpu grading off: 3 failures in a row"]
    assert not grader.can_grade((64, 96, 3), ev=0.0, grain=True)
    assert device.closed == 1
    runs = len(device.runs)
    with pytest.raises(GpuError):
        _grade(grader, artifacts, _frame())
    assert len(device.runs) == runs and len(lines) == 1
    grader.close()
    assert device.closed == 1


def test_an_unexpected_exception_in_a_grade_is_a_counted_gpu_error():
    grader, device, _, artifacts = _open()
    device.fail_next = 3
    device.error = MemoryError("out of graphics memory")
    for _ in range(3):
        with pytest.raises(GpuError, match="MemoryError"):
            _grade(grader, artifacts, _frame())
    assert not grader.enabled


def test_a_failure_while_loading_the_tile_is_counted_and_the_tile_is_loaded_again():
    class TileFails(FakeDevice):
        fail_tile = False

        def load_tile(self, tile):
            if self.fail_tile and tile is not None:
                self.fail_tile = False
                raise GpuError("GL error after the tile: 0x505")
            super().load_tile(tile)

    artifacts = tile_artifacts()
    device = TileFails()
    grader = GpuGrader(artifacts, device, log=lambda _line: None)
    device.fail_tile = True
    frame = _frame(120, 200)
    with pytest.raises(GpuError, match="the tile"):
        _grade(grader, artifacts, frame)
    assert np.array_equal(_grade(grader, artifacts, frame, seed=2)[..., :3],
                          _expected(artifacts, frame, 2))


# --- fix round 1 -------------------------------------------------------------


def test_grade_raises_only_gpu_error_even_when_the_placement_draw_fails():
    class BadGenerator:
        def integers(self, *args, **kwargs):
            raise TypeError("not a generator")

    grader, device, lines, artifacts = _open()
    runs = len(device.runs)
    gains = stats_gains(_frame(), artifacts.normalize)
    for _ in range(3):
        with pytest.raises(GpuError, match="TypeError: not a generator"):
            grader.grade(_frame(), gains, BadGenerator())
    assert len(device.runs) == runs
    assert not grader.enabled and lines == ["gpu grading off: 3 failures in a row"]


@pytest.mark.parametrize("name", ["stats_gains", "reference_pass", "pass_tables"])
def test_any_exception_in_the_self_test_is_a_self_test_gpu_error(monkeypatch, name):
    def broken(*args, **kwargs):
        raise ValueError("bad numbers")

    monkeypatch.setattr(grader_module, name, broken)
    device = FakeDevice()
    with pytest.raises(GpuError, match="self-test: ValueError: bad numbers"):
        GpuGrader(tile_artifacts(), device, log=lambda _line: None)
    assert device.closed == 1
    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append, device_factory=FakeDevice) is None
    assert lines == ["gpu grading off: self-test: ValueError: bad numbers"]


def test_the_self_test_leaves_the_grain_tile_cache_alone():
    from pifilm import grain as grain_module

    artifacts = tile_artifacts()
    grain_module._tile.cache_clear()
    warmed = grain_tile(artifacts.grain, 4056 / artifacts.grain.frame_width_mm)
    other = grain_tile(artifacts.grain, 640 / artifacts.grain.frame_width_mm)
    before = grain_module._tile.cache_info()
    _, device, _, _ = _open(artifacts)
    after = grain_module._tile.cache_info()
    assert (after.currsize, after.misses) == (before.currsize, before.misses)
    # Both warmed tiles are still the cached objects.
    assert grain_tile(artifacts.grain, 4056 / artifacts.grain.frame_width_mm) is warmed
    assert grain_tile(artifacts.grain, 640 / artifacts.grain.frame_width_mm) is other
    # And the self-test's tile is the one grain_tile would have made.
    small = dataclasses.replace(artifacts.grain, tile_px=grader_module.SELF_TEST_TILE_PX)
    assert device.tile_loads[1] == (64, 64)
    assert np.array_equal(device.tile, grain_tile(small, 96 / small.frame_width_mm))
