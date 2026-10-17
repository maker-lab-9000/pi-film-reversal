"""``pifilm-gpu-check``: is this machine's GPU grading right, and how fast.

The hardware acceptance tool of the GPU grade. The grader's self-test compares
a 96 x 64 frame with the NumPy reference at every start; this does the same on
a whole photograph (or a synthetic frame of the IMX477's size), prints what
each pass took beside the CPU pipeline on the same frame, and exits 0 only
when no pixel is more than one level from the reference. The difference from
the CPU pipeline is printed for information: the CPU rounds to bytes four
times where the pass rounds once (``pifilm.gpu.reference`` says where that
shows), so it is not what passes or fails. A look with halation is checked
with it: its glow is made on the CPU, as the grader makes it, timed and printed
beside the statistics, and handed to the reference.

Run it on the Pi with the capture service stopped, so the timings are the
pass's own:

    pifilm-gpu-check --artifacts DIR --image ~/Pictures/pifilm/<day>/<name>_original.jpg

Only the GPU runs and the CPU pipeline are timed. The reference is NumPy on a
full frame and takes far longer than either; the output says so before it
starts, so a silent minute is not mistaken for a hang.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import numpy as np

from ..artifacts import Artifacts, ArtifactsError
from ..capture.camera import synthetic_frame
from ..grain import grain_tile, tile_placement
from ..halation import prepare
from ..imageio import load_rgb
from ..normalize import apply_gains_u8, stats_gains
from ..pipeline import Pipeline
from .grader import SELF_TEST_TOLERANCE, Device, GpuError, GpuGrader
from .reference import pass_tables, reference_pass

# The IMX477's still, (height, width): the frame the Pi grades.
SYNTHETIC_SHAPE = (3040, 4056)
# Every run and both paths place the grain from this seed.
SEED = 1234


def _frame(image: str | None, shape: tuple[int, int] = SYNTHETIC_SHAPE) -> np.ndarray:
    if image is None:
        return synthetic_frame(*shape)
    return load_rgb(os.path.expanduser(image))[0]


def _levels(count: int) -> str:
    return f"{count} level" + ("" if count == 1 else "s")


def _difference(out: Callable[[str], None], label: str, got: np.ndarray,
                want: np.ndarray) -> int:
    """Print how far ``got`` is from ``want``; return the largest difference."""
    diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
    share = [float((diff > k).mean()) * 100.0 for k in (0, 1, 2)]
    worst = int(diff.max())
    out(f"  {label}: max {_levels(worst)}, mean {diff.mean():.3f}; "
        f"above 0: {share[0]:.3f} %, above 1: {share[1]:.3f} %, above 2: {share[2]:.3f} %")
    return worst


def _stages(timings: dict) -> str:
    return ", ".join(f"{stage} {ms:.1f}" for stage, ms in timings.items())


def _check(grader: GpuGrader, artifacts: Artifacts, frame: np.ndarray, runs: int,
           out: Callable[[str], None]) -> int:
    height, width = frame.shape[:2]
    out(f"frame {width}x{height}")
    if not grader.can_grade(frame.shape, ev=0.0, grain=True):
        out(f"FAILED: the GPU cannot grade this frame (its largest texture is "
            f"{grader.max_texture}); the capture service grades such a frame on the CPU")
        return 1
    grain = artifacts.grain
    grain_on = grain.enabled and grain.strength > 0

    started = time.perf_counter()
    gains = stats_gains(frame, artifacts.normalize)
    out(f"statistics on the CPU: {(time.perf_counter() - started) * 1000.0:.1f} ms "
        "(paid per photograph on either path)")
    hp = None
    if artifacts.halation.active:
        started = time.perf_counter()
        hp = prepare(apply_gains_u8(frame, gains), artifacts.halation)
        out(f"halation on the CPU: {(time.perf_counter() - started) * 1000.0:.1f} ms "
            "(the glow, paid per photograph on the GPU path)")

    out("GPU pass (the first run also makes the textures and loads the tile):")
    gpu = None
    for number in range(1, runs + 1):
        timings: dict = {}
        started = time.perf_counter()
        try:
            gpu = grader.grade(frame, gains, np.random.default_rng(SEED), timings=timings)
        except GpuError as exc:
            out(f"FAILED: run {number}: {exc}")
            return 1
        total = (time.perf_counter() - started) * 1000.0
        out(f"  run {number}: total {total:.1f} ms (upload {timings.get('gpu_upload')}, "
            f"shader {timings.get('gpu_shader')}, read-back {timings.get('gpu_readback')})")

    out("CPU pipeline, same frame:")
    pipeline = Pipeline(artifacts)
    pipeline.warm((height, width))
    cpu = None
    for label in ("first", "again"):       # the first also warms OpenCV's threads
        stages: dict = {}
        started = time.perf_counter()
        cpu, _ = pipeline.process(frame, rng=np.random.default_rng(SEED), timings=stages)
        total = (time.perf_counter() - started) * 1000.0
        out(f"  {label}: total {total:.1f} ms ({_stages(stages)})")

    out("comparing with the NumPy reference (not timed; this step is slow, "
        "up to a minute or two for a 12 MP frame on the Pi) ...")
    tile = placement = None
    if grain_on:
        tile = grain_tile(grain, max(height, width) / grain.frame_width_mm)
        placement = tile_placement(grain, np.random.default_rng(SEED))
    want = reference_pass(frame, pass_tables(gains, grain), artifacts.lut.table, tile, placement,
                          halation=hp)
    worst = _difference(out, "against the reference", gpu[..., :3], want)
    _difference(out, "against the CPU pipeline", gpu[..., :3], cpu)
    if worst > SELF_TEST_TOLERANCE:
        out(f"FAILED: the GPU is {_levels(worst)} from the reference "
            f"(at most {SELF_TEST_TOLERANCE})")
        return 1
    out(f"ok: the GPU is within {_levels(SELF_TEST_TOLERANCE)} of the reference")
    return 0


def main(argv: list[str] | None = None, *, out: Callable[[str], None] = print,
         device_factory: Callable[[], Device] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade one frame on the GPU, compare it with the reference and time it.")
    parser.add_argument("--artifacts", required=True, metavar="DIR",
                        help="the look to grade with (its grain must be two_scale_tile, or off)")
    parser.add_argument("--image", metavar="FILE",
                        help="a photograph to grade (default: a synthetic 4056 x 3040 frame)")
    parser.add_argument("--runs", type=int, default=3, metavar="N",
                        help="GPU passes to time (default: 3)")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    try:
        artifacts = Artifacts.load(args.artifacts)
        frame = _frame(args.image)
    except (ArtifactsError, OSError, ValueError) as exc:
        out(f"error: {exc}")
        return 1
    grader = GpuGrader.open(artifacts, log=out, device_factory=device_factory)
    if grader is None:
        out("no GPU grading on this machine: the capture service would grade on the CPU")
        return 1
    try:
        return _check(grader, artifacts, frame, args.runs, out)
    finally:
        grader.close()


if __name__ == "__main__":
    raise SystemExit(main())
