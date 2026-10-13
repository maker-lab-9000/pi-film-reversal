"""The grader: one photograph's grade on a GPU device, and what happens when
the device cannot be trusted.

``GpuGrader`` drives a narrow ``Device`` (the real one is ``pifilm.gpu.egl``;
the tests use a fake built on ``pifilm.gpu.reference``). It owns the decisions
around the pass, none of the graphics calls:

**The self-test.** The graphics code cannot run in the test suite, so nothing
before the Pi proves that the driver computes what the reference does. At set
up the grader grades a small fixed frame and compares it with
``reference_pass``; more than one level anywhere, or any error, and there is
no grader for this run. One level is the room for the GPU's own rounding of an
exact half and for its float arithmetic. The frame has gradients and mid-tones
on purpose: a flat frame normalises to black or white, where the grain's
envelope is zero, and the grain would go untested. Its tile is a small one of
the artifact's grain (``SELF_TEST_TILE_PX``), so the frame wraps round it and
the start-up does not pay for a full tile at a scale no photograph has.

**One thread.** A graphics context belongs to the thread that made it. The
grader remembers that thread; from any other, ``can_grade`` says no, ``grade``
raises before touching the device, and ``close`` only marks the grader off.

**Failures.** Every failure of the device surfaces as ``GpuError``, whatever
it was underneath, so the caller has one thing to catch before it grades the
same photograph on the CPU. Three in a row switch the grader off for the run
and release the device. ``open`` never raises: no GPU is an ordinary state
(a Mac, a Pi without the driver), reported in one log line.

**The seed.** The grain's placement is drawn by ``pifilm.grain.tile_placement``
exactly when the CPU's ``add_grain`` would draw it, so a shot's seed puts the
grain in the same place on both paths, and a grade without grain draws nothing.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from ..artifacts import Artifacts
from ..grain import GrainParams, grain_tile, tile_placement
from ..normalize import Gains, stats_gains
from .reference import PassTables, pass_tables, reference_pass, supports

# The self-test: a frame of this (height, width), a tile of this size, placed
# here. The orientation has all three bits set, so the transpose and both flips
# run; the frame is wider than the tile, so the wrap runs too.
SELF_TEST_SHAPE = (64, 96)
SELF_TEST_TILE_PX = 64
SELF_TEST_PLACEMENT = (41, 23, 7)
# Levels the device may be from the reference before it is not believed.
SELF_TEST_TOLERANCE = 1


class GpuError(Exception):
    """The GPU cannot be used, or a pass on it failed. The CPU grades instead."""


class Device(Protocol):
    """What the grader needs of a GPU. ``pifilm.gpu.egl`` is the real one."""

    max_texture: int

    def load_tables(self, tables: PassTables) -> None: ...     # per photograph; cheap

    def load_lut(self, table: np.ndarray) -> None: ...         # once

    def load_tile(self, tile: np.ndarray | None) -> None: ...  # once per tile; None: no grain

    def run(self, rgb_u8: np.ndarray, *, use_tone: bool,
            placement: tuple[int, int, int] | None,
            timings: dict | None = None) -> np.ndarray: ...    # (H, W, 4) uint8; raises GpuError

    def close(self) -> None: ...


def self_test_frame() -> np.ndarray:
    """The fixed frame of the self-test: three crossing gradients with a cast
    (mid-tones, where the grain is strongest and the tone curve bends), and a
    full grey ramp along the bottom so the black and white points are found."""
    height, width = SELF_TEST_SHAPE
    y, x = np.mgrid[0:height, 0:width]
    frame = np.stack([24 + 2 * x, 30 + 3 * y, 210 - x - y], axis=-1)
    frame[height - 8:] = ((x[height - 8:] * 255) // (width - 1))[..., None]
    return frame.astype(np.uint8)


def _grain_on(grain: GrainParams) -> bool:
    """Whether the CPU's ``add_grain`` would draw from the generator."""
    return grain.enabled and grain.strength > 0


def _unsupported(grain: GrainParams) -> str:
    return f"the GPU pass has no {grain.model!r} grain"


class GpuGrader:
    MAX_FAILURES = 3

    def __init__(self, artifacts: Artifacts, device: Device, *,
                 log: Callable[[str], None] = print) -> None:
        """Load the LUT and run the self-test. Raises ``GpuError`` (with the
        device closed) when this artifact or this device cannot be used."""
        self._artifacts = artifacts
        self._device = device
        self._log = log
        self._thread = threading.get_ident()
        self._enabled = False
        self._closed = False
        self._failures = 0
        self._tile_key: tuple[int, float] | None = None
        try:
            self._set_up()
        except BaseException:
            self._close_device()
            raise
        self._enabled = True

    @classmethod
    def open(cls, artifacts: Artifacts, *, log: Callable[[str], None] = print,
             device_factory: Callable[[], Device] | None = None) -> GpuGrader | None:
        """A working grader, or None with one ``gpu grading off:`` line. Never
        raises. Call it on the thread that will grade."""
        def say(line: str) -> None:
            try:
                log(line)
            except Exception:  # noqa: BLE001 - a broken log must not decide the backend
                pass

        try:
            # Asked first so an artifact the pass cannot grade costs no context.
            if not supports(artifacts.grain):
                raise GpuError(_unsupported(artifacts.grain))
            if device_factory is None:
                from .egl import open_device as device_factory
            device = device_factory()
            grader = cls(artifacts, device, log=log)
        except Exception as exc:  # noqa: BLE001 - whatever went wrong, the CPU grades
            say(f"gpu grading off: {exc}")
            return None
        renderer = getattr(device, "renderer", None)
        say(f"gpu grading on ({renderer + ', ' if renderer else ''}self-test passed)")
        return grader

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def max_texture(self) -> int:
        """The longest side of a frame the device can take."""
        return self._device.max_texture

    def can_grade(self, shape: tuple[int, ...], *, ev: float, grain: bool) -> bool:
        """Whether ``grade`` may be called for a frame of ``shape`` now, from
        this thread. Never logs and never raises: a photograph the pass does
        not cover is an ordinary one for the CPU. ``grain`` needs no check (an
        artifact whose grain the pass cannot make never gets a grader)."""
        try:
            return (
                self._enabled
                and threading.get_ident() == self._thread
                and float(ev) == 0.0
                and len(shape) == 3
                and shape[2] == 3
                and min(shape[:2]) > 0
                and max(shape[:2]) <= self._device.max_texture
            )
        except Exception:  # noqa: BLE001 - an odd shape or EV is a no, not an error
            return False

    def grade(self, rgb_u8: np.ndarray, gains: Gains, rng: np.random.Generator | None,
              *, grain: bool = True, timings: dict | None = None) -> np.ndarray:
        """Grade one frame with ``gains`` (from ``stats_gains``). Returns
        ``(H, W, 4)`` uint8, the fourth byte unused. ``timings``, when given,
        gets ``gpu_upload``, ``gpu_shader`` and ``gpu_readback`` in milliseconds.

        Raises ``GpuError`` on any failure; the caller then grades this frame
        on the CPU. The generator is advanced only by the placement draw, the
        same draw the CPU makes, so the caller can restore its state and get
        the CPU's own result."""
        if not self._enabled:
            raise GpuError("gpu grading is off")
        if threading.get_ident() != self._thread:
            raise GpuError("the grader belongs to the thread that opened it")
        params = self._artifacts.grain
        placement = None
        if grain and _grain_on(params):
            rng = rng if rng is not None else np.random.default_rng()
            placement = tile_placement(params, rng)
        else:
            params = dataclasses.replace(params, enabled=False)
        try:
            out = self._run(rgb_u8, gains, params, placement, timings)
        except GpuError:
            self._failures += 1
            if self._failures >= self.MAX_FAILURES:
                self._enabled = False
                self._close_device()
                self._say(f"gpu grading off: {self._failures} failures in a row")
            raise
        self._failures = 0
        return out

    def close(self) -> None:
        """Switch the grader off and release the device. Safe to call twice.
        From a thread other than the grader's it only switches it off: the
        context is not that thread's to release, and the process is ending."""
        self._enabled = False
        if threading.get_ident() == self._thread:
            self._close_device()

    # --- internals -----------------------------------------------------------

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:  # noqa: BLE001 - a broken log must not fail a grade
            pass

    def _close_device(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._device.close()
        except Exception:  # noqa: BLE001 - nothing more can be done with it
            pass

    def _set_up(self) -> None:
        grain = self._artifacts.grain
        if not supports(grain):
            raise GpuError(_unsupported(grain))
        if _grain_on(grain) and grain.tile_px > self._device.max_texture:
            raise GpuError(f"the grain tile ({grain.tile_px} px) is larger than the "
                           f"largest texture ({self._device.max_texture})")
        try:
            self._device.load_lut(self._artifacts.lut.table)
            # Every sampler of the shader has a texture from the start; a real
            # tile replaces this one when the first photograph with grain comes.
            self._device.load_tile(None)
        except GpuError:
            raise
        except Exception as exc:  # noqa: BLE001 - see _run
            raise GpuError(f"{type(exc).__name__}: {exc}") from exc
        try:
            self._self_test()
        except GpuError as exc:
            raise GpuError(f"self-test: {exc}") from exc

    def _self_test(self) -> None:
        frame = self_test_frame()
        gains = stats_gains(frame, self._artifacts.normalize)
        grain = self._artifacts.grain
        if _grain_on(grain):
            grain = dataclasses.replace(grain, tile_px=min(grain.tile_px, SELF_TEST_TILE_PX))
            placement = SELF_TEST_PLACEMENT
            tile = self._tile(grain, frame.shape)
        else:
            grain = dataclasses.replace(grain, enabled=False)
            placement = tile = None
        got = self._run(frame, gains, grain, placement, None)
        want = reference_pass(frame, pass_tables(gains, grain), self._artifacts.lut.table,
                              tile, placement)
        worst = int(np.abs(got[..., :3].astype(np.int16) - want.astype(np.int16)).max())
        if worst > SELF_TEST_TOLERANCE:
            raise GpuError(f"the device is {worst} levels from the reference "
                           f"(at most {SELF_TEST_TOLERANCE} allowed)")

    @staticmethod
    def _tile(grain: GrainParams, shape: tuple[int, ...]) -> np.ndarray:
        """The tile the CPU's ``add_grain`` uses for a frame of ``shape``."""
        return grain_tile(grain, max(shape[:2]) / grain.frame_width_mm)

    def _run(self, rgb_u8: np.ndarray, gains: Gains, grain: GrainParams,
             placement: tuple[int, int, int] | None, timings: dict | None) -> np.ndarray:
        """One pass on the device. Anything the device raises leaves as
        ``GpuError``: ctypes and NumPy have their own exceptions (a bad
        argument, no memory), and the caller's fallback catches one type."""
        try:
            tables = pass_tables(gains, grain)
            if placement is not None:
                key = (grain.tile_px, max(rgb_u8.shape[:2]) / grain.frame_width_mm)
                if key != self._tile_key:
                    # Forgotten first: a load that fails half way leaves no tile
                    # this key could be mistaken for.
                    self._tile_key = None
                    self._device.load_tile(self._tile(grain, rgb_u8.shape))
                    self._tile_key = key
            self._device.load_tables(tables)
            out = self._device.run(rgb_u8, use_tone=tables.use_tone, placement=placement,
                                   timings=timings)
        except GpuError:
            raise
        except Exception as exc:  # noqa: BLE001 - see the docstring
            raise GpuError(f"{type(exc).__name__}: {exc}") from exc
        expected = (*rgb_u8.shape[:2], 4)
        if getattr(out, "shape", None) != expected or out.dtype != np.uint8:
            raise GpuError(f"the device returned {getattr(out, 'shape', None)}, "
                           f"not {expected} bytes")
        return out
