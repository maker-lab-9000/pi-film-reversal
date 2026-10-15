"""A stand-in for the GPU device, built on the NumPy reference of the pass.

The real device (``pifilm.gpu.egl``) cannot run on a Mac. This one computes
what the shader would, with switches for the ways a real one goes wrong: a
graphics error on chosen runs, pixels off by some levels, a pass that leaves
out the grain or the tone curve.
"""

from __future__ import annotations

import dataclasses
import threading

import numpy as np

from pifilm.artifacts import Artifacts
from pifilm.gpu.grader import GpuError
from pifilm.gpu.reference import reference_pass
from pifilm.grain import GrainParams

TILE_GRAIN = GrainParams(model="two_scale_tile", strength=0.025, tile_px=256)


def tile_artifacts(**changes) -> Artifacts:
    """The bundled look with a grain the pass can make."""
    return dataclasses.replace(Artifacts.default(), **{"grain": TILE_GRAIN, **changes})


class FakeDevice:
    renderer = "Fake GPU"

    def __init__(self, *, max_texture: int = 4096, off_by: int = 0,
                 skip_grain: bool = False, skip_tone: bool = False) -> None:
        self.max_texture = max_texture
        self.off_by = off_by
        self.skip_grain = skip_grain
        self.skip_tone = skip_tone
        self.fail_next = 0            # this many coming runs raise GpuError
        self.error: Exception | None = None   # raised instead of GpuError when set
        self.lut = None
        self.tile = None
        self.tables = None
        self.tile_loads: list[tuple[int, int] | None] = []
        self.table_loads = 0
        self.runs: list[dict] = []
        self.threads: set[int] = set()
        self.closed = 0

    def _seen(self) -> None:
        self.threads.add(threading.get_ident())

    def load_tables(self, tables) -> None:
        self._seen()
        self.tables = tables
        self.table_loads += 1

    def load_lut(self, table: np.ndarray) -> None:
        self._seen()
        self.lut = table

    def load_tile(self, tile: np.ndarray | None) -> None:
        self._seen()
        self.tile = tile
        self.tile_loads.append(None if tile is None else tile.shape)

    def run(self, rgb_u8, *, use_tone, placement, timings=None) -> np.ndarray:
        self._seen()
        if self.closed:
            raise GpuError("the device is closed")
        self.runs.append({"shape": rgb_u8.shape, "use_tone": use_tone, "placement": placement,
                          "gain": self.tables.gain.copy()})
        if self.fail_next > 0:
            self.fail_next -= 1
            raise self.error or GpuError("GL error after grading: 0x505")
        tables = dataclasses.replace(self.tables, use_tone=use_tone and not self.skip_tone)
        grain = placement is not None and not self.skip_grain
        rgb = reference_pass(rgb_u8, tables, self.lut, self.tile if grain else None,
                             placement if grain else None)
        if self.off_by:
            # Towards the middle, so the shift is never lost to clipping.
            wide = rgb.astype(np.int16)
            rgb = np.where(wide < 128, wide + self.off_by, wide - self.off_by).astype(np.uint8)
        out = np.full((*rgb.shape[:2], 4), 255, np.uint8)
        out[..., :3] = rgb
        if timings is not None:
            timings.update(gpu_upload=1.5, gpu_shader=2.5, gpu_readback=3.5)
        return out

    def close(self) -> None:
        self.closed += 1
