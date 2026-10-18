"""The Parr pipeline: normalise, halation, apply the LUT, add grain.

The order is fixed and matters. The LUT was fitted on normalised input, so
normalisation comes first; grain is a property of the developed film, so it
goes on last. Halation (``pifilm.halation``), when the look has it, sits
between normalisation and the LUT: the glow forms as the film is exposed, so it
is made from the exposure the LUT is handed, and the LUT's roll-off then treats
the glow as it treats any other light. A look without halation grades exactly
as before there was the step. The same ``Pipeline`` serves full-resolution
captures, the low-resolution live preview (``grain=False``, ``halation=False``)
and batch reprocessing, which is what guarantees the preview shows the grade
the capture will get, less the two film effects a viewfinder does without.

``info`` returns the gains that were applied, whether either gain hit its
clamp, the LUT's content hash and a hash of the normalisation parameters. The
capture app writes all of it to its log so a surprising frame can be explained
after the fact. Both hashes are needed: normalisation is a per-artifact
setting, so the same LUT under two different ``NormalizeParams`` produces two
different grades that ``lut_sha1`` alone cannot tell apart.

Exposure compensation (``ev``) has to survive normalisation. The camera
honours EV by exposing darker or brighter, but normalisation exists to bring
every frame to one exposure, so on its own it hands the LUT the same image at
-2 EV as at 0: the first shots with the viewfinder's EV buttons (2026-09-24)
came out 2 stops apart as originals and identical as graded files. So the
compensation is taken out of the frame (a ``2 ** -ev`` gain in linear light),
the frame is normalised as the plain auto exposure it would have been, and the
compensation is put back before the LUT. Undoing it first, rather than only
re-applying it afterwards, keeps the result exactly ``ev`` stops from the
uncompensated grade even where normalisation's clamps cannot fully restore a
very dark frame - re-applying alone darkened such a frame twice. Before the
LUT is where film takes exposure: the negative is exposed differently and
developed the same. The value is recorded as ``ev_comp`` so a regrade can
reproduce it. Halation follows the compensation: it is made from the frame as
exposed, the one the LUT is given.

The GPU. ``process`` can be handed a grader (``pifilm.gpu.grader``), which does
normalise, halation, LUT and grain as one pass on the Pi's GPU (it makes the
glow itself, on the CPU, and is told whether to). The statistics stay here,
on the CPU, from the same subsample (``stats_gains``), so the audit fields are
the same whichever path graded; only the pixels differ, by the CPU's rounding
between its stages. The grader is asked first whether it can take the frame
(``can_grade``: a compensated exposure, a frame too wide, another thread are
all ordinary reasons to say no, and none is logged). If it then fails, the same
frame is graded on the CPU at once. The grader had already drawn the grain's
placement from the generator by then, so the generator is put back to where it
was before the attempt: the CPU grade of a failed GPU attempt is byte for byte
the grade there would have been with no GPU, and the shot's seed still
reproduces its grain.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .artifacts import Artifacts
from .color import linear_to_srgb, srgb_to_linear
from .gpu.grader import GpuError
from .gpu.shader import BACKEND
from .grain import add_grain, warm
from .halation import apply_cpu, prepare
from .normalize import Gains, normalize_u8, stats_gains


def _lap(timings: dict | None, stage: str, started: float) -> float:
    """Record the milliseconds since ``started`` under ``stage``; return now."""
    now = time.perf_counter()
    if timings is not None:
        timings[stage] = round((now - started) * 1000.0, 1)
    return now


class Pipeline:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self._filter = artifacts.lut.to_pillow()
        self.normalize_sha1 = hashlib.sha1(
            json.dumps(artifacts.normalize.to_dict(), sort_keys=True).encode()
        ).hexdigest()

    def warm(self, shape: tuple[int, int]) -> None:
        """Prepare for frames of ``(height, width)``, so the first photograph does
        not pay for what is built once (the tile grain's tile)."""
        warm(self.artifacts.grain, shape)

    def process(
        self,
        rgb_u8: np.ndarray,
        *,
        grain: bool = True,
        rng: np.random.Generator | None = None,
        ev: float = 0.0,
        timings: dict | None = None,
        grader=None,
        used: dict | None = None,
        halation: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Grade one frame. ``timings``, when given, is filled with the milliseconds
        each stage took (``normalize``, ``halation`` when it ran, ``lut``, and
        ``grain`` when it ran). It is an out-parameter and not part of the
        returned audit dict, because that dict is what a regrade is reproduced
        from and a duration is not reproducible.

        ``halation=False`` leaves the look's halation out, as ``grain=False`` its
        grain (the live view does both); the audit still names the look's
        halation (``halation_sha1``, when it has one), as it names its LUT.

        ``grader``, when given and able to take this frame, grades it on the GPU
        (see the module docstring); the stages are then ``normalize`` (the
        statistics), ``gpu`` (the whole pass) and the grader's ``gpu_upload``,
        ``gpu_shader`` and ``gpu_readback``, and its ``halation`` (the glow)
        when that ran. ``used``, when given, is another out-parameter:
        ``used["backend"]`` says what graded the frame (``BACKEND``, ``"gpu-2"``,
        or ``"cpu"``), and ``used["gpu_error"]`` is there when the GPU was tried
        and failed, for the caller to log.

        The image returned is ``(H, W, 3)``, or ``(H, W, 4)`` with an unused
        fourth byte when the GPU graded it (``save_jpeg`` takes either). Without
        a grader nothing here differs from before there was one."""
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError(
                f"process() expects an RGB uint8 array of shape (H, W, 3); "
                f"got dtype {rgb_u8.dtype} shape {rgb_u8.shape}"
            )
        ev = float(ev)
        with_grain = bool(grain and self.artifacts.grain.enabled)
        with_halation = bool(halation and self.artifacts.halation.active)
        if used is not None:
            used["backend"] = "cpu"
        if grader is not None and grader.can_grade(rgb_u8.shape, ev=ev, grain=with_grain):
            # The generator as the caller gave it, for the CPU if the GPU fails.
            state = rng.bit_generator.state if rng is not None else None
            gpu_timings: dict = {}
            try:
                started = time.perf_counter()
                gains = stats_gains(rgb_u8, self.artifacts.normalize)
                started = _lap(gpu_timings, "normalize", started)
                graded = grader.grade(rgb_u8, gains, rng, grain=with_grain,
                                      halation=with_halation, timings=gpu_timings)
                _lap(gpu_timings, "gpu", started)
            except GpuError as exc:
                if rng is not None:
                    rng.bit_generator.state = state
                if used is not None:
                    used["gpu_error"] = str(exc)
            else:
                if timings is not None:
                    timings.update(gpu_timings)
                if used is not None:
                    used["backend"] = BACKEND
                return graded, self._audit(gains, ev)
        started = time.perf_counter()
        if ev != 0.0:
            rgb_u8 = _ev_table(-ev)[rgb_u8]
        normalised, gains = normalize_u8(rgb_u8, self.artifacts.normalize)
        if ev != 0.0:
            normalised = _ev_table(ev)[normalised]
        started = _lap(timings, "normalize", started)
        if with_halation:
            # From the frame the LUT is given: normalised, compensation put back.
            normalised = apply_cpu(normalised, prepare(normalised, self.artifacts.halation))
            started = _lap(timings, "halation", started)
        graded = self.artifacts.lut.apply_pillow(normalised, self._filter)
        started = _lap(timings, "lut", started)
        if with_grain:
            graded = add_grain(graded, self.artifacts.grain, rng)
            _lap(timings, "grain", started)
        return graded, self._audit(gains, ev)

    def _audit(self, gains: Gains, ev: float) -> dict:
        """The record's audit fields. The same for the same gains, whichever
        path graded. ``halation_sha1`` follows ``normalize_sha1`` only for a
        look with halation, so a look without it records what it always did."""
        audit = {
            "wb_gains": [round(float(g), 4) for g in gains.wb],
            "exposure_gain": round(float(gains.exposure), 4),
            "levels": (
                {k: (v if isinstance(v, str) else round(float(v), 4))
                 for k, v in gains.levels.items()}
                if gains.levels else None
            ),
            "clamped": dict(gains.clamped),
            "lut_sha1": self.artifacts.lut_sha1,
            "normalize_sha1": self.normalize_sha1,
        }
        if self.artifacts.halation.active:
            audit["halation_sha1"] = self.artifacts.halation.sha1
        audit["ev_comp"] = ev
        return audit


def _ev_table(ev: float) -> np.ndarray:
    """256-entry sRGB table for a ``2 ** ev`` exposure gain in linear light."""
    codes = np.arange(256, dtype=np.float32) / 255.0
    scaled = linear_to_srgb(np.clip(srgb_to_linear(codes) * 2.0 ** ev, 0.0, 1.0))
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
