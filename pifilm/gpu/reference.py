"""What the GPU pass computes, in NumPy.

The shader (``pifilm.gpu.shader``) cannot run on a Mac or in the test suite,
so this module stands in for it. ``reference_pass`` is the fragment shader
step for step, with the shader's own names (``b``, ``c``, ``y``, ``s``, ``lo``,
``f``, ``c00`` ... ``g``, ``row``, ``col``), in float32 like the shader's
``highp`` float, rounded to bytes once at the end. It has three uses: the
tests compare it with the CPU pipeline, the grader's self-test on the Pi
compares the GPU with it at every start (so a driver that changes the output
is caught that day), and the check command does the same on a full frame.
``pass_tables`` builds the float tables the shader and this reference both
read, so the two cannot be handed different numbers.

Where the result differs from the CPU path (``Pipeline.process``), and why
that is expected. The CPU rounds to bytes four times: after the per-channel
tables, after the luma tone curve, after the LUT, and inside the grain. This
pass rounds once. Measured on three real 12 MP photographs with this
emulation (2026-10-13): 0.23 to 0.29 levels on average without grain, about
0.6 with, the grain in the same place for the same seed, and on one
photograph up to 10 levels on 0.06 % of the pixels.

Against the float path that never rounds (``apply_gains_float`` then
``LUT3D.apply_numpy``) this pass is exact to a level almost everywhere, and
the CPU pipeline is as far from that path as it is from this one. So the
difference is the CPU's rounding. Taken apart on synthetic frames with the
bundled starter look (``tests/test_gpu_reference.py``), from largest:

* **Saturated colours under the luma tone curve.** The CPU applies that curve
  through a byte YCrCb round trip, which does not return a saturated colour
  to itself: (0, 231, 0) comes back (1, 233, 0) with the curve doing nothing.
  A level or two before the LUT is a level or two after it in most of the
  cube, but the starter LUT is steep at the edge of the gamut, and there the
  graded pixel moves 5 to 20 levels. This pass adds the change in Y to R, G
  and B and clamps, with no round trip. These are the largest differences
  (up to 20 levels on strong primaries), they need the luma tone curve, and
  an artifact with per-channel tone or no levels stayed within 9.
* **A steep tone curve on a rounded Y.** A dark frame gets a curve several
  times steeper than 1 near black. The CPU rounds Y to a byte before the
  curve, so half a level becomes a few. This raises the average (up to 0.8 of
  a level on a dark synthetic frame) more than the maximum.
* **The LUT read from rounded bytes.** With no luma step the CPU still hands
  the LUT a rounded normalised pixel. A quarter of a level on average, a few
  levels where the LUT is steep.
* **The grain.** From the same graded bytes the two grains differ by at most
  2 levels (the CPU adds to a Y byte and returns through byte chroma; its
  strength table is indexed by that byte, here interpolated at the float
  luminance). It adds a few tenths to the average and little to the maximum.

One place where this pass is itself approximate, as the shader is: the tone
curve is a 256-entry table read with linear interpolation. With a gamma below
1 the curve leaves zero at the black point with an unbounded slope, and within
a level of that point the table is up to 8 levels from the curve. The CPU
reads the same table at a byte, so this is no worse than today.

Two things the shader does that this does not copy, both far below a level:
the tile is a 16-bit float texture on the GPU (about three decimal digits, on
a term of at most a few levels), and the GPU's final rounding of an exact
half is its own business. The self-test's tolerance of one level covers both.

**Halation** (``pifilm.halation``) is a step of the pass, after the tone step
and before the LUT, where the CPU runs it between normalisation and the LUT.
Each channel goes to linear light, is expanded (``x / (1 - hr x)``), mixed
towards the glow by ``hw`` (strength times weights), compressed back
(``x / (1 + hr x)``) and returned to sRGB. The glow itself, a blur of a reduced
copy of the frame, is not the shader's work: ``pifilm.halation.prepare`` makes
it on the CPU from the normalised bytes for both paths, so the two mix the same
numbers, and the shader only reads it back at full size with a hand-written
bilinear read at half-pixel centres, edges clamped (``_glow_at`` here), which is
what ``cv2.resize``'s ``INTER_LINEAR`` does on the CPU. The CPU path rounds to
bytes once more, after the mix, where the pass does not. On a mid-tone random
frame with a bright square (``tests/test_gpu_reference.py``, 2026-10-17) the
two are 0.77 levels apart on average and 7 at most at strength 0.4, against
0.86 and 8 with no halation: the extra rounding does not show beside the
others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..color import linear_to_srgb, srgb_to_linear
from ..grain import GrainParams
from ..halation import HalationPass
from ..normalize import Gains, _apply_levels_linear, _tone_curve

# The shader's LUMA constant: the weights of the Y in YCrCb (BT.601), which is
# the luma the CPU's tone curve and grain work on.
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
# Rows are worked in bands of about this many pixels: the eight LUT reads are
# each a float RGB copy of the band, and a 12 MP frame at once would need
# gigabytes on a Pi that is also holding the photograph.
BAND_PIXELS = 1 << 19


@dataclass(frozen=True)
class PassTables:
    """The 256-entry float tables of one photograph's pass."""

    chan: np.ndarray      # (256, 3) float32: per-channel normalisation
    tone: np.ndarray      # (256,) float32: luma tone curve (identity when unused)
    use_tone: bool
    gain: np.ndarray      # (256,) float32: grain strength by luminance, 0..1 units


def supports(grain: GrainParams) -> bool:
    """True when the pass can reproduce this artifact's grain: grain disabled or
    strength 0, or the ``two_scale_tile`` model. The Gaussian and ``two_scale``
    models draw a fresh noise field per photograph, which the shader cannot."""
    return not _grain_on(grain) or grain.model == "two_scale_tile"


def _grain_on(grain: GrainParams) -> bool:
    """Whether ``add_grain`` would change the frame at all."""
    return grain.enabled and grain.strength > 0


def pass_tables(gains: Gains, grain: GrainParams) -> PassTables:
    """The normalisation and the grain's strength as float tables.

    ``chan`` and ``tone`` are ``gains_to_luts`` and ``tone_lut`` of
    ``pifilm.normalize`` without their rounding to bytes. ``gain`` is
    ``strength`` times the envelope, the table ``add_grain`` indexes by the Y
    byte (there scaled by 255); all zeros when the grain is off. A grain model
    the pass cannot make is refused rather than given the wrong envelope.
    """
    if not supports(grain):
        raise ValueError(f"the GPU pass has no {grain.model!r} grain; check supports() first")
    levels = np.arange(256, dtype=np.float32) / 255.0
    lin = srgb_to_linear(levels)
    use_tone = gains.levels is not None and gains.levels.get("tone", "channel") == "luma"
    chan = np.empty((256, 3), np.float32)
    for channel in range(3):
        x = lin * gains.combined[channel]
        if gains.levels is not None and not use_tone:
            x = _apply_levels_linear(x, gains.levels)
        else:
            x = np.clip(x, 0.0, 1.0)
        chan[:, channel] = linear_to_srgb(x)
    tone = np.clip(_tone_curve(levels, gains.levels), 0.0, 1.0) if use_tone else levels
    if _grain_on(grain):
        gain = grain.strength * np.interp(levels, grain.envelope_luma, grain.envelope_rel)
    else:
        gain = np.zeros(256)
    return PassTables(chan=chan, tone=np.asarray(tone, np.float32), use_tone=use_tone,
                      gain=np.asarray(gain, np.float32))


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """GLSL ``mix``: ``a * (1 - t) + b * t``."""
    return a * (1.0 - t) + b * t


def _table(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The shader's ``table``: a 256-entry table read at ``v`` in 0..1 with
    linear interpolation between entries."""
    x = np.clip(v, 0.0, 1.0) * np.float32(255.0)
    lo = np.minimum(np.floor(x), np.float32(254.0))
    i = lo.astype(np.intp)
    return _mix(t[i], t[i + 1], x - lo)


def reference_pass(rgb_u8: np.ndarray, tables: PassTables, lut: np.ndarray,
                   tile: np.ndarray | None,
                   placement: tuple[int, int, int] | None,
                   halation: HalationPass | None = None) -> np.ndarray:
    """What the shader computes, in NumPy float32: returns ``(H, W, 3)`` uint8.

    ``lut`` is ``LUT3D.table`` (indexed ``[r, g, b]``). ``tile`` is
    ``pifilm.grain.grain_tile`` for this frame and ``placement`` the
    ``(oy, ox, orientation)`` of ``pifilm.grain.tile_placement``; both None
    means no grain (the shader's ``use_grain == 0``). ``halation`` is
    ``pifilm.halation.prepare`` of this frame's normalised bytes; None means
    no halation (``use_halation == 0``).
    """
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(
            f"reference_pass() expects an RGB uint8 array of shape (H, W, 3); "
            f"got dtype {rgb_u8.dtype} shape {rgb_u8.shape}"
        )
    if (tile is None) != (placement is None):
        raise ValueError("tile and placement are given together, or neither is")
    height, width = rgb_u8.shape[:2]
    out = np.empty((height, width, 3), np.uint8)
    rows = max(1, BAND_PIXELS // max(width, 1))
    for top in range(0, height, rows):
        out[top:top + rows] = _band(rgb_u8[top:top + rows], top, (height, width), tables, lut,
                                    tile, placement, halation)
    return out


def _glow_at(glow: np.ndarray, top: int, rows: int, cols: int,
             height: int, width: int) -> np.ndarray:
    """The shader's read of the glow for rows ``top .. top + rows`` and columns
    ``0 .. cols`` of a ``height`` x ``width`` frame: ``(rows, cols, 3)`` float32.
    Bilinear by hand, at half-pixel centres, edges clamped, x first, then y."""
    small_h, small_w = glow.shape[:2]

    def axis(first: int, count: int, small: int, full: int):
        # vec2 u = (vec2(px) + 0.5) * vec2(gs) / vec2(textureSize(img, 0)) - 0.5;
        index = np.arange(first, first + count).astype(np.float32)
        u = (index + np.float32(0.5)) * np.float32(small) / np.float32(full) - np.float32(0.5)
        # u = clamp(u, vec2(0.0), vec2(gs - 1));
        u = np.clip(u, np.float32(0.0), np.float32(small - 1))
        # ivec2 i0 = ivec2(floor(u));  ivec2 i1 = min(i0 + 1, gs - 1);  vec2 t = u - vec2(i0);
        lo = np.floor(u)
        i0 = lo.astype(np.intp)
        i1 = np.minimum(i0 + 1, small - 1)
        return i0, i1, u - lo

    y0, y1, ty = axis(top, rows, small_h, height)
    x0, x1, tx = axis(0, cols, small_w, width)
    tx, ty = tx[None, :, None], ty[:, None, None]
    # vec3 B = mix(mix(G(i0), G(ivec2(i1.x, i0.y)), t.x),
    #              mix(G(ivec2(i0.x, i1.y)), G(i1), t.x), t.y);
    upper = _mix(glow[np.ix_(y0, x0)], glow[np.ix_(y0, x1)], tx)
    lower = _mix(glow[np.ix_(y1, x0)], glow[np.ix_(y1, x1)], tx)
    return _mix(upper, lower, ty).astype(np.float32, copy=False)


def _band(b: np.ndarray, top: int, size: tuple[int, int], tables: PassTables, lut: np.ndarray,
          tile: np.ndarray | None, placement: tuple[int, int, int] | None,
          halation: HalationPass | None) -> np.ndarray:
    """The shader's ``main`` for the rows of ``b``, the first being row ``top``
    of a frame of ``size`` (height, width). The comments quote the shader line
    each step stands for."""
    # vec3 c = vec3(texelFetch(chan, ivec2(b.r, 0), 0).r, ...g, ...b);
    chan = tables.chan
    c = np.stack([chan[b[..., 0], 0], chan[b[..., 1], 1], chan[b[..., 2], 2]], axis=-1)
    if tables.use_tone:
        # float y = dot(c, LUMA);  c = clamp(c + (table(tone, y) - y), 0.0, 1.0);
        y = c @ LUMA
        c = np.clip(c + (_table(tables.tone, y) - y)[..., None], 0.0, 1.0)
    if halation is not None:
        # The shader's ``l`` is ``lin`` here. ``step(edge, v)`` is 1 when
        # ``v >= edge``, so the first branch of each ``mix`` is ``v < edge``.
        # vec3 l = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
        hr = np.float32(halation.headroom)
        lin = np.where(c < np.float32(0.04045), c / np.float32(12.92),
                       ((c + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4))
        # vec3 x = l / (1.0 - hr * l);
        x = lin / (np.float32(1.0) - hr * lin)
        big = _glow_at(halation.glow, top, b.shape[0], b.shape[1], *size)
        # x = x + hw * (B - x);  l = x / (1.0 + hr * x);
        x = x + np.asarray(halation.mix, np.float32) * (big - x)
        lin = x / (np.float32(1.0) + hr * x)
        # c = clamp(mix(l * 12.92, 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055,
        #               step(0.0031308, l)), 0.0, 1.0);
        c = np.clip(np.where(lin < np.float32(0.0031308), lin * np.float32(12.92),
                             np.float32(1.055) * lin ** np.float32(1.0 / 2.4)
                             - np.float32(0.055)), 0.0, 1.0).astype(np.float32, copy=False)

    # vec3 s = c * float(n - 1);  lo = min(floor(s), float(n - 2));  f = s - lo;
    n = lut.shape[0]
    s = c * np.float32(n - 1)
    lo = np.minimum(np.floor(s), np.float32(n - 2))
    i = lo.astype(np.intp)
    f = s - lo
    ir, ig, ib = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0, None], f[..., 1, None], f[..., 2, None]
    c00 = _mix(lut[ir, ig, ib], lut[ir + 1, ig, ib], fr)
    c10 = _mix(lut[ir, ig + 1, ib], lut[ir + 1, ig + 1, ib], fr)
    c01 = _mix(lut[ir, ig, ib + 1], lut[ir + 1, ig, ib + 1], fr)
    c11 = _mix(lut[ir, ig + 1, ib + 1], lut[ir + 1, ig + 1, ib + 1], fr)
    g = np.clip(_mix(_mix(c00, c10, fg), _mix(c01, c11, fg), fb), 0.0, 1.0)

    if tile is not None:
        # int row = (px.y + offset.y) % S;  int col = (px.x + offset.x) % S;
        oy, ox, orient = placement
        size = tile.shape[0]
        row = ((np.arange(b.shape[0]) + top + oy) % size)[:, None]
        col = ((np.arange(b.shape[1]) + ox) % size)[None, :]
        row, col = np.broadcast_arrays(row, col)
        if orient & 4:
            row, col = col, row
        if orient & 1:
            row = size - 1 - row
        if orient & 2:
            col = size - 1 - col
        # g += table(gain, dot(g, LUMA)) * texelFetch(tile, ivec2(col, row), 0).r;
        g = g + (_table(tables.gain, g @ LUMA) * tile[row, col])[..., None]

    # o = vec4(clamp(g, 0.0, 1.0), 1.0);  then the GPU's one rounding to bytes.
    return np.rint(np.clip(g, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
