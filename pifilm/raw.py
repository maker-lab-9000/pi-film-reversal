"""The camera's raw frame: a DNG reader and the tables of the camera's tuning file.

The DNG developer (the raw look, phase 1). A look trained on the DNG needs
the sensor's own numbers, before the ISP has chosen a tone curve or clipped a
highlight, so this module reads them, the parts of the libcamera tuning file
that say how the ISP would have corrected them, and ``develop`` applies those
corrections in the ISP's order to give an 8-bit sRGB picture close to the
camera's own JPEG (on an IMX477 room shot, within 4 levels on average).

Why our own reader
------------------
Picamera2 writes one kind of DNG: little-endian TIFF, one IFD, the Bayer
samples in one uncompressed strip, 12 bits packed two pixels to three bytes
(or plain 16-bit). Reading that takes a page of ``struct`` and NumPy, where a
general raw library would be a new dependency on the Pi and the Mac for one
file layout. So the reader accepts that layout and nothing else: a big-endian
file, compression, tiles, several strips, another bit depth or a missing tag
is a ``RawError`` that names what it found, never a guess. A value above the
white level is refused too; it is how a wrong unpack shows itself.

The 12-bit order is TIFF's: the first pixel is the first byte and the high
half of the second (``b0 << 4 | b1 >> 4``), the second pixel the low half of
the second byte and the third (``(b1 & 0x0F) << 8 | b2``). A swapped nibble
gives a picture that looks almost right, so the reader was checked against
LibRaw on a real IMX477 DNG (equal in every pixel) and a test pins the order
on three hand-made bytes.

The CFA pattern is the top-left 2 x 2 read row by row, as the DNG states it
(0 red, 1 green, 2 blue), so ``"BGGR"`` means blue at (0, 0) and red at
(1, 1). Converting that to an OpenCV demosaic code is the developer's job,
because OpenCV names its codes by a different corner.

Why the tuning file
-------------------
The ISP corrects the frame with tables from the camera's tuning file, chosen
by the colour temperature its AWB measured: lens shading (``rpi.alsc``), the
colour matrix (``rpi.ccm``) and a tone curve (``rpi.contrast``). Using the same
tables makes the developed DNG close to the camera's JPEG, which is the base
the look is trained from. ``tuning_tables`` takes the parsed JSON (never a
path) and checks every table's size, so a wrong or partial file is refused at
load, not halfway through a frame.

The colour matrices and the lens-shading grids are calibrated at a few colour
temperatures. ``ccm_at`` and ``alsc_at`` interpolate linearly in kelvin
between the two neighbours, as libcamera does, and clamp outside the range:
extrapolating a colour matrix past its last calibration can give any colour.

The luminance table is a vignetting gain on all channels that the ISP applies
only in part: ``luminance_strength`` (0.5 on the IMX477) moves each gain
towards 1 as ``1 + (lut - 1) * strength``. That is the formula of libcamera's
``src/ipa/rpi/controller/rpi/alsc.cpp`` (raspberrypi/libcamera at
c682979b, ``addLuminanceRb`` and ``addLuminanceG``, lines 792 and 800), and a
missing ``luminance_strength`` is 1, as it is there. The grids are 16 columns
by 12 rows over the whole frame, stored row by row.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


class RawError(ValueError):
    """A DNG or tuning file this module does not read."""


@dataclass(frozen=True)
class RawFrame:
    """The Bayer samples of one DNG and the numbers needed to scale them."""

    bayer: np.ndarray  # (H, W) uint16, values 0..white
    cfa: str  # "BGGR", "RGGB", "GRBG" or "GBRG", top-left 2x2 read row by row
    black: float  # one black level (refused when the four differ)
    white: float
    as_shot_neutral: tuple[float, float, float] | None
    bits: int | None = None  # bits per sample in the file; None when not known


# TIFF field types: (struct code, bytes per value).
_TYPES = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    5: ("II", 8),
    6: ("b", 1),
    7: ("B", 1),
    8: ("h", 2),
    9: ("i", 4),
    10: ("ii", 8),
    11: ("f", 4),
    12: ("d", 8),
}

_TAG_NAMES = {
    256: "ImageWidth",
    257: "ImageLength",
    258: "BitsPerSample",
    259: "Compression",
    262: "PhotometricInterpretation",
    273: "StripOffsets",
    277: "SamplesPerPixel",
    279: "StripByteCounts",
    33421: "CFARepeatPatternDim",
    33422: "CFAPattern",
    50714: "BlackLevel",
    50717: "WhiteLevel",
}
_AS_SHOT_NEUTRAL = 50728
_TILE_TAGS = (322, 323, 324, 325)
# Tags that would change what the samples mean, which this reader does not
# apply: a linearisation curve, an active area smaller than the frame.
_UNREAD_TAGS = {50712: "LinearizationTable", 50829: "ActiveArea"}
_ORIENTATION = 274
_DEFAULT_CROP = (50719, 50720)  # origin, size
_BLACK_DELTAS = (50715, 50716)
_CFA_COLOURS = "RGB"
_PATTERNS = ("BGGR", "RGGB", "GRBG", "GBRG")


def _read_ifd(data: bytes) -> dict[int, tuple]:
    """The first IFD's fields as {tag: values}, rationals as floats."""
    if len(data) < 8:
        raise RawError(f"not a TIFF file: {len(data)} bytes")
    if data[:4] == b"MM\0*":
        raise RawError("big-endian TIFF; Picamera2 writes little-endian")
    if data[:4] != b"II*\0":
        raise RawError(f"not a TIFF file: header {data[:4]!r}")
    (offset,) = struct.unpack_from("<I", data, 4)
    if offset + 2 > len(data):
        raise RawError(f"IFD offset {offset} is past the end of the file ({len(data)} bytes)")
    (count,) = struct.unpack_from("<H", data, offset)
    if offset + 2 + 12 * count > len(data):
        raise RawError(f"IFD of {count} entries runs past the end of the file")
    fields: dict[int, tuple] = {}
    for i in range(count):
        tag, typ, n, _ = struct.unpack_from("<HHII", data, offset + 2 + 12 * i)
        if typ not in _TYPES:
            continue
        code, size = _TYPES[typ]
        start = offset + 2 + 12 * i + 8
        if size * n > 4:
            (start,) = struct.unpack_from("<I", data, start)
        if start + size * n > len(data):
            raise RawError(f"tag {tag} runs past the end of the file")
        values = struct.unpack_from(f"<{n * len(code)}{code[0]}", data, start)
        if typ in (5, 10):
            pairs = zip(values[0::2], values[1::2], strict=True)
            values = tuple(a / b if b else float("nan") for a, b in pairs)
        fields[tag] = values
    return fields


def _one(fields: dict[int, tuple], tag: int) -> float:
    values = fields[tag]
    if len(values) != 1:
        raise RawError(f"{_TAG_NAMES[tag]} has {len(values)} values, expected 1")
    return values[0]


def _unpack12(strip: bytes, count: int) -> np.ndarray:
    """Two pixels from every three bytes, TIFF order, without a Python loop."""
    triples = np.frombuffer(strip, dtype=np.uint8).reshape(-1, 3).astype(np.uint16)
    out = np.empty(count, dtype=np.uint16)
    out[0::2] = (triples[:, 0] << 4) | (triples[:, 1] >> 4)
    out[1::2] = ((triples[:, 1] & 0x0F) << 8) | triples[:, 2]
    return out


def read_dng(data: bytes) -> RawFrame:
    """Parse a Picamera2 DNG: little-endian TIFF, one IFD, uncompressed CFA strip,
    12- or 16-bit. Anything else (big-endian, compressed, tiles, several strips,
    other bit depths, missing tags) raises RawError naming what was found."""
    fields = _read_ifd(data)
    missing = [name for tag, name in _TAG_NAMES.items() if tag not in fields]
    if missing:
        raise RawError(f"missing DNG tags: {', '.join(missing)}")
    if any(tag in fields for tag in _TILE_TAGS):
        raise RawError("tiled image; Picamera2 writes one strip")
    for tag, name in _UNREAD_TAGS.items():
        if tag in fields:
            raise RawError(f"{name} (tag {tag}) is not applied by this reader")
    if _ORIENTATION in fields and tuple(fields[_ORIENTATION]) != (1,):
        raise RawError(f"orientation {fields[_ORIENTATION][0]}; only 1 (as stored) is read")

    compression = _one(fields, 259)
    if compression != 1:
        raise RawError(f"compression {compression}; only uncompressed (1) is read")
    photometric = _one(fields, 262)
    if photometric != 32803:
        raise RawError(f"photometric {photometric}; only CFA (32803) is read")
    samples = _one(fields, 277)
    if samples != 1:
        raise RawError(f"samples per pixel {samples}; a CFA image has 1")
    bits = _one(fields, 258)
    if bits not in (12, 16):
        raise RawError(f"{bits} bits per sample; only 12 and 16 are read")

    if tuple(fields[33421]) != (2, 2):
        raise RawError(f"CFARepeatPatternDim {tuple(fields[33421])}; only 2 x 2 is read")
    codes = fields[33422]
    if len(codes) != 4 or any(c not in (0, 1, 2) for c in codes):
        raise RawError(f"CFA pattern codes {tuple(codes)}; expected four of 0, 1, 2")
    cfa = "".join(_CFA_COLOURS[c] for c in codes)
    if cfa not in _PATTERNS:
        raise RawError(f"CFA pattern {cfa}; only {', '.join(_PATTERNS)} are read")

    blacks = set(fields[50714])
    if len(blacks) != 1:
        raise RawError(f"unequal black levels {tuple(fields[50714])}; one level is expected")
    for tag in _BLACK_DELTAS:
        if tag in fields and any(fields[tag]):
            raise RawError(f"black level deltas (tag {tag}) are not read")
    black = float(blacks.pop())
    white = float(_one(fields, 50717))
    # A rational with a zero denominator reads as NaN, and every comparison with
    # NaN is false, so it would slip past the checks below without this one.
    if not (np.isfinite(black) and np.isfinite(white)):
        raise RawError(f"black level {black:g} or white level {white:g} is not finite")
    if black >= white:
        raise RawError(f"black level {black:g} is not below the white level {white:g}")

    offsets, counts = fields[273], fields[279]
    if len(offsets) != 1 or len(counts) != 1:
        raise RawError(f"{len(offsets)} strips; Picamera2 writes one")
    width, height = int(_one(fields, 256)), int(_one(fields, 257))
    origin, size = (tuple(fields.get(tag, ())) for tag in _DEFAULT_CROP)
    if (origin and origin != (0, 0)) or (size and size != (width, height)):
        raise RawError(
            f"DefaultCrop origin {origin or '-'} size {size or '-'}; only the full "
            f"{width} x {height} frame is read"
        )
    if bits == 12 and width % 2:
        raise RawError(f"width {width} is odd; 12-bit rows must pack whole pixel pairs")
    expected = width * height * bits // 8
    if counts[0] != expected:
        raise RawError(
            f"strip holds {counts[0]} bytes, expected {expected} "
            f"for {width} x {height} at {bits} bits"
        )
    start = offsets[0]
    if start + expected > len(data):
        raise RawError(f"strip at {start} of {expected} bytes runs past the end of the file")
    strip = data[start : start + expected]

    if bits == 12:
        flat = _unpack12(strip, width * height)
    else:
        flat = np.frombuffer(strip, dtype="<u2").astype(np.uint16)
    bayer = flat.reshape(height, width)
    top = int(bayer.max()) if bayer.size else 0
    if top > white:
        raise RawError(f"value {top} is above the white level {white:g}; a wrong unpack?")

    neutral = None
    if _AS_SHOT_NEUTRAL in fields:
        values = fields[_AS_SHOT_NEUTRAL]
        if len(values) != 3:
            raise RawError(f"AsShotNeutral has {len(values)} values, expected 3")
        neutral = tuple(float(v) for v in values)
    return RawFrame(
        bayer=bayer, cfa=cfa, black=black, white=white, as_shot_neutral=neutral, bits=int(bits)
    )


_GRID = (12, 16)  # rows, columns of the lens-shading tables


@dataclass(frozen=True)
class RawTuning:
    """The tables of a libcamera tuning file that the developer uses."""

    black_level: float  # at 16 bits, as in the file
    ccms: tuple[tuple[float, np.ndarray], ...]  # (ct, 3x3) sorted by ct
    alsc_cr: tuple[tuple[float, np.ndarray], ...]  # (ct, (12, 16)) sorted by ct
    alsc_cb: tuple[tuple[float, np.ndarray], ...]
    luminance: np.ndarray  # (12, 16), luminance_lut with luminance_strength applied
    gamma_curve: np.ndarray  # (33, 2) in 0..1, x then y


def _blocks(tuning: dict) -> dict:
    """The algorithm blocks by name: version 2 lists them, version 1 has them at the top."""
    if not isinstance(tuning, dict):
        raise RawError(f"tuning is a {type(tuning).__name__}, expected a dict")
    algorithms = tuning.get("algorithms")
    if algorithms is None:
        return tuning
    if not isinstance(algorithms, list):
        raise RawError("tuning 'algorithms' is not a list")
    blocks: dict = {}
    for entry in algorithms:
        if isinstance(entry, dict):
            blocks.update(entry)
    return blocks


def _block(blocks: dict, name: str) -> dict:
    block = blocks.get(name)
    if not isinstance(block, dict):
        raise RawError(f"tuning has no {name} block")
    return block


def _floats(values, size: int, what: str) -> np.ndarray:
    if not isinstance(values, list | tuple):
        raise RawError(f"{what} is missing or not a list")
    if len(values) != size:
        raise RawError(f"{what} has {len(values)} values, expected {size}")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RawError(f"{what} holds a value that is not a number") from exc
    if array.ndim != 1:
        raise RawError(f"{what} is not a flat list of numbers")
    if not np.all(np.isfinite(array)):
        raise RawError(f"{what} holds a value that is not finite")
    return array


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _calibrations(entries, key: str, size: int, shape: tuple, what: str):
    """``[{"ct": K, key: [...]}, ...]`` as ((ct, table), ...) sorted by ct."""
    if not isinstance(entries, list) or not entries:
        raise RawError(f"{what} has no calibrations")
    tables = []
    for entry in entries:
        if not isinstance(entry, dict) or "ct" not in entry:
            raise RawError(f"{what} has an entry without a ct")
        ct = float(_floats([entry["ct"]], 1, f"{what} ct")[0])
        table = _floats(entry.get(key), size, f"{what} {key} at {ct:g} K")
        tables.append((ct, _frozen(table.reshape(shape))))
    tables.sort(key=lambda pair: pair[0])
    cts = [ct for ct, _ in tables]
    for a, b in zip(cts, cts[1:], strict=False):
        if a == b:
            raise RawError(f"{what} has two calibrations at {a:g} K")
    return tuple(tables)


def tuning_tables(tuning: dict) -> RawTuning:
    """The developer's tables from a parsed tuning file; RawError when one is missing
    or malformed."""
    blocks = _blocks(tuning)

    black = _block(blocks, "rpi.black_level").get("black_level")
    black_level = float(_floats([black], 1, "rpi.black_level black_level")[0])

    ccms = _calibrations(_block(blocks, "rpi.ccm").get("ccms"), "ccm", 9, (3, 3), "rpi.ccm")

    alsc = _block(blocks, "rpi.alsc")
    cells = _GRID[0] * _GRID[1]
    alsc_cr = _calibrations(alsc.get("calibrations_Cr"), "table", cells, _GRID, "rpi.alsc Cr")
    alsc_cb = _calibrations(alsc.get("calibrations_Cb"), "table", cells, _GRID, "rpi.alsc Cb")
    lut = _floats(alsc.get("luminance_lut"), cells, "rpi.alsc luminance_lut").reshape(_GRID)
    strength = float(_floats([alsc.get("luminance_strength", 1.0)], 1, "luminance_strength")[0])
    # libcamera alsc.cpp: lambda * ((luminanceLut[i] - 1) * luminanceStrength + 1).
    luminance = 1.0 + (lut - 1.0) * strength

    curve = _block(blocks, "rpi.contrast").get("gamma_curve")
    if not isinstance(curve, list) or len(curve) < 4 or len(curve) % 2:
        size = len(curve) if isinstance(curve, list) else "no"
        raise RawError(f"rpi.contrast gamma_curve has {size} values, expected (x, y) pairs")
    gamma = _floats(curve, len(curve), "rpi.contrast gamma_curve").reshape(-1, 2) / 65535.0
    if np.any(np.diff(gamma[:, 0]) <= 0):
        raise RawError("rpi.contrast gamma_curve x values are not increasing")

    return RawTuning(
        black_level=black_level,
        ccms=ccms,
        alsc_cr=alsc_cr,
        alsc_cb=alsc_cb,
        luminance=_frozen(luminance),
        gamma_curve=_frozen(gamma),
    )


def _interpolate(tables: tuple[tuple[float, np.ndarray], ...], ct: float) -> np.ndarray:
    """Linear in ct between the two neighbours; the end table outside the range."""
    if not np.isfinite(ct):
        raise RawError(f"colour temperature {ct} is not a finite number")
    if ct <= tables[0][0]:
        return tables[0][1].copy()
    if ct >= tables[-1][0]:
        return tables[-1][1].copy()
    for (ct0, t0), (ct1, t1) in zip(tables, tables[1:], strict=False):
        if ct <= ct1:
            w = (ct - ct0) / (ct1 - ct0)
            return (1.0 - w) * t0 + w * t1
    raise AssertionError("unreachable")  # pragma: no cover


def ccm_at(t: RawTuning, ct: float) -> np.ndarray:
    """The 3 x 3 colour matrix at colour temperature ``ct`` (K), clamped at the ends."""
    return _interpolate(t.ccms, float(ct))


def alsc_at(t: RawTuning, ct: float) -> tuple[np.ndarray, np.ndarray]:
    """The (cr, cb) lens-shading grids at colour temperature ``ct`` (K), clamped at the ends."""
    return _interpolate(t.alsc_cr, float(ct)), _interpolate(t.alsc_cb, float(ct))


# --- the developer -------------------------------------------------------

TONE_CURVES = ("isp", "soft-1", "soft-1.5")

# Rows per band for the colour matrix and the tone curve, so the float picture
# is never held at full size (the output aside).
_BAND_ROWS = 256

# OpenCV names a Bayer code after the second and third pixels of its second
# row, not after the top-left 2 x 2 that the DNG gives, so the four names are
# shifted by one site diagonally: a frame whose top-left is R G / G B
# ("RGGB") is OpenCV's "BayerBG". This is the one place the mapping lives;
# a test develops a frame of colour patches through each pattern to prove it.
_OPENCV_BAYER = {
    "RGGB": "COLOR_BayerBG2RGB",
    "BGGR": "COLOR_BayerRG2RGB",
    "GRBG": "COLOR_BayerGB2RGB",
    "GBRG": "COLOR_BayerGR2RGB",
}
_SITE_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _sites(cfa: str, colour: str) -> list[tuple[int, int]]:
    """The (row, column) offsets in the 2 x 2 tile where ``colour`` sits."""
    return [offset for offset, c in zip(_SITE_OFFSETS, cfa, strict=True) if c == colour]


def _linear(frame: RawFrame) -> np.ndarray:
    """The Bayer plane as float32 linear light, black at 0 and white at 1."""
    plane = frame.bayer.astype(np.float32)
    plane -= np.float32(frame.black)
    plane *= np.float32(1.0 / (frame.white - frame.black))
    np.clip(plane, 0.0, 1.0, out=plane)
    return plane


def _upsample(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A (12, 16) grid at full size, bilinear between cell centres.

    ``cv2.resize`` with ``INTER_LINEAR`` puts the centre of cell ``i`` at
    ``(i + 0.5) * W / 16``, the centres of 16 equal divisions of the frame,
    and holds the edge cells' values out to the frame's edges.
    """
    cv2 = _require_cv2()
    h, w = shape
    return cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _shade(plane: np.ndarray, cfa: str, cr: np.ndarray, cb: np.ndarray, lum: np.ndarray) -> None:
    """Lens shading in place: red sites x cr x lum, blue x cb x lum, green x lum.

    The calibrated Cr and Cb tables are the R/G and B/G of a flat field, so
    they carry that light's colour (about 2.9 and 1.5 on the IMX477) as well as
    the shading. libcamera divides each by its smallest value before use
    (``compensateLambdasForCal`` in ``alsc.cpp``, lines 572 to 583, with the
    adaptive lambdas at 1 as on a table reset), leaving only the shading; the
    colour is the white balance's job. So do we. Without it the red sites
    would get the red gain twice and the picture would come out magenta. The
    luminance table is not divided: ``normalise`` there (line 742) discards
    its result, so the ISP applies the table as it is, and so do we.

    The grids are upsampled one at a time and dropped, so this holds at most
    two full-size gain planes beside the frame.
    """
    gain = _upsample(lum, plane.shape)
    for colour, grid in (("R", cr), ("B", cb)):
        colour_gain = _upsample(grid / grid.min(), plane.shape)
        for r, c in _sites(cfa, colour):
            gain[r::2, c::2] *= colour_gain[r::2, c::2]
        del colour_gain
    plane *= gain


def _white_balance(
    plane: np.ndarray, cfa: str, gains: tuple[float, float], digital_gain: float = 1.0
) -> None:
    """The shot's digital gain and ColourGains in place, then a clip to 1.

    The digital gain is AGC's: the ISP multiplies every site by it after the
    sensor's samples (``applyDG`` in libcamera's vc4 IPA), so the DNG does not
    carry it. It is 1.000 to 1.043 on the training photographs. It comes
    before the clip like the colour gains, so a site it pushes past 1 clips
    with the rest.

    The clip is what keeps a clipped highlight white. A pixel at the white
    level is 1 in all three channels; the red and blue gains (about 3 and 1.6
    in daylight) lift those sites past 1 while green stays at 1, and without
    the clip the colour matrix would turn that excess into a magenta cast.
    Clipped to 1 here, before the matrix, it is a neutral (1, 1, 1), and a
    matrix whose rows sum to 1 keeps it there.
    """
    red, blue = gains
    if digital_gain != 1.0:
        plane *= np.float32(digital_gain)
    for colour, gain in (("R", red), ("B", blue)):
        for r, c in _sites(cfa, colour):
            plane[r::2, c::2] *= np.float32(gain)
    np.minimum(plane, 1.0, out=plane)


def _demosaic(plane: np.ndarray, cfa: str) -> np.ndarray:
    """(H, W, 3) uint16 RGB, OpenCV's bilinear demosaic at 16 bits."""
    cv2 = _require_cv2()
    mosaic = np.rint(plane * np.float32(65535.0)).astype(np.uint16)
    return cv2.cvtColor(mosaic, getattr(cv2, _OPENCV_BAYER[cfa]))


def _colour(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """The colour matrix on linear camera RGB (..., 3), clipped to 0..1."""
    out = rgb @ ccm.T.astype(np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _soft_stops(tone: str) -> float | None:
    """None for ``"isp"``, the lift in stops for ``"soft-<e>"``; ValueError otherwise."""
    if tone == "isp":
        return None
    if tone.startswith("soft-"):
        try:
            stops = float(tone[len("soft-") :])
        except ValueError:
            stops = float("nan")
        if np.isfinite(stops) and stops > 0:
            return stops
    raise ValueError(f"unknown tone {tone!r}; expected 'isp' or 'soft-<stops>', as {TONE_CURVES}")


def _tone(x: np.ndarray, curve: np.ndarray, tone: str) -> np.ndarray:
    """Linear 0..1 to display 0..1 (float32).

    ``"isp"`` is the tuning's gamma curve, linear between its points: the
    curve the ISP encodes the camera's JPEG with, so its output is taken as
    sRGB as it stands. ``"soft-<e>"`` first maps ``x`` to
    ``x * 2**e / (1 + x * (2**e - 1))``, which keeps 0 and 1, lifts the
    shadows by ``e`` stops and rolls the highlights off, then applies the same
    curve: a flatter base for a look to work on.
    """
    stops = _soft_stops(tone)
    if stops is not None:
        k = np.float32(2.0**stops)
        x = x * k / (np.float32(1.0) + x * (k - np.float32(1.0)))
    out = np.interp(x, curve[:, 0], curve[:, 1]).astype(np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _to_u8(x: np.ndarray) -> np.ndarray:
    """Display 0..1 float32 to uint8, rounded to the nearest level."""
    return np.rint(x * np.float32(255.0)).astype(np.uint8)


def develop(
    frame: RawFrame,
    tuning: RawTuning,
    *,
    colour_gains: tuple[float, float],
    colour_temperature: float,
    tone: str = "isp",
    as_float: bool = False,
    digital_gain: float = 1.0,
) -> np.ndarray:
    """(H, W, 3) uint8 sRGB. colour_gains is the record's ColourGains (red, blue).

    The steps run in the ISP's order: black level, lens shading, digital gain
    and white balance, demosaic, colour matrix, tone curve. The lens-shading
    grids and the colour matrix are taken at ``colour_temperature`` (the
    record's ColourTemperature), as the ISP chose them for the shot;
    ``digital_gain`` is the record's DigitalGain.

    Two checks catch a frame paired with the wrong inputs. When the DNG has an
    AsShotNeutral, its reciprocal (relative to green) must be the given red and
    blue gains within 0.5 %: Picamera2 writes it from the same ColourGains, so
    a mismatch means the DNG and the record are not the same shot. When the
    frame's bit depth is known to be 12, the tuning's black level (in 16-bit
    units) scaled to 12 bits must be the frame's within 1 unit: another
    sensor's tuning fails here.

    Limits. Full-field sensor modes only: the lens-shading grids are stretched
    over the whole frame, where libcamera resamples them for a cropped mode
    and mirrors them for a flipped one; neither is done here. The ISP lays its
    grid in 128-pixel cells over 4096 x 3072 (a little past the 4056 x 3040
    frame) and stores the gains as u4.10 fixed point; the stretched grid and
    float gains differ from that by much less than a level, a known and
    negligible difference.

    With ``as_float`` the picture is returned as float32 in 0..1 after the
    tone curve, before rounding: ``np.rint(x * 255)`` of it is the uint8
    output exactly.

    Memory: at 12 MP each full-size float32 plane is 49 MB. The developer holds
    the plane and two gain planes during lens shading, then the 16-bit
    demosaiced picture (73 MB) and the output; the float RGB stage runs in
    bands of rows, so it is never held at full size.
    """
    _soft_stops(tone)  # an unknown tone fails before any work
    gains = tuple(float(g) for g in colour_gains)
    if len(gains) != 2 or not all(np.isfinite(g) and g > 0 for g in gains):
        raise RawError(f"colour gains {tuple(colour_gains)} must be two positive numbers")
    if frame.cfa not in _OPENCV_BAYER:
        raise RawError(f"CFA pattern {frame.cfa!r} is not one of {', '.join(_PATTERNS)}")
    digital_gain = float(digital_gain)
    if not (np.isfinite(digital_gain) and digital_gain > 0):
        raise RawError(f"digital gain {digital_gain} must be a positive number")
    if frame.as_shot_neutral is not None:
        n_r, n_g, n_b = frame.as_shot_neutral
        for name, neutral, gain in (("red", n_r, gains[0]), ("blue", n_b, gains[1])):
            implied = n_g / neutral if neutral > 0 else math.inf
            if not abs(implied / gain - 1.0) <= 0.005:
                raise RawError(
                    f"AsShotNeutral implies a {name} gain of {implied:.4f}, the record "
                    f"{gain:.4f}: is the DNG paired with the right record?"
                )
    if frame.bits == 12:
        expected = tuning.black_level / 16.0
        if abs(expected - frame.black) > 1.0:
            raise RawError(
                f"tuning black level {tuning.black_level:g} is {expected:g} at 12 bits, "
                f"the frame's {frame.black:g}: a tuning file for another sensor?"
            )
    ccm = ccm_at(tuning, colour_temperature)
    cr, cb = alsc_at(tuning, colour_temperature)

    plane = _linear(frame)
    _shade(plane, frame.cfa, cr, cb, tuning.luminance)
    _white_balance(plane, frame.cfa, gains, digital_gain)
    rgb16 = _demosaic(plane, frame.cfa)
    del plane

    h, w, _ = rgb16.shape
    out = np.empty((h, w, 3), dtype=np.float32 if as_float else np.uint8)
    scale = np.float32(1.0 / 65535.0)
    for top in range(0, h, _BAND_ROWS):
        band = rgb16[top : top + _BAND_ROWS].astype(np.float32) * scale
        toned = _tone(_colour(band, ccm), tuning.gamma_curve, tone)
        out[top : top + _BAND_ROWS] = toned if as_float else _to_u8(toned)
    return out


def _require_cv2():
    # Imported here so the reader and the tuning tables work without OpenCV.
    from pifilm._cv2 import require_cv2

    return require_cv2()
