"""The camera's raw frame: a DNG reader and the tables of the camera's tuning file.

This is the first half of the DNG developer (the raw look, phase 1). A look
trained on the DNG needs the sensor's own numbers, before the ISP has chosen a
tone curve or clipped a highlight, so this module reads them, and the parts of
the libcamera tuning file that say how the ISP would have corrected them.

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

    offsets, counts = fields[273], fields[279]
    if len(offsets) != 1 or len(counts) != 1:
        raise RawError(f"{len(offsets)} strips; Picamera2 writes one")
    width, height = int(_one(fields, 256)), int(_one(fields, 257))
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
    return RawFrame(bayer=bayer, cfa=cfa, black=black, white=white, as_shot_neutral=neutral)


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
