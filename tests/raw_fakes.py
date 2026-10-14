"""A writer for small Picamera2-style DNGs, so the tests never need a real one.

It lays out the file the way Picamera2 does (little-endian TIFF, one IFD, one
uncompressed CFA strip, 12-bit samples packed two to three bytes) from a
uint16 Bayer array. ``tags`` replaces or, with ``None``, removes any tag, and
``strips`` splits the strip in several, so a test can build each file the
reader must refuse. The 12-bit packing here is written out pixel pair by pixel
pair, apart from the reader's vectorised unpack, so the two cannot share a
mistake.
"""

from __future__ import annotations

import struct

import numpy as np

SHORT, LONG, RATIONAL, BYTE = 3, 4, 5, 1
_SIZES = {BYTE: 1, SHORT: 2, LONG: 4, RATIONAL: 8, 10: 8}
_CFA_CODES = {"R": 0, "G": 1, "B": 2}


def pack12(bayer: np.ndarray) -> bytes:
    """Two 12-bit pixels in three bytes, TIFF order: high bits first."""
    flat = [int(v) for v in np.asarray(bayer).ravel()]
    if len(flat) % 2:
        raise ValueError("12-bit packing needs an even pixel count")
    out = bytearray()
    for a, b in zip(flat[0::2], flat[1::2], strict=True):
        out += bytes([a >> 4, ((a & 0x0F) << 4) | (b >> 8), b & 0xFF])
    return bytes(out)


def _encode(typ: int, values, endian: str) -> bytes:
    if typ == BYTE:
        return bytes(int(v) for v in values)
    if typ == SHORT:
        return struct.pack(f"{endian}{len(values)}H", *values)
    if typ == LONG:
        return struct.pack(f"{endian}{len(values)}I", *values)
    if typ in (RATIONAL, 10):
        code = "I" if typ == RATIONAL else "i"
        parts = []
        for v in values:
            # NaN is written as 0/0, the zero denominator a broken file would have.
            parts += [0, 0] if v != v else [round(v * 1_000_000), 1_000_000]
        return struct.pack(f"{endian}{2 * len(values)}{code}", *parts)
    raise ValueError(f"type {typ}")


def make_dng(
    bayer: np.ndarray,
    *,
    bits: int = 12,
    cfa: str = "BGGR",
    black=(256, 256, 256, 256),
    white: int = 4095,
    as_shot_neutral=(0.3186, 1.0, 0.6277),
    tags: dict | None = None,
    strips: int = 1,
    big_endian: bool = False,
    pixel_bytes: bytes | None = None,
) -> bytes:
    """A DNG holding ``bayer``; the keyword arguments make the faulty ones."""
    bayer = np.asarray(bayer, dtype=np.uint16)
    height, width = bayer.shape
    endian = ">" if big_endian else "<"
    if pixel_bytes is None:
        if bits == 12:
            pixel_bytes = pack12(bayer)
        elif bits == 16:
            pixel_bytes = bayer.astype(f"{endian}u2").tobytes()
        else:
            raise ValueError(f"bits {bits}")

    chunk = -(-len(pixel_bytes) // strips)
    pieces = [pixel_bytes[i : i + chunk] for i in range(0, len(pixel_bytes), chunk)]

    entries: dict[int, tuple[int, list]] = {
        256: (LONG, [width]),
        257: (LONG, [height]),
        258: (SHORT, [bits]),
        259: (SHORT, [1]),
        262: (SHORT, [32803]),
        273: (LONG, [0] * len(pieces)),
        277: (SHORT, [1]),
        279: (LONG, [len(p) for p in pieces]),
        33421: (SHORT, [2, 2]),
        33422: (BYTE, [_CFA_CODES[c] for c in cfa]),
        50713: (SHORT, [2, 2]),
        50714: (SHORT, list(black)),
        50717: (SHORT, [white]),
    }
    if as_shot_neutral is not None:
        entries[50728] = (RATIONAL, list(as_shot_neutral))
    for tag, value in (tags or {}).items():
        if value is None:
            entries.pop(tag, None)
        else:
            entries[tag] = value

    count = len(entries)
    ifd_offset = 8
    data_offset = ifd_offset + 2 + 12 * count + 4
    extra = bytearray()
    strip_offsets = []
    position = data_offset
    for piece in pieces:
        strip_offsets.append(position)
        position += len(piece)
    if 273 in entries and entries[273][1] == [0] * len(pieces):
        entries[273] = (LONG, strip_offsets)

    blobs = b"".join(pieces)
    extra_base = data_offset + len(blobs)
    ifd = bytearray(struct.pack(f"{endian}H", count))
    for tag in sorted(entries):
        typ, values = entries[tag]
        payload = _encode(typ, values, endian)
        if len(payload) <= 4:
            field = payload.ljust(4, b"\0")
        else:
            field = struct.pack(f"{endian}I", extra_base + len(extra))
            extra += payload
            if len(extra) % 2:
                extra += b"\0"
        ifd += struct.pack(f"{endian}HHI", tag, typ, len(values)) + field
    ifd += struct.pack(f"{endian}I", 0)

    magic = b"MM\0*" if big_endian else b"II*\0"
    header = magic + struct.pack(f"{endian}I", ifd_offset)
    return header + bytes(ifd) + blobs + bytes(extra)


def small_tuning(changes: dict | None = None) -> dict:
    """A tuning dict with the structure of the IMX477 file and easy numbers."""
    def grid(value: float) -> list[float]:
        return [value] * 192

    blocks = {
        "rpi.black_level": {"black_level": 4096},
        "rpi.ccm": {
            "ccms": [
                {"ct": 5000, "ccm": [2.0, -0.5, -0.5, -0.5, 2.0, -0.5, -0.5, -0.5, 2.0]},
                {"ct": 3000, "ccm": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
            ]
        },
        "rpi.alsc": {
            "luminance_strength": 0.5,
            "calibrations_Cr": [
                {"ct": 3000, "table": grid(1.0)},
                {"ct": 6000, "table": grid(2.0)},
            ],
            "calibrations_Cb": [
                {"ct": 3000, "table": grid(3.0)},
                {"ct": 6000, "table": grid(1.0)},
            ],
            "luminance_lut": [1.0 + i / 191 for i in range(192)],
        },
        "rpi.contrast": {
            "ce_enable": 1,
            "gamma_curve": [
                v for i in range(33) for v in (min(65535, i * 2048), min(65535, i * 4096))
            ],
        },
    }
    for name, value in (changes or {}).items():
        if value is None:
            blocks.pop(name, None)
        else:
            blocks[name] = value
    return {
        "version": 2.0,
        "target": "bcm2835",
        "algorithms": [{"rpi.awb": {}}] + [{name: block} for name, block in blocks.items()],
    }
