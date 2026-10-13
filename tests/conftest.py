import struct
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _state_home_under_tmp(tmp_path_factory, monkeypatch):
    """Keep every test out of the real ``~/.local/state``.

    ``pifilm-capture`` grades in the background by default and keeps its pending
    entries under ``$XDG_STATE_HOME``, so a test that runs ``main`` without
    ``--state-dir`` would otherwise read, and could write, the developer's own.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("xdg-state")))


@pytest.fixture(autouse=True)
def _temporary_files_under_pytest(tmp_path_factory, monkeypatch):
    """Keep every test's temporary files inside pytest's own tree.

    A ``CaptureSession`` makes a private ``pifilm-cards-*`` directory for its
    exposure cards and quick grades and, like the service, never removes it, so a
    run used to leave dozens behind in the system's temporary directory. pytest
    prunes its own tree.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path_factory.mktemp("tmp")))


def _s15f16(x: float) -> int:
    """ICC s15Fixed16Number."""
    return int(round(x * 65536.0))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + b"\0" * 4 + struct.pack(">3i", _s15f16(x), _s15f16(y), _s15f16(z))


def _curv_tag(gamma: float) -> bytes:
    """A one-entry curve tag, which ICC reads as a plain gamma value."""
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256)))


def _desc_tag(text: str) -> bytes:
    body = text.encode("ascii") + b"\0"
    return b"desc" + b"\0" * 4 + struct.pack(">I", len(body)) + body + b"\0" * 88


def build_rgb_icc_profile(description: str, gamma: float, primaries) -> bytes:
    """A minimal valid ICC v2 RGB matrix-shaper profile, built in memory.

    Enough for littlecms to construct a transform: white point, three
    colourant XYZ tags, three tone curves, a description and a copyright.
    """
    tags = {
        b"desc": _desc_tag(description),
        b"wtpt": _xyz_tag(0.9642, 1.0, 0.8249),  # D50, which ICC requires
        b"rXYZ": _xyz_tag(*primaries[0]),
        b"gXYZ": _xyz_tag(*primaries[1]),
        b"bXYZ": _xyz_tag(*primaries[2]),
        b"rTRC": _curv_tag(gamma),
        b"gTRC": _curv_tag(gamma),
        b"bTRC": _curv_tag(gamma),
        b"cprt": _desc_tag("public domain"),
    }
    offset = 128 + 4 + len(tags) * 12
    table, data = b"", b""
    for sig, payload in tags.items():
        padding = (-len(payload)) % 4
        table += sig + struct.pack(">II", offset, len(payload))
        data += payload + b"\0" * padding
        offset += len(payload) + padding
    body = struct.pack(">I", len(tags)) + table + data

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + len(body))  # total size
    header[4:8] = b"none"                                # preferred CMM
    header[8:12] = struct.pack(">I", 0x02100000)         # version 2.1
    header[12:16] = b"mntr"                              # display device class
    header[16:20] = b"RGB "                              # data colour space
    header[20:24] = b"XYZ "                              # profile connection space
    struct.pack_into(">6H", header, 24, 2026, 9, 3, 0, 0, 0)
    header[36:40] = b"acsp"                              # required signature
    header[64:68] = struct.pack(">I", 0)                 # perceptual intent
    header[68:80] = struct.pack(">3i", _s15f16(0.9642), _s15f16(1.0), _s15f16(0.8249))
    return bytes(header) + body


@pytest.fixture(scope="session")
def wide_gamut_icc() -> bytes:
    """Adobe RGB (1998) primaries adapted to D50, gamma 2.2."""
    return build_rgb_icc_profile(
        "Test Wide Gamut RGB",
        2.2,
        [
            (0.6097, 0.3111, 0.0195),
            (0.2053, 0.6257, 0.0609),
            (0.1492, 0.0632, 0.7448),
        ],
    )
