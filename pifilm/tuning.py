"""Which libcamera tuning the camera runs with, as a fingerprint, and whether a look agrees.

A trained look is fitted to the colours the camera produced under one tuning
file. An OS or libcamera update can replace that file, and every photograph
then shifts with no error and no changed ``lut_sha1``. A look can carry the
fingerprint of the tuning it was trained under (``tuning_sha1`` in
``params.json``), the camera fingerprints the tuning it opens with, and
``check`` turns the comparison into one log line. Warn only: a camera that will
not shoot after an update in the field is worse than colours that can be
regraded from the kept originals.

**The fingerprint is of the parsed JSON, not the file's bytes.** It is the SHA-1
of the dict serialised with sorted keys and no whitespace, so reformatting the
file or reordering its keys is not a change, and any changed value is.

**Which tuning.** With ``--tuning-file NAME`` it is the very dict the camera is
opened with (``loaded``), not a second read that could find another file on the
search path. Without it, the camera runs with the sensor's own default file,
``<model>.json`` from libcamera's tuning directories, so that is what is loaded
and fingerprinted. ``LIBCAMERA_RPI_TUNING_FILE`` is ignored on purpose:
Picamera2 (0.3.x) removes it from the environment when it opens a camera
without a tuning, so the camera never uses it, and honouring it here would make
``pifilm-tuning`` and the service disagree. A failure in the automatic branch
is a reason, never an exception: the camera opens anyway. A failure loading a
given name propagates, because the camera could not open with it either.

Nothing here imports Picamera2; the loader is passed in, so this runs and tests
on a Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from .artifacts import ArtifactsError, validate_tuning_sha1


def fingerprint(obj: object) -> str:
    """Forty lowercase hex digits identifying a parsed tuning."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_argument(text: str) -> str:
    """An argparse ``type`` for a ``tuning_sha1`` given on a command line, checked
    exactly as the artifact loader checks the field."""
    try:
        return validate_tuning_sha1(text, "the value")
    except ArtifactsError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def short(sha1: str) -> str:
    """The first 12 hex digits, for log lines."""
    return sha1[:12]


@dataclass(frozen=True)
class Resolved:
    """The tuning in use, as far as it could be fingerprinted."""

    label: str            # what a record's tuning_file says: the name given, or "auto:<model>"
    source: str | None    # where the dict came from (a path or a name), for the CLI and the log
    sha1: str | None      # None when it could not be fingerprinted
    reason: str | None    # why not, when sha1 is None


def _phrase(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def resolve(tuning_file: str | None, model: str, *,
            load: Callable[[str], dict], loaded: dict | None = None) -> Resolved:
    """Fingerprint the tuning the camera uses (see the module docstring for the order)."""
    if tuning_file is not None:
        tuning = loaded if loaded is not None else load(tuning_file)
        return Resolved(label=tuning_file, source=tuning_file,
                        sha1=fingerprint(tuning), reason=None)

    label = f"auto:{model}"
    source = f"{model.lower()}.json"
    try:
        tuning = load(source)
    except Exception as exc:  # any loader failure: the camera still opens
        return Resolved(label, source, None, f"cannot load {source}: {_phrase(exc)}")
    try:
        sha1 = fingerprint(tuning)
    except (TypeError, ValueError) as exc:
        return Resolved(label, source, None, f"cannot fingerprint {source}: {_phrase(exc)}")
    return Resolved(label, source, sha1, None)


def check(look_sha1: str | None, camera: Resolved | None) -> tuple[bool | None, str | None]:
    """``(tuning_matches, log line)``; ``(None, None)`` when there is nothing to check."""
    if look_sha1 is None or camera is None:
        return None, None
    if camera.sha1 is None:
        return None, f"tuning: cannot check the look's tuning ({camera.reason})"
    if camera.sha1 == look_sha1:
        return True, f"tuning: matches the look ({short(look_sha1)})"
    return False, (
        f"tuning: DIFFERS from the look: camera {short(camera.sha1)} ({camera.label}), "
        f"look {short(look_sha1)}; colours may be off"
    )
