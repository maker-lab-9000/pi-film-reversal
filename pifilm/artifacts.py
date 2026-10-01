"""Loading, validating and publishing the trained artifact.

An artifact is a `.cube` LUT plus `params.json` holding the normalisation the
LUT was fitted against, the grain settings, and training provenance. Three
concerns live here.

**Where the default comes from.** The shipped look is package data at
``pifilm/data/``, found with ``importlib.resources``. That makes every
command work from any working directory and puts the look inside a built
wheel. ``--artifacts DIR`` overrides it with a directory on disk.

**Validating rather than trusting.** The JSON root must be an object with a
known version; each section must be an object; the LUT file must exist,
parse, and hash to the ``lut_sha1`` recorded beside it. That last check is
what makes a mixed artifact impossible to load: if a training run wrote a
new LUT and then failed before rewriting ``params.json``, the hashes
disagree and loading fails loudly instead of grading with mismatched
parameters.

**Publishing atomically.** The trainer writes everything into a staging
directory, this module loads and validates that directory, and only then is
it moved into place with ``os.replace`` on the directory. A capture process
reading concurrently sees either the whole old artifact or the whole new
one.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .grain import GrainParams
from .lut import LUT3D, CubeError, read_cube, sha1_hex, write_cube
from .normalize import NormalizeParams

PARAMS_VERSION = 3
# Version 3 exists only to carry a non-Gaussian grain model: an older build ignores
# the extra grain fields and would grain at the Gaussian model's settings, so such
# an artifact must be one it refuses. Gaussian-grain artifacts stay at version 2.
GRAIN_MODEL_VERSION = 3
DEFAULT_LUT_FILE = "pifilm.cube"


class ArtifactsError(Exception):
    """Artifact directory missing, incomplete or invalid."""


def _require_object(value: object, name: str, path: Path) -> dict:
    if not isinstance(value, dict):
        raise ArtifactsError(f"{path}: '{name}' must be a JSON object, got {type(value).__name__}")
    return value


@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    training: dict
    path: Path
    lut_sha1: str

    @classmethod
    def load(cls, dir_path: str | Path) -> Artifacts:
        path = Path(dir_path)
        params_path = path / "params.json"
        if not params_path.is_file():
            raise ArtifactsError(
                f"{params_path} not found. Run pifilm-train, pass --artifacts DIR, "
                "or reinstall the package to restore the bundled default."
            )
        try:
            raw = json.loads(params_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactsError(f"{params_path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ArtifactsError(f"{params_path}: top level must be a JSON object")

        version = raw.get("version")
        # `type(...) is int`, not isinstance: bool subclasses int, so
        # {"version": true} would otherwise pass and then compare as 1.
        if type(version) is not int or not 1 <= version <= PARAMS_VERSION:
            raise ArtifactsError(
                f"{params_path}: unsupported params version {version!r} "
                f"(this build reads 1 to {PARAMS_VERSION})"
            )

        # Structural checks before file I/O, so a malformed section is reported as
        # such instead of being pre-empted by a missing-LUT error.
        training = _require_object(raw.get("training", {}), "training", params_path)
        try:
            normalize = NormalizeParams.from_dict(
                _require_object(raw.get("normalize", {}), "normalize", params_path)
            )
            grain = GrainParams.from_dict(
                _require_object(raw.get("grain", {}), "grain", params_path)
            )
        except (ValueError, TypeError) as exc:
            # TypeError too: a field-level type error such as
            # {"wb_gain_min": "oops"} reaches math.isfinite and raises TypeError,
            # which would otherwise escape unwrapped past this loader.
            raise ArtifactsError(f"{params_path}: {exc}") from exc
        if grain.model != "gaussian" and version < GRAIN_MODEL_VERSION:
            raise ArtifactsError(
                f"{params_path}: grain model {grain.model!r} needs params version "
                f"{GRAIN_MODEL_VERSION}, file says {version}"
            )

        lut_file = raw.get("lut_file", DEFAULT_LUT_FILE)
        # Validated before use: `path / 5` raises TypeError, which would escape
        # this loader unwrapped, and a name like "../../x.cube" would read
        # outside the artifact directory.
        if not isinstance(lut_file, str) or not lut_file:
            raise ArtifactsError(
                f"{params_path}: 'lut_file' must be a non-empty string, "
                f"got {type(lut_file).__name__}"
            )
        if Path(lut_file).is_absolute() or ".." in Path(lut_file).parts:
            raise ArtifactsError(
                f"{params_path}: 'lut_file' must be a plain name inside the artifact "
                f"directory, got {lut_file!r}"
            )
        lut_path = path / lut_file
        if not lut_path.is_file():
            raise ArtifactsError(f"LUT file {lut_path} not found (named in {params_path})")
        try:
            lut = read_cube(lut_path)
        except CubeError as exc:
            raise ArtifactsError(str(exc)) from exc

        actual = sha1_hex(lut)
        recorded = raw.get("lut_sha1")
        # Required, not optional. Making it optional would let a params.json that
        # simply omits the field load a LUT it was never paired with, which is the
        # exact failure this check exists to catch. There is no earlier schema
        # version to stay compatible with: PARAMS_VERSION starts at 2.
        if not isinstance(recorded, str):
            raise ArtifactsError(
                f"{params_path}: 'lut_sha1' is required and must be a string, "
                f"got {type(recorded).__name__}"
            )
        if recorded != actual:
            raise ArtifactsError(
                f"{path}: lut_sha1 in params.json ({recorded}) does not match {lut_path.name} "
                f"({actual}). The artifact is mixed; re-run training or restore it."
            )

        return cls(
            lut=lut,
            normalize=normalize,
            grain=grain,
            training=training,
            path=path,
            lut_sha1=actual,
        )

    @classmethod
    def default(cls) -> Artifacts:
        """The artifact shipped inside the package.

        Resolved as a sub-path of ``pifilm`` rather than as the package
        ``pifilm.data``. ``pifilm/data`` has no ``__init__.py``, so
        ``resources.files("pifilm.data")`` yields a ``MultiplexedPath``,
        and ``as_file`` on one of those materialises a *temporary* copy that
        is deleted when the context exits — leaving ``Artifacts.path``
        pointing at a directory that no longer exists, and copying the
        950 KB table on every startup. A sub-path of a real package is a real
        directory, so it survives and costs nothing.
        """
        data_dir = Path(str(resources.files("pifilm") / "data"))
        if not data_dir.is_dir():
            raise ArtifactsError(
                f"packaged artifact directory {data_dir} is missing; reinstall the package"
            )
        return cls.load(data_dir)

    @classmethod
    def resolve(cls, dir_path: str | Path | None) -> Artifacts:
        return cls.load(dir_path) if dir_path is not None else cls.default()


def write_artifact(
    dir_path: str | Path,
    lut: LUT3D,
    normalize: NormalizeParams,
    grain: GrainParams,
    training: dict | None = None,
    lut_file: str = DEFAULT_LUT_FILE,
) -> Path:
    """Write a complete artifact into ``dir_path`` (creating it) and return the directory."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    write_cube(lut, path / lut_file, title="pifilm")
    # Hash what will actually be read back, not the in-memory table. The .cube
    # format stores six decimals, so a fitted LUT loses about 5e-7 per value on
    # the way to disk. Recording the pre-write hash would make every non-identity
    # artifact fail its own integrity check on load — and an identity table would
    # not reveal it, because its values are exact at six decimals.
    payload = {
        "version": GRAIN_MODEL_VERSION if grain.model != "gaussian" else 2,
        "lut_file": lut_file,
        "lut_sha1": sha1_hex(read_cube(path / lut_file)),
        "normalize": normalize.to_dict(),
        "grain": grain.to_dict(),
        "training": training or {},
    }
    (path / "params.json").write_text(json.dumps(payload, indent=2) + "\n")
    return path


def publish(staging_dir: str | Path, dest_dir: str | Path) -> Path:
    """Validate ``staging_dir``, then swap it into ``dest_dir``.

    The artifact is never published half-written: ``Artifacts.load`` runs
    against the staging directory first, so an incomplete or inconsistent set
    is rejected before anything moves.

    On the swap itself, be precise about the guarantee. ``os.replace`` cannot
    rename onto a non-empty directory, so replacing an existing artifact takes
    two steps, and between them the destination briefly does not exist. A
    reader in that window gets "not found" rather than a mixed artifact, which
    is the failure mode that matters; it is two back-to-back metadata calls
    with no work between them. Fully closing the window would need a symlink
    indirection, which the design does not call for.
    """
    staging = Path(staging_dir)
    dest = Path(dest_dir)
    Artifacts.load(staging)  # raises ArtifactsError before anything is moved

    dest.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if dest.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{dest.name}.old.", dir=dest.parent))
            os.rmdir(backup)
            os.replace(dest, backup)
        os.replace(staging, dest)
    except OSError as exc:
        if backup is not None and not dest.exists():
            os.replace(backup, dest)
        raise ArtifactsError(f"could not publish {staging} to {dest}: {exc}") from exc
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return dest
