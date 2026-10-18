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

**The tuning it was trained under.** An optional top-level ``tuning_sha1`` is
the fingerprint of the libcamera tuning the camera ran with when the training
photographs were taken (``pifilm.tuning``). It is top-level and optional so an
older build, which ignores unknown top-level keys, still loads a stamped look:
no params version change, and ``lut_sha1``, ``normalize_sha1`` and the pixels
are unaffected. When present it is validated (forty lowercase hex digits):
anything else would never equal a camera's fingerprint, so a typo would report
every shot as a mismatch instead of failing once at load. ``stamp_tuning`` adds
it to an existing artifact through a temporary file and ``os.replace``, so a
failure part way leaves the old ``params.json``.

**Halation and version 5.** An optional ``halation`` section holds the
``HalationParams`` (``pifilm.halation``). A build from before halation ignores
unknown sections and would grade the look without its glow, so a look with
halation switched on must be at params version 5, which such a build refuses. A
look without the section, or with it switched off, keeps its version and its
exact pixels. ``stamp_halation`` writes the section into an existing artifact
the way ``stamp_tuning`` writes its key, and raises the version (never lowers
it) when the halation is on.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .grain import GrainParams
from .halation import HalationParams
from .lut import LUT3D, CubeError, read_cube, sha1_hex, write_cube
from .normalize import NormalizeParams

PARAMS_VERSION = 5
# Version 3 exists only to carry a non-Gaussian grain model: an older build ignores
# the extra grain fields and would grain at the Gaussian model's settings, so such
# an artifact must be one it refuses. Gaussian-grain artifacts stay at version 2.
GRAIN_MODEL_VERSION = 3
# Version 4 carries the tiled HP5 grain, whose pixels differ from version 3's.
GRAIN_MODEL_VERSIONS = {"gaussian": 2, "two_scale": GRAIN_MODEL_VERSION, "two_scale_tile": 4}
# Version 5 carries halation, which an older build would ignore and grade without.
HALATION_VERSION = 5
DEFAULT_LUT_FILE = "pifilm.cube"
_TUNING_SHA1 = re.compile(r"[0-9a-f]{40}")


class ArtifactsError(Exception):
    """Artifact directory missing, incomplete or invalid."""


def _require_object(value: object, name: str, path: Path) -> dict:
    if not isinstance(value, dict):
        raise ArtifactsError(f"{path}: '{name}' must be a JSON object, got {type(value).__name__}")
    return value


def validate_tuning_sha1(value: object, path: Path | str) -> str:
    """The one check of ``tuning_sha1``, for loading, writing and stamping."""
    if not isinstance(value, str) or not _TUNING_SHA1.fullmatch(value):
        raise ArtifactsError(
            f"{path}: 'tuning_sha1' must be forty lowercase hex digits, got {value!r}"
        )
    return value


@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    training: dict
    path: Path
    lut_sha1: str
    tuning_sha1: str | None = None
    halation: HalationParams = field(default_factory=HalationParams)

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
            halation = HalationParams()
            if "halation" in raw:
                halation = HalationParams.from_dict(
                    _require_object(raw["halation"], "halation", params_path)
                )
        except (ValueError, TypeError) as exc:
            # TypeError too: a field-level type error such as
            # {"wb_gain_min": "oops"} reaches math.isfinite and raises TypeError,
            # which would otherwise escape unwrapped past this loader.
            raise ArtifactsError(f"{params_path}: {exc}") from exc
        if grain.model != "gaussian" and version < GRAIN_MODEL_VERSIONS[grain.model]:
            raise ArtifactsError(
                f"{params_path}: grain model {grain.model!r} needs params version "
                f"{GRAIN_MODEL_VERSIONS[grain.model]}, file says {version}"
            )
        if halation.active and version < HALATION_VERSION:
            raise ArtifactsError(
                f"{params_path}: halation needs params version {HALATION_VERSION}, "
                f"file says {version}"
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

        tuning_sha1 = None
        if "tuning_sha1" in raw:
            tuning_sha1 = validate_tuning_sha1(raw["tuning_sha1"], params_path)

        return cls(
            lut=lut,
            normalize=normalize,
            grain=grain,
            training=training,
            path=path,
            lut_sha1=actual,
            tuning_sha1=tuning_sha1,
            halation=halation,
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
    tuning_sha1: str | None = None,
    halation: HalationParams | None = None,
) -> Path:
    """Write a complete artifact into ``dir_path`` (creating it) and return the directory.

    ``tuning_sha1``, when given, is written top-level (see the module docstring).
    ``halation``, when given, is written after ``grain``; switched on, it makes the
    file version 5 at least.
    """
    path = Path(dir_path)
    if tuning_sha1 is not None:
        validate_tuning_sha1(tuning_sha1, path / "params.json")
    path.mkdir(parents=True, exist_ok=True)
    write_cube(lut, path / lut_file, title="pifilm")
    # Hash what will actually be read back, not the in-memory table. The .cube
    # format stores six decimals, so a fitted LUT loses about 5e-7 per value on
    # the way to disk. Recording the pre-write hash would make every non-identity
    # artifact fail its own integrity check on load — and an identity table would
    # not reveal it, because its values are exact at six decimals.
    payload = {
        "version": max(
            GRAIN_MODEL_VERSIONS[grain.model],
            HALATION_VERSION if halation is not None and halation.active else 0,
        ),
        "lut_file": lut_file,
        "lut_sha1": sha1_hex(read_cube(path / lut_file)),
        "normalize": normalize.to_dict(),
        "grain": grain.to_dict(),
    }
    if halation is not None:
        payload["halation"] = halation.to_dict()
    payload["training"] = training or {}
    if tuning_sha1 is not None:
        payload["tuning_sha1"] = tuning_sha1
    (path / "params.json").write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _rewrite_params(path: Path, change: Callable[[dict], None], verb: str) -> Path:
    """Apply ``change`` to the artifact's ``params.json`` in place; return its path.

    Every key ``change`` leaves alone is kept, in its order. The new file is
    written beside the old one, fsynced and moved over it with ``os.replace``, so
    a reader, or a failure part way, sees the whole old file or the whole new
    one. The artifact is loaded before (an already broken one is refused
    untouched) and after, to confirm. The file's mode is kept. The ``.cube`` is
    not touched, so ``lut_sha1`` still holds.
    """
    params_path = path / "params.json"
    # Refuse an artifact that is already unloadable (a mixed lut_sha1, say) before
    # anything is written, so a failed stamp leaves params.json as it was.
    Artifacts.load(path)
    try:
        raw = json.loads(params_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactsError(f"cannot read {params_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArtifactsError(f"{params_path}: top level must be a JSON object")
    change(raw)

    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".params.", suffix=".json.tmp", dir=path)
    except OSError as exc:
        raise ArtifactsError(f"could not {verb} {params_path}: {exc}") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(raw, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp makes the file 0600 and os.replace would carry that onto
        # params.json; keep the mode the artifact had.
        shutil.copymode(params_path, tmp_name)
        os.replace(tmp_name, params_path)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise ArtifactsError(f"could not {verb} {params_path}: {exc}") from exc
    Artifacts.load(path)
    return params_path


def stamp_tuning(dir_path: str | Path, sha1: str) -> Path:
    """Set ``tuning_sha1`` in the artifact at ``dir_path``; return its ``params.json``.

    An earlier stamp is replaced where it stands; everything else is as
    ``_rewrite_params`` keeps it.
    """
    path = Path(dir_path)
    validate_tuning_sha1(sha1, path / "params.json")

    def change(raw: dict) -> None:
        raw["tuning_sha1"] = sha1

    return _rewrite_params(path, change, "stamp")


def stamp_halation(dir_path: str | Path, params: HalationParams) -> Path:
    """Set the ``halation`` section of the artifact at ``dir_path``; return its ``params.json``.

    Switched on, the version is raised to 5 if it is lower; switched off, the
    version is left as it is (never lowered: the grain may need it). An earlier
    section is replaced where it stands; everything else is as
    ``_rewrite_params`` keeps it.
    """

    def change(raw: dict) -> None:
        raw["halation"] = params.to_dict()
        if params.active:
            # The load before has checked the version is a whole number in range.
            raw["version"] = max(raw.get("version", 0), HALATION_VERSION)

    return _rewrite_params(Path(dir_path), change, "stamp halation in")


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
