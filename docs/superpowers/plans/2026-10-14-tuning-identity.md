# Tuning-File Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** The Pi fingerprints the libcamera tuning it runs with, a trained look can carry the fingerprint it was trained under, and a difference is reported (log, records, an amber `TUNING` tag on the viewfinder) without ever stopping the camera.

**Architecture:** A pure module `pifilm/tuning.py` defines the fingerprint, how the tuning in use is resolved (the Picamera2 loader and the environment passed in, so it tests on a Mac) and the comparison. The artifact gains an optional, validated `tuning_sha1`. The Picamera2 backend fingerprints at open and reports it through `StreamInfo`; the capture session compares at start-up; the viewfinder reads a flag through the controller snapshot, like the developing count.

**Tech Stack:** Python 3.12, stdlib `hashlib`/`json`, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-14-tuning-identity-design.md` (approved by the owner 2026-10-14: warn only). Read it first; it is the authority. Its tables give the exact log lines and record fields.

## Global Constraints

- Always `.venv/bin/pytest`, `.venv/bin/ruff`. Suite: `.venv/bin/pytest -q -m 'not slow'` (about 130 s; allow 400 s; macOS has no `timeout` command). Lint: `.venv/bin/ruff check .`. Both pass before every commit.
- Never touch, stage or delete the untracked directory named `" "` (one space) at the repo root. Stage by explicit path only.
- Never read, print or source `.env`.
- Never refuse to start, never raise out of camera open or session start because of the fingerprint or the check. A tuning that cannot be fingerprinted is one log line, not an error.
- No change to graded pixels, `lut_sha1`, `normalize_sha1` or `PARAMS_VERSION`. Older builds must still load a stamped artifact (they ignore unknown top-level keys; keep it top-level).
- Hardware imports stay lazy: `pifilm/tuning.py` and the CLI import nothing from Picamera2 at module level; the package imports on a Mac.
- Fingerprint is exactly `hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()`.
- Existing tests pass unchanged, except expected records gaining `tuning_sha1`/`tuning_matches` where a fake camera now reports one.
- Match the surrounding code: module docstrings carry the reasons, comments are plain sentences saying why, no em dashes in prose.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Work on branch `feat/tuning-hash`. Do not push, open pull requests or switch branches. Do not edit `README.md`.

## Review Focus

- `--tuning-file` given as a bare name (`imx477.json`, the deployed case) must fingerprint the very dict the camera is opened with, not a re-read of some other file.
- A tuning file that is not valid JSON or is missing (auto case) must give a camera that opens and one `cannot fingerprint` line, not an exception.
- A look stamped with an uppercase, short or non-string `tuning_sha1` must fail to load with `ArtifactsError`, not slip through and never match.
- Stamping must keep every other key of `params.json` and leave the artifact loadable; a failure part way must leave the old `params.json` intact.
- The V4L2 camera and `--fake` must behave exactly as before: no fingerprint, no check, no line, no tag.

---

### Task 1: the fingerprint, the look's field, stamping, `pifilm-tuning`, the trainer flag

**Files:**
- Create: `pifilm/tuning.py`, `pifilm/capture/tuningcli.py`, `tests/test_tuning.py`
- Modify: `pifilm/artifacts.py`, `pifilm/train/fit.py`, `pyproject.toml` (`pifilm-tuning = "pifilm.capture.tuningcli:main"` under `[project.scripts]`)
- Test: `tests/test_tuning.py`, additions to the existing artifacts and trainer-CLI tests

**Produces (later tasks use these exactly):**

```python
# pifilm/tuning.py
ENV_VAR = "LIBCAMERA_RPI_TUNING_FILE"

def fingerprint(obj: object) -> str: ...      # the Global Constraints formula

@dataclass(frozen=True)
class Resolved:
    label: str            # what the record's tuning_file says: the name given, or f"auto:{model}"
    source: str | None    # where the dict came from, for the CLI and the log (path or name)
    sha1: str | None      # None when it could not be fingerprinted
    reason: str | None    # why not, when sha1 is None

def resolve(tuning_file: str | None, model: str, *,
            load: Callable[[str], dict], environ: Mapping[str, str],
            loaded: dict | None = None) -> Resolved:
    """tuning_file given: fingerprint `loaded` when passed (the dict the camera was
    opened with), else load(tuning_file); a failure here propagates (the camera would
    not open either). Not given: environ[ENV_VAR] (read and json-parsed from that
    path) when set, else load(f"{model.lower()}.json"); any failure in this branch
    returns sha1=None with a one-phrase reason."""

def check(look_sha1: str | None, camera: Resolved | None) -> tuple[bool | None, str | None]:
    """(tuning_matches, log line). (None, None) when the look has no fingerprint or
    camera is None (V4L2, fake). The lines exactly as the spec's table, with the
    first 12 hex digits."""

def short(sha1: str) -> str: ...              # sha1[:12]

# pifilm/artifacts.py
class Artifacts: ...  tuning_sha1: str | None = None   # new last field, default None
def write_artifact(..., tuning_sha1: str | None = None) -> Path   # written top-level when given
def stamp_tuning(dir_path: str | Path, sha1: str) -> Path
    """Rewrite params.json with tuning_sha1 set, all other keys kept in order:
    validate sha1 first (ArtifactsError), write a temp file in the same directory,
    fsync, os.replace, then Artifacts.load(dir) to confirm. The .cube is untouched."""
```

**Behaviour:**

- Validation in `Artifacts.load`: when the key is present it must be a `str` matching `^[0-9a-f]{40}$`; otherwise `ArtifactsError` naming `params.json` and the field. Absent means `None`. One helper used by both `load` and `stamp_tuning`/`write_artifact`.
- `pifilm-tuning` (argparse): `--tuning-file NAME` (optional), `--sha1 HEX` (instead of resolving), `--stamp DIR`. Without `--sha1` it imports Picamera2 lazily inside `main` and uses `Picamera2.load_tuning_file` as `load`. It does not open the camera (the service usually holds it), so the auto case takes the sensor model from `--model` (default `imx477`); `--help` says so. Prints `tuning <source>: <sha1>`; with `--stamp` also `stamped <dir>/params.json`. Exit 0; 1 with `error: ...` when resolution yields no fingerprint, Picamera2 is missing (and no `--sha1`), or stamping fails. `--sha1` is validated like the field. `main(argv, *, out=print, load=None, environ=None)` so tests inject the loader.
- `pifilm-train --tuning-sha1 HEX`: validated (argparse error on bad value), passed to `write_artifact`. Find where the trainer calls `write_artifact` and thread it through `FitConfig` or the call, whichever the code already does for similar provenance. The value also goes into the training report if the report already lists provenance fields; otherwise leave the report alone.

**Tests:**

1. `fingerprint` is the formula; key order and whitespace of the source JSON do not change it; one changed nested value does.
2. `resolve`: given name with `loaded` (load not called), given name without `loaded` (load called with the name), given name whose load raises (propagates); auto with the env var pointing at a temp JSON file; auto with the env var pointing at a missing file and at invalid JSON (sha1 None, reason set, no raise); auto with no env var (load called with `"imx477.json"` for model `"IMX477"`); auto whose load raises (sha1 None).
3. `check`: all four rows of the spec's table, exact lines.
4. Artifacts: absent field loads as `None`; valid field round-trips through `write_artifact`; uppercase, 39 digits, a number and `null` each raise `ArtifactsError`; `lut_sha1` and `Pipeline(...).normalize_sha1` are the same with and without the field.
5. `stamp_tuning`: keeps every other key and their order, artifact loads with the new value, the `.cube` bytes are unchanged; an invalid sha1 leaves `params.json` byte-identical; if `os.replace` is made to fail (monkeypatch), the old `params.json` is intact and no temp file is left.
6. CLI: `--sha1 X --stamp DIR` on a temp artifact; resolution through an injected `load`; a resolution with no fingerprint exits 1; bad `--sha1` exits non-zero.
7. Trainer CLI: `--tuning-sha1` reaches `params.json` (use the smallest existing trainer CLI test fixture; if the trainer test is marked slow, test the argument wiring at the level the existing tests do).

Commits: small, one per concern.

---

### Task 2: the camera fingerprints, the session checks, records carry it

**Files:**
- Modify: `pifilm/capture/camera.py` (`StreamInfo.tuning_sha1: str | None = None`, emitted by `to_dict()` right after `tuning_file` when not None), `pifilm/capture/picamera.py`, `pifilm/capture/app.py`
- Test: additions to the existing Picamera2 backend tests (they use a fake `Picamera2` module) and the capture app/session tests

**Consumes:** `pifilm.tuning.resolve`, `check`, `Resolved` from Task 1; `Artifacts.tuning_sha1`.

**Produces:**

```python
# Picamera2Camera
@property
def tuning(self) -> Resolved | None: ...   # set at open; the V4L2 camera and FakeCamera have none (use getattr(camera, "tuning", None))

# CaptureSession
tuning_mismatch: bool                    # True only when check() returned False
```

**Behaviour:**

- In `Picamera2Camera.__init__`: keep the existing `load_tuning_file` call; after the camera is open and `model` is read, call `resolve(tuning_file, model, load=Picamera2.load_tuning_file, environ=os.environ, loaded=tuning)`. Wrap only the auto branch's failure (already returned as `sha1=None` by `resolve`); an unexpected exception from `resolve` in the auto case must also not stop the camera: catch `Exception`, log, and treat as `sha1=None`. When `sha1` is None print the spec's `tuning: cannot fingerprint the tuning in use (<reason>)` through the same logging/print path the backend already uses for its warnings. Pass the sha1 into `_stream_info` so `StreamInfo.tuning_sha1` is set.
- In the session (find where it is constructed with the camera and the pipeline/artifacts, and where the start-up lines like `gpu grading on ...` are logged; the check belongs once per run, at start-up, after both exist): `matches, line = check(artifacts.tuning_sha1, getattr(camera, "tuning", None))`; log `line` when not None; keep `matches`; `self.tuning_mismatch = matches is False`.
- Records: wherever a record or a background shot's `fields` spreads `**self.camera.stream_info.to_dict()` (inline single, background single, both double paths, the composite: search for every occurrence), add `**({"tuning_matches": matches} if matches is not None else {})` immediately after it. Recovered grades use the fields stored in the pending entry; do not add anything at recovery.

**Tests:**

1. Backend with the fake Picamera2: explicit `tuning_file` gives `stream_info.to_dict()["tuning_sha1"] == fingerprint(<the dict the fake load returned>)` and `load_tuning_file` is called once (not re-read); auto with the fake loader raising gives no `tuning_sha1`, a `cannot fingerprint` line, and an open camera; auto with a working fake loader gives the fingerprint.
2. Session with a fake camera exposing `tuning` and an artifact with `tuning_sha1`: equal gives the `matches` line, records with `tuning_matches: true`, `tuning_mismatch` False; different gives the `DIFFERS` line, `tuning_matches: false`, `tuning_mismatch` True, and the photograph is still taken and graded; look without fingerprint gives no line and no field; camera without `tuning` (plain `FakeCamera`) gives no line and no field.
3. Background mode: the record written by the grade worker carries `tuning_matches`; a record recovered from a pending entry carries the value stored in the entry (write the entry with `true`, recover in a session whose check says `false`, expect `true`).
4. Inline mode and a double exposure carry the field too.

Commits: small, one per concern.

---

### Task 3: the viewfinder tag and the docs

**Files:**
- Modify: `pifilm/capture/controller.py` (`ControllerSnapshot.tuning_mismatch: bool = False`, read from the session with `getattr(session, "tuning_mismatch", False)` the way `pending_grades` is read), `pifilm/display/liveview.py` (`LiveState.tuning_mismatch: bool = False`; the tag), `pifilm/display/viewfinder.py` (pass it from the snapshot where `developing=` is passed)
- Docs: `docs/known-issues.md`, `docs/training.md` (step 7, "Pin the tuning file for a trained look"), `docs/lcd-viewfinder.md` (screen layout section near "Developing count"; a short acceptance subsection in section 7 with the spec's four hardware steps), `docs/how-it-works.md` (where records' camera fields or reproducibility are described), `CLAUDE.md` (artifacts bullet: optional `tuning_sha1`; capture section: the check, warn only; Commands: `pifilm-tuning`), `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md` (step 4: built, not yet run on the Pi)
- Test: additions to the existing liveview, viewfinder and controller tests

**Behaviour:**

- The tag: text `TUNING`, amber like the developing count, in the top-left tag row after the battery tag and the developing count (or where those would be when absent), same height and padding as the battery tag. It lives in `_tags_tile` (which already lays out the row because positions depend on the battery text's width); give it a new keyword argument `tuning: bool = False`, and draw the tile when any of double, battery, developing or tuning is present. Keep the tile cache keys correct: whatever caches `_tags_tile` must include the new argument.
- Check the tile still fits the region reserved for it (`_TAGS_BOX` or equivalent) at the widest battery text (`AC 100%`) plus a two-digit developing count at both scales in use (1 and 2); widen the reserved box if needed, and make sure it does not overlap the controls or the shutter button (the hit-test regions). Say in the report what width you measured.
- Docs: the known issue's status becomes "checked at start-up, warn only (step 4)", with what the owner sees and how to stamp a look; `training.md` step 7 gains the stamp command (`pifilm-tuning --tuning-file imx477.json --stamp <artifacts dir>`, run on the Pi after deploying) and `pifilm-train --tuning-sha1`.

**Tests:**

1. `LiveState(tuning_mismatch=True)` renders the tag (pixels in the expected place differ from the render without it); without it, the render is byte-identical to before.
2. The tag with the widest battery text and a developing count of 9 or more stays inside the tag box at scales 1 and 2.
3. The snapshot carries `tuning_mismatch` from the session; a session without the attribute gives False.
4. The viewfinder passes it into `LiveState`.

Commits: one for code, one for docs.
