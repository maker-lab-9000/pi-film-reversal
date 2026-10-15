# Tuning-file identity: design

Amended 2026-10-14 (whole-branch review): item 2 of "Which tuning is fingerprinted"
is dropped. Picamera2 0.3.x removes `LIBCAMERA_RPI_TUNING_FILE` from the environment
when it opens a camera without a tuning, so the camera never uses it, and honouring it
would make `pifilm-tuning` and the service disagree. Without `--tuning-file` the
fingerprint is of `Picamera2.load_tuning_file(f"{model.lower()}.json")` only, and the
variable is ignored.

Date: 2026-10-14. Step 4 of `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`.
Branch `feat/tuning-hash`, pull request into `pipeline-improvements`. Approved by the
owner in conversation on 2026-10-14: warn only, never refuse to start.

## Goal

A trained look is fitted to the colours the camera produced under one libcamera
tuning file. An OS or libcamera update can replace that file; today nothing notices,
and every photograph shifts with no error and no changed `lut_sha1`
(`docs/known-issues.md`, "A libcamera update could silently retune a deployed
trained LUT"). This step turns that into a check: the Pi fingerprints the tuning it
runs with, a trained look can carry the fingerprint it was trained under, and a
difference is reported in the log, in every record and on the viewfinder.

The deployed service runs `--tuning-file imx477.json` with the trained IMX477 look.

## The fingerprint

`tuning_sha1`: SHA-1 of the tuning's parsed JSON, serialised with
`json.dumps(obj, sort_keys=True, separators=(",", ":"))` and UTF-8 encoded. Parsed,
not the file's bytes, so a change of whitespace or key order is not a change, and any
changed value is. Forty lowercase hex digits.

Which tuning is fingerprinted (Picamera2 backend only):

1. `--tuning-file NAME` given: the dict `Picamera2.load_tuning_file(NAME)` returns,
   the same dict the camera is opened with. Hashed before it is passed on.
2. No `--tuning-file`, and `LIBCAMERA_RPI_TUNING_FILE` set in the environment: that
   file, read and parsed.
3. Neither: `Picamera2.load_tuning_file(f"{model.lower()}.json")`, the sensor's
   default file from the directories Picamera2 searches, which are libcamera's.
4. Any failure in 2 or 3 (file missing, not JSON): no fingerprint, one log line
   `tuning: cannot fingerprint the tuning in use (<reason>)`. The camera still opens.
   Case 1 cannot fail separately: a tuning file that cannot be loaded already stops
   the camera from opening, as today.

The V4L2 (USB) backend has no tuning file: no fingerprint, nothing checked, no line.

## The look's fingerprint

`params.json` gains an optional top-level `tuning_sha1` (forty lowercase hex digits).
The loader ignores unknown top-level keys today, so older builds read a stamped
artifact unchanged: no params version change. This build validates the field when
present (`ArtifactsError` for any other value) and exposes it as
`Artifacts.tuning_sha1: str | None`. `lut_sha1`, `normalize_sha1` and the pixels are
unaffected.

Ways in:

- `pifilm-train --tuning-sha1 HEX`: written into the new artifact. Without it the
  artifact has none, as today. (The trainer reads a folder of photographs, not
  capture records, so it cannot find the fingerprint itself.)
- `pifilm-tuning` (new command, run on the Pi):
  - `pifilm-tuning [--tuning-file NAME]` prints the fingerprint of the tuning the
    camera would use, resolved exactly as above, and which file it came from.
  - `pifilm-tuning [--tuning-file NAME] --stamp DIR` also writes it into the
    artifact at `DIR`: `params.json` is rewritten through a temporary file in the
    same directory and `os.replace`, with every other key kept, then the artifact
    is loaded to confirm. The `.cube` is not touched.
  - `--sha1 HEX` in place of resolving the tuning, so a look can be stamped on the Mac.
  It needs Picamera2 only when it resolves a tuning (not with `--sha1`).

The bundled starter (`pifilm/data/`) is not trained and carries no fingerprint.

## The check

At start-up, once the camera is open and the artifact loaded:

| Look has `tuning_sha1` | Camera fingerprint | Result |
| --- | --- | --- |
| no | any | nothing checked, no line, no `tuning_matches` |
| yes | none (could not fingerprint) | one line `tuning: cannot check the look's tuning ...`; no marker; no `tuning_matches` |
| yes | equal | one line `tuning: matches the look (<first 12 hex>)`; `tuning_matches: true` |
| yes | different | one line `tuning: DIFFERS from the look: camera <12 hex> (<label>), look <12 hex>; colours may be off`; `tuning_matches: false`; the viewfinder marker |

Warn only: the service never refuses to start because of it (the owner's decision:
after an update in the field a camera that will not shoot is worse than colours
that can be regraded from the kept originals).

## Records

- `tuning_sha1` beside `tuning_file` in the camera fields of every record from the
  Picamera2 backend that has a fingerprint (it comes from `StreamInfo.to_dict()`, so
  every kind of record carries it: inline, background, recovered, double).
- `tuning_matches` (`true`/`false`) in every record when the check ran (look and
  camera both have a fingerprint), placed right after the camera fields
  (`**stream_info.to_dict()`), so it is set when the photograph is taken.
- A background shot's record fields are written into its pending entry at capture
  time, so a grade recovered at a later start keeps the `tuning_sha1` and
  `tuning_matches` of the run that took the photograph.

## On the screen

While the camera's fingerprint differs from the look's, the live view shows an amber
`TUNING` tag in the top-left tag row, after the battery tag and the developing count,
in the same tag style. It stays for the whole run (the tuning cannot change while the
service runs). Absent otherwise. It comes to the viewfinder the way the developing
count does: `ControllerSnapshot.tuning_mismatch` read from the session, into
`LiveState.tuning_mismatch`.

## Code

- `pifilm/tuning.py` (new): `fingerprint(obj) -> str`, `resolve(tuning_file, model,
  load, environ) -> (label, sha1 | None, reason | None)` with the Picamera2 loader and
  the environment passed in (so it tests on a Mac), `check(look_sha1, camera_sha1) ->
  tuple[bool | None, str | None]` (the `tuning_matches` value and the log line).
- `pifilm/artifacts.py`: optional `tuning_sha1` (validated), `Artifacts.tuning_sha1`,
  `write_artifact(..., tuning_sha1=None)`, `stamp_tuning(dir, sha1)`.
- `pifilm/capture/camera.py`: `StreamInfo.tuning_sha1`.
- `pifilm/capture/picamera.py`: computes the fingerprint at open.
- `pifilm/capture/app.py`: the check at start-up, its log line, `tuning_matches` in
  records, `CaptureSession.tuning_mismatch`.
- `pifilm/capture/controller.py`, `pifilm/display/liveview.py`, `viewfinder.py`: the
  marker.
- `pifilm/train/fit.py`: `--tuning-sha1`.
- `pifilm/capture/tuningcli.py`: `pifilm-tuning`; entry in `pyproject.toml`.
- Docs: `docs/known-issues.md` (the entry becomes "checked, warn only"),
  `docs/training.md` step 7 (stamp after deploying), `docs/lcd-viewfinder.md` (the
  tag; acceptance), `docs/how-it-works.md`, `CLAUDE.md`, the roadmap. No README change.

## Acceptance (hardware)

1. `pifilm-tuning --tuning-file imx477.json` prints a fingerprint and the file.
2. `pifilm-tuning --tuning-file imx477.json --stamp <the service's artifacts>`; the
   service restarts; the log says `tuning: matches the look`; records carry
   `tuning_sha1` and `tuning_matches: true`; no tag on screen.
3. Copy `imx477.json`, change one value (for example a `rpi.ccm` coefficient), run the
   service by hand with `--tuning-file` naming the copy: the log says `DIFFERS`, the
   amber `TUNING` tag shows, records say `tuning_matches: false`, photographs are still
   taken and graded.
4. Back to the unit's own command: tag gone.

## Not in this step

- Fingerprinting the libcamera version or the sensor's firmware.
- Refusing to start, or regrading automatically.
- Finding the fingerprint from the training photographs' records.
