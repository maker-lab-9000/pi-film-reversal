# Background Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** The camera is free for the next photograph once the original is on disk and a quick grade exists; the full-resolution grade runs on a queue behind the shutter.

**Architecture:** A small, camera-free `GradeQueue` (one worker thread, bounded, with on-disk pending entries for crash recovery). `CaptureSession` splits a shot into a capture side (read, save original, quick grade, enqueue) and a grade side (grade, save, record) that the queue runs. The controller exposes how many grades are waiting; the live view draws that number.

**Tech Stack:** Python 3.12, NumPy, Pillow, OpenCV through `pifilm._cv2.require_cv2()`, pytest, ruff. Threads from the standard library only.

**Spec:** `docs/superpowers/specs/2026-10-13-background-grading-design.md` (approved by the owner 2026-10-13). Read it first; it is the authority.

## Global Constraints

- Always `.venv/bin/pytest`, `.venv/bin/ruff`. Suite: `.venv/bin/pytest -q -m 'not slow'` (about 75 s; give it a 300 s timeout). Lint: `.venv/bin/ruff check .` (E, F, I, B, UP; line length 100). Both must pass before every commit.
- Never touch, stage or delete the untracked directory named `" "` (one space) at the repo root. Stage files by explicit path, never `git add -A` or `git add .`.
- Never read, print or source `.env`.
- Never `import cv2` directly; use `require_cv2()`. Never `Image.open` for pixels; use `pifilm.imageio.load_rgb` / `save_jpeg`.
- Match the surrounding code: module docstrings carry the design reasons, comments are plain sentences saying why, no em dashes in prose.
- With background grading off (`CaptureSession` default, `--grade inline`), behaviour, files and records are exactly as before this plan. Existing tests must pass unchanged, except where a task says otherwise.
- The graded file for a shot must be byte-identical between inline and background grading (same frame, seed, EV, artifact).
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Work on branch `feat/background-grade`. Do not push, do not open pull requests, do not switch branches.
- The Pi has 4 GB of RAM; a 12 MP RGB frame is about 37 MB. Do not copy full frames without need.

## Review Focus

- A grade that raises (disk full, unreadable original) must not kill the worker, lose the pending entry, or block later shots.
- A crash between "graded file saved", "record appended" and "entry deleted" must not leave a duplicate record or a missing one after recovery.
- A full queue must make the next capture wait, not fail and not drop a photograph; stopping the service while it waits must not deadlock.
- Stopping (normal close) finishes queued grades before the process exits; a second close is harmless.
- Recovered entries for a different output folder, or with a missing original, must not stop start-up.

---

### Task 1: `GradeQueue` and pending entries

**Files:**
- Create: `pifilm/capture/gradequeue.py`
- Test: `tests/test_gradequeue.py`

No camera, no pipeline, no display in this module. It knows nothing about what a grade is.

**Interfaces (produces):**

```python
ENTRY_VERSION = 1
MAX_ATTEMPTS = 2

def default_state_dir() -> Path:
    """$XDG_STATE_HOME/pifilm/pending, or ~/.local/state/pifilm/pending."""

def write_entry(state_dir: Path, name: str, payload: dict) -> Path:
    """Write <state_dir>/<name>.json atomically (temp file in the same directory,
    flush, os.fsync, os.replace; directory created with mode 0o700). The payload
    gains "version": ENTRY_VERSION and "attempts": 0 unless it has them. Returns the path."""

def read_entries(state_dir: Path) -> list[tuple[Path, dict]]:
    """Every *.json entry, sorted by file name. A missing directory is an empty list.
    An entry that is not valid JSON, not an object, or of another version is renamed
    to *.failed and skipped (never raises)."""

def note_failure(path: Path) -> bool:
    """Add one to the entry's "attempts" (rewritten atomically). At MAX_ATTEMPTS the
    file is renamed to <name>.failed and False is returned; otherwise True. A missing
    file returns False. Never raises."""

def remove_entry(path: Path | None) -> None:
    """Delete the entry; a missing file or None is fine."""

@dataclass
class GradeJob:
    payload: Any                    # opaque to the queue; the session's own job data
    entry: Path | None = None       # its pending-entry file, if any
    queued_at: float = 0.0          # time.perf_counter() when accepted, set by the queue
    started_at: float = 0.0         # time.perf_counter() when the worker took it, set by the queue

class GradeQueue:
    def __init__(self, run: Callable[[GradeJob], None], *, depth: int = 3,
                 backlog: Iterable[GradeJob] = (), log: Callable[[str], None] = print) -> None: ...
    def submit(self, job: GradeJob) -> bool: ...
    @property
    def pending(self) -> int: ...
    def close(self, timeout: float | None = None) -> bool: ...
```

**Behaviour:**

- One daemon worker thread named `pifilm-grade`, started in `__init__`.
- The worker runs every `backlog` job first, in order, then jobs from `submit`, in order. `backlog` is not bounded by `depth`.
- `submit` blocks while `depth` jobs are waiting (the job being run does not count as waiting). It returns `True` when the job is accepted. After `close` has been called it returns `False` at once, and a `submit` that is blocked when `close` is called also returns `False` (no deadlock); the caller then still owns the job.
- `pending` is waiting jobs plus backlog not yet run plus the one running. It is 0 only when nothing is left to do. Safe to read from any thread.
- `run(job)` raising any `Exception`: the worker logs `grade failed: <exc>` through `log`, calls `note_failure(job.entry)` when the job has an entry, and carries on with the next job. `run` returning normally: the queue does nothing with the entry (the session removes it, after writing its record).
- `close(timeout)` stops accepting, lets the worker finish everything already accepted (and the backlog), and joins it. Returns `True` when the worker finished, `False` on timeout. Calling it twice is fine.
- `depth` below 1 raises `ValueError`.

**Tests (`tests/test_gradequeue.py`), each asserting real behaviour with `threading.Event`s, no sleeps longer than needed and a 5 s ceiling on every wait:**

1. Jobs run in submission order, each with `queued_at` and `started_at` set and `started_at >= queued_at`.
2. Backlog jobs run before submitted jobs.
3. With `depth=2` and a `run` held on an event: two more submits are accepted, a third blocks (checked from a helper thread), and it is accepted once the held job is released.
4. `pending` counts running + waiting + backlog, and returns to 0.
5. A `run` that raises: later jobs still run; the log got one `grade failed:` line; the entry's `attempts` is 1; a second failure of the same entry renames it `*.failed`.
6. `close()` finishes accepted jobs and returns `True`; `submit` afterwards returns `False`; a second `close()` returns `True`.
7. A `submit` blocked on a full queue returns `False` when `close()` is called, and `close()` still returns (the held job released by the test).
8. `close(timeout=0.05)` with a held job returns `False`.
9. `write_entry` then `read_entries` round-trips the payload with `version` and `attempts: 0`; the directory mode is 0o700; no temp file is left behind.
10. `read_entries`: missing directory gives `[]`; a file with invalid JSON, a JSON array, and an entry with `"version": 99` are each renamed `*.failed` and skipped; valid ones come back sorted by name.
11. `note_failure` on a missing path returns `False`; `remove_entry(None)` and a missing path do not raise.
12. `default_state_dir` honours `XDG_STATE_HOME` (monkeypatched) and falls back to `~/.local/state/pifilm/pending`.
13. `depth=0` raises `ValueError`.

Commit: `gradequeue: a bounded grade worker with pending entries on disk`.

---

### Task 2: the session grades behind the shutter

**Files:**
- Modify: `pifilm/capture/app.py` (`CaptureResult`, `CaptureSession`, `_announce`, `main` and its argument parser, the stop paths)
- Modify: `pifilm/capture/controller.py` (`ControllerSnapshot.pending_grades`, closing the session's grades)
- Test: `tests/test_app.py`, `tests/test_controller.py` (find the existing controller tests' file by name; add there)

**Consumes:** Task 1's `GradeQueue`, `GradeJob`, `write_entry`, `read_entries`, `remove_entry`, `default_state_dir` exactly as specified above.

**Produces (Task 3 relies on these):**
- `ControllerSnapshot.pending_grades: int` (default 0), the last field of the dataclass.
- `CaptureSession.pending_grades -> int` property (0 when grading inline).

**`CaptureSession` changes:**

- New keyword arguments, all with defaults that keep today's behaviour: `background: bool = False`, `state_dir: Path | None = None` (None means `default_state_dir()`), `queue_depth: int = 3`, `log: Callable[[str], None] = print`.
- `background=False`: no queue, no state directory touched, no new record fields. Exactly today's code path.
- `background=True`: the session owns a `GradeQueue(self._run_grade, depth=queue_depth, backlog=<recovered jobs>, log=log)`.
- `CaptureResult` gains `graded: Path | None = None`: where the graded file is or will be. Inline: equal to `pifilm`. Background: the final path, while `pifilm` is the quick grade. Exposure 1 of a double: `None`.

**A background single shot (`capture`)**, in this order:

1. Read the frame, start `stages` (unchanged).
2. Draw the grain seed (same call and same position in the seed stream as today, so a seeded session gives the same seeds inline and in background).
3. `_write_original` (unchanged).
4. Build the capture-time record fields: everything in today's record that does not come from grading (`timestamp`, `original`, `pifilm` (the graded file's name), `frame_source`, `shutter_us`, `iso`, `colour_temperature_k`, `grain_seed`, `params_version`, `package_version`, the stream info, `camera_metadata`, `dng`) plus `ev_comp`.
5. `write_entry(state_dir, f"{day_dir.name}_{stem}", {...})` with: `out_root` (absolute string), `day` (`day_dir.name`), `graded` (file name), `originals` (list of paths relative to `out_root`, one item), `seed`, `ev`, `double` (None), `record` (step 4's dict).
6. The quick grade, timed as stage `preview`: reduce the frame by the integer factor `max(1, round(max(height, width) / 1014))` with `cv2.resize(..., interpolation=cv2.INTER_AREA)` (factor 1: no resize), `pipeline.process(small, grain=False, ev=ev)`, `save_jpeg` into the session's private temporary directory (the one `_exposure_card` uses; reuse its create-if-gone logic) as `preview-{stem}.jpg`. Keep only the 16 newest `preview-*.jpg` there.
7. `queue.submit(GradeJob(payload=<frame, seed, ev, paths, record fields, stages, shutter time>, entry=<step 5's path>))`. If it returns `False` (the session is closing), grade inline right there with the same function the worker uses, so the photograph is still finished.
8. Return `CaptureResult(original, preview_path, record, graded=graded_path)` where `record` is step 4's dict plus `shutter_to_ready_ms` (shutter to now) and `stage_ms` (a copy of the stages so far).

**The grade side (`_run_grade(job)`, on the worker):**

1. If the job carries no frame (a recovered entry), load it: `load_rgb(out_root / originals[0])`; for a double, load both and `composite` them. A missing original raises (the queue then counts the failure).
2. Stage `queued` = `started_at - queued_at` in ms (0.0 for a recovered job). For a double, stage `composite`.
3. `pipeline.process(rgb, rng=np.random.default_rng(seed), ev=ev, timings=stages)`; `pipeline_ms` as today.
4. Stage `save_graded`: `save_jpeg(graded, graded_path)`.
5. Append the record: capture-time fields + the pipeline's info + `pipeline_ms` + `shutter_to_ready_ms` + `shutter_to_saved_ms` (shutter to now) + `stage_ms` + `"grade": "background"`. Field order: keep today's order for the fields that exist today; new fields after them. A recovered job has no shutter time: it omits `shutter_to_ready_ms`, `shutter_to_saved_ms` and `pipeline_ms` keeps its meaning, and it carries `"recovered": true`.
6. `remove_entry(job.entry)`.

Before step 3 of a recovered job, if `captures.jsonl` in the day folder already has a line whose `pifilm` equals this graded name, the grade had finished and only the entry was left behind: remove the entry and return without grading (no duplicate record).

**Double exposure in background:** exposure 1 is unchanged (nothing graded, card shown, no entry). Exposure 2: the quick grade is `composite` of the two reduced frames, then `process(..., grain=False, ev=0.0)`; the entry has two `originals` (exposure 1 first), `ev` 0.0 and `double: {"ev_comp": [ev1, ev2], "method": COMPOSITE_METHOD}`; the job holds both full frames; the grade side composites at full size and grades at `ev=0.0`. The final record is today's exposure-2 record plus the new fields. The pending frame is taken off the session before anything that can fail, as today.

**Recovery (in `__init__`, background only):** `read_entries(state_dir)`; entries whose `out_root` is not this session's absolute `out_root` are left alone; the others become backlog jobs without frames, in name order. Log one line `recovering N unfinished grade(s)` when N > 0.

**Other session members:**
- `pending_grades` property: the queue's `pending`, 0 inline.
- `close_grades(timeout: float | None = None) -> bool`: close the queue (True inline). Idempotent.

**`_announce`:** when the record has no `wb_gains` (a background result), print `Saved {original name} in {shutter_to_ready_ms:.0f} ms; grading {graded name} in the background`. Inline output unchanged.

**Controller:** `ControllerSnapshot.pending_grades` filled in `snapshot()` from `getattr(session, "pending_grades", 0)`, guarded so a session that raises or returns a non-int gives 0. In the worker's `finally`, before the camera is closed, call the session's `close_grades()` when it has one (exceptions swallowed, the camera must still be closed).

**`main`:** new option `--grade {background,inline}`, default `background`, help text in the file's style; `--state-dir PATH` (default: `default_state_dir()`), help says it holds unfinished grades. Both `CaptureSession(...)` call sites pass `background=args.grade == "background"`, `state_dir=...`. Anywhere `main` builds a session without a controller that closes it, call `close_grades()` on the way out.

**Tests to add (existing tests must stay green untouched):**

1. Background shot with a seeded session: after `close_grades()`, the graded file exists and is byte-identical to the graded file of an inline session with the same seed and the same `FakeCamera` frame.
2. The result returns before the grade finishes: with the pipeline's `process` for full-size frames held on an event, `capture()` returns, `result.pifilm` exists and is a readable JPEG in a directory outside `out_root`, `result.graded` does not exist yet, `pending_grades == 1`; after release and `close_grades()`, `result.graded` exists and `pending_grades == 0`.
3. The final record: has every key the inline record has, plus `shutter_to_ready_ms`, `grade == "background"`, `stage_ms` containing `preview`, `queued`, `normalize`, `lut`, `grain`, `save_graded`; `shutter_to_saved_ms >= shutter_to_ready_ms`. The inline record has none of the new keys.
4. Records are appended in shot order for three quick shots.
5. The pending entry exists while the grade is held and is gone after it finishes.
6. Recovery, single: write originals and an entry by hand (or kill a held session by never releasing it and building a second session on the same `out_root` and `state_dir`); the new session grades it, the record has `recovered: true` and no `shutter_to_saved_ms`, the entry is gone.
7. Recovery skips an entry whose graded name is already in `captures.jsonl`: entry removed, no second line.
8. Recovery leaves an entry for another `out_root` in place, and a missing original counts a failure (attempts 1) without stopping the session from taking a new shot.
9. Background double exposure: exposure 1 writes no entry and no graded file; exposure 2's quick grade exists; after `close_grades()` the `_double_graded.jpg` is byte-identical to the inline one for the same frames and seed, and the record has `double.index == 2` and `grade == "background"`.
10. A full queue makes `capture()` wait: `queue_depth=1`, grades held; the third capture blocks until one is released.
11. `submit` refused because the session is closing: the shot is graded inline and its record written.
12. Quick grade size: a 4056 x 3040 frame gives a 1014 x 760 preview; a 640 x 480 frame is not reduced.
13. Only the 16 newest previews are kept.
14. A grade that raises (make `save_jpeg` fail for the graded path once): the log has `grade failed:`, the entry has `attempts == 1`, the next shot still grades.
15. Controller: `snapshot().pending_grades` follows the session; a session without the property gives 0; `close()` calls `close_grades()` before closing the camera, and still closes the camera when `close_grades` raises.
16. CLI: `--grade inline` and the default reach `CaptureSession` as `background=False` / `True`; `--state-dir` reaches it (follow how the existing CLI tests inspect session construction).
17. `_announce` on a background result prints the "grading ... in the background" line; inline output unchanged.

Check that `Pipeline.process` is safe to call from two threads at once (the quick grade on the capture worker while the grade worker runs a full frame): read `pifilm/pipeline.py`, `pifilm/lut.py` `apply_pillow` and `pifilm/normalize.py` for shared mutable state, and say what you found in the report. If something is shared, guard it and add a test.

Commits: one for the controller, one for the session and CLI (or more, small).

---

### Task 3: the developing count on the live view, and the docs

**Files:**
- Modify: `pifilm/display/liveview.py`, `pifilm/display/viewfinder.py`
- Test: `tests/test_liveview.py`, `tests/test_viewfinder.py` (use the files where the existing tests of these modules live)
- Modify: `docs/lcd-viewfinder.md`, `docs/how-it-works.md`, `CLAUDE.md`, `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`
- Do NOT modify `README.md` (the owner approves README text from a draft first).

**Consumes:** `ControllerSnapshot.pending_grades: int` from Task 2.

**Live view:**
- `LiveState` gains `developing: int = 0` (last field).
- When `developing > 0`, `render_live` draws that number in the amber the outlines use, in the bundled Barlow Condensed font at the size of the other small readouts, immediately left of the battery gauge, vertically centred on it. It must not overlap the battery, the `2x` mark or anything else at 320 x 240 base units, for one and two digits; numbers above 99 draw as `99`. At 0 nothing is drawn and the frame is pixel-identical to today's.
- Read the module first: positions are in base units scaled by `scale`, tiles are cached; follow both.
- `ViewfinderLoop` passes the controller snapshot's `pending_grades` into `LiveState` on every live frame (`getattr(snapshot, "pending_grades", 0)`, since tests use fake controllers).

**Tests:**
1. `developing=0` renders exactly as before (compare with a render of a state without the field set).
2. `developing=3` changes pixels only inside a box left of the battery (assert the bounding box of the difference lies within the expected region and does not intersect the battery's or the `2x` mark's rectangles), at `scale` 1 and 2.
3. `developing=12` and `developing=250` (drawn as 99) stay inside that region.
4. The viewfinder passes the count through: a fake controller snapshot with `pending_grades=2` leads to `LiveState.developing == 2` in the rendered state (follow how existing viewfinder tests capture the state or the frame).

**Docs (plain, short sentences, the files' own style; no em dashes):**
- `docs/lcd-viewfinder.md`: in the screen-layout section, the developing count and what it means (safe to switch off at zero; a full queue of three is why the shutter may wait). In the hardware-acceptance section, add the seven acceptance steps from the spec as a table like the existing ones, results left empty.
- `docs/how-it-works.md`: where the capture pipeline is described, the new order of a shot (original saved, quick grade shown, full grade behind the shutter, record written when the grade is saved), the pending entries and recovery, and `--grade inline`.
- `CLAUDE.md`: in the Capture section, `gradequeue.py` and the session's two sides; `--grade`, `--state-dir`; the new record fields; in the Display section, the developing count. Keep entries as dense as their neighbours.
- Roadmap: under step 6, a short "Built" note pointing to the spec; Pi numbers are added later by the controller of this plan, leave a line `Pi numbers: to be measured.`

Commits: one for the live view, one for the docs.
