# Background grading: design

Date: 2026-10-13. Step 6 of `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`.
Branch `feat/background-grade`, pull request into `pipeline-improvements`.

## Goal

The camera is ready for the next photograph as soon as the frame is safe on disk
and there is something to show. The full-resolution grade happens behind it.

Today (Pi 4, IMX477, tile grain, 2026-10-13) a shot holds the camera for 3,651 ms:

| Stage | Time | After this change |
| --- | --- | --- |
| Capture (camera request, array copy, DNG) | 1,099 ms | waits, as now |
| Save original JPEG | 343 ms | waits, as now |
| Save DNG | 27 ms | waits, as now |
| Quick grade for the screen and the Stick | new, about 150 ms (estimate) | waits |
| Normalise, LUT, grain at full resolution | 1,804 ms | behind the shutter |
| Save graded JPEG | 371 ms | behind the shutter |

Expected wait between photographs: about 1.6 s instead of 3.65 s. To be measured.

The graded file itself does not change: same pipeline, same input, same seed.

## What happens on a shot

1. Read the full-resolution frame (unchanged).
2. Write the original JPEG and the DNG (unchanged). From here the photograph
   survives anything.
3. Write a **pending entry** for the grade (see "Power loss").
4. Make the **quick grade**: the frame reduced four times each way (1014 x 760
   for the IMX477), through the same pipeline without grain, saved as a JPEG in a
   private temporary directory (where the "Exposure 1/2" cards already live, so
   the Nextcloud sync never sees it).
5. Hand the full frame to the **grade queue** and finish the capture job. The
   job's picture is the quick grade.
6. The grade worker, one thread, takes jobs in order: normalise, LUT, grain, save
   `*_graded.jpg`, append the record to `captures.jsonl`, delete the pending entry.

The review screen and the Stick need no change: both show the job's picture, which
is now the quick grade. The Stick's protocol and firmware are untouched; it simply
gets its picture sooner.

The quick grade is not the final file. It has no grain, and its normalisation is
measured on the reduced frame, so tone can differ slightly from the graded file.
On a 640 x 480 screen and a 240 x 135 thumbnail neither is visible.

## The grade queue

- One worker thread, `pifilm-grade`. Jobs finish in the order they were shot, so
  `captures.jsonl` stays in shot order.
- Each waiting job holds its 12 MP frame in memory, about 37 MB (74 MB for a
  double exposure, which holds two).
- **Depth: 3 waiting jobs.** When the queue is full the next capture waits for a
  place, with the processing screen showing, exactly like a slow shot today. The
  camera can never get further ahead than three photographs, and memory stays
  under about 150 MB (220 MB if all are doubles).
- Shooting as fast as possible, a grade takes longer than the wait between shots
  (about 2.2 s against 1.6 s), so a long burst settles to the grading rate after
  the first few frames. Single shots and short bursts get the full gain.

## Double exposure

- Exposure 1 is unchanged: original saved, frame kept, card shown, nothing graded.
- Exposure 2: the quick grade is the composite of the two reduced frames. The
  full-resolution composite and its grade go to the queue as one job holding both
  frames.

## Records

One line per photograph in `captures.jsonl`, written when its grade is saved, with
the fields it has today. Changes:

- `shutter_to_ready_ms` (new): shutter to the camera being free again. This is
  the number the photographer feels.
- `shutter_to_saved_ms` keeps its meaning: shutter to the graded file on disk. It
  now includes time spent waiting in the queue.
- `stage_ms` gains `preview` (the quick grade) and `queued` (waiting for the
  worker).
- `grade: "background"`, so a reader of old and new records can tell them apart.

`pifilm-process` is unaffected: it reads the same fields as before.

## Power loss and crashes

Before a job is queued, a small JSON file is written to a state directory outside
the photo folder (`~/.local/state/pifilm/pending/`): the original's path (both
originals for a double), the grain seed, the EV, the camera fields of the record,
and the name the graded file will have. It is deleted when the record is written.

At start-up the service looks for pending entries and finishes them on the same
worker before new jobs: it loads the original JPEG from disk and grades it with
the stored seed, and the record carries `recovered: true`. A recovered grade starts
from the saved JPEG rather than the frame in memory, so it can differ from what
the uninterrupted grade would have been by JPEG rounding; that is why it is marked.

An entry that fails twice is renamed `*.failed` and reported in the log, so one
unreadable file cannot keep the service busy at every start.

What can be lost: nothing that was acknowledged. A photograph whose original was
not yet written when the power went is lost, as today.

## Stopping

- A normal stop (systemd, the X728's low-battery shutdown) finishes the queue
  first: at most three grades, under ten seconds.
- The camera-stall exit keeps its 10 s deadline. Anything not graded by then is
  recovered at the next start.

## On the screen

While grades are waiting, the live view shows a small amber number beside the
battery: how many photographs are still developing. It disappears at zero. This
tells the photographer when it is safe to switch off, and that a full queue is
why the shutter is waiting.

## Switching it off

`--grade inline` restores today's behaviour (grade before the camera is free);
`--grade background` is the default. The switch exists for comparison on the
hardware and as a way back if the background load upsets the live view.

`CaptureSession` grades inline unless it is given a queue, so the existing tests
and the other loops keep their behaviour until the app wires the queue in.

## Code

- `pifilm/capture/gradequeue.py` (new): `GradeQueue` (worker, bounded queue,
  `submit`, `pending`, `drain`), `GradeJob` (frames, seed, EV, paths, record
  fields, capture-side timings), the pending-entry files and start-up recovery.
  No camera and no display in it; tested with a fake pipeline.
- `pifilm/capture/app.py`: `CaptureSession.capture` and `_capture_double` split
  into "capture side" and "grade side"; the grade side is what the queue runs.
  The `--grade` flag. Stop paths drain the queue.
- `pifilm/capture/controller.py`: `ControllerSnapshot.pending_grades`, read from
  the session like the double-exposure state.
- `pifilm/display/liveview.py`, `viewfinder.py`: the developing count.
- Docs: `docs/lcd-viewfinder.md`, `docs/how-it-works.md`, `README.md` pipeline
  section (draft to the owner first), `CLAUDE.md`.

## Risks to check on the hardware

- **Live view under load.** The grade now runs while the live view is drawing.
  The Pi 4 has four cores and the heavy work releases Python's lock, but the
  preview may stutter while a grade runs, and the known preview stall
  (`docs/known-issues.md`) must not become more frequent.
- **A capture during a grade** shares the CPU and memory bus with it, so both
  get slower. The record's numbers will show by how much.
- **Memory.** Three waiting frames, the frame being graded and its working copies
  peak near 400 MB. Fine on a 4 GB or 8 GB Pi 4; to be confirmed on this one.

## Acceptance (hardware)

1. A single shot: review appears, camera free in about 1.6 s; the graded file
   appears a few seconds later; the record has `shutter_to_ready_ms`.
2. Five shots as fast as the camera allows: all five graded files and records
   present, in order; the developing count rises and returns to zero.
3. A double exposure: card, then the composite's review; one graded file.
4. Stick: picture arrives after each shot, as before.
5. Power cut with the count above zero (pull the supply with the UPS off, or
   `kill -9`): after restart the missing graded files appear, records marked
   `recovered`.
6. `--grade inline`: behaves as before this change.
7. Ten minutes of ordinary use: no preview stall, live view acceptable.

## Not in this step

- GPU grading (step 7). It replaces what the worker runs; the queue stays.
- Moving the original's JPEG encode behind the shutter. It would save a further
  0.34 s, at the price of a window in which only the DNG (when enabled) is on
  disk. Left as it is; can be revisited with numbers.
- A graded live view.
