# Known issues

Understood-but-unfixed defects and hardware limitations. Each entry records the
symptom, the verified root cause, what was ruled out, the current decision, and
how to confirm it is resolved.

## IMX708 second-capture "Camera frontend has timed out" on the Raspberry Pi 3B

**Status:** **resolved 2026-09-19** by the move to the Raspberry Pi 4. No code
change was needed and native 12 MP capture was kept — see
[Resolution](#resolution) for the measurements. The entry stays because the
failure mode is instructive and a Pi 3B is still affected.

**First seen:** 2026-09-18, Raspberry Pi 3B + Camera Module 3 Wide (IMX708),
`pifilm-capture` on `main`.

### Symptom

The first capture succeeds; the second (and later) captures log a libcamera
frontend timeout and can stall:

```
WARN  V4L2 /dev/video0[..:cap]: Dequeue timer of 1000000.00us has expired!
ERROR RPI  Camera frontend has timed out!
ERROR RPI  Please check that your camera sensor connector is attached securely.
ERROR RPI  Alternatively, try another cable and/or sensor.
```

In an interactive run both photos still saved (~20 s each); under the systemd
service / remote (Stick) path the stall escalated to a capture that never
returned (observed waiting >150 s).

### Root cause (verified)

It is **CPU/processing time at full resolution**, not memory and not the DNG:

- The camera is configured for the IMX708's native **4608×2592 (~12 MP)** with an
  RGB grading stream plus a raw stream (`pifilm/capture/picamera.py`,
  `create_still_configuration(..., buffer_count=2)`).
- Grading one 12 MP frame — normalize → 3D LUT → grain — takes **~20 s on the
  Pi 3B** (measured: `Saved … in 20394 ms`). During that grade the capture loop
  does not service the camera, so libcamera's 1-second dequeue watchdog on the
  CSI frontend expires and logs "frontend has timed out". The free-running camera
  then stalls; sometimes it recovers (the photo still saves), sometimes the next
  `capture_request()` blocks — the >150 s hang.

### Ruled out during diagnosis

- **DNG / raw write.** Reproduced with `--no-dng`; the raw stream is still
  configured and the second capture still timed out, so writing the DNG is not
  the trigger.
- **CMA / RAM.** The 256 MiB CMA pool had ~68 MiB free and `dmesg` showed no
  allocation failures; both photos saved. Memory pressure is not the cause.

### Environment

Raspberry Pi 3B, Raspberry Pi OS Trixie, libcamera 0.7.2, picamera2 0.3.37,
IMX708 (`imx708_wide`) at 4608×2592, `buffer_count=2`, CMA 256 MiB shared with
`vc4-kms-v3d`.

### Resolution

Measured on a Raspberry Pi 4 Model B running the migrated SD, 2026-09-19, with
the same 4608×2592 configuration and `buffer_count=2`:

| Metric | Pi 3B | Pi 4 |
|---|---|---|
| Grade one 12 MP frame (`pipeline_ms`) | ~20,400 ms | **~3,290 ms** |
| Shutter to saved (`shutter_to_saved_ms`) | — | **~5,300 ms** |
| `Camera frontend has timed out` | on the second capture | **0 across every capture in the boot** |

The Pi 4 grades a 12 MP frame roughly six times faster, so the capture loop no
longer leaves the camera unserviced long enough to trip libcamera's one-second
dequeue watchdog. Repeated captures complete with consistent timings
(3286.7 / 3284.8 / 3303.7 ms across three shots).

Native 12 MP is therefore kept, and the configurable grading resolution
considered during diagnosis was **not** needed.

### If it regresses — the fallback

Stop grading at full 12 MP: make the capture/grading resolution configurable (for
example a half-res `2304×1296` binned stream, or `1920×1080`), keeping full
resolution only for the saved original and DNG. Grading 2–3 MP is several times
faster, which closes the camera-servicing gap that trips the watchdog. Raising
`buffer_count` and/or CMA is a secondary lever where there is room.

## Picamera2 stops handing over preview frames on the Raspberry Pi 4 (viewfinder freezes)

**Status:** **cause unknown; contained 2026-10-10.** The freeze no longer hangs
the process: it is logged with Picamera2's state and the service restarts
itself. Seen twice.

**First seen:** 2026-10-10, Raspberry Pi 4 + IMX477, `python3-picamera2`
0.3.37-1, `python3-libcamera` 0.7.2+rpt20260817-1, the DSI viewfinder with the
GPIO shutter button, about an hour after boot and twenty to thirty seconds
after a double exposure had saved normally. Nothing on the screen had been
tapped.

### Symptom

The live image froze. A shutter press logged `button: capture requested` and
nothing followed. The process stayed up (SSH worked, systemd reported
`active (running)`), so nothing restarted it.

### What was measured on the frozen process

- `py-spy dump`: the viewfinder thread was inside Picamera2's
  `capture_request()`, waiting for the next preview frame while holding the
  camera lock; the capture worker was waiting for that lock.
- The sensor was still streaming: `unicam_capture0` in `/proc/interrupts`
  counted 594 interrupts in 3 seconds.
- Every thread in the process was at 0.0% CPU, libcamera's `CameraManager`
  threads included.
- No libcamera error in the journal, no kernel message, `throttled=0x0`,
  1.8 GB of memory free.

So frames were arriving at the Pi and nothing was consuming them, which is how
the camera looks when libcamera has no buffers queued. Why the preview buffers
stopped being recycled is not known.

### Ruled out

- The sensor, its ribbon cable and the power supply (interrupts still counting,
  no undervoltage).
- The display, the touch panel and the button: their threads were idle and
  healthy.
- A lost photo: the last capture's files were complete.

### Containment

`Picamera2Camera.read` now waits a bounded time for its frame (5 s for a
preview frame, 30 s for a still, plus four times a fixed shutter) and raises
`CameraStalled`. The log line carries Picamera2's state:

```
camera: no preview frame within 5.0 s (started=True completed=0 frames=41234)
```

After three stalled preview reads in a row the viewfinder ends and
`pifilm-capture` exits 1 with
`error: camera stopped delivering frames: ...; exiting for a restart`, so
systemd (`Restart=on-failure`) reopens the camera. If closing the stalled
camera itself hangs, the process is ended after 10 seconds regardless. A shot
requested during the stall is lost. By the timeouts the camera should be back
about 20 seconds after the freeze; that has not been timed on the Pi, because
the stall cannot be produced on demand.

Only the LCD viewfinder has this exit. In the headless Stick-only mode a
stalled still fails that one capture with the same message and the process
keeps running.

### Second occurrence

2026-10-13, about 18:38, during the GPU grading test (PR #60): the same three
`no preview frame within 5.0 s (started=True completed=0 ...)` lines, then the
exit and a restart by systemd, which worked. It came about 7 s after the
viewfinder restarted the live view for a white-balance change; no grade was
running. The GPU's self-test had passed at that start and at the three before
and after it. Whether GPU grading or the live-view restart makes the stall more
likely is not known from two occurrences.

### The journal is kept across reboots (2026-10-15)

Raspberry Pi OS keeps the journal in memory (`Storage=volatile` in
`/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf`), so a reboot or a
power cut lost every stall logged before it. That happened on 2026-10-15: several
restarts during outdoor use, and nothing left to read once the Pi had been switched
off and on. `scripts/pi_journal.py` keeps it on the card instead (a drop-in that
sorts after the OS's, capped at 200 MB and always leaving 1 GB free). Once, on the
Pi:

```bash
cd ~/repos/pi-film-reversal
python3 scripts/pi_journal.py              # prints the plan, changes nothing
sudo python3 scripts/pi_journal.py --apply
```

After restarts, every boot's starts, stalls and error exits, the lines before each
exit, and any undervoltage or out-of-memory message from the kernel (no sudo
needed; the user is in `adm`):

```bash
python3 scripts/pi_journal.py --report
```

`sudo python3 scripts/pi_journal.py --undo --apply` goes back to memory only.

### If it happens again

The restart destroys the evidence, so the log line is the record: note
`started`, `completed` and `frames` from it, and what the camera had done in
the minute before.

```bash
journalctl -u pifilm-capture.service --since "-10 min" --no-pager | grep -B 30 "no preview frame"
```

## Live white-balance changes are ignored after a photo (Pi 4, IMX477)

**Status:** **cause unknown; contained 2026-10-12** by restarting the live view
when a white-balance change has not taken effect. The wrong colour lasts about a
second and a half and the picture blinks once; the root cause is not fixed.

**First seen:** 2026-10-12, Raspberry Pi 4 + IMX477, libcamera 0.7.2,
Picamera2 0.3.37, `pifilm-capture --display waveshare35dsi`.

### Symptom

A white balance is selected on the viewfinder (`WB` cell, `+`/`-`) and the live
view keeps the old colour, once for over a minute and sometimes until a photo is
taken. The photo itself is right: each still re-applies the whole configuration.
Returning `WB` to `AUTO` fails the same way.

### What was measured

From the viewfinder's own journal lines (`control:` after each step, `camera:`
when a frame's metadata disagrees with the selection), on 2026-10-12:

- Before the first photo after start-up, a live white-balance change is applied
  within 2 s: the camera reports the new value.
- After a photo or a restart of the live view, that is after any mode switch, a
  live white-balance change (to a fixed value or back to auto) is usually not
  applied: 11 of 11 first sends were ignored.
- Sending the controls again cured 2 of about 7.
- Restarting the live view cured it every time (3 of 3).
- A live shutter change after a photo was applied within 2 s (one observation).
- In a stand-alone script using the same `Picamera2Camera` class, live changes
  after a photo were applied and the picture's R/G and B/G followed, so the
  camera and the ISP work in isolation.
- A 300-change stress test of Picamera2 alone lost none.
- A first containment that allowed one restart per 60 s left the user stuck in
  the wrong colour after the first restart.

### Ruled out

- **The display path.** The picture ratios in the journal (the means of the
  preview frame's channels, R/G and B/G) agree with the value the camera reports,
  so the screen shows what the camera delivers.
- **A lost update in Picamera2 as the sole cause.** Sending the same controls
  again rarely helps.

### Containment

`ViewfinderLoop._verify_controls` (`pifilm/display/viewfinder.py`) compares each
new frame's metadata with the fixed controls. A white balance that disagrees for
1.5 s and 6 camera frames (4 s for a return to auto, which libcamera's own
damping holds still for about 2 s) restarts the live view through
`Picamera2Camera.restart_preview`, a `switch_mode` to the preview configuration,
which already holds the selection. It is not resent first. Shutter and ISO are
resent after 3 s and 6 frames and the live view is restarted after two resends
without effect. There is one restart per control per selection, and never two within
5 s; a step that changes that control, or leaving the live view, allows another,
and agreement in between does not. The auto case restarts at most once per return
to auto: after it auto is not watched again until a white balance is next fixed,
because on a capped lens or a dark scene auto reports the old value for good. What the user sees and
the exact lines are in
[`lcd-viewfinder.md`](lcd-viewfinder.md#manual-controls).

Run on the Pi 4 on 2026-10-12 with this containment in place: four white-balance
changes made after a photo (to 4000 K, 5600 K, 4000 K and 6500 K) were each
reported `not in the live view after 1.5 s`, restarted once, and reported
`applied after restarting the live view` within a second, with the picture's
R/G and B/G following (for example 1.53 / 0.66 to 2.09 / 0.04 for 2533 K to
4000 K). Shutter changes after a photo applied live with no restart. With the
frame thread, taps at a 1 s shutter answered at once and the journal kept
about 10 fps at fast shutters.

### If someone investigates further

Collect the journal around a failure:

```bash
journalctl -u pifilm-capture | grep -E 'control:|camera:'
```

- `control: WB 7500K selected; camera reports ...; picture R/G ... B/G ...` is
  written about 2 s after every step: what was selected, what the camera
  reported, and what the picture looked like.
- `camera: ... not in the live view after ...; restarting the live view` and
  `... resending` are written when the camera's report disagrees, and
  `camera: ... applied after ...` or `... not applied even after restarting the
  live view` say how it ended.

The open question is what differs between the app and the stand-alone script
that took live changes after a photo.

## A libcamera update could silently retune a deployed trained LUT

**Status:** checked at start-up, warn only (step 4 of the
[pipeline roadmap](superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md)).
The check works only for a look stamped with its tuning's fingerprint.

**Description:** a trained LUT is fitted against the colour science its source
corpus was captured under, which is one libcamera tuning file. If that file
changes (an OS or libcamera package update, or a sensor read differently), the
camera feeds a different colour response through an unchanged LUT. There is no
error and no changed `lut_sha1`. The Picamera2 backend defaults `--tuning-file`
to libcamera's automatic choice for the detected sensor (`tuning_file:
auto:<model>` in `captures.jsonl`); the example systemd unit runs with this
default.

**The check.** The Pi fingerprints the tuning it opens the camera with: the
SHA-1 of the file's parsed JSON, so a change of whitespace or key order does not
count and any changed value does. A look can carry the fingerprint it was
trained under, as `tuning_sha1` in its `params.json`. At start-up the service
compares the two and writes one line to the journal:

- `tuning: matches the look (<12 hex>)`: nothing to do.
- `tuning: DIFFERS from the look: camera <12 hex> (<tuning file>), look <12 hex>; colours may be off`:
  the amber `TUNING` tag shows on the viewfinder for the whole run (see
  [docs/lcd-viewfinder.md](lcd-viewfinder.md#tuning-warning)).
- `tuning: cannot check the look's tuning (<reason>)`: the camera's tuning
  could not be read, so nothing was compared.

Nothing is checked, and no line is written, for a look without a fingerprint
(the bundled starter, or a trained look not yet stamped) or for a USB camera.
The service never refuses to start over this: photographs are still taken and
graded, and the originals are kept, so they can be regraded later. From this
build on, every record from the Pi camera carries `tuning_sha1` when the tuning
could be fingerprinted, and when the check ran also `tuning_matches` (`true` or
`false`).

**What to do.** When you deploy a trained look, pin `--tuning-file` to the file
the corpus was captured under and stamp the look on the Pi (see
[docs/training.md](training.md), Step 7):

```sh
pifilm-tuning --tuning-file imx477.json --stamp <the service's artifacts dir>
sudo systemctl restart pifilm-capture.service
```

Run the stamp as the service user, not with `sudo`; only the restart needs it.
Stamp the Mac's copy of the look as well (`pifilm-tuning --sha1 <hex> --stamp
<dir>`), or a later `rsync` from the Mac puts back the unstamped `params.json`
and the check goes quiet.

If the tag shows after an update, the tuning file changed under the look.
Either put the old tuning file back and name it with `--tuning-file`, or retrain
the look on frames taken under the new tuning. Stamping the look again with the
new fingerprint only hides the warning; it does not make the colours right.

## Capture records written before 2026-09-19 cannot be told apart by `lut_sha1`

**Status:** understood, not fixed retroactively. New records carry the missing
field; old ones cannot be rewritten without guessing.

**First seen:** 2026-09-19, reviewing the Phase 6 normalisation freeze
([write-up](experiments/2026-09-19-imx708-normalisation.md)).

### Symptom

Two `captures.jsonl` records — one from before the Phase 6 freeze, one from
after — can carry the **same `lut_sha1`** and still have been graded
differently. The bundled starter's LUT did not change in Phase 6; its
normalisation did (`white_balance` true -> false,
`levels_lift_highlight_ref` null -> 0.02).

### Root cause

`lut_sha1` is the content hash of the `.cube` only. Normalisation lives in the
artifact's `params.json` and was never hashed into the record, so the record
identified half of the grade. `Pipeline.process` now also records
`normalize_sha1` (a SHA-1 of `NormalizeParams.to_dict()`, sorted keys), and
`CaptureSession` writes it into every line — but only for captures taken from
2026-09-19 onward.

### Workaround for old records

The record still carries the *applied* `wb_gains`, `exposure_gain` and
`levels`, which is what the normaliser actually did to that frame. Feeding
those values back reconstructs the grade without knowing which parameter set
produced them; the missing piece is only the parameter identity, not the
result.

### How to confirm it is resolved for new captures

`normalize_sha1` is present in every new `captures.jsonl` line
(`tests/test_app.py::test_log_line_carries_full_provenance`) and differs
between two artifacts whose normalisation differs
(`tests/test_pipeline.py::test_normalize_sha1_identifies_the_normalisation`).

## `pifilm-process` regrades without the capture's EV compensation

Since 2026-09-24 the capture path re-applies the viewfinder's EV compensation
after normalisation (see the `pifilm/pipeline.py` docstring) and records it as
`ev_comp` in `captures.jsonl`. Batch regrading with `pifilm-process` does not read
the log, so it grades every original at EV 0: a shot taken at −2 EV regrades to
the same brightness as a plain one. Captures before 2026-09-24 have no `ev_comp`
field and were graded at EV 0 whatever the buttons said.

Workaround: regrade such a file from Python with
`Pipeline(Artifacts.default()).process(rgb, ev=<ev_comp>)`.

## Multi-frame merge: limits not yet checked on the Pi's frames

The merge (`pifilm/stack.py`) was tuned on synthetic frames. Three limits are
known and left as they are until real bursts show whether they matter.

- **Ghosts are found by luma only.** A moving subject is caught by its brightness
  difference from the reference. Something that changes colour at the same
  brightness (a red and a green car of equal luma passing) is averaged and can
  appear faintly twice.
- **`MIN_CORRELATION` (0.5) is untuned.** A frame whose ECC fit ends under it is
  dropped. A scene with little texture (sky, a plain wall) at high gain may fall
  under it, and the photograph is then the reference alone or a merge of fewer
  frames. This is a safe failure: noisier, never misaligned. The record's
  `stack.aligned` shows which frames were used.
- **Ghost weighting limits the noise drop at high gain.** `GHOST_SCALE` (0.05
  linear luma) assumes the noise between two frames is a few hundredths. At high
  gain in the dark the noise can approach it, so frames weigh less than 1 where
  only noise differs, and the noise falls by less than 1/√n.

How to check: take multi-frame shots of a still scene in dim light (high gain) and
of a plain wall, read `stack.aligned` and compare the merge's noise with a single
frame at the same exposure. Changing either constant changes pixels, so a
regrade of older records would differ.
