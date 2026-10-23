"""``pifilm-capture``: live preview, press SPACE, get two JPEGs.

Structure
---------
``CaptureSession`` owns the camera, the pipeline and the output folder, and
knows how to take one capture or produce one preview frame. Two thin loops
drive it, both taking injectable key sources so the whole flow is testable
with ``FakeCamera``:

* ``run_preview_loop`` draws the graded feed, or only saved captures, in an OpenCV window.
  Capture display mode reads the camera only when SPACE is pressed. Every GUI
  call is inside the guard, because a build without GUI support can fail at
  ``imshow`` rather than at ``namedWindow``, and a failure there must fall
  back to headless rather than end the session.
* ``run_headless_loop`` reads single keys from the terminal.
* ``--display`` runs the LCD viewfinder (``pifilm/display/viewfinder.py``) on the
  Pi's SPI panel instead of any OpenCV window. It shares one
  ``CaptureController`` with the remote API, so a tap on the panel and a Stick
  request are the same kind of job and never two owners of the camera.
* ``--shutter-gpio`` adds a GPIO push button (``pifilm/capture/button.py``) as one
  more trigger through that shared ``CaptureController``; the remote API or the
  button each select the shared-controller loops.

A dropped frame prints and continues in both loops. The spec promises the
session survives frame read failures, and that promise is only worth
anything if it also holds while the preview is running.

Capture semantics
-----------------
``SPACE`` acquires one fresh frame and saves exactly that frame; the
displayed frame is not re-used. ``capture()`` calls ``camera.read()`` once,
so the saved original and the graded image always come from one acquisition.

Output layout
-------------
``OUT/YYYY-MM-DD/HHMMSS_original.jpg`` holds the original for the shot.
For the USB camera (V4L2) it is the camera's own JPEG bytes, written verbatim.
For Picamera2, which has no separate camera JPEG, it is a single JPEG encode of
the ISP's RGB frame; that frame is always the camera's own full-quality
rendering, so it is named ``_original.jpg`` whether or not a DNG sidecar is
also saved. ``HHMMSS_graded.jpg`` is the graded version. ``_ungraded.jpg`` is
used only for the V4L2 fallback, when there is no camera JPEG to save
verbatim (raw mode unsupported, or a captured buffer failed validation). When
the backend supplies a raw frame and DNG saving is enabled, a ``<stem>.dng``
sidecar is written alongside the original; ``--no-dng`` only omits that
sidecar and never changes the original's name. One JSON line per capture
lands in ``captures.jsonl``.

That line is an audit record, not a status message. It carries the LUT hash,
the normalisation hash and the grain seed, which together let anyone regenerate
the graded file from the original — the LUT hash alone is not enough, because
normalisation is a per-artifact setting that changes the grade under an
unchanged LUT; the negotiated stream format, so a camera that quietly
dropped to a different mode is visible; two timings, because the pipeline cost
and the time from shutter to durable file are different numbers and only the
second is what the user waits for; ``shutter_us`` (the fixed exposure time under
shutter priority, null on auto); ``iso`` (the fixed ISO, null on auto);
``camera_metadata`` (per-shot camera metadata)
when present; and ``dng`` (the sidecar filename) when a DNG was written.

Grading behind the shutter
--------------------------
With ``background=True`` (``--grade background``, the command's default) a shot is
split in two. The capture side reads the frame, saves the original, writes a
pending entry, makes a quick grade for the screen and the Stick, and hands the
full frame to a ``GradeQueue``; the camera is then free. The grade side
(``_run_grade``, on the queue's one worker) grades at full size, saves the graded
file, appends the record and removes the entry, in that order.

Why that order on the grade side: a crash after the file and before the record
leaves the entry, so the next start grades again and writes the one record; a
crash after the record and before the entry is removed leaves an entry whose
record already exists, which the next start sees in ``captures.jsonl`` and only
removes. Neither leaves a photograph without a record or with two. An entry can
also come back after it was removed (its removal is not synced to disk), and the
same check covers that.

Why the grade side syncs the graded file and the log before it removes the entry:
on ext4 a removal can reach the disk before data that was only written (delayed
allocation), so a power cut a few seconds after a grade could otherwise leave no
entry, no record and an empty graded file, with nothing left to say a grade is
owed. The inline path never had an entry to remove and is not synced. A log whose
last line was cut short by a power cut gets a newline before the next record, so
a record is never joined to half of another.

Stopping: ``main`` turns SIGTERM, which is how systemd stops the service, into the
same exit as Ctrl-C, so its clean-up runs. The clean-up finishes the photographs
taken in this run (the grade running and those waiting) before the process ends;
grades recovered from an earlier run that have not started are left, with their
entries, for the next start, so a stop is never longer than this run's queue.
Once a clean-up has begun SIGTERM is ignored, in every mode, so a second one
cannot interrupt it.

Why the original is synced before the entry is written (background only): the
entry is synced, so without this a power cut could leave an entry naming an
original that never reached the disk, which no start could ever finish.

Why the entry is written before the quick grade: from the moment the original is
on disk the photograph must be finishable without this process, and the quick
grade is the first thing after it that can fail.

The graded file is the same in both modes, byte for byte: same frame, same seed
drawn at the same place in the seed stream, same pipeline call, same encoder. The
quick grade is not that file. It has no grain and is normalised on the reduced
frame (the look's halation too, which is sized in mm of the frame, so its glow is
the same share of the picture), and it lives in the session's private temporary
directory, never under ``OUT``, so the Nextcloud sync never uploads it.

The record is written by the grade side, so it lands when the graded file does,
in shot order (one worker). It gains ``shutter_to_ready_ms`` (shutter to the
camera being free, the wait the photographer feels), ``grade: "background"``, and
the stages ``preview`` and ``queued``. A grade finished at the next start has no
shutter to measure from: it carries ``recovered: true`` instead of the two
shutter timings, and it was made from the saved JPEG, not the frame in memory.

``background=False`` is the code as it was before any of this: no queue, no
state directory, and none of these record fields. (Its records do carry
``pipeline_backend``, always ``"cpu"``; see below.)

Grading on the GPU
------------------
With ``grade_backend="auto"`` (``--grade-backend auto``, the command's default) and
grading behind the shutter, the grade worker opens a ``GpuGrader`` before its
first job and hands it to ``Pipeline.process`` with every full-resolution grade.
It is opened and closed on the worker (the queue's ``on_start`` and ``on_stop``)
because a graphics context belongs to the thread that made it. Opening never
fails the start: without a GPU, or with a look the pass does not cover, it logs
one line and the worker grades on the CPU as before.

Opening can do something worse than fail: it can hang, inside the driver, with
no time limit, on the only grade worker. Then no grade ever runs, the fourth
shot waits for a place that never frees, the stop waits for systemd's kill, and
the next start would do it all again. So the worker writes a marker file,
``gpu-opening``, before it opens the GPU and removes it when the open returns,
with a grader or without. A start that finds the marker does not open the GPU:
it logs one line naming the file and grades on the CPU, and the marker stays
until a person deletes it. The marker sits beside the state directory, not in
it, so it is never taken for a pending entry. Where it cannot be written there
is no protection and the open goes ahead.

Everything else stays on the CPU: the quick grade, the live preview, a session
grading inline, and the one shot a closing queue refuses, which is graded on the
capture's thread (the grader says no to any thread but its own). A photograph the
GPU fails on is graded on the CPU in the same call, from the same seed, and one
line in the log names it; it is never lost and never waits for another start.

Every record of a graded photograph says what graded it: ``pipeline_backend`` is
``"gpu-2"`` (``pifilm.gpu.shader.BACKEND``; the number is the shader's version)
or ``"cpu"``. It is the newest field, so it comes last of the new ones: after
``stage_ms`` in an inline record, at the end of a background one. A GPU grade is
not the CPU grade byte for byte (see ``pifilm.pipeline``), and this field is how
``pifilm-process`` knows.

The tuning check
----------------
A trained look is fitted to the colours of one libcamera tuning file, and an OS
update can replace that file without any error. So when a session starts, with
the camera open and the look loaded, it compares the fingerprint the look was
stamped with (``tuning_sha1`` in ``params.json``) against the one the camera
reports (``Picamera2Camera.tuning``; see ``pifilm.tuning``) and logs one line:
matches, DIFFERS, or cannot check. Warn only: a camera that will not shoot after
an update in the field is worse than colours that can be regraded from the kept
originals. A look without a fingerprint, or a camera without a tuning (V4L2,
``--fake``), checks nothing and logs nothing. When both fingerprints are known,
every record gains ``tuning_matches`` right after the camera fields, and
``tuning_mismatch`` (always present, False unless they differ) is what the
viewfinder reads. A background shot stores the field in its pending entry, so a
grade recovered at a later start says what the run that took it saw.

The disk-space warning
----------------------
``main`` has the session make a ``DiskMonitor`` (``pifilm.capture.disk``) over
the photo folder and ``/`` and runs it for every loop, started and closed beside
the UPS poller: the viewfinder shows its warning, and its log lines (through the
session's log, like ``tuning:``) are as useful headless. ``disk_warning`` is what
the controller's snapshot carries to the viewfinder. Warn only: a shot is never
refused for space.

The network addresses
---------------------
Beside it ``main`` has the session make a ``NetworkMonitor``
(``pifilm.capture.network``) for every loop, for the same reasons: the viewfinder
shows the Pi's ``eth0`` and ``wlan0`` addresses, and the ``network:`` lines say in
the journal which address it had when. ``network`` is what the snapshot carries.

The settings menu
-----------------
``main`` reads ``settings.json`` (``pifilm.capture.settings``) beside the state
directory once the backend is known and before the panel or the camera is opened,
so a saved value is in effect from the first frame: the unit's flags are the
defaults, a value saved in the menu wins, and each ``settings:`` line says so in
the journal. Only a real Picamera2 takes them (DNG saving, the two auto-exposure
modes, the ISP's sharpness, contrast and saturation and the short exposure mode
are its controls); under ``--fake`` or V4L2 the arguments stay as given, so a
saved value can never trip the Picamera2-only check and stop the service
starting. The ``SettingsControl`` gets one applier per row the backend can
change (none for the shutter delay, which is read at each press, and none yet
for multi-frame, which is stored but not acted on), and the viewfinder gets
the rows the camera supports (none of the auto-exposure ones under
``--ae-lock``, whose setters would change nothing, and only the image controls
the camera lists), the look's name (the ``--artifacts`` folder) and, with
``--nextcloud-env`` and a panel, an ``Upload`` that ``main`` closes beside the
network monitor. Only the viewfinder's loop thread sets or resets a setting; the
control has no lock.

The shutter delay (``shutter_delay``) has no applier because nothing in the
session holds it: the LCD's shutter and the GPIO button read it at each press and
pass it to ``CaptureController.submit``, whose worker waits it out before the
capture; the Stick and the keyboard never wait. The viewfinder reads it from its
``SettingsControl`` only with setting rows (Picamera2), and the button through a
getter ``main`` gives it that answers 0.0 on any other backend, so a value saved
on the Pi never delays a ``--fake`` or V4L2 shot. The record's
``shutter_delay_s`` (Picamera2 records only, right after the camera's stream
fields, and kept in a background shot's pending entry) is the delay waited.

The halation setting
--------------------
The menu's HALATION row (``halation`` in ``settings.json``; no flag, the unit's
value is on) leaves the look's halation out of the photographs that follow, and
puts it back. The choice is the shutter's: ``capture`` reads ``halation`` beside
the EV and the shutter speed, and that value grades the shot's quick grade and
its full grade, however long the grade waits while the row is tapped again. A
background shot keeps it in its pending entry (``"halation"``), so a grade
recovered at a later start is the one the photographer asked for; an entry from
before the setting has none and means on, as it was graded then. A double
exposure takes the value at its second shutter, the one that makes the
composite. The live view has no halation either way. The record says what was
done (the pipeline's audit fields): ``halation_sha1`` when the glow was applied,
``"halation": "off"`` in its place when a look with halation was graded without
it, which ``pifilm-process`` reads to regrade the same way, and neither for a
look without halation, whose records are as they always were. Like the other
settings it is taken only on a real Picamera2, the only backend with the menu.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import select
import signal
import sys
import tempfile
import termios
import threading
import time
import tty
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import current_thread, main_thread

import numpy as np

from .. import __version__
from .._cv2 import require_cv2
from ..artifacts import PARAMS_VERSION, Artifacts, ArtifactsError
from ..display import DisplayError
from ..display.viewfinder import ViewfinderLoop
from ..double import COMPOSITE_METHOD, composite
from ..gpu.grader import GpuGrader
from ..imageio import load_rgb, save_jpeg
from ..pipeline import Pipeline
from ..tuning import check as check_tuning
from .button import (
    MAX_PIN,
    MIN_PIN,
    WAVESHARE28_PINS,
    X728_PINS,
    ButtonError,
    ShutterButton,
    open_shutter_button,
)
from .camera import Camera, CameraError, FakeCamera, Frame, V4L2Camera
from .controller import CaptureController, JobSnapshot
from .disk import DEFAULT_WARN_PERCENT, DiskMonitor, DiskWarning, watched_paths
from .errors import CameraStalled
from .gradequeue import (
    GradeJob,
    GradeQueue,
    default_state_dir,
    read_entries,
    remove_entry,
    report_failure,
    write_entry,
)
from .network import Addresses, NetworkMonitor
from .picamera import COLOUR_TEMPERATURE_RANGE_K, Picamera2Camera
from .power import UPS_CHOICES, PowerError, build_ups
from .remote import RemoteCaptureServer
from .settings import IMAGE_KEYS, SETTINGS_FILE, Settings, SettingsControl
from .settings import load as load_settings
from .thumbnail import render_exposure_card
from .upload import Upload

cv2 = require_cv2()

DEFAULT_OUT = Path("~/Pictures/pifilm")
WINDOW_NAME = "Parr  [SPACE capture | P toggle grade | Q quit]"
CAPTURE_WINDOW_NAME = "Parr captures  [SPACE capture | Q quit]"

# The quick grade's long edge, in pixels: a quarter of the IMX477's 4056, which is
# more than the 640 x 480 panel and the Stick's 240 x 135 thumbnail can show.
QUICK_LONG_EDGE = 1014
# Quick grades kept in the private directory. Only the newest is ever shown; the
# rest are kept for a client still downloading an earlier one.
QUICK_KEEP = 16
# How long the grade side waits to be told how long the shutter was held (see
# ``_GradeWork.ready``). The capture side says so within microseconds of handing
# the job over; the ceiling is for a capture ended by an interrupt before it could.
READY_WAIT_S = 5.0
# The file that says a start is, or was, in the middle of opening the GPU (see
# "Grading on the GPU" in the module docstring). Beside the state directory.
GPU_MARKER = "gpu-opening"
GRADE_BACKENDS = ("auto", "cpu")


@dataclass
class CaptureResult:
    original: Path
    pifilm: Path
    record: dict
    # (index, of) for a double-exposure shot, None for a single one. After exposure 1
    # ``pifilm`` is the "Exposure 1/2" card, not a photo: nothing is graded until the
    # second exposure lands on the same "negative" (see ``pifilm.double``).
    exposure: tuple[int, int] | None = None
    # Where the graded file is, or will be once its background grade finishes. Graded
    # inline it is ``pifilm``; behind the shutter ``pifilm`` is the quick grade and
    # this is the final path. None after exposure 1 of a double: nothing is graded.
    graded: Path | None = None


@dataclass
class _GradeWork:
    """One photograph's grade: the payload of a ``GradeJob``.

    The first nine fields are exactly what the pending entry holds, so a grade
    can be done again from the entry alone. The rest exists only for a shot taken
    by this process; a recovered grade has ``frames`` None and reads its originals
    from disk.
    """

    day: str  # the day folder's name; the graded file and its record go there
    graded: str  # the graded file's name
    originals: list[str]  # relative to out_root; two for a double, exposure 1 first
    seed: int
    ev: float
    record: dict  # the record's capture-time fields
    double: dict | None = None  # {"ev_comp": [ev1, ev2], "method": ...} for a double
    # The menu's HALATION as it was at the shutter: False leaves the look's glow
    # out of both grades. An entry written before there was the setting has none,
    # and was graded with the look's halation, so a missing value means True.
    halation: bool = True
    frames: tuple[np.ndarray, ...] | None = None
    stages: dict[str, float] = field(default_factory=dict)
    shutter: float | None = None  # time.perf_counter() at the shutter
    # Shutter to the camera being free. Known only after the job is handed over, by
    # which time the worker may already be grading it, so the worker waits on
    # ``ready`` before it reads the number for the record.
    ready_ms: float | None = None
    ready: threading.Event = field(default_factory=threading.Event)

    def entry(self, out_root: Path) -> dict:
        return {
            "out_root": str(out_root),
            "day": self.day,
            "graded": self.graded,
            "originals": list(self.originals),
            "seed": self.seed,
            "ev": self.ev,
            "double": self.double,
            "halation": self.halation,
            "record": self.record,
        }

    @classmethod
    def from_entry(cls, payload: dict) -> _GradeWork:
        """The work a pending entry describes. Raises on anything it cannot trust.

        An entry is read back from disk at start-up, so every name in it is checked
        before it is joined to the output folder: a damaged or foreign file must
        fail its grade (and be set aside by the queue), not write somewhere else.
        """
        day, graded = payload["day"], payload["graded"]
        originals, record, double = payload["originals"], payload["record"], payload.get("double")
        for name in (day, graded):
            if (
                not isinstance(name, str) or not name or Path(name).name != name
                or name in (".", "..")
            ):
                raise ValueError(f"pending entry has a bad name: {name!r}")
        if (
            not isinstance(originals, list)
            or len(originals) != (2 if double is not None else 1)
            or not all(isinstance(name, str) and name for name in originals)
            or any(Path(name).is_absolute() or ".." in Path(name).parts for name in originals)
        ):
            raise ValueError(f"pending entry has bad originals: {originals!r}")
        if not isinstance(record, dict) or not isinstance(double, dict | None):
            raise ValueError("pending entry has a bad record")
        halation = payload.get("halation", True)
        if not isinstance(halation, bool):
            raise ValueError(f"pending entry has a bad halation: {halation!r}")
        return cls(day, graded, originals, int(payload["seed"]), float(payload["ev"]),
                   record, double, halation)


def _quick_frame(rgb: np.ndarray) -> np.ndarray:
    """The frame reduced by a whole factor to about ``QUICK_LONG_EDGE`` on its long edge.

    A whole factor, so every output pixel is the plain mean of a block of sensor
    pixels (``INTER_AREA``), which is cheap and does not alias. A frame already
    that small is returned as it is, not copied.
    """
    height, width = rgb.shape[:2]
    factor = max(1, round(max(height, width) / QUICK_LONG_EDGE))
    if factor == 1:
        return rgb
    size = (max(1, width // factor), max(1, height // factor))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _sync_file(path: Path) -> None:
    """Force a file that was just written onto the disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _already_recorded(day_dir: Path, graded: str) -> bool:
    """Whether ``captures.jsonl`` in ``day_dir`` already has the record for ``graded``."""
    try:
        lines = (day_dir / "captures.jsonl").read_text().splitlines()
    except (OSError, ValueError):
        return False
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a line cut short by a power cut says nothing about this grade
        if isinstance(record, dict) and record.get("pifilm") == graded:
            return True
    return False


@dataclass
class _PendingExposure:
    """Exposure 1 of a double, held in memory until exposure 2 arrives."""

    rgb: np.ndarray
    original: str  # relative to out_root, e.g. "2026-09-30/120000_original.jpg"
    ev: float


@contextlib.contextmanager
def _stage(stages: dict[str, float], name: str):
    """Time the enclosed block into ``stages[name]``, in milliseconds."""
    started = time.perf_counter()
    try:
        yield
    finally:
        stages[name] = round((time.perf_counter() - started) * 1000.0, 1)


def _delay_fields(frame: Frame, delay_s: float) -> dict[str, float]:
    """``shutter_delay_s`` for a Picamera2 frame, nothing for any other.

    The shutter delay is a setting, and settings apply on Picamera2 only, so a
    V4L2 or fake record never carries it. It is the delay waited for this shot
    (0.0 for the Stick and the keyboard), not the menu's value at the time.
    """
    return {"shutter_delay_s": float(delay_s)} if frame.source == "picamera2" else {}


def _capture_stages(frame: Frame, shutter: float) -> dict[str, float]:
    """Start a capture's stage timings (``stage_ms`` in its record) with the camera.

    ``capture`` is the whole ``camera.read()``; a backend that breaks its own time
    down (Picamera2: the mode switch and request, the copy out of the buffer, the
    DNG) adds ``capture_<stage>`` beside it. Later stages are added as they run, so
    the record says where a slow shot went. Durations are not reproducible and so
    stay out of the audit fields a regrade is checked against.
    """
    stages = {"capture": round((time.perf_counter() - shutter) * 1000.0, 1)}
    for name, value in (getattr(frame, "timings", None) or {}).items():
        stages[f"capture_{name}"] = float(value)
    return stages


class CaptureSession:
    def __init__(
        self,
        camera: Camera,
        pipeline: Pipeline,
        out_root: str | Path,
        now: Callable[[], datetime] | None = None,
        seed_rng: np.random.Generator | None = None,
        package_version: str = __version__,
        save_dng: bool = True,
        *,
        background: bool = False,
        state_dir: Path | None = None,
        queue_depth: int = 3,
        log: Callable[[str], None] = print,
        grade_backend: str = "cpu",
        halation: bool = True,
    ) -> None:
        if grade_backend not in GRADE_BACKENDS:
            raise ValueError(
                f"grade_backend must be one of {GRADE_BACKENDS}, got {grade_backend!r}"
            )
        self.camera = camera
        self.pipeline = pipeline
        self.out_root = Path(out_root).expanduser()
        self._log = log
        # The tuning check (see the module docstring): once, before anything else
        # this start logs or grades. Warn only; the camera works either way.
        look = getattr(getattr(pipeline, "artifacts", None), "tuning_sha1", None)
        matches, line = check_tuning(look, getattr(camera, "tuning", None))
        if line is not None:
            self._say(line)
        self._tuning_fields: dict = {} if matches is None else {"tuning_matches": matches}
        self.tuning_mismatch = matches is False
        # The disk-space check (see "The disk-space warning" in the module
        # docstring); None until ``watch_disk`` makes one.
        self._disk: DiskMonitor | None = None
        # The network addresses (see "The network addresses"); None until
        # ``watch_network`` makes a monitor.
        self._network: NetworkMonitor | None = None
        self._now = now or datetime.now
        self._seed_rng = seed_rng or np.random.default_rng()
        self._package_version = package_version
        self._save_dng = save_dng
        # The menu's HALATION (see "The halation setting" in the module docstring):
        # True grades with the look's halation, False leaves it out. Set by the
        # viewfinder's thread, read once per shot at the shutter.
        self.halation = bool(halation)
        # Double exposure: touched only on the controller's worker thread. Memory
        # only, so a restart always begins with the mode off at 0/2.
        self.double_exposure = False
        self._pending: _PendingExposure | None = None
        self._card_dir: Path | None = None
        self._previews: deque[Path] = deque()
        # Grading behind the shutter (see the module docstring). Without a queue
        # every shot is graded inline and the state directory is never touched.
        self._queue: GradeQueue | None = None
        # The GPU grader (see the module docstring). Made, used and closed on the
        # grade worker; None on every other path and whenever there is no GPU.
        self._grader: GpuGrader | None = None
        if background:
            self._state_dir = Path(state_dir) if state_dir is not None else default_state_dir()
            gpu = grade_backend == "auto"
            self._queue = GradeQueue(
                self._run_grade, depth=queue_depth, backlog=self._unfinished(), log=log,
                on_start=self._open_grader if gpu else None,
                on_stop=self._close_grader if gpu else None,
            )
        # Off the start-up path: a shot taken before it finishes just builds the
        # same thing itself.
        still_size = getattr(camera, "still_size", None)
        if still_size is not None:
            threading.Thread(
                target=pipeline.warm, args=((still_size[1], still_size[0]),),
                name="pifilm-warm", daemon=True,
            ).start()

    def _open_grader(self) -> None:
        """On the grade worker, before its first job. ``open`` logs its one line
        (on, or off and why) and returns None when the CPU is to grade."""
        marker = self._state_dir.parent / GPU_MARKER
        try:
            left_behind = marker.exists()
        except OSError:
            left_behind = False
        if left_behind:
            self._say("gpu grading off: the last start did not finish opening the GPU; "
                      f"delete {marker} to try again")
            return
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("pifilm-capture was opening the GPU for grading. If this file "
                              "is still here after a start, that open never returned.\n")
        except OSError:
            pass  # no marker, no protection; the open is still worth trying
        try:
            self._grader = GpuGrader.open(self.pipeline.artifacts, log=self._say)
        finally:
            # Whatever it returned, it returned: the next start may try again.
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass

    def _close_grader(self) -> None:
        """On the grade worker, after its last job: the thread that may release
        the graphics context."""
        grader, self._grader = self._grader, None
        if grader is not None:
            grader.close()

    def _allocate(self, suffix: str) -> tuple[Path, str, datetime]:
        t = self._now()
        day_dir = self.out_root / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base = t.strftime("%H%M%S")
        stem, k = base, 1
        while (day_dir / f"{stem}_{suffix}.jpg").exists():
            k += 1
            stem = f"{base}-{k}"
        return day_dir, stem, t

    @property
    def save_dng(self) -> bool:
        """Whether a still's DNG is saved beside its original."""
        return self._save_dng

    @save_dng.setter
    def save_dng(self, on: bool) -> None:
        # The camera holds its own copy (it decides whether to extract the DNG at
        # all), so the change goes there too when it has one.
        self._save_dng = bool(on)
        setter = getattr(self.camera, "set_save_dng", None)
        if setter is not None:
            setter(self._save_dng)

    @property
    def double_state(self) -> tuple[bool, int]:
        """(mode on, exposures taken towards the current pair: 0 or 1)."""
        return self.double_exposure, (1 if self._pending is not None else 0)

    def set_double_exposure(self, enabled: bool) -> None:
        """Turning the mode off drops a pending exposure 1; its original stays on disk.

        Turning it on while already on keeps the pending frame, so a repeated tap
        cannot silently restart a half-made pair.
        """
        self.double_exposure = bool(enabled)
        if not self.double_exposure:
            self._pending = None

    def capture(self, *, delay_s: float = 0.0) -> CaptureResult:
        """Take one shot. ``delay_s`` is the shutter delay the controller waited
        before calling (0.0 for the Stick); a Picamera2 record carries it as
        ``shutter_delay_s``, right after the camera's stream fields."""
        shutter = time.perf_counter()
        delay = float(delay_s)
        # Read before the frame: the EV the exposure was made at. The grade has to
        # re-apply it, or normalisation cancels it (see ``pifilm.pipeline``).
        ev = float(getattr(self.camera, "ev", 0.0))
        # Also read before the frame: the shutter the exposure was made at (None = auto).
        shutter_us = getattr(self.camera, "shutter_us", None)
        # And the manual white balance (None = auto).
        wb_k = getattr(self.camera, "colour_temperature_k", None)
        # And the manual ISO (None = auto).
        iso = getattr(self.camera, "iso", None)
        # And the menu's HALATION: the photograph is graded as it was set now,
        # whenever its grade runs.
        halation = self.halation
        frame = self.camera.read()
        stages = _capture_stages(frame, shutter)
        if self.double_exposure:
            return self._capture_double(
                frame, ev, shutter, stages, shutter_us=shutter_us, wb_k=wb_k, iso=iso,
                halation=halation, delay_s=delay,
            )

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        if self._queue is not None:
            return self._capture_background(
                frame, ev, shutter, stages, seed, shutter_us=shutter_us, wb_k=wb_k, iso=iso,
                halation=halation, delay_s=delay,
            )
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(
            frame.rgb, rng=np.random.default_rng(seed), ev=ev, timings=stages,
            halation=halation,
        )
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        day_dir, stem, t, original, dng_name = self._write_original(frame, stages)
        with _stage(stages, "save_graded"):
            pifilm = save_jpeg(graded, day_dir / f"{stem}_graded.jpg")
        shutter_to_saved_ms = (time.perf_counter() - shutter) * 1000.0

        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": pifilm.name,
            "frame_source": frame.source,
            **info,
            "shutter_us": shutter_us,
            "iso": iso,
            "colour_temperature_k": wb_k,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            **_delay_fields(frame, delay),
            **self._tuning_fields,
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round(shutter_to_saved_ms, 1),
            "stage_ms": stages,
            "pipeline_backend": "cpu",
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        self._append_record(day_dir, record)
        return CaptureResult(original, pifilm, record, graded=pifilm)

    def _capture_background(
        self, frame: Frame, ev: float, shutter: float, stages: dict[str, float], seed: int, *,
        shutter_us: int | None, wb_k: int | None, iso: int | None, halation: bool,
        delay_s: float,
    ) -> CaptureResult:
        """A single shot graded behind the shutter: everything up to handing it over."""
        day_dir, stem, t, original, dng_name = self._write_original(frame, stages)
        graded = day_dir / f"{stem}_graded.jpg"
        # The inline record without what grading adds. ``ev_comp`` marks where the
        # pipeline's fields go, so the finished record keeps the inline order.
        fields = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": graded.name,
            "frame_source": frame.source,
            "ev_comp": ev,
            "shutter_us": shutter_us,
            "iso": iso,
            "colour_temperature_k": wb_k,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            **_delay_fields(frame, delay_s),
            **self._tuning_fields,
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        work = _GradeWork(
            day_dir.name, graded.name, [self._relative(original)], seed, ev, fields,
            halation=halation, frames=(frame.rgb,), stages=stages, shutter=shutter,
        )
        return self._hand_over(work, stem, original, graded, exposure=None)

    def _hand_over(
        self, work: _GradeWork, stem: str, original: Path, graded: Path, *,
        exposure: tuple[int, int] | None,
    ) -> CaptureResult:
        """Entry, quick grade, queue: the end of the capture side of a background shot."""
        assert self._queue is not None and work.frames is not None and work.shutter is not None
        job = GradeJob(payload=work, entry=self._write_entry(work, stem))
        try:
            preview = self._quick_grade(work, stem)
        except Exception:
            # No picture for the screen, so the shot is reported as failed, but the
            # photograph is on disk and its grade must still be done in this run.
            self._submit(job, shutter=work.shutter)
            raise
        # A copy, taken now: once the job is handed over the worker adds its own
        # stages to the job's dict.
        stages_so_far = dict(work.stages)
        self._submit(job, shutter=work.shutter)
        record = {**work.record, "shutter_to_ready_ms": work.ready_ms, "stage_ms": stages_so_far}
        return CaptureResult(original, preview, record, exposure=exposure, graded=graded)

    def _write_entry(self, work: _GradeWork, stem: str) -> Path | None:
        """Write the shot's pending entry; None (and one log line) when that fails.

        A state directory that cannot be written costs the protection against a
        power cut, not the photograph: the grade still runs from memory.
        """
        try:
            base = name = f"{work.day}_{stem}"
            k = 1
            # Another output folder's unfinished entry may share this name (the same
            # second on another day's clock); it is not ours to overwrite.
            while (self._state_dir / f"{name}.json").exists():
                k += 1
                name = f"{base}-{k}"
            return write_entry(self._state_dir, name, work.entry(self.out_root.absolute()))
        except (OSError, ValueError, TypeError) as exc:
            self._say(f"warning: could not write the pending entry for {work.graded} ({exc}); "
                      "this grade will not survive a power cut")
            return None

    def _quick_grade(self, work: _GradeWork, stem: str) -> Path:
        """The picture for the screen and the Stick: reduced, no grain, never under ``out_root``."""
        assert work.frames is not None
        with _stage(work.stages, "preview"):
            small = [_quick_frame(rgb) for rgb in work.frames]
            rgb = composite(small[0], small[1]) if len(small) == 2 else small[0]
            quick, _ = self.pipeline.process(rgb, grain=False, ev=work.ev,
                                             halation=work.halation)
            path = save_jpeg(quick, self._private_dir() / f"preview-{stem}.jpg")
        # The same name can come round again on another day; the older file is
        # then already replaced and must not be deleted as if it were old.
        if path in self._previews:
            self._previews.remove(path)
        self._previews.append(path)
        while len(self._previews) > QUICK_KEEP:
            self._previews.popleft().unlink(missing_ok=True)
        return path

    def _submit(self, job: GradeJob, *, shutter: float) -> None:
        """Hand the job to the queue, waiting for a place if it is full.

        A queue that refuses is closing. The photograph is then finished here, on
        the capture's own thread, after the grades already accepted (so records
        stay in shot order and two full frames are never graded at once).
        """
        assert self._queue is not None
        work: _GradeWork = job.payload
        asked = time.perf_counter()
        try:
            accepted = self._queue.submit(job)
        finally:
            # Set whatever happened: the worker waits on it before writing the record.
            work.ready_ms = round((time.perf_counter() - shutter) * 1000.0, 1)
            work.ready.set()
        if accepted:
            return
        self._queue.close()
        job.queued_at, job.started_at = asked, time.perf_counter()
        try:
            self._run_grade(job)
        except Exception as exc:
            # As the queue does for its own jobs: the entry stays, one attempt used.
            report_failure(job, exc, self._say)

    def _run_grade(self, job: GradeJob) -> None:
        """The grade side of a shot: grade, save, record, remove the entry.

        Runs on the queue's worker (or on the capture thread for the one shot a
        closing queue refused). Anything raised here is the queue's to count; the
        entry is left for another attempt.
        """
        work = job.payload if isinstance(job.payload, _GradeWork) else _GradeWork.from_entry(
            job.payload
        )
        recovered = work.frames is None
        day_dir = self.out_root / work.day
        if recovered and _already_recorded(day_dir, work.graded):
            # The grade had finished; only the entry was left behind.
            remove_entry(job.entry)
            return
        stages = work.stages
        stages["queued"] = (
            0.0 if recovered else round(max(0.0, job.started_at - job.queued_at) * 1000.0, 1)
        )
        frames = work.frames
        if frames is None:
            frames = tuple(load_rgb(self.out_root / name)[0] for name in work.originals)
        else:
            with _stage(stages, "sync_original"):
                self._sync_originals(work, day_dir)
        t0 = time.perf_counter()
        if work.double is not None:
            with _stage(stages, "composite"):
                rgb = composite(frames[0], frames[1])
        else:
            rgb = frames[0]
        # Offered only when there is one, so the call without a GPU is the call
        # there always was. On the capture thread (a refused shot) the grader
        # says no by thread and the CPU grades.
        used: dict = {}
        gpu = {"grader": self._grader, "used": used} if self._grader is not None else {}
        graded, info = self.pipeline.process(
            rgb, rng=np.random.default_rng(work.seed), ev=work.ev, timings=stages,
            halation=work.halation, **gpu,
        )
        if "gpu_error" in used:
            self._say(f"gpu grade failed ({work.graded}): {used['gpu_error']}; "
                      "graded on the CPU")
        pipeline_ms = (time.perf_counter() - t0) * 1000.0
        with _stage(stages, "save_graded"):
            # Synced inside the stage: "saved" means on the disk, and the entry
            # below must not be removed before the file it stands for is there.
            _sync_file(save_jpeg(graded, day_dir / work.graded))
        saved_ms = None
        if work.shutter is not None:
            saved_ms = (time.perf_counter() - work.shutter) * 1000.0
            # Bounded: a capture ended by an interrupt may never set it, and this
            # worker must not wait for ever. The grade is finished either way,
            # once, and its record then goes without ``shutter_to_ready_ms``.
            work.ready.wait(READY_WAIT_S)

        # Today's order for today's fields, the new ones after them. The fields that
        # close the inline record close this one too.
        closing = ("double",) if "double" in work.record else ("camera_metadata", "dng")
        record: dict = {}
        for key, value in work.record.items():
            if key == "ev_comp":
                record.update(info)
            elif key not in closing:
                record[key] = value
        record.update(info)  # an entry without ``ev_comp`` still gets them; no reorder otherwise
        record["pipeline_ms"] = round(pipeline_ms, 1)
        if saved_ms is not None:
            record["shutter_to_saved_ms"] = round(saved_ms, 1)
        record["stage_ms"] = stages
        record.update({key: work.record[key] for key in closing if key in work.record})
        if work.ready_ms is not None:
            record["shutter_to_ready_ms"] = work.ready_ms
        record["grade"] = "background"
        if recovered:
            record["recovered"] = True
        record["pipeline_backend"] = used.get("backend", "cpu")
        self._append_record(day_dir, record, durable=True)
        remove_entry(job.entry)

    def _unfinished(self) -> list[GradeJob]:
        """Jobs for the pending entries a previous run of this output folder left.

        Entries of another output folder are left where they are. Nothing in an
        entry is trusted or even looked at here beyond that: this runs at start-up
        and must not raise, so a damaged entry fails later, on the worker.
        """
        mine = str(self.out_root.absolute())
        jobs = [
            GradeJob(payload=payload, entry=path)
            for path, payload in read_entries(self._state_dir)
            if payload.get("out_root") == mine
        ]
        if jobs:
            self._say(f"recovering {len(jobs)} unfinished grade(s)")
        return jobs

    def _say(self, message: str) -> None:
        try:
            self._log(message)
        except Exception:
            pass  # a closed stdout must not cost a photograph

    def watch_disk(self, threshold: int = DEFAULT_WARN_PERCENT, **kwargs) -> DiskMonitor:
        """A disk monitor for the photo folder and ``/``, logging through the session.

        Not started: the caller starts and closes it, as ``main`` does the UPS poller.
        ``kwargs`` go to ``DiskMonitor`` (tests pass fake ``statvfs`` and ``stat``).
        """
        self._disk = DiskMonitor(watched_paths(self.out_root), threshold, log=self._say,
                                 **kwargs)
        return self._disk

    @property
    def disk_warning(self) -> DiskWarning:
        """``(label, percent)`` for each filesystem at or above the threshold, photos
        first; empty when none is, or when nothing watches the disk. Never touches it."""
        return self._disk.warning if self._disk is not None else ()

    def watch_network(self, **kwargs) -> NetworkMonitor:
        """A monitor of the eth0 and wlan0 addresses, logging through the session.

        Not started: the caller starts and closes it, as ``main`` does the disk monitor.
        ``kwargs`` go to ``NetworkMonitor`` (tests pass a fake ``read``).
        """
        self._network = NetworkMonitor(log=self._say, **kwargs)
        return self._network

    @property
    def network(self) -> Addresses:
        """``(label, address)`` for each of eth0 and wlan0 that is up with an IPv4
        address, ETH first; empty when neither is, or when nothing watches. Never
        touches the network."""
        return self._network.addresses if self._network is not None else ()

    @property
    def pending_grades(self) -> int:
        """Photographs whose graded file is not on disk yet; 0 when grading inline."""
        return self._queue.pending if self._queue is not None else 0

    def close_grades(self, timeout: float | None = None) -> bool:
        """Finish the grades already accepted and stop the worker.

        True when nothing is left running (always, when grading inline); False when
        ``timeout`` ran out first. Safe to call again, and from any thread: it
        takes no lock a capture holds, so it also releases a capture that is
        waiting for a place in a full queue.
        """
        return self._queue.close(timeout) if self._queue is not None else True

    def _capture_double(
        self, frame: Frame, ev: float, shutter: float, stages: dict[str, float], *,
        shutter_us: int | None, wb_k: int | None, iso: int | None, halation: bool,
        delay_s: float,
    ) -> CaptureResult:
        # The pending frame is taken off the session before anything that can fail, so
        # any failure from here on returns the count to 0/2 (spec section 5). A camera
        # error is raised by read() before this point and leaves 1/2 in place for a retry.
        pending, self._pending = self._pending, None
        day_dir, stem, t, original, dng_name = self._write_original(frame, stages)
        camera_fields = {
            "frame_source": frame.source,
            "shutter_us": shutter_us,
            "iso": iso,
            "colour_temperature_k": wb_k,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            **_delay_fields(frame, delay_s),
            **self._tuning_fields,
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        if pending is None:
            record = {
                "timestamp": t.isoformat(timespec="seconds"),
                "original": original.name,
                **camera_fields,
                "ev_comp": ev,
                "shutter_to_saved_ms": round((time.perf_counter() - shutter) * 1000.0, 1),
                "double": {"index": 1, "of": 2},
            }
            # The card before the log line: a failed exposure 1 leaves no ``index: 1``
            # record behind for an exposure 2 that will never follow it.
            with _stage(stages, "card"):
                card = self._exposure_card(1, 2)
            record["stage_ms"] = stages
            self._append_record(day_dir, record)
            self._pending = _PendingExposure(frame.rgb, self._relative(original), ev)
            return CaptureResult(original, card, record, exposure=(1, 2))

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        if self._queue is not None:
            graded = day_dir / f"{stem}_double_graded.jpg"
            double = {"ev_comp": [pending.ev, ev], "method": COMPOSITE_METHOD}
            originals = [pending.original, self._relative(original)]
            # Today's exposure-2 record without what grading adds (see
            # ``_capture_background``); the composite is graded at ev=0.
            fields = {
                "timestamp": t.isoformat(timespec="seconds"),
                "original": original.name,
                "pifilm": graded.name,
                **camera_fields,
                "ev_comp": 0.0,
                "grain_seed": seed,
                "params_version": PARAMS_VERSION,
                "double": {
                    "index": 2, "of": 2, "method": COMPOSITE_METHOD,
                    "originals": originals, "ev_comp": [pending.ev, ev],
                },
            }
            # The composite is graded with HALATION as it was at this, the
            # second, shutter: the one that makes the photograph.
            work = _GradeWork(
                day_dir.name, graded.name, originals, seed, 0.0, fields, double, halation,
                frames=(pending.rgb, frame.rgb), stages=stages, shutter=shutter,
            )
            return self._hand_over(work, stem, original, graded, exposure=(2, 2))
        t0 = time.perf_counter()
        # Each frame goes in as shot, EV included; the composite is graded at ev=0.
        with _stage(stages, "composite"):
            combined = composite(pending.rgb, frame.rgb)
        graded, info = self.pipeline.process(
            combined, rng=np.random.default_rng(seed), ev=0.0, timings=stages,
            halation=halation,
        )
        pipeline_ms = (time.perf_counter() - t0) * 1000.0
        with _stage(stages, "save_graded"):
            pifilm = save_jpeg(graded, day_dir / f"{stem}_double_graded.jpg")
        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": pifilm.name,
            **camera_fields,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round((time.perf_counter() - shutter) * 1000.0, 1),
            "stage_ms": stages,
            "pipeline_backend": "cpu",
            "double": {
                "index": 2, "of": 2, "method": COMPOSITE_METHOD,
                "originals": [pending.original, self._relative(original)],
                "ev_comp": [pending.ev, ev],
            },
        }
        self._append_record(day_dir, record)
        return CaptureResult(original, pifilm, record, exposure=(2, 2), graded=pifilm)

    def _sync_originals(self, work: _GradeWork, day_dir: Path) -> None:
        """Force a fresh shot's originals and DNG onto the disk, first thing on the
        grade worker.

        The camera does not wait for this (see ``_write_original``), so for the
        second or so until the worker gets here, longer behind a full queue, a
        power cut can leave a pending entry whose original is empty; that
        photograph is lost and its entry ends as ``*.failed``. Before grading went
        behind the shutter nothing was synced at all and the same cut could take
        the last half minute of photographs. A file that has gone is the grade's
        problem to report, not this step's.
        """
        names = [self.out_root / name for name in work.originals]
        dng = work.record.get("dng")
        if dng:
            names.append(day_dir / dng)
        for path in names:
            try:
                _sync_file(path)
            except OSError:
                pass

    def _write_original(
        self, frame: Frame, stages: dict[str, float],
    ) -> tuple[Path, str, datetime, Path, str | None]:
        has_original = frame.jpeg is not None or frame.source == "picamera2"
        suffix = "original" if has_original else "ungraded"
        day_dir, stem, t = self._allocate(suffix)
        original = day_dir / f"{stem}_{suffix}.jpg"
        # Not synced here. Forcing the original and the DNG onto the SD card took
        # 0.2 s and 0.6 to 1.0 s on the Pi 4 (2026-10-13) and ate most of what
        # grading behind the shutter gave; the grade worker syncs them as its
        # first step instead (``_sync_originals``).
        with _stage(stages, "save_original"):
            if frame.jpeg is not None:
                original.write_bytes(frame.jpeg)
            else:
                save_jpeg(frame.rgb, original)
        dng_name = None
        if frame.dng is not None and self._save_dng:
            with _stage(stages, "save_dng"):
                dng_path = day_dir / f"{stem}.dng"
                dng_path.write_bytes(frame.dng)
            dng_name = dng_path.name
        return day_dir, stem, t, original, dng_name

    def _relative(self, path: Path) -> str:
        """``YYYY-MM-DD/name`` under ``out_root``: a pair can straddle midnight, so the
        two originals of one double may sit in different day folders."""
        return path.relative_to(self.out_root).as_posix()

    @staticmethod
    def _append_record(day_dir: Path, record: dict, *, durable: bool = False) -> None:
        """Append one line to the day's log.

        ``durable`` is the grade side's: the line is synced to disk before the
        caller removes the pending entry, and a log left without a final newline
        (a line cut short by a power cut) is given one first.
        """
        path = day_dir / "captures.jsonl"
        if not durable:
            with path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            return
        line = (json.dumps(record) + "\n").encode()
        with path.open("a+b") as fh:
            size = fh.seek(0, os.SEEK_END)
            if size:
                fh.seek(size - 1)
                if fh.read(1) != b"\n":
                    line = b"\n" + line
            fh.write(line)  # append mode: always at the end, wherever the read left off
            fh.flush()
            os.fsync(fh.fileno())

    def _exposure_card(self, index: int, of: int) -> Path:
        """The Stick's picture for a partial double; never under ``out_root``.

        Everything under ``out_root`` is uploaded by the Nextcloud sync, and a card is
        not a photograph, so it lives in a private temporary directory. The directory is
        re-created if it has gone: systemd-tmpfiles ages /tmp under a long-running
        service (the unit has no PrivateTmp), and a vanished card must not fail a shot.
        """
        path = self._private_dir() / f"exposure-{index}-of-{of}.jpg"
        if not path.exists():
            path.write_bytes(render_exposure_card(index, of))
        return path

    def _private_dir(self) -> Path:
        """The session's temporary directory for pictures that are not photographs."""
        if self._card_dir is None:
            self._card_dir = Path(tempfile.mkdtemp(prefix="pifilm-cards-"))
        self._card_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return self._card_dir

    def preview_frame(self, graded: bool = True, size: tuple[int, int] = (640, 360)) -> np.ndarray:
        # full=False: the live preview must not pay Picamera2's per-frame autofocus
        # cycle or DNG extraction cost; capture() below keeps the full=True default.
        small = _resize_to_fit(self.camera.read(full=False).rgb, size)
        if graded:
            # The live view is a viewfinder: no grain and no halation. The quick
            # grade and the photograph have the halation.
            small, _ = self.pipeline.process(
                small, grain=False, halation=False, ev=float(getattr(self.camera, "ev", 0.0)),
            )
        return small


def _announce(result: CaptureResult, out: Callable[[str], None]) -> None:
    r = result.record
    if result.exposure == (1, 2):
        out(
            f"Saved {result.original.name} as exposure 1/2 in "
            f"{r['shutter_to_saved_ms']:.0f} ms; waiting for the second exposure"
        )
        return
    if "wb_gains" not in r:
        # Graded behind the shutter: the record so far has nothing from the grade.
        graded = result.graded.name if result.graded is not None else result.pifilm.name
        out(
            f"Saved {result.original.name} in {r['shutter_to_ready_ms']:.0f} ms; "
            f"grading {graded} in the background"
        )
        return
    clamps = [k for k, v in r["clamped"].items() if v]
    note = f" (clamped: {', '.join(clamps)})" if clamps else ""
    out(
        f"Saved {result.pifilm.name} + {result.original.name} in "
        f"{r['shutter_to_saved_ms']:.0f} ms; wb={r['wb_gains']} "
        + (f"levels gamma={r['levels']['gamma']} stretch={r['levels']['stretch']}"
           if r.get("levels") else f"exposure={r['exposure_gain']}")
        + note
    )


def run_headless_loop(
    session: CaptureSession,
    read_key: Callable[[], str | None],
    out: Callable[[str], None] = print,
    *,
    controller: CaptureController | None = None,
) -> int:
    out("Headless mode: SPACE to capture, Q to quit.")
    count = 0
    owned_controller = controller is None
    controller = controller or CaptureController(session)
    try:
        while True:
            key = read_key()
            if key is None:
                continue
            if key == " ":
                job = controller.submit(uuid.uuid4().hex)
                if job.error_code is not None:
                    out(f"error: {job.error_message or job.error_code}")
                    continue
                out("Processing photo...")
                completed = _wait_for_terminal_job(controller, job.request_id)
                if completed.state == "complete":
                    assert isinstance(completed.result, CaptureResult)
                    _announce(completed.result, out)
                    count += 1
                else:
                    out(f"error: {completed.error_message or completed.error_code}")
            elif key.lower() == "q":
                return count
    finally:
        if owned_controller:
            controller.close()


def _wait_for_terminal_job(controller: CaptureController, request_id: str) -> JobSnapshot:
    while True:
        job = controller.status(request_id)
        assert job is not None
        if job.state in {"complete", "failed"}:
            return job
        time.sleep(0.01)


def _resize_to_fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, fitted, interpolation=cv2.INTER_AREA)


def _fit_display(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit an image into the display, using black borders instead of cropping/stretching."""
    width, height = size
    fitted = _resize_to_fit(frame, size)
    image_height, image_width = fitted.shape[:2]
    left, top = (width - image_width) // 2, (height - image_height) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + image_height, left:left + image_width] = fitted
    return canvas


def _window_size(window_name: str, previous: tuple[int, int]) -> tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
        if width > 1 and height > 1:
            # GTK reports the fitted image rectangle, but exposes the entire widget's
            # width/height ratio through this property. Qt returns a ratio-mode enum.
            try:
                ratio = cv2.getWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO)
                if np.isfinite(ratio) and ratio > 0 and ratio != cv2.WINDOW_FREERATIO:
                    if width / height < ratio:
                        width = round(height * ratio)
                    else:
                        height = round(width / ratio)
            except cv2.error:
                pass
            return width, height
    except cv2.error:
        # Some backends cannot report geometry; let the window preserve image proportions.
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    return previous


def _capture_prompt(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    label = "SPACE to capture"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width / 640, height / 360)
    thickness = max(1, round(scale * 2))
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    position = ((width - text_width) // 2, (height + text_height) // 2)
    cv2.putText(screen, label, position, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return screen


def _capture_loading_screen(size: tuple[int, int]) -> np.ndarray:
    """Draw TV-style colour bars in OpenCV's BGR order."""
    width, height = size
    screen = np.zeros((height, width, 3), dtype=np.uint8)
    colours = [
        (191, 191, 191), (0, 191, 191), (191, 191, 0), (0, 191, 0),
        (191, 0, 191), (0, 0, 191), (191, 0, 0),
    ]
    for index, colour in enumerate(colours):
        left, right = index * width // 7, (index + 1) * width // 7
        screen[:height * 3 // 4, left:right] = colour
    label = "Processing photo..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width / 640, height / 360) * 0.7
    thickness = max(1, round(scale))
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    position = ((width - text_width) // 2, (height * 7 // 8) + text_height // 2)
    cv2.putText(screen, label, position, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return screen


def run_preview_loop(
    session: CaptureSession,
    window_name: str = WINDOW_NAME,
    *,
    captures_only: bool = False,
    read_key: Callable[[], str | None] | None = None,
    controller: CaptureController | None = None,
) -> bool:
    """Run the windowed loop. Returns False if this build cannot show a window."""
    if controller is not None and not captures_only:
        raise ValueError("a shared capture controller requires capture-only display mode")
    owned_controller = captures_only and controller is None
    controller = controller or (CaptureController(session) if captures_only else None)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    except cv2.error:
        if owned_controller:
            assert controller is not None
            controller.close()
        return False
    graded = True
    try:
        try:
            # GTK's first imshow resets the drawable to the image's size. Let that
            # allocation finish BEFORE fullscreen, or it can leave a tiny drawable
            # inside a fullscreen (white) outer window. Do not read geometry yet.
            display_size = (640, 360)
            displayed_frame = _capture_prompt(display_size)
            cv2.resizeWindow(window_name, *display_size)
            cv2.imshow(window_name, displayed_frame)
            cv2.waitKey(50)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(50)
            captured_frame = None  # Keep the full-resolution photo for subsequent display resizes.
            active_request_id = None
            displayed_request_id = None
        except cv2.error:
            return False
        while True:
            try:
                size = _window_size(window_name, display_size)
                if size != display_size:
                    cv2.resizeWindow(window_name, *size)
                if captures_only and size != display_size:
                    displayed_frame = (
                        _capture_prompt(size) if captured_frame is None
                        else _fit_display(captured_frame, size)
                    )
                    cv2.imshow(window_name, displayed_frame)
                resized = size != display_size
                display_size = size
                if captures_only:
                    assert controller is not None
                    snapshot = controller.snapshot()
                    # Completed jobs may arrive entirely between GUI polls, so
                    # observing only the active request would miss fast remote shots.
                    job = snapshot.last_completed_job
                    new_photo = job is not None and job.request_id != displayed_request_id
                    if new_photo:
                        assert job is not None
                        assert isinstance(job.result, CaptureResult)
                        _announce(job.result, print)
                        frame, _ = load_rgb(job.result.pifilm)
                        captured_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        displayed_frame = _fit_display(captured_frame, display_size)
                        cv2.imshow(window_name, displayed_frame)
                        displayed_request_id = job.request_id
                    if active_request_id is not None:
                        tracked = controller.status(active_request_id)
                        if tracked is not None and tracked.state == "failed":
                            print(f"error: {tracked.error_message or tracked.error_code}")
                            cv2.imshow(window_name, displayed_frame)
                    active = snapshot.active_job
                    if active is not None:
                        if active.request_id != active_request_id or resized or new_photo:
                            cv2.imshow(window_name, _capture_loading_screen(display_size))
                        active_request_id = active.request_id
                    else:
                        active_request_id = None
                if not captures_only:
                    frame = session.preview_frame(graded)
                    cv2.imshow(window_name, _fit_display(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), display_size,
                    ))
                key = cv2.waitKey(30 if captures_only else 1) & 0xFF
            except CameraError as exc:
                print(f"error: {exc}")
                continue
            except cv2.error:
                return False
            if read_key is not None:
                terminal_key = read_key()
                if terminal_key:
                    key = ord(terminal_key)
            if key == ord(" "):
                try:
                    if captures_only:
                        if active_request_id is not None:
                            continue
                        assert controller is not None
                        job = controller.submit(uuid.uuid4().hex)
                        if job.error_code is not None:
                            print(f"error: {job.error_message or job.error_code}")
                            continue
                        active_request_id = job.request_id
                        cv2.imshow(window_name, _capture_loading_screen(display_size))
                        # imshow queues a repaint; pump GUI events before blocking on capture.
                        cv2.waitKey(1)
                    else:
                        try:
                            result = session.capture()
                            _announce(result, print)
                        except (CameraError, OSError) as exc:
                            print(f"error: {exc}")
                except cv2.error:
                    return False
            elif not captures_only and key in (ord("p"), ord("P")):
                graded = not graded
            elif key in (ord("q"), ord("Q"), 27):
                return True
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if owned_controller:
            assert controller is not None
            controller.close()


class TerminalKeys:
    """Put the terminal in cbreak mode and read single keys without Enter."""

    def __enter__(self) -> TerminalKeys:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


def has_display() -> bool:
    return sys.platform == "darwin" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def _remote_listen(value: str) -> tuple[str, int]:
    """Parse the intentionally simple IPv4/hostname ``host:port`` CLI value."""
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("expected host:port")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be a number") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return host, port


def _white_balance(value: str) -> int:
    """Parse the ``--white-balance`` colour temperature in kelvin."""
    low, high = COLOUR_TEMPERATURE_RANGE_K
    try:
        kelvin = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of kelvin, got {value!r}"
        ) from None
    if not low <= kelvin <= high:
        raise argparse.ArgumentTypeError(f"must be within {low}-{high} K, got {kelvin}")
    return kelvin


def _colour_gains(value: str) -> tuple[float, float]:
    """Parse the ``R,B`` fixed colour-gains CLI value."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected R,B")
    try:
        r, b = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("colour gains must be numbers") from exc
    if r <= 0 or b <= 0:
        raise argparse.ArgumentTypeError("colour gains must be positive numbers")
    return r, b


def _picamera2_available() -> bool:
    """Probe only the optional runtime package, without opening a camera."""
    try:
        __import__("picamera2")
    except (ImportError, OSError):
        return False
    return True


def _drop_display_on_v4l2(args: argparse.Namespace) -> None:
    """Downgrade ``--display`` to a warning when the backend is V4L2.

    The shipped service unit passes ``--display waveshare35dsi``; on a deployment
    whose camera is USB, refusing the flag would exit 2 and systemd would
    restart the service forever. The viewfinder genuinely cannot run there (the
    V4L2 backend has no preview-mode split and the meter needs libcamera
    metadata), so say so once and serve the Stick.
    """
    if args.display == "none":
        return
    print(
        "warning: display unavailable (the V4L2 backend has no preview mode); "
        "continuing without it",
        file=sys.stderr,
    )
    args.display = "none"


def _reject_picamera2_only_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, reason: str,
) -> None:
    """Refuse Picamera2-only options whenever no real Picamera2 will be opened,
    whether that is V4L2 (however it was chosen) or ``--fake``.

    Covers ``--tuning-file``, ``--autofocus``, ``--af-range``, ``--ae-lock``,
    ``--awb-lock``, ``--colour-gains``, ``--white-balance``, ``--ae-constraint``
    and ``--ae-metering``. ``--device`` (which itself selects V4L2) and ``--no-dng``
    (wired independently into ``CaptureSession`` and meaningful on any backend)
    are deliberately excluded, and so is ``--display``: it is a warning rather
    than an error (see ``_drop_display_on_v4l2``), and it runs against
    ``FakeCamera`` under ``--fake`` so the panel and its touch mapping can be
    exercised without a sensor.
    """
    picamera_only = [
        ("--tuning-file", args.tuning_file is not None),
        ("--autofocus", args.autofocus is not None),
        ("--af-range", args.af_range is not None),
        ("--ae-lock", args.ae_lock),
        ("--awb-lock", args.awb_lock),
        ("--colour-gains", args.colour_gains is not None),
        ("--white-balance", args.white_balance is not None),
        ("--ae-constraint", args.ae_constraint is not None),
        ("--ae-metering", args.ae_metering is not None),
    ]
    offenders = [name for name, given in picamera_only if given]
    if offenders:
        parser.error(f"{', '.join(offenders)} {reason}")


def _open_display(args: argparse.Namespace, out_root: Path):
    """Open the requested display and touch, or None for --display none.

    Raises DisplayError; the caller downgrades that to a warning so a service
    with a broken or absent LCD keeps serving the Stick instead of crash-looping.
    """
    if args.display == "none":
        return None
    if args.display == "fake":
        from ..display.fake import FileDisplay, NoTouch
        return FileDisplay(Path(out_root).expanduser() / "viewfinder-last.png"), NoTouch()
    if args.display == "waveshare35dsi":
        from ..display import evtouch, kms
        display = kms.open_waveshare35dsi(args.display_rotate)
        try:
            touch = evtouch.open_goodix_touch(
                (display.width, display.height), args.display_rotate,
            )
        except DisplayError:
            display.close()
            raise
        return display, touch
    from ..display.cst3530 import open_waveshare28_touch
    from ..display.st7789 import open_waveshare28
    display = open_waveshare28(args.display_rotate)
    try:
        touch = open_waveshare28_touch(args.display_rotate)
    except DisplayError:
        display.close()
        raise
    return display, touch


def _close_display(display_pair) -> None:
    """Release the panel when the process is giving up before the loop starts."""
    if display_pair is None:
        return
    display, touch = display_pair
    touch.close()
    display.close()


def _serve_until_interrupt() -> int:
    """Hold the process open for the remote API with no terminal controls."""
    print("Remote capture API active without terminal controls. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


# How long the clean-up may take after the camera stalls before the process is
# ended regardless. Closing a camera that has stopped answering can itself wait
# for ever, and a process that never exits is never restarted.
STALL_EXIT_DEADLINE = 10.0


def _arm_exit_deadline(seconds: float, code: int) -> None:
    timer = threading.Timer(seconds, os._exit, (code,))
    timer.daemon = True
    timer.start()


class _Terminated(KeyboardInterrupt):
    """SIGTERM, raised in the main thread so a stop unwinds exactly as Ctrl-C does."""


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """While inside, the first SIGTERM raises ``_Terminated`` in the main thread.

    systemd stops the service with SIGTERM, and Python's default for it ends the
    process on the spot: no ``finally`` runs, so the queued grades were left for
    the next start to recover. Raising makes every loop ``main`` can sit in end
    the way it already does for Ctrl-C. The viewfinder installs its own handler
    over this one while it runs (it sets an event instead of raising). Every
    clean-up then begins with ``_ignore_sigterm``, so no later SIGTERM can break
    into it, whichever handler took the first.

    ``signal.signal`` only works in the main thread, so anywhere else this does
    nothing; a caller there cannot be sent a signal's exception anyway.
    """
    if current_thread() is not main_thread():
        yield
        return
    fired = False

    def handler(*_: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise _Terminated()

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _ignore_sigterm() -> None:
    """From here on SIGTERM does nothing: the clean-up is already under way.

    Called at the start of every clean-up. Whichever handler took the first
    SIGTERM (``main``'s, or the viewfinder's, which leaves ``main``'s unfired), a
    second one must not raise inside ``controller.close()`` and leave the grades
    unfinished and the camera open. ``main`` puts the original handler back when
    it returns. Like the handler it replaces, it is only set from the main thread.
    """
    if current_thread() is main_thread():
        signal.signal(signal.SIGTERM, _no_signal)


def _no_signal(*_: object) -> None:
    return None


def _run_viewfinder(
    camera, controller, display_pair, power, args, *,
    settings: SettingsControl | None = None, upload: Upload | None = None,
    look: str = "", menu_rows: tuple[str, ...] | None = None, look_halation: bool = False,
) -> int:
    display, touch = display_pair
    stop = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    loop = ViewfinderLoop(
        camera, controller, display, touch,
        power_snapshot=power.snapshot if power is not None else None,
        touch_debug=args.touch_debug,
        dim_after=args.display_dim_after,
        off_after=args.display_off_after,
        settings=settings,
        upload=upload,
        look=look,
        menu_rows=menu_rows,
        look_halation=look_halation,
    )
    print("LCD viewfinder running. Tap the shutter to capture; Ctrl-C to stop.")
    failure: DisplayError | None = None
    stalled: CameraStalled | None = None
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        pass
    except DisplayError as exc:
        failure = exc
    except CameraStalled as exc:
        stalled = exc
        _arm_exit_deadline(STALL_EXIT_DEADLINE, 1)
    finally:
        # Not ``previous`` yet: that is ``main``'s handler, which raises, and a
        # SIGTERM while the panel closes would leave the rest of the clean-up undone.
        _ignore_sigterm()
        touch.close()
        display.close()
    if stalled is not None:
        # Unlike a dead screen, this leaves nothing worth serving: the Stick and
        # the button would only queue shots behind the same camera. Exit 1 so
        # systemd restarts the service, which reopens the camera.
        print(f"error: camera stopped delivering frames: {stalled}; exiting for a restart",
              file=sys.stderr)
        return 1
    if failure is None:
        return 0
    print(f"error: display failed repeatedly: {failure}", file=sys.stderr)
    # A dead screen must cost the screen only. Exiting here would be a restart
    # loop under systemd, and the Stick would get a few seconds of service per
    # cycle. Without a remote API the process exits 1 and systemd restarts it.
    # That restart is also what brings a GPIO button back: the button has no
    # loop of its own to keep the process alive.
    if not args.remote_listen:
        return 1
    # Serving goes on, so a stop has to work again.
    signal.signal(signal.SIGTERM, previous)
    return _serve_until_interrupt()


def _look_name(artifacts: Path | None) -> str:
    """The ``LOOK`` row: the ``--artifacts`` folder's name, or "" for the starter.

    The shipped unit names the bundled starter by its path (``.../pifilm/data``),
    and ``data`` would say nothing on the screen, so that folder counts as no
    ``--artifacts`` at all and the viewfinder shows its starter label.
    """
    if not artifacts:
        return ""
    path = Path(artifacts)
    if path.name == "data" and path.parent.name == "pifilm":
        return ""
    return path.name


def _disk_percent(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole percent") from None
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError(f"{value} is not from 1 to 100")
    return value


def _gpio_pin(text: str) -> int:
    """An argparse type: a BCM pin number the shutter button may use."""
    try:
        pin = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a BCM pin number") from None
    if not MIN_PIN <= pin <= MAX_PIN:
        raise argparse.ArgumentTypeError(
            f"BCM {pin} is not usable; choose a pin between {MIN_PIN} and {MAX_PIN}"
        )
    return pin


def _button_capture(
    controller: CaptureController, delay: Callable[[], float] = lambda: 0.0,
) -> None:
    """One press of the GPIO shutter button: the same request an LCD tap makes.

    Runs on gpiozero's callback thread. ``submit`` only takes the controller's
    lock and queues the job, so the press returns at once (the shutter delay,
    ``delay()`` read now, is waited on the controller's worker); a capture
    already in progress (or a controller that is shutting down) answers with a
    failed snapshot, and the press is dropped, never queued.
    """
    job = controller.submit(str(uuid.uuid4()), delay_s=delay())
    if job.state == "failed":
        print(f"button: press ignored ({job.error_code})")
    else:
        print("button: capture requested")


def _idle_seconds(text: str) -> float:
    value = float(text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more seconds, got {text}")
    return value


def main(argv: list[str] | None = None) -> int:
    # Held until the clean-up in ``_main`` is over, so a stop finishes the grades.
    with _sigterm_as_interrupt():
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="pifilm-capture",
        description="Capture Parr-graded photos from a supported camera.",
    )
    parser.add_argument("--camera", choices=("v4l2", "picamera2"))
    parser.add_argument("--device", default=None, help="index, /dev/videoN or /dev/v4l/by-id/...")
    parser.add_argument(
        "--tuning-file",
        help="Picamera2 tuning filename or absolute path "
             "(default: libcamera's automatic choice for the detected sensor)",
    )
    parser.add_argument(
        "--autofocus", choices=("continuous", "auto", "manual"), default=None,
        help="Picamera2 autofocus mode (default: continuous)",
    )
    parser.add_argument(
        "--af-range", choices=("normal", "macro", "full"), default=None,
        help="Picamera2 autofocus range (default: normal)",
    )
    parser.add_argument(
        "--ae-lock", action="store_true",
        help=(
            "Picamera2: disable auto-exposure at whatever value it holds right "
            "before capture starts (not a converged/settled value); for controlled "
            "shoots. The viewfinder's SHUTTER and ISO controls are then greyed out "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--awb-lock", action="store_true",
        help=(
            "Picamera2: disable auto white balance at whatever value it holds right "
            "before capture starts (not a converged/settled value); use --colour-gains "
            "instead to set a known white balance (default: auto)"
        ),
    )
    parser.add_argument(
        "--colour-gains", type=_colour_gains, default=None, metavar="R,B",
        help="Picamera2: fixed colour gains R,B; implies AWB disabled (controlled shoots)",
    )
    parser.add_argument(
        "--white-balance", type=_white_balance, default=None, metavar="K",
        help=(
            "Picamera2: start with the white balance fixed at K kelvin "
            f"({COLOUR_TEMPERATURE_RANGE_K[0]}-{COLOUR_TEMPERATURE_RANGE_K[1]}; higher is "
            "warmer). The viewfinder's WB control can change it or return to auto "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--ae-constraint", choices=("normal", "highlight", "shadows"), default=None,
        help=(
            "Picamera2 AE constraint: highlight protects bright regions from clipping "
            "(default: normal)"
        ),
    )
    parser.add_argument(
        "--ae-metering", choices=("centre", "spot", "matrix"), default=None,
        help="Picamera2 AE metering (default: centre-weighted)",
    )
    parser.add_argument(
        "--no-dng", action="store_true", help="disable the Picamera2 DNG sidecar output",
    )
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-preview", action="store_true", help="disable the live preview")
    parser.add_argument(
        "--show-captures", action="store_true",
        help="display each saved graded photo until the next capture (disables live preview)",
    )
    parser.add_argument("--fake", action="store_true", help="synthetic camera, no hardware")
    parser.add_argument("--seed", type=int, default=None, help="seed the grain seed generator")
    parser.add_argument(
        "--remote-listen", type=_remote_listen, metavar="HOST:PORT",
        help="serve the authenticated remote capture API (requires PIFILM_REMOTE_TOKEN)",
    )
    parser.add_argument(
        "--ups", choices=UPS_CHOICES, default="none",
        help="UPS to read the Pi's battery from and publish in /v1/status (default: none)",
    )
    parser.add_argument(
        "--display", choices=("none", "waveshare35dsi", "waveshare28", "fake"), default="none",
        help="LCD viewfinder with exposure meter (Picamera2 or --fake only): "
             "'waveshare35dsi' is the 3.5 inch DSI panel, 'waveshare28' the 2.8 inch SPI "
             "panel, 'fake' writes OUT/viewfinder-last.png instead of driving a panel",
    )
    parser.add_argument("--display-rotate", type=int, choices=(0, 180), default=0,
                        help="rotate the LCD image and touch mapping by 180 degrees")
    parser.add_argument("--touch-debug", action="store_true",
                        help="print raw and mapped touch coordinates (orientation check)")
    parser.add_argument("--display-dim-after", type=_idle_seconds, default=60.0,
                        metavar="SECONDS",
                        help="halve the LCD backlight after this long untouched; 0 disables "
                             "(no effect on the DSI panel, which has no brightness control)")
    parser.add_argument("--display-off-after", type=_idle_seconds, default=300.0,
                        metavar="SECONDS",
                        help="turn the LCD screen off after this long untouched "
                             "(a tap wakes it); 0 disables")
    parser.add_argument(
        "--shutter-gpio", type=_gpio_pin, default=None, metavar="BCM",
        help="take a photo when a push button wired between this BCM pin and ground is "
             "pressed (for example 21: physical pin 40, with ground on pin 39)",
    )
    parser.add_argument(
        "--grade", choices=("background", "inline"), default="background",
        help="'background' frees the camera once the original is saved and grades the "
             "full-resolution file behind the shutter; 'inline' grades before the camera "
             "is free, as before (default: background)",
    )
    parser.add_argument(
        "--grade-backend", choices=GRADE_BACKENDS, default="auto",
        help="'auto' grades the full-resolution file on the Pi's GPU when it can (a "
             "background grade, a look with the tile grain, a working GPU) and on the CPU "
             "otherwise; 'cpu' always grades on the CPU, as before (default: auto)",
    )
    parser.add_argument(
        "--state-dir", type=Path, default=None, metavar="PATH",
        help="where unfinished background grades are kept so a restart can finish them "
             "(default: $XDG_STATE_HOME/pifilm/pending, or ~/.local/state/pifilm/pending)",
    )
    parser.add_argument(
        "--nextcloud-env", type=Path, default=None, metavar="PATH",
        help="the .env of the Nextcloud sync, for the settings menu's UPLOAD row; this "
             "process never reads it and only hands the path to the sync it starts "
             "(default: no UPLOAD row)",
    )
    parser.add_argument(
        "--disk-warn-percent", type=_disk_percent, default=DEFAULT_WARN_PERCENT, metavar="N",
        help="warn (a log line, and an amber PHOTOS or SD tag on the LCD) while the photo "
             f"folder's filesystem or / is N%% full or more, 1-100 (default: "
             f"{DEFAULT_WARN_PERCENT}); photographs are still taken",
    )
    args = parser.parse_args(argv)

    if args.shutter_gpio is not None:
        if args.ups == "x728" and args.shutter_gpio in X728_PINS:
            parser.error(f"--shutter-gpio {args.shutter_gpio} is used by the X728 UPS")
        if args.display == "waveshare28" and args.shutter_gpio in WAVESHARE28_PINS:
            parser.error(f"--shutter-gpio {args.shutter_gpio} is used by the 2.8 inch LCD")

    if args.white_balance is not None and (args.awb_lock or args.colour_gains is not None):
        parser.error("--white-balance cannot be combined with --awb-lock or --colour-gains")

    if args.camera == "picamera2" and args.device is not None:
        parser.error("--device cannot be used with --camera picamera2")
    if args.camera == "v4l2":
        _drop_display_on_v4l2(args)
        _reject_picamera2_only_flags(parser, args, reason="cannot be used with --camera v4l2")

    autofocus = args.autofocus or "continuous"
    af_range = args.af_range or "normal"

    remote_token = os.environ.get("PIFILM_REMOTE_TOKEN") if args.remote_listen else None
    if args.remote_listen and not remote_token:
        print("error: --remote-listen requires PIFILM_REMOTE_TOKEN", file=sys.stderr)
        return 2

    power = None
    try:
        power = build_ups(args.ups)
    except PowerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        pipeline = Pipeline(Artifacts.resolve(args.artifacts))
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # The backend is resolved before the panel is opened, and the panel before
    # the camera. Preview mode is a continuous binned stream plus a mode switch
    # per shot, so it is only worth paying for once there is a panel to feed;
    # opening the camera first would leave a failed display with the camera
    # stuck in preview mode for the whole session, and opening the panel before
    # knowing the backend would claim the SPI bus only to abandon it on V4L2.
    backend = None
    if not args.fake:
        backend = args.camera
        if backend is None:
            backend = "v4l2" if args.device is not None else (
                "picamera2" if _picamera2_available() else "v4l2"
            )
        if backend == "v4l2":
            _drop_display_on_v4l2(args)
    if args.fake:
        # Argument errors must precede any hardware side effect: the panel below
        # claims the panel (SPI, GPIO and I2C, or the DRM output) and may pulse a reset line.
        _reject_picamera2_only_flags(parser, args, reason="cannot be used with --fake")
    # The menu's settings (see "The settings menu"): the unit's flags are the
    # defaults, and what the menu saved wins over them. Only a real Picamera2
    # takes them, so under --fake or V4L2 the arguments stay as given.
    settings_path = (args.state_dir or default_state_dir()).parent / SETTINGS_FILE
    settings_defaults = Settings(
        save_dng=not args.no_dng,
        ae_constraint=args.ae_constraint or "normal",
        ae_metering=args.ae_metering or "centre",
    )
    effective, overrides, lines = load_settings(settings_path, settings_defaults)
    for line in lines:
        print(line)
    image = {"sharpness": 1.0, "contrast": 1.0, "saturation": 1.0}
    short_exposure = False
    if backend == "picamera2":
        image = {key: getattr(effective, key) for key in IMAGE_KEYS}
        short_exposure = effective.short_exposure
        args.no_dng = not effective.save_dng
        args.ae_constraint = effective.ae_constraint
        args.ae_metering = effective.ae_metering
    display_pair = None
    if args.display != "none":
        try:
            display_pair = _open_display(args, args.out)
        except DisplayError as exc:
            print(f"warning: display unavailable ({exc}); continuing without it", file=sys.stderr)
    try:
        if args.fake:
            camera: Camera = FakeCamera()
        else:
            if backend == "picamera2":
                camera = Picamera2Camera(
                    args.tuning_file,
                    save_dng=not args.no_dng,
                    autofocus=autofocus,
                    af_range=af_range,
                    ae_lock=args.ae_lock,
                    awb_lock=args.awb_lock,
                    colour_gains=args.colour_gains,
                    colour_temperature=args.white_balance,
                    ae_constraint=args.ae_constraint or "normal",
                    ae_metering=args.ae_metering or "centre",
                    preview=True if display_pair is not None else None,
                    short_exposure=short_exposure,
                    **image,
                )
            else:
                _reject_picamera2_only_flags(
                    parser, args, reason="cannot be used with the selected V4L2 backend"
                )
                camera = V4L2Camera(args.device)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _close_display(display_pair)
        return 2

    session = CaptureSession(
        camera,
        pipeline,
        args.out,
        seed_rng=np.random.default_rng(args.seed),
        save_dng=not args.no_dng,
        background=args.grade == "background",
        state_dir=args.state_dir,
        grade_backend=args.grade_backend,
        halation=effective.halation if backend == "picamera2" else True,
    )
    settings = SettingsControl(settings_path, settings_defaults, overrides, log=print)
    menu_rows: tuple[str, ...] | None = None
    if backend == "picamera2":
        settings.register("save_dng", lambda on: setattr(session, "save_dng", on))
        settings.register("halation", lambda on: setattr(session, "halation", on))
        settings.register("ae_constraint", lambda name: camera.set_ae_constraint(name))
        settings.register("ae_metering", lambda name: camera.set_ae_metering(name))
        for key in IMAGE_KEYS:
            settings.register(
                key, lambda value, key=key: camera.set_image_control(key, value),
            )
        settings.register("short_exposure", lambda on: camera.set_short_exposure(on))
        # Under --ae-lock the AE setters would succeed and change nothing.
        exposure = getattr(camera, "exposure_adjustable", True) is not False
        menu_rows = ("save_dng", "halation") + (
            ("ae_constraint",)
            if exposure and getattr(camera, "ae_constraint_supported", False) else ()
        ) + (
            ("ae_metering",)
            if exposure and getattr(camera, "ae_metering_supported", False) else ()
        ) + tuple(
            key for key in IMAGE_KEYS
            if key in getattr(camera, "image_controls_supported", ())
        )
    # Read at each button press (see "The settings menu"); 0.0 off Picamera2.
    def shutter_delay() -> float:
        return settings.current.shutter_delay if backend == "picamera2" else 0.0

    look = _look_name(args.artifacts)
    # Whether the HALATION row has anything to turn off.
    halation_params = getattr(getattr(pipeline, "artifacts", None), "halation", None)
    look_halation = bool(getattr(halation_params, "active", False))
    upload: Upload | None = None
    controller: CaptureController | None = None
    remote: RemoteCaptureServer | None = None
    button: ShutterButton | None = None
    disk: DiskMonitor | None = None
    network: NetworkMonitor | None = None
    if display_pair is not None:
        controller = CaptureController(session)
    try:
        if power is not None:
            power.start()
            print(f"Reading the {args.ups} UPS every 10 s.")
        # For every loop, not only the viewfinder: the log lines help headless too.
        # Looked up like ``close_grades`` below, so a session without it runs unwatched.
        watch_disk = getattr(session, "watch_disk", None)
        if callable(watch_disk):
            disk = watch_disk(args.disk_warn_percent)
            disk.start()
        watch_network = getattr(session, "watch_network", None)
        if callable(watch_network):
            network = watch_network()
            network.start()
        if args.nextcloud_env is not None and display_pair is not None:
            upload = Upload(args.nextcloud_env.expanduser(), log=print)
        menu = {"settings": settings, "upload": upload, "look": look, "menu_rows": menu_rows,
                "look_halation": look_halation}
        if args.shutter_gpio is not None:
            try:
                button = open_shutter_button(args.shutter_gpio)
            except ButtonError as exc:
                print(
                    f"warning: shutter button unavailable ({exc}); continuing without it",
                    file=sys.stderr,
                )
        # The remote API and the button are both triggers that arrive on another
        # thread, so either one needs the shared controller and the loops that
        # tolerate it. A button that failed to open leaves the mode as it was.
        shared = bool(args.remote_listen) or button is not None
        if shared:
            controller = controller or CaptureController(session)
        if button is not None:
            assert controller is not None
            button.on_press(functools.partial(_button_capture, controller, shutter_delay))
            print(f"Shutter button on BCM {args.shutter_gpio}.")
        if args.remote_listen:
            assert controller is not None
            try:
                remote = RemoteCaptureServer(
                    controller, remote_token, args.remote_listen,
                    power=power.snapshot if power else None,
                )
                remote.start()
            except OSError as exc:
                print(f"error: could not start remote listener: {exc}", file=sys.stderr)
                return 2
            host, port = args.remote_listen
            print(f"Remote capture API listening on {host}:{port}.")
        if shared:
            try:
                if display_pair is not None:
                    assert controller is not None
                    return _run_viewfinder(
                        camera, controller, display_pair, power, args, **menu,
                    )
                if has_display() and (args.show_captures or not args.no_preview):
                    if sys.stdin.isatty():
                        with TerminalKeys() as keys:
                            displayed = run_preview_loop(
                                session, CAPTURE_WINDOW_NAME, captures_only=True,
                                read_key=lambda: keys.read(timeout=0), controller=controller,
                            )
                    else:
                        displayed = run_preview_loop(
                            session, CAPTURE_WINDOW_NAME, captures_only=True, controller=controller,
                        )
                    if displayed:
                        return 0
                    print("Capture display unavailable; remote API remains active.")
                if sys.stdin.isatty():
                    with TerminalKeys() as keys:
                        run_headless_loop(session, keys.read, controller=controller)
                    return 0
                return _serve_until_interrupt()
            except KeyboardInterrupt:
                return 0
        if display_pair is not None:
            assert controller is not None
            return _run_viewfinder(camera, controller, display_pair, power, args, **menu)
        if args.show_captures and has_display():
            if sys.stdin.isatty():
                with TerminalKeys() as keys:
                    displayed = run_preview_loop(
                        session, CAPTURE_WINDOW_NAME, captures_only=True,
                        read_key=lambda: keys.read(timeout=0),
                    )
            else:
                displayed = run_preview_loop(session, CAPTURE_WINDOW_NAME, captures_only=True)
            if displayed:
                return 0
            print("Capture display unavailable; falling back to headless mode.")
        elif args.show_captures:
            print("No display attached; falling back to headless mode.")
        elif not args.no_preview and has_display():
            if run_preview_loop(session):
                return 0
            print("Preview unavailable (OpenCV built without GUI); falling back to headless mode.")
        if not sys.stdin.isatty():
            print(
                "error: stdin is not a terminal, so keys cannot be read. Run from a terminal, "
                "or use pifilm-process for files.",
                file=sys.stderr,
            )
            return 2
        with TerminalKeys() as keys:
            run_headless_loop(session, keys.read)
        return 0
    except _Terminated:
        return 0  # stopped by SIGTERM; the clean-up below is the whole of the exit
    finally:
        _ignore_sigterm()
        if button is not None:
            # The GPIO library can raise while joining its thread or releasing
            # the line; that must not skip the closes below, which release the
            # camera and let a capture in flight finish.
            try:
                button.close()
            except Exception as exc:
                print(
                    f"warning: shutter button did not close cleanly ({exc})",
                    file=sys.stderr,
                )
        if power is not None:
            power.close()
        if disk is not None:
            disk.close()
        if network is not None:
            network.close()
        if upload is not None:
            upload.close()
        if remote is not None:
            remote.close()
        if controller is not None:
            # Its worker finishes the queued grades, then closes the camera.
            controller.close()
        else:
            # No controller, or one a loop owned and has already closed: finishing
            # the grades twice is harmless.
            close_grades = getattr(session, "close_grades", None)
            if callable(close_grades):
                try:
                    close_grades()
                except Exception as exc:
                    print(f"warning: grades did not finish cleanly ({exc})", file=sys.stderr)
            camera.close()


if __name__ == "__main__":
    sys.exit(main())
