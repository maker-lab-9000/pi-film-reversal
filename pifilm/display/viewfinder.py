"""The viewfinder state machine: LIVE ⇄ REVIEW.

Every shot goes through the shared ``CaptureController`` exactly like a Stick
request, so the LCD is one more trigger and one more screen, never a second
owner of the camera. A finished job is noticed through the controller's
``finished_count``, which is how a Stick shot appears here without the
controller knowing about displays.
Double exposure adds a third state, NOTICE: after exposure 1 of a pair there is
nothing developed to review, so a short "Exposure 1/2" message replaces the
review and the loop returns to LIVE for the second exposure.
The live screen (``liveview.py``) is select-and-adjust: a tap on one of the strip's
three cells (SHUTTER, ISO, WB) selects it, and one ``+``/``-`` pair steps whichever
is selected. While a job is processing the colour bars hide the cells and the
keys, so those taps are ignored then, like the ``2x`` toggle's. Which cells the
camera can change is decided once, at start-up; a cell it cannot change is drawn
grey and is never selected. Each control is automatic
while the camera holds ``None`` for it, and the first step out of auto lands on the
step nearest what the camera is metering, so the picture does not jump.

Press feedback. Actions fire on release (that is what makes a tap a tap, and why a
resting finger never repeats a step), but a key that only reacted on release would
feel dead, so ``+``, ``-`` and the shutter button are drawn lit while a finger is
on them. The finger's position is recorded on every touch poll for that. A tap can
also be over between two frames, so a tap on a key lights it for
``PRESS_FLASH_SECONDS`` as well.

Under ``run()`` touch is polled on its own thread ("pifilm-touch"), not inside
the frame loop. At the long shutter speeds (up to 1 s) the preview read blocks
for a whole frame, so a frame loop that polled touch would look at the panel
about once a second. Touch is polled, not latched, on either panel, and ``TapDetector``
needs to see the finger down and then up within its hold limit, so a
quick tap would fall between two polls and never register. The thread polls
every ``touch_period`` of real time and queues finished taps; ``step()`` takes
at most one per call and handles it on the loop's clock as before. While the
thread runs it is the only code that touches the touch device, the
``TapDetector`` and the touch-error log state. ``step()`` driven directly (the
tests) has no thread and polls inline, exactly as before.

Under ``run()`` the camera is read on its own thread too ("pifilm-frames"), for
the other half of the same problem: a loop that waited a second for each frame
also took a second to act on a queued tap and to redraw the key and the value it
changed. The loop now turns over at its own rate whatever the shutter speed.
Each step draws the most recent frame with the controls as they are at that
moment, and the meter, the focus measure and the metered values are worked out
once per camera frame, not once per redraw. The slow picture at a slow shutter
is the exposure and stays; the slow controls are gone.
The frame thread reads only when asked: every live step sets ``_want_frames``
and the thread takes that as a request for one frame, so the camera is read no
more often than the loop turns over (as when the loop read it itself), and not
at all while the screen is off or showing colour bars, a notice or a review. At
a slow shutter the request is already waiting when a read returns, so the reads
run back to back. A read already in flight when the live view is left still
finishes, and its frame is the one shown when the live view comes back; it is
shown but not counted, because the ``viewfinder: N fps`` line counts frames the
camera delivered to the live view and is read as the camera's rate.
While the thread runs it is the only code that reads preview frames; the
camera's control setters are still called from the loop, which
``Picamera2Camera`` allows beside a preview read (see its locks). ``step()``
driven directly has no thread and reads its frame inline, exactly as before.

Each new frame is also the check that the camera took the controls it was sent.
On the Pi 4 a fixed white balance has been seen not to reach the live view until
a photo re-applied the configuration, so a frame whose metadata disagrees with
the fixed shutter, ISO or white balance for long enough is logged and acted on:
shutter and ISO are sent again and, if that changes nothing, the live view is
restarted; white balance restarts the live view at once (see
``CONTROL_RESEND_SECONDS`` and ``_verify_controls``). Each successful step is
also traced in the journal about two seconds later (``CONTROL_TRACE_SECONDS``).

Display and touch objects are duck-typed (``show``/``close``, ``read``/``close``)
and the clock is injectable, so the loop is tested without hardware. The loop
never closes the camera: the caller stops it through the stop event and only
then closes the controller, which owns the camera's lifetime.

``CameraError`` is imported from ``pifilm.capture.errors`` rather than
``pifilm.capture.camera`` on purpose: that module runs ``require_cv2()`` at
import time, and nothing under ``pifilm/display/`` may need OpenCV.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ..capture.errors import CameraError, CameraStalled
from ..imageio import load_rgb
from . import DisplayError, iso, shutter, stepscale, whitebalance
from .idle import IdleDimmer, Screen
from .liveview import (
    DASH,
    PRESS_KEYS,
    SELECT_ACTIONS,
    Action,
    Cell,
    LiveState,
    hit,
    render_live,
)
from .meter import FocusTracker, MeterReading, compute_reading, focus_score, format_shutter
from .touch import Tap, TapDetector
from .ui import iso_label, render_message, render_processing, render_review, text_or_dash

RATE_LOG_INTERVAL = 10.0
# How long a tapped key stays lit after the finger is up.
PRESS_FLASH_SECONDS = 0.15
# The longest press that still counts as a tap. The detector's own default is
# 0.6 s, which was right when nothing showed a press; now the keys light under
# the finger, which invites a longer, more deliberate press, and a key that lit
# and then did nothing on release reads as a fault.
TAP_MAX_HOLD = 1.5
# Cell indexes, taken from the screen's own order so the two cannot drift.
SHUTTER_CELL, ISO_CELL, WB_CELL = (
    SELECT_ACTIONS.index(action)
    for action in (Action.SELECT_SHUTTER, Action.SELECT_ISO, Action.SELECT_WB)
)
# Consecutive preview reads that may time out before the loop gives the camera up.
CAMERA_STALL_LIMIT = 3
# How long run() waits for the frame thread once the loop has ended. A read at
# the slowest shutter takes a second, so this covers any healthy one; a read
# that is waiting out a stall (five seconds and more) is left behind instead.
FRAME_THREAD_JOIN_SECONDS = 2.0
# Verifying the fixed controls. Each new frame's metadata is compared with the
# selection, because on the Pi 4 a control sent to the running camera does not
# always take effect, and a disagreement that lasts is written to the journal
# and acted on. The journal line is the evidence; the cause is not known.
#
# Measured on the Pi 4 + IMX477 (libcamera 0.7.2, Picamera2 0.3.37) on
# 2026-10-12, from these journal lines:
# - before the first photo after start-up a live white-balance change is
#   applied within 2 s;
# - after a photo or a restart of the live view (after any mode switch) a live
#   white-balance change, to a fixed value or back to auto, is usually not
#   applied: 11 of 11 first sends were ignored, sending again cured 2 of about
#   7, and restarting the live view cured it every time (3 of 3);
# - a live shutter change after a photo was applied within 2 s (one case).
# The same camera class driven from a stand-alone script took live changes
# after a photo, and 300 control changes against Picamera2 alone lost none.
#
# So the remedy depends on the control. Shutter and ISO are sent again when the
# disagreement has lasted this long over at least this many new frames (both: a
# new value takes a few frames to show at any shutter speed, and at 1 s three
# seconds is only three frames), and after this many resends that changed
# nothing the live view is restarted, which reconfigures the camera as a photo
# does.
CONTROL_RESEND_SECONDS = 3.0
CONTROL_RESEND_FRAMES = 6
CONTROL_RESENDS_BEFORE_RESTART = 2
# White balance is never resent, since that so rarely helped: the live view is
# restarted as soon as the disagreement has lasted this long over this many new
# frames. After a photo a new white balance therefore shows in the old colour
# for about a second and a half, then the picture blinks once and is right.
WB_RESTART_SECONDS = 1.5
WB_RESTART_FRAMES = 6
# A return to auto gets longer: libcamera's own damping was measured holding
# the old value for about 2 s before it starts to drift, and a restart inside
# that would be a false alarm.
WB_AUTO_RESTART_SECONDS = 4.0
# One restart per control per selection. A restart that did not help is not
# repeated (a 60 s hold-off was tried first and left the user stuck in the
# wrong colour after the first restart, 2026-10-12), and neither is one that
# helped for a moment: a control that agrees for a frame and then disagrees
# again would otherwise buy a new restart every time and blink for as long as
# it flapped. Only a step that changes that control, or coming back to the live
# view, allows another. Whatever the selections, two restarts are never closer
# than this, counted from the end of the first, so no run of taps can make the
# picture blink continuously.
CONTROL_RESTART_MIN_INTERVAL = 5.0
# The camera by itself applies a live control correctly, even after a photo
# (tried on the Pi: the reported value and the picture both change), so the
# fault is somewhere in how this app runs it, and the evidence has to come from
# inside the app. Each successful step is therefore followed by one journal
# line: what was selected, what the camera reports and what the picture looks
# like, taken from the first new frame at least this long after the step, by
# when a control that is going to take has taken.
CONTROL_TRACE_SECONDS = 2.0
# The cell names as the trace line writes them, in cell order, and the camera
# attribute that holds each selection.
_TRACE_NAMES = ("SHUTTER", "ISO", "WB")
_SELECTION_ATTRIBUTES = ("shutter_us", "iso", "colour_temperature_k")
# The control check's keys for each cell: white balance has one for a fixed
# value and one for a return to auto.
_CHECK_KEYS = (("shutter",), ("iso",), ("wb", "awb"))
# A camera that never takes the control is resent to every few seconds, but
# the line is written this often at most, per control.
CONTROL_LOG_INTERVAL = 30.0
# How far the camera's report may be from the selection and still agree: the
# sensor rounds an exposure time to whole lines and a gain to its own steps.
WB_TOLERANCE_K = 100.0
EXPOSURE_TOLERANCE = 0.05
# Automatic white balance that reports the last fixed value to within this has
# not started: once running it drifts by tens of kelvin from frame to frame.
AWB_UNMOVED_K = 1.0
# Idle dimming: half brightness after a minute untouched, off after five. The
# normal level is the panel driver's own default (st7789.BACKLIGHT_DEFAULT).
DIM_AFTER = 60.0
OFF_AFTER = 300.0
FULL_BACKLIGHT = 80
# Double exposure: how long the "Exposure 1/2" notice holds before the live view
# returns for the second exposure (a tap ends it sooner), and the processing labels.
NOTICE_SECONDS = 1.2
PROCESSING_FIRST = "Exposure 1 of 2..."
PROCESSING_DOUBLE = "Developing double exposure..."


@dataclass
class _Episode:
    """One unbroken disagreement of one control with one selected value.

    ``began`` is when it was first seen and ``since`` when its current wait
    started, both on the loop's clock; ``frames`` counts the new camera frames
    of that wait. ``awaiting`` names the remedy ("resend" or "restart") whose
    effect has yet to be seen. Whether the control's restart has been used is
    not the episode's to know: that outlives it (``_restart_spent``).
    """

    began: float
    since: float
    frames: int = 0
    resends: int = 0
    awaiting: str | None = None


@dataclass(frozen=True)
class _Grab:
    """What one preview read produced: a frame, or the error it raised.

    ``seq`` numbers the reads, so the loop can tell a new result from the one
    it has already handled.
    """

    seq: int
    frame: Any = None
    error: CameraError | None = None


class ViewfinderLoop:
    def __init__(
        self, camera: Any, controller: Any, display: Any, touch: Any, *,
        clock: Any = time, power_snapshot: Callable[[], Any] | None = None,
        frame_period: float = 0.1, review_timeout: float = 30.0,
        max_display_failures: int = 5, log: Callable[..., None] = print,
        touch_debug: bool = False, dim_after: float = DIM_AFTER,
        off_after: float = OFF_AFTER, full_backlight: int = FULL_BACKLIGHT,
        touch_period: float = 0.02,
    ) -> None:
        self._camera, self._controller = camera, controller
        self._display, self._touch = display, touch
        self._clock, self._power = clock, power_snapshot
        self._frame_period, self._review_timeout = frame_period, review_timeout
        self._max_failures, self._log, self._touch_debug = max_display_failures, log, touch_debug
        # The layout is in 320x240 base units; a 640-wide panel draws it at 2x and
        # reports taps in its own pixels. The tap detector's drag limit is in panel
        # pixels too, so it grows with the scale.
        self._scale = max(1, int(getattr(display, "width", 320)) // 320)
        self._taps = TapDetector(max_hold=TAP_MAX_HOLD, max_move=20.0 * self._scale)
        self._touch_period = touch_period
        # Set only while run()'s touch thread is alive; step() then reads taps
        # from it instead of polling the panel itself.
        self._tap_queue: queue.Queue[Tap] | None = None
        # Set only while run()'s frame thread is alive; step() then asks it for
        # frames through this event instead of reading the camera itself.
        self._want_frames: threading.Event | None = None
        # The frame thread's latest result. It is written by that thread and
        # read by step(), once per step into a local: one assignment of an
        # immutable record is the only cross-thread traffic, as with the finger
        # below, so it needs no lock.
        self._grab: _Grab | None = None
        # The rest is the loop's own. The number of the last result handled;
        # the picture, meter reading and metered white balance of the last good
        # frame, which every redraw until the next one reuses; and whether the
        # previous step was a live one.
        self._grab_seq = 0
        self._view: tuple[Any, MeterReading, Any] | None = None
        self._was_live = False
        # Verifying the fixed controls (see CONTROL_RESEND_SECONDS): the
        # disagreement in progress for each control, by its key; when each
        # control's resend line was last written and how many have been held
        # back since; the controls whose one restart for the current selection
        # has been used, and those for which its failure has been reported;
        # when the last restart ended; and the last white balance that was
        # fixed, until auto is seen to have left it or has been restarted for.
        self._episodes: dict[str, _Episode] = {}
        self._resend_logged: dict[str, tuple[float, int]] = {}
        self._restart_spent: set[str] = set()
        self._restart_told: set[str] = set()
        self._restarted_at: float | None = None
        self._last_fixed_k: float | None = None
        # The step waiting for its trace line: the cell and when it was stepped.
        self._trace: tuple[int, float] | None = None
        self.state = "LIVE"
        # Which of the three controls this camera can change, by cell. A fixed
        # shutter or ISO needs a backend with the setter (Picamera2, the fake)
        # whose auto-exposure was not locked from the command line
        # (``exposure_adjustable``, False under --ae-lock). Manual white balance
        # likewise needs one that has not had its white balance fixed already,
        # and whose libcamera has a settable ColourTemperature (both reported as
        # ``wb_adjustable``).
        exposure_ok = getattr(camera, "exposure_adjustable", True) is not False
        self._available = (
            exposure_ok and callable(getattr(camera, "set_shutter", None)),
            exposure_ok and callable(getattr(camera, "set_iso", None)),
            callable(getattr(camera, "set_colour_temperature", None))
            and getattr(camera, "wb_adjustable", True) is not False,
        )
        # The cell that + and - act on: the first one the camera can change, kept
        # until another is tapped. None when it can change none of them.
        self.selected: int | None = next(
            (cell for cell, ok in enumerate(self._available) if ok), None)
        # What auto is using now, by cell: exposure time (us), ISO, kelvin. Each
        # keeps its last usable value, so a frame without metadata or a camera
        # error does not send the next first step back to the default.
        self._metered: list[float | None] = [None, None, None]
        # Where the finger is, in panel pixels, or None when it is up. Written by
        # whoever polls touch (the touch thread under run()) and read by step():
        # one assignment of an immutable tuple is the only cross-thread traffic,
        # so it needs no lock.
        self._finger: tuple[int, int] | None = None
        # A tapped key and the loop-clock time until which it stays lit.
        self._flash: tuple[Action, float] | None = None
        self._seen_finished = controller.snapshot().finished_count
        self._review_since = 0.0
        self._notice_since = 0.0
        self._display_failures = 0
        self._camera_stalls = 0
        self._touch_logged_at: float | None = None
        self._touch_suppressed = 0
        self._frames = 0
        self._rate_since = clock.monotonic()
        self._processing_shown = False
        # The focus gauge's "best seen" mark. It is never reset on a state change:
        # the peak fades on wall time alone, so a review or a spell of colour bars
        # leaves the mark where a few seconds of decay put it.
        self._focus = FocusTracker()
        # Activity is a touch or a capture (in progress or just finished, from
        # this screen or the Stick). The backlight is written only when its level
        # changes; None forces the first step to set it. A display that cannot
        # dim (``--display fake``) gets no idle handling at all: treating it as
        # off would freeze a screen that is still lit.
        if not callable(getattr(display, "backlight", None)):
            dim_after = off_after = 0.0
        # A panel that can only be on or off (the DSI one has no brightness control)
        # skips the dim step and keeps the off step.
        if getattr(display, "dimmable", True) is False:
            dim_after = 0.0
        self._idle = IdleDimmer(
            full=full_backlight, dim_after=dim_after, off_after=off_after,
            now=clock.monotonic(),
        )
        self._backlight: int | None = None

    # -- plumbing -------------------------------------------------------------

    def _poll_tap(self, monotonic: Callable[[], float] | None = None) -> Tap | None:
        """Read the panel once and feed the tap detector.

        ``monotonic`` is the time source for the detector and the error log:
        the loop's clock when polled inline, real time on the touch thread.
        """
        monotonic = monotonic or self._clock.monotonic
        try:
            points = self._touch.read()
        except DisplayError as exc:
            # The panel shares I2C1 with the X728 gauge and RTC, so a single NAK
            # is a glitch, not a dead panel: report no finger and carry on. It
            # does not count toward the display breaker, which is about the
            # screen the user is looking at, and feeding the detector an empty
            # list also stops a half-seen press from wedging it down forever.
            # The log is rate-limited because a panel that never answers would
            # otherwise write a line every frame for as long as the service runs.
            now = monotonic()
            if (
                self._touch_logged_at is None
                or now - self._touch_logged_at >= RATE_LOG_INTERVAL
            ):
                more = f" ({self._touch_suppressed} more)" if self._touch_suppressed else ""
                self._log(f"touch: {exc}{more}")
                self._touch_logged_at, self._touch_suppressed = now, 0
            else:
                self._touch_suppressed += 1
            points = []
        except Exception:
            # Anything else is the caller's to report (the touch thread logs it
            # and carries on), but the finger last seen is no longer known to
            # be there: forget it, or its key stays lit while the fault lasts.
            self._finger = None
            raise
        # Recorded on every poll, an empty one included, so a key never stays
        # lit after the finger has left or the panel has stopped answering.
        self._finger = (points[0].x, points[0].y) if points else None
        if self._touch_debug and points:
            self._log(f"touch: {[(p.x, p.y) for p in points]}")
        return self._taps.feed(points, monotonic())

    def _next_tap(self) -> Tap | None:
        """One tap for this step: from the touch thread's queue when it runs,
        otherwise by polling the panel inline."""
        taps = self._tap_queue
        if taps is None:
            return self._poll_tap()
        try:
            return taps.get_nowait()
        except queue.Empty:
            return None

    def _touch_worker(self, stop: threading.Event, halt: threading.Event,
                      taps: queue.Queue[Tap]) -> None:
        """Poll the panel on real time until the loop stops, queueing taps.

        ``halt`` is the loop's own signal for when ``run()`` ends by an
        exception (a display breaker trip, Ctrl-C), which does not set the
        caller's ``stop``. An unexpected error is logged and polling carries on
        rather than letting the thread die and silently take touch with it.
        """
        while not (stop.is_set() or halt.is_set()):
            try:
                tap = self._poll_tap(time.monotonic)
            except Exception as exc:
                self._log(f"touch: {exc}")
                tap = None
            if tap is not None:
                taps.put(tap)
            halt.wait(self._touch_period)

    def _read_grab(self, seq: int) -> _Grab:
        """One preview read as a record: the frame, or the ``CameraError`` it raised."""
        try:
            return _Grab(seq, frame=self._camera.read(full=False))
        except CameraError as exc:
            return _Grab(seq, error=exc)

    def _frame_worker(self, stop: threading.Event, halt: threading.Event,
                      want: threading.Event, seq: int) -> None:
        """Read one preview frame for each request until the loop stops.

        ``want`` is the loop's request and is taken (cleared) before the read,
        so a request made during a long read starts the next one at once and a
        loop that has left the live view starts none. A camera error is the
        loop's to judge (it logs it and counts stalls), so it is published like
        a frame. Anything else is logged here and the thread carries on, as the
        touch thread does: dying would freeze the picture without a word.
        After either the thread waits a frame period, so a camera that fails
        at once is not asked again in a tight loop.
        """
        while not (stop.is_set() or halt.is_set()):
            if not want.wait(self._frame_period):
                continue
            want.clear()
            if halt.is_set():
                break
            seq += 1
            try:
                grab = self._read_grab(seq)
            except Exception as exc:
                self._log(f"frames: {exc}")
                halt.wait(self._frame_period)
                continue
            self._grab = grab
            if grab.error is not None:
                halt.wait(self._frame_period)

    def _control_check_entering_live(self) -> None:
        """Start the control check afresh on coming back to the live view.

        Runs on the first live step after colour bars, a review, a notice or a
        dark screen (and on the very first step). A still re-applies the whole
        configuration, so what the frames before it showed says nothing about
        the ones after: every episode ends, every control may have its restart
        again, and a pending trace line is dropped, since it would describe the
        photo's reconfiguration and not the step.
        """
        self._episodes.clear()
        self._restart_spent.clear()
        self._restart_told.clear()
        self._trace = None

    def _control_check_stepped(self, cell: int, changed: bool) -> None:
        """Start the waits again after a ``+``/``-`` step on ``cell``.

        A value has just been sent, so every control gets its full wait and
        the resend line may be written at once. Only a step that changed the
        selection makes it a new selection with a restart of its own, and only
        for the control that was stepped.
        """
        self._episodes.clear()
        self._resend_logged.clear()
        if changed:
            self._restart_spent.difference_update(_CHECK_KEYS[cell])
            self._restart_told.difference_update(_CHECK_KEYS[cell])

    def _note_fixed_wb(self) -> None:
        kelvin = getattr(self._camera, "colour_temperature_k", None)
        if stepscale.is_usable(kelvin):
            self._last_fixed_k = float(kelvin)

    def _control_checks(
        self, metadata: dict,
    ) -> list[tuple[str, str, tuple[str, str] | None]]:
        """Compare a frame's metadata with the fixed controls.

        One entry for each control that could be judged: its key, its name for
        the log, and ``None`` when the camera agrees or the two halves of the
        complaint when it does not. A control on auto has nothing to agree
        with, one whose cell the camera cannot change is not this screen's, and
        a missing or meaningless metadata value cannot tell either way: none of
        those is listed. The exception on auto is a white balance that was
        fixed earlier and has not moved at all since it was released.
        """
        camera = self._camera
        checks: list[tuple[str, str, tuple[str, str] | None]] = []
        us = getattr(camera, "shutter_us", None)
        exposure = metadata.get("ExposureTime")
        if self._available[SHUTTER_CELL] and stepscale.is_usable(us) \
                and stepscale.is_usable(exposure):
            what = f"shutter {shutter.label(us)}"
            agrees = abs(exposure - us) <= EXPOSURE_TOLERANCE * us
            checks.append(("shutter", what, None if agrees else (
                f"{what} not in the live view",
                f" (camera reports {format_shutter(exposure)})",
            )))
        fixed_iso = getattr(camera, "iso", None)
        gain = metadata.get("AnalogueGain")
        if self._available[ISO_CELL] and stepscale.is_usable(fixed_iso) \
                and stepscale.is_usable(gain):
            what = f"ISO {iso.label(fixed_iso)}"
            agrees = abs(gain - fixed_iso / 100) <= EXPOSURE_TOLERANCE * fixed_iso / 100
            checks.append(("iso", what, None if agrees else (
                f"{what} not in the live view",
                f" (camera reports ISO {round(gain * 100)})",
            )))
        kelvin = getattr(camera, "colour_temperature_k", None)
        reported = metadata.get("ColourTemperature")
        if not self._available[WB_CELL] or not stepscale.is_usable(reported):
            return checks
        if stepscale.is_usable(kelvin):
            what = f"white balance {round(kelvin)}K"
            agrees = abs(reported - kelvin) <= WB_TOLERANCE_K
            checks.append(("wb", what, None if agrees else (
                f"{what} not in the live view", f" (camera reports {round(reported)}K)",
            )))
        elif self._last_fixed_k is not None:
            what = "auto white balance"
            if abs(reported - self._last_fixed_k) <= AWB_UNMOVED_K:
                checks.append(("awb", what, (
                    f"{what} has not moved from {round(self._last_fixed_k)}K", "",
                )))
            else:
                # Auto is running: reaching the old value again later is not a fault.
                self._last_fixed_k = None
                checks.append(("awb", what, None))
        return checks

    @staticmethod
    def _picture(rgb: Any) -> str:
        """The frame's colour for a journal line: channel means as R/G and B/G.

        It is what the picture looks like, beside what the camera says it did:
        a white balance the camera reports but the picture does not show is
        only visible this way. Worked out when a line is written, never for
        every frame.
        """
        try:
            red, green, blue = (float(rgb[..., channel].mean()) for channel in range(3))
        except Exception:
            red = green = blue = 0.0
        if not stepscale.is_usable(green):
            return "picture R/G - B/G -"
        return f"picture R/G {red / green:.2f} B/G {blue / green:.2f}"

    def _trace_step(self, rgb: Any, metadata: dict, now: float) -> None:
        """Write the pending step's trace line once its frame has come.

        All three reported values are written whatever was stepped, with a
        dash for one the frame does not carry: a step of one control that
        moved another is exactly the kind of thing the line is for.
        """
        if self._trace is None or now - self._trace[1] < CONTROL_TRACE_SECONDS:
            return
        cell = self._trace[0]
        self._trace = None
        selection = getattr(self._camera, _SELECTION_ATTRIBUTES[cell], None)
        if not stepscale.is_usable(selection):
            selected = "AUTO"
        elif cell == SHUTTER_CELL:
            selected = shutter.label(selection)
        elif cell == ISO_CELL:
            selected = iso.label(selection)
        else:
            selected = f"{round(selection)}K"
        kelvin = metadata.get("ColourTemperature")
        gains = metadata.get("ColourGains")
        exposure = metadata.get("ExposureTime")
        gain = metadata.get("AnalogueGain")
        if (
            isinstance(gains, (list, tuple)) and len(gains) == 2
            and all(stepscale.is_usable(value) for value in gains)
        ):
            gains_text = f"({gains[0]:.2f}, {gains[1]:.2f})"
        else:
            gains_text = DASH
        self._log(
            f"control: {_TRACE_NAMES[cell]} {selected} selected; camera reports "
            f"{f'{round(kelvin)}K' if stepscale.is_usable(kelvin) else DASH} "
            f"gains {gains_text}, "
            f"exposure {format_shutter(exposure) if stepscale.is_usable(exposure) else DASH}, "
            f"ISO {round(gain * 100) if stepscale.is_usable(gain) else DASH}; "
            f"{self._picture(rgb)}"
        )

    def _verify_controls(self, metadata: dict, now: float, rgb: Any = None) -> None:
        """Check one new frame against the fixed controls; act when overdue.

        Called once per camera frame, never for a redraw: only a new frame is
        new evidence. A control that agrees ends its episode; one the frame
        cannot tell about keeps its episode but starts its wait again, so the
        frame that brings the metadata back is not acted on at once. For one
        that disagrees, shutter and ISO are sent again and, after
        ``CONTROL_RESENDS_BEFORE_RESTART`` resends without effect, the live
        view is restarted; white balance goes straight to the restart (see the
        constants for why). A control has one restart per selection
        (``_restart_spent``), whether or not it agreed in between: once it is
        used, a disagreement is said once, shutter and ISO go on being resent
        and white balance is left alone. A restart for an auto white balance
        that has not moved also ends the watch on auto: on a capped lens or in
        the dark auto reports the old value for good, and that must not cost a
        restart after every photo.

        A failed resend is logged and survived like any other camera error from
        a control; so is a failed restart, except that one which times out is a
        stalled camera and is counted as one (it raises at
        ``CAMERA_STALL_LIMIT``). The lines that announce a resend or a restart
        end with the picture's colour, from ``rgb``.
        """
        self._note_fixed_wb()
        restart = getattr(self._camera, "restart_preview", None)
        may_restart = callable(restart) and (
            self._restarted_at is None
            or now - self._restarted_at >= CONTROL_RESTART_MIN_INTERVAL
        )
        restart_line: str | None = None
        resend_due = False
        checks = self._control_checks(metadata)
        judged = {kind for kind, _, _ in checks}
        for kind, episode in self._episodes.items():
            if kind not in judged:
                episode.since, episode.frames = now, 0
        for kind, what, fault in checks:
            if fault is None:
                self._resend_logged.pop(kind, None)
                episode = self._episodes.pop(kind, None)
                if episode is not None and episode.awaiting == "restart":
                    self._log(f"camera: {what} applied after restarting the live view")
                elif episode is not None and episode.awaiting == "resend":
                    self._log(f"camera: {what} applied after resend")
                continue
            before, after = fault
            episode = self._episodes.get(kind)
            if episode is None:
                episode = self._episodes[kind] = _Episode(began=now, since=now)
            episode.frames += 1
            elapsed = f"{now - episode.began:.1f} s"
            if kind in ("wb", "awb"):
                seconds = WB_AUTO_RESTART_SECONDS if kind == "awb" else WB_RESTART_SECONDS
                if episode.frames < WB_RESTART_FRAMES or now - episode.since < seconds:
                    continue
                if kind in self._restart_spent:
                    # The restart has had a full wait and the camera
                    # disagrees. Said once, and nothing more is tried.
                    if kind not in self._restart_told:
                        self._restart_told.add(kind)
                        self._log(
                            f"camera: {what} not applied even after restarting the live view"
                        )
                elif not callable(restart):
                    self._restart_spent.add(kind)
                    self._restart_told.add(kind)
                    self._log(
                        f"camera: {before} after {elapsed}{after}; cannot restart the "
                        f"live view; {self._picture(rgb)}"
                    )
                elif may_restart and restart_line is None:
                    restart_line = (
                        f"camera: {before} after {elapsed}{after}; restarting the "
                        f"live view; {self._picture(rgb)}"
                    )
                continue
            if (
                episode.frames < CONTROL_RESEND_FRAMES
                or now - episode.since < CONTROL_RESEND_SECONDS
            ):
                continue
            episode.since, episode.frames = now, 0
            spent = kind in self._restart_spent
            if spent and kind not in self._restart_told:
                self._restart_told.add(kind)
                self._log(f"camera: {what} not applied even after restarting the live view")
            if (
                not spent
                and episode.resends >= CONTROL_RESENDS_BEFORE_RESTART
                and may_restart
                and restart_line is None
            ):
                restart_line = (
                    f"camera: {what} still not in the live view after "
                    f"{episode.resends} resends; restarting the live view; "
                    f"{self._picture(rgb)}"
                )
                continue
            episode.resends += 1
            episode.awaiting = "resend"
            resend_due = True
            logged = self._resend_logged.get(kind)
            if logged is not None and now - logged[0] < CONTROL_LOG_INTERVAL:
                self._resend_logged[kind] = (logged[0], logged[1] + 1)
                continue
            more = f" ({logged[1]} more)" if logged is not None and logged[1] else ""
            self._log(
                f"camera: {before} after {elapsed}{after}; resending{more}; "
                f"{self._picture(rgb)}"
            )
            self._resend_logged[kind] = (now, 0)
        if restart_line is not None:
            self._log(restart_line)
            # The restart re-applies every control, so it is the one restart of
            # every control with an episode open now.
            self._restart_spent.update(self._episodes)
            if self._episodes.pop("awb", None) is not None:
                # Auto is not watched again until a white balance is next fixed.
                self._last_fixed_k = None
            failure: CameraError | None = None
            try:
                # On the loop thread, and it waits for the frame thread's read
                # in flight (one exposure) and then for the camera to restart:
                # the screen and the keys stand still that long. Accepted for
                # a restart; the alternative is a third thread.
                restart()
            except CameraError as exc:
                failure = exc
            # Timed from the end, not the start: a restart can take seconds,
            # and neither the five seconds to the next one nor the wait to see
            # this one's effect may be shortened by that.
            done = self._clock.monotonic()
            self._restarted_at = done
            for episode in self._episodes.values():
                episode.since, episode.frames, episode.awaiting = done, 0, None
            if failure is not None:
                # Not restarted, so there is nothing to report as surviving it;
                # the one attempt is spent all the same.
                self._restart_told.update(self._episodes)
                self._camera_failed(failure)
                return
            for episode in self._episodes.values():
                episode.awaiting = "restart"
            # The time the restart took is not the live view's frame rate.
            self._restart_rate_window()
            return
        if not resend_due:
            return
        resend = getattr(self._camera, "reassert_controls", None)
        if callable(resend):
            try:
                resend()
            except CameraError as exc:
                self._log(f"camera: {exc}")

    def _camera_failed(self, exc: CameraError) -> None:
        """Log a camera error from the live view; raise it when the camera has stalled out.

        One late frame is survivable. A camera that has stopped delivering
        them does not recover inside this process, so at ``CAMERA_STALL_LIMIT``
        consecutive stalls the error is raised, the loop ends and the caller
        exits for a restart.
        """
        self._log(f"camera: {exc}")
        self._restart_rate_window()
        if isinstance(exc, CameraStalled):
            self._camera_stalls += 1
            if self._camera_stalls >= CAMERA_STALL_LIMIT:
                raise exc

    def _restart_rate_window(self) -> None:
        """Begin a fresh frame-rate window after a gap that was not the live view.

        The printed rate is what acceptance item 1 is read against, so the up
        to 30 s a review screen is held, or a stretch of camera errors, must
        not be averaged in as dropped frames. It counts camera frames, not
        redraws: under run() the same frame is redrawn until the next arrives.
        """
        self._frames, self._rate_since = 0, self._clock.monotonic()

    def _show(self, image: Image.Image) -> None:
        try:
            self._display.show(image)
        except DisplayError as exc:
            self._display_failures += 1
            self._log(f"display: {exc} ({self._display_failures}/{self._max_failures})")
            if self._display_failures >= self._max_failures:
                raise
            return
        self._display_failures = 0

    def _apply_backlight(self, now: float) -> None:
        """Drive the panel's backlight to the idle level, if it has one.

        A display without ``backlight`` is skipped (and was given no idle
        delays in ``__init__``). A failure is logged once per level and not retried
        every frame, and it never stops the viewfinder: a stuck backlight is a
        brighter screen, not a broken camera.
        """
        level = self._idle.backlight(now)
        if level == self._backlight:
            return
        self._backlight = level
        setter = getattr(self._display, "backlight", None)
        if not callable(setter):
            return
        try:
            setter(level)
        except Exception as exc:
            self._log(f"display: backlight {level}%: {exc}")

    def _step_selected(self, up: bool) -> None:
        """Move the selected control one step; ``up`` is the ``+`` key.

        ``+`` is a faster shutter, a higher ISO or a warmer white balance. A
        control the camera rejects is logged and left as it was: the setters
        change nothing when they raise.
        """
        cell = self.selected
        if cell is None:
            return
        camera, metered = self._camera, self._metered[cell]
        if cell == SHUTTER_CELL:
            step = shutter.faster if up else shutter.slower
            value = step(getattr(camera, "shutter_us", None), metered)
            setter = camera.set_shutter
        elif cell == ISO_CELL:
            # Steps the sensor's gain cannot reach are left off the scale, so
            # the first step from auto never lands on one the camera refuses.
            max_gain = getattr(camera, "max_gain", None)
            max_iso = max_gain * 100 if stepscale.is_usable(max_gain) else None
            step = iso.higher if up else iso.lower
            value = step(getattr(camera, "iso", None), metered, max_iso)
            setter = camera.set_iso
        else:
            step = whitebalance.warmer if up else whitebalance.cooler
            value = step(getattr(camera, "colour_temperature_k", None), metered)
            setter = camera.set_colour_temperature
        # Before the setter: a white balance stepped from fixed to auto between
        # two frames must still be remembered as the value auto has to leave.
        self._note_fixed_wb()
        before = getattr(camera, _SELECTION_ATTRIBUTES[cell], None)
        try:
            setter(value)
        except CameraError as exc:
            self._log(f"camera: {exc}")
        # A rejected step, or + on the top step, leaves the control as it was.
        changed = getattr(camera, _SELECTION_ATTRIBUTES[cell], None) != before
        self._control_check_stepped(cell, changed)
        if changed:
            # Trace it, in place of any step still waiting.
            self._trace = (cell, self._clock.monotonic())

    def _cells(self, reading: MeterReading, metered_k: Any) -> tuple[Cell, Cell, Cell]:
        """The strip's three read-outs for this frame.

        A fixed control shows the value the camera holds; an automatic one shows
        what this frame was metered at, or a dash when the frame does not say.
        A control the camera cannot change shows the metered value too, but
        without the ``AUTO`` tag: under --ae-lock, --awb-lock or --colour-gains
        it is fixed, only not by this screen.
        Automatic ISO reads ``MAX`` when the gain has nowhere left to go, which
        is the warning that a fixed shutter this fast will underexpose.
        """
        camera = self._camera
        us = getattr(camera, "shutter_us", None)
        fixed_iso = getattr(camera, "iso", None)
        kelvin = getattr(camera, "colour_temperature_k", None)
        if us is not None:
            shutter_text = shutter.label(us)
        else:
            shutter_text = reading.shutter or DASH
        if fixed_iso is not None:
            iso_text = iso.label(fixed_iso)
        elif reading.iso_max:
            iso_text = "MAX"
        else:
            iso_text = str(reading.iso) if reading.iso is not None else DASH
        if kelvin is not None:
            wb_text = whitebalance.value_label(kelvin)
        elif stepscale.is_usable(metered_k):
            wb_text = whitebalance.value_label(metered_k)
        else:
            wb_text = DASH
        shutter_ok, iso_ok, wb_ok = self._available
        return (
            Cell("SHUTTER", shutter_text, auto=shutter_ok and us is None, enabled=shutter_ok),
            Cell("ISO", iso_text, auto=iso_ok and fixed_iso is None, enabled=iso_ok),
            Cell("WB", wb_text, auto=wb_ok and kelvin is None, enabled=wb_ok),
        )

    def _pressed(self, now: float) -> Action | None:
        """The key to draw lit: the one under the finger, else a tap's flash.

        ``now`` is the time this step began, not the time the frame is drawn:
        when the step reads the camera itself, a one-second preview read
        between the two outlasts the flash, and a tap handled in this step must
        still be seen in its frame.
        """
        finger = self._finger
        if finger is not None:
            action = hit(finger[0], finger[1], self._scale)
            if action in PRESS_KEYS:
                return action
        if self._flash is not None:
            action, until = self._flash
            if now < until:
                return action
            self._flash = None
        return None

    def _review_image(self, job: Any) -> Image.Image:
        """Render the review screen for a finished job, whatever state it is in.

        The caption goes through ``compute_reading`` rather than formatting the
        metadata here, so a zero or non-numeric ``ExposureTime`` is handled by
        the meter's guards and the review agrees with the live readout instead
        of doing its own arithmetic. A double's caption is marked ``2x`` and
        shows the second exposure's values. Everything is caught: this
        runs on the loop thread, and after Task 9 that is the process's main thread, so an
        unreadable file or a malformed record must cost one screen, not the
        Stick server.
        """
        if job.state != "complete" or job.result is None:
            return render_message(
                "Capture failed", job.error_message or job.error_code or "", scale=self._scale,
            )
        try:
            rgb, _ = load_rgb(job.result.pifilm)
            meta = job.result.record.get("camera_metadata") or {}
            reading = compute_reading(meta, rgb, None)
            caption = f"{text_or_dash(reading.shutter)}  {iso_label(reading.iso)}"
            if getattr(job.result, "exposure", None) == (2, 2):
                caption = f"2x  {caption}"
            return render_review(rgb, caption, scale=self._scale)
        except Exception as exc:
            self._log(f"review: {exc}")
            return render_message("Review unavailable", str(exc)[:60], scale=self._scale)

    @staticmethod
    def _processing_label(snap: Any) -> str:
        """Which exposure of a pair is on its way. The count is the one before the job."""
        if not getattr(snap, "double_exposure", False):
            return "Processing photo..."
        return PROCESSING_DOUBLE if snap.exposures_taken == 1 else PROCESSING_FIRST

    # -- states -----------------------------------------------------------------

    def step(self) -> None:
        # No frames are wanted until this step turns out to be a live one, so
        # the frame thread starts no read behind colour bars, a review, a
        # notice or a dark screen.
        want = self._want_frames
        if want is not None:
            want.clear()
        was_live, self._was_live = self._was_live, False
        ask = True
        tap = self._next_tap()
        now = self._clock.monotonic()
        if tap is not None:
            # On a dark screen the user cannot see what they touch, so the tap
            # only wakes it (and, being dropped here, lights no key). Dimmed, the
            # buttons are visible and the tap acts.
            if self._idle.screen(now) is Screen.OFF:
                tap = None
            self._idle.wake(now)
        snap = self._controller.snapshot()
        if snap.active_job is not None or snap.finished_count != self._seen_finished:
            self._idle.wake(now)
        self._apply_backlight(now)
        if snap.finished_count != self._seen_finished and snap.last_finished_job is not None:
            self._seen_finished = snap.finished_count
            self._processing_shown = False
            self._flash = None
            job = snap.last_finished_job
            if job.state == "complete" and getattr(job.result, "exposure", None) == (1, 2):
                self.state = "NOTICE"
                self._notice_since = self._clock.monotonic()
                self._show(render_message(
                    "Exposure 1/2", "frame the second exposure", scale=self._scale,
                ))
                return
            self.state = "REVIEW"
            self._review_since = self._clock.monotonic()
            self._show(self._review_image(job))
            return
        if self.state == "NOTICE":
            held = self._clock.monotonic() - self._notice_since
            if tap is not None or held >= NOTICE_SECONDS:
                self.state = "LIVE"
                self._restart_rate_window()
            return
        if self.state == "REVIEW":
            held = self._clock.monotonic() - self._review_since
            if tap is not None or held >= self._review_timeout:
                self.state = "LIVE"
                self._processing_shown = False
                self._restart_rate_window()
            return
        if tap is not None:
            action = hit(tap.x, tap.y, self._scale)
            if snap.active_job is not None and action is not Action.SHUTTER:
                # While a job runs the colour bars hide the cells, the keys and
                # the 2x toggle (spec section 5), so a tap there was not aimed at
                # them: no step, no selection, no toggle and no flash. Only the
                # shutter tap goes on, to the controller, which answers busy.
                action = Action.NONE
            if action in PRESS_KEYS:
                self._flash = (action, now + PRESS_FLASH_SECONDS)
            if action is Action.SHUTTER:
                self._controller.submit(str(uuid.uuid4()))
                # The still waits for the camera behind any preview read, so
                # this step asks the frame thread for none: at a 1 s shutter a
                # read started now would add a second of shutter lag.
                ask = False
            elif action in SELECT_ACTIONS:
                # hit() knows nothing of the camera: a grey cell still reports
                # its tap, and is ignored here.
                cell = SELECT_ACTIONS.index(action)
                if self._available[cell]:
                    self.selected = cell
            elif action in (Action.PLUS, Action.MINUS):
                self._step_selected(action is Action.PLUS)
            elif action is Action.DOUBLE_TOGGLE:
                self._controller.set_double_exposure(not snap.double_exposure)
        if snap.active_job is not None:
            # The grade takes about three seconds, and the camera belongs to the
            # capture worker for the still in front of it: show the same colour
            # bars the Stick and the OpenCV window show rather than a stale live
            # frame. Drawn once per job, because re-blitting identical bars at
            # 10 fps would only occupy the SPI bus. Touch keeps being polled
            # above, so a shutter tap still reaches the controller (which
            # answers busy). A key's flash must not outlive the live view it
            # was drawn on.
            self._flash = None
            if not self._processing_shown:
                self._show(render_processing(self._processing_label(snap), scale=self._scale))
                self._processing_shown = True
            self._restart_rate_window()
            return
        if self._idle.screen(now) is Screen.OFF:
            # Dark screen: neither read nor draw the preview, which is most of the
            # loop's CPU. Touch and the controller are still polled above.
            self._restart_rate_window()
            return
        self._was_live = True
        if not was_live:
            self._control_check_entering_live()
        if want is None:
            # No frame thread (step() driven directly): read one frame here.
            grab = self._read_grab(self._grab_seq + 1)
        else:
            grab = self._grab
            if ask:
                want.set()
        fresh = grab is not None and grab.seq != self._grab_seq
        if fresh:
            self._grab_seq = grab.seq
            if grab.error is not None:
                self._camera_failed(grab.error)
                return
            self._camera_stalls = 0
            frame = grab.frame
            power = self._power() if self._power is not None else None
            # Manual focus aid: the bar is the absolute sharpness of the frame's
            # centre, the tracker only remembers the best of the last few seconds
            # for the mark.
            level = focus_score(frame.rgb)
            peak = self._focus.update(level, self._clock.monotonic())
            max_gain = getattr(self._camera, "max_gain", None)
            reading = compute_reading(
                frame.metadata, frame.rgb, power, focus=level, focus_peak=peak,
                max_gain=max_gain if stepscale.is_usable(max_gain) else None,
            )
            metadata = frame.metadata or {}
            metered_k = metadata.get("ColourTemperature")
            for cell, value in enumerate(
                (metadata.get("ExposureTime"), reading.iso, metered_k)
            ):
                if stepscale.is_usable(value):
                    self._metered[cell] = float(value)
            self._view = (frame.rgb, reading, metered_k)
            checked_at = self._clock.monotonic()
            self._trace_step(frame.rgb, metadata, checked_at)
            self._verify_controls(metadata, checked_at, frame.rgb)
        if self._view is None:
            # Nothing has arrived yet: the first frame of a slow shutter.
            return
        # Drawn every step, new frame or not: the cells, the selection and the
        # lit key are read now, so a tap shows at the loop's rate, not the camera's.
        rgb, reading, metered_k = self._view
        self._show(render_live(
            rgb, reading,
            LiveState(
                cells=self._cells(reading, metered_k), selected=self.selected,
                pressed=self._pressed(now),
                double=(snap.double_exposure, snap.exposures_taken),
                # Read at every frame: the count falls while no capture is running.
                developing=getattr(snap, "pending_grades", 0),
                # The tuning cannot change while the service runs; read like the count.
                tuning_mismatch=getattr(snap, "tuning_mismatch", False) is True,
            ),
            scale=self._scale,
        ))
        if not fresh:
            return
        if want is not None and not was_live:
            # The first result after a pause came from a read that was in flight
            # when the live view was left: worth showing, but not a frame of the
            # rate window that has just begun.
            return
        self._frames += 1
        now = self._clock.monotonic()
        if now - self._rate_since >= RATE_LOG_INTERVAL:
            self._log(f"viewfinder: {self._frames / (now - self._rate_since):.1f} fps")
            self._restart_rate_window()

    def run(self, stop: threading.Event) -> None:
        taps: queue.Queue[Tap] = queue.Queue()
        halt = threading.Event()
        worker = threading.Thread(
            target=self._touch_worker, args=(stop, halt, taps),
            name="pifilm-touch", daemon=True,
        )
        self._tap_queue = taps
        want = threading.Event()
        grabber = threading.Thread(
            target=self._frame_worker, args=(stop, halt, want, self._grab_seq),
            name="pifilm-frames", daemon=True,
        )
        self._grab = None
        self._want_frames = want
        # The touch device was opened seconds before this loop polls it, and the
        # kernel queued whatever touched the panel meanwhile. Read once and throw
        # the result away, past the tap detector: otherwise the first poll reports
        # that stale press, the second its release, and a finger on the panel
        # during start-up takes a photo.
        try:
            self._touch.read()
        except DisplayError:
            pass
        worker.start()
        grabber.start()
        try:
            while not stop.is_set():
                started = self._clock.monotonic()
                self.step()
                elapsed = self._clock.monotonic() - started
                self._clock.sleep(max(0.0, self._frame_period - elapsed))
        finally:
            # The caller closes the touch device as soon as run() returns, so
            # the thread must be done with it first.
            halt.set()
            worker.join(1.0)
            if worker.is_alive():
                self._log("touch: poll thread did not stop within 1 s")
            self._tap_queue = None
            # The frame thread is woken so it sees the halt at once. One that is
            # inside a read cannot be interrupted; it is given the time a healthy
            # read takes and otherwise left behind (it is a daemon, publishes to
            # a loop nobody steps any more, and ends with its read).
            want.set()
            grabber.join(FRAME_THREAD_JOIN_SECONDS)
            if grabber.is_alive():
                self._log(
                    f"frames: read thread did not stop within {FRAME_THREAD_JOIN_SECONDS:g} s"
                )
            self._want_frames = None
