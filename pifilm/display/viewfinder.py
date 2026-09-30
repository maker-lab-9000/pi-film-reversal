"""The viewfinder state machine: LIVE ⇄ REVIEW.

Every shot goes through the shared ``CaptureController`` exactly like a Stick
request, so the LCD is one more trigger and one more screen, never a second
owner of the camera. A finished job is noticed through the controller's
``finished_count``, which is how a Stick shot appears here without the
controller knowing about displays.
Double exposure adds a third state, NOTICE: after exposure 1 of a pair there is
nothing developed to review, so a short "Exposure 1/2" message replaces the
review and the loop returns to LIVE for the second exposure.
Shutter taps go straight to the camera, like EV taps, so they also work while a
job is processing.

Under ``run()`` touch is polled on its own thread ("pifilm-touch"), not inside
the frame loop. At the long shutter speeds (up to 1 s) the preview read blocks
for a whole frame, so a frame loop that polled touch would look at the panel
about once a second. The CST3530 is polled, not latched, and ``TapDetector``
needs to see the finger down and then up within its 0.6 s hold limit, so a
quick tap would fall between two polls and never register. The thread polls
every ``touch_period`` of real time and queues finished taps; ``step()`` takes
at most one per call and handles it on the loop's clock as before. While the
thread runs it is the only code that touches the touch device, the
``TapDetector`` and the touch-error log state. ``step()`` driven directly (the
tests) has no thread and polls inline, exactly as before.

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
from typing import Any

from PIL import Image

from ..capture.errors import CameraError
from ..imageio import load_rgb
from . import DisplayError, shutter
from .cst3530 import Tap, TapDetector
from .idle import IdleDimmer, Screen
from .meter import FocusTracker, compute_reading, focus_score, format_ev
from .ui import (
    Action,
    hit,
    iso_label,
    render_live,
    render_message,
    render_processing,
    render_review,
    text_or_dash,
)

EV_STEP = 1.0 / 3.0
EV_LIMIT = 2.0
RATE_LOG_INTERVAL = 10.0
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
        self._taps = TapDetector()
        self._touch_period = touch_period
        # Set only while run()'s touch thread is alive; step() then reads taps
        # from it instead of polling the panel itself.
        self._tap_queue: queue.Queue[Tap] | None = None
        self.state = "LIVE"
        self.ev_comp = float(getattr(camera, "ev", 0.0))
        # Shutter priority exists only on backends that can fix the exposure time
        # (Picamera2, the fake); on V4L2 the buttons are neither drawn nor live.
        self._shutter_ok = callable(getattr(camera, "set_shutter", None))
        self._metered_us: float | None = None
        self._seen_finished = controller.snapshot().finished_count
        self._review_since = 0.0
        self._notice_since = 0.0
        self._display_failures = 0
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

    def _restart_rate_window(self) -> None:
        """Begin a fresh frame-rate window after a gap that was not the live view.

        The printed rate is what acceptance item 1 is read against, so the up
        to 30 s a review screen is held, or a stretch of camera errors, must
        not be averaged in as dropped frames.
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

    def _set_ev(self, value: float) -> None:
        value = max(-EV_LIMIT, min(EV_LIMIT, round(value / EV_STEP) * EV_STEP))
        try:
            self._camera.set_ev(value)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            return
        self.ev_comp = value

    def _step_shutter(self, action: Action) -> None:
        step = shutter.faster if action is Action.SHUTTER_FASTER else shutter.slower
        value = step(getattr(self._camera, "shutter_us", None), self._metered_us)
        try:
            self._camera.set_shutter(value)
        except CameraError as exc:
            self._log(f"camera: {exc}")

    def _review_image(self, job: Any) -> Image.Image:
        """Render the review screen for a finished job, whatever state it is in.

        The caption goes through ``compute_reading`` rather than formatting the
        metadata here, so a zero or non-numeric ``ExposureTime`` is handled by
        the meter's guards and the review agrees with the live readout instead
        of doing its own arithmetic. A double's composite was graded at EV 0 from
        two frames shot at their own EVs, so its caption shows the pair's recorded
        EVs (``EV e1/e2``), not wherever the dial is now. Everything is caught: this
        runs on the loop thread, and after Task 9 that is the process's main thread, so an
        unreadable file or a malformed record must cost one screen, not the
        Stick server.
        """
        if job.state != "complete" or job.result is None:
            return render_message("Capture failed", job.error_message or job.error_code or "")
        try:
            rgb, _ = load_rgb(job.result.pifilm)
            meta = job.result.record.get("camera_metadata") or {}
            reading = compute_reading(meta, rgb, self.ev_comp, None)
            ev_text = format_ev(self.ev_comp)
            double = getattr(job.result, "exposure", None) == (2, 2)
            if double:
                evs = (job.result.record.get("double") or {}).get("ev_comp")
                if isinstance(evs, list) and len(evs) == 2:
                    ev_text = "/".join(format_ev(float(ev)) for ev in evs)
            caption = f"{text_or_dash(reading.shutter)}  {iso_label(reading.iso)}  EV {ev_text}"
            if double:
                caption = f"2x  {caption}"
            return render_review(rgb, caption)
        except Exception as exc:
            self._log(f"review: {exc}")
            return render_message("Review unavailable", str(exc)[:60])

    @staticmethod
    def _processing_label(snap: Any) -> str:
        """Which exposure of a pair is on its way. The count is the one before the job."""
        if not getattr(snap, "double_exposure", False):
            return "Processing photo..."
        return PROCESSING_DOUBLE if snap.exposures_taken == 1 else PROCESSING_FIRST

    # -- states -----------------------------------------------------------------

    def step(self) -> None:
        tap = self._next_tap()
        now = self._clock.monotonic()
        if tap is not None:
            # On a dark screen the user cannot see what they touch, so the tap
            # only wakes it. Dimmed, the buttons are visible and the tap acts.
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
            job = snap.last_finished_job
            if job.state == "complete" and getattr(job.result, "exposure", None) == (1, 2):
                self.state = "NOTICE"
                self._notice_since = self._clock.monotonic()
                self._show(render_message("Exposure 1/2", "frame the second exposure"))
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
            action = hit(tap.x, tap.y)
            if action is Action.SHUTTER:
                self._controller.submit(str(uuid.uuid4()))
            elif action is Action.EV_MINUS:
                self._set_ev(self.ev_comp - EV_STEP)
            elif action is Action.EV_PLUS:
                self._set_ev(self.ev_comp + EV_STEP)
            elif (action in (Action.SHUTTER_FASTER, Action.SHUTTER_SLOWER)
                  and self._shutter_ok):
                self._step_shutter(action)
            elif action is Action.DOUBLE_TOGGLE and snap.active_job is None:
                # Ignored while a job runs (spec section 5): the badge is hidden behind
                # the colour bars, so a tap there was not aimed at it.
                self._controller.set_double_exposure(not snap.double_exposure)
        if snap.active_job is not None:
            # The grade takes about three seconds, and the camera belongs to the
            # capture worker for the still in front of it: show the same colour
            # bars the Stick and the OpenCV window show rather than a stale live
            # frame. Drawn once per job, because re-blitting identical bars at
            # 10 fps would only occupy the SPI bus. Touch keeps being polled
            # above, so a shutter tap still reaches the controller (which
            # answers busy) and EV taps still work.
            if not self._processing_shown:
                self._show(render_processing(self._processing_label(snap)))
                self._processing_shown = True
            self._restart_rate_window()
            return
        if self._idle.screen(now) is Screen.OFF:
            # Dark screen: neither read nor draw the preview, which is most of the
            # loop's CPU. Touch and the controller are still polled above.
            self._restart_rate_window()
            return
        try:
            frame = self._camera.read(full=False)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            self._restart_rate_window()
            return
        exposure = (frame.metadata or {}).get("ExposureTime")
        if isinstance(exposure, (int, float)) and exposure > 0:
            self._metered_us = float(exposure)
        power = self._power() if self._power is not None else None
        # Manual focus aid: the bar is the absolute sharpness of the frame's centre,
        # the tracker only remembers the best of the last few seconds for the mark.
        level = focus_score(frame.rgb)
        peak = self._focus.update(level, self._clock.monotonic())
        reading = compute_reading(
            frame.metadata, frame.rgb, self.ev_comp, power, focus=level, focus_peak=peak,
            shutter_us=getattr(self._camera, "shutter_us", None),
            max_gain=getattr(self._camera, "max_gain", None),
        )
        self._show(render_live(
            frame.rgb, reading, (snap.double_exposure, snap.exposures_taken),
            shutter_buttons=self._shutter_ok,
        ))
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
        worker.start()
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
