"""The viewfinder state machine: LIVE ⇄ REVIEW.

Every shot goes through the shared ``CaptureController`` exactly like a Stick
request, so the LCD is one more trigger and one more screen, never a second
owner of the camera. A finished job is noticed through the controller's
``finished_count``, which is how a Stick shot appears here without the
controller knowing about displays.

Display and touch objects are duck-typed (``show``/``close``, ``read``/``close``)
and the clock is injectable, so the loop is tested without hardware. The loop
never closes the camera: the caller stops it through the stop event and only
then closes the controller, which owns the camera's lifetime.

``CameraError`` is imported from ``pifilm.capture.errors`` rather than
``pifilm.capture.camera`` on purpose: that module runs ``require_cv2()`` at
import time, and nothing under ``pifilm/display/`` may need OpenCV.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from PIL import Image

from ..capture.errors import CameraError
from ..imageio import load_rgb
from . import DisplayError
from .cst3530 import Tap, TapDetector
from .meter import compute_reading, format_ev
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


class ViewfinderLoop:
    def __init__(
        self, camera: Any, controller: Any, display: Any, touch: Any, *,
        clock: Any = time, power_snapshot: Callable[[], Any] | None = None,
        frame_period: float = 0.1, review_timeout: float = 30.0,
        max_display_failures: int = 5, log: Callable[..., None] = print,
        touch_debug: bool = False,
    ) -> None:
        self._camera, self._controller = camera, controller
        self._display, self._touch = display, touch
        self._clock, self._power = clock, power_snapshot
        self._frame_period, self._review_timeout = frame_period, review_timeout
        self._max_failures, self._log, self._touch_debug = max_display_failures, log, touch_debug
        self._taps = TapDetector()
        self.state = "LIVE"
        self.ev_comp = float(getattr(camera, "ev", 0.0))
        self._seen_finished = controller.snapshot().finished_count
        self._review_since = 0.0
        self._display_failures = 0
        self._touch_logged_at: float | None = None
        self._touch_suppressed = 0
        self._frames = 0
        self._rate_since = clock.monotonic()
        self._processing_shown = False

    # -- plumbing -------------------------------------------------------------

    def _poll_tap(self) -> Tap | None:
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
            now = self._clock.monotonic()
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
        return self._taps.feed(points, self._clock.monotonic())

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

    def _set_ev(self, value: float) -> None:
        value = max(-EV_LIMIT, min(EV_LIMIT, round(value / EV_STEP) * EV_STEP))
        try:
            self._camera.set_ev(value)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            return
        self.ev_comp = value

    def _review_image(self, job: Any) -> Image.Image:
        """Render the review screen for a finished job, whatever state it is in.

        The caption goes through ``compute_reading`` rather than formatting the
        metadata here, so a zero or non-numeric ``ExposureTime`` is handled by
        the meter's guards and the review agrees with the live readout instead
        of doing its own arithmetic. Everything is caught: this runs on the loop
        thread, and after Task 9 that is the process's main thread, so an
        unreadable file or a malformed record must cost one screen, not the
        Stick server.
        """
        if job.state != "complete" or job.result is None:
            return render_message("Capture failed", job.error_message or job.error_code or "")
        try:
            rgb, _ = load_rgb(job.result.pifilm)
            meta = job.result.record.get("camera_metadata") or {}
            reading = compute_reading(meta, rgb, self.ev_comp, None)
            caption = (
                f"{text_or_dash(reading.shutter)}  {iso_label(reading.iso)}  "
                f"EV {format_ev(self.ev_comp)}"
            )
            return render_review(rgb, caption)
        except Exception as exc:
            self._log(f"review: {exc}")
            return render_message("Review unavailable", str(exc)[:60])

    # -- states -----------------------------------------------------------------

    def step(self) -> None:
        tap = self._poll_tap()
        snap = self._controller.snapshot()
        if snap.finished_count != self._seen_finished and snap.last_finished_job is not None:
            self._seen_finished = snap.finished_count
            self.state = "REVIEW"
            self._processing_shown = False
            self._review_since = self._clock.monotonic()
            self._show(self._review_image(snap.last_finished_job))
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
        if snap.active_job is not None:
            # The grade takes about three seconds, and the camera belongs to the
            # capture worker for the still in front of it: show the same colour
            # bars the Stick and the OpenCV window show rather than a stale live
            # frame. Drawn once per job, because re-blitting identical bars at
            # 10 fps would only occupy the SPI bus. Touch keeps being polled
            # above, so a shutter tap still reaches the controller (which
            # answers busy) and EV taps still work.
            if not self._processing_shown:
                self._show(render_processing())
                self._processing_shown = True
            self._restart_rate_window()
            return
        try:
            frame = self._camera.read(full=False)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            self._restart_rate_window()
            return
        power = self._power() if self._power is not None else None
        reading = compute_reading(frame.metadata, frame.rgb, self.ev_comp, power)
        self._show(render_live(frame.rgb, reading))
        self._frames += 1
        now = self._clock.monotonic()
        if now - self._rate_since >= RATE_LOG_INTERVAL:
            self._log(f"viewfinder: {self._frames / (now - self._rate_since):.1f} fps")
            self._restart_rate_window()

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            started = self._clock.monotonic()
            self.step()
            elapsed = self._clock.monotonic() - started
            self._clock.sleep(max(0.0, self._frame_period - elapsed))
