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
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from PIL import Image

from ..capture.camera import CameraError
from ..imageio import load_rgb
from . import DisplayError
from .cst3530 import Tap, TapDetector
from .meter import compute_reading, format_ev, format_shutter
from .ui import Action, hit, render_live, render_message, render_review

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
        self._frames = 0
        self._rate_since = clock.monotonic()

    # -- plumbing -------------------------------------------------------------

    def _poll_tap(self) -> Tap | None:
        points = self._touch.read()
        if self._touch_debug and points:
            self._log(f"touch: {[(p.x, p.y) for p in points]}")
        return self._taps.feed(points, self._clock.monotonic())

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
        if job.state != "complete" or job.result is None:
            return render_message("Capture failed", job.error_message or job.error_code or "")
        rgb, _ = load_rgb(job.result.pifilm)
        meta = job.result.record.get("camera_metadata") or {}
        parts = []
        if "ExposureTime" in meta:
            parts.append(format_shutter(float(meta["ExposureTime"])))
        if "AnalogueGain" in meta:
            parts.append(f"ISO {int(round(float(meta['AnalogueGain']) * 100 / 10) * 10)}")
        parts.append(f"EV {format_ev(self.ev_comp)}")
        return render_review(rgb, "  ".join(parts))

    # -- states -----------------------------------------------------------------

    def step(self) -> None:
        tap = self._poll_tap()
        snap = self._controller.snapshot()
        if snap.finished_count != self._seen_finished and snap.last_finished_job is not None:
            self._seen_finished = snap.finished_count
            self.state = "REVIEW"
            self._review_since = self._clock.monotonic()
            self._show(self._review_image(snap.last_finished_job))
            return
        if self.state == "REVIEW":
            held = self._clock.monotonic() - self._review_since
            if tap is not None or held >= self._review_timeout:
                self.state = "LIVE"
            return
        if tap is not None:
            action = hit(tap.x, tap.y)
            if action is Action.SHUTTER:
                self._controller.submit(str(uuid.uuid4()))
            elif action is Action.EV_MINUS:
                self._set_ev(self.ev_comp - EV_STEP)
            elif action is Action.EV_PLUS:
                self._set_ev(self.ev_comp + EV_STEP)
        try:
            frame = self._camera.read(full=False)
        except CameraError as exc:
            self._log(f"camera: {exc}")
            return
        power = self._power() if self._power is not None else None
        reading = compute_reading(frame.metadata, frame.rgb, self.ev_comp, power)
        self._show(render_live(frame.rgb, reading, busy=snap.active_job is not None))
        self._frames += 1
        now = self._clock.monotonic()
        if now - self._rate_since >= RATE_LOG_INTERVAL:
            self._log(f"viewfinder: {self._frames / (now - self._rate_since):.1f} fps")
            self._frames, self._rate_since = 0, now

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            started = self._clock.monotonic()
            self.step()
            elapsed = self._clock.monotonic() - started
            self._clock.sleep(max(0.0, self._frame_period - elapsed))
