# Camera standby: stop the stream when nobody can use it

Status: design, 2026-09-25. Builds on LCD idle dimming (PR #37).

## Goal

Stop the IMX477's stream (`Picamera2.stop()`) when neither the Stick nor the
LCD can use the camera, and start it again the moment either can, or when any
capture is requested. The aim is battery life on the X728: the camera should
cost nothing while the Pi sits idle in a bag.

## Measured on the Pi (2026-09-25)

`scripts/measure_camera_standby.py`, IMX477 in the viewfinder's preview mode
(2028x1520 binned raw, 640x480 main), dim room, exposure at its ceiling:

| Quantity | Result |
|---|---|
| `stop()` | 17-29 ms |
| `start()` | 14-21 ms |
| First frame after `start()` | 191-224 ms (median 191) |
| Auto-exposure back on its pre-stop exposure x gain | 2 frames, ~325 ms: libcamera's AGC keeps its state across stop/start |
| Sensor runtime PM while stopped (`10-001a`) | `active`: the driver keeps the sensor powered; only the stream stops |
| Wall power, streaming vs stopped (USB meter) | about 1 W lower stopped |
| CPU while streaming, no frames read | 3.2 % |

Consequences for the design:

- **Worth it.** About 1 W is a large share of a Pi 4's draw. The saving comes
  from halting sensor readout, the CSI receiver and the ISP, not from cutting
  sensor power.
- **Waking is cheap (~0.2 s) and exposure is already right**, so there is no
  exposure seeding on resume and no "waking camera" screen.
- `Picamera2.close()` (full power-down, re-open ~2 s) is not needed and is out
  of scope.

## The rule

The stream runs while **the Stick is connected** or **the LCD is lit** (full
or dim). Otherwise, once `--camera-standby-after` seconds (default 60) have
passed since the last capture, the camera goes to standby.

| Situation | Camera |
|---|---|
| Stick connected (its screen on or off) | streaming, always |
| No Stick, LCD full or dim | streaming |
| No Stick, LCD off (5 min idle, PR #37) | standby |
| No Stick, no LCD (headless) | standby 60 s after the last capture |
| Any capture request while in standby | resumes first (~0.2 s), then captures |
| `--camera-standby-after 0` | never standby (today's behaviour) |

**"Stick connected"** means an authenticated request reached the remote server
in the last 10 s (`STICK_HEARTBEAT_WINDOW`). The Stick polls `GET /v1/status`
every 500 ms in every client state, with its screen on or off, from Wi-Fi join
until it powers down (`CaptureClient::nextWork`), so no firmware change is
needed. Unauthenticated requests never count, so a stray client on the hotspot
cannot keep the camera awake.

## Components

### 1. Camera: `standby()`, `resume()`, `streaming`

`Picamera2Camera` gains:

- `standby()`: `Picamera2.stop()` under `_request_lock`; does nothing if it is
  already stopped.
- `resume()`: `Picamera2.start()` under `_request_lock`; does nothing if it is
  already streaming. It starts whichever configuration was running before the
  stop (the preview configuration when there is one).
- `streaming: bool`, read-only.
- **`read()` resumes implicitly.** A read while in standby calls `resume()`
  first. That makes standby a latency matter only, never a correctness one: the
  capture worker, the viewfinder and the OpenCV preview loops keep working
  unchanged, whatever the policy decided.

`V4L2Camera` gets no-op `standby`/`resume` and `streaming = True`: UVC cameras
are out of scope. `FakeCamera` records the calls for tests.

### 2. Policy: `CameraPower` (pure)

`pifilm/capture/standby.py`, in the style of `pifilm/display/idle.py`: no
threads, no hardware, an injectable clock.

```python
class CameraPower:
    def __init__(self, *, standby_after: float, now: float,
                 has_display: bool) -> None: ...
    def stick_seen(self, now: float) -> None: ...        # remote server, authenticated request
    def display_awake(self, awake: bool) -> None: ...    # viewfinder: FULL/DIM -> True, OFF -> False
    def captured(self, now: float) -> None: ...          # controller: a job started or finished
    def should_stream(self, now: float) -> bool: ...
```

`should_stream` is true when `standby_after == 0`, or the Stick was seen within
10 s, or `has_display` and the display is awake, or a capture happened within
`standby_after` seconds. A new `CameraPower` starts with the display awake and
the last capture at `now`, so the service streams for its first minute.

### 3. Remote server: heartbeat

`RemoteCaptureServer` takes an optional `on_client_seen: Callable[[], None]`
and calls it after `_authorized` succeeds, on every endpoint. `GET /v1/status`
gains an additive `"camera_streaming": bool` field for diagnosis; the Stick
ignores unknown fields.

### 4. Controller: captures count as activity

`CaptureController` takes an optional `on_capture: Callable[[], None]`, called
when a job starts and when it finishes. Resuming needs no controller code,
since `read()` resumes implicitly. A resume that fails raises `CameraError`
from `read()`, and the job fails with the normal `camera_error` the Stick
already displays.

### 5. Viewfinder: report the screen state

`ViewfinderLoop` takes an optional `on_screen: Callable[[bool], None]` and calls
it whenever `_apply_backlight` changes the level: `True` for full or dim,
`False` for off. A display without a backlight never reaches off, so it keeps
the camera awake. The loop already skips `read()` while off, so it never wakes
the camera by itself; the first read after a waking tap resumes it.

### 6. Wiring: `StandbyWorker`

A daemon thread in `pifilm-capture`, started with the controller, closed before
the camera. Once a second it asks `CameraPower.should_stream(now)` and calls
`camera.standby()` or `camera.resume()` when the answer differs from
`camera.streaming`. It logs one line per transition:

```text
camera: standby (no Stick for 10 s, LCD off)
camera: streaming (Stick connected)
```

The worker runs only in the modes where nothing reads the camera continuously
behind the policy's back: `--display waveshare28`/`fake`, and headless
`--no-preview`. The OpenCV live preview and `--show-captures` loops read every
frame, so each read would resume what the worker had just stopped. Standby is
disabled there, and `pifilm-capture` says so in one line if
`--camera-standby-after` was given explicitly.

`--camera-standby-after SECONDS` (default 60, `0` disables) is added beside
`--display-off-after`. The systemd unit needs no change. Worker errors
(`CameraError` from stop/start) are logged and retried on the next tick; they
never stop the service.

## Races, and why they are harmless

- **Policy stops the camera while a capture starts.** Both take
  `_request_lock`. If standby wins, the capture's `read()` resumes (~0.2 s
  later). If the capture wins, the stale standby runs right after it; the
  worker's next tick, at most a second later, sees `captured()` and resumes. The
  cost is one wasted stop/start (~0.2 s), never a failed shot.
- **Tap on a dark LCD.** The viewfinder reports awake before its next
  `read()`, and that read resumes the camera, so the live view is back about
  0.2 s after the backlight.
- **The Stick disappears mid-session.** 10 s later, if the LCD is off or absent
  and no capture happened in the last `standby_after`, the camera stops. Its
  next poll, when it reconnects, restarts the stream within a second (the
  worker's tick).

## Testing

- `CameraPower`: each row of the rule table, the 10 s heartbeat edge, `0`
  disables, a new instance streams at first, display-less behaviour.
- `Picamera2Camera` with the existing Picamera2 test double: standby/resume
  are idempotent, run under the lock, `read()` resumes, and resume restores the
  preview configuration.
- Remote server: authenticated requests call `on_client_seen`, rejected ones do
  not; `/v1/status` carries `camera_streaming`.
- Controller: `on_capture` fires at job start and end, including a failed job.
- Viewfinder: `on_screen` fires on the FULL/DIM to OFF transitions only.
- `StandbyWorker` with a fake clock and camera: transitions, one log line
  each, errors retried.
- CLI: the default, override, `0`, and negative rejected; no worker in the
  OpenCV preview and `--show-captures` modes.

## Hardware acceptance

Added to `docs/lcd-viewfinder.md` §7:

1. Stick on, LCD idle past 5 min: the journal shows no `camera: standby`, and
   a Stick shot takes the usual time.
2. Stick off, LCD reaches off: `camera: standby` within about 11 s of the Stick
   going quiet; the USB meter drops by about 1 W.
3. Tap the dark LCD: live view within about 0.5 s; `camera: streaming`.
4. Stick powered on during standby: `camera: streaming` within a second of its
   first poll; the first shot is normal.
5. Headless (no panel): standby 60 s after the last capture; a Stick-less
   capture from standby succeeds.

## Out of scope

`Picamera2.close()`/re-open and cutting the camera board's power; a wake
endpoint (the heartbeat covers it); Stick firmware changes; V4L2 cameras.
