"""Measure what camera standby (Picamera2.stop/start) costs and saves on the Pi.

Hardware probe for the camera-standby design: run it ON the Pi with the
``pifilm-capture`` service stopped (it needs the camera to itself). It opens the
camera the way the LCD viewfinder does (``Picamera2Camera(preview=True)``) and
reports, per stop/start cycle:

- how long ``stop()`` and ``start()`` take, and the first frame after start;
- how many frames and milliseconds auto-exposure needs to settle back to the
  exposure x gain it had before the stop;
- the sensor's kernel power state (``/sys/bus/i2c/devices/*-001a/power/
  runtime_status``) while stopped: ``suspended`` means the IMX477 driver really
  powered the sensor down, ``active`` means stop() only paused the stream.

Then it holds the camera streaming, and stopped, for ``--power-seconds`` each,
printing wall-clock marks so a USB power meter can be read for both phases, with
CPU use from /proc/stat and SoC temperature alongside.

    sudo systemctl stop pifilm-capture.service
    .venv/bin/python scripts/measure_camera_standby.py
    sudo systemctl start pifilm-capture.service
"""

from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SETTLE_TOLERANCE = 0.05   # exposure x gain within 5 % of the pre-stop value
SETTLE_STABLE = 0.02      # and changing less than 2 % frame to frame
SETTLE_MAX_FRAMES = 120


def sensor_power_states() -> dict[str, str]:
    states = {}
    for path in glob.glob("/sys/bus/i2c/devices/*-001a/power/runtime_status"):
        states[path.split("/")[5]] = Path(path).read_text().strip()
    return states


def cpu_times() -> tuple[int, int]:
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields), idle


def cpu_percent(since: tuple[int, int]) -> float:
    total, idle = cpu_times()
    d_total, d_idle = total - since[0], idle - since[1]
    return 100.0 * (d_total - d_idle) / d_total if d_total else 0.0


def soc_temp() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except OSError:
        return None


def exposure_product(meta: dict) -> float:
    return (
        float(meta.get("ExposureTime", 0))
        * float(meta.get("AnalogueGain", 0))
        * float(meta.get("DigitalGain", 1.0))
    )


def settle(picam, target: float, started: float) -> tuple[int, float, float]:
    """Frames and ms from ``started`` until AE is back on ``target`` and steady."""
    previous = None
    for frame in range(1, SETTLE_MAX_FRAMES + 1):
        product = exposure_product(picam.capture_metadata())
        steady = previous is not None and abs(product - previous) <= SETTLE_STABLE * previous
        if target and abs(product - target) <= SETTLE_TOLERANCE * target and steady:
            return frame, (time.perf_counter() - started) * 1000, product
        previous = product
    return -1, (time.perf_counter() - started) * 1000, previous or 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--off-seconds", type=float, default=5.0,
                        help="how long each stop lasts (lets the sensor power down)")
    parser.add_argument("--power-seconds", type=float, default=60.0,
                        help="length of each power-meter phase; 0 skips them")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    from pifilm.capture.picamera import Picamera2Camera

    camera = Picamera2Camera(preview=True, save_dng=False)
    picam = camera._camera  # the probe drives Picamera2 directly, under no other user
    info = camera.stream_info
    print(f"camera: {info.sensor_mode}, preview {camera.preview_size}, tuning {info.tuning_file}")
    print(f"sensor power while streaming: {sensor_power_states() or 'no *-001a node found'}")
    try:
        time.sleep(args.warmup_seconds)
        rows = []
        for cycle in range(1, args.cycles + 1):
            before = exposure_product(picam.capture_metadata())
            t = time.perf_counter()
            picam.stop()
            stop_ms = (time.perf_counter() - t) * 1000
            time.sleep(min(1.0, args.off_seconds))
            power_off = sensor_power_states()
            time.sleep(max(0.0, args.off_seconds - 1.0))
            t = time.perf_counter()
            picam.start()
            start_ms = (time.perf_counter() - t) * 1000
            picam.capture_metadata()
            first_ms = (time.perf_counter() - t) * 1000
            frames, settle_ms, after = settle(picam, before, t)
            rows.append((stop_ms, start_ms, first_ms, frames, settle_ms))
            print(
                f"cycle {cycle}: stop {stop_ms:.0f} ms | start {start_ms:.0f} ms | "
                f"first frame {first_ms:.0f} ms | AE settled "
                + (f"after {frames} frames, {settle_ms:.0f} ms" if frames > 0
                   else f"NOT within {SETTLE_MAX_FRAMES} frames")
                + f" (exp x gain {before / 1000:.0f} -> {after / 1000:.0f}) | "
                f"sensor while stopped: {power_off or '?'}"
            )
        good = [r for r in rows if r[3] > 0]
        if good:
            start, first, frames, ms = (
                statistics.median(r[i] for r in good) for i in (1, 2, 3, 4)
            )
            print(f"median: start {start:.0f} ms, first frame {first:.0f} ms, "
                  f"AE settled {ms:.0f} ms ({frames} frames)")

        if args.power_seconds > 0:
            for label, stopped in (("STREAMING", False), ("STOPPED", True)):
                if stopped:
                    picam.stop()
                since = cpu_times()
                print(f"\n{time.strftime('%H:%M:%S')}  phase {label} for "
                      f"{args.power_seconds:.0f} s: read the power meter now")
                time.sleep(args.power_seconds)
                print(f"{time.strftime('%H:%M:%S')}  {label} done: CPU {cpu_percent(since):.1f} %, "
                      f"SoC {soc_temp()} C, sensor {sensor_power_states()}")
            picam.start()
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
