"""Experiment: photograph each scene twice, libcamera's adaptive contrast on and off.

Step 5 of docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md,
"Adaptive contrast off". Run ON the Pi with the capture service stopped (the
camera can have one owner):

    sudo systemctl stop pifilm-capture
    .venv/bin/python scripts/experiments/contrast_pairs.py
    sudo systemctl start pifilm-capture

Press Enter to take a pair, type q and Enter to stop. Keep the camera still
during a pair (a tripod or a steady surface): the two shots are about 3 s apart.

Why the second shot copies the first one's exposure: ``rpi.contrast`` (the
adaptive contrast enhancement, ``ce_enable`` in the tuning file) works on the
picture after the exposure has been chosen, so the exposure and the white
balance must not differ between the two, or the comparison measures them
instead. The first shot runs with automatic exposure and white balance, the
way the service does (``--ae-constraint highlight``, the service's tuning
file); its ``ExposureTime``, ``AnalogueGain`` and ``ColourGains`` are then fixed
for the second shot, which differs only in ``ce_enable: 0``. The camera has to
be reopened for that, because a tuning file is read when the camera opens.

Writes ``~/contrast-pairs/<date>/HHMMSS_ce-on.jpg`` and ``HHMMSS_ce-off.jpg``
(quality 95, the camera's main stream, like the service's ``_original.jpg``)
and one line per pair in ``pairs.jsonl`` with both shots' metadata. Nothing in
the service or the photo folder is touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pifilm.imageio import save_jpeg

SETTLE_S = 1.5   # automatic exposure and white balance settle in about a second
FIXED_FRAMES = 4  # frames to let the fixed controls take hold


def contrast_off(tuning: dict) -> dict:
    """A copy of ``tuning`` with the adaptive contrast enhancement switched off."""
    off = copy.deepcopy(tuning)
    for algo in off.get("algorithms", []):
        if "rpi.contrast" in algo:
            algo["rpi.contrast"]["ce_enable"] = 0
            return off
    raise SystemExit("the tuning file has no rpi.contrast block; nothing to switch off")


def _open(Picamera2, tuning: dict):
    cam = Picamera2(tuning=tuning)
    size = tuple(cam.sensor_resolution)
    cam.configure(cam.create_still_configuration(main={"size": size, "format": "RGB888"},
                                                 buffer_count=2))
    return cam


def _auto_controls(cam, lc) -> dict:
    controls = {"ExposureValue": 0.0}
    try:
        controls["AeConstraintMode"] = lc.AeConstraintModeEnum.Highlight
    except AttributeError:
        pass
    return controls


def _fixed_controls(cam, md: dict, lc) -> dict:
    controls = {"ExposureTime": int(md["ExposureTime"]),
                "AnalogueGain": float(md["AnalogueGain"]),
                "AwbEnable": False, "ColourGains": tuple(md["ColourGains"])}
    # libcamera 0.5 takes manual exposure through the modes; older stacks through AeEnable.
    if "ExposureTimeMode" in cam.camera_controls:
        controls["ExposureTimeMode"] = lc.ExposureTimeModeEnum.Manual
        controls["AnalogueGainMode"] = lc.AnalogueGainModeEnum.Manual
    else:
        controls["AeEnable"] = False
    return controls


def _shoot(cam, wait_frames: int | None) -> tuple[np.ndarray, dict]:
    cam.start()
    try:
        if wait_frames is None:
            time.sleep(SETTLE_S)
        else:
            for _ in range(wait_frames):
                cam.capture_metadata()
        request = cam.capture_request()
        try:
            bgr = request.make_array("main")
            rgb = np.ascontiguousarray(bgr[..., ::-1])   # RGB888 is BGR in memory
            md = request.get_metadata()
        finally:
            request.release()
    finally:
        cam.stop()
    return rgb, md


def _keep(md: dict) -> dict:
    names = ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
             "ColourTemperature", "Lux", "FocusFoM")
    return {k: md[k] for k in names if k in md}


def take_pair(Picamera2, lc, tuning_on: dict, tuning_off: dict, out: Path) -> dict:
    stem = datetime.now().strftime("%H%M%S")
    cam = _open(Picamera2, tuning_on)
    try:
        cam.set_controls(_auto_controls(cam, lc))
        on, md_on = _shoot(cam, None)
    finally:
        cam.close()
    cam = _open(Picamera2, tuning_off)
    try:
        cam.set_controls(_fixed_controls(cam, md_on, lc))
        off, md_off = _shoot(cam, FIXED_FRAMES)
    finally:
        cam.close()
    save_jpeg(on, out / f"{stem}_ce-on.jpg")
    save_jpeg(off, out / f"{stem}_ce-off.jpg")
    record = {"time": stem, "ce_on": _keep(md_on), "ce_off": _keep(md_off)}
    with (out / "pairs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="Take scene pairs with adaptive contrast on and off.")
    ap.add_argument("--tuning-file", default="imx477.json",
                    help="the service's tuning file (default imx477.json)")
    ap.add_argument("--out", type=Path, default=Path.home() / "contrast-pairs")
    args = ap.parse_args()

    from libcamera import controls as lc
    from picamera2 import Picamera2

    tuning_on = Picamera2.load_tuning_file(args.tuning_file)
    tuning_off = contrast_off(tuning_on)
    out = args.out / datetime.now().strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    print(f"pairs go to {out}")
    while True:
        answer = input("Enter: take a pair, q: stop > ").strip().lower()
        if answer == "q":
            return 0
        r = take_pair(Picamera2, lc, tuning_on, tuning_off, out)
        on, off = r["ce_on"], r["ce_off"]
        print(f"{r['time']}: {on.get('Lux', 0):.0f} lux; exposure {on.get('ExposureTime')} us / "
              f"{off.get('ExposureTime')} us, gain {on.get('AnalogueGain', 0):.2f} / "
              f"{off.get('AnalogueGain', 0):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
