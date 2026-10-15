"""Develop the DNGs behind a folder of capture originals, under the same names.

Phase 2 of docs/superpowers/specs/2026-10-14-raw-look-design.md: the raw look is
trained exactly like the JPEG look, on the same photographs, so its sources are
the DNGs of those photographs developed by ``pifilm.raw`` and saved the way the
camera's originals are (quality-95 JPEG, the same file names). Each name
``<day>_<HHMMSS>_original.jpg`` is looked up as ``<HHMMSS>.dng`` with its record in
``<photos>/<day>/captures.jsonl`` (or in ``--folder`` for a single day).

    .venv/bin/python scripts/experiments/develop_folder.py --tuning data/tuning/imx477.json \\
        --tone isp --photos ~/Downloads/PiTrainingSource --photos ~/Downloads/PiCameraPhotos \\
        data/source-imx477-v3-highlight data/source-imx477-v3-highlight-dng-isp
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pifilm.imageio import save_jpeg
from pifilm.raw import develop, read_dng, tuning_tables


def _records(photos: list[Path]) -> dict[tuple[str, str], tuple[Path, dict]]:
    found = {}
    for root in photos:
        for log in root.glob("*/captures.jsonl"):
            for line in log.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("dng"):
                    found[(log.parent.name, r["dng"].removesuffix(".dng"))] = (log.parent, r)
    return found


def _one(job: tuple[str, str, Path, dict, str, str]) -> str:
    tuning_path, tone, folder, record, name, out = job
    tuning = tuning_tables(json.loads(Path(tuning_path).read_text()))
    md = record["camera_metadata"]
    frame = read_dng((folder / record["dng"]).read_bytes())
    rgb = develop(frame, tuning, colour_gains=tuple(md["ColourGains"]),
                  colour_temperature=md["ColourTemperature"],
                  digital_gain=md.get("DigitalGain", 1.0), tone=tone)
    save_jpeg(rgb, Path(out) / name)
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--tone", default="isp")
    ap.add_argument("--photos", action="append", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    records = _records(args.photos)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, missing = [], []
    for p in sorted(args.src.glob("*_original.jpg")):
        day, hhmmss = p.name.split("_")[:2]
        hit = records.get((day, hhmmss))
        if hit is None:
            missing.append(p.name)
            continue
        jobs.append((args.tuning, args.tone, hit[0], hit[1], p.name, str(args.out)))
    if missing:
        raise SystemExit(f"no DNG for {len(missing)} photographs, e.g. {missing[:3]}")
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, name in enumerate(pool.map(_one, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
