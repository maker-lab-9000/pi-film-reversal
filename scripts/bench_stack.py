"""Time the multi-frame merge (``pifilm.stack.merge``) on this machine.

    .venv/bin/python scripts/bench_stack.py --folder ~/Pictures/pifilm/2026-10-24 --shot 142233
    .venv/bin/python scripts/bench_stack.py --frames ref.jpg b.jpg c.jpg d.jpg

``--folder`` reads ``captures.jsonl`` there and takes the frames a record's ``stack``
lists (``--shot`` is the time stamp, ``142233``, or the original's name,
``142233_original.jpg``). ``--frames`` takes JPEGs directly, the first one the
reference. The frames are loaded once; ``merge`` then runs ``--runs`` times (3 by
default) and the medians of ``align_ms``, ``merge_ms`` and the wall time of the
call are printed, with the frames used and the peak resident memory of the
process. No camera, no ``.env``: it runs on the Pi with the repo's venv, to decide
whether six frames is affordable there, and on the Mac.
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

from pifilm.capture.batch import CAPTURE_LOG, load_capture_log
from pifilm.imageio import load_rgb
from pifilm.stack import merge


class BenchError(Exception):
    pass


def frames_from_log(folder: Path, shot: str) -> list[Path]:
    """The record's frame files, the reference first."""
    name = shot if shot.endswith(".jpg") else f"{shot}_original.jpg"
    if not (folder / CAPTURE_LOG).is_file():
        raise BenchError(f"no {CAPTURE_LOG} in {folder}")
    entries, _bad = load_capture_log(folder)
    entry = entries.get(name)
    if entry is None:
        raise BenchError(f"no record for {name} in {folder / CAPTURE_LOG}")
    if entry.frames is None or len(entry.frames) < 2:
        raise BenchError(f"the record for {name} is not a multi-frame photograph")
    paths = [folder / frame for frame in entry.frames]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise BenchError(f"frames missing from {folder}: {', '.join(missing)}")
    return paths


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / (1 << 10)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", type=Path, help="a capture folder with captures.jsonl")
    source.add_argument("--frames", type=Path, nargs="+", help="JPEGs, the reference first")
    ap.add_argument("--shot", help="with --folder: HHMMSS or HHMMSS_original.jpg")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)
    if args.folder is not None and not args.shot:
        ap.error("--folder needs --shot")
    if args.frames is not None and args.shot:
        ap.error("--shot goes with --folder")
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    try:
        if args.folder is not None:
            paths = frames_from_log(args.folder, args.shot)
        else:
            paths = list(args.frames)
            if len(paths) < 2:
                raise BenchError("--frames needs at least two JPEGs")
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise BenchError(f"no such file: {', '.join(missing)}")
        frames = [load_rgb(p)[0] for p in paths]
        if any(f.shape != frames[0].shape for f in frames):
            raise BenchError("the frames differ in size")
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    height, width = frames[0].shape[:2]
    print(f"{len(frames)} frames of {width}x{height}, reference {paths[0].name}")
    aligns, merges, totals = [], [], []
    used = 0
    for run in range(1, args.runs + 1):
        start = time.perf_counter()
        _result, info = merge(frames, reference=0)
        total = (time.perf_counter() - start) * 1000.0
        used = sum(f["used"] for f in info["frames"])
        aligns.append(info["align_ms"])
        merges.append(info["merge_ms"])
        totals.append(total)
        print(f"run {run}: align_ms {info['align_ms']:.1f}, merge_ms {info['merge_ms']:.1f}, "
              f"total_ms {total:.1f}")
    print(f"median of {args.runs}: align_ms {statistics.median(aligns):.1f}, "
          f"merge_ms {statistics.median(merges):.1f}, total_ms {statistics.median(totals):.1f}")
    print(f"frames used: {used} of {len(frames)}")
    print(f"peak RSS: {peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
