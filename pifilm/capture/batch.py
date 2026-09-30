"""``pifilm-process``: regrade a folder of images with the current artifact.

Two hazards make this more than a loop, and both come from the shape of a
real capture folder, which holds ``<time>_original.jpg`` next to
``<time>_graded.jpg``:

* **Double grading.** Feeding that folder to a naive globber grades the
  already-graded files a second time. Files matching ``*_graded.*`` are
  therefore always skipped, and when a folder contains any ``_original`` or
  ``_ungraded`` files, only those are processed unless ``--all`` is given.
* **Lost exposure intent.** The Pi grades with the EV the frame was shot at
  (normalisation would otherwise cancel it; see ``pifilm.pipeline``) and a
  logged grain seed. A regrade that ignored ``captures.jsonl`` undid every
  deliberate EV and drew new grain, so when the folder has that log, each
  frame is graded with its logged ``ev_comp`` and ``grain_seed``: the same
  artifact then reproduces the Pi's file, up to the re-encoded original.
  Double-exposure frames are regraded singly, with their own EV and fresh
  grain, because their logged seed belongs to the composite.
* **Clobbering.** ``a.jpg`` and ``a.png`` would produce the same output
  name, and re-running would silently overwrite. Same-stem inputs get their
  extension folded into the output name, existing outputs are skipped unless
  ``--overwrite``, and an output directory equal to or inside the input is
  refused outright.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import Artifacts, ArtifactsError
from ..imageio import list_images, load_rgb, save_jpeg
from ..pipeline import Pipeline

GRADED_SUFFIXES = ("_graded",)
SOURCE_SUFFIXES = ("_original", "_ungraded")


CAPTURE_LOG = "captures.jsonl"


@dataclass
class BatchResult:
    written: list[Path] = field(default_factory=list)
    skipped_graded: int = 0
    skipped_existing: int = 0
    from_log: int = 0
    ev_applied: int = 0
    doubles: int = 0
    bad_log_lines: int = 0


@dataclass(frozen=True)
class LogEntry:
    ev: float
    seed: int | None
    double: bool = False


def _log_entry(record: dict) -> LogEntry:
    double = record.get("double")
    if isinstance(double, dict):
        evs = double.get("ev_comp")
        ev = evs[1] if double.get("index") == 2 and isinstance(evs, list) else record.get(
            "ev_comp")
        return LogEntry(float(ev or 0.0), None, double=True)
    seed = record.get("grain_seed")
    return LogEntry(float(record.get("ev_comp") or 0.0),
                    int(seed) if seed is not None else None)


def load_capture_log(in_dir: Path) -> tuple[dict[str, LogEntry], int]:
    """``original`` filename -> logged EV and seed, and the count of unusable lines."""
    path = Path(in_dir) / CAPTURE_LOG
    entries: dict[str, LogEntry] = {}
    bad = 0
    if not path.exists():
        return entries, bad
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not an object")
            if "original" not in record:
                continue
            entries[record["original"]] = _log_entry(record)
        except (ValueError, TypeError, IndexError):
            bad += 1
    return entries, bad


def _is_graded(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in GRADED_SUFFIXES)


def _is_source(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in SOURCE_SUFFIXES)


def select_inputs(paths: Sequence[Path], all_files: bool = False) -> list[Path]:
    """Graded outputs are never inputs; capture folders default to their originals."""
    candidates = [p for p in paths if not _is_graded(p)]
    if all_files:
        return candidates
    sources = [p for p in candidates if _is_source(p)]
    return sources if sources else candidates


def output_path(src: Path, out_dir: Path, disambiguate: bool) -> Path:
    stem = f"{src.stem}_{src.suffix.lstrip('.').lower()}" if disambiguate else src.stem
    return Path(out_dir) / f"{stem}_graded.jpg"


def _check_directories(in_dir: Path, out_dir: Path) -> None:
    in_res, out_res = in_dir.resolve(), out_dir.resolve()
    if out_res == in_res or in_res in out_res.parents:
        raise ValueError(
            f"output directory {out_dir} is the same as, or inside, the input directory "
            f"{in_dir}; choose a separate destination"
        )


def process_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    artifacts_dir: str | Path | None = None,
    grain: bool = True,
    all_files: bool = False,
    overwrite: bool = False,
    use_log: bool = True,
) -> BatchResult:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    _check_directories(in_dir, out_dir)

    every = list_images(in_dir)
    chosen = select_inputs(every, all_files=all_files)
    stems = [p.stem for p in chosen]
    ambiguous = {s for s in stems if stems.count(s) > 1}

    pipeline = Pipeline(Artifacts.resolve(artifacts_dir))
    result = BatchResult(skipped_graded=sum(1 for p in every if _is_graded(p)))
    log: dict[str, LogEntry] = {}
    if use_log:
        log, result.bad_log_lines = load_capture_log(in_dir)
    for src in chosen:
        dest = output_path(src, out_dir, disambiguate=src.stem in ambiguous)
        if dest.exists() and not overwrite:
            result.skipped_existing += 1
            continue
        entry = log.get(src.name)
        ev, rng = 0.0, None
        if entry is not None:
            result.from_log += 1
            result.ev_applied += entry.ev != 0.0
            result.doubles += entry.double
            ev = entry.ev
            if entry.seed is not None:
                rng = np.random.default_rng(entry.seed)
        rgb, _meta = load_rgb(src)
        graded, _info = pipeline.process(rgb, grain=grain, rng=rng, ev=ev)
        result.written.append(save_jpeg(graded, dest))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pifilm-process",
        description="Regrade a folder of images with the Parr LUT.",
    )
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--no-grain", action="store_true", help="skip film grain")
    parser.add_argument(
        "--all", action="store_true", help="process every image, not just originals"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    parser.add_argument(
        "--ignore-log", action="store_true",
        help=f"grade every frame at EV 0 with fresh grain, ignoring {CAPTURE_LOG}",
    )
    args = parser.parse_args(argv)

    if not args.in_dir.is_dir() or not list_images(args.in_dir):
        print(f"error: no images found in {args.in_dir}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    try:
        result = process_dir(
            args.in_dir,
            args.out_dir,
            args.artifacts,
            grain=not args.no_grain,
            all_files=args.all,
            overwrite=args.overwrite,
            use_log=not args.ignore_log,
        )
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    notes = []
    if result.from_log:
        notes.append(f"{result.from_log} from {CAPTURE_LOG} ({result.ev_applied} with EV)")
    if result.doubles:
        notes.append(f"{result.doubles} double-exposure frames graded singly")
    if result.bad_log_lines:
        notes.append(f"{result.bad_log_lines} unreadable {CAPTURE_LOG} lines ignored")
    if result.skipped_graded:
        notes.append(f"{result.skipped_graded} already-graded skipped")
    if result.skipped_existing:
        notes.append(f"{result.skipped_existing} existing outputs kept (use --overwrite)")
    suffix = f" ({', '.join(notes)})" if notes else ""
    print(
        f"Processed {len(result.written)} image(s) into {args.out_dir} "
        f"in {time.perf_counter() - t0:.1f}s{suffix}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
