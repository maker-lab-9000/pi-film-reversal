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
  artifact then reproduces the Pi's file, up to the re-encoded original. A USB
  camera's original is the camera's own JPEG, which is what the Pi graded, so
  that regrade is exact. The Pi camera's original is a quality-95 JPEG of the
  frame, and the Pi graded the frame itself, so the regrade starts from a
  slightly different picture: about 1 to 2 levels on average, more where the
  look's LUT is steep (measured 2026-10-14 on sixteen IMX477 photographs, GPU
  graded: mean 0.7 to 2.1, at most 18 to 27).
  Double-exposure frames are regraded singly, with their own EV and fresh
  grain, because their logged seed belongs to the composite.
* **Halation switched off.** The viewfinder's menu can leave a look's halation
  out of a photograph, and its record then says ``"halation": "off"`` where
  ``halation_sha1`` would be. Such a frame is regraded without halation too, or
  the regrade would add a glow the Pi's file never had. ``--ignore-log`` grades
  with the look's halation, as it ignores the rest of the log.
* **A file the Pi graded on its GPU.** The regrade here is always the CPU's.
  A record whose ``pipeline_backend`` starts with ``gpu`` was graded by the
  one-pass shader, which does not round between its stages as the CPU does,
  so the regraded file is close to the Pi's and not the same bytes (see
  ``pifilm.pipeline``). The tool says so once per run. Records graded on the
  CPU, and records from before the field existed, reproduce as closely as the
  original allows (exactly for a USB camera, above).
* **A multi-frame photograph.** Its record's ``stack`` lists the burst's
  files, the reference first (``<time>_original.jpg``, then
  ``<time>_frame2.jpg`` and on), and the Pi graded their merge. The regrade
  merges the same files with ``pifilm.stack.merge(frames, reference=0)``, as the
  Pi's grade worker does, then grades the merge with the logged EV and seed, so
  it reproduces the Pi's file as closely as the frames on disk allow (exactly
  for a grade the Pi recovered from those JPEGs). A listed frame missing from
  the folder, unreadable, or of another size than the reference leaves the
  reference to be graded alone (that photograph only; the run goes on), and
  the run says how many. ``--ignore-log`` grades the reference alone. The ``_frameN`` files are
  never inputs themselves, whatever ``--all`` says: each is part of a
  photograph, not one.
* **Clobbering.** ``a.jpg`` and ``a.png`` would produce the same output
  name, and re-running would silently overwrite. Same-stem inputs get their
  extension folded into the output name, existing outputs are skipped unless
  ``--overwrite``, and an output directory equal to or inside the input is
  refused outright.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import Artifacts, ArtifactsError
from ..imageio import list_images, load_rgb, save_jpeg
from ..pipeline import Pipeline
from ..stack import merge

GRADED_SUFFIXES = ("_graded",)
SOURCE_SUFFIXES = ("_original", "_ungraded")
# A multi-frame burst's other frames (``<time>_frame2`` and on): never an input.
BURST_FRAME = re.compile(r"_frame\d+$")


CAPTURE_LOG = "captures.jsonl"


@dataclass
class BatchResult:
    written: list[Path] = field(default_factory=list)
    skipped_graded: int = 0
    skipped_existing: int = 0
    from_log: int = 0
    ev_applied: int = 0
    doubles: int = 0
    gpu_graded: int = 0  # regraded here on the CPU, graded by the Pi on its GPU
    halation_off: int = 0  # regraded without the look's halation, as the Pi graded them
    stacks: int = 0  # multi-frame photographs regraded from the merge of their frames
    # Multi-frame photographs graded from the reference alone: a listed frame
    # missing, unreadable or of another size.
    stack_frames_missing: int = 0
    bad_log_lines: int = 0


@dataclass(frozen=True)
class LogEntry:
    ev: float
    seed: int | None
    double: bool = False
    gpu: bool = False  # the Pi graded this record on its GPU
    halation: bool = True  # False: the record says the look's halation was left out
    # A multi-frame record's file names, the reference first; None for one frame.
    frames: tuple[str, ...] | None = None


def _stack_frames(record: dict) -> tuple[str, ...] | None:
    stack = record.get("stack")
    if stack is None:
        return None
    frames = stack.get("frames") if isinstance(stack, dict) else None
    if (not isinstance(frames, list) or not frames
            or not all(isinstance(name, str) and name for name in frames)):
        raise ValueError(f"bad stack: {stack!r}")
    # Names only, in the record's own folder: never a path out of it.
    if any(Path(name).name != name for name in frames):
        raise ValueError(f"bad stack frame names: {frames!r}")
    return tuple(frames)


def _log_entry(record: dict) -> LogEntry:
    halation = record.get("halation") != "off"
    double = record.get("double")
    if isinstance(double, dict):
        evs = double.get("ev_comp")
        ev = evs[1] if double.get("index") == 2 and isinstance(evs, list) else record.get(
            "ev_comp")
        # Not marked as a GPU record: a double's frame is regraded singly, which
        # is a different picture from the Pi's composite whatever graded that.
        return LogEntry(float(ev or 0.0), None, double=True, halation=halation)
    seed = record.get("grain_seed")
    backend = record.get("pipeline_backend")
    return LogEntry(float(record.get("ev_comp") or 0.0),
                    int(seed) if seed is not None else None,
                    gpu=isinstance(backend, str) and backend.startswith("gpu"),
                    halation=halation, frames=_stack_frames(record))


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


def _is_burst_frame(path: Path) -> bool:
    return BURST_FRAME.search(path.stem) is not None


def _is_source(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in SOURCE_SUFFIXES)


def select_inputs(paths: Sequence[Path], all_files: bool = False) -> list[Path]:
    """Graded outputs and a burst's other frames are never inputs; capture
    folders default to their originals."""
    candidates = [p for p in paths if not _is_graded(p) and not _is_burst_frame(p)]
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
        ev, rng, halation = 0.0, None, True
        if entry is not None:
            result.from_log += 1
            result.ev_applied += entry.ev != 0.0
            result.doubles += entry.double
            result.gpu_graded += entry.gpu
            result.halation_off += not entry.halation
            ev, halation = entry.ev, entry.halation
            if entry.seed is not None:
                rng = np.random.default_rng(entry.seed)
        rgb, _meta = load_rgb(src)
        if entry is not None and entry.frames is not None and len(entry.frames) > 1:
            others = [in_dir / name for name in entry.frames[1:]]
            merged = None
            if entry.frames[0] == src.name and all(p.is_file() for p in others):
                # As the Pi's grade worker: the reference first, merged onto it. A
                # frame that cannot be read, or of another size (merge's
                # ValueError), costs this photograph its merge, not the run.
                try:
                    merged, _info = merge([rgb] + [load_rgb(p)[0] for p in others],
                                          reference=0)
                except (OSError, ValueError):
                    merged = None
            if merged is not None:
                rgb = merged
                result.stacks += 1
            else:
                result.stack_frames_missing += 1
        graded, _info = pipeline.process(rgb, grain=grain, rng=rng, ev=ev, halation=halation)
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
        help=f"grade every frame at EV 0 with fresh grain and the look's halation, "
             f"and a multi-frame photograph from its reference alone, ignoring {CAPTURE_LOG}",
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
    if result.stacks:
        notes.append(f"{result.stacks} multi-frame merged")
    if result.stack_frames_missing:
        notes.append(f"{result.stack_frames_missing} multi-frame with frames missing or "
                     "unusable, reference graded alone")
    if result.halation_off:
        notes.append(f"{result.halation_off} without halation, as shot")
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
    if result.gpu_graded:
        print(
            f"Note: the Pi graded {result.gpu_graded} of these on its GPU, and this regrade "
            "is the CPU's. Those files match the Pi's within a few levels on average (more "
            "on strongly saturated colours), not byte for byte."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
