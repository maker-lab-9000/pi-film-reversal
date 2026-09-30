"""``pifilm-train``: fit the Parr LUT from two folders of images.

The sequence (spec section 6):

1. ``dataset.build_corpus`` splits each corpus **by image** and samples the
   halves separately, so the evaluation later is genuinely held out.
2. ``transport.hue_weights`` reweights the target's hues toward the
   source's, reducing content bias. This is a heuristic; see the note in
   ``transport.py`` about what it does not guarantee.
3. ``transport.iterative_distribution_transfer`` gives every training source
   pixel a Parr partner. ``strength`` blends between identity and the
   full transport.
4. ``lutfit.fit_lut`` fits a smooth LUT to those pairs.
5. ``evaluate.evaluate`` measures the result on the held-out images with a
   paired evaluator, and ``check_gates`` turns the numbers into pass or fail.
   If either corpus was too small to hold images back, this falls back to
   training pixels, warns, and records ``held_out_eval: false``.
6. Everything is written into a staging directory and published atomically,
   so an interrupted run can never leave a new LUT beside old parameters.

A failing gate does not delete the artifact: you may want to inspect it. It
sets exit code 3 and names the gate, so a script cannot mistake it for
success.

``--proxy-source`` exists because the trainer cannot tell whether a folder of
photographs came from the U20CAM. Passing it records the fact so users of
the shipped artifact know to retrain.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import __version__
from ..artifacts import publish, write_artifact
from ..color import oklab_to_srgb
from ..grain import GrainParams
from ..highlight import protect_highlights
from ..imageio import list_images
from ..lut import LUT3D
from ..normalize import NormalizeParams
from .dataset import CorpusTooSmall, PixelPool, SampleConfig, build_corpus
from .evaluate import check_gates, evaluate
from .lutfit import enforce_grey_axis, enforce_monotone, fit_lut
from .report import write_report
from .transport import hue_weights, iterative_distribution_transfer

MIN_SOURCE_IMAGES = 30
MIN_TARGET_IMAGES = 200


@dataclass
class FitConfig:
    lut_size: int = 33
    iterations: int = 40
    hue_bins: int = 24
    chroma_floor: float = 0.03
    lambda_smooth: float = 1e-2
    lambda_identity: float = 1.0
    strength: float = 1.0
    seed: int = 0
    neutral_axis_cap: float = 0.005   # halved from 0.01: 0.01 left indoor whites visibly cool
    highlights: float = 0.0

    def __post_init__(self) -> None:
        if not 2 <= self.lut_size <= 65:
            raise ValueError(f"lut_size must be in 2..65, got {self.lut_size}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.hue_bins < 1:
            raise ValueError(f"hue_bins must be positive, got {self.hue_bins}")
        if not 0.0 <= self.chroma_floor < 0.5:
            raise ValueError(f"chroma_floor must be in [0, 0.5), got {self.chroma_floor}")
        if self.neutral_axis_cap < 0:
            raise ValueError(f"neutral_axis_cap must be non-negative, got {self.neutral_axis_cap}")
        if not math.isfinite(self.highlights) or not 0.0 <= self.highlights <= 1.0:
            raise ValueError(
                f"highlights must be finite and in [0, 1], got {self.highlights!r}"
            )
        for name in ("lambda_smooth", "lambda_identity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        # partner = src + strength * (moved - src): 1 is the learned look, below
        # blends toward identity, above EXTRAPOLATES the learned difference. The
        # first real fit had the right shape at about a third of the amplitude
        # people expect, so >1 is a legitimate knob. Capped at 2 because beyond
        # that the transport's out-of-gamut clipping dominates the result.
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError(f"strength must be in [0, 2], got {self.strength}")


@dataclass
class FitResult:
    lut: LUT3D
    transported_lab: np.ndarray
    target_weights: np.ndarray


def fit(
    source_pool: PixelPool,
    target_pool: PixelPool,
    cfg: FitConfig,
    progress: Callable[[str], None] | None = None,
) -> FitResult:
    say = progress or (lambda _m: None)
    rng = np.random.default_rng(cfg.seed)
    src_lab, tgt_lab = source_pool.lab, target_pool.lab

    say("reweighting target hues toward the source histogram")
    weights = hue_weights(src_lab, tgt_lab, cfg.hue_bins, cfg.chroma_floor)

    say(f"iterative distribution transfer, {cfg.iterations} rounds")
    moved = iterative_distribution_transfer(src_lab, tgt_lab, weights, cfg.iterations, rng)
    partner_lab = src_lab + cfg.strength * (moved - src_lab)

    say(f"fitting {cfg.lut_size}^3 LUT by regularised least squares")
    lut = fit_lut(
        source_pool.srgb,
        np.clip(oklab_to_srgb(partner_lab), 0.0, 1.0),
        n=cfg.lut_size,
        lambda_smooth=cfg.lambda_smooth,
        lambda_identity=cfg.lambda_identity,
        neutral_axis_cap=cfg.neutral_axis_cap,
    )
    if cfg.highlights > 0.0:
        say(f"protecting highlights at {cfg.highlights}")
        lut = enforce_monotone(
            enforce_grey_axis(enforce_monotone(protect_highlights(lut, cfg.highlights)))
        )
    return FitResult(lut, partner_lab.astype(np.float32), weights)


def _corpus_licences(corpus_dir: Path) -> dict | None:
    """What pifilm-fetch recorded about the licences of this corpus, if anything."""
    manifest = Path(corpus_dir) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        m = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    return {"policy": m.get("licence_policy"), "counts": m.get("licences")}


def _code_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dependency_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "Pillow", "opencv-python"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def train(
    source_dir: str | Path,
    target_dir: str | Path,
    out_dir: str | Path,
    cfg: FitConfig,
    sample_cfg: SampleConfig,
    grain: GrainParams | None,
    proxy_source: bool = False,
    allow_small: bool = False,
    target_levels: bool = False,
    source_levels: bool = True,
    target_median: float | None = None,
    source_white_balance: bool = True,
    source_lift_highlight_ref: float | None = None,
    source_gamma_min: float | None = None,
    source_gamma_max: float | None = None,
    command: str = "",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, list]:
    say = progress or (lambda _m: None)
    source_dir, target_dir, out_dir = Path(source_dir), Path(target_dir), Path(out_dir)

    source_normalize = NormalizeParams(
        white_balance=source_white_balance,
        levels=source_levels,
        levels_lift_highlight_ref=source_lift_highlight_ref,
        **_gamma_limits(source_gamma_min, source_gamma_max),
    )
    target_normalize = NormalizeParams(
        white_balance=False, levels=target_levels, levels_target_median=target_median
    )
    source = build_corpus(source_dir, source_normalize, sample_cfg, MIN_SOURCE_IMAGES,
                          "source", allow_small, say)
    target = build_corpus(target_dir, target_normalize, sample_cfg, MIN_TARGET_IMAGES,
                          "target", allow_small, say)

    t0 = time.perf_counter()
    result = fit(source.train_pool, target.train_pool, cfg, say)
    fit_seconds = time.perf_counter() - t0

    # The evaluation is only held out if BOTH sides kept images back. A corpus
    # under 1/val_fraction images yields an empty validation split, and falling
    # back to training pixels while still calling the result "held-out" would
    # report memorisation as generalisation.
    source_held_out = len(source.val_pool.srgb) > 0
    target_held_out = len(target.val_pool.srgb) > 0
    held_out_eval = source_held_out and target_held_out
    if held_out_eval:
        say("evaluating on held-out images")
    else:
        pairs = (("source", source_held_out), ("target", target_held_out))
        short = [name for name, ok in pairs if not ok]
        say(f"WARNING: no held-out images for the {' and '.join(short)} corpus; "
            "evaluating on training pixels, which measures memorisation, not generalisation")
    val_weights = hue_weights(
        source.val_pool.lab, target.val_pool.lab, cfg.hue_bins, cfg.chroma_floor
    ) if target_held_out else None
    metrics = evaluate(
        lut=result.lut,
        val_src=source.val_pool if source_held_out else source.train_pool,
        val_tgt=target.val_pool if target_held_out else target.train_pool,
        val_weights=val_weights,
        train_src=source.train_pool,
        train_tgt=target.train_pool,
        train_weights=result.target_weights,
        transported_lab=result.transported_lab,
        n_bins=cfg.hue_bins,
        chroma_floor=cfg.chroma_floor,
        seed=cfg.seed,
    )
    metrics["held_out_eval"] = held_out_eval
    gates = check_gates(metrics)

    training = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_revision": _code_revision(),
        "package_version": __version__,
        "dependency_versions": _dependency_versions(),
        "command": command,
        "target": {
            "dir": str(target_dir),
            "n_images": len(target.train_paths) + len(target.val_paths),
            "corpus_sha1": target.corpus_sha1,
            "licences": _corpus_licences(target_dir),
            "normalize": target_normalize.to_dict(),
            "n_pixels": int(len(target.train_pool.srgb)),
            "clamp_rate": target.train_pool.clamp_rate,
            "profiles": target.train_pool.profiles,
        },
        "source": {
            "dir": str(source_dir),
            "n_images": len(source.train_paths) + len(source.val_paths),
            "corpus_sha1": source.corpus_sha1,
            "licences": _corpus_licences(source_dir),
            "n_pixels": int(len(source.train_pool.srgb)),
            "proxy": proxy_source,
            "clamp_rate": source.train_pool.clamp_rate,
            "profiles": source.train_pool.profiles,
        },
        "split": {
            "val_fraction": sample_cfg.val_fraction,
            "n_source_val_images": len(source.val_paths),
            "n_target_val_images": len(target.val_paths),
            "seed": sample_cfg.seed,
        },
        "fit": {**asdict(cfg), "fit_seconds": round(fit_seconds, 1)},
        "sample": asdict(sample_cfg),
        "metrics": {k: v for k, v in metrics.items() if k != "hue_bins"},
    }

    # publish() creates out_dir's parent, but mkdtemp needs it to exist first.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pifilm-staging-", dir=out_dir.parent))
    try:
        write_artifact(staging, result.lut, source_normalize, grain or GrainParams(), training)
        say("writing report")
        write_report(staging / "report", result.lut, metrics, gates, source, target,
                     source_normalize, target_normalize, sample_cfg)
        publish(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    say(f"published {out_dir}; report in {out_dir / 'report'}")
    return metrics, gates


def _gamma_limits(gamma_min: float | None, gamma_max: float | None) -> dict:
    """NormalizeParams overrides for the levels gamma clamp; None keeps its default.

    The clamp decides how far normalisation may move a frame the median rule
    misjudges. The 0.5 floor lifted a deliberately dark street to slate grey,
    and the 2.0 ceiling crushed a face against a white wall to a tenth of its
    brightness (IMX477 frames 160707 and 155821, 2026-09-30). The LUT is fitted
    on normalised input, so a changed clamp needs a retrain, not a params edit.
    """
    limits = {}
    if gamma_min is not None:
        limits["levels_gamma_min"] = gamma_min
    if gamma_max is not None:
        limits["levels_gamma_max"] = gamma_max
    return limits


def build_parser() -> argparse.ArgumentParser:
    """Split out so the defaults can be asserted against FitConfig directly."""
    parser = argparse.ArgumentParser(prog="pifilm-train", description="Fit the Parr LUT.")
    parser.add_argument("--source", type=Path, required=True, help="folder of camera photos")
    parser.add_argument("--target", type=Path, default=Path("data/references"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    # Defaults come from FitConfig so there is ONE source of truth. They were
    # duplicated here, which silently pinned the CLI to the old values when the
    # dataclass defaults changed -- every trained artifact used them.
    fd = FitConfig()
    parser.add_argument("--strength", type=float, default=fd.strength,
                        help="0 = no change, 1 = the learned look, up to 2 exaggerates it")
    parser.add_argument("--lut-size", type=int, default=fd.lut_size)
    parser.add_argument("--iterations", type=int, default=fd.iterations)
    parser.add_argument("--hue-bins", type=int, default=fd.hue_bins)
    parser.add_argument("--lambda-smooth", type=float, default=fd.lambda_smooth)
    parser.add_argument("--lambda-identity", type=float, default=fd.lambda_identity)
    parser.add_argument("--neutral-axis-cap", type=float, default=fd.neutral_axis_cap,
                        help="max Oklab chroma the LUT may give a neutral input; 0 = fully neutral")
    parser.add_argument(
        "--highlights", type=float, default=fd.highlights,
        help="protect coloured highlights: 0 = off, up to 1 removes the top tone lift",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grain-strength", type=float, default=GrainParams().strength)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--pixels-per-image", type=int, default=3000)
    parser.add_argument("--max-pixels", type=int, default=400_000)
    parser.add_argument("--no-source-levels", dest="source_levels", action="store_false",
                        help="normalise camera frames with a gain instead of levels (A/B only)")
    parser.add_argument(
        "--no-source-white-balance", dest="source_white_balance", action="store_false",
        help="do not grey-world the source frames; use for cameras whose ISP has already "
             "white-balanced them (Picamera2/IMX708). Recorded in the artifact and applied "
             "identically by the Pi",
    )
    parser.add_argument(
        "--source-lift-highlight-ref", type=float, default=None, metavar="FRACTION",
        help="clipping-aware lift: fraction of pixels at the ceiling at which the levels "
             "lift is fully suppressed (e.g. 0.05); default off. Recorded in the artifact",
    )
    parser.add_argument(
        "--source-gamma-min", type=float, default=None, metavar="GAMMA",
        help="strongest levels lift allowed on a camera frame (default 0.5). Higher keeps "
             "a deliberately dark scene dark. Recorded in the artifact",
    )
    parser.add_argument(
        "--source-gamma-max", type=float, default=None, metavar="GAMMA",
        help="strongest levels darkening allowed on a camera frame (default 2.0). Lower "
             "keeps a subject against a bright wall from being crushed. Recorded in the "
             "artifact",
    )
    parser.add_argument("--target-levels", action=argparse.BooleanOptionalAction, default=False,
                        help="stretch target levels; default uses exposure matching only")
    parser.add_argument("--target-median", type=float, default=None,
                        help="median the target scans are gamma-normalised to; default = the "
                             "exposure target. Lower keeps the film's density")
    parser.add_argument("--proxy-source", action="store_true",
                        help="mark the source as stand-in photos, not U20CAM shots")
    parser.add_argument("--allow-small", action="store_true",
                        help="proceed with a corpus below the recommended minimum")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for label, path in (("source", args.source), ("target", args.target)):
        if not path.is_dir() or not list_images(path):
            print(f"error: {label} folder {path} does not exist or has no images", file=sys.stderr)
            return 1
    if args.source.resolve() == args.target.resolve():
        print("error: source and target are the same folder", file=sys.stderr)
        return 1

    try:
        cfg = FitConfig(
            lut_size=args.lut_size, iterations=args.iterations, hue_bins=args.hue_bins,
            lambda_smooth=args.lambda_smooth, lambda_identity=args.lambda_identity,
            neutral_axis_cap=args.neutral_axis_cap,
            strength=args.strength, seed=args.seed, highlights=args.highlights,
        )
        sample_cfg = SampleConfig(
            max_side=args.max_side, pixels_per_image=args.pixels_per_image,
            max_pixels=args.max_pixels, val_fraction=args.val_fraction, seed=args.seed,
        )
        grain = GrainParams(strength=args.grain_strength)
        # Validated here and thrown away: train() builds the real one, but only after
        # reading both corpora, and a rejected --source-lift-highlight-ref should be a
        # one-line usage error like every other bad flag. __post_init__ is the single
        # copy of the range checks, so this is a fail-fast call, not duplicated logic.
        NormalizeParams(
            white_balance=args.source_white_balance,
            levels=args.source_levels,
            levels_lift_highlight_ref=args.source_lift_highlight_ref,
            **_gamma_limits(args.source_gamma_min, args.source_gamma_max),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        metrics, gates = train(
            args.source, args.target, args.out, cfg, sample_cfg, grain,
            proxy_source=args.proxy_source, allow_small=args.allow_small,
            target_levels=args.target_levels, target_median=args.target_median,
            source_levels=args.source_levels,
            source_white_balance=args.source_white_balance,
            source_lift_highlight_ref=args.source_lift_highlight_ref,
            source_gamma_min=args.source_gamma_min, source_gamma_max=args.source_gamma_max,
            command=" ".join(["pifilm-train", *(argv or sys.argv[1:])]), progress=print,
        )
    # Deliberately narrow: a ValueError from inside the fit (a diverging LUT, say) is a
    # bug worth a traceback, not a usage error. Bad flag values are already rejected by
    # the config block above, so nothing user-facing reaches here but a short corpus.
    except CorpusTooSmall as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    label = "held-out" if metrics["held_out_eval"] else "TRAINING (not held out)"
    print(
        f"{label} distance to Parr: {metrics['swd_before']:.5f} -> "
        f"{metrics['swd_after']:.5f} (seed spread {metrics['swd_seed_spread']:.5f})"
    )
    failed = [g for g in gates if not g.passed]
    for gate in gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    if failed:
        print(
            "error: artifact written but "
            + ", ".join(g.name for g in failed)
            + " did not pass; see the report before using it",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
