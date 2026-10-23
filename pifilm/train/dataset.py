"""Turn folders of images into the pixel pools the fitter and evaluator use.

Both corpora go through the same steps:

1. Crop ``crop_frac`` from every edge. Slide scans carry film rebate, mount
   shadow or scanner bed; camera frames can have vignetted corners.
2. Downscale so the long side is ``max_side``. Colour statistics do not need
   full resolution.
3. Normalise with the same code the Pi runs. Sources get white balance and
   a median-to-target exposure gain; targets are given
   ``NormalizeParams(white_balance=False, levels=False)`` by default: exposure
   matching without white balance or black/white-point stretching. The caller
   may explicitly enable target levels stretching for flat scans.
4. Sample pixels whose Oklab lightness is inside ``(l_min, l_max)``.
   Near-black pixels are borders and crushed shadows, near-white are blown
   highlights and scanner glare; neither says anything about how the film
   renders colour.

Why the split happens here, before sampling
-------------------------------------------
Metrics computed on the pixels a LUT was fitted to measure how well the fit
memorised its training data, not whether the look generalises. So each
corpus is split **by image** first, and only then sampled. Splitting after
sampling would be worse than useless: pixels from the same photograph are
highly correlated, so a "held-out" pixel drawn from a training image leaks
almost everything about its neighbours, and the reported improvement would
be inflated in a way no seed average would reveal.

Weighting images
----------------
A ``weights.json`` beside the images (``{"file.jpg": 3}``) takes that many
times ``pixels_per_image`` from the files it names; every other image counts
once. It exists for a reference set where a few images are the look itself
(one's own scans of the film) and the rest are a broader family. Copying
those files instead would put copies on both sides of the split by image
and leak them into the held-out evaluation. The weights are part of the
corpus identity, so changing them changes ``corpus_sha1``.

Diagnostics travel with the pool
--------------------------------
``PixelPool`` carries the white balance and exposure gains applied to each
image, how often a gain hit its clamp, and which ICC profiles were seen. A
corpus where normalisation is clamping constantly, or which is secretly half
Adobe RGB, produces a misleading fit; the report publishes these so it is
visible rather than silent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._cv2 import require_cv2
from ..color import srgb_to_oklab
from ..imageio import list_images, load_rgb
from ..normalize import Gains, NormalizeParams, normalize_float

cv2 = require_cv2()


class CorpusTooSmall(ValueError):
    """A corpus has too few images for a statistically meaningful fit."""


WEIGHTS_FILE = "weights.json"
MAX_WEIGHT = 10.0


def load_weights(dir_path: str | Path) -> dict[str, float]:
    """Read ``weights.json`` from an image folder; ``{}`` when there is none.

    Every name must be an image in the folder and every weight a number in
    (0, MAX_WEIGHT]: a typo would otherwise silently weight nothing.
    """
    path = Path(dir_path) / WEIGHTS_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of file name to weight")
    names = {p.name for p in list_images(dir_path)}
    weights: dict[str, float] = {}
    for name, w in data.items():
        if name not in names:
            raise ValueError(f"{path}: {name!r} is not in {dir_path}")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"{path}: the weight of {name!r} must be a number")
        if not w > 0:
            raise ValueError(f"{path}: the weight of {name!r} must be positive")
        if w > MAX_WEIGHT:
            raise ValueError(f"{path}: the weight of {name!r} must be at most {MAX_WEIGHT:g}")
        weights[name] = float(w)
    return weights


@dataclass
class SampleConfig:
    crop_frac: float = 0.06
    max_side: int = 512
    pixels_per_image: int = 3000
    l_min: float = 0.02
    l_max: float = 0.98
    max_pixels: int = 400_000
    val_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.crop_frac < 0.4:
            raise ValueError(f"crop_frac must be in [0, 0.4), got {self.crop_frac}")
        if self.max_side < 16:
            raise ValueError(f"max_side must be at least 16, got {self.max_side}")
        if self.pixels_per_image < 1:
            raise ValueError(f"pixels_per_image must be positive, got {self.pixels_per_image}")
        if self.max_pixels < 1:
            raise ValueError(f"max_pixels must be positive, got {self.max_pixels}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if not 0.0 <= self.l_min < self.l_max <= 1.0:
            raise ValueError(f"l_min ({self.l_min}) must be below l_max ({self.l_max})")


@dataclass
class PixelPool:
    srgb: np.ndarray
    n_images: int
    clamp_rate: float = 0.0
    wb_gains: list = field(default_factory=list)
    exposure_gains: list = field(default_factory=list)
    profiles: dict = field(default_factory=dict)
    _lab: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            self._lab = srgb_to_oklab(self.srgb)
        return self._lab


@dataclass
class CorpusSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    train_pool: PixelPool
    val_pool: PixelPool
    corpus_sha1: str
    weights: dict[str, float] = field(default_factory=dict)


def crop_and_resize(rgb_u8: np.ndarray, crop_frac: float, max_side: int) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    dy, dx = int(round(h * crop_frac)), int(round(w * crop_frac))
    cropped = rgb_u8[dy : h - dy or None, dx : w - dx or None]
    ch, cw = cropped.shape[:2]
    scale = max_side / max(ch, cw)
    if scale >= 1.0:
        return np.ascontiguousarray(cropped)
    size = (max(1, int(round(cw * scale))), max(1, int(round(ch * scale))))
    return cv2.resize(np.ascontiguousarray(cropped), size, interpolation=cv2.INTER_AREA)


def prepare_image(
    rgb_u8: np.ndarray, normalize_params: NormalizeParams, cfg: SampleConfig
) -> tuple[np.ndarray, Gains]:
    small = crop_and_resize(rgb_u8, cfg.crop_frac, cfg.max_side).astype(np.float32) / 255.0
    return normalize_float(small, normalize_params)


def sample_pixels(
    rgb: np.ndarray, n: int, l_min: float, l_max: float, rng: np.random.Generator
) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    lightness = srgb_to_oklab(flat)[:, 0]
    keep = np.flatnonzero((lightness > l_min) & (lightness < l_max))
    if len(keep) > n:
        keep = rng.choice(keep, n, replace=False)
    return flat[keep]


def split_paths(paths: Sequence, val_fraction: float, seed: int) -> tuple[list, list]:
    """Split by image, deterministically. Validation gets ``floor(n * fraction)`` images."""
    ordered = list(paths)
    n_val = int(math.floor(len(ordered) * val_fraction))
    if n_val == 0:
        return ordered, []
    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(ordered), n_val, replace=False).tolist())
    train = [p for i, p in enumerate(ordered) if i not in val_idx]
    val = [p for i, p in enumerate(ordered) if i in val_idx]
    return train, val


def build_pool(
    paths: Sequence[Path],
    normalize_params: NormalizeParams,
    cfg: SampleConfig,
    progress: Callable[[str], None] | None = None,
    weights: dict[str, float] | None = None,
) -> PixelPool:
    rng = np.random.default_rng(cfg.seed)
    chunks: list[np.ndarray] = []
    wb_gains: list[list[float]] = []
    exposure_gains: list[float] = []
    profiles: dict[str, int] = {}
    clamped = 0
    for i, path in enumerate(paths, start=1):
        rgb, meta = load_rgb(path)
        profiles[meta.profile] = profiles.get(meta.profile, 0) + 1
        prepared, gains = prepare_image(rgb, normalize_params, cfg)
        wb_gains.append([round(float(g), 4) for g in gains.wb])
        exposure_gains.append(round(float(gains.exposure), 4))
        clamped += int(any(gains.clamped.values()))
        n = int(round(cfg.pixels_per_image * (weights or {}).get(Path(path).name, 1.0)))
        chunks.append(sample_pixels(prepared, n, cfg.l_min, cfg.l_max, rng))
        if progress and i % 100 == 0:
            progress(f"  {i}/{len(paths)} images sampled")

    pixels = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), np.float32)
    if len(pixels) > cfg.max_pixels:
        pixels = pixels[rng.choice(len(pixels), cfg.max_pixels, replace=False)]
    return PixelPool(
        srgb=np.ascontiguousarray(pixels, dtype=np.float32),
        n_images=len(paths),
        clamp_rate=round(clamped / max(len(paths), 1), 4),
        wb_gains=wb_gains,
        exposure_gains=exposure_gains,
        profiles=profiles,
    )


def corpus_sha1(paths: Sequence[Path], weights: dict[str, float] | None = None) -> str:
    """Hash the bytes of every file, so equal names with different content differ.

    Weights, when there are any, are hashed too; without them the value is
    what it always was, so older artifacts keep their identity.
    """
    h = hashlib.sha1()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode())
        h.update(hashlib.sha1(p.read_bytes()).digest())
    if weights:
        h.update(json.dumps(weights, sort_keys=True).encode())
    return h.hexdigest()


def build_corpus(
    dir_or_paths: str | Path | Sequence[Path],
    normalize_params: NormalizeParams,
    cfg: SampleConfig,
    minimum: int,
    label: str,
    allow_small: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CorpusSplit:
    is_dir = isinstance(dir_or_paths, (str, Path))
    paths = list_images(dir_or_paths) if is_dir else [Path(p) for p in dir_or_paths]
    weights = load_weights(dir_or_paths) if is_dir else {}
    if len(paths) < minimum and not allow_small:
        raise CorpusTooSmall(
            f"{label} corpus has {len(paths)} images, fewer than the {minimum} needed for a "
            f"meaningful fit. Add more images, or pass --allow-small to proceed anyway."
        )
    train_paths, val_paths = split_paths(paths, cfg.val_fraction, cfg.seed)
    if progress:
        progress(f"{label}: {len(train_paths)} train, {len(val_paths)} validation images")
        if weights:
            progress(f"{label}: {len(weights)} images weighted by {WEIGHTS_FILE}")
    return CorpusSplit(
        train_paths=train_paths,
        val_paths=val_paths,
        train_pool=build_pool(train_paths, normalize_params, cfg, progress, weights),
        val_pool=build_pool(val_paths, normalize_params, cfg, progress, weights),
        corpus_sha1=corpus_sha1(paths, weights),
        weights=weights,
    )
