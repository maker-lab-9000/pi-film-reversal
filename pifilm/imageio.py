"""Image file I/O, and the one place colour management happens.

Every pixel that enters this project comes through ``load_rgb``, which does
two things no other module should have to remember:

**EXIF orientation.** A camera that was held sideways records the rotation
as a tag rather than rotating the pixels. The trainer crops a fixed 6% from
each edge, so an unoriented image gets the wrong edges cropped, and a
portrait frame would be sampled as landscape. ``ImageOps.exif_transpose``
resolves the tag into real pixel order.

**ICC profiles.** The whole project is a colour measurement, so treating an
Adobe RGB or ProPhoto scan as if it were sRGB would bake a systematic error
into the learned look - the more so because the target corpus is archival
scans whose colour management is part of what we are matching. When a
profile is embedded, the image is converted to sRGB with ``ImageCms``. When
none is present, sRGB is assumed, which is the web convention and correct
for Commons JPEGs. When one is present but unreadable, sRGB is assumed and
the failure is recorded rather than swallowed, so the report can count it.

``ImageMeta`` travels with the pixels so the trainer can publish profile
statistics: a corpus that is secretly half Adobe RGB should be visible, not
silently averaged in.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_EXIF_ORIENTATION_TAG = 274


@dataclass
class ImageMeta:
    profile: str
    oriented: bool
    profile_error: str | None
    width: int
    height: int


@lru_cache(maxsize=1)
def srgb_profile():
    """The sRGB profile used as the working space, built once."""
    return ImageCms.createProfile("sRGB")


def _describe(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileDescription(profile).strip() or "unnamed profile"
    except Exception:  # noqa: BLE001 - a profile can be readable but have no description
        return "unnamed profile"


def load_rgb(path: str | Path, *, colour_manage: bool = True) -> tuple[np.ndarray, ImageMeta]:
    """Load an image as sRGB RGB uint8, applying EXIF orientation and ICC conversion."""
    path = Path(path)
    with Image.open(path) as im:
        exif = im.getexif()
        oriented = bool(exif.get(_EXIF_ORIENTATION_TAG, 1) not in (1, None))
        im = ImageOps.exif_transpose(im)

        raw_profile = im.info.get("icc_profile")
        profile_name = "sRGB (assumed)"
        profile_error: str | None = None

        if raw_profile and colour_manage:
            # Built outside the try on purpose: if the working-space profile
            # itself cannot be created, that is an environment fault, not a bad
            # source profile, and it should surface rather than be reported as
            # "invalid" against the image.
            destination = srgb_profile()
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(raw_profile))
                profile_name = _describe(src)
                # Do not convert the mode first: littlecms needs the image in the
                # colour space the profile describes. Pre-converting a LAB or CMYK
                # image to RGB makes the transform unbuildable, and the fallback
                # below would then silently mislabel a perfectly good profile.
                im = ImageCms.profileToProfile(
                    im, src, destination, renderingIntent=0, outputMode="RGB"
                )
            except Exception as exc:  # noqa: BLE001 - any malformed profile falls back to sRGB
                profile_name = "invalid"
                profile_error = f"{type(exc).__name__}: {exc}"
        elif raw_profile:
            profile_name = "sRGB (assumed, colour management off)"

        # np.asarray on a PIL image is read-only. Callers legitimately draw on
        # what they get back (overlays on a preview frame), and the writability
        # of a public return value must not depend on which branch produced it.
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    return rgb, ImageMeta(
        profile=profile_name,
        oriented=oriented,
        profile_error=profile_error,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )


def save_jpeg(
    rgb_u8: np.ndarray,
    path: str | Path,
    quality: int = 95,
    embed_srgb: bool = True,
    subsampling: int = 0,
) -> Path:
    """Write an sRGB JPEG. ``subsampling=0`` means 4:4:4, full chroma resolution.

    Pillow's default is 4:2:0, which halves chroma resolution in both axes.
    On a natural frame that costs little on average (mean Oklab dE 0.0126
    against 0.0102) but a great deal at saturated edges: the 99th-percentile
    error is 0.032 dE, which exceeds the fitted LUT's own 0.025 accuracy
    floor. In other words, on the pixels where Parr's character
    actually lives, the file format would introduce more error than the
    learned grade itself carries. The cost is about twice the bytes — 2.4 MB
    against 4.9 MB for a 1080p frame — which is the right trade for a project
    whose entire purpose is colour, and for files that double as the archival
    record. Pass ``subsampling=2`` for 4:2:0 if storage matters more.

    The image is ``(H, W, 3)``, or ``(H, W, 4)`` with the picture in bytes 0 to 2
    and a fourth byte that is ignored. That is what the GPU grade reads back
    (it cannot read three bytes a pixel), and the encoder takes it as it is:
    repacking 12 MP to three bytes cost more than it saved (2026-10-13). The
    file is byte for byte the one the same pixels give as three bytes.
    """
    rgb_u8 = np.asarray(rgb_u8)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] not in (3, 4) or rgb_u8.dtype != np.uint8:
        raise ValueError(
            f"save_jpeg expects a uint8 array of shape (H, W, 3) or (H, W, 4); "
            f"got dtype {rgb_u8.dtype} shape {rgb_u8.shape}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_u8.shape[2] == 4:
        # Pillow reads the caller's buffer in place (no copy when it is
        # contiguous, as a read-back is); "RGBX" is RGB with a padding byte.
        height, width = rgb_u8.shape[:2]
        image = Image.frombuffer(
            "RGBX", (width, height), np.ascontiguousarray(rgb_u8), "raw", "RGBX", 0, 1
        )
    else:
        image = Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB")
    kwargs = {}
    if embed_srgb:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    image.save(path, "JPEG", quality=quality, subsampling=subsampling, **kwargs)
    return path


def list_images(dir_path: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
