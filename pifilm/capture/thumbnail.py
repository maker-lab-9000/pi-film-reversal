"""Small, bounded JPEG previews for the remote capture protocol."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from ..display.ui import PROCESSING_BARS, PROCESSING_DIM

THUMBNAIL_SIZE = (240, 135)
MAX_JPEG_BYTES = 64 * 1024


class ThumbnailError(ValueError):
    """The saved grade cannot safely be represented by the remote thumbnail."""


def fitted_jpeg(
    source: Path,
    *,
    size: tuple[int, int] = THUMBNAIL_SIZE,
    max_bytes: int = MAX_JPEG_BYTES,
) -> bytes:
    """Return an aspect-preserving RGB JPEG on a black ``size`` canvas.

    ``source`` is deliberately a trusted capture result path supplied by the
    server; this function never interprets an HTTP path or filename.
    """
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ThumbnailError("saved graded image is invalid") from exc

    width, height = size
    if width <= 0 or height <= 0 or max_bytes <= 0:
        raise ThumbnailError("invalid thumbnail limits")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    canvas.paste(image, (left, top))

    for quality in range(90, 19, -5):
        data = io.BytesIO()
        canvas.save(data, format="JPEG", quality=quality, optimize=True)
        encoded = data.getvalue()
        if len(encoded) <= max_bytes:
            return encoded
    raise ThumbnailError(f"thumbnail exceeds {max_bytes // 1024} KiB transfer limit")


def render_exposure_card(
    index: int, of: int, *, size: tuple[int, int] = THUMBNAIL_SIZE,
) -> bytes:
    """What the Stick shows after exposure ``index`` of a double exposure.

    Exposure 1 has no photo yet: the film has not been developed. The Stick's
    protocol expects an image for every finished capture, so it gets the same dimmed
    colour bars the LCD shows while working, labelled with the count.
    """
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(PROCESSING_BARS):
        left, right = i * width // 7, (i + 1) * width // 7
        dimmed = tuple(round(c * PROCESSING_DIM) for c in colour)
        draw.rectangle((left, 0, right - 1, height * 3 // 4 - 1), fill=dimmed)
    draw.text((width // 2, height * 7 // 8), f"Exposure {index}/{of}",
              fill=(255, 255, 255), font=ImageFont.load_default(size=16), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
