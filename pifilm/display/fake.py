"""Hardware-free stand-ins for ``--display fake``: frames go to a PNG, no touch."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class FileDisplay:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def show(self, image: Image.Image) -> None:
        image.save(self.path)

    def close(self) -> None:
        return None


class NoTouch:
    def read(self) -> list:
        return []

    def close(self) -> None:
        return None
