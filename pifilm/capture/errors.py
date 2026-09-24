"""Capture exceptions, in a module with no third-party imports.

``CameraError`` lives here rather than in ``camera.py`` because that module
runs ``require_cv2()`` at import time. Anything that only needs to *catch* a
camera failure — the LCD viewfinder in ``pifilm/display/`` is the case that
forced this split — would otherwise drag OpenCV in with it, and OpenCV is
deliberately not a base dependency of this package. ``camera.py`` re-exports
the name, so every existing ``from .camera import CameraError`` still works.
"""

from __future__ import annotations


class CameraError(Exception):
    """Camera could not be opened or read."""
