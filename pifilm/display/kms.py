"""Waveshare 3.5" DSI LCD (E), 640x480, driven through DRM/KMS with ``pykms``.

Unlike the 2.8" SPI panel, this one is a kernel display: the ``waveshare_35DSI``
overlay registers it as connector ``DSI-1`` and the app only has to hand the
kernel finished frames. ``pykms`` (apt ``python3-kms++``) is already on the Pi as
a Picamera2 dependency, and it is imported only inside ``open_waveshare35dsi`` so
this module imports on a Mac.

Frames go into one of two dumb XRGB8888 buffers and the buffers are swapped, so
the panel never shows a half-drawn frame. The alternative, writing into
``/dev/fb0``, was measured to work (16 bpp, 2026-10-10) and is the documented
fallback, but it tears and it shares the screen with the login console. While
this process holds the display the kernel console is kept off the panel, and it
returns when the card is closed.

The service must own the screen, so the Pi boots to the console: under a desktop
session the compositor is the DRM master and opening the output here fails. That
failure, like every other at open, is a ``DisplayError`` and costs the screen
only; ``pifilm-capture`` carries on headless for the Stick.

The panel has no brightness control (``/sys/class/backlight`` is empty), so
``backlight`` is on/off: 0 powers the output down and anything else powers it up.
``dimmable = False`` tells the viewfinder loop to skip its half-brightness step.

``KmsDisplay`` holds the logic and is tested against a fake backend;
``_PykmsBackend`` is the only code that touches ``pykms`` and is proven on the Pi
(see the acceptance checklist in docs/lcd-viewfinder.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from . import DisplayError

CONNECTOR = "DSI-1"
PIXEL_FORMAT = "XR24"  # XRGB8888: bytes in memory are B, G, R, X


def pack_xrgb8888(rgb: np.ndarray, rotate: int = 0) -> np.ndarray:
    """An (H, W, 3) RGB array as (H, W, 4) B, G, R, X bytes, optionally turned 180."""
    if rotate == 180:
        rgb = rgb[::-1, ::-1]
    out = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., 0] = rgb[..., 2]
    out[..., 1] = rgb[..., 1]
    out[..., 2] = rgb[..., 0]
    out[..., 3] = 255
    return out


class KmsDisplay:
    dimmable = False

    def __init__(self, backend: Any, *, rotate: int = 0) -> None:
        if rotate not in (0, 180):
            raise DisplayError(f"rotate must be 0 or 180, got {rotate}")
        self._backend, self._rotate = backend, rotate
        self.width, self.height = int(backend.width), int(backend.height)
        self._front = 0
        self._on = True

    def show(self, image: Image.Image) -> None:
        if image.size != (self.width, self.height):
            raise DisplayError(
                f"frame must be {self.width}x{self.height}, "
                f"got {image.size[0]}x{image.size[1]}"
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        back = 1 - self._front
        try:
            self._backend.write(back, pack_xrgb8888(np.asarray(image), self._rotate))
            self._backend.flip(back)
        except (OSError, RuntimeError) as exc:
            raise DisplayError(f"display swap failed: {exc}") from exc
        self._front = back

    def backlight(self, percent: int) -> None:
        on = int(percent) > 0
        if on == self._on:
            return
        try:
            self._backend.power(on)
        except (OSError, RuntimeError) as exc:
            raise DisplayError(f"display power {'on' if on else 'off'} failed: {exc}") from exc
        self._on = on

    def close(self) -> None:
        try:
            if not self._on:
                self._backend.power(True)
        except (OSError, RuntimeError):
            pass
        finally:
            self._backend.close()


class _PykmsBackend:
    """The pykms calls. No logic beyond what the hardware needs."""

    def __init__(self, pykms: Any) -> None:
        self._pykms = pykms
        self._card = pykms.Card()
        resources = pykms.ResourceManager(self._card)
        self._conn = resources.reserve_connector(CONNECTOR)
        self._crtc = resources.reserve_crtc(self._conn)
        self._mode = self._conn.get_default_mode()
        self.width, self.height = self._mode.hdisplay, self._mode.vdisplay
        self._fbs = [
            pykms.DumbFramebuffer(self._card, self.width, self.height, PIXEL_FORMAT)
            for _ in range(2)
        ]
        self._maps = []
        for fb in self._fbs:
            if fb.stride(0) != self.width * 4:
                raise RuntimeError(
                    f"unexpected buffer stride {fb.stride(0)} for width {self.width}"
                )
            flat = np.frombuffer(fb.map(0), dtype=np.uint8)
            self._maps.append(
                flat[: self.height * self.width * 4].reshape(self.height, self.width, 4)
            )
        self._crtc.set_mode(self._conn, self._fbs[0], self._mode)
        self._plane = self._crtc.primary_plane

    def write(self, index: int, pixels: np.ndarray) -> None:
        np.copyto(self._maps[index], pixels)

    def flip(self, index: int) -> None:
        request = self._pykms.AtomicReq(self._card)
        request.add(self._plane, "FB_ID", self._fbs[index].id)
        result = request.commit_sync()
        if result:
            raise OSError(f"atomic commit returned {result}")

    def power(self, on: bool) -> None:
        # The connector's DPMS property is refused on this panel ("commit failed",
        # measured 2026-10-10); switching the CRTC's ACTIVE state is what works.
        request = self._pykms.AtomicReq(self._card)
        request.add(self._crtc, "ACTIVE", 1 if on else 0)
        result = request.commit_sync(allow_modeset=True)
        if result:
            raise OSError(f"ACTIVE commit returned {result}")

    def close(self) -> None:
        # Dropping the last references closes the card, which hands the screen back
        # to the kernel console.
        self._maps.clear()
        self._fbs.clear()
        self._plane = self._crtc = self._conn = self._card = None


def open_waveshare35dsi(rotate: int = 0) -> KmsDisplay:
    """Open the DSI panel. Raises DisplayError with the cause and the fix."""
    if rotate not in (0, 180):
        raise DisplayError(f"rotate must be 0 or 180, got {rotate}")
    try:
        import pykms
    except ImportError as exc:
        raise DisplayError(
            "display libraries missing; install python3-kms++ from apt"
        ) from exc
    try:
        backend = _PykmsBackend(pykms)
    except PermissionError as exc:
        raise DisplayError(
            f"no permission for the DSI display: {exc}; add the service user to the "
            "video and render groups and log in again"
        ) from exc
    except Exception as exc:  # pykms raises plain RuntimeError/ValueError for most faults
        raise DisplayError(
            f"cannot open the DSI display on {CONNECTOR}: {exc}; check the "
            "dtoverlay=waveshare_35DSI line in config.txt and the cable, and that the Pi "
            "boots to the console (a desktop session owns the screen)"
        ) from exc
    return KmsDisplay(backend, rotate=rotate)
