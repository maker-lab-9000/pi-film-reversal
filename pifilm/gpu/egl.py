"""The GPU device: a windowless EGL context and the handful of OpenGL ES 3
calls of the pass, through ``ctypes`` on the system's ``libEGL`` and
``libGLESv2``. No new dependency.

This is the ``Gpu`` class of ``scripts/experiments/gpu_pass_spike.py`` (branch
``spike/gpu-pass``) shaped into the grader's ``Device``. That class ran
correctly on the target (Pi 4, V3D 4.2, OpenGL ES 3.1, Mesa 26.2) on
2026-10-13, and nothing here can run on a Mac or in the test suite, where a
recording stand-in for the libraries checks the calls and not their effect. So
the call sequence, the constants and the formats are the spike's, and every
difference from it is listed here to be checked on the hardware:

* A texture that is replaced (the frame's pair when the frame's size changes,
  the tile, the LUT) is deleted first, and so is the old framebuffer. The
  spike leaked them, which a script may and a service may not.
* ``close`` releases the context: nothing current, then ``eglDestroyContext``.
  The display is left initialised (``eglTerminate`` would end it for the whole
  process).
* ``EGL_PLATFORM`` is set to ``surfaceless`` around the display's creation
  only and then put back, so nothing else in the process inherits it.
* ``argtypes`` are declared on the GL functions that take a pointer, as the
  spike did for EGL only, and on the two EGL functions ``close`` adds.
* After a GL error the remaining error flags are read off, so one failed pass
  is not reported again by the next.
* The set-up, the tables, the LUT and the tile are each checked for a GL
  error (the spike checked after allocation and after a pass), and a frame or
  a table of the wrong shape is refused before it reaches a GL call.
* With no grain the tile is one zero texel, so every sampler has a texture.
* The spike's printing and its query of the preferred read-back format are
  gone; the four-byte upload and three-byte read-back it tried are too.

What the hardware showed and this relies on: the frame goes up as three bytes
a pixel (``GL_RGB8``), and comes back as four (``GL_RGBA``). A three-byte
read-back fails with GL error 0x502 and is not attempted.

Everything that fails is a ``GpuError`` carrying the EGL or GL code: the
caller grades on the CPU and logs it. The libraries are loaded inside
``open_device``, never at import. A context belongs to the thread that made
it: open, use and close a device on one thread (``GpuGrader`` enforces it).
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p

import numpy as np

from .grader import GpuError
from .reference import PassTables
from .shader import FRAGMENT, VERTEX

EGL_NONE = 0x3038
GL_TEXTURE_2D, GL_TEXTURE_3D = 0x0DE1, 0x806F
GL_RGBA, GL_RGB, GL_RED, GL_UNSIGNED_BYTE, GL_FLOAT = 0x1908, 0x1907, 0x1903, 0x1401, 0x1406
GL_RGBA8, GL_RGB8, GL_RGB32F, GL_R32F, GL_R16F = 0x8058, 0x8051, 0x8815, 0x822E, 0x822D
GL_NEAREST, GL_CLAMP = 0x2600, 0x812F
GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 = 0x8D40, 0x8CE0
# The textures a pass reads besides the frame; all must be loaded before one.
LOADED = ("chan", "tone", "lut", "gain", "tile")


def open_device() -> EglDevice:
    """A device with a current context on the calling thread. Raises ``GpuError``."""
    try:
        return EglDevice()
    except GpuError:
        raise
    except (OSError, AttributeError) as exc:
        # A library that is not there, or one without a function this needs.
        raise GpuError(str(exc)) from exc


def ptr(array: np.ndarray) -> c_void_p:
    return array.ctypes.data_as(c_void_p)


def _declare(egl, gl) -> None:
    """Argument and return types for every function that takes or returns a
    pointer. Without them ctypes passes a handle as a 32-bit int."""
    egl.eglGetDisplay.restype = c_void_p
    egl.eglGetDisplay.argtypes = [c_void_p]
    egl.eglInitialize.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    egl.eglChooseConfig.argtypes = [
        c_void_p, POINTER(c_int), POINTER(c_void_p), c_int, POINTER(c_int)]
    egl.eglCreateContext.restype = c_void_p
    egl.eglCreateContext.argtypes = [
        c_void_p, c_void_p, c_void_p, POINTER(c_int)]
    egl.eglMakeCurrent.argtypes = [c_void_p] * 4
    egl.eglDestroyContext.argtypes = [c_void_p] * 2
    gl.glGetString.restype = c_char_p
    gl.glGetIntegerv.argtypes = [c_uint, POINTER(c_int)]
    gl.glShaderSource.argtypes = [c_uint, c_int, POINTER(c_char_p), c_void_p]
    gl.glGetShaderiv.argtypes = [c_uint, c_uint, POINTER(c_int)]
    gl.glGetProgramiv.argtypes = [c_uint, c_uint, POINTER(c_int)]
    gl.glGetShaderInfoLog.argtypes = [c_uint, c_int, c_void_p, c_char_p]
    gl.glGetProgramInfoLog.argtypes = [c_uint, c_int, c_void_p, c_char_p]
    gl.glGetUniformLocation.argtypes = [c_uint, c_char_p]
    for name in ("glGenVertexArrays", "glGenTextures", "glGenFramebuffers",
                 "glDeleteTextures", "glDeleteFramebuffers"):
        getattr(gl, name).argtypes = [c_int, POINTER(c_uint)]
    gl.glTexSubImage2D.argtypes = [c_uint, c_int, c_int, c_int, c_int, c_int,
                                   c_uint, c_uint, c_void_p]
    gl.glTexSubImage3D.argtypes = [c_uint, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                                   c_uint, c_uint, c_void_p]
    gl.glReadPixels.argtypes = [c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p]


class EglDevice:
    def __init__(self) -> None:
        self.egl = self.gl = None
        self._dpy = self._ctx = None
        self.textures: dict[str, int] = {}
        self.size = (0, 0)              # (width, height) the frame's textures are for
        self._fbo = 0
        self.max_texture = 0
        self.renderer: str | None = None
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        previous = os.environ.get("EGL_PLATFORM")
        os.environ["EGL_PLATFORM"] = "surfaceless"
        try:
            self.egl = egl = ctypes.CDLL("libEGL.so.1")
            self.gl = gl = ctypes.CDLL("libGLESv2.so.2")
            _declare(egl, gl)
            dpy = egl.eglGetDisplay(None)
            major, minor = c_int(), c_int()
            if not dpy or not egl.eglInitialize(dpy, byref(major), byref(minor)):
                raise GpuError(f"EGL init failed: 0x{egl.eglGetError():x}")
        finally:
            if previous is None:
                os.environ.pop("EGL_PLATFORM", None)
            else:
                os.environ["EGL_PLATFORM"] = previous
        self._dpy = dpy
        egl.eglBindAPI(0x30A0)                      # EGL_OPENGL_ES_API
        want = (c_int * 5)(0x3040, 0x0040,          # RENDERABLE_TYPE = ES3
                           0x3033, 0x0001, EGL_NONE)  # SURFACE_TYPE = PBUFFER
        config, count = c_void_p(), c_int()
        egl.eglChooseConfig(dpy, want, byref(config), 1, byref(count))
        if count.value < 1:
            want = (c_int * 3)(0x3040, 0x0040, EGL_NONE)
            egl.eglChooseConfig(dpy, want, byref(config), 1, byref(count))
        if count.value < 1:
            raise GpuError("no EGL config for OpenGL ES 3")
        attrs = (c_int * 5)(0x3098, 3, 0x30FB, 0, EGL_NONE)
        ctx = egl.eglCreateContext(dpy, config, None, attrs)
        if not ctx:
            raise GpuError(f"no context: 0x{egl.eglGetError():x}")
        self._ctx = ctx
        if not egl.eglMakeCurrent(dpy, None, None, ctx):
            raise GpuError(f"no context: 0x{egl.eglGetError():x}")
        self.renderer = (gl.glGetString(0x1F01) or b"").decode(errors="replace") or None
        value = c_int()
        gl.glGetIntegerv(0x0D33, byref(value))
        self.max_texture = value.value
        gl.glPixelStorei(0x0CF5, 1)                 # UNPACK_ALIGNMENT
        gl.glPixelStorei(0x0D05, 1)                 # PACK_ALIGNMENT
        self.program = self._program()
        self._check("set-up")

    def _shader(self, kind: int, source: bytes) -> int:
        gl = self.gl
        shader = gl.glCreateShader(kind)
        text = c_char_p(source)
        gl.glShaderSource(shader, 1, byref(text), None)
        gl.glCompileShader(shader)
        ok = c_int()
        gl.glGetShaderiv(shader, 0x8B81, byref(ok))
        if not ok.value:
            log = ctypes.create_string_buffer(4096)
            gl.glGetShaderInfoLog(shader, 4096, None, log)
            raise GpuError("shader: " + log.value.decode(errors="replace"))
        return shader

    def _program(self) -> int:
        gl = self.gl
        program = gl.glCreateProgram()
        gl.glAttachShader(program, self._shader(0x8B31, VERTEX))
        gl.glAttachShader(program, self._shader(0x8B30, FRAGMENT))
        gl.glLinkProgram(program)
        ok = c_int()
        gl.glGetProgramiv(program, 0x8B82, byref(ok))
        if not ok.value:
            log = ctypes.create_string_buffer(4096)
            gl.glGetProgramInfoLog(program, 4096, None, log)
            raise GpuError("link: " + log.value.decode(errors="replace"))
        vao = c_uint()
        gl.glGenVertexArrays(1, byref(vao))
        gl.glBindVertexArray(vao)
        return program

    def _texture(self, name: str, target: int) -> int:
        """A new texture under ``name``, bound; the one it replaces is deleted."""
        gl = self.gl
        self._delete(name)
        tex = c_uint()
        gl.glGenTextures(1, byref(tex))
        gl.glBindTexture(target, tex)
        for pname in (0x2801, 0x2800):
            gl.glTexParameteri(target, pname, GL_NEAREST)
        for pname in (0x2802, 0x2803, 0x8072):
            gl.glTexParameteri(target, pname, GL_CLAMP)
        self.textures[name] = tex.value
        return tex.value

    def _delete(self, name: str) -> None:
        old = self.textures.pop(name, None)
        if old:
            self.gl.glDeleteTextures(1, byref(c_uint(old)))

    def _table(self, name: str, values: np.ndarray) -> None:
        """A 256x1 float table: one channel (R32F) or three (RGB32F). Made on
        the first call, refilled on the later ones."""
        gl = self.gl
        values = np.ascontiguousarray(values, np.float32)
        if name in self.textures:
            gl.glBindTexture(GL_TEXTURE_2D, self.textures[name])
        else:
            self._texture(name, GL_TEXTURE_2D)
            gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F if values.ndim == 1 else GL_RGB32F,
                              256, 1)
        fmt = GL_RED if values.ndim == 1 else GL_RGB
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, fmt, GL_FLOAT, ptr(values))

    def load_tables(self, tables: PassTables) -> None:
        self._require_open()
        named = (("chan", tables.chan, (256, 3)), ("tone", tables.tone, (256,)),
                 ("gain", tables.gain, (256,)))
        for name, values, shape in named:
            # The upload reads 256 entries whatever it is given.
            if np.shape(values) != shape:
                raise GpuError(f"table {name} has shape {np.shape(values)}, not {shape}")
        for name, values, _ in named:
            self._table(name, values)
        self._check("the tables")

    def load_tile(self, tile: np.ndarray | None) -> None:
        self._require_open()
        if tile is None:
            tile = np.zeros((1, 1), np.float32)
        if tile.ndim != 2 or tile.shape[0] != tile.shape[1]:
            raise GpuError(f"the tile has shape {tile.shape}, not a square")
        gl, size = self.gl, tile.shape[0]
        data = np.ascontiguousarray(tile, np.float32)
        self._texture("tile", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16F, size, size)
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size, size, GL_RED, GL_FLOAT, ptr(data))
        self.tile_size = size
        self._check("the tile")

    def load_lut(self, table: np.ndarray) -> None:
        self._require_open()
        n = table.shape[0]
        if table.shape != (n, n, n, 3):
            raise GpuError(f"the LUT has shape {table.shape}, not (n, n, n, 3)")
        gl = self.gl
        # Texture (x, y, z) is the LUT's (r, g, b), as the shader indexes it.
        data = np.ascontiguousarray(table.transpose(2, 1, 0, 3), np.float32)
        self._texture("lut", GL_TEXTURE_3D)
        gl.glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGB32F, n, n, n)
        gl.glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, n, n, n, GL_RGB, GL_FLOAT, ptr(data))
        self.n = n
        self._check("the LUT")

    def _allocate(self, width: int, height: int) -> None:
        """The frame's input and output textures and the framebuffer, for this
        size; the previous ones are deleted."""
        gl = self.gl
        self.size = (0, 0)      # until this succeeds, no size is allocated
        if self._fbo:
            gl.glDeleteFramebuffers(1, byref(c_uint(self._fbo)))
            self._fbo = 0
        self._texture("in", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, width, height)
        self._texture("out", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height)
        fbo = c_uint()
        gl.glGenFramebuffers(1, byref(fbo))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        self._fbo = fbo.value
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, self.textures["out"], 0)
        status = gl.glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != 0x8CD5:
            raise GpuError(f"framebuffer incomplete: 0x{status:x}")
        gl.glViewport(0, 0, width, height)
        self._check("allocation")
        self.size = (width, height)

    def _require_open(self) -> None:
        if self._ctx is None:
            raise GpuError("the device is closed")

    def _check(self, where: str) -> None:
        gl = self.gl
        error = gl.glGetError()
        if error:
            # Read off the other flags (bounded: a lost context reports for ever).
            for _ in range(16):
                if not gl.glGetError():
                    break
            raise GpuError(f"GL error after {where}: 0x{error:x}")

    def run(self, rgb_u8: np.ndarray, *, use_tone: bool,
            placement: tuple[int, int, int] | None,
            timings: dict | None = None) -> np.ndarray:
        """One pass. ``placement`` is (oy, ox, orientation) or None for no
        grain. Returns the read-back buffer, four bytes a pixel."""
        self._require_open()
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise GpuError(f"the frame is {rgb_u8.dtype} {rgb_u8.shape}, not (H, W, 3) bytes")
        height, width = rgb_u8.shape[:2]
        if max(width, height) > self.max_texture:
            raise GpuError(f"{width}x{height} exceeds {self.max_texture}")
        missing = [name for name in LOADED if name not in self.textures]
        if missing:
            raise GpuError("not loaded: " + ", ".join(missing))
        gl = self.gl
        if (width, height) != self.size:
            self._allocate(width, height)
        pixels = np.ascontiguousarray(rgb_u8)
        t = time.perf_counter()
        gl.glBindTexture(GL_TEXTURE_2D, self.textures["in"])
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                           GL_RGB, GL_UNSIGNED_BYTE, ptr(pixels))
        gl.glFinish()
        upload = time.perf_counter() - t
        t = time.perf_counter()
        program = self.program
        gl.glUseProgram(program)
        units = (("img", GL_TEXTURE_2D, "in"), ("chan", GL_TEXTURE_2D, "chan"),
                 ("tone", GL_TEXTURE_2D, "tone"), ("lut", GL_TEXTURE_3D, "lut"),
                 ("gain", GL_TEXTURE_2D, "gain"), ("tile", GL_TEXTURE_2D, "tile"))
        for unit, (uniform, target, name) in enumerate(units):
            gl.glActiveTexture(0x84C0 + unit)
            gl.glBindTexture(target, self.textures[name])
            gl.glUniform1i(gl.glGetUniformLocation(program, uniform.encode()), unit)
        gl.glActiveTexture(0x84C0)
        oy, ox, orientation = (int(v) for v in placement or (0, 0, 0))
        for uniform, value in (("n", self.n), ("S", self.tile_size), ("orient", orientation),
                               ("use_tone", int(use_tone)),
                               ("use_grain", int(placement is not None))):
            gl.glUniform1i(gl.glGetUniformLocation(program, uniform.encode()), value)
        gl.glUniform2i(gl.glGetUniformLocation(program, b"offset"), ox, oy)
        gl.glDrawArrays(0x0004, 0, 3)
        gl.glFinish()
        shader = time.perf_counter() - t
        t = time.perf_counter()
        out = np.empty((height, width, 4), np.uint8)
        gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, ptr(out))
        readback = time.perf_counter() - t
        self._check("grading")
        if timings is not None:
            timings["gpu_upload"] = round(upload * 1000.0, 1)
            timings["gpu_shader"] = round(shader * 1000.0, 1)
            timings["gpu_readback"] = round(readback * 1000.0, 1)
        return out

    def close(self) -> None:
        """Release the context (and with it every texture). Safe to call twice."""
        egl, dpy, ctx = self.egl, self._dpy, self._ctx
        self._dpy = self._ctx = None
        self.textures = {}
        self.size = (0, 0)
        self._fbo = 0
        if egl is None or not dpy:
            return
        egl.eglMakeCurrent(dpy, None, None, None)
        if ctx:
            egl.eglDestroyContext(dpy, ctx)
