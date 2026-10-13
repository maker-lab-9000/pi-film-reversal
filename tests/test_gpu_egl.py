"""The EGL device, as far as a machine without a GPU can check it.

The graphics calls themselves only run on the Pi. What can be pinned here:
importing the module loads nothing, a machine without the libraries gets a
``GpuError``, and, against a recording stand-in for the two libraries, the
device makes the calls the hardware spike made, reallocates on a new frame
size, turns every failure into ``GpuError`` and releases its context.
"""

import ctypes
import os
import sys

import numpy as np
import pytest

from pifilm.gpu import egl
from pifilm.gpu.grader import GpuError, GpuGrader
from pifilm.gpu.reference import PassTables
from tests.gpu_fakes import tile_artifacts

GL_TEXTURE_2D, GL_TEXTURE_3D = 0x0DE1, 0x806F
GL_RGBA, GL_RGB, GL_RED, GL_UNSIGNED_BYTE, GL_FLOAT = 0x1908, 0x1907, 0x1903, 0x1401, 0x1406
GL_RGBA8, GL_RGB8, GL_RGB32F, GL_R32F, GL_R16F = 0x8058, 0x8051, 0x8815, 0x822E, 0x822D


def _target(arg):
    """The ctypes object behind a ``byref``."""
    return arg._obj


def _address(arg) -> int:
    return arg.value if isinstance(arg, ctypes.c_void_p) else arg


class _Function:
    def __init__(self, lib, name):
        self.lib, self.name = lib, name

    def __call__(self, *args):
        # What ctypes itself would refuse on the Pi is refused here: with
        # declared types, the wrong count or an argument that does not convert;
        # without, anything but an int, None, bytes or a ctypes object (a NumPy
        # integer, for one).
        argtypes = self.__dict__.get("argtypes")
        if argtypes is not None:
            assert len(args) == len(argtypes), self.name
            for kind, arg in zip(argtypes, args, strict=True):
                kind.from_param(arg)
        else:
            for arg in args:
                assert arg is None or type(arg) in (int, bytes) or isinstance(
                    arg, (ctypes._SimpleCData, ctypes.Array, ctypes._Pointer)), (self.name, arg)
        self.lib.calls.append((self.name, args))
        handler = getattr(self.lib, "_" + self.name, None)
        return handler(*args) if handler else 0


class FakeLibs:
    """libEGL and libGLESv2 in one object: records every call and answers the
    few that the device reads an answer from."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.loaded: list[str] = []
        self.platform_at_display: list[str | None] = []
        self.errors: list[int] = []          # glGetError answers, then 0
        self.display = 0xD15
        self.configs = 1
        self.context = 0xC7
        self.compile_ok = 1
        self.next_id = 10
        self.uniforms: dict[bytes, int] = {}
        self.uploads: dict[str, bytes] = {}

    def __getattr__(self, name):
        if not name.startswith(("egl", "gl")):
            raise AttributeError(name)
        function = _Function(self, name)
        self.__dict__[name] = function
        return function

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args_of(self, name: str) -> list[tuple]:
        return [args for called, args in self.calls if called == name]

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _eglGetDisplay(self, native):
        self.platform_at_display.append(os.environ.get("EGL_PLATFORM"))
        return self.display

    def _eglInitialize(self, dpy, major, minor):
        _target(major).value, _target(minor).value = 1, 5
        return 1

    def _eglGetError(self):
        return 0x3001

    def _eglChooseConfig(self, dpy, want, config, size, count):
        _target(count).value = self.configs
        return 1

    def _eglCreateContext(self, dpy, config, share, attrs):
        return self.context

    def _eglMakeCurrent(self, dpy, draw, read, ctx):
        return 1

    def _glGetString(self, name):
        return {0x1F01: b"V3D 4.2.14.0", 0x1F02: b"OpenGL ES 3.1 Mesa 26.2"}.get(name)

    def _glGetIntegerv(self, name, value):
        _target(value).value = {0x0D33: 4096}.get(name, 0)

    def _glCreateShader(self, kind):
        return self._id()

    def _glCreateProgram(self):
        return self._id()

    def _glGetShaderiv(self, shader, name, ok):
        _target(ok).value = self.compile_ok

    def _glGetProgramiv(self, program, name, ok):
        _target(ok).value = 1

    def _glGetShaderInfoLog(self, shader, size, length, log):
        log.value = b"ERROR: 0:3: syntax error"

    def _gen(self, count, ids):
        _target(ids).value = self._id()

    _glGenTextures = _glGenVertexArrays = _glGenFramebuffers = _gen

    def _glCheckFramebufferStatus(self, target):
        return 0x8CD5

    def _glGetError(self):
        return self.errors.pop(0) if self.errors else 0

    def _glGetUniformLocation(self, program, name):
        return self.uniforms.setdefault(name, len(self.uniforms))

    def _glTexSubImage2D(self, target, level, x, y, width, height, fmt, kind, data):
        size = width * height * {GL_RGB: 3, GL_RED: 1}[fmt] * (4 if kind == GL_FLOAT else 1)
        self.uploads[f"{width}x{height}:{fmt:x}"] = ctypes.string_at(_address(data), size)

    def _glTexSubImage3D(self, target, level, x, y, z, w, h, d, fmt, kind, data):
        self.uploads["lut"] = ctypes.string_at(_address(data), w * h * d * 3 * 4)

    def _glReadPixels(self, x, y, width, height, fmt, kind, data):
        ctypes.memset(_address(data), 7, width * height * 4)


@pytest.fixture
def libs(monkeypatch):
    fake = FakeLibs()

    def load(name):
        fake.loaded.append(name)
        return fake

    monkeypatch.setattr(ctypes, "CDLL", load)
    return fake


def _tables() -> PassTables:
    ramp = np.arange(256, dtype=np.float32) / 255.0
    return PassTables(chan=np.stack([ramp] * 3, axis=-1), tone=ramp, use_tone=True,
                      gain=ramp * 0.02)


def _lut(n: int = 5) -> np.ndarray:
    return np.random.default_rng(1).random((n, n, n, 3)).astype(np.float32)


def _ready(libs) -> egl.EglDevice:
    device = egl.open_device()
    device.load_lut(_lut())
    device.load_tile(np.zeros((8, 8), np.float32))
    device.load_tables(_tables())
    return device


def _frame(height: int = 4, width: int = 6) -> np.ndarray:
    return np.random.default_rng(2).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _uniform(libs, name: bytes):
    location = libs.uniforms[name]
    values = [args[1:] for call, args in libs.calls
              if call in ("glUniform1i", "glUniform2i") and args[0] == location]
    return values[-1]


# --- without a GPU -----------------------------------------------------------


def test_importing_the_module_loads_no_graphics_library():
    # In a fresh interpreter, so the answer does not depend on what other tests
    # have loaded: ctypes.CDLL is replaced before the import and must not be called.
    import subprocess

    code = (
        "import ctypes\n"
        "def refuse(*a, **k): raise SystemExit('loaded ' + repr(a))\n"
        "ctypes.CDLL = refuse\n"
        "import pifilm.gpu.egl, pifilm.gpu.grader\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_open_device_raises_gpu_error_on_a_machine_without_egl():
    before = os.environ.get("EGL_PLATFORM")
    with pytest.raises(GpuError, match="libEGL"):
        egl.open_device()
    assert os.environ.get("EGL_PLATFORM") == before


def test_open_with_the_real_device_returns_none_on_a_machine_without_egl():
    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append) is None
    assert len(lines) == 1 and lines[0].startswith("gpu grading off: ")
    assert "libEGL" in lines[0]


# --- set-up ------------------------------------------------------------------


@pytest.mark.parametrize("previous", [None, "wayland"])
def test_the_platform_variable_is_set_for_the_display_only(libs, monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv("EGL_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("EGL_PLATFORM", previous)
    egl.open_device()
    assert libs.platform_at_display == ["surfaceless"]
    assert os.environ.get("EGL_PLATFORM") == previous


def test_set_up_makes_the_spikes_calls_in_its_order(libs):
    device = egl.open_device()
    assert libs.loaded == ["libEGL.so.1", "libGLESv2.so.2"]
    assert libs.names()[:6] == ["eglGetDisplay", "eglInitialize", "eglBindAPI",
                                "eglChooseConfig", "eglCreateContext", "eglMakeCurrent"]
    assert libs.args_of("eglGetDisplay") == [(None,)]
    assert libs.args_of("eglBindAPI") == [(0x30A0,)]
    assert list(libs.args_of("eglChooseConfig")[0][1]) == [0x3040, 0x0040, 0x3033, 0x0001, 0x3038]
    assert list(libs.args_of("eglCreateContext")[0][3]) == [0x3098, 3, 0x30FB, 0, 0x3038]
    assert libs.args_of("eglMakeCurrent") == [(0xD15, None, None, 0xC7)]
    assert libs.args_of("glPixelStorei") == [(0x0CF5, 1), (0x0D05, 1)]
    assert [args[0] for args in libs.args_of("glCreateShader")] == [0x8B31, 0x8B30]
    assert device.max_texture == 4096
    assert device.renderer == "V3D 4.2.14.0"


def test_pointer_arguments_are_declared(libs):
    egl.open_device()
    assert libs.eglGetDisplay.restype is ctypes.c_void_p
    assert libs.eglCreateContext.restype is ctypes.c_void_p
    assert libs.glGetString.restype is ctypes.c_char_p
    assert libs.eglMakeCurrent.argtypes == [ctypes.c_void_p] * 4
    assert libs.eglDestroyContext.argtypes == [ctypes.c_void_p] * 2
    for name in ("glTexSubImage2D", "glTexSubImage3D", "glReadPixels"):
        assert getattr(libs, name).argtypes[-1] is ctypes.c_void_p


def test_a_missing_library_is_a_gpu_error(monkeypatch):
    def load(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setenv("EGL_PLATFORM", "x11")
    with pytest.raises(GpuError, match="libEGL.so.1: cannot open"):
        egl.open_device()
    assert os.environ["EGL_PLATFORM"] == "x11"


def test_a_missing_function_is_a_gpu_error(libs):
    class Without(FakeLibs):
        def __getattr__(self, name):
            if name == "glGetString":      # as ctypes on a library without the symbol
                raise AttributeError("undefined symbol: glGetString")
            return super().__getattr__(name)

    libs.__class__ = Without
    with pytest.raises(GpuError, match="undefined symbol: glGetString"):
        egl.open_device()


def test_no_display_is_a_gpu_error_with_the_egl_code(libs):
    libs.display = 0
    with pytest.raises(GpuError, match="EGL init failed: 0x3001"):
        egl.open_device()


def test_no_es3_config_is_a_gpu_error(libs):
    libs.configs = 0
    with pytest.raises(GpuError, match="no EGL config"):
        egl.open_device()
    assert len(libs.args_of("eglChooseConfig")) == 2      # with, then without, a pbuffer


def test_no_context_is_a_gpu_error_with_the_egl_code(libs):
    libs.context = 0
    with pytest.raises(GpuError, match="no context: 0x3001"):
        egl.open_device()


def test_a_shader_that_does_not_compile_is_a_gpu_error_and_releases_the_context(libs):
    libs.compile_ok = 0
    with pytest.raises(GpuError, match="shader: ERROR: 0:3: syntax error"):
        egl.open_device()
    assert libs.args_of("eglDestroyContext") == [(0xD15, 0xC7)]


# --- textures ----------------------------------------------------------------


def test_the_lut_goes_up_as_float_with_red_along_x(libs):
    device = egl.open_device()
    table = _lut()
    device.load_lut(table)
    assert libs.args_of("glTexStorage3D") == [(GL_TEXTURE_3D, 1, GL_RGB32F, 5, 5, 5)]
    assert libs.args_of("glTexSubImage3D")[0][:10] == (
        GL_TEXTURE_3D, 0, 0, 0, 0, 5, 5, 5, GL_RGB, GL_FLOAT)
    assert libs.uploads["lut"] == np.ascontiguousarray(table.transpose(2, 1, 0, 3)).tobytes()


def test_the_tile_is_a_16_bit_float_texture_and_none_is_one_zero(libs):
    device = egl.open_device()
    device.load_tile(np.ones((8, 8), np.float32))
    device.load_tile(None)
    assert libs.args_of("glTexStorage2D") == [
        (GL_TEXTURE_2D, 1, GL_R16F, 8, 8), (GL_TEXTURE_2D, 1, GL_R16F, 1, 1)]
    assert libs.uploads[f"1x1:{GL_RED:x}"] == b"\0\0\0\0"
    # The first tile's texture is deleted when the second replaces it.
    assert len(libs.args_of("glDeleteTextures")) == 1


def test_the_tables_are_made_once_and_then_only_refilled(libs):
    device = egl.open_device()
    device.load_tables(_tables())
    device.load_tables(_tables())
    assert libs.args_of("glTexStorage2D") == [
        (GL_TEXTURE_2D, 1, GL_RGB32F, 256, 1), (GL_TEXTURE_2D, 1, GL_R32F, 256, 1),
        (GL_TEXTURE_2D, 1, GL_R32F, 256, 1)]
    assert len(libs.args_of("glTexSubImage2D")) == 6
    assert libs.args_of("glDeleteTextures") == []
    assert libs.uploads[f"256x1:{GL_RGB:x}"] == _tables().chan.tobytes()


@pytest.mark.parametrize("bad", [
    lambda d: d.load_tables(PassTables(chan=np.zeros((256,), np.float32),
                                       tone=np.zeros(256, np.float32), use_tone=False,
                                       gain=np.zeros(256, np.float32))),
    lambda d: d.load_tables(PassTables(chan=np.zeros((256, 3), np.float32),
                                       tone=np.zeros(255, np.float32), use_tone=False,
                                       gain=np.zeros(256, np.float32))),
    lambda d: d.load_tile(np.zeros((8, 9), np.float32)),
    lambda d: d.load_lut(np.zeros((5, 5, 4, 3), np.float32)),
])
def test_a_table_of_the_wrong_shape_is_refused_before_any_upload(libs, bad):
    device = egl.open_device()
    with pytest.raises(GpuError):
        bad(device)
    assert libs.uploads == {}


def test_a_gl_error_while_loading_is_a_gpu_error_with_the_code(libs):
    device = egl.open_device()
    libs.errors = [0x505]
    with pytest.raises(GpuError, match="0x505"):
        device.load_lut(_lut())


# --- a pass ------------------------------------------------------------------


def test_a_pass_uploads_three_bytes_draws_and_reads_back_four(libs):
    device = _ready(libs)
    frame = _frame()
    start = len(libs.calls)
    timings: dict = {}
    out = device.run(frame, use_tone=True, placement=(3, 5, 6), timings=timings)

    assert out.shape == (4, 6, 4) and out.dtype == np.uint8 and (out == 7).all()
    assert (GL_TEXTURE_2D, 1, GL_RGB8, 6, 4) in libs.args_of("glTexStorage2D")
    assert (GL_TEXTURE_2D, 1, GL_RGBA8, 6, 4) in libs.args_of("glTexStorage2D")
    assert libs.args_of("glViewport") == [(0, 0, 6, 4)]
    assert libs.uploads[f"6x4:{GL_RGB:x}"] == frame.tobytes()
    upload = [args for args in libs.args_of("glTexSubImage2D") if args[4:6] == (6, 4)][0]
    assert upload[:8] == (GL_TEXTURE_2D, 0, 0, 0, 6, 4, GL_RGB, GL_UNSIGNED_BYTE)
    assert libs.args_of("glReadPixels")[0][:6] == (0, 0, 6, 4, GL_RGBA, GL_UNSIGNED_BYTE)
    assert libs.args_of("glDrawArrays") == [(0x0004, 0, 3)]

    # Upload, finish, draw, finish, read: the order the timings are split by.
    order = [name for name in libs.names()[start:]
             if name in ("glFinish", "glDrawArrays", "glReadPixels")]
    assert order == ["glFinish", "glDrawArrays", "glFinish", "glReadPixels"]
    assert set(timings) == {"gpu_upload", "gpu_shader", "gpu_readback"}
    assert all(isinstance(v, float) and v >= 0 and round(v, 1) == v for v in timings.values())

    assert _uniform(libs, b"offset") == (5, 3)            # (ox, oy)
    assert _uniform(libs, b"orient") == (6,)
    assert _uniform(libs, b"use_grain") == (1,)
    assert _uniform(libs, b"use_tone") == (1,)
    assert _uniform(libs, b"n") == (5,)
    assert _uniform(libs, b"S") == (8,)
    # Each sampler has its own unit.
    samplers = [_uniform(libs, name)[0]
                for name in (b"img", b"chan", b"tone", b"lut", b"gain", b"tile")]
    assert samplers == [0, 1, 2, 3, 4, 5]


def test_a_pass_without_grain_or_tone_says_so_to_the_shader(libs):
    device = _ready(libs)
    device.run(_frame(), use_tone=False, placement=None)
    assert _uniform(libs, b"use_grain") == (0,)
    assert _uniform(libs, b"use_tone") == (0,)
    assert _uniform(libs, b"offset") == (0, 0)


def test_the_frames_textures_are_kept_for_the_same_size_and_remade_for_another(libs):
    device = _ready(libs)
    device.run(_frame(4, 6), use_tone=False, placement=None)
    device.run(_frame(4, 6), use_tone=False, placement=None)
    frame_storage = [args for args in libs.args_of("glTexStorage2D")
                     if args[2] in (GL_RGB8, GL_RGBA8)]
    assert len(frame_storage) == 2
    assert libs.args_of("glDeleteTextures") == [] and libs.args_of("glDeleteFramebuffers") == []

    device.run(_frame(6, 4), use_tone=False, placement=None)      # same pixels, other shape
    frame_storage = [args for args in libs.args_of("glTexStorage2D")
                     if args[2] in (GL_RGB8, GL_RGBA8)]
    assert frame_storage[2:] == [(GL_TEXTURE_2D, 1, GL_RGB8, 4, 6),
                                 (GL_TEXTURE_2D, 1, GL_RGBA8, 4, 6)]
    assert len(libs.args_of("glDeleteTextures")) == 2
    assert len(libs.args_of("glDeleteFramebuffers")) == 1
    assert libs.args_of("glViewport")[-1] == (0, 0, 4, 6)
    assert libs.args_of("glReadPixels")[-1][:4] == (0, 0, 4, 6)


def test_a_gl_error_in_a_pass_is_a_gpu_error_and_the_next_pass_is_clean(libs):
    device = _ready(libs)
    device.run(_frame(), use_tone=False, placement=None)
    libs.errors = [0x502, 0x505]
    with pytest.raises(GpuError, match="GL error after grading: 0x502"):
        device.run(_frame(), use_tone=False, placement=None)
    assert libs.errors == []                    # the second flag was read off too
    assert device.run(_frame(), use_tone=False, placement=None).shape == (4, 6, 4)


def test_a_failed_allocation_is_made_again_for_the_same_size(libs):
    device = _ready(libs)
    libs.errors = [0x505]
    with pytest.raises(GpuError, match="allocation: 0x505"):
        device.run(_frame(), use_tone=False, placement=None)
    device.run(_frame(), use_tone=False, placement=None)
    frame_storage = [args for args in libs.args_of("glTexStorage2D")
                     if args[2] in (GL_RGB8, GL_RGBA8)]
    assert len(frame_storage) == 4


def test_an_incomplete_framebuffer_is_a_gpu_error_with_its_status(libs):
    device = _ready(libs)
    libs._glCheckFramebufferStatus = lambda target: 0x8CD6
    with pytest.raises(GpuError, match="framebuffer incomplete: 0x8cd6"):
        device.run(_frame(), use_tone=False, placement=None)


@pytest.mark.parametrize("frame", [
    np.zeros((4, 6), np.uint8), np.zeros((4, 6, 4), np.uint8), np.zeros((4, 6, 3), np.float32),
    np.zeros((4, 5000, 3), np.uint8),
])
def test_a_frame_the_device_cannot_take_is_refused_before_any_call(libs, frame):
    device = _ready(libs)
    start = len(libs.calls)
    with pytest.raises(GpuError):
        device.run(frame, use_tone=False, placement=None)
    assert len(libs.calls) == start


def test_a_pass_before_the_textures_are_loaded_is_refused(libs):
    device = egl.open_device()
    with pytest.raises(GpuError, match="lut"):
        device.run(_frame(), use_tone=False, placement=None)


# --- close -------------------------------------------------------------------


def test_close_releases_the_context_once_and_a_pass_afterwards_is_refused(libs):
    device = _ready(libs)
    device.close()
    device.close()
    assert libs.args_of("eglMakeCurrent")[1:] == [(0xD15, None, None, None)]
    assert libs.args_of("eglDestroyContext") == [(0xD15, 0xC7)]
    start = len(libs.calls)
    with pytest.raises(GpuError, match="closed"):
        device.run(_frame(), use_tone=False, placement=None)
    with pytest.raises(GpuError, match="closed"):
        device.load_tables(_tables())
    assert len(libs.calls) == start


def test_the_grader_drives_the_device_end_to_end(libs):
    # The recording libraries return a flat 7, so the self-test must refuse the
    # device: the whole path from open to a closed context, with no GPU.
    lines: list[str] = []
    assert GpuGrader.open(tile_artifacts(), log=lines.append) is None
    assert len(lines) == 1 and lines[0].startswith("gpu grading off: self-test: ")
    assert libs.args_of("eglDestroyContext") == [(0xD15, 0xC7)]
