"""Throwaway hardware spike 2: the whole grade in one GPU pass on the Pi 4
(step 7 of the pipeline-improvements roadmap). Not product code; nothing imports it.

Normalise (per-channel table, luma tone table), the 3D LUT and the tiled HP5 grain
in one fragment shader, in float, rounded once. It answers: how long the pass takes
against the CPU pipeline on the same frame, whether a three-byte upload and a
four-byte JPEG encode avoid the repacking that made the LUT-only spike no faster,
how far the result is from the CPU's, and whether it all runs on a worker thread.

Run on the Pi with the capture service stopped:

    .venv/bin/python scripts/experiments/gpu_pass_spike.py \
        --artifacts artifacts/imx477-2026-09-v4-dark-hp5-tile \
        --image ~/Pictures/pifilm/<day>/<name>_original.jpg
"""

from __future__ import annotations

import argparse
import ctypes
import io
import os
import threading
import time
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p

import numpy as np

EGL_NONE = 0x3038
GL_TEXTURE_2D, GL_TEXTURE_3D = 0x0DE1, 0x806F
GL_RGBA, GL_RGB, GL_RED, GL_UNSIGNED_BYTE, GL_FLOAT = 0x1908, 0x1907, 0x1903, 0x1401, 0x1406
GL_RGBA8, GL_RGB8, GL_RGB32F, GL_R32F, GL_R16F = 0x8058, 0x8051, 0x8815, 0x822E, 0x822D
GL_NEAREST, GL_CLAMP = 0x2600, 0x812F
GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 = 0x8D40, 0x8CE0

VERTEX = b"""#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
FRAGMENT = b"""#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
uniform sampler2D img;     // the frame, bytes
uniform sampler2D chan;    // 256x1 RGB: per-channel normalisation, float
uniform sampler2D tone;    // 256x1 R: luma tone curve, float
uniform sampler3D lut;     // n^3 RGB
uniform sampler2D gain;    // 256x1 R: grain strength by luminance, in 0..1 units
uniform sampler2D tile;    // SxS R: unit-variance grain
uniform int n;
uniform int S;
uniform ivec2 offset;      // (ox, oy)
uniform int orient;
uniform int use_tone;
uniform int use_grain;
out vec4 o;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
float table(sampler2D t, float v) {
  float x = clamp(v, 0.0, 1.0) * 255.0;
  float lo = min(floor(x), 254.0);
  int i = int(lo);
  return mix(texelFetch(t, ivec2(i, 0), 0).r, texelFetch(t, ivec2(i + 1, 0), 0).r, x - lo);
}
vec3 at(ivec3 i) { return texelFetch(lut, i, 0).rgb; }
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  ivec3 b = ivec3(texelFetch(img, px, 0).rgb * 255.0 + 0.5);
  vec3 c = vec3(texelFetch(chan, ivec2(b.r, 0), 0).r,
                texelFetch(chan, ivec2(b.g, 0), 0).g,
                texelFetch(chan, ivec2(b.b, 0), 0).b);
  if (use_tone == 1) {
    float y = dot(c, LUMA);
    c = clamp(c + (table(tone, y) - y), 0.0, 1.0);
  }
  vec3 s = c * float(n - 1);
  vec3 lo = min(floor(s), float(n - 2));
  ivec3 i = ivec3(lo);
  vec3 f = s - lo;
  vec3 c00 = mix(at(i), at(i + ivec3(1, 0, 0)), f.r);
  vec3 c10 = mix(at(i + ivec3(0, 1, 0)), at(i + ivec3(1, 1, 0)), f.r);
  vec3 c01 = mix(at(i + ivec3(0, 0, 1)), at(i + ivec3(1, 0, 1)), f.r);
  vec3 c11 = mix(at(i + ivec3(0, 1, 1)), at(i + ivec3(1, 1, 1)), f.r);
  vec3 g = clamp(mix(mix(c00, c10, f.g), mix(c01, c11, f.g), f.b), 0.0, 1.0);
  if (use_grain == 1) {
    int row = (px.y + offset.y) % S;
    int col = (px.x + offset.x) % S;
    if ((orient & 4) != 0) { int t = row; row = col; col = t; }
    if ((orient & 1) != 0) row = S - 1 - row;
    if ((orient & 2) != 0) col = S - 1 - col;
    g += table(gain, dot(g, LUMA)) * texelFetch(tile, ivec2(col, row), 0).r;
  }
  o = vec4(clamp(g, 0.0, 1.0), 1.0);
}
"""


def cma_free_mb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("CmaFree"):
            return int(line.split()[1]) / 1024
    return float("nan")


def ptr(array: np.ndarray) -> c_void_p:
    return array.ctypes.data_as(c_void_p)


class Gpu:
    def __init__(self, platform: str) -> None:
        os.environ["EGL_PLATFORM"] = platform
        self.egl = egl = ctypes.CDLL("libEGL.so.1")
        self.gl = gl = ctypes.CDLL("libGLESv2.so.2")
        egl.eglGetDisplay.restype = c_void_p
        egl.eglGetDisplay.argtypes = [c_void_p]
        egl.eglInitialize.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
        egl.eglChooseConfig.argtypes = [
            c_void_p, POINTER(c_int), POINTER(c_void_p), c_int, POINTER(c_int)]
        egl.eglCreateContext.restype = c_void_p
        egl.eglCreateContext.argtypes = [
            c_void_p, c_void_p, c_void_p, POINTER(c_int)]
        egl.eglMakeCurrent.argtypes = [c_void_p] * 4
        gl.glGetString.restype = c_char_p
        native = None
        if platform == "gbm":
            gbm = ctypes.CDLL("libgbm.so.1")
            gbm.gbm_create_device.restype = c_void_p
            fd = os.open("/dev/dri/renderD128", os.O_RDWR)
            native = gbm.gbm_create_device(fd)
        dpy = egl.eglGetDisplay(native)
        major, minor = c_int(), c_int()
        if not dpy or not egl.eglInitialize(dpy, byref(major), byref(minor)):
            raise RuntimeError(f"EGL init failed: 0x{egl.eglGetError():x}")
        egl.eglBindAPI(0x30A0)                      # EGL_OPENGL_ES_API
        want = (c_int * 5)(0x3040, 0x0040,          # RENDERABLE_TYPE = ES3
                           0x3033, 0x0001, EGL_NONE)  # SURFACE_TYPE = PBUFFER
        config, count = c_void_p(), c_int()
        egl.eglChooseConfig(dpy, want, byref(config), 1, byref(count))
        if count.value < 1:
            want = (c_int * 3)(0x3040, 0x0040, EGL_NONE)
            egl.eglChooseConfig(dpy, want, byref(config), 1, byref(count))
        if count.value < 1:
            raise RuntimeError("no EGL config for OpenGL ES 3")
        attrs = (c_int * 5)(0x3098, 3, 0x30FB, 0, EGL_NONE)
        ctx = egl.eglCreateContext(dpy, config, None, attrs)
        if not ctx or not egl.eglMakeCurrent(dpy, None, None, ctx):
            raise RuntimeError(f"no context: 0x{egl.eglGetError():x}")
        print(f"EGL {major.value}.{minor.value}, platform {platform}")
        for name, code in (("vendor", 0x1F00), ("renderer", 0x1F01),
                           ("version", 0x1F02)):
            print(f"  GL {name}: {(gl.glGetString(code) or b'?').decode()}")
        value = c_int()
        gl.glGetIntegerv(0x0D33, byref(value))
        self.max_2d = value.value
        gl.glGetIntegerv(0x8073, byref(value))
        print(f"  max texture {self.max_2d}, max 3D texture {value.value}")
        gl.glPixelStorei(0x0CF5, 1)                 # UNPACK_ALIGNMENT
        gl.glPixelStorei(0x0D05, 1)                 # PACK_ALIGNMENT
        self.program = self._program()
        self.textures: dict[str, int] = {}
        self.size = (0, 0)

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
            raise RuntimeError("shader: " + log.value.decode())
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
            raise RuntimeError("link: " + log.value.decode())
        vao = c_uint()
        gl.glGenVertexArrays(1, byref(vao))
        gl.glBindVertexArray(vao)
        return program

    def _texture(self, name: str, target: int) -> int:
        gl = self.gl
        tex = c_uint()
        gl.glGenTextures(1, byref(tex))
        gl.glBindTexture(target, tex)
        for pname in (0x2801, 0x2800):
            gl.glTexParameteri(target, pname, GL_NEAREST)
        for pname in (0x2802, 0x2803, 0x8072):
            gl.glTexParameteri(target, pname, GL_CLAMP)
        self.textures[name] = tex.value
        return tex.value

    def table(self, name: str, values: np.ndarray) -> None:
        """A 256x1 float table: one channel (R32F) or three (RGB32F)."""
        gl = self.gl
        values = np.ascontiguousarray(values, np.float32)
        self._texture(name, GL_TEXTURE_2D)
        if values.ndim == 1:
            gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 256, 1)
            gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RED, GL_FLOAT, ptr(values))
        else:
            gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB32F, 256, 1)
            gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RGB, GL_FLOAT, ptr(values))

    def update_table(self, name: str, values: np.ndarray) -> None:
        gl = self.gl
        values = np.ascontiguousarray(values, np.float32)
        gl.glBindTexture(GL_TEXTURE_2D, self.textures[name])
        fmt = GL_RED if values.ndim == 1 else GL_RGB
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, fmt, GL_FLOAT, ptr(values))

    def load_tile(self, tile: np.ndarray) -> None:
        gl, size = self.gl, tile.shape[0]
        data = np.ascontiguousarray(tile, np.float32)
        self._texture("tile", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16F, size, size)
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size, size, GL_RED, GL_FLOAT, ptr(data))
        self.tile_size = size

    def load_lut(self, table: np.ndarray) -> None:
        gl, n = self.gl, table.shape[0]
        data = np.ascontiguousarray(table.transpose(2, 1, 0, 3), np.float32)
        self._texture("lut", GL_TEXTURE_3D)
        gl.glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGB32F, n, n, n)
        gl.glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, n, n, n, GL_RGB, GL_FLOAT, ptr(data))
        self.n = n

    def allocate(self, width: int, height: int, three_byte: bool) -> None:
        gl = self.gl
        if max(width, height) > self.max_2d:
            raise RuntimeError(f"{width}x{height} exceeds {self.max_2d}")
        self.three_byte = three_byte
        self._texture("in", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8 if three_byte else GL_RGBA8, width, height)
        self._texture("out", GL_TEXTURE_2D)
        gl.glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height)
        fbo = c_uint()
        gl.glGenFramebuffers(1, byref(fbo))
        gl.glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, self.textures["out"], 0)
        status = gl.glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != 0x8CD5:
            raise RuntimeError(f"framebuffer incomplete: 0x{status:x}")
        gl.glViewport(0, 0, width, height)
        fmt, kind = c_int(), c_int()
        gl.glGetIntegerv(0x8B9B, byref(fmt))
        gl.glGetIntegerv(0x8B9A, byref(kind))
        print(f"  preferred read-back format 0x{fmt.value:x}, type 0x{kind.value:x}")
        self._check("allocation")

    def _check(self, where: str) -> None:
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"GL error after {where}: 0x{error:x}")

    def grade(self, rgb: np.ndarray, *, tone: bool, grain: tuple[int, int, int] | None,
              read_rgb: bool = False) -> tuple[np.ndarray, dict]:
        """One pass. ``grain`` is (oy, ox, orientation) or None. Returns the
        read-back buffer (four bytes a pixel, or three with ``read_rgb``)."""
        gl = self.gl
        height, width = rgb.shape[:2]
        times = {}
        t = time.perf_counter()
        gl.glBindTexture(GL_TEXTURE_2D, self.textures["in"])
        if self.three_byte:
            gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                               GL_RGB, GL_UNSIGNED_BYTE, ptr(np.ascontiguousarray(rgb)))
        else:
            rgba = np.empty((height, width, 4), np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = 255
            times["to_rgba"] = time.perf_counter() - t
            t = time.perf_counter()
            gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                               GL_RGBA, GL_UNSIGNED_BYTE, ptr(rgba))
        gl.glFinish()
        times["upload"] = time.perf_counter() - t
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
        oy, ox, orientation = grain or (0, 0, 0)
        for uniform, value in (("n", self.n), ("S", self.tile_size), ("orient", orientation),
                               ("use_tone", int(tone)), ("use_grain", int(grain is not None))):
            gl.glUniform1i(gl.glGetUniformLocation(program, uniform.encode()), value)
        gl.glUniform2i(gl.glGetUniformLocation(program, b"offset"), ox, oy)
        gl.glDrawArrays(0x0004, 0, 3)
        gl.glFinish()
        times["shader"] = time.perf_counter() - t
        t = time.perf_counter()
        out = np.empty((height, width, 3 if read_rgb else 4), np.uint8)
        gl.glReadPixels(0, 0, width, height, GL_RGB if read_rgb else GL_RGBA,
                        GL_UNSIGNED_BYTE, ptr(out))
        times["readback"] = time.perf_counter() - t
        self._check("grading")
        return out, times


def float_tables(gains):
    """The normalisation as float tables: ``gains_to_luts`` and ``tone_lut`` of
    ``pifilm.normalize`` without their rounding to bytes."""
    from pifilm.color import linear_to_srgb, srgb_to_linear
    from pifilm.normalize import _apply_levels_linear, _tone_curve

    levels = np.arange(256, dtype=np.float32) / 255.0
    lin = srgb_to_linear(levels)
    luma_tone = gains.levels is not None and gains.levels.get("tone", "channel") == "luma"
    chan = np.empty((256, 3), np.float32)
    for c in range(3):
        x = lin * gains.combined[c]
        if gains.levels is not None and not luma_tone:
            x = _apply_levels_linear(x, gains.levels)
        else:
            x = np.clip(x, 0.0, 1.0)
        chan[:, c] = linear_to_srgb(x)
    tone = np.clip(_tone_curve(levels, gains.levels), 0.0, 1.0) if luma_tone else levels
    return chan, np.asarray(tone, np.float32), luma_tone


def diff_report(label: str, got: np.ndarray, want: np.ndarray) -> None:
    diff = np.abs(got.astype(np.int16) - want.astype(np.int16))
    share = [float((diff > k).mean()) * 100 for k in (0, 1, 2, 4)]
    print(f"  {label}: max {int(diff.max())} levels, mean {diff.mean():.3f}; "
          f"differing by >0: {share[0]:.2f}%, >1: {share[1]:.3f}%, "
          f">2: {share[2]:.4f}%, >4: {share[3]:.4f}%")


def report(label: str, times: dict) -> None:
    parts = "  ".join(f"{k} {v * 1000:.0f}" for k, v in times.items())
    print(f"  {label}: total {sum(times.values()) * 1000:.0f} ms  ({parts})")


def timed(label: str, fn):
    t = time.perf_counter()
    result = fn()
    print(f"  {label}: {(time.perf_counter() - t) * 1000:.0f} ms")
    return result


def run(args) -> None:
    from PIL import Image

    from pifilm.artifacts import Artifacts
    from pifilm.grain import grain_tile
    from pifilm.imageio import load_rgb
    from pifilm.normalize import compute_gains
    from pifilm.pipeline import Pipeline

    artifacts = Artifacts.load(args.artifacts)
    grain = artifacts.grain
    if grain.model != "two_scale_tile":
        raise SystemExit("this spike needs an artifact with the two_scale_tile grain")
    frame, _ = load_rgb(os.path.expanduser(args.image))
    height, width = frame.shape[:2]
    print(f"frame {width}x{height}, thread {threading.current_thread().name}")
    seed = 1234

    print("CPU pipeline, same frame:")
    pipeline = Pipeline(artifacts)
    pipeline.warm((height, width))
    for _ in range(2):
        stages: dict = {}
        cpu, _ = pipeline.process(frame, rng=np.random.default_rng(seed), timings=stages)
        print("  " + "  ".join(f"{k} {v:.0f}" for k, v in stages.items())
              + f"  = {sum(stages.values()):.0f} ms")
    cpu_clean, _ = pipeline.process(frame, grain=False)

    print(f"CMA free before: {cma_free_mb():.0f} MB")
    gpu = Gpu(args.platform)
    t = time.perf_counter()
    step = max(1, int(np.ceil(np.sqrt(height * width / 300_000))))
    gains = compute_gains(frame[::step, ::step].astype(np.float32) / 255.0, artifacts.normalize)
    chan, tone, luma_tone = float_tables(gains)
    stats_ms = (time.perf_counter() - t) * 1000
    levels = np.arange(256, dtype=np.float32) / 255.0
    envelope = np.interp(levels, grain.envelope_luma, grain.envelope_rel)
    gpu.table("chan", chan)
    gpu.table("tone", tone)
    gpu.table("gain", grain.strength * envelope)
    gpu.load_lut(artifacts.lut.table)
    tile = grain_tile(grain, max(height, width) / grain.frame_width_mm)
    timed("tile upload", lambda: gpu.load_tile(tile))
    print(f"  statistics and tables on the CPU: {stats_ms:.0f} ms (paid per photograph)")

    rng = np.random.default_rng(seed)
    oy, ox = (int(v) for v in rng.integers(0, grain.tile_px, 2))
    placement = (oy, ox, int(rng.integers(0, 8)))

    for three_byte in (True, False):
        print(f"GPU pass, {'three' if three_byte else 'four'}-byte upload:")
        try:
            gpu.allocate(width, height, three_byte)
            for number in range(3):
                t = time.perf_counter()
                gpu.update_table("chan", chan)
                gpu.update_table("tone", tone)
                tables = time.perf_counter() - t
                out, times = gpu.grade(frame, tone=luma_tone, grain=placement)
                report(f"run {number + 1}", {"tables": tables, **times})
        except RuntimeError as exc:
            print(f"  failed: {exc}")
            continue
        print(f"  CMA free: {cma_free_mb():.0f} MB")
        diff_report("with grain, against the CPU", out[..., :3], cpu)
        clean, _ = gpu.grade(frame, tone=luma_tone, grain=None)
        diff_report("without grain, against the CPU", clean[..., :3], cpu_clean)
        try:
            three, times = gpu.grade(frame, tone=luma_tone, grain=placement, read_rgb=True)
            same = np.array_equal(three, out[..., :3])
            report(f"three-byte read-back (same pixels: {same})", times)
        except RuntimeError as exc:
            print(f"  three-byte read-back not available: {exc}")
            while gpu.gl.glGetError():
                pass

    print("JPEG encode (quality 95, 4:4:4, to memory):")
    rgb = np.ascontiguousarray(out[..., :3])

    def encode(image):
        image.save(io.BytesIO(), "JPEG", quality=95, subsampling=0)

    timed("four bytes to three in NumPy", lambda: np.ascontiguousarray(out[..., :3]))
    timed("from the three-byte array", lambda: encode(Image.fromarray(rgb)))
    timed("straight from the four-byte buffer",
          lambda: encode(Image.frombuffer("RGBX", (width, height), out, "raw", "RGBX", 0, 1)))
    timed("four-byte buffer through Pillow's convert",
          lambda: encode(Image.frombuffer("RGBX", (width, height), out, "raw", "RGBX", 0, 1)
                         .convert("RGB")))
    if args.save:
        Image.fromarray(rgb).save(args.save + "-gpu.png")
        Image.fromarray(cpu).save(args.save + "-cpu.png")
        print(f"saved {args.save}-gpu.png and {args.save}-cpu.png")
    print("done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--image", required=True, help="a full-resolution original JPEG")
    parser.add_argument("--platform", default="surfaceless", choices=("surfaceless", "gbm"))
    parser.add_argument("--save", help="path prefix: also write the two results as PNG")
    args = parser.parse_args()
    # On a worker thread, as the product would run it: the context is made and
    # used there, with the main thread idle.
    failure: list[BaseException] = []

    def work() -> None:
        try:
            run(args)
        except BaseException as exc:  # noqa: BLE001 - reported below
            failure.append(exc)

    thread = threading.Thread(target=work, name="spike-grade")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


if __name__ == "__main__":
    main()
