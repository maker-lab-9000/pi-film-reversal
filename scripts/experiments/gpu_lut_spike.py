"""Throwaway hardware spike: a 3D LUT on the Pi 4's GPU (step 2 of the
pipeline-improvements roadmap). Not product code; nothing imports it.

Run on the Pi with the capture service stopped:

    .venv/bin/python scripts/experiments/gpu_lut_spike.py \
        --artifacts artifacts/imx477-2026-09-v4-dark
    .venv/bin/python scripts/experiments/gpu_lut_spike.py \
        --artifacts artifacts/imx477-2026-09-v4-dark --camera

It uses plain ctypes on libEGL and libGLESv2 (no Python GL binding needed),
a windowless context, texelFetch with the interpolation done in highp float,
and a half-float LUT texture.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_void_p

import numpy as np

EGL_NONE = 0x3038
GL_TEXTURE_2D, GL_TEXTURE_3D = 0x0DE1, 0x806F
GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, GL_FLOAT = 0x1908, 0x1907, 0x1401, 0x1406
GL_RGBA8, GL_RGB16F = 0x8058, 0x881B
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
precision highp sampler2D;
precision highp sampler3D;
uniform sampler2D img;
uniform sampler3D lut;
uniform int n;
out vec4 o;
vec3 at(ivec3 i) { return texelFetch(lut, i, 0).rgb; }
void main() {
  vec3 c = texelFetch(img, ivec2(gl_FragCoord.xy), 0).rgb * float(n - 1);
  vec3 lo = min(floor(c), float(n - 2));
  ivec3 i = ivec3(lo);
  vec3 f = c - lo;
  vec3 c00 = mix(at(i), at(i + ivec3(1, 0, 0)), f.r);
  vec3 c10 = mix(at(i + ivec3(0, 1, 0)), at(i + ivec3(1, 1, 0)), f.r);
  vec3 c01 = mix(at(i + ivec3(0, 0, 1)), at(i + ivec3(1, 0, 1)), f.r);
  vec3 c11 = mix(at(i + ivec3(0, 1, 1)), at(i + ivec3(1, 1, 1)), f.r);
  o = vec4(mix(mix(c00, c10, f.g), mix(c01, c11, f.g), f.b), 1.0);
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

    def load_lut(self, table: np.ndarray) -> None:
        gl, n = self.gl, table.shape[0]
        data = np.ascontiguousarray(table.transpose(2, 1, 0, 3), np.float32)
        self._texture("lut", GL_TEXTURE_3D)
        gl.glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGB16F, n, n, n)
        gl.glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, n, n, n,
                           GL_RGB, GL_FLOAT, ptr(data))
        self.n = n

    def allocate(self, width: int, height: int) -> None:
        gl = self.gl
        if max(width, height) > self.max_2d:
            raise RuntimeError(f"{width}x{height} exceeds {self.max_2d}")
        for name in ("in", "out"):
            self._texture(name, GL_TEXTURE_2D)
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
        self.size = (width, height)
        error = gl.glGetError()
        if error:
            raise RuntimeError(f"GL error after allocation: 0x{error:x}")

    def grade(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        gl = self.gl
        height, width = rgb.shape[:2]
        times = {}
        t = time.perf_counter()
        rgba = np.empty((height, width, 4), np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        times["to_rgba"] = time.perf_counter() - t
        t = time.perf_counter()
        gl.glBindTexture(GL_TEXTURE_2D, self.textures["in"])
        gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                           GL_RGBA, GL_UNSIGNED_BYTE, ptr(rgba))
        gl.glFinish()
        times["upload"] = time.perf_counter() - t
        t = time.perf_counter()
        gl.glUseProgram(self.program)
        gl.glActiveTexture(0x84C0)
        gl.glBindTexture(GL_TEXTURE_2D, self.textures["in"])
        gl.glActiveTexture(0x84C1)
        gl.glBindTexture(GL_TEXTURE_3D, self.textures["lut"])
        gl.glUniform1i(gl.glGetUniformLocation(self.program, b"img"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, b"lut"), 1)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, b"n"), self.n)
        gl.glDrawArrays(0x0004, 0, 3)
        gl.glFinish()
        times["shader"] = time.perf_counter() - t
        t = time.perf_counter()
        out = np.empty((height, width, 4), np.uint8)
        gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, ptr(out))
        times["readback"] = time.perf_counter() - t
        t = time.perf_counter()
        result = np.ascontiguousarray(out[..., :3])
        times["to_rgb"] = time.perf_counter() - t
        error = gl.glGetError()
        if error:
            raise RuntimeError(f"GL error after grading: 0x{error:x}")
        return result, times


def synthetic(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
    mix = np.array([0.7, 0.2, 0.5], np.float32)
    frame = x * mix + y * (1 - mix) + rng.normal(0, 6, (height, width, 3))
    return np.clip(frame, 0, 255).astype(np.uint8)


def compare(lut, rgb: np.ndarray, gpu_out: np.ndarray) -> None:
    sample = (slice(None, None, 8), slice(None, None, 8))
    ref = lut.apply_numpy(rgb[sample].astype(np.float32) / 255.0)
    ref = np.clip(np.round(ref * 255.0), 0, 255).astype(np.int16)
    diff = np.abs(gpu_out[sample].astype(np.int16) - ref)
    share = [float((diff > k).mean()) * 100 for k in (0, 1, 2)]
    print(f"  against the CPU reference: max {int(diff.max())} levels, "
          f"mean {diff.mean():.3f}; differing by >0: {share[0]:.2f}%, "
          f">1: {share[1]:.3f}%, >2: {share[2]:.3f}%")


def report(label: str, times: dict) -> None:
    parts = "  ".join(f"{k} {v * 1000:.0f}" for k, v in times.items())
    print(f"  {label}: total {sum(times.values()) * 1000:.0f} ms  ({parts})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--platform", default="surfaceless",
                        choices=("surfaceless", "gbm"))
    parser.add_argument("--size", default="4056x3040")
    parser.add_argument("--camera", action="store_true",
                        help="also take a real still with the textures loaded")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split("x"))

    from pifilm.artifacts import Artifacts
    lut = Artifacts.load(args.artifacts).lut
    print(f"CMA free before: {cma_free_mb():.0f} MB")
    gpu = Gpu(args.platform)
    gpu.load_lut(lut.table)
    gpu.allocate(width, height)
    print(f"CMA free with {width}x{height} textures: {cma_free_mb():.0f} MB")

    frame = synthetic(width, height)
    for run in range(3):
        out, times = gpu.grade(frame)
        report(f"GPU run {run + 1}", times)
    compare(lut, frame, out)
    t = time.perf_counter()
    lut.apply_pillow(frame)
    print(f"  CPU (Pillow) LUT, same frame: "
          f"{(time.perf_counter() - t) * 1000:.0f} ms")

    if args.camera:
        from pifilm.capture.picamera import Picamera2Camera
        print("Opening the camera with the textures still loaded...")
        camera = Picamera2Camera(tuning_file="imx477.json",
                                 ae_constraint="highlight", preview=True,
                                 save_dng=False)
        try:
            for shot in range(2):
                still = camera.read(full=True)
                print(f"  still {shot + 1}: {still.rgb.shape}, "
                      f"camera stages {getattr(still, 'timings', None)}, "
                      f"CMA free {cma_free_mb():.0f} MB")
                if still.rgb.shape[:2] == (height, width):
                    out, times = gpu.grade(still.rgb)
                    report("GPU on the still", times)
                    compare(lut, still.rgb, out)
            camera.read(full=False)
            print("  preview frame after the stills: ok")
        finally:
            camera.close()
    print("done")


if __name__ == "__main__":
    main()
