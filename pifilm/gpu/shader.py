"""The grade as one fragment shader: normalise, halation, 3D LUT, tile grain.

This is the source the Pi's GPU compiles (OpenGL ES 3.0, V3D on the Pi 4). It
is the shader of ``scripts/experiments/gpu_pass_spike.py`` (branch
``spike/gpu-pass``), character for character, but for the halation step (the
``glow``, ``use_halation``, ``hr`` and ``hw`` uniforms, the ``G`` helper and
the ``use_halation`` block, added with version 2): the spike's text compiled
and ran correctly on the target on 2026-10-13, and it cannot be compiled on a
Mac or in the test suite, so any edit here, the halation step included, is
unproven until it has run on the Pi.
``pifilm.gpu.reference`` is the same maths in NumPy, step for step and name
for name; read the two side by side, and change them together.

What the shader relies on, which the device code must provide:

* Every texture is read with ``texelFetch``, never filtered: the input bytes
  exactly, the tables and the LUT interpolated by hand in ``highp`` float. The
  GPU's own filtering has too few fractional bits for a LUT.
* ``lut`` is indexed ``(r, g, b)`` as texture ``(x, y, z)``.
* ``gl_FragCoord`` counts rows in the order the frame was uploaded, so
  ``px.y`` is the array's row and the tile lands where the CPU puts it.
* ``offset`` is ``(ox, oy)``; ``orient`` is the orientation of
  ``pifilm.grain.tile_placement``.
* ``glow`` is the halation glow of ``pifilm.halation.prepare``, an RGB float
  texture of its reduced ``(w, h)`` (one zero texel without halation, so the
  sampler always has a texture); ``hr`` is the headroom and ``hw`` the
  strength times the weights. The shader reads it bilinearly by hand at the
  pixel's centre, ``vec2(px) + 0.5``, the half-pixel mapping of the CPU's
  ``cv2.resize``.
* The vertex shader draws one triangle over the whole target from
  ``gl_VertexID`` alone: three vertices, no buffers.

``SHADER_VERSION`` is part of a photograph's record (``pipeline_backend``).
Raise it with any change to the pass that can move a pixel, here or in the
tables and textures it is given.
"""

from __future__ import annotations

SHADER_VERSION = 2
# The value of a record's ``pipeline_backend`` for a photograph this pass graded.
BACKEND = f"gpu-{SHADER_VERSION}"

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
uniform sampler2D glow;    // (w, h) RGB float: the reduced, blurred, expanded glow
uniform int use_halation;
uniform float hr;          // headroom c
uniform vec3 hw;           // strength * weights
out vec4 o;
const vec3 LUMA = vec3(0.299, 0.587, 0.114);
float table(sampler2D t, float v) {
  float x = clamp(v, 0.0, 1.0) * 255.0;
  float lo = min(floor(x), 254.0);
  int i = int(lo);
  return mix(texelFetch(t, ivec2(i, 0), 0).r, texelFetch(t, ivec2(i + 1, 0), 0).r, x - lo);
}
vec3 at(ivec3 i) { return texelFetch(lut, i, 0).rgb; }
vec3 G(ivec2 i) { return texelFetch(glow, i, 0).rgb; }
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
  if (use_halation == 1) {
    vec3 l = mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
    vec3 x = l / (1.0 - hr * l);
    ivec2 gs = textureSize(glow, 0);
    vec2 u = (vec2(px) + 0.5) * vec2(gs) / vec2(textureSize(img, 0)) - 0.5;
    u = clamp(u, vec2(0.0), vec2(gs - 1));
    ivec2 i0 = ivec2(floor(u));
    ivec2 i1 = min(i0 + 1, gs - 1);
    vec2 t = u - vec2(i0);
    vec3 B = mix(mix(G(i0), G(ivec2(i1.x, i0.y)), t.x),
                 mix(G(ivec2(i0.x, i1.y)), G(i1), t.x), t.y);
    x = x + hw * (B - x);
    l = x / (1.0 + hr * x);
    c = clamp(mix(l * 12.92, 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055,
                  step(0.0031308, l)), 0.0, 1.0);
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
