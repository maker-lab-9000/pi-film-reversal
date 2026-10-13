# GPU grading: design

Date: 2026-10-13. Step 7 of `docs/superpowers/plans/2026-10-13-pipeline-improvements-roadmap.md`.
Branch `feat/gpu-grade`, pull request into `pipeline-improvements`.

## Goal

The background grade of a photograph (normalise, 3D LUT, grain) runs as one pass
on the Pi 4's GPU instead of three passes on the CPU. The CPU path stays as the
reference, the fallback, and what `pifilm-process` and the Mac use.

## What was measured (Pi 4, IMX477, 12 MP, 2026-10-13, `scripts/experiments/gpu_pass_spike.py`)

| | CPU | GPU, one pass |
| --- | --- | --- |
| Statistics and tables | in normalise | 102 ms (stays on the CPU) |
| Normalise + LUT + grain | 1,728 ms | upload 142, shader 239, read-back 180 = 562 ms |
| Total | 1,728 ms | about 665 ms |

- A three-byte upload works, so the frame is not repacked. A three-byte read-back
  does not (GL error); the result comes back as four bytes a pixel.
- The JPEG encoder takes the four-byte buffer directly (371 ms against 386 ms from
  three bytes), so nothing is unpacked either.
- Against the CPU's result on a real photograph: without grain at most 4 levels,
  0.23 on average; with grain at most 5, 0.58 on average. The grain lands in the
  same place for the same seed.
- It ran on a worker thread, and free graphics memory did not move (493 MB).

So a grade goes from about 1.7 s to about 0.7 s. Grading is already behind the
shutter, so this does not change the wait between photographs (1.8 to 2.0 s). It
shortens the background job, about 3.3 to 4 s today, by about a second: the graded
file exists sooner and a burst drains faster.

Not measured: the GPU pass while this process also owns the display and the live
view is running. The spike ran with the service stopped. That is the first
hardware check.

## What runs where

- **GPU:** the grade worker's full-resolution grade, when all of these hold: a GPU
  grader was set up, the artifact's grain is `two_scale_tile` (or grain is off),
  the frame fits the largest texture (4096; the IMX477's 4056 fits), the EV is 0,
  and the caller is the thread that owns the graphics context.
- **CPU, as now:** everything else. The quick grade for the screen, `--grade
  inline`, a shot the closing queue refused (graded on the capture thread),
  `pifilm-process`, the trainer, artifacts with the Gaussian or `two_scale` grain,
  frames wider than 4096 (the IMX708), the double exposure's composite (1.4 s; a
  candidate for later).

## The pass

Statistics stay on the CPU and are the same code as today (`compute_gains` on the
same subsample), so the audit fields of a record (`wb_gains`, `exposure_gain`,
`levels`, `clamped`) are identical whichever path graded.

The shader, per pixel, all in `highp` float:

1. Read the input bytes exactly (`texelFetch`).
2. Per-channel normalisation: a 256-entry float table per channel (white balance
   and exposure, or the per-channel tone curve), indexed by the byte.
3. Luma tone, when the artifact uses it: `Y = 0.299R + 0.587G + 0.114B`, the tone
   curve read from a 256-entry float table with linear interpolation, and the
   change in Y added to all three channels (what the CPU does through YCrCb).
4. The 3D LUT: eight `texelFetch`es and trilinear interpolation written out in
   the shader. Never the GPU's own filtering. Same interpolation as the CPU.
5. Grain: the same tile as the CPU (`pifilm.grain.grain_tile`), placed by the
   same offset and orientation drawn from the shot's seed, scaled by the
   envelope at the graded pixel's luminance, added to all three channels.
6. Clamp and write. One rounding to bytes, by the GPU, at the end.

Textures: the frame as three bytes a pixel; tables and the LUT in 32-bit float;
the tile in 16-bit float. No dithering: the grain already breaks up banding, and
a second noise would be a look change.

This is more precise than the CPU path, which rounds to bytes after normalising,
after the tone curve, after the LUT and inside the grain. That is where the
differences above come from.

## Agreement with the CPU, and how it is kept honest

The GPU code cannot run on the Mac or in the test suite. Three things stand in:

1. **A NumPy reference of the shader** (`pifilm/gpu/reference.py`): the same
   steps in float, written to be read beside the shader. Tested on the Mac
   against the CPU pipeline with the tolerances measured above (without grain:
   at most 6 levels and under 0.5 on average; with grain: at most 8 and under 1.0).
2. **A self-test on the Pi at start-up:** the grader grades a small fixed frame
   and compares it with the NumPy reference. More than 1 level anywhere, or any
   graphics error, and GPU grading is switched off for this run with one log
   line. A driver update that changes the output is caught the day it happens.
3. **A check command** (`pifilm-gpu-check --artifacts DIR [--image FILE]`): the
   same comparison on a full frame, with timings. This is the hardware acceptance
   tool and replaces the spike script.

## Failures

- Set-up fails (no GPU, no driver, the self-test fails): one log line, CPU for the
  run. The service starts either way.
- A grade fails on the GPU: that photograph is graded on the CPU at once, so it is
  never lost and never waits. Three GPU failures in a row switch the GPU off for
  the run.
- The fallback is silent on the screen and explicit in the log and in the record.

## Records and reproducibility

- New field `pipeline_backend`: `"gpu-1"` or `"cpu"`, in every record written by a
  build that has this feature. The number is the shader's version; any change to
  the pass that can move pixels raises it.
- `pifilm-process` regrades on the CPU. For a record marked `gpu-1` the result
  matches the Pi's file within the tolerances above, not byte for byte; the tool
  says so once when it meets such a record. Records graded on the CPU still
  reproduce exactly. This is the accepted cost named in the roadmap.

## Switch

`--grade-backend auto|cpu`, default `auto` (the GPU when it can be used). `cpu`
is today's behaviour exactly.

## Code

- `pifilm/gpu/egl.py`: the windowless EGL context and the handful of GL calls,
  through `ctypes` on the system's `libEGL` and `libGLESv2`. No new dependency.
  Imported lazily, like the other hardware modules.
- `pifilm/gpu/shader.py`: the shader source and `SHADER_VERSION`.
- `pifilm/gpu/reference.py`: the NumPy reference and the float tables (shared by
  the grader and the tests).
- `pifilm/gpu/grader.py`: `GpuGrader` (set-up, self-test, `grade(rgb, gains,
  placement) -> four-byte image`, `close`). Owns its textures; allocated once for
  the camera's still size and reused.
- `pifilm/gpu/check.py`: the check command.
- `pifilm/pipeline.py`: `Pipeline.process` gains an optional `grader`; with one,
  and when the conditions above hold, the pass runs on it. The audit dict gains
  nothing; the backend used is reported through the `timings`-style out-parameter.
- `pifilm/grain.py`: the tile placement (offset and orientation from the seed) is
  factored out so the CPU and the GPU draw it the same way.
- `pifilm/imageio.py`: `save_jpeg` accepts a four-byte image.
- `pifilm/capture/app.py`: the grade worker creates the grader on its own thread
  before its first job; `--grade-backend`; `pipeline_backend` in records.
- Docs: `docs/how-it-works.md`, `CLAUDE.md`, `README.md` (draft to the owner
  first), the roadmap.

## Memory

Input and output textures for a 12 MP frame, the tile and the tables: about
110 MB of graphics memory, held while the service runs. The spike showed no drop
in the pool the camera and display share. The Pi has 4 GB.

## Risks to check on the hardware

- **Beside the display.** The app owns the screen through KMS in the same
  process. The GPU context is windowless and on the render device, so they
  should not meet; not yet tried together.
- **Live view while the GPU grades.** The CPU is freer than today during a grade,
  but the GPU and the display share memory bandwidth.
- **The camera stall** (`docs/known-issues.md`) must not become more frequent.

## Acceptance (hardware)

1. Service starts with the display on; the log says GPU grading is on and the
   self-test passed.
2. `pifilm-gpu-check` on a real photograph: within 1 level of the reference;
   timings near the spike's.
3. Ten photographs: records say `pipeline_backend: gpu-1`; `stage_ms` shows the
   grade near 0.7 s; graded files look right at 100 %, grain included.
4. A double exposure and a Stick shot work as before.
5. `--grade-backend cpu`: records say `cpu`; timings as before.
6. `pifilm-process` on today's folder: runs, and notes the GPU records.
7. Ten minutes of ordinary use: no preview stall, live view no worse than now.

## Not in this step

- Tetrahedral interpolation, dithering: look changes, to be judged by gallery.
- A graded live view from the same shader.
- The double exposure's composite on the GPU.
- Tiling for frames wider than 4096.
