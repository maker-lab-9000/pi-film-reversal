# GPU Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** The background grade of a photograph runs as one pass on the Pi 4's GPU, with the CPU path as reference and fallback.

**Architecture:** A NumPy reference of the pass (testable on a Mac) defines the maths. A `GpuGrader` drives a narrow `Device` interface; the real device is `ctypes` on EGL and GLES 3, a fake one built on the reference stands in for it in tests. `Pipeline.process` uses a grader when given one and falls back to the CPU on any GPU error. The grade worker owns the grader.

**Tech Stack:** Python 3.12, NumPy, Pillow, OpenCV via `require_cv2()`, `ctypes` on `libEGL.so.1` and `libGLESv2.so.2` (Pi only, imported lazily), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-13-gpu-grading-design.md` (approved 2026-10-13). Read it first; it is the authority.

**Proven GL code to port, not to reinvent:** `scripts/experiments/gpu_pass_spike.py` on branch `spike/gpu-pass` (read it with `git show origin/spike/gpu-pass:scripts/experiments/gpu_pass_spike.py`). It ran correctly on the target Pi on 2026-10-13: EGL set-up, the shader, texture formats, the three-byte upload, the four-byte read-back. The GL code cannot be run on this Mac, so every departure from the spike's calls is a risk; keep its call sequence and constants.

## Global Constraints

- Always `.venv/bin/pytest`, `.venv/bin/ruff`. Suite: `.venv/bin/pytest -q -m 'not slow'` (about 90 s; allow 300 s; macOS has no `timeout` command). Lint: `.venv/bin/ruff check .`. Both pass before every commit.
- Never touch, stage or delete the untracked directory named `" "` (one space) at the repo root. Stage by explicit path only.
- Never read, print or source `.env`.
- Never `import cv2` directly; use `require_cv2()`. Pixels enter and leave through `pifilm.imageio`.
- The package must import, and the whole suite must pass, on a Mac with no EGL: nothing under `pifilm/` may load `libEGL`/`libGLESv2` at import time. Only `pifilm/gpu/egl.py` touches them, inside functions.
- With `--grade-backend cpu`, with `--grade inline`, in `pifilm-process` and in the trainer, graded pixels are byte-identical to before this plan. Existing tests pass unchanged unless a task says otherwise (the one allowed change: an expected record gaining `pipeline_backend`).
- A photograph is never lost or left ungraded because of the GPU: every GPU failure ends in a CPU grade of that same photograph.
- Match the surrounding code: module docstrings carry the reasons, comments are plain sentences saying why, no em dashes in prose.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Work on branch `feat/gpu-grade`. Do not push, open pull requests or switch branches. Do not edit `README.md`.

## Review Focus

- A GPU error at set-up, in the self-test, or in the middle of a grade must end in a CPU grade and a log line, never an exception out of the grade worker and never a hang.
- The grader used from a thread other than the one that created it must not touch GL (it must report that it cannot grade).
- A frame of a different size from the last one (a different sensor mode, the self-test's small frame) must be graded correctly, not with stale textures.
- An artifact the pass does not cover (Gaussian or `two_scale` grain, EV not 0, a frame wider than the largest texture) must go to the CPU without a log line per photograph.
- The seed must place the grain identically on both paths: the generator draws must be the same calls in the same order.

---

### Task 1: the reference pass, the shader source, shared grain placement, four-byte JPEG

**Files:**
- Create: `pifilm/gpu/__init__.py` (docstring only), `pifilm/gpu/reference.py`, `pifilm/gpu/shader.py`
- Modify: `pifilm/grain.py`, `pifilm/normalize.py`, `pifilm/imageio.py`
- Test: `tests/test_gpu_reference.py`, additions to `tests/test_grain.py`, `tests/test_normalize.py`, `tests/test_imageio.py` (use the existing files for those modules, whatever their exact names)

**Produces (later tasks use these exactly):**

```python
# pifilm/grain.py
def tile_placement(params: GrainParams, rng: np.random.Generator) -> tuple[int, int, int]:
    """(oy, ox, orientation) for one shot: the same three draws, in the same
    order, that add_grain makes for the two_scale_tile model."""
# add_grain's tile branch must call it, and its output must stay byte-identical.

# pifilm/normalize.py
def stats_gains(rgb_u8: np.ndarray, params: NormalizeParams,
                max_stats_pixels: int = 300_000) -> Gains:
    """The gains normalize_u8 computes, from the same strided subsample."""
# normalize_u8 must call it; its output stays byte-identical.

# pifilm/imageio.py
# save_jpeg accepts an (H, W, 4) uint8 array as well: bytes 0..2 are RGB, the
# fourth is ignored (Pillow mode "RGBX", no copy of the pixels). The JPEG is
# byte-identical to saving the same pixels as (H, W, 3).

# pifilm/gpu/shader.py
SHADER_VERSION = 1
BACKEND = f"gpu-{SHADER_VERSION}"      # the record's pipeline_backend value
VERTEX: bytes
FRAGMENT: bytes                         # the spike's shader, verbatim unless a comment says why

# pifilm/gpu/reference.py
@dataclass(frozen=True)
class PassTables:
    chan: np.ndarray      # (256, 3) float32: per-channel normalisation
    tone: np.ndarray      # (256,) float32: luma tone curve (identity when unused)
    use_tone: bool
    gain: np.ndarray      # (256,) float32: grain strength by luminance, 0..1 units; zeros when grain is off

def pass_tables(gains: Gains, grain: GrainParams) -> PassTables: ...
def reference_pass(rgb_u8: np.ndarray, tables: PassTables, lut: np.ndarray,
                   tile: np.ndarray | None,
                   placement: tuple[int, int, int] | None) -> np.ndarray:
    """What the shader computes, in NumPy float32: returns (H, W, 3) uint8.
    lut is LUT3D.table; tile and placement None means no grain."""
def supports(grain: GrainParams) -> bool:
    """True when the pass can reproduce this artifact's grain: grain disabled or
    strength 0, or the two_scale_tile model."""
```

**Behaviour:**

- `pass_tables` and `reference_pass` are the spike's `float_tables` and the NumPy emulation of the shader: per-channel table indexed by the byte; when `use_tone`, `y = dot(c, (0.299, 0.587, 0.114))`, `c = clip(c + (table(tone, y) - y), 0, 1)`; trilinear LUT (the same as `LUT3D.apply_numpy`), clamped to 0..1; when grain, `g += table(gain, dot(g, luma)) * tile[row, col]` with `row = (r + oy) % S`, `col = (c + ox) % S`, then if `orientation & 4` swap row and col, if `& 1` `row = S-1-row`, if `& 2` `col = S-1-col`; clamp; `rint(x * 255)`. `table(t, v)` is linear interpolation between the 256 entries at `clip(v, 0, 1) * 255`.
- `gain` is `strength * interp(levels, envelope_luma, envelope_rel)`.
- Write `reference_pass` to be read beside the shader: same step order, same names. Process in row bands if needed to keep peak memory reasonable on a 12 MP frame (it is used by the Pi check command on full frames).
- The module docstring of `reference.py` says why it exists (the GPU code cannot run in the test suite) and where its results differ from the CPU path (the CPU rounds to bytes after normalising, after the tone curve, after the LUT and inside the grain).

**Tests:**

1. `tile_placement`: for several seeds, equals what `add_grain` used (compare `add_grain` output with a hand-placed tile, as the existing tile test does), and `add_grain` output for `two_scale_tile` is byte-identical to before (keep an inline copy of the old three draws in the test).
2. `stats_gains` equals the gains inside `normalize_u8` (same `combined`, same `levels`); `normalize_u8` output unchanged.
3. `save_jpeg` with a four-byte array writes the same bytes as with the three-byte array of the same pixels, whatever the fourth byte holds; a two-channel array still raises.
4. `reference_pass` against the CPU pipeline (`Pipeline.process`) on at least three different frames (use `pifilm.capture.camera.synthetic_frame` and other varied synthetic frames with gradients, saturated colours, near-black and near-white areas, at about 600 x 400), with an artifact made from `Artifacts.default()` and a `two_scale_tile` grain of strength 0.025, tile_px 256: without grain, mean absolute difference below 0.5 levels and the 99.9th percentile at most 4; with grain (same seed on both sides, placement from `tile_placement`), mean below 1.0 and the 99.9th percentile at most 5. The maximum is checked loosely (at most 16): on real photographs a laptop emulation saw up to 10 levels on 0.06 % of pixels. Find out where the largest differences come from (expected: saturated pixels, where the CPU's byte YCrCb round trip clips) and say so in the reference's docstring.
5. With both luma tone and per-channel tone artifacts (`levels_tone` "luma" and "channel" in `NormalizeParams`), and with `levels` off.
6. `orientation` and offset: for each of the 8 orientations, `reference_pass` on a flat mid-grey frame equals a direct computation with `pifilm.grain._oriented` and `place_tile`.
7. Grain off (`tile=None`): a frame of pure black and pure white is unchanged by an identity LUT and identity tables.
8. `supports`: true for disabled grain, zero strength and `two_scale_tile`; false for `gaussian` and `two_scale` with positive strength.
9. `FRAGMENT` and `VERTEX` are bytes starting with `#version 300 es`; `BACKEND == "gpu-1"`.

Commits: small, one per concern.

---

### Task 2: the device, the grader, the check command

**Files:**
- Create: `pifilm/gpu/egl.py`, `pifilm/gpu/grader.py`, `pifilm/gpu/check.py`
- Modify: `pyproject.toml` (`pifilm-gpu-check = "pifilm.gpu.check:main"` under `[project.scripts]`)
- Test: `tests/test_gpu_grader.py`, `tests/test_gpu_check.py`

**Consumes:** Task 1's `PassTables`, `pass_tables`, `reference_pass`, `supports`, `VERTEX`, `FRAGMENT`, `BACKEND`, `stats_gains`, `tile_placement`, `grain_tile`.

**Produces:**

```python
# pifilm/gpu/grader.py
class GpuError(Exception): ...

class Device(Protocol):
    max_texture: int
    def load_tables(self, tables: PassTables) -> None: ...     # per photograph; cheap
    def load_lut(self, table: np.ndarray) -> None: ...         # once
    def load_tile(self, tile: np.ndarray | None) -> None: ...  # once per tile; None: no grain
    def run(self, rgb_u8: np.ndarray, *, use_tone: bool,
            placement: tuple[int, int, int] | None,
            timings: dict | None = None) -> np.ndarray: ...    # (H, W, 4) uint8; raises GpuError
    def close(self) -> None: ...

class GpuGrader:
    MAX_FAILURES = 3
    def __init__(self, artifacts: Artifacts, device: Device, *,
                 log: Callable[[str], None] = print) -> None: ...   # loads the LUT, runs the self-test; raises GpuError
    @classmethod
    def open(cls, artifacts: Artifacts, *, log=print,
             device_factory: Callable[[], Device] | None = None) -> GpuGrader | None: ...
    @property
    def enabled(self) -> bool: ...
    def can_grade(self, shape: tuple[int, ...], *, ev: float, grain: bool) -> bool: ...
    def grade(self, rgb_u8: np.ndarray, gains: Gains, rng: np.random.Generator | None,
              *, grain: bool = True, timings: dict | None = None) -> np.ndarray: ...
    def close(self) -> None: ...

# pifilm/gpu/egl.py
def open_device() -> Device: ...     # raises GpuError; the only code that loads libEGL / libGLESv2
```

**Behaviour:**

- `egl.py` is the spike's `Gpu` class turned into a `Device`: surfaceless EGL platform, an ES 3 context, the program built from `shader.VERTEX`/`FRAGMENT`, RGB8 input texture with a three-byte upload, RGBA8 output texture on a framebuffer, 32-bit float tables and LUT, 16-bit float tile, `glFinish` after upload and draw, `glReadPixels` as RGBA. Textures for the frame are (re)allocated when the frame's size differs from the last one; old ones deleted. Every GL or EGL failure, a missing library, a shader that does not compile, a GL error after a call sequence: `GpuError` with the code in the message. `timings`, when given, gets `gpu_upload`, `gpu_shader`, `gpu_readback` in milliseconds (0.1 precision). `close()` releases the context and is safe to call twice. Set `EGL_PLATFORM=surfaceless` only around display creation and restore the previous value of the environment variable. Keep the module small and close to the spike; it cannot be run here, so add no cleverness. Put ctypes `argtypes`/`restype` on every function that takes or returns a pointer (as the spike does).
- `GpuGrader.__init__`: remembers the creating thread (`threading.get_ident()`), loads the LUT, builds and loads the tile lazily per frame scale (`grain_tile(grain, max(h, w) / grain.frame_width_mm)`, reloaded only when the scale changes), then runs the **self-test**: a fixed 96 x 64 synthetic frame (deterministic, no random seed from the clock), gains from `stats_gains`, a fixed placement, graded on the device and compared with `reference_pass`; any difference above 1 level, or a `GpuError`, raises `GpuError("self-test: ...")` after closing the device. When `supports(artifacts.grain)` is false the constructor raises `GpuError` naming the grain model (the caller logs once and uses the CPU).
- `GpuGrader.open`: never raises. Builds the device (`device_factory or egl.open_device`), then the grader; on `GpuError` (or any `OSError`/`AttributeError` from ctypes) logs one line `gpu grading off: <reason>` and returns None. On success logs one line `gpu grading on (<renderer string if the device exposes one>, self-test passed)`.
- `can_grade`: True only when enabled, the calling thread is the creating thread, `ev == 0.0`, `max(shape[:2]) <= device.max_texture`, and the frame is (H, W, 3). Never logs, never raises.
- `grade`: `pass_tables(gains, grain)` (zero gain table and no tile when `grain=False`), `tile_placement(artifacts.grain, rng)` exactly when the CPU's `add_grain` would draw (grain on, enabled, strength above 0), device run. Returns the four-byte image. On `GpuError`: counts the failure, at `MAX_FAILURES` in a row disables itself and logs `gpu grading off: 3 failures in a row`, and re-raises. A success resets the count. `timings` gets the device's three numbers.
- `close`: closes the device; afterwards `enabled` is False. Safe twice.
- `check.py` (`pifilm-gpu-check --artifacts DIR [--image FILE] [--runs N]`): opens a grader, grades the image (or a synthetic 4056 x 3040 frame) N times (default 3), prints per-run timings, the CPU pipeline's time for the same frame, the difference from `reference_pass` (max, mean, share above 0/1/2 levels) and from the CPU pipeline, and exits 0 when the difference from the reference is at most 1 level, 1 otherwise or when the GPU cannot be opened. Printing through an injectable `out` so it can be tested with a fake device.

**Tests (a `FakeDevice` built on `reference_pass`, with switches to fail on set-up, on the Nth run, or to return pixels off by k levels):**

1. With the fake device, `grade` returns four-byte pixels whose RGB equals `reference_pass` with the placement the same seed gives `add_grain`.
2. Self-test: a device off by 2 levels makes `__init__` raise `GpuError` and closes the device; off by 1 passes.
3. `open`: returns None and logs exactly one `gpu grading off:` line when the factory raises `GpuError`, `OSError`, or the self-test fails; returns a grader and logs one `gpu grading on` line otherwise. On this Mac, `GpuGrader.open(artifacts)` with the real factory returns None without raising (no EGL here).
4. Unsupported grain (`gaussian`, strength above 0): `open` returns None with the model named in the log.
5. `can_grade`: false from another thread, for `ev=0.5`, for a frame wider than `max_texture`, for a two-dimensional frame, after `close`, after being disabled; true otherwise.
6. Failures: one failing run re-raises and leaves it enabled; a success in between resets the count; three in a row disable it and log once.
7. Frame size change: a 96 x 64 frame then a 200 x 120 frame then 96 x 64 again all grade correctly (the fake device records the sizes it was asked for).
8. Tile reuse: the tile is loaded once for two frames of the same size and again when the scale changes; `grain=False` uses no tile and a zero gain table, and draws nothing from the generator.
9. `timings` receives the device's keys.
10. `check`: with a fake device, prints timings and differences and returns 0; with a device off by 2 levels returns 1; with no GPU returns 1 and says so.
11. Importing `pifilm.gpu.egl` on this Mac does not raise (it loads nothing until `open_device` is called), and `open_device()` here raises `GpuError`.

Commits: device, grader, check command.

---

### Task 3: the pipeline and the capture session use the grader

**Files:**
- Modify: `pifilm/pipeline.py`, `pifilm/capture/gradequeue.py`, `pifilm/capture/app.py`, `pifilm/capture/batch.py`
- Modify docs: `docs/how-it-works.md`, `CLAUDE.md`, `docs/lcd-viewfinder.md` (hardware acceptance table from the spec), the roadmap's step 7 ("Built" note). Not `README.md`.
- Test: `tests/test_pipeline.py`, `tests/test_gradequeue.py`, `tests/test_app.py`, the batch tests' file

**Consumes:** Task 2's `GpuGrader` (`open`, `can_grade`, `grade`, `close`, `enabled`), `GpuError`; Task 1's `stats_gains`, `BACKEND`.

**Behaviour:**

- `Pipeline.process(..., grader=None, used: dict | None = None)`: when `grader` is given and `grader.can_grade(rgb_u8.shape, ev=ev, grain=<whether grain will run>)`: compute `gains = stats_gains(rgb_u8, params)` (stage `normalize` in `timings`), call `grader.grade(...)` (stage `gpu`, plus the grader's three sub-stages), and return the four-byte image with the same audit dict the CPU path returns for those gains. On `GpuError`, fall through to the CPU path for the same frame (the generator must not have been advanced in a way that changes the CPU result: draw the placement only inside `grade`, and on failure rebuild the generator from the caller's seed; since `process` receives a generator, not a seed, take a copy of its state before the GPU attempt and restore it before the CPU attempt, and test that a failed GPU attempt gives the byte-identical CPU result). `used`, when given, gets `used["backend"]` = `BACKEND` or `"cpu"`. Without a grader the function is exactly as before. The docstring states that the result is (H, W, 3), or (H, W, 4) with an unused fourth byte when the GPU graded.
- `GradeQueue(..., on_start: Callable[[], None] | None = None, on_stop: Callable[[], None] | None = None)`: `on_start` runs on the worker thread before the first job (and before the backlog), `on_stop` on the worker thread after the last, each wrapped so an exception is logged and never kills the worker.
- `CaptureSession(..., grade_backend: str = "cpu")` (`"auto"` or `"cpu"`): with `"auto"` and background grading, the queue's `on_start` opens `GpuGrader.open(pipeline.artifacts, log=self._log)` on the worker thread and `on_stop` closes it. `_run_grade` passes the grader to `Pipeline.process` (the capture thread's refused-submit path passes it too; `can_grade` refuses there by thread). `save_jpeg` gets whatever `process` returned. Inline sessions and `"cpu"` never open a grader.
- Records: every record that has `params_version` gains `pipeline_backend` (`"gpu-1"` or `"cpu"`), background and inline, as the last of the new fields (after `grade` for background records, after `stage_ms` for inline ones, before the closing `camera_metadata`/`dng`/`double` fields, following how the existing code orders new fields). Exposure 1 of a double has no grade and no such field. A recovered grade records what actually graded it.
- `main`: `--grade-backend {auto,cpu}`, default `auto`, help in the file's style, passed to the session.
- `pifilm-process` (`batch.py`): when a folder's log has a record with `pipeline_backend` starting `gpu`, print once per run: the Pi graded these on its GPU, so the regraded files match within a few levels, not byte for byte. Nothing else changes.
- Docs: describe what runs where, the self-test, the fallback, `pipeline_backend`, `--grade-backend`, `pifilm-gpu-check`, and the reproducibility note; the seven acceptance steps as a table with empty results. Plain short sentences, no em dashes.

**Tests (fake grader objects; no GL):**

1. `process` with a fake grader that can grade: returns its image, `used["backend"] == "gpu-1"`, the audit dict equals the CPU path's audit dict for the same frame, `timings` has `normalize` and `gpu` and no `lut`/`grain`.
2. `process` with a grader whose `can_grade` is False: byte-identical to no grader, `used["backend"] == "cpu"`, `grade` never called.
3. `process` when `grade` raises `GpuError`: the result is byte-identical to the CPU result for the same seed (generator state restored), `used["backend"] == "cpu"`.
4. `process` without `grader`/`used`: unchanged (existing tests).
5. `GradeQueue`: `on_start` runs on the worker thread before any job, `on_stop` after the last and after `close`; an `on_start` that raises is logged and jobs still run.
6. Session, `grade_backend="auto"` with `GpuGrader.open` monkeypatched to a fake: opened once, on the `pifilm-grade` thread; the record says `gpu-1`; the graded file is the fake's pixels; `close_grades()` closes it.
7. Session with `open` returning None: records say `cpu`, graded file byte-identical to today's.
8. Session where the fake grader fails on one photograph: that photograph's graded file exists and is byte-identical to the CPU grade, its record says `cpu`; the next says `gpu-1`.
9. `grade_backend="cpu"` and inline sessions never call `GpuGrader.open`; their records say `cpu`.
10. The refused-submit path (closing queue) grades on the CPU and records `cpu`.
11. A recovered grade with a grader records `gpu-1`.
12. CLI: `--grade-backend` default and `cpu` reach the session.
13. Batch: the note is printed once for a log with a `gpu-1` record and not for a CPU-only log; output files unchanged.

Commits: pipeline, queue hooks, session and CLI, batch, docs.
