# Double Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A touch toggle on the LCD viewfinder makes every two shots one film-style double
exposure. Two ungraded frames are added as light, then graded once, with a `0/2` / `1/2`
badge showing progress.

**Architecture:**
- A pure `pifilm/double.py` merges two sRGB frames in linear light, one stop off each.
- `CaptureSession` owns the mode and the pending first frame. Exposure 1 saves only its
  original. Exposure 2 saves its original and grades the composite once through the
  existing `Pipeline`.
- `CaptureController` forwards a toggle through its work queue and exposes the state in
  its snapshot. The viewfinder draws the badge, handles the toggle tap and shows a brief
  notice after exposure 1.
- The Stick gets an "Exposure 1/2" card through the unchanged image endpoint.

**Tech Stack:** Python 3.11+, NumPy, Pillow, pytest; ruff (E, F, I, B, UP; line length 100).

**Spec:** `docs/superpowers/specs/2026-09-30-double-exposure-design.md`

## Global Constraints

- Use `.venv/bin/pytest` and `.venv/bin/ruff`, run from the repo root. Never use a system Python.
- Nothing under `pifilm/display/` or `pifilm/double.py` may import OpenCV; use `pifilm._cv2.require_cv2()` elsewhere, never `import cv2`.
- Pixels enter only through `pifilm.imageio` (`load_rgb`, `save_jpeg`). No raw `Image.open` in new capture code.
- Composite maths: linear light, `L = 0.5·L₁ + 0.5·L₂`, via `pifilm.color.srgb_to_linear` / `linear_to_srgb`. Method name `"linear_mean"`.
- Grading order is unchanged: `normalize → LUT → grain`, run **once**, on the composite, with `ev=0.0`.
- With the mode off, `CaptureSession.capture()` behaves exactly as before. Every existing test must pass unchanged, except the `GatedSession` helper in `tests/test_viewfinder.py`, which Task 7 extends.
- Files: exposure 2 writes `<stem>_double_graded.jpg`. The exposure card is never written under `out_root`: the Nextcloud sync uploads everything there.
- The default font has no glyphs beyond ASCII: labels are `2x`, `2x 0/2`, `2x 1/2`, `Exposure 1/2`, `Exposure 1 of 2...`, `Developing double exposure...`.
- The mode and pending frame are memory only. They start off at 0/2 on every start.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **A plain SPACE/headless or `--show-captures` session receives an exposure-1 result**
   (a Stick shot while the mode is on). `_announce` must not `KeyError` on a record with no
   grade keys; it prints a one-line "exposure 1/2" message. Test in Task 3.
2. **The viewfinder sees a finished exposure-1 job but a stale `exposures_taken`.** The
   controller must update the double state in the same locked section that bumps
   `finished_count`. Test in Task 4 (one snapshot shows both).
3. **The Stick downloads `/image.jpg` for exposure 1.** It must get a valid ≤64 KiB JPEG
   (the card), not `image_unavailable`. Test in Task 5.
4. **Toggling on when already on at 1/2** (a double tap, or two taps before the snapshot
   catches up) must keep the pending frame. Toggling off must drop it. Test in Task 3.
5. **A frame-size change between exposures**, or any failure after exposure 2 is read,
   must fail the job and return to 0/2. It must not leave a pending frame that fails
   forever. Test in Task 3.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `pifilm/double.py` | Create | `composite()` and the rationale docstring |
| `pifilm/capture/thumbnail.py` | Modify | `render_exposure_card()` |
| `pifilm/capture/app.py` | Modify | `CaptureResult.exposure`, the session's double-exposure state, `_announce` guard |
| `pifilm/capture/controller.py` | Modify | Queued toggle and snapshot fields |
| `pifilm/capture/remote.py` | Modify | `double_exposure` in `/v1/status` |
| `pifilm/display/ui.py` | Modify | Badge, `DOUBLE_BOX`, `Action.DOUBLE_TOGGLE` |
| `pifilm/display/viewfinder.py` | Modify | Toggle tap, NOTICE state, processing labels, review caption |
| `tests/test_double.py` | Create | Composite maths and the card |
| `tests/test_double_capture.py` | Create | Session behaviour |
| `tests/test_capture_controller.py`, `tests/test_remote.py`, `tests/test_display_ui.py`, `tests/test_viewfinder.py` | Modify | New tests |
| `docs/lcd-viewfinder.md`, `docs/how-it-works.md`, `CLAUDE.md` | Modify | Docs |

---

### Task 1: Composite maths (`pifilm/double.py`)

**Files:**
- Create: `pifilm/double.py`
- Test: `tests/test_double.py`

**Interfaces:**
- Produces: `COMPOSITE_METHOD: str = "linear_mean"`; `composite(first: np.ndarray, second: np.ndarray) -> np.ndarray` (uint8 `(H, W, 3)` in, uint8 `(H, W, 3)` out; raises `ValueError`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_double.py
import numpy as np
import pytest

from pifilm.double import COMPOSITE_METHOD, composite


def _flat(value, shape=(4, 6)):
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_black_plus_black_stays_black():
    assert np.array_equal(composite(_flat(0), _flat(0)), _flat(0))


def test_black_lets_the_other_exposure_through_at_one_stop_under():
    """Darkness adds no light: white over black is white at half its linear light."""
    out = composite(_flat(0), _flat(255))
    assert np.all(out == 188)  # linear 0.5 -> sRGB 0.7354 -> 187.5 -> 188


def test_two_matching_exposures_make_one_normal_exposure():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    out = composite(frame, frame)
    assert np.abs(out.astype(int) - frame.astype(int)).max() <= 1


def test_colours_add_as_light_red_plus_green_is_yellow():
    red = _flat(0)
    red[..., 0] = 255
    green = _flat(0)
    green[..., 1] = 255
    out = composite(red, green)
    assert np.all(out[..., 0] == 188) and np.all(out[..., 1] == 188)
    assert np.all(out[..., 2] == 0)


def test_order_does_not_matter():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    assert np.array_equal(composite(a, b), composite(b, a))


def test_output_is_uint8_rgb_of_the_input_shape():
    out = composite(_flat(10, (3, 5)), _flat(200, (3, 5)))
    assert out.dtype == np.uint8 and out.shape == (3, 5, 3)


@pytest.mark.parametrize("second", [
    np.zeros((4, 7, 3), dtype=np.uint8),     # different size
    np.zeros((4, 6, 3), dtype=np.float32),   # wrong dtype
    np.zeros((4, 6), dtype=np.uint8),        # not RGB
])
def test_mismatched_or_malformed_frames_are_refused(second):
    with pytest.raises(ValueError):
        composite(_flat(0), second)


def test_method_name_is_recorded_verbatim():
    assert COMPOSITE_METHOD == "linear_mean"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_double.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pifilm.double'`

- [ ] **Step 3: Write the implementation**

```python
# pifilm/double.py
"""Two exposures on one frame of colour negative film.

Film records exposure H (light x time) at each point, and a second exposure adds
to the first: H = H1 + H2. Darkness adds almost nothing, so where one frame is dark
the other shows through at full strength (the classic silhouette filled with
trees); where both are bright, detail separation is lost. Colours add as light,
not as paint: red + green gives yellow. The film's characteristic curve then acts
once, on the sum.

So the composite is made here, before any grading, and ``Pipeline.process`` runs
once on the result. Grading each frame first would apply the LUT's tone curve twice
and add values after the curve, which is a flat 50/50 blend, not added light; it
would also lay grain down twice and normalise each frame and then the mix, which
``pifilm.normalize`` does not support.

Each frame is taken one stop down (x0.5 in linear light) before they are added.
That is the usual film starting point for two equal exposures: where the frames
match, the total is one normal exposure. The camera auto-exposes every frame to a
full exposure, so a plain sum would sit a stop over and clip before the LUT's
shoulder could act; the mean never exceeds 1.0. Each frame goes in as shot, its EV
compensation included, so a frame shot darker contributes less light. The caller
grades the composite with ``ev=0``: normalisation then sets the pair's overall
exposure, and the ratio between the two exposures is what survives.
"""

from __future__ import annotations

import numpy as np

from .color import linear_to_srgb, srgb_to_linear

COMPOSITE_METHOD = "linear_mean"


def composite(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Expose two sRGB uint8 frames onto one "negative"; return sRGB uint8."""
    for name, frame in (("first", first), ("second", second)):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"composite() expects RGB uint8 frames of shape (H, W, 3); "
                f"{name} is dtype {frame.dtype} shape {frame.shape}"
            )
    if first.shape != second.shape:
        raise ValueError(
            f"double exposure frames differ in size: {first.shape} vs {second.shape}"
        )
    linear = 0.5 * (srgb_to_linear(first / 255.0) + srgb_to_linear(second / 255.0))
    return np.clip(np.round(linear_to_srgb(linear) * 255.0), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/test_double.py && .venv/bin/ruff check pifilm/double.py tests/test_double.py`
Expected: all PASS; ruff prints `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add pifilm/double.py tests/test_double.py
git commit -m "Double exposure: linear-light composite of two frames

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Exposure card for the Stick (`render_exposure_card`)

**Files:**
- Modify: `pifilm/capture/thumbnail.py`
- Test: `tests/test_double.py` (append)

**Interfaces:**
- Produces: `render_exposure_card(index: int, of: int, *, size: tuple[int, int] = THUMBNAIL_SIZE) -> bytes`. It returns JPEG bytes of dimmed colour bars with `Exposure {index}/{of}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_double.py`)

```python
import io

from PIL import Image

from pifilm.capture.thumbnail import (
    MAX_JPEG_BYTES,
    THUMBNAIL_SIZE,
    fitted_jpeg,
    render_exposure_card,
)


def test_exposure_card_is_a_stick_sized_jpeg(tmp_path):
    data = render_exposure_card(1, 2)
    assert len(data) <= MAX_JPEG_BYTES
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "JPEG" and img.size == THUMBNAIL_SIZE
    # The remote /image.jpg path serves it through fitted_jpeg from a file.
    path = tmp_path / "card.jpg"
    path.write_bytes(data)
    assert len(fitted_jpeg(path)) <= MAX_JPEG_BYTES


def test_exposure_card_text_depends_on_the_index():
    assert render_exposure_card(1, 2) != render_exposure_card(2, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_double.py -k card`
Expected: FAIL with `ImportError: cannot import name 'render_exposure_card'`

- [ ] **Step 3: Implement** (in `pifilm/capture/thumbnail.py`)

Change the Pillow import to `from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError`. Add `from ..display.ui import PROCESSING_BARS, PROCESSING_DIM` below it; `pifilm.display.ui` is Pillow-only. Then append:

```python
def render_exposure_card(
    index: int, of: int, *, size: tuple[int, int] = THUMBNAIL_SIZE,
) -> bytes:
    """What the Stick shows after exposure ``index`` of a double exposure.

    Exposure 1 has no photo yet: the film has not been developed. The Stick's
    protocol expects an image for every finished capture, so it gets the same dimmed
    colour bars the LCD shows while working, labelled with the count.
    """
    width, height = size
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(PROCESSING_BARS):
        left, right = i * width // 7, (i + 1) * width // 7
        dimmed = tuple(round(c * PROCESSING_DIM) for c in colour)
        draw.rectangle((left, 0, right - 1, height * 3 // 4 - 1), fill=dimmed)
    draw.text((width // 2, height * 7 // 8), f"Exposure {index}/{of}",
              fill=(255, 255, 255), font=ImageFont.load_default(size=16), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/test_double.py tests/test_thumbnail.py && .venv/bin/ruff check pifilm/capture/thumbnail.py tests/test_double.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/thumbnail.py tests/test_double.py
git commit -m "Double exposure: Exposure 1/2 card for the Stick

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `CaptureSession` double-exposure mode

**Files:**
- Modify: `pifilm/capture/app.py`: `CaptureResult` (~line 98), `CaptureSession` (~104–192), `_announce` (~194)
- Test: `tests/test_double_capture.py` (create)

**Interfaces:**
- Consumes: `composite`, `COMPOSITE_METHOD` (Task 1); `render_exposure_card` (Task 2).
- Produces:
  - `CaptureResult.exposure: tuple[int, int] | None = None`.
  - `CaptureSession.double_exposure: bool`.
  - `CaptureSession.set_double_exposure(enabled: bool) -> None`.
  - `CaptureSession.double_state -> tuple[bool, int]` (property: enabled, exposures taken, 0 or 1).
  - Exposure 1's result: `pifilm` = the card path, `exposure=(1, 2)`.
  - Exposure 2's result: `pifilm` = `<stem>_double_graded.jpg`, `exposure=(2, 2)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_double_capture.py
import json

import numpy as np
import pytest

from pifilm.artifacts import Artifacts, write_artifact
from pifilm.capture.app import CaptureSession, _announce
from pifilm.capture.camera import CameraError, FakeCamera
from pifilm.double import composite
from pifilm.grain import GrainParams
from pifilm.lut import LUT3D
from pifilm.normalize import NormalizeParams
from pifilm.pipeline import Pipeline

DARK = np.full((90, 160, 3), 10, dtype=np.uint8)
BRIGHT = np.full((90, 160, 3), 230, dtype=np.uint8)


@pytest.fixture
def pipeline(tmp_path):
    d = tmp_path / "art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(d))


def _session(tmp_path, pipeline, frames=(DARK, BRIGHT)):
    camera = FakeCamera([f.copy() for f in frames])
    return CaptureSession(camera, pipeline, tmp_path / "shots",
                          seed_rng=np.random.default_rng(0))


def _records(session):
    lines = []
    for f in sorted(session.out_root.glob("*/captures.jsonl")):
        lines += [json.loads(line) for line in f.read_text().splitlines()]
    return lines


def test_mode_is_off_by_default_and_single_shots_are_unchanged(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    assert session.double_state == (False, 0)
    result = session.capture()
    assert result.exposure is None
    assert result.pifilm.name.endswith("_graded.jpg")
    assert "double" not in _records(session)[0]


def test_exposure_one_saves_only_the_original_and_returns_the_card(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    result = session.capture()
    assert result.exposure == (1, 2)
    assert session.double_state == (True, 1)
    assert result.original.exists()
    day = result.original.parent
    assert not list(day.glob("*_graded.jpg"))
    # The card is served to the Stick but never lands in the synced photo folder.
    assert result.pifilm.exists()
    assert session.out_root not in result.pifilm.parents
    (record,) = _records(session)
    assert record["double"] == {"index": 1, "of": 2}
    assert "pifilm" not in record and record["original"] == result.original.name


def test_exposure_two_grades_the_composite_once(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    calls = []
    real = session.pipeline.process

    def spy(rgb, **kw):
        calls.append((rgb.copy(), kw))
        return real(rgb, **kw)

    monkeypatch.setattr(session.pipeline, "process", spy)
    session.set_double_exposure(True)
    first = session.capture()
    second = session.capture()

    assert len(calls) == 1  # graded once: one normalisation, one LUT, one grain pass
    graded_input, kw = calls[0]
    assert np.array_equal(graded_input, composite(DARK, BRIGHT))
    assert kw["ev"] == 0.0
    assert second.exposure == (2, 2)
    assert second.pifilm.name.endswith("_double_graded.jpg") and second.pifilm.exists()
    assert session.double_state == (True, 0)
    record = _records(session)[1]
    assert record["pifilm"] == second.pifilm.name
    assert record["double"]["index"] == 2 and record["double"]["method"] == "linear_mean"
    assert record["double"]["originals"] == [first.original.name, second.original.name]
    assert record["double"]["ev_comp"] == [0.0, 0.0]
    assert isinstance(record["grain_seed"], int) and "lut_sha1" in record


def test_each_exposure_keeps_the_ev_it_was_shot_at(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.camera.set_ev(-1.0)
    session.capture()
    session.camera.set_ev(0.0)
    session.capture()
    assert _records(session)[1]["double"]["ev_comp"] == [-1.0, 0.0]


def test_the_count_cycles_back_for_the_next_pair(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    assert [session.capture().exposure for _ in range(4)] == [(1, 2), (2, 2), (1, 2), (2, 2)]


def test_toggling_off_at_one_of_two_drops_the_pending_frame(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    session.set_double_exposure(False)
    assert session.double_state == (False, 0)
    session.set_double_exposure(True)
    assert session.capture().exposure == (1, 2)


def test_toggling_on_again_keeps_the_pending_frame(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    session.set_double_exposure(True)
    assert session.double_state == (True, 1)
    assert session.capture().exposure == (2, 2)


def test_a_camera_error_on_exposure_two_keeps_one_of_two(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()
    real = session.camera.read

    def broken(*, full=True):
        raise CameraError("no frame")

    monkeypatch.setattr(session.camera, "read", broken)
    with pytest.raises(CameraError):
        session.capture()
    assert session.double_state == (True, 1)
    monkeypatch.setattr(session.camera, "read", real)
    assert session.capture().exposure == (2, 2)


def test_a_failure_after_exposure_two_is_read_returns_to_zero(tmp_path, pipeline, monkeypatch):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    session.capture()

    def boom(*a, **k):
        raise RuntimeError("grade failed")

    monkeypatch.setattr(session.pipeline, "process", boom)
    with pytest.raises(RuntimeError):
        session.capture()
    assert session.double_state == (True, 0)
    assert len(list(session.out_root.glob("*/*_original.jpg"))) == 0  # decoded source
    assert len(list(session.out_root.glob("*/*_ungraded.jpg"))) == 2  # both kept


def test_frames_of_different_sizes_fail_and_reset(tmp_path, pipeline):
    small = np.full((45, 80, 3), 128, dtype=np.uint8)
    session = _session(tmp_path, pipeline, frames=(DARK, small))
    session.set_double_exposure(True)
    session.capture()
    with pytest.raises(ValueError):
        session.capture()
    assert session.double_state == (True, 0)


def test_announce_handles_an_exposure_one_result(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    session.set_double_exposure(True)
    lines = []
    _announce(session.capture(), lines.append)
    assert len(lines) == 1 and "exposure 1/2" in lines[0]
    _announce(session.capture(), lines.append)
    assert "_double_graded.jpg" in lines[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_double_capture.py`
Expected: FAIL with `AttributeError: 'CaptureSession' object has no attribute 'double_state'`

- [ ] **Step 3: Implement in `pifilm/capture/app.py`**

3a. Imports: add `import tempfile` (with the stdlib imports). Add `from ..double import COMPOSITE_METHOD, composite`. Change the camera import to `from .camera import Camera, CameraError, FakeCamera, Frame, V4L2Camera`. Add `from .thumbnail import render_exposure_card`.

3b. Replace the `CaptureResult` dataclass and add `_PendingExposure` after it:

```python
@dataclass
class CaptureResult:
    original: Path
    pifilm: Path
    record: dict
    # (index, of) for a double-exposure shot, None for a single one. After exposure 1
    # ``pifilm`` is the "Exposure 1/2" card, not a photo: nothing is graded until the
    # second exposure lands on the same "negative" (see ``pifilm.double``).
    exposure: tuple[int, int] | None = None


@dataclass
class _PendingExposure:
    """Exposure 1 of a double, held in memory until exposure 2 arrives."""

    rgb: np.ndarray
    original: str
    ev: float
```

3c. Add to the end of `CaptureSession.__init__`:

```python
        # Double exposure: touched only on the controller's worker thread. Memory
        # only, so a restart always begins with the mode off at 0/2.
        self.double_exposure = False
        self._pending: _PendingExposure | None = None
        self._card_dir: Path | None = None
```

3d. Replace `capture()` with the version below, and add the new methods after it.
The single-shot path produces the same files and record as before; only the
original/DNG writing and the log append moved into helpers.

```python
    @property
    def double_state(self) -> tuple[bool, int]:
        """(mode on, exposures taken towards the current pair: 0 or 1)."""
        return self.double_exposure, (1 if self._pending is not None else 0)

    def set_double_exposure(self, enabled: bool) -> None:
        """Turning the mode off drops a pending exposure 1; its original stays on disk.

        Turning it on while already on keeps the pending frame, so a repeated tap
        cannot silently restart a half-made pair.
        """
        self.double_exposure = bool(enabled)
        if not self.double_exposure:
            self._pending = None

    def capture(self) -> CaptureResult:
        shutter = time.perf_counter()
        # Read before the frame: the EV the exposure was made at. The grade has to
        # re-apply it, or normalisation cancels it (see ``pifilm.pipeline``).
        ev = float(getattr(self.camera, "ev", 0.0))
        frame = self.camera.read()
        if self.double_exposure:
            return self._capture_double(frame, ev, shutter)

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(
            frame.rgb, rng=np.random.default_rng(seed), ev=ev,
        )
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        day_dir, stem, t, original, dng_name = self._write_original(frame)
        pifilm = save_jpeg(graded, day_dir / f"{stem}_graded.jpg")
        shutter_to_saved_ms = (time.perf_counter() - shutter) * 1000.0

        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": pifilm.name,
            "frame_source": frame.source,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round(shutter_to_saved_ms, 1),
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        self._append_record(day_dir, record)
        return CaptureResult(original, pifilm, record)

    def _capture_double(self, frame: Frame, ev: float, shutter: float) -> CaptureResult:
        # The pending frame is taken off the session before anything that can fail, so
        # any failure from here on returns the count to 0/2 (spec section 5). A camera
        # error is raised by read() before this point and leaves 1/2 in place for a retry.
        pending, self._pending = self._pending, None
        day_dir, stem, t, original, dng_name = self._write_original(frame)
        camera_fields = {
            "frame_source": frame.source,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            **({"camera_metadata": frame.metadata} if frame.metadata else {}),
            **({"dng": dng_name} if dng_name else {}),
        }
        if pending is None:
            record = {
                "timestamp": t.isoformat(timespec="seconds"),
                "original": original.name,
                **camera_fields,
                "ev_comp": ev,
                "shutter_to_saved_ms": round((time.perf_counter() - shutter) * 1000.0, 1),
                "double": {"index": 1, "of": 2},
            }
            self._append_record(day_dir, record)
            card = self._exposure_card(1, 2)
            self._pending = _PendingExposure(frame.rgb, original.name, ev)
            return CaptureResult(original, card, record, exposure=(1, 2))

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        # Each frame goes in as shot, EV included; the composite is graded at ev=0.
        graded, info = self.pipeline.process(
            composite(pending.rgb, frame.rgb), rng=np.random.default_rng(seed), ev=0.0,
        )
        pipeline_ms = (time.perf_counter() - t0) * 1000.0
        pifilm = save_jpeg(graded, day_dir / f"{stem}_double_graded.jpg")
        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "pifilm": pifilm.name,
            **camera_fields,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round((time.perf_counter() - shutter) * 1000.0, 1),
            "double": {
                "index": 2, "of": 2, "method": COMPOSITE_METHOD,
                "originals": [pending.original, original.name],
                "ev_comp": [pending.ev, ev],
            },
        }
        self._append_record(day_dir, record)
        return CaptureResult(original, pifilm, record, exposure=(2, 2))

    def _write_original(self, frame: Frame) -> tuple[Path, str, datetime, Path, str | None]:
        has_original = frame.jpeg is not None or frame.source == "picamera2"
        suffix = "original" if has_original else "ungraded"
        day_dir, stem, t = self._allocate(suffix)
        original = day_dir / f"{stem}_{suffix}.jpg"
        if frame.jpeg is not None:
            original.write_bytes(frame.jpeg)
        else:
            save_jpeg(frame.rgb, original)
        dng_name = None
        if frame.dng is not None and self._save_dng:
            dng_path = day_dir / f"{stem}.dng"
            dng_path.write_bytes(frame.dng)
            dng_name = dng_path.name
        return day_dir, stem, t, original, dng_name

    @staticmethod
    def _append_record(day_dir: Path, record: dict) -> None:
        with (day_dir / "captures.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _exposure_card(self, index: int, of: int) -> Path:
        """The Stick's picture for a partial double; never under ``out_root``.

        Everything under ``out_root`` is uploaded by the Nextcloud sync, and a card is
        not a photograph, so it lives in a private temporary directory.
        """
        if self._card_dir is None:
            self._card_dir = Path(tempfile.mkdtemp(prefix="pifilm-cards-"))
        path = self._card_dir / f"exposure-{index}-of-{of}.jpg"
        if not path.exists():
            path.write_bytes(render_exposure_card(index, of))
        return path
```

3e. In `_announce`, insert directly after `r = result.record`:

```python
    if result.exposure == (1, 2):
        out(
            f"Saved {result.original.name} as exposure 1/2 in "
            f"{r['shutter_to_saved_ms']:.0f} ms; waiting for the second exposure"
        )
        return
```

- [ ] **Step 4: Run the new and existing capture tests**

Run: `.venv/bin/pytest -q tests/test_double_capture.py tests/test_app.py && .venv/bin/ruff check pifilm/capture/app.py tests/test_double_capture.py`
Expected: all PASS. `test_app.py` passing shows the single-shot path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/app.py tests/test_double_capture.py
git commit -m "Double exposure: capture session composites two frames, grades once

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Controller toggle and snapshot

**Files:**
- Modify: `pifilm/capture/controller.py`
- Test: `tests/test_capture_controller.py` (append)

**Interfaces:**
- Consumes: the session's `set_double_exposure(bool)` and `double_state` (Task 3), both optional on the session.
- Produces:
  - `CaptureController.set_double_exposure(enabled: bool) -> None`.
  - `ControllerSnapshot.double_exposure: bool = False`.
  - `ControllerSnapshot.exposures_taken: int = 0`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_capture_controller.py`)

```python
class DoubleSession:
    """Minimal session with the double-exposure surface of ``CaptureSession``."""

    def __init__(self) -> None:
        self.camera = Mock()
        self.enabled = False
        self.taken = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    @property
    def double_state(self):
        return self.enabled, self.taken

    def set_double_exposure(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.taken = 0

    def capture(self):
        self.started.set()
        self.release.wait(timeout=2)
        if self.enabled:
            self.taken = (self.taken + 1) % 2
        return Result("saved")


def _until(pred):
    for _ in range(200):
        if pred():
            return
        threading.Event().wait(0.01)
    raise AssertionError("condition not met")


def test_double_exposure_toggle_reaches_the_snapshot():
    session = DoubleSession()
    controller = CaptureController(session)
    try:
        assert controller.snapshot().double_exposure is False
        controller.set_double_exposure(True)
        _until(lambda: controller.snapshot().double_exposure)
        assert controller.snapshot().exposures_taken == 0
    finally:
        controller.close()


def test_a_finished_job_and_its_exposure_count_arrive_in_one_snapshot():
    """A display must never see exposure 1 finished with the count still at 0."""
    session = DoubleSession()
    controller = CaptureController(session)
    try:
        controller.set_double_exposure(True)
        _until(lambda: controller.snapshot().double_exposure)
        controller.submit("one")
        _until(lambda: controller.snapshot().finished_count == 1)
        snap = controller.snapshot()
        assert snap.finished_count == 1 and snap.exposures_taken == 1
    finally:
        controller.close()


def test_a_toggle_sent_during_a_capture_applies_after_it():
    session = DoubleSession()
    session.enabled = True
    session.release.clear()
    controller = CaptureController(session)
    try:
        controller.submit("busy")
        assert session.started.wait(timeout=1)
        controller.set_double_exposure(False)
        assert session.enabled is True  # not applied mid-capture
        session.release.set()
        _until(lambda: controller.snapshot().double_exposure is False)
        assert controller.snapshot().finished_count == 1
    finally:
        session.release.set()
        controller.close()


def test_a_session_without_double_exposure_ignores_the_toggle():
    session = BlockingSession()
    session.release.set()
    controller = CaptureController(session)
    try:
        controller.set_double_exposure(True)
        controller.submit("plain")
        wait_for(controller, "plain", "complete")
        assert controller.snapshot().double_exposure is False
    finally:
        controller.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_capture_controller.py -k "double or toggle"`
Expected: FAIL with `AttributeError: 'CaptureController' object has no attribute 'set_double_exposure'`

- [ ] **Step 3: Implement in `pifilm/capture/controller.py`**

Add after `ControllerSnapshot`'s `last_finished_job` field:

```python
    # Double exposure, as the session reported it when the last job or toggle
    # finished. Updated under the same lock as ``finished_count``, so a display that
    # sees a job finish also sees the count that job left behind.
    double_exposure: bool = False
    exposures_taken: int = 0
```

Add above `class CaptureController`:

```python
@dataclass(frozen=True)
class _DoubleToggle:
    enabled: bool
```

In `__init__`, change the queue line and add the cached state before the worker starts:

```python
        self._double = self._read_double()
        self._work: queue.Queue[str | _DoubleToggle | None] = queue.Queue()
```

Add the public method after `submit`:

```python
    def set_double_exposure(self, enabled: bool) -> None:
        """Queue a double-exposure toggle behind any capture already accepted.

        It runs on the worker thread, which is the only thread that touches the
        session, so it can never change the mode in the middle of a pair's capture.
        """
        with self._lock:
            if self._closed:
                return
            self._work.put(_DoubleToggle(bool(enabled)))
```

Extend `snapshot()`'s constructor call with `self._double[0], self._double[1]` after
`self._last_finished_job`.

In `_run`, replace the top of the loop body:

```python
                item = self._work.get()
                if item is None:
                    return
                if isinstance(item, _DoubleToggle):
                    self._apply_toggle(item.enabled)
                    continue
                request_id = item
```

Add these helpers. In `_finish`, read the state before taking the lock and store it
inside the lock:

```python
    def _read_double(self) -> tuple[bool, int]:
        state = getattr(self._session, "double_state", None)
        if not isinstance(state, tuple) or len(state) != 2:
            return False, 0
        return bool(state[0]), int(state[1])

    def _apply_toggle(self, enabled: bool) -> None:
        setter = getattr(self._session, "set_double_exposure", None)
        if callable(setter):
            setter(enabled)
        double = self._read_double()
        with self._lock:
            self._double = double
```

```python
    def _finish(self, request_id, state, *, error_code=None, result=None, error_message=None):
        double = self._read_double()
        with self._lock:
            ...existing body unchanged...
            self._double = double
```

(Keep `_finish`'s existing type annotations; only the `double = ...` line and the
`self._double = double` line are new.)

- [ ] **Step 4: Run the controller tests**

Run: `.venv/bin/pytest -q tests/test_capture_controller.py && .venv/bin/ruff check pifilm/capture/controller.py tests/test_capture_controller.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/controller.py tests/test_capture_controller.py
git commit -m "Double exposure: controller queues the toggle and snapshots the count

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remote status and the Stick's exposure-1 image

**Files:**
- Modify: `pifilm/capture/remote.py:205-232` (`_status_payload`)
- Test: `tests/test_remote.py` (append)

**Interfaces:**
- Consumes: `ControllerSnapshot.double_exposure` and `exposures_taken` (Task 4); `CaptureSession`'s exposure-1 card result (Task 3).
- Produces: `/v1/status` JSON key `"double_exposure": {"enabled": bool, "taken": int}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_remote.py`; reuse its `remote` fixture, `_request` (returns `(status, headers, bytes)`) and `_json` (returns `(status, headers, dict)`))

```python
def test_status_reports_double_exposure(remote):
    server, _session = remote
    status, _headers, payload = _json(server, "GET", "/v1/status")
    assert status == 200
    assert payload["double_exposure"] == {"enabled": False, "taken": 0}


def test_exposure_one_image_is_the_card(tmp_path):
    from pifilm.artifacts import Artifacts, write_artifact
    from pifilm.capture.app import CaptureSession
    from pifilm.capture.camera import FakeCamera, synthetic_frame
    from pifilm.grain import GrainParams
    from pifilm.lut import LUT3D
    from pifilm.normalize import NormalizeParams
    from pifilm.pipeline import Pipeline

    art = tmp_path / "art"
    write_artifact(art, LUT3D.identity(9), NormalizeParams(), GrainParams())
    session = CaptureSession(FakeCamera([synthetic_frame(48, 64)]),
                             Pipeline(Artifacts.load(art)), tmp_path / "shots")
    controller = CaptureController(session)
    server = RemoteCaptureServer(controller, "secret-token", ("127.0.0.1", 0))
    server.start()
    try:
        controller.set_double_exposure(True)
        request_id = str(uuid.uuid4())
        _request(server, "POST", "/v1/captures", json.dumps({"request_id": request_id}))
        for _ in range(200):
            if controller.status(request_id).state == "complete":
                break
            time.sleep(0.01)
        status, _headers, body = _request(server, "GET", f"/v1/captures/{request_id}/image.jpg")
        assert status == 200 and body[:2] == b"\xff\xd8"
        _status, _headers, payload = _json(server, "GET", "/v1/status")
        assert payload["double_exposure"] == {"enabled": True, "taken": 1}
    finally:
        server.close()
        controller.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_remote.py -k "double or card"`
Expected: FAIL with `KeyError: 'double_exposure'`

- [ ] **Step 3: Implement.** In `_status_payload`'s returned dict, add after `"pi_battery"`:

```python
            # Ignored by current Stick firmware; lets any client show the 0/2, 1/2 count.
            "double_exposure": {
                "enabled": snapshot.double_exposure,
                "taken": snapshot.exposures_taken,
            },
```

- [ ] **Step 4: Run the remote tests**

Run: `.venv/bin/pytest -q tests/test_remote.py && .venv/bin/ruff check pifilm/capture/remote.py tests/test_remote.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pifilm/capture/remote.py tests/test_remote.py
git commit -m "Double exposure: report the mode and count in /v1/status

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Badge and toggle hit region (`pifilm/display/ui.py`)

**Files:**
- Modify: `pifilm/display/ui.py`
- Test: `tests/test_display_ui.py` (append)

**Interfaces:**
- Produces:
  - `DOUBLE_BOX = (4, 4, 74, 24)`.
  - `Action.DOUBLE_TOGGLE`.
  - `render_live(frame_rgb, reading, double: tuple[bool, int] | None = None)`: `None` draws no badge, so existing callers are unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_display_ui.py`, adding `DOUBLE_BOX` to its import list from `pifilm.display.ui`)

```python
def _box_pixels(img):
    x0, y0, x1, y1 = DOUBLE_BOX
    return np.asarray(img)[y0:y1 + 1, x0:x1 + 1]


def test_no_badge_unless_asked():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    assert np.array_equal(
        np.asarray(render_live(frame, _reading())),
        np.asarray(render_live(frame, _reading(), None)),
    )
    # the corner is the plain frame
    assert np.all(_box_pixels(render_live(frame, _reading()))[6:10, 2:6] == 200)


def test_badge_off_is_dark_on_is_amber():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    off = np.asarray(render_live(frame, _reading(), (False, 0)))
    on = np.asarray(render_live(frame, _reading(), (True, 0)))
    x, y = DOUBLE_BOX[0] + 4, (DOUBLE_BOX[1] + DOUBLE_BOX[3]) // 2
    assert off[y, x].max() < 150
    assert tuple(on[y, x]) == AMBER


def test_badge_shows_progress():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    zero = _box_pixels(render_live(frame, _reading(), (True, 0)))
    one = _box_pixels(render_live(frame, _reading(), (True, 1)))
    assert not np.array_equal(zero, one)


def test_badge_tap_is_the_double_toggle():
    x0, y0, x1, y1 = DOUBLE_BOX
    assert hit((x0 + x1) // 2, (y0 + y1) // 2) is Action.DOUBLE_TOGGLE
    assert hit(x1 + HIT_MARGIN, y1 + HIT_MARGIN) is Action.DOUBLE_TOGGLE
    assert hit(x1 + HIT_MARGIN + 1, (y0 + y1) // 2) is Action.NONE


def test_badge_region_overlaps_no_other_control():
    x0, y0, x1, y1 = DOUBLE_BOX
    for x in range(max(0, x0 - HIT_MARGIN), x1 + HIT_MARGIN + 1):
        for y in range(max(0, y0 - HIT_MARGIN), y1 + HIT_MARGIN + 1):
            assert hit(x, y) is Action.DOUBLE_TOGGLE
    # clear of the focus bar's backing, which starts at FOCUS_BAR_Y0 - 4
    assert y1 + HIT_MARGIN < FOCUS_BAR_Y0 - 4
    assert hit(*SHUTTER_CENTRE) is Action.SHUTTER
```

(`HIT_MARGIN` must be added to the test file's import list too.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_display_ui.py -k badge`
Expected: FAIL with `ImportError: cannot import name 'DOUBLE_BOX'`

- [ ] **Step 3: Implement in `pifilm/display/ui.py`**

After the focus-bar constants:

```python
# Double-exposure toggle: a pill in the top-left corner. It sits above the focus
# bar's backing (y >= FOCUS_BAR_Y0 - 4) even with its hit margin, and away from the
# battery badge (top-right) and the shutter (right edge).
DOUBLE_BOX = (4, 4, 74, 24)
```

Add `DOUBLE_TOGGLE = "double_toggle"` to `Action`.

Change `render_live`'s signature to
`def render_live(frame_rgb: np.ndarray, reading: MeterReading, double: tuple[bool, int] | None = None) -> Image.Image:`
(wrap it to 100 columns). Just before `# battery badge`, add:

```python
    if double is not None:
        _draw_double_badge(draw, double[0], double[1], small)
```

Add the helper below `_draw_focus_bar`:

```python
def _draw_double_badge(draw: ImageDraw.ImageDraw, enabled: bool, taken: int, font: Any) -> None:
    """The double-exposure toggle. Off: an outlined ``2x``. On: amber, with progress.

    ASCII only: the default font has no multiplication sign.
    """
    x0, y0, x1, y1 = DOUBLE_BOX
    if enabled:
        draw.rounded_rectangle(DOUBLE_BOX, radius=8, fill=AMBER + (255,))
        label, colour = f"2x {taken}/2", (0, 0, 0, 255)
    else:
        draw.rounded_rectangle(DOUBLE_BOX, radius=8, fill=(0, 0, 0, BAR_ALPHA),
                               outline=(255, 255, 255, 255), width=1)
        label, colour = "2x", (255, 255, 255, 255)
    draw.text(((x0 + x1) // 2, (y0 + y1) // 2), label, fill=colour, font=font, anchor="mm")
```

In `hit()`, after the shutter check and before the bar check:

```python
    x0, y0, x1, y1 = DOUBLE_BOX
    if x0 - HIT_MARGIN <= x <= x1 + HIT_MARGIN and y0 - HIT_MARGIN <= y <= y1 + HIT_MARGIN:
        return Action.DOUBLE_TOGGLE
```

- [ ] **Step 4: Run the UI tests**

Run: `.venv/bin/pytest -q tests/test_display_ui.py && .venv/bin/ruff check pifilm/display/ui.py tests/test_display_ui.py`
Expected: all PASS. If `test_badge_off_is_dark_on_is_amber` lands on a text pixel, move
the sample point `x` to `DOUBLE_BOX[0] + 3`; the text is centred and never reaches
there.

- [ ] **Step 5: Commit**

```bash
git add pifilm/display/ui.py tests/test_display_ui.py
git commit -m "Double exposure: 2x badge and toggle hit region on the viewfinder

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Viewfinder flow (`pifilm/display/viewfinder.py`)

**Files:**
- Modify: `pifilm/display/viewfinder.py`
- Test: `tests/test_viewfinder.py` (extend `GatedSession`, then append tests)

**Interfaces:**
- Consumes:
  - `Action.DOUBLE_TOGGLE`, `DOUBLE_BOX`, and `render_live(..., double=)` (Task 6).
  - `CaptureController.set_double_exposure` and the snapshot's `double_exposure` / `exposures_taken` (Task 4).
  - `CaptureResult.exposure` (Task 3).
- Produces:
  - New state `"NOTICE"`.
  - `NOTICE_SECONDS = 1.2`.
  - `PROCESSING_FIRST = "Exposure 1 of 2..."`.
  - `PROCESSING_DOUBLE = "Developing double exposure..."`.

- [ ] **Step 1: Extend `GatedSession`** in `tests/test_viewfinder.py` so the controller can
reach the wrapped session's double-exposure surface. Add this method to the class:

```python
    def __getattr__(self, name):
        # Only called for attributes the wrapper lacks: double_state and
        # set_double_exposure fall through to the real session.
        return getattr(self._session, name)
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_viewfinder.py`, adding `DOUBLE_BOX` to the `pifilm.display.ui` import and `NOTICE_SECONDS` to the `pifilm.display.viewfinder` import)

```python
def _badge_centre():
    x0, y0, x1, y1 = DOUBLE_BOX
    return (x0 + x1) // 2, (y0 + y1) // 2


def _enable_double(loop, touch, clock, ctl):
    _tap(loop, touch, clock, _badge_centre())
    _until(lambda: ctl.snapshot().double_exposure)


def test_badge_tap_toggles_double_exposure(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    _tap(loop, touch, clock, _badge_centre())
    _until(lambda: not ctl.snapshot().double_exposure)


def test_live_view_draws_the_badge_from_the_snapshot(controller, monkeypatch):
    camera, ctl = controller
    seen = []
    real = viewfinder.render_live

    def spy(frame, reading, double=None):
        seen.append(double)
        return real(frame, reading, double)

    monkeypatch.setattr(viewfinder, "render_live", spy)
    loop, touch, display, clock = _loop(camera, ctl)
    loop.step()
    assert seen[-1] == (False, 0)
    _enable_double(loop, touch, clock, ctl)
    loop.step()
    assert seen[-1] == (True, 0)


def test_exposure_one_shows_a_notice_not_a_review(controller, monkeypatch):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    assert loop.state == "NOTICE"
    assert calls["review"] == []
    assert calls["message"][-1][0] == "Exposure 1/2"
    clock.t += NOTICE_SECONDS - 0.1
    loop.step()
    assert loop.state == "NOTICE"
    clock.t += 0.2
    loop.step()
    assert loop.state == "LIVE"


def test_a_tap_ends_the_notice_early(controller):
    camera, ctl = controller
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    _tap(loop, touch, clock, (160, 120))
    assert loop.state == "LIVE"
    assert ctl.snapshot().finished_count == 1  # the tap did not fire the shutter


def test_exposure_two_reviews_the_composite_with_a_2x_caption(controller, monkeypatch):
    camera, ctl = controller
    calls = _record_renderers(monkeypatch)
    loop, touch, display, clock = _loop(camera, ctl)
    _enable_double(loop, touch, clock, ctl)
    ctl.submit("exp-1")
    _until(lambda: ctl.snapshot().finished_count == 1)
    loop.step()
    ctl.submit("exp-2")
    _until(lambda: ctl.snapshot().finished_count == 2)
    loop.step()
    assert loop.state == "REVIEW"
    assert calls["review"][-1].startswith("2x")
    job = ctl.snapshot().last_finished_job
    assert job.result.pifilm.name.endswith("_double_graded.jpg")


def test_processing_labels_follow_the_pair(tmp_path, monkeypatch):
    camera, ctl, session = _gated_controller(tmp_path)
    labels = []
    real = viewfinder.render_processing

    def spy(label="Processing photo..."):
        labels.append(label)
        return real(label)

    monkeypatch.setattr(viewfinder, "render_processing", spy)
    try:
        loop, touch, display, clock = _loop(camera, ctl)
        _enable_double(loop, touch, clock, ctl)
        for request_id, expected in (("a", "Exposure 1 of 2..."),
                                     ("b", "Developing double exposure...")):
            session.started.clear()
            session.release.clear()
            ctl.submit(request_id)
            _until(lambda: session.started.is_set())
            loop.step()
            assert labels[-1] == expected
            session.release.set()
            _until(lambda: ctl.status(request_id).state == "complete")
            loop.step()   # NOTICE or REVIEW
            _tap(loop, touch, clock, (160, 120))  # back to LIVE
    finally:
        session.release.set()
        ctl.close()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py -k "double or notice or badge or exposure or labels"`
Expected: FAIL with `ImportError: cannot import name 'NOTICE_SECONDS'`

- [ ] **Step 4: Implement in `pifilm/display/viewfinder.py`**

Add to the module docstring's first paragraph:

```
Double exposure adds a third state, NOTICE: after exposure 1 of a pair there is
nothing developed to review, so a short "Exposure 1/2" message replaces the
review and the loop returns to LIVE for the second exposure.
```

After `FULL_BACKLIGHT`:

```python
# Double exposure: how long the "Exposure 1/2" notice holds before the live view
# returns for the second exposure (a tap ends it sooner), and the processing labels.
NOTICE_SECONDS = 1.2
PROCESSING_FIRST = "Exposure 1 of 2..."
PROCESSING_DOUBLE = "Developing double exposure..."
```

In `__init__`, next to `self._review_since = 0.0`, add `self._notice_since = 0.0`.

Replace the "new finished job" block at the top of `step()` (the `if snap.finished_count
!= self._seen_finished ...` block) with:

```python
        if snap.finished_count != self._seen_finished and snap.last_finished_job is not None:
            self._seen_finished = snap.finished_count
            self._processing_shown = False
            job = snap.last_finished_job
            if job.state == "complete" and getattr(job.result, "exposure", None) == (1, 2):
                self.state = "NOTICE"
                self._notice_since = self._clock.monotonic()
                self._show(render_message("Exposure 1/2", "frame the second exposure"))
                return
            self.state = "REVIEW"
            self._review_since = self._clock.monotonic()
            self._show(self._review_image(job))
            return
        if self.state == "NOTICE":
            held = self._clock.monotonic() - self._notice_since
            if tap is not None or held >= NOTICE_SECONDS:
                self.state = "LIVE"
                self._restart_rate_window()
            return
```

In the LIVE tap dispatch, add a branch:

```python
            elif action is Action.DOUBLE_TOGGLE and snap.active_job is None:
                # Ignored while a job runs (spec section 5): the badge is hidden behind
                # the colour bars, so a tap there was not aimed at it.
                self._controller.set_double_exposure(not snap.double_exposure)
```

Add a test for it alongside the others in Step 2:

```python
def test_badge_taps_during_processing_are_ignored(tmp_path):
    camera, ctl, session = _gated_controller(tmp_path)
    try:
        loop, touch, display, clock = _loop(camera, ctl)
        ctl.submit("busy")
        _until(lambda: session.started.is_set())
        _tap(loop, touch, clock, _badge_centre())
        session.release.set()
        _until(lambda: ctl.snapshot().finished_count == 1)
        assert ctl.snapshot().double_exposure is False
    finally:
        session.release.set()
        ctl.close()
```

In the active-job block, replace `self._show(render_processing())` with
`self._show(render_processing(self._processing_label(snap)))`, and add the method in the
plumbing section:

```python
    @staticmethod
    def _processing_label(snap: Any) -> str:
        """Which exposure of a pair is on its way. The count is the one before the job."""
        if not getattr(snap, "double_exposure", False):
            return "Processing photo..."
        return PROCESSING_DOUBLE if snap.exposures_taken == 1 else PROCESSING_FIRST
```

Change the live render call to
`self._show(render_live(frame.rgb, reading, (snap.double_exposure, snap.exposures_taken)))`.

In `_review_image`, just before `return render_review(rgb, caption)`:

```python
            if getattr(job.result, "exposure", None) == (2, 2):
                caption = f"2x  {caption}"
```

- [ ] **Step 5: Run the viewfinder tests and the full fast suite**

Run: `.venv/bin/pytest -q tests/test_viewfinder.py && .venv/bin/pytest -q -m 'not slow' && .venv/bin/ruff check .`
Expected: all PASS; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add pifilm/display/viewfinder.py tests/test_viewfinder.py
git commit -m "Double exposure: viewfinder toggle, 1/2 notice and composite review

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/lcd-viewfinder.md` (§5 table, new §5 subsection, §7 checklist)
- Modify: `docs/how-it-works.md` (a new subsection at the end of "At capture time, on the Pi")
- Modify: `CLAUDE.md` (Capture section)

- [ ] **Step 1: `docs/lcd-viewfinder.md`.** Add a row to the §5 table, after the battery badge row:

```markdown
| `2x` toggle (pill, top-left) | Outlined `2x` when off; amber `2x 0/2` / `2x 1/2` when on | Turns double exposure on or off. Off at `1/2` discards the pending first exposure (its original stays on disk) |
```

Add a subsection after "### Idle dimming" and its paragraphs, before "## 6.":

```markdown
### Double exposure

With `2x` on, every two shots become one picture, the way two exposures on one frame
of colour negative film do. Light adds: a dark area in one frame lets the other show
through, and bright areas stack. Any trigger counts: the shutter button, the Stick,
or both mixed.

1. **Exposure 1:** colour bars labelled `Exposure 1 of 2...`, then `Exposure 1/2` for
   about a second, then the live view with the badge at `1/2`. Only the original is
   saved; the Stick shows an "Exposure 1/2" card.
2. **Exposure 2:** colour bars labelled `Developing double exposure...`, then the
   review screen with the composite (caption starts `2x`). The badge returns to
   `0/2` and the mode stays on.

Files: each exposure keeps its own `HHMMSS_original.jpg` (and `.dng`). The composite
is `HHMMSS_double_graded.jpg`, named after exposure 2. Its `captures.jsonl` line has a
`double` block naming both originals, their EVs and the method (`linear_mean`).
Exposure compensation per frame is the film shooter's control: shoot the frame you
want to recede at −1 EV. The mode is not remembered across restarts. A camera error on
exposure 2 keeps `1/2`, so just shoot again. See [how it works](how-it-works.md#double-exposure)
for why the frames are added before grading.
```

Append to the §7 checklist:

```markdown
11. Double exposure: tap `2x` → badge `2x 0/2` in amber. Shoot a dark silhouette
    against a bright window, then a textured subject: `Exposure 1/2` shows briefly and
    the badge reads `1/2`; the second shot reviews a composite in which the texture
    fills the silhouette. The day folder gains two originals and one
    `_double_graded.jpg`, and no `_graded.jpg` for exposure 1. Repeat with the Stick
    as the trigger: it shows the `Exposure 1/2` card, then the composite. Tap `2x` at
    `1/2`: the badge goes to the outlined `2x`. `systemctl restart pifilm-capture`:
    the mode comes back off.
```

- [ ] **Step 2: `docs/how-it-works.md`.** Insert before `## At training time, on the Mac`:

```markdown
### Double exposure

With the viewfinder's `2x` toggle on, two captures make one picture
(`pifilm/double.py`). Film adds exposures, H = H₁ + H₂, and is developed once, so the
two *ungraded* frames are added in linear light and the sum goes through the three
steps above once. Grading each frame and blending the results would apply the LUT's
tone curve twice, add values after the curve instead of light before it, lay grain
down twice, and normalise twice, which normalisation does not support. Each frame is
taken one stop down before the sum (`0.5·L₁ + 0.5·L₂`), the usual film starting
point: the camera auto-exposes every frame fully, so a straight sum would sit a stop
over and clip before the LUT's shoulder could act. Each frame keeps the EV it was
shot at, so a frame shot darker contributes less light. The composite is graded at
EV 0, and normalisation sets the pair's overall exposure.
```

- [ ] **Step 3: `CLAUDE.md`.** In the Capture list, after the `app.py` bullet, add:

```markdown
- Double exposure: `CaptureSession` owns the mode and pending first frame; `pifilm/double.py`
  adds the two ungraded frames in linear light (one stop down each), and the composite is
  graded once. Exposure 1's `CaptureResult.pifilm` is an "Exposure 1/2" card in a temp dir
  (never under the synced output folder); toggled via `CaptureController.set_double_exposure`.
```

- [ ] **Step 4: Verify links and lint**

Run: `grep -n "double-exposure" docs/lcd-viewfinder.md && grep -n "### Double exposure" docs/how-it-works.md && .venv/bin/ruff check .`
Expected: the anchor link and the heading both found; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add docs/lcd-viewfinder.md docs/how-it-works.md CLAUDE.md
git commit -m "Double exposure: document the toggle, files, rationale and acceptance

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
