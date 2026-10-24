"""The multi-frame merge (pifilm.stack): alignment, noise, ghosts, drops, bands."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pifilm._cv2 import require_cv2
from pifilm.color import SRGB8_LINEAR
from pifilm.stack import METHOD, MIN_CORRELATION, REASONS, merge, merge_aligned, unused_entry

cv2 = require_cv2()

_REFERENCE_ENTRY = {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "rho": None, "used": True,
                    "reason": None}

H, W = 1200, 1800  # reduced by 2 for alignment


def _scene(seed: int = 1, height: int = H, width: int = W) -> np.ndarray:
    """Textured RGB float scene in 30..220 (blurred noise at two scales)."""
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(rng.random((height, width, 3), dtype=np.float32), (0, 0), 3)
    coarse = cv2.GaussianBlur(rng.random((height, width, 3), dtype=np.float32), (0, 0), 25)
    img = (fine - fine.mean()) / fine.std() + 2.0 * (coarse - coarse.mean()) / coarse.std()
    img = 125.0 + 30.0 * img
    return np.clip(img, 30.0, 220.0).astype(np.float32)


def _truth(dx: float, dy: float, angle_deg: float, width: int = W, height: int = H):
    """A 2x3 map from reference pixels to frame pixels: rotation about the centre
    plus a shift."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    m = np.empty((2, 3))
    m[:, :2] = rot
    m[:, 2] = centre - rot @ centre + np.array([dx, dy])
    return m


def _frame_from(scene: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """frame(M x) = scene(x): warp the scene forward by ``truth``."""
    h, w = scene.shape[:2]
    out = cv2.warpAffine(scene, truth, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _corner_error(found: dict, truth: np.ndarray, width: int = W, height: int = H) -> float:
    """Largest distance at the four corners between the found and the true map."""
    est = _truth(found["dx"], found["dy"], found["angle_deg"], width, height)
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1],
                        [width - 1, height - 1, 1]], dtype=np.float64).T
    return float(np.max(np.linalg.norm(est @ corners - truth @ corners, axis=0)))


def test_known_shifts_and_rotations_are_recovered():
    scene = _scene()
    truths = [_truth(0, 0, 0), _truth(6.3, -4.7, 0.0), _truth(-11.2, 3.4, 0.4),
              _truth(2.5, 8.1, -0.7)]
    frames = [_frame_from(scene, t) for t in truths]
    out, info = merge(frames, reference=0)
    assert out.shape == frames[0].shape and out.dtype == np.uint8
    assert info["reference"] == 0
    assert info["method"] == METHOD == "ecc-euclidean-ghost-weighted-mean"
    assert len(info["frames"]) == 4
    for entry, truth in zip(info["frames"], truths, strict=True):
        assert entry["used"] is True and entry["reason"] is None
        assert _corner_error(entry, truth) < 0.5
    assert info["frames"][0]["rho"] is None  # the reference is not fitted
    assert all(MIN_CORRELATION <= e["rho"] <= 1.0 for e in info["frames"][1:])
    # The aligned frames agree with the reference away from the reflected border.
    inner = (slice(60, H - 60), slice(60, W - 60))
    diff = np.abs(out[inner].astype(int) - frames[0][inner].astype(int))
    assert diff.mean() < 1.5
    json.dumps(info)  # plain floats and bools only
    for entry in info["frames"]:
        assert list(entry) == ["dx", "dy", "angle_deg", "rho", "used", "reason"]
        assert all(type(entry[k]) is float for k in ("dx", "dy", "angle_deg"))
    assert type(info["align_ms"]) is float and type(info["merge_ms"]) is float


def test_reference_other_than_first():
    scene = _scene(seed=3)
    truths = [_truth(5.0, 2.0, 0.2), _truth(0, 0, 0), _truth(-3.0, 7.5, -0.3)]
    frames = [_frame_from(scene, t) for t in truths]
    _, info = merge(frames, reference=1)
    assert info["reference"] == 1
    assert info["frames"][1] == _REFERENCE_ENTRY
    for i in (0, 2):
        assert _corner_error(info["frames"][i], truths[i]) < 0.5


@pytest.mark.parametrize("n", [2, 4, 6])
def test_noise_falls_about_as_one_over_root_n(n):
    scene = _scene(seed=5)
    scene[300:700, 500:1100] = 128.0          # a flat grey patch
    rng = np.random.default_rng(100 + n)
    sigma = 4.0
    frames = [_u8(scene + rng.normal(0, sigma, scene.shape)) for _ in range(n)]
    out, info = merge(frames, reference=0)
    assert all(e["used"] for e in info["frames"])
    patch = (slice(320, 680), slice(520, 1080))
    single = frames[0][patch].astype(np.float64).std()
    merged = out[patch].astype(np.float64).std()
    expected = single / math.sqrt(n)
    assert 0.8 * expected < merged < 1.3 * expected


def test_moving_square_comes_only_from_the_reference():
    scene = _scene(seed=7)
    frames = []
    for i in range(4):
        img = scene.copy()
        top, left = 400, 300 + 250 * i
        img[top:top + 160, left:left + 160] = 250.0
        frames.append(_u8(img))
    out, info = merge(frames, reference=2)
    assert all(e["used"] for e in info["frames"])
    # Where the square is in the reference, the reference's pixels.
    sq = (slice(400, 560), slice(800, 960))
    assert np.abs(out[sq].astype(int) - frames[2][sq].astype(int)).max() <= 1
    # Where it was in the other frames, no ghost: the scene, as the reference has it.
    for i in (0, 1, 3):
        left = 300 + 250 * i
        ghost = (slice(400, 560), slice(left, left + 160))
        if i == 1:  # overlaps the reference's square: only check the free part
            ghost = (slice(400, 560), slice(left, 800))
        # A ghost (a plain mean) would be about a quarter of the way to white; the
        # alignment moves a few hundredths of a pixel, a level or two at edges.
        diff = np.abs(out[ghost].astype(int) - frames[2][ghost].astype(int))
        assert diff.max() <= 6 and diff.mean() < 0.5
        plain = np.mean([SRGB8_LINEAR[f[ghost]] for f in frames], axis=0)
        assert (plain - SRGB8_LINEAR[frames[2][ghost]]).max() > 0.15


def test_a_huge_shift_is_dropped():
    scene = _scene(seed=9)
    big = 0.2 * W
    truths = [_truth(0, 0, 0), _truth(3.0, -2.0, 0.0), _truth(big, 0.0, 0.0)]
    frames = [_frame_from(scene, t) for t in truths]
    out, info = merge(frames, reference=0)
    assert info["frames"][1]["used"] is True
    assert info["frames"][2]["used"] is False
    # ECC does not reach a shift that large: it ends on a poor fit.
    assert info["frames"][2]["reason"] in ("shift", "low_correlation", "ecc_failed")
    # The result equals the merge of the two used frames alone.
    alone, _ = merge(frames[:2], reference=0)
    assert np.array_equal(out, alone)


def test_an_unrelated_frame_is_dropped():
    frames = [_u8(_scene(seed=11)), _u8(_scene(seed=12))]
    out, info = merge(frames, reference=0)
    assert info["frames"][1]["used"] is False
    assert info["frames"][1]["reason"] in ("ecc_failed", "low_correlation")
    assert out is frames[0]


def test_one_frame_comes_back_unchanged():
    frame = _u8(_scene(seed=13, height=240, width=320))
    out, info = merge([frame], reference=0)
    assert np.array_equal(out, frame)
    assert info["frames"] == [_REFERENCE_ENTRY]
    json.dumps(info)


def test_bands_give_the_same_result_as_one_band():
    rng = np.random.default_rng(17)
    scene = _scene(seed=17, height=300, width=400)
    ref = _u8(scene + rng.normal(0, 5, scene.shape))
    others = [_u8(scene + rng.normal(0, 5, scene.shape)) for _ in range(3)]
    others[1][100:150, 100:200] = 255
    whole = merge_aligned(ref, others, band_pixels=300 * 400)
    for band_pixels in (400, 7 * 400 + 13, 64 * 400):
        assert np.array_equal(merge_aligned(ref, others, band_pixels=band_pixels), whole)


def test_merge_is_deterministic():
    scene = _scene(seed=19)
    frames = [_frame_from(scene, t) for t in (_truth(0, 0, 0), _truth(1.5, 2.5, 0.1))]
    a, ia = merge(frames, reference=0)
    b, ib = merge(frames, reference=0)
    assert np.array_equal(a, b)
    assert ia["frames"] == ib["frames"]


def test_identical_frames_merge_to_the_frame():
    frame = _u8(_scene(seed=23, height=200, width=300))
    out = merge_aligned(frame, [frame.copy(), frame.copy()])
    assert np.array_equal(out, frame)


def test_merge_is_a_mean_in_linear_light():
    a = np.full((4, 4, 3), 50, np.uint8)
    b = np.full((4, 4, 3), 54, np.uint8)  # close: weight near 1
    out = merge_aligned(a, [b])
    la, lb = SRGB8_LINEAR[50], SRGB8_LINEAR[54]
    d = lb - la
    w = math.exp(-((d / 0.05) ** 2))
    expected = (la + w * lb) / (1 + w)
    from pifilm.color import linear_to_srgb
    assert out[0, 0, 0] == int(np.round(float(linear_to_srgb(np.float32(expected))) * 255))


@pytest.mark.parametrize(
    "frames, reference, match",
    [
        ([], 0, "at least one"),
        ([np.zeros((4, 4), np.uint8)], 0, "RGB uint8"),
        ([np.zeros((4, 4, 3), np.float32)], 0, "RGB uint8"),
        ([np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8)], 0, "differ"),
        ([np.zeros((4, 4, 3), np.uint8)], 1, "frame index"),
        ([np.zeros((4, 4, 3), np.uint8)], -1, "frame index"),
    ],
)
def test_bad_input_is_refused(frames, reference, match):
    with pytest.raises(ValueError, match=match):
        merge(frames, reference=reference)


def test_a_large_roll_is_dropped():
    scene = _scene(seed=29)
    truths = [_truth(0, 0, 0), _truth(0, 0, 3.0)]
    frames = [_frame_from(scene, t) for t in truths]
    out, info = merge(frames, reference=0)
    assert info["frames"][1]["used"] is False
    assert info["frames"][1]["reason"] == "angle"
    assert out is frames[0]


def _two_frames():
    scene = _scene(seed=31, height=240, width=320)
    return [_u8(scene), _u8(scene)]


def test_ecc_that_raises_is_recorded_as_ecc_failed(monkeypatch):
    def failing(*args, **kwargs):
        raise cv2.error("no convergence")

    monkeypatch.setattr(cv2, "findTransformECC", failing)
    frames = _two_frames()
    out, info = merge(frames, reference=0)
    assert out is frames[0]
    assert info["frames"][1] == {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "rho": None,
                                 "used": False, "reason": "ecc_failed"}
    json.dumps(info)


def test_a_low_correlation_is_recorded_with_its_rho(monkeypatch):
    monkeypatch.setattr(cv2, "findTransformECC",
                        lambda *a, **k: (0.31234567, np.eye(2, 3, dtype=np.float32)))
    frames = _two_frames()
    out, info = merge(frames, reference=0)
    assert out is frames[0]
    entry = info["frames"][1]
    assert (entry["rho"], entry["used"], entry["reason"]) == (0.3123, False, "low_correlation")
    assert type(entry["rho"]) is float


def test_a_non_finite_correlation_is_ecc_failed(monkeypatch):
    monkeypatch.setattr(cv2, "findTransformECC",
                        lambda *a, **k: (float("nan"), np.eye(2, 3, dtype=np.float32)))
    _, info = merge(_two_frames(), reference=0)
    assert info["frames"][1]["rho"] is None and info["frames"][1]["reason"] == "ecc_failed"
    json.dumps(info, allow_nan=False)


def test_unused_entry_is_a_dropped_frame_for_a_reason():
    assert unused_entry("unreadable") == {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "rho": None,
                                          "used": False, "reason": "unreadable"}
    assert set(REASONS) == {"ecc_failed", "low_correlation", "shift", "angle", "unreadable"}
    with pytest.raises(ValueError, match="unknown reason"):
        unused_entry("blurred")


def test_a_good_fit_beyond_the_shift_limit_is_recorded_as_shift(monkeypatch):
    warp = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]], np.float32)  # 40 of 320 px
    monkeypatch.setattr(cv2, "findTransformECC", lambda *a, **k: (0.97, warp))
    frames = _two_frames()
    out, info = merge(frames, reference=0)
    entry = info["frames"][1]
    assert out is frames[0]
    assert (entry["rho"], entry["used"], entry["reason"]) == (0.97, False, "shift")
    assert entry["dx"] == 40.0
