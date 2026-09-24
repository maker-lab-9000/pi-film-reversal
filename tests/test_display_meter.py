import numpy as np
import pytest

from pifilm.display.meter import (
    FOCUS_DECAY_PER_SECOND,
    FocusTracker,
    compute_reading,
    focus_score,
    format_ev,
    format_shutter,
)


def _grey(value, shape=(48, 64, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_format_shutter():
    assert format_shutter(4000) == "1/250"
    assert format_shutter(33333) == "1/30"
    assert format_shutter(1_300_000) == "1.3s"
    assert format_shutter(1_000_000) == "1.0s"


def test_format_ev():
    assert format_ev(0.0) == "0"
    assert format_ev(1 / 3) == "+0.3"
    assert format_ev(-4 / 3) == "-1.3"


def test_reading_from_metadata():
    meta = {"ExposureTime": 4000, "AnalogueGain": 2.0, "DigitalGain": 1.5, "Lux": 640.4}
    r = compute_reading(meta, _grey(118), 1 / 3, None)
    assert r.shutter == "1/250"
    assert r.iso == 300
    assert r.lux == pytest.approx(640.4)
    assert r.ev_comp == pytest.approx(1 / 3)
    assert r.battery_percent is None and r.external_power is None


def test_mid_grey_frame_reads_zero_deviation():
    # sRGB 118 ≈ 18 % linear reflectance
    r = compute_reading({}, _grey(118), 0.0, None)
    assert r.deviation_ev == pytest.approx(0.0, abs=0.05)
    assert r.clip_pct == 0.0


def test_dark_and_bright_frames_swing_and_clamp():
    assert compute_reading({}, _grey(0), 0.0, None).deviation_ev == -3.0
    assert compute_reading({}, _grey(255), 0.0, None).deviation_ev == pytest.approx(2.47, abs=0.05)


def test_clip_pct_counts_any_channel_at_ceiling():
    frame = _grey(50)
    frame[:24, :, 1] = 254
    assert compute_reading({}, frame, 0.0, None).clip_pct == pytest.approx(50.0)


def test_missing_metadata_gives_none_fields_without_raising():
    r = compute_reading(None, _grey(118), 0.0, None)
    assert r.shutter is None and r.iso is None and r.lux is None


def test_battery_fields_come_from_power():
    class P:
        percent = 87
        external_power = True

    r = compute_reading({}, _grey(118), 0.0, P())
    assert (r.battery_percent, r.external_power) == (87, True)


def _scene(shape=(96, 128), seed=0):
    """A textured test scene: rectangles of random size and grey level, so it has
    edges at every scale and orientation like a real subject, not one frequency."""
    rng = np.random.default_rng(seed)
    frame = np.full(shape, 128.0)
    for _ in range(60):
        h, w = rng.integers(3, shape[0] // 3), rng.integers(3, shape[1] // 3)
        y, x = rng.integers(0, shape[0] - h), rng.integers(0, shape[1] - w)
        frame[y:y + h, x:x + w] = rng.integers(20, 236)
    return np.repeat(frame.astype(np.uint8)[:, :, None], 3, axis=2)


def _box(rgb, k):
    out = rgb.astype(np.float32)
    pad = k // 2
    padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(out)
    for dy in range(k):
        for dx in range(k):
            acc += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return acc / (k * k)


def _blur(rgb, k=5):
    """Near-Gaussian blur (three box passes, sigma ~ k/2) without OpenCV. A single
    box turns an edge into a straight ramp, which no lens does and which the
    re-blur in ``focus_score`` cannot tell from sharp at its middle."""
    return np.clip(_box(_box(_box(rgb, k), k), k), 0, 255).astype(np.uint8)


def _contrast(rgb, gain):
    return np.clip(128 + (rgb.astype(np.float32) - 128) * gain, 0, 255).astype(np.uint8)


def _noisy(rgb, sigma, seed=1):
    noise = np.random.default_rng(seed).normal(0, sigma, rgb.shape)
    return np.clip(rgb + noise, 0, 255).astype(np.uint8)


def test_focus_score_is_zero_for_a_flat_frame():
    assert focus_score(_grey(128)) == 0.0


def test_focus_score_reads_high_on_a_sharp_scene_and_low_on_a_blurred_one():
    """An absolute reading: a badly blurred frame must read low on its own,
    with no sharp frame seen first. This is the defect found on the IMX477: the
    old bar was relative to its own recent peak, so a lens left out of focus read
    full."""
    sharp = _scene()
    assert focus_score(sharp) >= 0.8
    assert focus_score(_blur(sharp, 15)) <= 0.2


def test_focus_score_falls_steadily_as_the_blur_grows():
    sharp = _scene()
    scores = [focus_score(sharp)] + [focus_score(_blur(sharp, k)) for k in (3, 5, 9, 15)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] - scores[-1] > 0.6


def test_focus_score_does_not_depend_on_the_scene_contrast():
    """A dim, flat subject in sharp focus must read as sharp as a bright one."""
    sharp, soft = _scene(), _blur(_scene(), 9)
    assert focus_score(_contrast(sharp, 0.3)) == pytest.approx(focus_score(sharp), abs=0.1)
    assert focus_score(_contrast(soft, 0.3)) == pytest.approx(focus_score(soft), abs=0.1)


def test_sensor_noise_does_not_make_a_blurred_frame_read_sharp():
    """High ISO in a dim room (1/15 s, ISO 1580 on the IMX477) puts pixel-scale
    noise on every frame; a plain Laplacian reads that noise as detail."""
    soft = _contrast(_blur(_scene(), 15), 0.3)
    for sigma in (4, 8):
        assert focus_score(_noisy(soft, sigma)) <= 0.2


def test_pure_noise_reads_zero_rather_than_sharp():
    assert focus_score(_noisy(_grey(128), 8)) <= 0.1


def test_focus_score_only_looks_at_the_central_half():
    """Detail outside the central region must not move the score: the user
    focuses on what is in the middle of the frame."""
    flat = _grey(128, shape=(80, 120, 3))
    assert focus_score(flat) == 0.0
    edged = flat.copy()
    edged[:20] = _scene((20, 120))
    edged[-20:] = _scene((20, 120))
    edged[:, :30] = _scene((80, 30))
    edged[:, -30:] = _scene((80, 30))
    assert focus_score(edged) == 0.0


def test_focus_score_returns_a_plain_float_in_the_unit_range():
    score = focus_score(_scene())
    assert isinstance(score, float) and 0.0 <= score <= 1.0


def test_focus_tracker_peak_follows_the_best_level_seen():
    tracker = FocusTracker()
    assert tracker.update(0.3, 0.0) == pytest.approx(0.3)
    assert tracker.update(0.9, 0.1) == pytest.approx(0.9)
    # Racked past best focus: the level falls, the mark stays near the best.
    assert tracker.update(0.4, 0.2) == pytest.approx(0.9 * FOCUS_DECAY_PER_SECOND ** 0.1)


def test_focus_tracker_peak_decays_so_a_new_subject_is_not_held_to_an_old_one():
    tracker = FocusTracker(decay_per_second=0.5)
    tracker.update(0.8, 0.0)
    assert tracker.update(0.1, 2.0) == pytest.approx(0.2)


def test_focus_tracker_peak_never_sits_below_the_current_level():
    tracker = FocusTracker()
    tracker.update(0.9, 0.0)
    assert tracker.update(0.95, 30.0) == pytest.approx(0.95)


def test_focus_tracker_clamps_to_the_unit_range():
    tracker = FocusTracker()
    assert tracker.update(-5.0, 0.0) == 0.0
    assert tracker.update(7.0, 0.1) == 1.0


def test_reading_carries_focus_when_given_and_none_otherwise():
    r = compute_reading({}, _grey(118), 0.0, None)
    assert r.focus is None and r.focus_peak is None
    r = compute_reading({}, _grey(118), 0.0, None, focus=0.4, focus_peak=0.7)
    assert r.focus == pytest.approx(0.4) and r.focus_peak == pytest.approx(0.7)
