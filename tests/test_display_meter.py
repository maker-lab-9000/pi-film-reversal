import numpy as np
import pytest

from pifilm.display.meter import (
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


def _checkerboard(shape=(48, 64), block=4):
    ys, xs = np.indices(shape)
    frame = np.where(((ys // block) + (xs // block)) % 2 == 0, 20, 235).astype(np.uint8)
    return np.repeat(frame[:, :, None], 3, axis=2)


def _blur(rgb, k=5):
    """Box-blur with a cumulative sum, so the test needs no OpenCV."""
    out = rgb.astype(np.float32)
    pad = k // 2
    padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(out)
    for dy in range(k):
        for dx in range(k):
            acc += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return (acc / (k * k)).astype(np.uint8)


def test_focus_score_is_zero_for_a_flat_frame():
    assert focus_score(_grey(128)) == 0.0


def test_focus_score_falls_when_the_frame_is_blurred():
    sharp = _checkerboard()
    assert focus_score(sharp) > focus_score(_blur(sharp)) > 0.0


def test_focus_score_only_looks_at_the_central_half():
    """Detail outside the central region must not move the score: the user
    focuses on what is in the middle of the frame."""
    flat = _grey(128, shape=(80, 120, 3))
    assert focus_score(flat) == 0.0
    edged = flat.copy()
    edged[:20] = _checkerboard((20, 120))
    edged[-20:] = _checkerboard((20, 120))
    edged[:, :30] = _checkerboard((80, 30))
    edged[:, -30:] = _checkerboard((80, 30))
    assert focus_score(edged) == 0.0


def test_focus_score_returns_a_plain_float():
    assert isinstance(focus_score(_checkerboard()), float)


def test_focus_tracker_first_score_sets_the_peak_and_reads_full():
    tracker = FocusTracker()
    assert tracker.update(100.0, 0.0) == pytest.approx(1.0)


def test_focus_tracker_drops_when_focus_is_racked_past():
    tracker = FocusTracker(decay_per_second=0.5)
    tracker.update(100.0, 0.0)
    # One frame later the peak has decayed only slightly (0.5 ** 0.1), so a quarter of
    # the sharpness still reads as roughly a quarter of the bar.
    level = tracker.update(25.0, 0.1)
    assert level == pytest.approx(25.0 / (100.0 * 0.5**0.1), rel=1e-6)
    assert 0.2 < level < 0.3


def test_focus_tracker_peak_decays_so_a_new_subject_can_peak_again():
    tracker = FocusTracker(decay_per_second=0.5)
    tracker.update(100.0, 0.0)
    # 2 s later the peak has halved twice: 100 -> 25, so a score of 25 reads full again.
    assert tracker.update(25.0, 2.0) == pytest.approx(1.0)


def test_focus_tracker_rising_score_always_reads_full():
    tracker = FocusTracker()
    for t, score in enumerate([10.0, 50.0, 400.0]):
        assert tracker.update(score, t * 0.1) == pytest.approx(1.0)


def test_focus_tracker_reads_zero_on_a_featureless_scene():
    tracker = FocusTracker()
    assert tracker.update(0.0, 0.0) == 0.0
    assert tracker.update(0.0, 1.0) == 0.0


def test_focus_tracker_level_is_clamped_to_the_unit_range():
    tracker = FocusTracker()
    tracker.update(100.0, 0.0)
    assert 0.0 <= tracker.update(-5.0, 0.1) <= 1.0


def test_reading_carries_focus_when_given_and_none_otherwise():
    assert compute_reading({}, _grey(118), 0.0, None).focus is None
    assert compute_reading({}, _grey(118), 0.0, None, focus=0.4).focus == pytest.approx(0.4)
