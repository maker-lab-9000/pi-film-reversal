import numpy as np
import pytest

from pifilm.display.meter import compute_reading, format_ev, format_shutter


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
