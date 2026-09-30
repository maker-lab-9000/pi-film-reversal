# tests/test_display_shutter.py
import pytest

from pifilm.display.shutter import (
    DEFAULT_START_US,
    SHUTTER_STEPS,
    SHUTTER_STEPS_US,
    faster,
    label,
    slower,
)


def test_series_is_one_third_stops_from_1_2000_to_1_second():
    assert len(SHUTTER_STEPS_US) == 34
    assert SHUTTER_STEPS_US[0] == 500 and SHUTTER_STEPS_US[-1] == 1_000_000
    assert list(SHUTTER_STEPS_US) == sorted(set(SHUTTER_STEPS_US))
    assert [us for us, _ in SHUTTER_STEPS] == list(SHUTTER_STEPS_US)


def test_labels_are_ascii_and_conventional():
    assert label(None) == "A"
    assert label(4000) == "1/250"
    assert label(1563) == "1/640"
    assert label(300_000) == "0.3s"
    assert label(1_000_000) == "1s"
    assert all(text.isascii() for _, text in SHUTTER_STEPS)


def test_from_auto_the_first_tap_lands_on_the_metered_step():
    assert faster(None, 4100.0) == 4000
    assert slower(None, 4100.0) == 4000
    # nearest in stops, not in microseconds
    assert faster(None, 14_000.0) == 12_500


def test_from_auto_without_a_metered_value_starts_at_1_125():
    assert DEFAULT_START_US == 8000
    assert faster(None, None) == 8000
    assert slower(None, 0.0) == 8000


def test_steps_move_one_third_stop():
    assert faster(4000, None) == 3125
    assert slower(4000, None) == 5000


def test_fastest_step_stays_put_and_slowest_returns_to_auto():
    assert faster(500, None) == 500
    assert slower(1_000_000, None) is None


def test_an_off_series_value_snaps_to_the_nearest_step_first():
    assert faster(4100, None) == 3125
    assert slower(4100, None) == 5000


@pytest.mark.parametrize("fn", [faster, slower])
def test_every_step_is_reachable(fn):
    seen = set()
    value = 500 if fn is slower else 1_000_000
    for _ in range(40):
        seen.add(value)
        value = fn(value, None)
        if value is None:
            break
    assert set(SHUTTER_STEPS_US) <= seen
