from pifilm.display import cst3530, touch
from pifilm.display.touch import Tap, TapDetector, TouchPoint


def test_a_quick_press_and_release_is_a_tap_at_the_press_position():
    detector = TapDetector()
    assert detector.feed([TouchPoint(100, 50, 0)], 0.0) is None
    assert detector.feed([TouchPoint(104, 52, 0)], 0.1) is None
    assert detector.feed([], 0.2) == Tap(100, 50)


def test_a_hold_or_a_drag_is_not_a_tap():
    detector = TapDetector()
    detector.feed([TouchPoint(100, 50, 0)], 0.0)
    assert detector.feed([], 1.0) is None
    detector.feed([TouchPoint(100, 50, 0)], 2.0)
    detector.feed([TouchPoint(160, 50, 0)], 2.1)
    assert detector.feed([], 2.2) is None


def test_max_move_is_configurable_for_a_denser_panel():
    detector = TapDetector(max_move=40.0)
    detector.feed([TouchPoint(100, 50, 0)], 0.0)
    detector.feed([TouchPoint(130, 50, 0)], 0.1)
    assert detector.feed([], 0.2) == Tap(100, 50)


def test_the_2_8_inch_driver_still_exports_the_shared_types():
    assert cst3530.TouchPoint is touch.TouchPoint
    assert cst3530.Tap is touch.Tap
    assert cst3530.TapDetector is touch.TapDetector
