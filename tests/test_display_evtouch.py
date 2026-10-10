import struct

import pytest

from pifilm.display import DisplayError
from pifilm.display.evtouch import (
    ABS_MT_POSITION_X,
    ABS_MT_POSITION_Y,
    ABS_MT_SLOT,
    ABS_MT_TRACKING_ID,
    EV_ABS,
    EV_SYN,
    EventTouch,
    SlotTracker,
    decode_events,
    open_goodix_touch,
)
from pifilm.display.touch import TapDetector, TouchPoint

SYN = (EV_SYN, 0, 0)


def _pack(*events):
    return b"".join(struct.pack("<qqHHi", 0, 0, *event) for event in events)


def _press(x, y, slot=0, tracking_id=7):
    return [(EV_ABS, ABS_MT_SLOT, slot), (EV_ABS, ABS_MT_TRACKING_ID, tracking_id),
            (EV_ABS, ABS_MT_POSITION_X, x), (EV_ABS, ABS_MT_POSITION_Y, y), SYN]


def _release(slot=0):
    return [(EV_ABS, ABS_MT_SLOT, slot), (EV_ABS, ABS_MT_TRACKING_ID, -1), SYN]


class Feed:
    """A scripted device: one chunk per ``EventTouch.read``.

    ``read`` keeps calling until it gets nothing, so each non-empty chunk is
    followed by one empty answer. An empty chunk stands for a poll with no events.
    """

    def __init__(self, *chunks):
        self.chunks = list(chunks)
        self._drained = False

    def __call__(self):
        if self._drained:
            self._drained = False
            return b""
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        self._drained = bool(chunk)
        return chunk


def _touch(feed, rotate=0):
    return EventTouch(feed, (0, 639), (0, 479), (640, 480), rotate=rotate)


def test_decode_events_reads_whole_records_and_drops_a_partial_tail():
    data = _pack((EV_ABS, ABS_MT_POSITION_X, 300), SYN) + b"\x01\x02\x03"
    assert decode_events(data) == [(EV_ABS, ABS_MT_POSITION_X, 300), SYN]


def test_finger_down_reports_one_point_and_release_reports_none():
    feed = Feed(_pack(*_press(320, 240)), b"", _pack(*_release()))
    touch = _touch(feed)
    assert touch.read() == [TouchPoint(320, 240, 0)]
    assert touch.read() == [TouchPoint(320, 240, 0)]   # still held, nothing new
    assert touch.read() == []


def test_a_press_and_release_inside_one_read_is_still_seen_once():
    touch = _touch(Feed(_pack(*_press(100, 80), *_release())))
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == []


def test_a_fast_tap_reaches_the_tap_detector():
    touch = _touch(Feed(_pack(*_press(100, 80), *_release())))
    detector = TapDetector()
    assert detector.feed(touch.read(), 0.00) is None
    tap = detector.feed(touch.read(), 0.02)
    assert (tap.x, tap.y) == (100, 80)


def test_state_is_committed_only_on_syn_report():
    """A read that ends mid-report must not expose the previous touch's position."""
    first = _pack(*_press(600, 400), *_release())
    half = _pack((EV_ABS, ABS_MT_SLOT, 0), (EV_ABS, ABS_MT_TRACKING_ID, 8))
    rest = _pack((EV_ABS, ABS_MT_POSITION_X, 50), (EV_ABS, ABS_MT_POSITION_Y, 60), SYN)
    touch = _touch(Feed(first, half, rest))
    assert touch.read() == [TouchPoint(600, 400, 0)]
    assert touch.read() == []                           # press not yet complete
    assert touch.read() == [TouchPoint(50, 60, 0)]      # never (600, 400)


def test_other_slots_are_ignored():
    touch = _touch(Feed(_pack(*_press(100, 80)), _pack(*_press(500, 300, slot=1, tracking_id=9)),
                        _pack(*_release(slot=1))))
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == [TouchPoint(100, 80, 0)]
    assert touch.read() == [TouchPoint(100, 80, 0)]     # slot 1 lifting is not slot 0 lifting


def test_coordinates_scale_and_clamp_to_the_screen():
    feed = Feed(_pack(*_press(1023, 767)), _pack(*_release()), _pack(*_press(5000, -40)))
    touch = EventTouch(feed, (0, 1023), (0, 767), (640, 480))
    assert touch.read() == [TouchPoint(639, 479, 0)]
    assert touch.read() == []
    assert touch.read() == [TouchPoint(639, 0, 0)]


def test_rotate_180_mirrors_both_axes():
    touch = _touch(Feed(_pack(*_press(0, 0))), rotate=180)
    assert touch.read() == [TouchPoint(639, 479, 0)]


def test_a_read_error_is_a_display_error():
    def broken():
        raise OSError(19, "No such device")

    with pytest.raises(DisplayError, match="touch read failed"):
        _touch(broken).read()


def test_bad_rotation_and_degenerate_range_are_rejected():
    with pytest.raises(DisplayError, match="rotate"):
        _touch(Feed(), rotate=90)
    with pytest.raises(DisplayError, match="range"):
        EventTouch(Feed(), (0, 0), (0, 479), (640, 480))


def test_slot_tracker_take_clears_the_pressed_latch():
    tracker = SlotTracker()
    tracker.feed(_press(10, 20) + _release())
    assert tracker.take() == (10, 20)
    assert tracker.take() is None


def test_open_reports_a_missing_device(tmp_path):
    with pytest.raises(DisplayError, match="Goodix Capacitive TouchScreen"):
        open_goodix_touch((640, 480), input_dir=tmp_path)


def test_close_calls_the_closer_once():
    closed = []
    touch = EventTouch(Feed(), (0, 639), (0, 479), (640, 480), closer=lambda: closed.append(1))
    touch.close()
    touch.close()
    assert closed == [1]
