import pytest

from pifilm.display.iso import DEFAULT_START, ISO_STEPS, higher, label, lower


def test_walk_up_from_auto_with_a_metered_value_then_to_the_top():
    seen, value = [], None
    for _ in range(5):
        value = higher(value, 380)
        seen.append(value)
    assert seen == [400, 800, 1600, 1600, 1600]


def test_walk_down_to_auto():
    seen, value = [], 1600
    for _ in range(6):
        value = lower(value, None)
        seen.append(value)
    assert seen == [800, 400, 200, 100, None, 400]


def test_lower_from_the_bottom_is_auto():
    assert lower(100, 380) is None


@pytest.mark.parametrize("metered", [None, 0, float("nan")])
def test_without_a_metered_value_the_start_is_400(metered):
    assert DEFAULT_START == 400
    assert higher(None, metered) == 400
    assert lower(None, metered) == 400


def test_labels():
    assert label(400) == "400"
    assert all(label(i).isascii() for i in ISO_STEPS)
