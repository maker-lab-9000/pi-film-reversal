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


def test_steps_above_the_sensors_limit_are_not_offered():
    """An IMX477-like limit of gain 8 is ISO 800: 1600 would be refused by the camera."""
    assert higher(None, 1250, max_iso=800) == 800      # nearest overall is 1600
    assert lower(None, 1250, max_iso=800) == 800
    assert higher(800, None, max_iso=800) == 800       # the top of the shortened scale
    assert higher(400, None, max_iso=800) == 800
    assert lower(800, None, max_iso=800) == 400
    assert higher(None, None, max_iso=800) == 400      # the default is still on the scale


def test_a_limit_between_steps_keeps_the_steps_below_it():
    assert higher(800, None, max_iso=1599.9) == 800
    assert higher(800, None, max_iso=2226) == 1600


def test_a_default_start_above_the_limit_falls_to_the_top_offered_step():
    assert higher(None, None, max_iso=200) == 200
    assert lower(None, None, max_iso=200) == 200


def test_a_limit_below_every_step_leaves_the_lowest_step():
    assert higher(None, 1250, max_iso=50) == 100
    assert higher(100, None, max_iso=50) == 100
    assert lower(100, None, max_iso=50) is None


@pytest.mark.parametrize("bad", [None, 0, -8, float("nan"), float("inf"), True, "800"])
def test_a_meaningless_limit_is_no_limit(bad):
    assert higher(800, None, max_iso=bad) == 1600
