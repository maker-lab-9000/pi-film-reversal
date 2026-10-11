import pytest

from pifilm.display.stepscale import down, is_auto, nearest, up

STEPS = (100, 200, 400)
BAD = [None, 0, -3, float("nan"), float("inf"), "x", True, False]


@pytest.mark.parametrize(("value", "expected"), [
    (140, 100), (150, 200), (282, 200), (283, 400),
    (1, 100), (10_000, 400), (100, 100), (400, 400),
])
def test_nearest_is_measured_in_log2(value, expected):
    assert nearest(STEPS, value) == expected


def test_an_exact_tie_goes_to_the_lower_step():
    assert nearest((100, 400), 200) == 100


@pytest.mark.parametrize("bad", BAD)
def test_unusable_values_count_as_auto(bad):
    assert is_auto(bad)


@pytest.mark.parametrize("good", [1, 0.5, 400, 4249.0])
def test_positive_finite_numbers_are_not_auto(good):
    assert not is_auto(good)


@pytest.mark.parametrize("bad", BAD)
def test_from_auto_without_a_usable_metered_value_both_directions_use_default(bad):
    assert up(STEPS, None, bad, 200) == 200
    assert down(STEPS, None, bad, 200) == 200


def test_from_auto_with_a_metered_value_both_directions_start_at_the_nearest():
    assert up(STEPS, None, 150, 100) == 200
    assert down(STEPS, None, 150, 100) == 200


def test_a_bad_current_counts_as_auto():
    assert up(STEPS, True, 150, 100) == 200
    assert down(STEPS, 0, None, 400) == 400


def test_up_and_down_from_a_step():
    assert up(STEPS, 100, None, 200) == 200
    assert up(STEPS, 200, None, 200) == 400
    assert down(STEPS, 400, None, 200) == 200
    assert down(STEPS, 200, None, 200) == 100


def test_from_between_steps_the_next_step_in_that_direction():
    assert up(STEPS, 250, None, 200) == 400
    assert down(STEPS, 250, None, 200) == 200


def test_above_the_top_up_stays_and_down_lands_on_the_top():
    assert up(STEPS, 900, None, 200) == 400
    assert down(STEPS, 900, None, 200) == 400


def test_below_the_bottom_up_lands_on_the_bottom_and_down_is_auto():
    assert up(STEPS, 50, None, 200) == 100
    assert down(STEPS, 50, None, 200) is None


def test_down_at_the_bottom_is_auto_and_up_at_the_top_stays():
    assert down(STEPS, 100, None, 200) is None
    assert up(STEPS, 400, None, 200) == 400
