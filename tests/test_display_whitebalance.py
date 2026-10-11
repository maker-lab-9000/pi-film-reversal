import pytest

from pifilm.display.whitebalance import (
    DEFAULT_START_K,
    WB_STEPS_K,
    cooler,
    label,
    next_step,
    value_label,
    warmer,
)


def test_warmer_walks_up_from_the_metered_step_and_stays_on_the_top():
    seen, value = [], None
    for _ in range(7):
        value = warmer(value, 4249)
        seen.append(value)
    assert seen == [4000, 5000, 5600, 6500, 7500, 7500, 7500]


def test_cooler_walks_down_and_ends_in_auto():
    seen, value = [], 7500
    for _ in range(7):
        value = cooler(value, None)
        seen.append(value)
    assert seen == [6500, 5600, 5000, 4000, 3200, None, 5600]


def test_the_default_start_is_5600_without_a_metered_value():
    assert DEFAULT_START_K == 5600
    assert warmer(None, None) == 5600
    assert cooler(None, float("nan")) == 5600


def test_value_label_rounds_to_100k():
    assert value_label(5600) == "5600K"
    assert value_label(4249) == "4200K"
    assert value_label(6500) == "6500K"


def test_the_steps_cycle_from_auto_through_every_value_and_back():
    seen, value = [], None
    for _ in range(len(WB_STEPS_K) + 1):
        value = next_step(value)
        seen.append(value)
    assert seen == [3200, 4000, 5000, 5600, 6500, 7500, None]


@pytest.mark.parametrize(("current", "expected"), [
    (2500, 3200), (3300, 4000), (6500, 7500), (7400, 7500), (7500, None), (9000, None),
])
def test_a_value_between_steps_moves_to_the_next_one_up(current, expected):
    assert next_step(current) == expected


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
def test_a_meaningless_value_counts_as_auto(bad):
    assert next_step(bad) == WB_STEPS_K[0]


def test_labels_are_ascii_and_short():
    assert label(None) == "WB A"
    assert label(5600) == "WB 5600K"
    assert all(label(k).isascii() and len(label(k)) <= 8 for k in WB_STEPS_K)
