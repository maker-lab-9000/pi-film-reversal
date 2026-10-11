import pytest

from pifilm.display.whitebalance import WB_STEPS_K, label, next_step


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
