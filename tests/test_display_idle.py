import pytest

from pifilm.display.idle import IdleDimmer, Screen


def test_full_until_the_dim_delay_then_dim_then_off():
    idle = IdleDimmer(full=80, dim_after=60, off_after=300, now=0.0)
    assert idle.screen(59.9) is Screen.FULL and idle.backlight(59.9) == 80
    assert idle.screen(60.0) is Screen.DIM and idle.backlight(60.0) == 40
    assert idle.screen(299.9) is Screen.DIM
    assert idle.screen(300.0) is Screen.OFF and idle.backlight(300.0) == 0


def test_activity_restarts_both_timers():
    idle = IdleDimmer(full=80, dim_after=60, off_after=300, now=0.0)
    idle.wake(250.0)
    assert idle.screen(300.0) is Screen.FULL
    assert idle.screen(310.0) is Screen.DIM
    assert idle.screen(550.0) is Screen.OFF


def test_waking_from_off_restores_full_brightness():
    idle = IdleDimmer(full=80, dim_after=60, off_after=300, now=0.0)
    assert idle.screen(1000.0) is Screen.OFF
    idle.wake(1000.0)
    assert idle.screen(1000.0) is Screen.FULL and idle.backlight(1000.0) == 80


def test_dim_is_half_of_the_normal_level():
    assert IdleDimmer(full=100, dim_after=1, off_after=0, now=0.0).backlight(5.0) == 50


@pytest.mark.parametrize("dim_after, off_after, at, expected", [
    (0, 300, 200.0, Screen.FULL),   # dimming disabled: full until off
    (0, 300, 300.0, Screen.OFF),
    (60, 0, 10_000.0, Screen.DIM),  # turning off disabled: stays dim
    (0, 0, 10_000.0, Screen.FULL),  # both disabled: never changes
])
def test_zero_disables_a_step(dim_after, off_after, at, expected):
    assert IdleDimmer(full=80, dim_after=dim_after, off_after=off_after, now=0.0).screen(at) \
        is expected


def test_an_off_delay_shorter_than_the_dim_delay_skips_dimming():
    idle = IdleDimmer(full=80, dim_after=300, off_after=60, now=0.0)
    assert idle.screen(59.0) is Screen.FULL
    assert idle.screen(60.0) is Screen.OFF


def test_negative_delays_are_refused():
    with pytest.raises(ValueError):
        IdleDimmer(full=80, dim_after=-1, off_after=300, now=0.0)
