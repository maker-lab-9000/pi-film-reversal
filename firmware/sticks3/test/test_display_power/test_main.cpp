#ifdef PIO_UNIT_TESTING
#include <unity.h>
#else
#include <cassert>
#define TEST_ASSERT_TRUE(value) assert(value)
#define TEST_ASSERT_FALSE(value) assert(!(value))
#endif

#include "display_power.h"

namespace {

// One complete click: the press is held past the debounce window, then
// released and that release is also held past it.
void click(DisplayPower& power, uint32_t at_ms) {
  power.onButtonSample(true, at_ms);
  power.onButtonSample(true, at_ms + DisplayPower::kDebounceMs);
  power.onButtonSample(false, at_ms + 100);
  power.onButtonSample(false, at_ms + 100 + DisplayPower::kDebounceMs);
}

void test_display_starts_on_and_reports_no_pending_change(void) {
  DisplayPower power;

  TEST_ASSERT_TRUE(power.on());
  TEST_ASSERT_FALSE(power.consumeChanged());
}

void test_a_click_turns_the_display_off_and_a_second_turns_it_on(void) {
  DisplayPower power;

  click(power, 1000);
  TEST_ASSERT_FALSE(power.on());

  click(power, 5000);
  TEST_ASSERT_TRUE(power.on());
}

void test_holding_the_button_without_releasing_does_not_toggle(void) {
  DisplayPower power;

  for (uint32_t now = 0; now <= 5000; now += 10) {
    power.onButtonSample(true, now);
  }

  TEST_ASSERT_TRUE(power.on());
  TEST_ASSERT_FALSE(power.consumeChanged());

  // Letting go after the hold is still one click, not several.
  power.onButtonSample(false, 5010);
  power.onButtonSample(false, 5010 + DisplayPower::kDebounceMs);
  power.onButtonSample(false, 6000);
  TEST_ASSERT_FALSE(power.on());
  TEST_ASSERT_TRUE(power.consumeChanged());
  TEST_ASSERT_FALSE(power.consumeChanged());
}

void test_consume_changed_reports_once_per_toggle(void) {
  DisplayPower power;

  click(power, 0);
  TEST_ASSERT_TRUE(power.consumeChanged());
  TEST_ASSERT_FALSE(power.consumeChanged());
  TEST_ASSERT_FALSE(power.on());

  // Idle sampling between clicks reports nothing to apply.
  for (uint32_t now = 1000; now < 2000; now += 10) {
    power.onButtonSample(false, now);
  }
  TEST_ASSERT_FALSE(power.consumeChanged());

  click(power, 2000);
  TEST_ASSERT_TRUE(power.consumeChanged());
  TEST_ASSERT_FALSE(power.consumeChanged());
  TEST_ASSERT_TRUE(power.on());
}

void test_a_bounce_shorter_than_the_debounce_window_is_not_a_click(void) {
  DisplayPower power;

  power.onButtonSample(true, 0);
  power.onButtonSample(true, 10);
  power.onButtonSample(false, 20);
  power.onButtonSample(false, 200);

  TEST_ASSERT_TRUE(power.on());
  TEST_ASSERT_FALSE(power.consumeChanged());
}

#ifdef PIO_UNIT_TESTING
void run_all() {
  UNITY_BEGIN();
  RUN_TEST(test_display_starts_on_and_reports_no_pending_change);
  RUN_TEST(test_a_click_turns_the_display_off_and_a_second_turns_it_on);
  RUN_TEST(test_holding_the_button_without_releasing_does_not_toggle);
  RUN_TEST(test_consume_changed_reports_once_per_toggle);
  RUN_TEST(test_a_bounce_shorter_than_the_debounce_window_is_not_a_click);
  UNITY_END();
}
#endif

}  // namespace

#ifdef PIO_UNIT_TESTING
int main(int, char**) {
  run_all();
  return 0;
}
#else
int main() {
  test_display_starts_on_and_reports_no_pending_change();
  test_a_click_turns_the_display_off_and_a_second_turns_it_on();
  test_holding_the_button_without_releasing_does_not_toggle();
  test_consume_changed_reports_once_per_toggle();
  test_a_bounce_shorter_than_the_debounce_window_is_not_a_click();
  return 0;
}
#endif
