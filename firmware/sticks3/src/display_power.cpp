#include "display_power.h"

void DisplayPower::onButtonSample(bool pressed, uint32_t now_ms) {
  if (pressed != raw_button_) {
    raw_button_ = pressed;
    raw_changed_at_ = now_ms;
  }
  if (raw_button_ == stable_button_ || now_ms - raw_changed_at_ < kDebounceMs) {
    return;
  }
  stable_button_ = raw_button_;
  // Only the release edge of a press that survived the debounce toggles: a
  // press shorter than kDebounceMs never becomes stable, so its release is
  // not a click either.
  if (!stable_button_) {
    on_ = !on_;
    changed_ = true;
  }
}

bool DisplayPower::consumeChanged() {
  const bool changed = changed_;
  changed_ = false;
  return changed;
}
