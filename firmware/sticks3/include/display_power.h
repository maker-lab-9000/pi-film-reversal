#pragma once

#include <cstdint>

// Screen on/off toggle for the side button (BtnB), kept free of Arduino, M5,
// and display dependencies so the `native` Unity environment can cover it.
// The panel is the Stick's largest current draw, so a shoot can leave it dark
// and still fire the shutter; nothing here touches the capture state machine.
class DisplayPower {
 public:
  // Same debounce window as the shutter path: a contact bounce shorter than
  // this is never a click.
  static constexpr uint32_t kDebounceMs = 30;

  // Samples the side button. A complete click - a debounced press followed by
  // a release - toggles the panel. Toggling on the release rather than the
  // press means holding the button down changes nothing until it is let go,
  // and it cannot repeat while held.
  void onButtonSample(bool pressed, uint32_t now_ms);

  bool on() const { return on_; }

  // True once after each toggle, so the caller applies the change to the
  // panel exactly once instead of on every loop iteration.
  bool consumeChanged();

 private:
  bool on_ = true;
  bool changed_ = false;
  bool raw_button_ = false;
  bool stable_button_ = false;
  uint32_t raw_changed_at_ = 0;
};
