#pragma once

#include <cstddef>
#include <cstdint>

#include "capture_client.h"

class StickDisplay {
 public:
  static constexpr size_t kMaxJpegBytes = 64 * 1024;

  // Width reserved in the bottom-right corner for the battery badge, so the
  // READY caption is centred in the remaining space instead of under it.
  static constexpr int16_t kBatteryBadgeWidth = 56;
  static constexpr int16_t kBatteryBadgeHeight = 14;
  // Bottom-left badge for the Pi's UPS battery ("Pi 87%+"); the READY caption
  // is centred in the space between the two badges.
  static constexpr int16_t kPiBadgeWidth = 56;

  void begin();
  void render(const CaptureClient& client, uint32_t now_ms);
  // Panel power, toggled by the side button to save battery. While off the
  // panel sleeps at zero brightness and nothing is drawn; captures continue,
  // and turning it back on repaints the current screen in full.
  void setPower(bool on);
  bool powered() const { return powered_; }
  // Candidate bytes are decoded before they replace the persistent photo.
  bool decodeAndStore(const uint8_t* jpeg, size_t jpeg_size);
  bool hasPhoto() const { return photo_size_ > 0; }
  // Battery text for the corner badge ("87%", "87%+", "--%"). Redraws the badge
  // at once; render() also repaints it after every full-screen redraw.
  void setBatteryLabel(const char* label, bool low);
  // Pi battery text for the bottom-left badge; "Pi --%" when the Pi has no UPS.
  void setPiBatteryLabel(const char* label, bool low);

 private:
  void drawPhoto();
  void drawColourBars();
  void drawBarsOverlay(const char* title, const char* detail, uint32_t elapsed_seconds);
  void drawMessage(const char* title, const char* detail, uint32_t elapsed_seconds);
  // The seconds counter alone, repainted in place. A tick must not redraw the
  // whole screen: repainting the colour bars once a second is a visible blink.
  void drawOverlayElapsed(uint32_t elapsed_seconds);
  void drawMessageElapsed(uint32_t elapsed_seconds);
  void drawElapsedOnly(ClientState state, uint32_t elapsed_seconds);
  void drawBatteryBadge();

  uint8_t photo_[kMaxJpegBytes] = {};
  size_t photo_size_ = 0;
  char battery_label_[12] = "";
  bool battery_low_ = false;
  char pi_battery_label_[12] = "";
  bool pi_battery_low_ = false;
  bool powered_ = true;
  // Brightness as configured at boot, restored when the panel comes back.
  uint8_t brightness_ = 0;
  bool force_redraw_ = false;
  ClientState last_state_ = ClientState::Connecting;
  uint32_t last_elapsed_seconds_ = UINT32_MAX;
  bool last_ready_ = false;
  int16_t width_ = 0;
  int16_t height_ = 0;
};
