#include "display.h"

#include <M5Unified.h>
#include <JPEGDEC.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {
constexpr uint16_t kBars[] = {TFT_WHITE, TFT_YELLOW, TFT_CYAN, TFT_GREEN,
                              TFT_MAGENTA, TFT_RED, TFT_BLUE, TFT_BLACK};

int validateJpegDraw(JPEGDRAW*) {
  // Validation decodes every MCU but intentionally discards its pixels. M5GFX
  // does the visible decode only after this complete pass has succeeded.
  return 1;
}

bool decodesSuccessfully(const uint8_t* jpeg, size_t size) {
  if (jpeg == nullptr || size < 4 || size > StickDisplay::kMaxJpegBytes) return false;
  // JPEGDEC contains ~18 KiB of workspace, exceeding loopTask's 8 KiB stack.
  // Only the UI loop calls this function, so reuse one static decoder safely.
  static JPEGDEC decoder;
  if (!decoder.openRAM(const_cast<uint8_t*>(jpeg), static_cast<int>(size), validateJpegDraw)) return false;
  const bool decoded = decoder.decode(0, 0, 0) == 1;
  decoder.close();
  return decoded;
}
}

void StickDisplay::begin() {
  M5.Display.setRotation(1);
  width_ = M5.Display.width();
  height_ = M5.Display.height();
  // Read the working brightness once, so setPower(true) restores exactly what
  // the panel ran at. A panel that reports 0 here would never come back, so
  // fall back to M5Unified's own default.
  brightness_ = M5.Display.getBrightness();
  if (brightness_ == 0) brightness_ = 128;
  M5.Display.setTextDatum(middle_center);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(1);
  drawColourBars();
}

void StickDisplay::setPower(bool on) {
  if (on == powered_) return;
  powered_ = on;
  if (!on) {
    // Brightness first: sleep() alone leaves the backlight burning, which is
    // most of what this saves.
    M5.Display.setBrightness(0);
    M5.Display.sleep();
    Serial.println("[display] panel off");
    return;
  }
  M5.Display.wakeup();
  M5.Display.setBrightness(brightness_);
  // Whatever is in the panel's memory is stale by however long it was dark, and
  // render() draws only on a change. Ask for one full repaint of the current
  // screen, badge included.
  force_redraw_ = true;
  last_state_ = ClientState::Connecting;
  last_elapsed_seconds_ = UINT32_MAX;
  last_ready_ = false;
  Serial.println("[display] panel on");
}

void StickDisplay::drawColourBars() {
  constexpr size_t bar_count = sizeof(kBars) / sizeof(kBars[0]);
  const int16_t bar_width = std::max<int16_t>(1, width_ / static_cast<int16_t>(bar_count));
  for (size_t index = 0; index < bar_count; ++index) {
    const int16_t x = static_cast<int16_t>(index * bar_width);
    const int16_t w = index + 1 == bar_count ? width_ - x : bar_width;
    M5.Display.fillRect(x, 0, w, height_, kBars[index]);
  }
  M5.Display.fillRect(0, height_ - 24, width_, 24, TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(1);
  // Centred between the Pi badge (left) and the Stick badge (right).
  const int16_t caption_x = kPiBadgeWidth + (width_ - kPiBadgeWidth - kBatteryBadgeWidth) / 2;
  // 20 characters at 6 px = 120 px, inside the 128 px between the two 56 px badges.
  M5.Display.drawString("READY \xE2\x80\xA2 press button", caption_x, height_ - 12);
}

void StickDisplay::setBatteryLabel(const char* label, bool low) {
  std::strncpy(battery_label_, label == nullptr ? "" : label, sizeof(battery_label_) - 1);
  battery_label_[sizeof(battery_label_) - 1] = '\0';
  battery_low_ = low;
  drawBatteryBadge();
}

void StickDisplay::setPiBatteryLabel(const char* label, bool low) {
  std::strncpy(pi_battery_label_, label == nullptr ? "" : label, sizeof(pi_battery_label_) - 1);
  pi_battery_label_[sizeof(pi_battery_label_) - 1] = '\0';
  pi_battery_low_ = low;
  drawBatteryBadge();
}

void StickDisplay::drawBatteryBadge() {
  // The labels above are still stored while the panel is off, so the badge is
  // correct the moment the screen comes back.
  if (!powered_) return;
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(middle_center);
  const int16_t y = height_ - kBatteryBadgeHeight;
  if (battery_label_[0] != '\0') {
    const int16_t x = width_ - kBatteryBadgeWidth;
    M5.Display.fillRect(x, y, kBatteryBadgeWidth, kBatteryBadgeHeight, TFT_BLACK);
    // Red below the low threshold while discharging; white otherwise. The trailing
    // "+" in the label marks charging.
    M5.Display.setTextColor(battery_low_ ? TFT_RED : TFT_WHITE, TFT_BLACK);
    M5.Display.drawString(battery_label_, x + kBatteryBadgeWidth / 2, y + kBatteryBadgeHeight / 2);
  }
  if (pi_battery_label_[0] != '\0') {
    M5.Display.fillRect(0, y, kPiBadgeWidth, kBatteryBadgeHeight, TFT_BLACK);
    M5.Display.setTextColor(pi_battery_low_ ? TFT_RED : TFT_WHITE, TFT_BLACK);
    M5.Display.drawString(pi_battery_label_, kPiBadgeWidth / 2, y + kBatteryBadgeHeight / 2);
  }
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
}

void StickDisplay::drawMessageElapsed(uint32_t elapsed_seconds) {
  // Clear only the counter's own box (up to "120s" at size 1) before redrawing,
  // so "9s" fully replaces "10s" without touching the title or detail lines.
  M5.Display.fillRect(width_ / 2 - 18, height_ / 2 + 19, 36, 14, TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(middle_center);
  char elapsed[24];
  snprintf(elapsed, sizeof(elapsed), "%lus", static_cast<unsigned long>(elapsed_seconds));
  M5.Display.drawString(elapsed, width_ / 2, height_ / 2 + 26);
}

void StickDisplay::drawMessage(const char* title, const char* detail, uint32_t elapsed_seconds) {
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(2);
  M5.Display.drawString(title, width_ / 2, height_ / 2 - 22);
  M5.Display.setTextSize(1);
  M5.Display.drawString(detail, width_ / 2, height_ / 2 + 4);
  drawMessageElapsed(elapsed_seconds);
}

void StickDisplay::drawOverlayElapsed(uint32_t elapsed_seconds) {
  // The counter sits at the right end of the black top strip; clear its box only.
  M5.Display.fillRect(width_ - 32, 0, 32, 14, TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setTextDatum(middle_center);
  char elapsed[24];
  snprintf(elapsed, sizeof(elapsed), "%lus", static_cast<unsigned long>(elapsed_seconds));
  M5.Display.drawString(elapsed, width_ - 14, 7);
}

void StickDisplay::drawBarsOverlay(const char* title, const char* detail, uint32_t elapsed_seconds) {
  drawColourBars();
  M5.Display.fillRect(0, 0, width_, 32, TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(1);
  M5.Display.drawString(title, width_ / 2, 7);
  M5.Display.drawString(detail, width_ / 2, 18);
  drawOverlayElapsed(elapsed_seconds);
}

void StickDisplay::drawElapsedOnly(ClientState state, uint32_t elapsed_seconds) {
  switch (state) {
    case ClientState::Requesting:
    case ClientState::Processing:
    case ClientState::Downloading:
      drawOverlayElapsed(elapsed_seconds);
      break;
    case ClientState::Connecting:
      drawMessageElapsed(elapsed_seconds);
      break;
    case ClientState::Error:
      // The error screen shows a counter only when there is no photo behind it.
      if (!hasPhoto()) drawMessageElapsed(elapsed_seconds);
      break;
    case ClientState::Ready:
    case ClientState::Photo:
      break;
  }
}

void StickDisplay::drawPhoto() {
  if (photo_size_ == 0) {
    Serial.println("[display] drawPhoto: no stored photo");
    return;
  }
  const int16_t target_width = std::min<int16_t>(width_, (height_ * 240) / 135);
  const int16_t target_height = std::min<int16_t>(height_, (width_ * 135) / 240);
  const int16_t x = (width_ - target_width) / 2;
  const int16_t y = (height_ - target_height) / 2;
  M5.Display.fillScreen(TFT_BLACK);
  const bool drawn = M5.Display.drawJpg(photo_, photo_size_, x, y, target_width, target_height);
  Serial.printf("[display] drawPhoto: M5GFX drawJpg %u bytes at %d,%d %dx%d on %dx%d screen -> %s\n",
                static_cast<unsigned>(photo_size_), x, y, target_width, target_height, width_, height_,
                drawn ? "ok" : "FAILED");
}

bool StickDisplay::decodeAndStore(const uint8_t* jpeg, size_t jpeg_size) {
  if (!decodesSuccessfully(jpeg, jpeg_size)) {
    Serial.printf("[display] JPEGDEC validation FAILED for %u bytes; photo not replaced\n",
                  static_cast<unsigned>(jpeg_size));
    return false;
  }
  // M5GFX exposes a draw-based decoder. JPEGDEC completed the candidate
  // decode above, so only a verified image can replace the persistent buffer.
  const int16_t target_width = std::min<int16_t>(width_, (height_ * 240) / 135);
  const int16_t target_height = std::min<int16_t>(height_, (width_ * 135) / 240);
  const int16_t x = (width_ - target_width) / 2;
  const int16_t y = (height_ - target_height) / 2;
  const bool drawn = M5.Display.drawJpg(jpeg, jpeg_size, x, y, target_width, target_height);
  Serial.printf("[display] JPEGDEC validation ok; M5GFX drawJpg %u bytes at %d,%d %dx%d -> %s\n",
                static_cast<unsigned>(jpeg_size), x, y, target_width, target_height, drawn ? "ok" : "FAILED");
  std::memcpy(photo_, jpeg, jpeg_size);
  photo_size_ = jpeg_size;
  return true;
}

void StickDisplay::render(const CaptureClient& client, uint32_t now_ms) {
  // Nothing is drawn while the panel sleeps. decodeAndStore() is called
  // independently of this, so a capture taken in the dark is still stored and
  // appears when the screen is turned back on.
  if (!powered_) return;
  const ClientState state = client.state();
  const uint32_t elapsed = client.elapsedSeconds(now_ms);
  const bool ready = client.readyForCapture();
  // Only the photo screen draws anything that depends on readiness. During a
  // capture the Pi reports itself busy, which flips `ready`; that must not
  // trigger a full repaint of the colour bars.
  const bool ready_matters = state == ClientState::Photo;
  const bool screen_changed = force_redraw_ || state != last_state_ || (ready_matters && ready != last_ready_);
  const bool counter_changed = elapsed != last_elapsed_seconds_;
  if (!screen_changed && !counter_changed) return;
  if (state != last_state_) {
    Serial.printf("[display] render %s (pi ready=%d, stored photo=%u bytes)\n", clientStateName(state), ready,
                  static_cast<unsigned>(photo_size_));
  }
  force_redraw_ = false;
  last_state_ = state;
  last_elapsed_seconds_ = elapsed;
  last_ready_ = ready;
  if (!screen_changed) {
    // Once a second while waiting: repaint the seconds box, nothing else.
    drawElapsedOnly(state, elapsed);
    return;
  }
  switch (state) {
    case ClientState::Ready:
      drawColourBars();
      break;
    case ClientState::Connecting:
      drawMessage("CONNECTING", "Wi-Fi reconnecting", elapsed);
      break;
    case ClientState::Requesting:
      drawBarsOverlay("REQUESTING", "Sending capture request", elapsed);
      break;
    case ClientState::Processing:
      drawBarsOverlay("PROCESSING", "Waiting for film scan", elapsed);
      break;
    case ClientState::Downloading:
      drawBarsOverlay("DOWNLOADING", "Fetching 240 x 135 preview", elapsed);
      break;
    case ClientState::Photo:
      drawPhoto();
      if (!ready) {
        M5.Display.fillRect(0, 0, width_, 20, TFT_BLACK);
        M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
        M5.Display.setTextSize(1);
        M5.Display.drawString("Pi unavailable or busy", width_ / 2, 10);
      }
      break;
    case ClientState::Error:
      if (!hasPhoto()) {
        drawMessage(client.timedOut() ? "TIMEOUT" : "ERROR", client.errorDetail(), elapsed);
        break;
      }
      drawPhoto();
      M5.Display.fillRect(0, 0, width_, 20, TFT_BLACK);
      M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
      M5.Display.setTextSize(1);
      M5.Display.drawString(client.errorDetail(), width_ / 2, 10);
      break;
  }
  // Every branch above repainted the full screen, so the badge goes back on top.
  drawBatteryBadge();
}
