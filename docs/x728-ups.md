# Geekworm X728 v2.5 UPS: integration with the Pi

What the shield needs from the Pi, what the Pi needs from the OS, which pins
and addresses are involved, how to install Geekworm's power-button service, how
to set a shutdown policy that suits a battery camera, and how to verify each
piece. Sources: the [X728 hardware page](https://wiki.geekworm.com/X728) and
the [X728 script page](https://wiki.geekworm.com/X728-script), plus the scripts
in the `Trixie` branch of
[geekworm-com/x728-script](https://github.com/geekworm-com/x728-script), read
on 2026-09-10. Facts marked *verified* were checked on this Pi the same day.

## 1. What the X728 gives this project

- **Unattended power.** 2 × 18650 cells, 5.1 V at up to 5 A out, pass-through
  charging at 2.3 to 3.2 A. The Pi 3B, the camera and the hotspot draw about
  4 W; two 3,400 mAh cells give roughly 4 to 5 hours.
- **Safe shutdown.** A power button wired to the Pi through GPIO: a short hold
  reboots, a longer hold runs `poweroff` cleanly and the shield cuts power only
  after the OS has halted. Pulling the plug on a running Pi is the most likely
  way this project corrupts its SD card; the button removes that.
- **A real-time clock.** A DS1307 on I2C, backed by the cells. The Pi 3B has no
  RTC, so today every offline boot starts with the wrong date and captures are
  filed under it. This closes that item from `todo.md`.
- **A fuel gauge.** A MAX17040-class gauge on I2C reports cell voltage and state
  of charge. The plan in `docs/superpowers/plans/2026-09-10-ups-battery-on-stick.md`
  puts that number on the Stick next to the Stick's own battery.

## 2. State of this Pi before integration (verified 2026-09-10)

| Item | Found | Consequence |
|---|---|---|
| OS, kernel | Raspberry Pi OS Trixie, kernel 6.18, Pi 3 Model B Rev 1.2 | Use Geekworm's `Trixie` branch; libgpiod 2.x command syntax |
| Header I2C bus | **Disabled**: `raspi-config nonint get_i2c` = 1, `#dtparam=i2c_arm=on` still commented, only `/dev/i2c-2` (the HDMI bus) exists | Must be enabled before anything on the shield is visible |
| `gpiod`, `python3-libgpiod` | Installed, 2.2.1 | Geekworm's `xPWR.sh` and `xSoft.sh` run as shipped |
| `python3-gpiozero`, `python3-lgpio`, `python3-rpi-lgpio` | Installed; gpiozero 2.0.1 with `LGPIOFactory` | Python GPIO works without RPi.GPIO; `import RPi.GPIO` resolves to the lgpio shim |
| `python3-smbus2`, `python3-smbus`, `i2c-tools` | Installed | Fuel gauge readable from Python once I2C is on; `i2cdetect` is in `/usr/sbin` |
| RTC | None registered (`/sys/class/rtc/` empty, `RTC time: n/a`); `fake-hwclock` not present | DS1307 overlay needed |
| Power | `vcgencmd get_throttled` = `0x0` | No under-voltage while running from the shield |
| User `george` groups | includes `gpio`, `i2c` | The capture service can read the gauge and the PLD pin without root |
| `hwclock` | Not installed; Trixie ships it in `util-linux-extra` | Use `timedatectl` to read the RTC, or install the package |

Later the same day the first two rows were resolved on this Pi: `dtparam=i2c_arm=on`
and `dtoverlay=i2c-rtc,ds1307` are in `config.txt`, `/dev/rtc0` exists, and
`timedatectl` shows the RTC set.

## 3. Hardware facts and pin map

Cells: two 3.7 V 18650, correct polarity is your responsibility, and Geekworm
says **do not use cells with built-in protection circuits**: the protection
PCB trips under the shield's charge and discharge currents. Use good-quality
unprotected flat-top cells. Charge termination is 4.24 V, recharge starts
below 4.1 V.

Power in: the DC 5.5 × 2.1 jack at 5 V and at least 4 A, or the USB-C socket at
5 V and at least 3 A. **Never power the Pi through its own micro-USB socket
while the shield is fitted.** Fit the copper standoffs before mounting; the
board can otherwise touch the Pi's USB shells and short.

| BCM pin | Direction | Meaning | Used by |
|---|---|---|---|
| 5 | Pi input | Shutdown request from the shield. High pulse of 200 to 600 ms after a 1 to 2 s button hold means reboot; longer means power off | `xPWR.sh` |
| 12 | Pi output | "OS is up". Driven high at boot; the shield cuts power when it falls after halt | `xPWR.sh` |
| 26 | Pi output | Software shutdown request to the shield: pulse high for about 2 s, then low. The shield answers by raising pin 5, so the normal `poweroff` path runs | `xSoft.sh`, the low-battery script |
| 6 | Pi input | Power-loss detection. 0 = external power present, 1 = running on the cells | our capture service, optional alarm scripts |
| 20 | Pi output | Buzzer (v2.1 and later) | optional |
| 16 | Pi output | Battery charge enable (v2.5 and later, "advanced users only") | leave alone |

The optional shutter button uses BCM 21 (physical pin 40), which is not in this
table; `pifilm-capture` refuses `--shutter-gpio` on any pin that is, when
`--ups x728` is given. See [setup](setup.md), "Optional: shutter button".

GPIO chip is `gpiochip0` on the Pi 3. Any future GPIO use in this project, such
as a flash LED, must avoid pins 5, 6, 12, 16, 20 and 26.

| I2C address | Chip | Registers |
|---|---|---|
| `0x36` | MAX17040/MAX17041 fuel gauge | `0x02` VCELL: 16-bit big-endian, voltage in volts = value × 1.25 / 1000 / 16. `0x04` SOC: 16-bit big-endian, percent = value / 256, clamp to 100 |
| `0x68` | DS1307 real-time clock | handled by the kernel `i2c-rtc` overlay |

Button, from the hardware page: short press powers on; hold 1 to 2 s reboots;
hold 3 to 7 s safe shutdown; hold more than 8 s forces power off. An "auto
power on" jumper makes the Pi boot as soon as input power is applied.

## 4. Dependencies on the Pi

Nothing needs installing beyond what is already present. Three things need
configuring:

1. **Enable the header I2C bus.** Creates `/dev/i2c-1`, where the gauge and
   the RTC live.

   ```sh
   sudo raspi-config nonint do_i2c 0
   ```

2. **Enable the RTC overlay.** Add under the `[all]` section, or at the end, of
   `/boot/firmware/config.txt`:

   ```text
   dtoverlay=i2c-rtc,ds1307
   ```

3. **Reboot**, then confirm both chips answer and the clock exists:

   ```sh
   sudo /usr/sbin/i2cdetect -y 1        # expect 36 and UU (68 claimed by the rtc driver)
   ls /dev/rtc0 /sys/class/rtc/         # rtc0
   timedatectl                          # "RTC time:" shows a date, not n/a
   ```

   `timedatectl` is enough to read the RTC. The classic `hwclock` tool is not
   installed on Trixie; Debian 13 moved it to `util-linux-extra`, so
   `sudo apt install util-linux-extra` if you want `hwclock -r` and
   `hwclock -w`. You rarely need them: systemd-timesyncd writes the system
   time into the RTC whenever it has synchronised over the network, so one
   wired session with `System clock synchronized: yes` sets the chip. From
   then on the DS1307 seeds the system clock at every offline boot. It is
   powered from the 18650s, so removing both cells loses the time.

   Verified 2026-09-10 on this Pi: after enabling I2C and the overlay,
   `/dev/rtc0` appeared and `timedatectl` reported `RTC time: 2026-09-10
   12:40:18` in UTC, already set by timesyncd.

## 5. Install Geekworm's power-button service

This is the piece that turns the button into a clean `reboot` or `poweroff`
and lets the shield cut power afterwards. Run on the Pi:

```sh
cd ~/repos && git clone -b Trixie https://github.com/geekworm-com/x728-script
cd x728-script && chmod +x *.sh
sudo cp -f xPWR.sh xSoft.sh /usr/local/bin/
sudo cp -f x728-pwr.service /lib/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now x728-pwr
systemctl status x728-pwr --no-pager
```

The unit runs `/usr/local/bin/xPWR.sh 0 5 12` as root with `Restart=always`:
chip 0, shutdown pin 5, boot pin 12. It drives pin 12 high, then polls pin 5
every 200 ms; a pulse longer than 600 ms runs `poweroff`, a pulse between 200
and 600 ms runs `reboot`.

Software power-off from a shell, which pulses pin 26 and lets the same path run:

```sh
sudo /usr/local/bin/xSoft.sh 0 26
```

Geekworm suggests an alias `x728off` for it in `~/.bashrc`. The repository's
`uninstall.sh` reverses the install.

## 6. Shutdown policy for a battery camera

Geekworm ships two sample behaviours. Only one of them belongs on this device.

- **Power-loss shutdown** (`sample/x728-v2.x-plsd*.py`, `sample/test.sh`):
  when pin 6 goes high, wait 5 s, pulse pin 26, shut down. **Do not install
  this.** For a handheld camera, running on the cells is the normal state,
  not a fault. Pin 6 is still useful: the capture service reads it to show
  "external power present" on the Stick.
- **Low-battery shutdown** (`sample/x728-v2.x-asd.py`): read the gauge every
  2 s, and below 3.00 V wait 10 s and pulse pin 26. **Install this**, as a
  service, so a flat battery ends in a clean halt rather than a brownout
  mid-write. The sample uses `RPi.GPIO`, which on this Pi resolves to the
  `rpi-lgpio` shim and works; `smbus.SMBus(1)` needs the I2C bus from
  section 4.

```sh
sudo cp ~/repos/x728-script/sample/x728-v2.x-asd.py /usr/local/bin/x728-asd.py
sudo tee /lib/systemd/system/x728-asd.service >/dev/null <<'EOF'
[Unit]
Description=X728 low-battery safe shutdown
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/bin/x728-asd.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now x728-asd
```

3.00 V per cell is deep for 18650s; 3.2 V leaves margin and costs little
runtime. Edit the threshold in the copied script if you prefer. The script
also prints every 2 s; add `StandardOutput=null` to the unit if the journal
fills. When the capture service later reads the gauge itself (see the plan),
this script can be retired in favour of one owner of the shutdown decision.

## 7. How the shield interacts with `pifilm-capture`

- **Shutdown ordering** is handled by systemd: `poweroff` stops
  `pifilm-capture.service` before halting, and the shield cuts power only after
  pin 12 falls. A capture in flight at that moment loses at most its own files;
  `captures.jsonl` is appended one line per completed capture.
- **Two readers of one gauge** are fine: the low-battery script and the capture
  service both do single-word I2C reads through the kernel's bus lock.
- **Read the level anytime** with `pifilm-battery`, which does one gauge read and
  prints e.g. `Battery: 64%  3.85 V  on battery` — the same maths the Stick
  shows. `pifilm-battery --ups none` is a no-op for a Pi without the shield.
- **Pin 6 and the gauge are read by user `george`**, who is in `gpio` and
  `i2c`. Pins 5, 12 and 26 are root-only through Geekworm's service; the
  capture service never touches them.
- **Power budget.** The shield's 5 A output covers the Pi 3B, the USB camera
  and Wi-Fi with a wide margin. Charging draws up to 3.2 A on top, so the
  supply must be a 5 V adapter of 4 A or more on the DC jack, or 3 A on USB-C.
- **Auto power on** is a jumper choice. With it set, the camera boots whenever
  a charger is connected; without it, the button is the only power-on.

### Interpreting the reading

The MAX17040 is a **voltage-based** gauge ("ModelGauge"): it infers percent from
cell voltage against a single-18650 curve, not by counting charge in and out.
That has consequences worth knowing before trusting the number:

- **The percent and the shield's LED bar disagree by design.** The LEDs are a
  separate, coarse voltage/charge indicator; all of them lit means "high or
  charging", not a calibrated 100%. They are not reading the gauge.
- **The middle of the Li-ion curve is flat** (≈90 % to ≈30 % SOC spans only
  ≈4.0 V to ≈3.6 V), so voltage→percent is imprecise there and sags further
  under the Pi's load. Expect the percent to read conservatively.
- **The X728-C1 case (four cells, 1S4P) doubles capacity but not voltage.** The
  MAX17040 has no capacity/design register — it only reports where the pack
  voltage sits on a single-cell curve, so it cannot credit the extra cells, and
  its factory rate-compensation under-reads until a full charge/discharge cycle
  lets ModelGauge converge. Four cells also take roughly twice as long to charge,
  so a pack that looks "full" on the LEDs may simply not be full yet.
- **Trust the voltage.** Rested, ~4.1–4.2 V is genuinely full and ~3.6 V is
  nearly empty; `pifilm-battery` prints the voltage next to the percent for exactly
  this reason. Measured 2026-09-17 on the C1 four-cell pack: 3.99 V read as 58 %
  while the shield's LEDs showed full — a normal voltage-gauge disagreement, not
  a fault.

## 8. Verification checklist

Run these on the Pi after sections 4 and 5, with the cells fitted and the
charger connected.

```sh
sudo /usr/sbin/i2cdetect -y 1                       # 36 present, 68 shown as UU
python3 - <<'PY'
import struct, smbus2
bus = smbus2.SMBus(1)
def word(reg):
    raw = bus.read_word_data(0x36, reg)
    return struct.unpack("<H", struct.pack(">H", raw))[0]
print(f"cell {word(0x02) * 1.25 / 1000 / 16:.3f} V, charge {min(word(0x04) / 256, 100):.1f} %")
PY
gpioget --numeric -c 0 6                            # 0 with charger connected, 1 without
```

While `pifilm-capture --ups x728` is running it holds BCM 6, so `gpioget` reports the
line busy; stop the service first or read `external_power` from `/v1/status` instead.

```sh
timedatectl | grep 'RTC time'                       # a plausible date, not n/a
systemctl is-active x728-pwr x728-asd               # active, active
```

Then the physical tests, one at a time: unplug the charger and confirm `gpioget`
returns 1 and the Pi keeps running; hold the button 1 to 2 s and confirm a
clean reboot; hold it 3 to 7 s and confirm a clean power-off after which the
shield's LEDs go out; power on with a short press and confirm the hotspot and
`pifilm-capture` return unattended.

## 9. Rollback

```sh
sudo systemctl disable --now x728-asd x728-pwr
sudo rm -f /lib/systemd/system/x728-asd.service /usr/local/bin/x728-asd.py
cd ~/repos/x728-script && ./uninstall.sh
```

Leave I2C and the RTC overlay enabled; they are harmless without the shield.
