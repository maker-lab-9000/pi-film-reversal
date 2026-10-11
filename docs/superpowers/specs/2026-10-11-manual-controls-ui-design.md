# Manual controls and the SLR-finder viewfinder: design

Date: 2026-10-11. Approved by the user in conversation ("Let's go with C2 ... proceed").
Mock-up: `docs/superpowers/mockups/2026-10-11-manual-controls-ui.png` (left: all auto;
right: full manual with `+` pressed).

## Goal

The viewfinder becomes a camera with three manual controls (shutter, ISO, white
balance), each of which can also be automatic, and loses exposure compensation. The
live screen is redrawn in an "SLR finder" style: full-screen image, a solid black
information strip with amber read-outs, thin amber rings for the keys.

## What the user asked for

1. Remove the EV `-`/`+` buttons and the EV function; keep the exposure scale as a meter.
2. Add ISO: automatic or a fixed value.
3. With shutter, ISO and WB all fixed the camera is in full manual.
4. Restyle the live screen (option "C2" of the mock-ups), with the shutter button at the
   top right and `+`/`-` below it, so a slip while adjusting does not take a photo.
5. Amber outlines on the keys and the battery read-out.
6. `+`/`-` react visually when pressed, on touch-down (not only on release).

## Behaviour

### Controls

- Three cells in the bottom strip: SHUTTER, ISO, WB. A tap selects a cell. SHUTTER is
  selected at start-up. The selection is kept until another cell is tapped.
- One `+`/`-` pair changes the selected control by one step.
  - SHUTTER: `+` is faster, `-` slower; slower than 1 s returns to auto (unchanged scale,
    `pifilm/display/shutter.py`).
  - ISO: 100, 200, 400, 800, 1600. `+` is higher; `-` below 100 returns to auto; `+` at
    1600 stays at 1600.
  - WB: 3200, 4000, 5000, 5600, 6500, 7500 K. `+` is warmer (a higher number); `-`
    below 3200 returns to auto; `+` at 7500 stays.
- The first step out of auto, in either direction, lands on the step nearest (in log2)
  to what the camera is using now (metered exposure time, metered ISO, metered colour
  temperature), so the picture does not jump. Without a metered value: 1/125, ISO 400,
  5600 K.
- An automatic control shows `AUTO` beside its name and the live metered value; a
  fixed control shows only its name and the fixed value.
- A control the camera cannot change (no setter; white balance fixed by `--awb-lock`
  or `--colour-gains`; no settable `ColourTemperature`) is drawn grey, cannot be
  selected, and `+`/`-` do nothing while no selectable cell is selected.
- All three reset on restart (WB to `--white-balance` if given).

### Press feedback

- While a finger is down on `+`, `-` or the shutter button, that key is drawn filled
  amber. A tap shorter than one frame still lights its key for 0.15 s.
- Actions still fire on release, as now.

### Meter

- The exposure scale stays: 13 dots from -3 to +3 stops in half stops, the dot nearest
  the measured deviation lit amber. Lux and clip stay as small text above it.
- In auto ISO, when the gain is at its limit (`iso_max`), the ISO cell reads `MAX`.

### EV removal

- No EV buttons, read-out, or state in the viewfinder. The `--ev` flag, the cameras'
  `set_ev`, and `Picamera2Camera`'s `ev` parameter are removed.
- `camera.ev` stays as a constant `0.0` attribute and the `ExposureValue: 0.0` control is
  still sent, so `ev_comp` stays in every capture record (now always 0) and old records
  regrade exactly as before. `Pipeline.process(ev=...)` and `pifilm-process` do not change.
- The review caption becomes `shutter  ISO NNN`.

### Records

- `captures.jsonl` gains `iso` (the fixed ISO, null on auto), beside `shutter_us` and
  `colour_temperature_k`.

## Camera facts measured on the Pi 4 + IMX477 (libcamera 0.7.2, Picamera2 0.3.37)

2026-10-11, preview configuration, dim room:

| Set | ExposureTime | AnalogueGain | DigitalGain |
| --- | --- | --- | --- |
| auto | 66656 | 16.0 | 1.0 |
| AnalogueGain 1.0 | 66656 | 1.0 | 1.17 |
| AnalogueGain 4.0 | 65443 | 4.0 | 1.0 |
| AnalogueGain 16.0 | 65434 | 16.0 | 1.0 |
| AnalogueGain 4.0 + ExposureTime 8000 | 7996 | 4.0 | 1.0 |
| AnalogueGain 0 + ExposureTime 0 | 66656 | 15.75 | 1.0 |

- `camera_controls["AnalogueGain"]` is `(1.0, 22.26, 1.0)`; `AnalogueGainMode` exists `(0, 1, 0)`.
- A fixed gain holds while the exposure time stays automatic; both can be fixed; zero
  returns both to auto. The room was too dark to show the exposure time compensating
  (it was already at the frame limit), so that is for hardware acceptance.
- ISO 100 is gain 1.0, the convention the meter already uses.

## Appearance (base units, 320 x 240; drawn at the panel's integer scale)

- Font: Barlow Condensed (SIL OFL 1.1), bundled in `pifilm/data/fonts/`: SemiBold for
  values, Medium for labels. The other screens keep their current font.
- Colours: amber (255, 176, 0); dim amber (150, 106, 8); white (245, 245, 240);
  grey (125, 125, 120); strip black.
- Exact geometry is in the plan (Task 3). Shapes are anti-aliased.

## Out of scope

Review, processing and message screens; idle dimming; the Stick; the GPIO button;
an `--iso` start-up flag; hold-to-repeat on `+`/`-`.

## Hardware acceptance

1. Start-up: full-screen image, strip with three cells all `AUTO`, SHUTTER selected.
2. Select each cell; `+`/`-` change only the selected one; `-` below the bottom returns
   it to `AUTO`; the first step from auto does not jump the picture.
3. ISO fixed, shutter auto, in good light: the shutter read-out changes to compensate as
   ISO is stepped, and the meter stays near the centre.
4. Full manual: fix shutter and ISO; the picture gets darker and lighter with the
   steps and the meter dot follows.
5. `+`/`-` light while the finger is down, and on a quick tap.
6. A shot at fixed shutter + ISO + WB: the record has `shutter_us`, `iso`,
   `colour_temperature_k`, and `camera_metadata` agrees.
7. The shutter button is not hit while working `+`/`-`.
8. `2x`, focus gauge, battery, review, idle screen-off, Stick and GPIO button work as before.
9. About 10 fps is kept (`viewfinder: N fps` in the journal).
