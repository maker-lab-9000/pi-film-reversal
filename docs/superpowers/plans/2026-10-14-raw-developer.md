# DNG Developer (Raw Look, Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** `pifilm/raw.py` turns a Picamera2 DNG into an 8-bit sRGB picture using the camera's own tuning file, and a gallery shows the 141 training photographs and 13 held-out scenes developed that way beside the camera's JPEG, for the owner to judge as a base for a raw look.

**Architecture:** A small DNG reader (TIFF, one uncompressed strip, 12-bit packed Bayer) and a developer made of plain steps (black level, lens shading, white balance, demosaic, colour matrix, tone curve), each a pure NumPy/OpenCV function, with the corrections taken from the libcamera tuning file and the shot's metadata. No dependency is added. Nothing in capture, pipeline or training changes in this phase.

**Tech Stack:** Python 3.12, NumPy, OpenCV through `pifilm._cv2.require_cv2()`, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-14-raw-look-design.md` (approved by the owner 2026-10-14 with its recommendations: our own developer, CPU first, the same 141 photographs). Read it first; it is the authority.

## Facts measured on 2026-10-14 (use them; do not re-derive)

- A Picamera2 DNG (`~/Downloads/PiTrainingSource/2026-09-30/154700.dng`) is little-endian TIFF with one IFD: width 4056, height 3040, BitsPerSample 12, Compression 1, Photometric 32803 (CFA), one strip at StripOffsets with StripByteCounts 18,495,360 = 4056 x 3040 x 1.5, so 12-bit packed: two pixels in three bytes, TIFF bit order (pixel 0 = b0 << 4 | b1 >> 4, pixel 1 = (b1 & 0x0F) << 8 | b2). Verify that bit order against the reference below; it is the most likely thing to be wrong.
- CFARepeatDim (2, 2), CFAPattern bytes 02 01 01 00 (0 = red, 1 = green, 2 = blue): **BGGR**, matching the sensor mode `SBGGR12`. BlackLevel 256 (x4), WhiteLevel 4095. AsShotNeutral [0.3186, 1.0, 0.6277] (the reciprocal of the shot's `ColourGains` [3.139, 1.593]). ColorMatrix1 present (not used: the tuning file's colour matrix for the shot's colour temperature is what the ISP applies).
- Reference for the raw values: LibRaw 0.22 via `rawpy` in a scratch venv at `/private/tmp/claude-501/-Users-george-babanau-repos-raspberry-pi-martin-parr-project/870c50da-4e8c-4772-8272-e6e1c8eb42e9/scratchpad/rawvenv/bin/python` (`rawpy.imread(path).raw_image_visible`). Use it only to check the reader by hand and say so in the report; it is not a dependency and no test may import it.
- The tuning file is at `data/tuning/imx477.json` (gitignored; SHA-1 of its bytes `6fb48b343ddf2b928e58492adb0c37c472bc4210`, byte-identical to the Pi's `/usr/share/libcamera/ipa/rpi/vc4/imx477.json`). Version 2.0, target bcm2835. Relevant blocks:
  - `rpi.black_level`: `black_level` 4096 (16-bit units, = 256 at 12 bits).
  - `rpi.ccm`: `ccms`, a list of `{"ct": K, "ccm": [9 floats, row-major]}` at 2850, 2960, 3580, 4559, 5881, 7600 K. Applied to white-balanced camera RGB in linear light; output linear sRGB.
  - `rpi.alsc`: `calibrations_Cr` and `calibrations_Cb`, each a list of `{"ct": K, "table": [192 floats]}` at 2960, 4850, 5930 K; `luminance_lut` (192 floats); `luminance_strength` 0.5. Tables are a 16 x 12 grid (16 columns, 12 rows, row-major) over the full frame. Cr is the red gain, Cb the blue gain, the luminance table a gain on all channels.
  - `rpi.contrast`: `ce_enable` 1 (adaptive, not reproduced), `gamma_curve` 66 numbers = 33 (x, y) pairs in 0..65535: the ISP's tone curve from linear to output.
- Each capture record's `camera_metadata` has `ColourGains` [r, b] and `ColourTemperature` (K). The 141 training photographs and their DNGs are in `~/Downloads/PiTrainingSource/2026-09-30/` (record field `dng` names the file); the 13 held-out scenes are `data/test-scenes-imx477/<day>_<HHMMSS>_original.jpg`, their DNGs and records in `~/Downloads/PiCameraPhotos/<day>/`.

## Global Constraints

- Always `.venv/bin/pytest`, `.venv/bin/ruff`. Suite: `.venv/bin/pytest -q -m 'not slow'` (about 140 s; allow 400 s; macOS has no `timeout` command). Lint: `.venv/bin/ruff check .`. Both pass before every commit.
- No new dependency. NumPy and OpenCV (through `require_cv2()`; never `import cv2`) only; Pillow only through `pifilm.imageio` for writing pictures. `pifilm/raw.py` must import on a Mac and on the Pi.
- Pure functions with explicit inputs: the developer never reads the environment, a camera or a fixed path. The tuning dict, the DNG bytes and the shot's `ColourGains`/`ColourTemperature` are arguments.
- Deterministic: the same inputs give the same bytes on every run (no threads whose order changes results, no random numbers).
- Nothing in `pifilm/capture/`, `pifilm/pipeline.py`, `pifilm/train/`, `pifilm/gpu/` or any default changes in this plan.
- Never touch, stage or delete the untracked directory named `" "` (one space) at the repo root. Stage by explicit path only. Never read, print or source `.env`. Gallery output goes under `artifacts/` (gitignored); never commit photographs, DNGs or the tuning file.
- Match the surrounding code: module docstrings carry the reasons, comments are plain sentences saying why, no em dashes in prose.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Work on branch `plan/raw-look`. Do not push, open pull requests or switch branches. Do not edit `README.md`.

## Review Focus

- The 12-bit unpacking order and the Bayer pattern: a swapped nibble or a pattern off by one pixel gives a picture that looks almost right (wrong colours, a faint grid). The reader is checked against LibRaw on a real DNG, and the developer on a synthetic frame of known colour patches.
- OpenCV's Bayer conversion codes are named by a different corner than the TIFF CFA pattern; the BGGR mapping must be proven by a test, not assumed.
- Colour temperatures outside the calibrated range (below 2850 K or above 7600 K for the CCM, outside 2960 to 5930 K for lens shading) must clamp to the nearest table, not extrapolate.
- A clipped pixel (raw at the white level in any channel) must stay white or near white after white balance and the colour matrix, not turn magenta (the classic raw highlight cast): clip to 1.0 after white balance and before the matrix, and say so.
- Memory: a 12 MP frame in float32 is 146 MB for three channels; the developer must not hold more than a few such arrays at once (it will run on a 4 GB Pi later).

---

### Task 1: the DNG reader and the tuning tables

**Files:**
- Create: `pifilm/raw.py` (this task: reader and tuning part), `tests/test_raw.py`
- Test helper: a function in the test file (or `tests/raw_fakes.py`) that writes a minimal valid Picamera2-style DNG (one IFD with the tags listed in Facts) from a uint16 Bayer array, packing 12 bits the same way, so tests never need a real DNG.

**Produces (later tasks use these exactly):**

```python
# pifilm/raw.py
class RawError(ValueError): ...

@dataclass(frozen=True)
class RawFrame:
    bayer: np.ndarray            # (H, W) uint16, values 0..white
    cfa: str                     # "BGGR", "RGGB", "GRBG" or "GBRG", top-left 2x2 read row by row
    black: float                 # one black level (the four are equal on this camera; refuse unequal ones)
    white: float
    as_shot_neutral: tuple[float, float, float] | None

def read_dng(data: bytes) -> RawFrame:
    """Parse a Picamera2 DNG: little-endian TIFF, one IFD, uncompressed CFA strip,
    12- or 16-bit. Anything else (big-endian, compressed, tiles, several strips,
    other bit depths, missing tags) raises RawError naming what was found."""

@dataclass(frozen=True)
class RawTuning:
    black_level: float           # at 16 bits, as in the file
    ccms: tuple[tuple[float, np.ndarray], ...]       # (ct, 3x3) sorted by ct
    alsc_cr: tuple[tuple[float, np.ndarray], ...]    # (ct, (12, 16)) sorted by ct
    alsc_cb: tuple[tuple[float, np.ndarray], ...]
    luminance: np.ndarray        # (12, 16), luminance_lut with luminance_strength applied
    gamma_curve: np.ndarray      # (33, 2) in 0..1, x then y

def tuning_tables(tuning: dict) -> RawTuning: ...   # RawError when a block is missing or malformed
def ccm_at(t: RawTuning, ct: float) -> np.ndarray:  # linear interpolation in ct between neighbours, clamped at the ends
def alsc_at(t: RawTuning, ct: float) -> tuple[np.ndarray, np.ndarray]:   # (cr, cb) grids, same rule
```

**Behaviour:**

- `read_dng`: header `II*\0`, the IFD at the offset in bytes 4..8; read tags 256, 257, 258, 259, 262, 273, 277, 279, 33421, 33422, 50714, 50717, 50728; require Compression 1, Photometric 32803, SamplesPerPixel 1, CFARepeatDim (2, 2), one strip of exactly `width * height * bits / 8` bytes. Unpack 12-bit with NumPy only (vectorised, no Python loop over pixels). Map the CFA bytes to the pattern string. Values above WhiteLevel are a RawError (a sign of a wrong unpack).
- `luminance_strength`: the ISP applies the luminance table partially. Use `1 + (lut - 1) * strength` and state in the docstring that this matches libcamera's `rpi/controller/rpi/alsc.cpp`; if you can read that source (it is open: raspberrypi/libcamera on GitHub), confirm the formula and cite the line; if you cannot, say in the report that it is unconfirmed and how the Task 3 gallery checks it (vignetting in a flat sky).
- `gamma_curve` scaled from 0..65535 to 0..1.

**Tests:**

1. Round trip: a random 12-bit Bayer array written by the test helper reads back equal, for 12- and 16-bit, for each of the four CFA patterns.
2. The 12-bit bit order: a hand-made 3-byte strip `0xAB 0xCD 0xEF` reads as pixels `0xABC` and `0xDEF`.
3. Refusals: compressed, tiled, two strips, wrong byte count, big-endian, missing CFA tags, unequal black levels, a value above the white level: each a RawError with a message naming the problem.
4. `tuning_tables` on a small hand-made tuning dict with the same structure (do not load `data/tuning/imx477.json` in tests: it is not in a clone); missing `rpi.ccm` or `rpi.alsc`, a table of the wrong length: RawError.
5. `ccm_at`, `alsc_at`: exact at a calibrated ct, the midpoint halfway between two, clamped below the first and above the last.
6. In the report (not a test): `read_dng` on `154700.dng` equals `rawpy`'s `raw_image_visible` exactly (state the command and the result), and `tuning_tables(json.load(open("data/tuning/imx477.json")))` loads.

Commits: small, one per concern.

---

### Task 2: the developer

**Files:**
- Modify: `pifilm/raw.py` (developer part), `tests/test_raw.py`

**Consumes:** Task 1's `RawFrame`, `RawTuning`, `ccm_at`, `alsc_at`.

**Produces:**

```python
TONE_CURVES = ("isp", "soft-1", "soft-1.5")

def develop(frame: RawFrame, tuning: RawTuning, *, colour_gains: tuple[float, float],
            colour_temperature: float, tone: str = "isp") -> np.ndarray:
    """(H, W, 3) uint8 sRGB. colour_gains is the record's ColourGains (red, blue)."""
```

**Behaviour (in this order, each step a small private function so it can be tested alone):**

1. To float32 linear: `(bayer - black) / (white - black)`, clipped to 0..1.
2. Lens shading on the Bayer plane: upsample the (12, 16) grids bilinearly to (H, W) (`cv2.resize` with `INTER_LINEAR` is acceptable; grid cell centres at the frame's 16 x 12 equal divisions); multiply red sites by `cr * lum`, blue sites by `cb * lum`, green sites by `lum`.
3. White balance on the Bayer plane: red sites x `colour_gains[0]`, blue sites x `colour_gains[1]`, then clip to 1.0 (so a clipped highlight stays neutral; say so in a comment).
4. Demosaic: scale to uint16 (x 65535, rounded), `cv2.cvtColor` with the Bayer code that is correct for the frame's CFA, to RGB, back to float32. Prove the code with a test (below), and keep the table from CFA string to OpenCV code in one place with a comment on OpenCV's naming.
5. Colour matrix: `ccm_at(tuning, colour_temperature)` applied per pixel (`rgb @ ccm.T`), clipped to 0..1. Work in row bands if needed to keep memory to a few full-frame arrays.
6. Tone: `"isp"` is the tuning's `gamma_curve` (linear interpolation between its points, x linear in, y out, as display-referred 0..1); `"soft-<e>"` is the same curve applied after `f(x) = x * 2**e / (1 + x * (2**e - 1))`, which keeps 0 and 1, lifts the shadows by `e` stops and rolls the highlights off. Unknown names raise ValueError.
7. To uint8 with rounding.

The `"isp"` output is treated as already sRGB-encoded (the ISP's gamma curve is what the camera's JPEG is encoded with); do not apply an sRGB transfer on top.

**Tests:**

1. The demosaic code for each CFA: a synthetic frame of four flat colour patches (pure red, green, blue, grey) mosaicked with that pattern, developed with identity lens shading, unit gains, identity CCM and a linear tone curve, gives each patch's colour within a few levels in its interior (away from patch edges).
2. White balance: a grey patch with known gains comes out neutral.
3. A clipped patch (raw at white in all channels) with strong gains comes out white (all channels above 250), not tinted.
4. Lens shading: a flat raw frame with a vignetting luminance table comes out flatter than without it (centre to corner ratio closer to 1).
5. `ccm_at` is the matrix used: developing with ct between two calibrations equals developing with the interpolated matrix supplied by hand.
6. Tone curves: `"soft-1"` maps 0 to 0, 1 to 255, is monotone, and is brighter than `"isp"` in the shadows; unknown name raises.
7. Determinism: two runs give identical bytes.
8. In the report (not a test): wall time and peak memory (`tracemalloc` or `resource`) of `develop` on one real 12 MP DNG on this Mac.

Commits: small, one per concern.

---

### Task 3: the base gallery

**Files:**
- Create: `scripts/experiments/raw_base_gallery.py`
- No tests (an experiment script, like `scripts/experiments/exposure_from_lux.py`; read that file and follow its structure and HTML style).

**Consumes:** `read_dng`, `tuning_tables`, `develop`, `TONE_CURVES`.

**Behaviour:**

- Arguments: `--tuning data/tuning/imx477.json`, `--out artifacts/raw-base-gallery`, `--long-edge 1600`, and capture folders. For every record with a `dng` field whose DNG and original exist and whose `tuning_file` names imx477, develop the DNG with each tone curve, using the record's `camera_metadata` `ColourGains` and `ColourTemperature`.
- Inputs to run on (put the exact command in the module docstring): the 141 training photographs (`~/Downloads/PiTrainingSource/2026-09-30`), and the 13 held-out scenes (their folders under `~/Downloads/PiCameraPhotos/`; include only the 13 named in `data/test-scenes-imx477/` with a `--only-names` option or a list file, and mark them "held out" in the page).
- Page: one row per photograph, columns: the camera's `_original.jpg`, `isp`, `soft-1`, `soft-1.5`; under each the median linear luminance and the share of pixels at 255. Grouped by the record's Lux in the bins `exposure_from_lux.py` uses. A short header saying what each column is and that nothing here is graded (no LUT, no grain).
- Validation numbers, printed and in a `summary.json`: for the `isp` column against the camera's JPEG of the same shot, the mean absolute difference per channel and the mean difference per channel (a colour cast shows as a signed mean), across all photographs and per Lux bin. Expected: close but not equal (the ISP adds adaptive contrast, denoise, sharpening and adaptive lens shading). A signed mean beyond about 8 levels in one channel, or a centre-to-corner brightness difference against the JPEG, points at a developer bug; investigate before reporting and say what you found.
- **Does 8 bits hold what the raw has?** (Added 2026-10-14: the owner's aim is to grade on the DNG's full information; this decides whether an 8-bit developed picture is enough or a float path is needed.) For the 6 darkest and the 6 brightest scenes by Lux, one more row each: a 600 x 400 crop of the deepest shadow area (or, for the bright ones, the brightest area that is not clipped), developed with `soft-1.5`, pushed by +2 stops (shadow crops) or -2 stops (highlight crops) after development, shown twice: once from the uint8 result of `develop`, once from the same development kept in float32 (add a keyword `as_float: bool = False` to `develop` that returns the float32 0..1 picture before the final rounding, and push both the same way). Under each, the number of distinct levels in the crop's green channel after the push. Banding or lost texture in the uint8 crop that the float crop does not have means the float path is needed.
- Run it, and put in the report: the command, the run time, the validation table, and three observations from looking at the page yourself (open a few of the JPEGs with the Read tool): colour, lens shading in flat areas, highlights, shadow noise in the darkest scenes.

Commit: the script only.
