# A look trained on the DNG: design (draft for the owner)

Date: 2026-10-14. Asked for by the owner: work from the raw file, which keeps more
detail in shadows and highlights than the camera's JPEG, with a look retrained for
it. Status: **draft, awaiting the owner's approval.** The task-by-task
implementation plan follows once this is agreed.

This reverses an entry in the roadmap's "Considered and not planned" ("Grading from
the raw stream ... means rebuilding demosaic, lens shading, white balance, colour
matrix and denoise, and retraining every look"). That cost is real and is what
this document plans for. The current look stays deployed, unchanged, until a raw
look has beaten it in a gallery.

## What we have (checked 2026-10-14)

- **Training photographs.** The current LUT (`98e36716`, inside
  `imx477-2026-09-v4-dark`) was trained by
  `pifilm-train --source data/source-imx477-v3-highlight --target data/references-merged-2026-09-29 ...`.
  The 141 source photographs are the 2026-09-30 originals from
  `~/Downloads/PiTrainingSource/2026-09-30/` (byte-identical copies). That folder
  also holds all 141 DNGs (2.6 GB) and the `captures.jsonl` that names them, so no
  download from Nextcloud is needed for the sources.
- **Held-out scenes.** The 13 frames in `data/test-scenes-imx477/` all have their
  DNG on the Mac too (`~/Downloads/PiCameraPhotos/`).
- **The references** (`data/references-merged-2026-09-29`, 168 Parr photographs)
  are scans and JPEGs. They stay as they are: the look is still "what a Parr
  print looks like", only the input changes.
- **The DNGs.** Picamera2 writes the sensor's 12-bit Bayer data (white level 4095,
  black level 256) with the shot's white-balance gains, but no usable colour
  matrix (LibRaw reads it as zeros). Before lens shading, before the ISP's colour
  matrix, tone curve, adaptive contrast and denoise.
- **The tuning file** (`imx477.json`) holds what the ISP uses for those steps:
  colour matrices by colour temperature (`rpi.ccm`), lens-shading tables by colour
  temperature (`rpi.alsc`), the black level, the tone curve. Each record carries
  the shot's `ColourGains` and `ColourTemperature`, and since step 4 the tuning
  file's fingerprint.

## The approach

### One developer, in this repo, on both machines

A new module `pifilm/raw.py` turns a DNG (or the raw array in memory) into the RGB
picture the rest of the pipeline already takes:

1. Subtract the black level; scale to 0..1 by the white level.
2. Lens shading: the tuning file's `rpi.alsc` tables for the shot's colour
   temperature, interpolated like the ISP does.
3. White balance: the shot's `ColourGains`.
4. Demosaic: OpenCV's `cv2.demosaicing` (fast C, already a dependency; the edge-aware
   variant if the gallery shows colour fringes).
5. Colour matrix: `rpi.ccm` interpolated at the shot's `ColourTemperature`, to
   linear sRGB.
6. Tone: a fixed curve to 8-bit sRGB with a long shoulder and a lifted toe, chosen
   to keep what the JPEG throws away (the point of the exercise). One curve for
   every shot, a version number on it.

Why our own and not LibRaw (which developed the 2026-10-14 test): it must run on
the Pi at capture time and give exactly the same pixels as on the Mac at training
time, the same principle as normalisation (one piece of maths on both sides). LibRaw
does not know this camera's colour matrix or lens shading, and is not installed
on the Pi. OpenCV is on both.

Everything after the developer is today's pipeline: normalise, LUT, grain. The
trainer, its gates, the report, the GPU pass and `pifilm-process` stay as they are;
only what feeds them changes.

What is not rebuilt: denoise (the grain sits on top; the high-ISO shadows are
checked in the gallery first) and the ISP's adaptive contrast (the owner keeps it
on for the JPEG look; the raw look gets whatever contrast its curve and LUT give,
judged by gallery).

### The look

A new artifact, trained exactly as the current one but with the 141 DNGs developed
by `pifilm/raw.py` as sources, the same references, the same held-out scenes, the
same gates. Its `params.json` says what input it expects: `"input": "dng-1"` (the
developer's version), so a build or a camera that cannot give it that input refuses
the look instead of grading the JPEG with it. That is a params version (5) for
such artifacts only; every existing look keeps its version and its behaviour. It
also carries the tuning fingerprint (step 4): the developer reads the tuning file,
so a changed tuning file changes the input more directly than before.

### On the Pi

The capture already holds the raw frame (it writes the DNG). The grade worker
develops it, then grades as now. The quick preview for the screen and the Stick
stays on the camera's JPEG (it is shown for seconds; speed matters there). The
`_original.jpg` and the DNG are saved as now; the graded file is the raw look.
Records gain `input: "dng-1"`; `pifilm-process` regrades a raw-look record from its
DNG.

Cost on the Pi is unknown: demosaicing and the colour matrix on 12 MP in NumPy and
OpenCV is probably 0.5 to 1.5 s on top of today's background grade. It is behind
the shutter, so the wait between photographs does not change; the graded file
comes later. Measured in phase 3 before anything is wired in.

## Phases, each with a gate

1. **Developer and a base gallery (Mac).** `pifilm/raw.py` with tests; a gallery of
   the 141 sources and 13 held-out scenes: the camera's JPEG beside the developed
   DNG, ungraded, and a few with the highlights and shadows pushed to show what the
   raw keeps. **Gate (owner):** the developed pictures are a sound base (colour,
   no shading or fringes, shadow noise acceptable). The tone curve is chosen here.
2. **Train the raw look (Mac).** `pifilm-train` on the developed DNGs; the report
   and its gates; a gallery: the current look on the JPEG beside the raw look on
   the DNG, at full size, including night, backlit and bright scenes. **Gate
   (owner):** the raw look is better, or at least as good with more highlight and
   shadow detail. If not, stop here: nothing has changed on the Pi.
3. **Pi spike (throwaway).** Time the developer on the Pi at 12 MP, its memory, and
   that it gives the Mac's pixels exactly. **Gate:** the background grade stays
   acceptable (to be agreed with the numbers).
4. **Wire it in.** The `input` field and its refusal, the grade worker developing
   the raw frame, records, `pifilm-process` from DNG, docs. Its own plan.
5. **Hardware acceptance** with the raw look deployed; the JPEG look one flag away.

## Decisions for the owner

1. **Our own developer** (this document's recommendation) or LibRaw, accepting that
   LibRaw would then be needed on the Pi and its colour would not be the camera's.
2. **The GPU.** The developer runs on the CPU at first. Moving it into the GPU pass
   is possible later (the pass would take 16-bit raw instead of 8-bit RGB) and is
   left out of this plan.
3. **Training set.** The same 141 photographs (recommended: a like-for-like
   comparison with today's look), or more DNGs from Nextcloud as well (more
   variety, but the comparison is no longer like for like).

## Not in this plan

- Denoise, a float or 16-bit path through the LUT, the GPU developer.
- Retraining the JPEG look.
- The IMX708 (its tuning file and DNGs differ; the developer is written for any
  sensor whose tuning file has `rpi.ccm` and `rpi.alsc`, but only the IMX477 is
  tested).
