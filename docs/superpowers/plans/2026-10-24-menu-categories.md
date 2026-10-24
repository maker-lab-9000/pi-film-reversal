# Menu categories and a MENU key in the bottom strip

Date: 2026-10-24. Branch `feat/menu-categories` from `main` (e02e5da, which has the
IMAGE and STABILISE pages). No separate spec: the user's request is the authority:

> Redesign the menu and break it down in multiple categories, like:
> Camera (Image, AE options); Pipeline (Save DNG, halation, stabilise);
> Network (ETH, WLAN, Nextcloud upload button). Keep Look on the main page.
> Move the menu button from the live view into the base bar with shutter, ISO
> and WB. Fit it between WB and the exposure meter.

The user decides the look from rendered screens (Task 3), not from metrics.

## Global Constraints

- Read CLAUDE.md, `docs/lcd-viewfinder.md` and the module docstrings of every
  file you change (`pifilm/display/menu.py`, `liveview.py`, `viewfinder.py`);
  keep docstrings accurate. Geometry is in 320x240 base units at an integer
  `scale` (1 for the 2.8" SPI panel, 2 for the 3.5" DSI panel); everything must
  hold at both.
- Tests: `../martin-parr-project/.venv/bin/pytest -q -m 'not slow'` from the
  worktree; ruff `../martin-parr-project/.venv/bin/ruff check .`. Every commit
  passes both. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- No change to settings keys, values, `settings.json`, records, or what any
  setting does. This is navigation and layout only.
- Never read, print or source `.env`. No SSH.

### Menu tree (exact)

```
MENU (title none, the first page)
  CAMERA        >
  PIPELINE      >
  NETWORK       >
  LOOK          <look name>      (read-only, as today)
  RESET         (two taps, all settings, as today's first-page RESET)
CAMERA
  IMAGE         >                (today's IMAGE page, unchanged inside)
  AE CONSTRAINT <mode>
  AE METERING   <mode>
  BACK
PIPELINE
  SAVE DNG      ON/OFF
  HALATION      ON/OFF/-
  STABILISE     >                (today's STABILISE page, unchanged inside)
  BACK
NETWORK
  ETH           <address>/-      (read-only)
  WLAN          <address>/-      (read-only)
  UPLOAD        <status>         (tap starts an upload, as today)
  BACK
```

- Each page below the first has an amber, non-tappable title line (as the IMAGE
  page has today). BACK goes up one level (IMAGE/STABILISE back to CAMERA /
  PIPELINE; a category back to the first page). DONE (where the shutter is)
  closes the menu from any page; the menu always opens on the first page. The
  IMAGE and STABILISE pages keep their own RESET.
- A category row is dim (inactive, value `-`) when none of its rows is active
  (e.g., CAMERA under `--fake`); a row that is not available today stays as it
  is today (dim, or absent where it is absent today — e.g., UPLOAD only with
  `--nextcloud-env`). NETWORK is always tappable (ETH/WLAN are informative).

### MENU key in the bottom strip (exact intent)

- The MENU key leaves its place under `-` (today `MENU_CENTRE (287, 199)`,
  `MENU_HIT_BOX (264, 190, 311, 215)`) and goes into the bottom strip, between
  the WB cell and the exposure meter, vertically centred on the strip's two
  text lines. It keeps today's look (outlined key, `MENU` label, lit while
  pressed).
- Today the WB cell's hit column ends at x 191 and the meter's `-` glyph sits
  at x 198, so there is no room: make room by narrowing the cells' pitch
  and/or the meter's span (keep 13 dots; the meter text stays centred over the
  meter). Choose numbers so that: the three cells' widest values still fit
  ("1/1000", "1600", "7500K" and "AUTO"); the MENU key's hit box is at least
  32 x 26 units and does not overlap a cell's hit column or the meter; the
  meter keeps at least 5 units between dots. Name every constant and say why
  in a comment.
- The freed spot under `-` stays empty.

## Tasks

### Task 1: the menu tree

- Generalise the page mechanism in `viewfinder.py` (`_PAGES`, `_PAGE_RESETS`,
  `_PAGE_KEYS`, the first-page row builder, `_menu_tap`) into a tree: each page
  has a title, its rows, an optional page RESET, and a parent. Keep IMAGE and
  STABILISE exactly as they are inside.
- `menu.py`: anything needed so every page fits (rows end before y 240 at both
  scales; nothing overlaps DONE, hit circle from x 253).
- Tests: every page's rows and order; BACK from each page goes to its parent;
  DONE from each page closes; menu reopens on the first page; RESET on the
  first page resets all; IMAGE/STABILISE resets unchanged; category rows dim
  under `--fake`; UPLOAD only with an uploader; a GPIO press while the menu is
  open still takes a photograph and the menu stays on its page; geometry.

### Task 2: MENU key in the strip

- `liveview.py`: move the MENU key and its hit box as above; re-lay the cells
  and/or the meter; update the hit-testing (`MENU` vs cells vs meter text).
- Tests: the key is drawn in the strip, between WB's hit column and the meter,
  at both scales; a tap on it opens the menu; taps on each cell still select
  that cell; the widest values fit their cells; the meter dot positions and
  the meter text stay consistent; nothing is drawn where the old key was.

### Task 3: renders for the user, and docs

- `scripts/render_menu_preview.py` (no camera, no `.env`): writes PNGs at
  scale 2 of the live view (a sample frame, all three cells with typical
  values, meter showing, IMG/STAB tags on, the MENU key unpressed and pressed)
  and of every menu page (first, CAMERA, PIPELINE, NETWORK, IMAGE,
  STABILISE), into a folder given on the command line. A small test.
- Docs: CLAUDE.md (display section: the tree, the key's new place),
  `docs/lcd-viewfinder.md` (screen layout, menu table, acceptance steps
  renumbered/updated: open MENU from the strip, walk every page, BACK, DONE).
