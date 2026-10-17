"""Push captured photos to Nextcloud: a wrapper kept for the installed timer unit.

The sync lives in ``pifilm.nextcloud`` so the capture service can run it with
``python -m pifilm.nextcloud`` from its menu; the unit
``deploy/pifilm-nextcloud-sync.service.example`` still runs this path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifilm.nextcloud import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
