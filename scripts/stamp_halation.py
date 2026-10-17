"""Turn halation on (or off) in a look's params.json, in place.

    .venv/bin/python scripts/stamp_halation.py --artifacts artifacts/imx477-2026-09-v4-dark

The defaults are the values chosen from the galleries on 2026-10-17
(``pifilm.halation``). Turning it on raises the params version to 5, so a build from
before halation refuses the look rather than grading it without the glow.
"""

from __future__ import annotations

import argparse
import json
import sys

from pifilm.artifacts import ArtifactsError, stamp_halation
from pifilm.halation import HalationParams


def main(argv: list[str] | None = None) -> int:
    defaults = HalationParams()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--strength", type=float, default=defaults.strength)
    ap.add_argument("--sigma-mm", type=float, default=defaults.sigma_mm)
    ap.add_argument("--headroom", type=float, default=defaults.headroom)
    ap.add_argument("--off", action="store_true", help="write the section switched off")
    args = ap.parse_args(argv)
    try:
        params = HalationParams(enabled=not args.off, strength=args.strength,
                                sigma_mm=args.sigma_mm, headroom=args.headroom)
        path = stamp_halation(args.artifacts, params)
    except (ValueError, ArtifactsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"stamped {path}")
    print(json.dumps(params.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
