"""Read the repository's ``.env`` as ``KEY=VALUE`` data, never evaluating a value.

The one parser for it: the Nextcloud sync, ``scripts/deploy_remote.py`` and
``scripts/pi_hotspot.py`` all read the file through ``parse_env_file``. A value
is data only; the file is never sourced, so nothing in it runs.
"""

from __future__ import annotations

import re
from pathlib import Path

_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a deliberately small dotenv format without evaluating any value."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid environment variable name")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if "\x00" in value:
            raise ValueError(f"{path}:{line_number}: NUL bytes are not allowed")
        values[key] = value
    return values
