"""Safely provide local StickS3 settings to PlatformIO without writing secrets.

PlatformIO executes this file as a pre-build script.  It reads the repository's
local .env as data and appends only the four firmware settings required by the
device.  In particular, PI_SSH_PASSWORD is deliberately not a firmware value.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_FIRMWARE_KEYS = (
    "STICKS3_WIFI_SSID",
    "STICKS3_WIFI_PASSWORD",
    "STICKS3_API_BASE",
    "STICKS3_API_TOKEN",
)


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a deliberately small dotenv format without evaluating any value.

    ``pifilm/envfile.py`` holds a copy for the package; keep the two in step.
    """
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


def _required(values: Mapping[str, str], key: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ValueError(f"missing required {key} in local env file")
    return value


def firmware_defines(values: Mapping[str, str]) -> list[tuple[str, str]]:
    """Return exactly the credentials and API values the firmware consumes."""
    host = _required(values, "PI_HOST")
    api_base = values.get("PIFILM_REMOTE_URL") or f"http://{host}:8765"
    defines = {
        "STICKS3_WIFI_SSID": _required(values, "WIFI_SSID"),
        "STICKS3_WIFI_PASSWORD": _required(values, "WIFI_PASSWORD"),
        "STICKS3_API_BASE": api_base,
        "STICKS3_API_TOKEN": _required(values, "PIFILM_REMOTE_TOKEN"),
    }
    return [(key, defines[key]) for key in _FIRMWARE_KEYS]


def platformio_defines(env_file: Path) -> list[tuple[str, str]]:
    """Return no local values on a clean clone, preserving safe C++ defaults."""
    if not env_file.is_file():
        return []
    return firmware_defines(parse_env_file(env_file))


def platformio_env_file(project_dir: Path) -> Path:
    """Map PlatformIO's firmware project directory to the checkout-local env."""
    return project_dir.resolve().parents[1] / ".env"


def configure_platformio(env_file: Path | None = None) -> None:
    """Append local defines when PlatformIO imports this module as a pre-script."""
    try:
        from SCons.Script import Import  # type: ignore[import-not-found]
    except ImportError:
        return
    Import("env")
    build_env = globals()["env"]
    if env_file is None:
        env_file = platformio_env_file(Path(str(build_env["PROJECT_DIR"])))
    defines = platformio_defines(env_file)
    if not defines:
        return
    # PlatformIO must preserve C string quotes through SCons and the shell.
    # Plain quoted strings lose their quotes before reaching the compiler.
    build_env.Append(CPPDEFINES=[
        (key, build_env.StringifyMacro(value.replace("\\", "\\\\")))
        for key, value in defines
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate local StickS3 firmware configuration")
    parser.add_argument(
        "--env", type=Path, default=Path(".env"), help="local env file (default: .env)"
    )
    args = parser.parse_args(argv)
    firmware_defines(parse_env_file(args.env))
    print("StickS3 firmware configuration is valid; no secret file was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    configure_platformio()
