"""Push captured photos from the Pi to a Nextcloud folder over WebDAV.

Run this ON THE PI, driven by the same ``.env`` the other scripts read. It
mirrors ``~/Pictures/pifilm`` into one Nextcloud folder with rclone's WebDAV
backend -- the closest thing to rsync that a Nextcloud HTTP endpoint speaks.
``rclone copy`` only ever adds or updates; it never deletes a remote file, so
cleaning the SD card cannot erase the cloud archive.

Two guards keep it a quiet no-op off the wired LAN, so a systemd timer can fire
it every few minutes and it does nothing until the Pi is plugged into the home
network: it runs only when ``eth0`` holds a real home-LAN IPv4 (not a
169.254.x link-local address) and the Nextcloud host answers a TCP connect.

The Nextcloud password never reaches a command line, a process listing, or an
rclone config file on disk. It is obscured with ``rclone obscure`` (plaintext
on stdin) and handed to rclone through ``RCLONE_CONFIG_*`` environment
variables, which are readable only by the process owner.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firmware.sticks3.scripts.generate_config import parse_env_file

REMOTE_NAME = "nextcloud"
DEFAULT_TARGET_DIR = "MartinParr"
DEFAULT_SOURCE_DIR = "~/Pictures/pifilm"
DEFAULT_IFACE = "eth0"
DEFAULT_MIN_AGE = "30s"


def _required(values: Mapping[str, str], key: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ValueError(f"missing required {key} in local env file")
    return value


@dataclass(frozen=True)
class NextcloudConfig:
    base_url: str
    user: str
    password: str
    target_dir: str
    source_dir: Path

    def webdav_url(self) -> str:
        """Nextcloud's per-user WebDAV endpoint for the configured account."""
        base = self.base_url.rstrip("/")
        return f"{base}/remote.php/dav/files/{self.user}/"

    def host_port(self) -> tuple[str, int]:
        parts = urlsplit(self.base_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return host, port


def config_from_env(values: Mapping[str, str]) -> NextcloudConfig:
    base_url = _required(values, "NEXTCLOUD_URL")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            "NEXTCLOUD_URL must be an http(s) URL such as http://192.168.178.241:8083/, "
            f"got {base_url!r}"
        )
    source = values.get("NEXTCLOUD_SOURCE_DIR") or DEFAULT_SOURCE_DIR
    return NextcloudConfig(
        base_url=base_url,
        user=_required(values, "NEXTCLOUD_USER"),
        password=_required(values, "NEXTCLOUD_PASSWORD"),
        target_dir=values.get("NEXTCLOUD_TARGET_DIR") or DEFAULT_TARGET_DIR,
        source_dir=Path(source).expanduser(),
    )


def _iface_ipv4s(iface: str, run: Callable = subprocess.run) -> list[str]:
    result = run(
        ["ip", "-o", "-4", "addr", "show", "dev", iface],
        capture_output=True,
        text=True,
        check=False,
    )
    tokens = (result.stdout or "").split()
    return [tokens[i + 1].split("/")[0] for i, tok in enumerate(tokens) if tok == "inet"]


def home_lan_ipv4(iface: str = DEFAULT_IFACE, run: Callable = subprocess.run) -> str | None:
    """The interface's first routable IPv4, or None if it only has link-local."""
    for candidate in _iface_ipv4s(iface, run=run):
        try:
            address = ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            continue
        if not (address.is_link_local or address.is_loopback or address.is_unspecified):
            return str(address)
    return None


def ethernet_ready(iface: str = DEFAULT_IFACE, run: Callable = subprocess.run) -> bool:
    return home_lan_ipv4(iface, run=run) is not None


def nextcloud_reachable(
    host: str,
    port: int,
    timeout: float = 3.0,
    connect: Callable = socket.create_connection,
) -> bool:
    try:
        with connect((host, port), timeout):
            return True
    except OSError:
        return False


def obscure_password(plaintext: str, rclone: str = "rclone", run: Callable = subprocess.run) -> str:
    """Obscure the password via rclone, keeping the plaintext on stdin only."""
    result = run(
        [rclone, "obscure", "-"],
        input=plaintext,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def rclone_env(
    cfg: NextcloudConfig, obscured_password: str, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    prefix = f"RCLONE_CONFIG_{REMOTE_NAME.upper()}_"
    env[prefix + "TYPE"] = "webdav"
    env[prefix + "URL"] = cfg.webdav_url()
    env[prefix + "VENDOR"] = "nextcloud"
    env[prefix + "USER"] = cfg.user
    env[prefix + "PASS"] = obscured_password
    return env


def rclone_copy_command(
    cfg: NextcloudConfig,
    *,
    dry_run: bool,
    rclone: str = "rclone",
    min_age: str = DEFAULT_MIN_AGE,
) -> list[str]:
    command = [
        rclone,
        "copy",
        str(cfg.source_dir),
        f"{REMOTE_NAME}:{cfg.target_dir}",
        "--min-age",
        min_age,
        # The photo folder is its own ext4 volume (docs/photo-volume.md); its
        # root-only lost+found made every run fail with "permission denied".
        "--exclude",
        "/lost+found/**",
        "--verbose",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def run_sync(
    cfg: NextcloudConfig,
    *,
    apply: bool,
    iface: str = DEFAULT_IFACE,
    rclone: str = "rclone",
    min_age: str = DEFAULT_MIN_AGE,
    run: Callable = subprocess.run,
    connect: Callable = socket.create_connection,
    which: Callable = shutil.which,
    log: Callable = print,
) -> int:
    ipv4 = home_lan_ipv4(iface, run=run)
    if ipv4 is None:
        log(f"{iface} has no home-LAN IPv4; skipping Nextcloud sync")
        return 0
    host, port = cfg.host_port()
    if not nextcloud_reachable(host, port, connect=connect):
        log(f"Nextcloud {host}:{port} not reachable over {iface} ({ipv4}); skipping")
        return 0
    if which(rclone) is None:
        log(f"error: {rclone} not found; install it with: sudo apt install rclone")
        return 3
    try:
        obscured = obscure_password(cfg.password, rclone=rclone, run=run)
        env = rclone_env(cfg, obscured)
        command = rclone_copy_command(cfg, dry_run=not apply, rclone=rclone, min_age=min_age)
        result = run(command, env=env)
    except subprocess.CalledProcessError as exc:
        log(f"error: rclone failed: {exc}")
        return 4
    mode = "copied" if apply else "dry-run"
    target = f"{REMOTE_NAME}:{cfg.target_dir}"
    log(f"rclone {mode} {cfg.source_dir} -> {target} (exit {result.returncode})")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=Path, default=Path(".env"), help="local env file (default: .env)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform the copy (default: dry-run)"
    )
    parser.add_argument(
        "--iface",
        default=DEFAULT_IFACE,
        help=f"wired interface to gate on (default: {DEFAULT_IFACE})",
    )
    parser.add_argument("--rclone", default="rclone", help="rclone binary (default: rclone)")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_env(parse_env_file(args.env))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return run_sync(cfg, apply=args.apply, iface=args.iface, rclone=args.rclone)


if __name__ == "__main__":
    raise SystemExit(main())
