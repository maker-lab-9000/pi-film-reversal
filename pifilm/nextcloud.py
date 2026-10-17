"""Push captured photos from the Pi to a Nextcloud folder over WebDAV.

Run this ON THE PI, driven by the same ``.env`` the other scripts read. It
mirrors ``~/Pictures/pifilm`` into one Nextcloud folder with rclone's WebDAV
backend -- the closest thing to rsync that a Nextcloud HTTP endpoint speaks.
``rclone copy`` only ever adds or updates; it never deletes a remote file, so
cleaning the SD card cannot erase the cloud archive.

Two guards keep it a quiet no-op off the home network, so a systemd timer can
fire it every few minutes and it does nothing until the Pi is on the home LAN,
wired or Wi-Fi. First, ``eth0`` then ``wlan0`` (the ``--iface`` list, in order)
is checked for a routable IPv4 (not a 169.254.x link-local address); the first
interface that has one is chosen. Second, the Nextcloud host must answer a TCP
connect. Only that first interface is named; the connect itself follows the
routing table, so trying the second interface on a failure would test the same
route. Hotspot mode needs no special case: ``wlan0`` then holds 10.42.0.1, a
routable address, but Nextcloud is not reachable from the hotspot, so the
second guard skips the run.

The Nextcloud password never reaches a command line, a process listing, or an
rclone config file on disk. It is obscured with ``rclone obscure`` (plaintext
on stdin) and handed to rclone through ``RCLONE_CONFIG_*`` environment
variables, which are readable only by the process owner.

Only one sync runs at a time. The settings menu's upload and the timer's run
can start together, and two rclone copies over the same files would race on
the same remote paths and double the upload. ``main`` takes an exclusive,
non-blocking ``flock`` on ``--lock`` (default
``$XDG_STATE_HOME/pifilm/nextcloud-sync.lock``) before ``run_sync``; a run that
finds it held prints one line and exits 0, a quiet no-op like the other guards.
The kernel drops the lock when the process ends, so a killed run cannot leave
it stuck.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .envfile import parse_env_file

REMOTE_NAME = "nextcloud"
DEFAULT_TARGET_DIR = "MartinParr"
DEFAULT_SOURCE_DIR = "~/Pictures/pifilm"
DEFAULT_IFACES = ("eth0", "wlan0")
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


def home_lan_ipv4(iface: str, run: Callable = subprocess.run) -> str | None:
    """The interface's first routable IPv4, or None if it only has link-local."""
    for candidate in _iface_ipv4s(iface, run=run):
        try:
            address = ipaddress.IPv4Address(candidate)
        except ipaddress.AddressValueError:
            continue
        if not (address.is_link_local or address.is_loopback or address.is_unspecified):
            return str(address)
    return None


def first_home_lan_ipv4(
    ifaces: Sequence[str], run: Callable = subprocess.run
) -> tuple[str, str] | None:
    """The first interface in ``ifaces`` with a routable IPv4, and that address."""
    for iface in ifaces:
        ipv4 = home_lan_ipv4(iface, run=run)
        if ipv4 is not None:
            return iface, ipv4
    return None


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
    ifaces: Sequence[str] = DEFAULT_IFACES,
    rclone: str = "rclone",
    min_age: str = DEFAULT_MIN_AGE,
    run: Callable = subprocess.run,
    connect: Callable = socket.create_connection,
    which: Callable = shutil.which,
    log: Callable = print,
) -> int:
    chosen = first_home_lan_ipv4(ifaces, run=run)
    if chosen is None:
        log(f"no network on {' or '.join(ifaces)}; skipping Nextcloud sync")
        return 0
    iface, ipv4 = chosen
    host, port = cfg.host_port()
    if not nextcloud_reachable(host, port, connect=connect):
        log(f"Nextcloud {host}:{port} not reachable over {iface} ({ipv4}); skipping")
        return 0
    log(f"Nextcloud reachable over {iface} ({ipv4})")
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


def _default_lock() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "pifilm" / "nextcloud-sync.lock"


@contextlib.contextmanager
def _sync_lock(path: Path):
    """Yield True while this process holds the lock, False if another run does."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
        action="append",
        dest="ifaces",
        default=None,
        help=(
            "interface to gate on; repeat to try several in order, the first with a "
            f"routable IPv4 wins (default: {' then '.join(DEFAULT_IFACES)})"
        ),
    )
    parser.add_argument("--rclone", default="rclone", help="rclone binary (default: rclone)")
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="lock file that keeps one sync at a time "
        "(default: $XDG_STATE_HOME/pifilm/nextcloud-sync.lock)",
    )
    args = parser.parse_args(argv)
    try:
        cfg = config_from_env(parse_env_file(args.env))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ifaces = args.ifaces or list(DEFAULT_IFACES)
    with _sync_lock(args.lock or _default_lock()) as held:
        if not held:
            print("another Nextcloud sync is running; skipping")
            return 0
        return run_sync(cfg, apply=args.apply, ifaces=ifaces, rclone=args.rclone)


if __name__ == "__main__":
    raise SystemExit(main())
