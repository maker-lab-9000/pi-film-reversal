"""Turn the Pi into a field Wi-Fi access point, driven by the local .env.

Run this ON THE PI. Away from home a phone or laptop joins the Pi's own
network and reaches it over SSH at the hotspot address. ``WIFI_SSID`` /
``WIFI_PASSWORD`` in ``.env`` become the hotspot credentials and the optional
``PI_HOTSPOT_ADDRESS`` (a plain IPv4 address, default ``10.42.0.1``) is the
address the Pi owns on it, on a /24 it hands out by DHCP.

The connection is written as a NetworkManager keyfile with mode 0600 rather
than passed to ``nmcli`` on the command line, so the passphrase never appears
in a process listing or a shell history. Without ``--apply`` nothing is
written; the keyfile is printed with the passphrase redacted.

Activating the access point takes ``wlan0`` away from any home Wi-Fi it was
joined to. Be connected over Ethernet before running with ``--apply``.

The profile disables Protected Management Frames (``pmf=1``); the Pi 3B radio
cannot install the AES-CMAC key that NetworkManager's "optional" default asks
for, and the access point fails to start without this. Measured on
Raspberry Pi OS Trixie, NetworkManager 1.52, brcmfmac firmware 7.45.98.

It also disables Wi-Fi power save (``powersave=2``). The brcmfmac driver
otherwise enables power management on ``wlan0`` (observed when eth0's carrier
drops), which makes the hotspot drop clients: a client associates and gets a
lease, then re-associates every few minutes and its connections stall.
Measured on a Pi 4 running the migrated SD, 2026-09-19.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifilm.envfile import parse_env_file

CONNECTION_ID = "pifilm-ap"
DEFAULT_KEYFILE = Path("/etc/NetworkManager/system-connections/pifilm-ap.nmconnection")
DEFAULT_ADDRESS = "10.42.0.1"
_UUID_NAMESPACE = uuid.UUID("6f1c2b3e-7a4d-4c1e-9b8a-2d5f0e7c1a90")
_PSK_LINE = re.compile(r"^psk=.*$", re.MULTILINE)


def _required(values: Mapping[str, str], key: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ValueError(f"missing required {key} in local env file")
    return value


def _ssid(values: Mapping[str, str]) -> str:
    ssid = _required(values, "WIFI_SSID")
    if len(ssid.encode("utf-8")) > 32 or not ssid.isprintable():
        raise ValueError("WIFI_SSID must be printable and at most 32 bytes")
    return ssid


def _psk(values: Mapping[str, str]) -> str:
    psk = _required(values, "WIFI_PASSWORD")
    if not 8 <= len(psk) <= 63 or not psk.isascii() or not psk.isprintable():
        raise ValueError("WIFI_PASSWORD must be 8 to 63 printable ASCII characters (WPA2-PSK)")
    return psk


def _ap_address(values: Mapping[str, str]) -> str:
    """The IPv4 address the hotspot owns, as the /24 it serves."""
    value = values.get("PI_HOTSPOT_ADDRESS") or DEFAULT_ADDRESS
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise ValueError(
            f"PI_HOTSPOT_ADDRESS must be a plain IPv4 address such as {DEFAULT_ADDRESS}, "
            f"got {value!r}"
        ) from exc
    return f"{address}/24"


def hotspot_keyfile(values: Mapping[str, str]) -> str:
    """A NetworkManager keyfile for a WPA2 access point on wlan0 with DHCP for clients."""
    ssid = _ssid(values)
    psk = _psk(values)
    address = _ap_address(values)
    connection_uuid = uuid.uuid5(_UUID_NAMESPACE, ssid)
    return (
        "[connection]\n"
        f"id={CONNECTION_ID}\n"
        f"uuid={connection_uuid}\n"
        "type=wifi\n"
        "interface-name=wlan0\n"
        "autoconnect=true\n"
        "autoconnect-priority=10\n"
        "\n"
        "[wifi]\n"
        "mode=ap\n"
        f"ssid={ssid}\n"
        "band=bg\n"
        # powersave=2 disables NetworkManager Wi-Fi power save on the AP. The
        # brcmfmac driver otherwise enables power management on wlan0 (observed
        # when eth0's carrier drops), which makes the hotspot drop clients: a
        # client associates and gets a lease, then re-associates every few minutes
        # and its connections stall. Measured on a Pi 4 running the migrated SD,
        # 2026-09-19; unplugging Ethernet made it obvious in the field.
        "powersave=2\n"
        "\n"
        "[wifi-security]\n"
        "key-mgmt=wpa-psk\n"
        "proto=rsn\n"
        # pmf=1 disables Protected Management Frames. NetworkManager's default
        # ("optional") makes hostapd install an AES-CMAC group key at start-up;
        # the Pi 3B / Zero W BCM43430 firmware has no MFP, the driver does not
        # advertise CMAC, and the kernel refuses the key, so the AP never comes
        # up ("key setting validation failed"). Clients connect without PMF.
        "pmf=1\n"
        f"psk={psk}\n"
        "\n"
        "[ipv4]\n"
        "method=shared\n"
        f"address1={address}\n"
        "\n"
        "[ipv6]\n"
        "method=disabled\n"
    )


def redact_psk(keyfile: str) -> str:
    return _PSK_LINE.sub("psk=<redacted>", keyfile)


def apply_keyfile(
    keyfile: str,
    path: Path,
    run: Callable[..., object] = subprocess.run,
) -> None:
    """Write the keyfile root-only, then have NetworkManager load and activate it."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(keyfile)
    os.chmod(path, 0o600)
    run(["nmcli", "connection", "reload"], check=True)
    run(["nmcli", "connection", "up", CONNECTION_ID], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=Path, default=Path(".env"), help="local env file (default: .env)"
    )
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_KEYFILE,
        help="keyfile destination (default: NetworkManager system-connections)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="write the keyfile and activate the hotspot (requires root)",
    )
    args = parser.parse_args(argv)
    try:
        keyfile = hotspot_keyfile(parse_env_file(args.env))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.apply:
        print(redact_psk(keyfile), end="")
        print(
            "\nDry run: nothing written. Connect over Ethernet first, then run\n"
            f"  sudo {sys.executable} {sys.argv[0]} --env {args.env} --apply",
        )
        return 0
    if os.geteuid() != 0:
        print("error: --apply must run as root (use sudo)", file=sys.stderr)
        return 2
    try:
        apply_keyfile(keyfile, args.path)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: could not activate {CONNECTION_ID}: {exc}", file=sys.stderr)
        return 1
    print(f"Hotspot {CONNECTION_ID} written to {args.path} and activated.")
    print("Check with: nmcli connection show --active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
