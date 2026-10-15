"""The Pi's IPv4 addresses on Ethernet and Wi-Fi, for the viewfinder's ETH / WLAN tag.

Reaching the Pi over SSH, or pointing a browser or the Stick at it, needs its
address, and the Pi has no other screen to show it on: the address a router hands
out on Ethernet changes from network to network, and ``parr.local`` needs mDNS,
which not every network or client resolves. So the capture app reads the IPv4
address of ``eth0`` and of ``wlan0`` and the viewfinder shows them at the bottom
left of the picture, ``ETH 192.168.1.20   WLAN 192.168.178.56``, Ethernet first.
Nothing is drawn while neither has one. Information, not a warning, so the tag is
white, not amber.

How it is read. An interface counts while ``/sys/class/net/<if>/operstate`` reads
``up`` (a missing interface or file is not up): the kernel's own word for a cable
plugged in, or a Wi-Fi link associated, with no tool to run and nothing to parse.
Its address then comes from the ``SIOCGIFADDR`` ioctl on an ``AF_INET`` socket,
the call ``ifconfig`` makes: the standard library has it (``fcntl``), so there is
no ``ip`` subprocess every few seconds and no ``netifaces`` dependency. The ioctl
returns the interface's primary IPv4 address, the one a person would type; an
interface with none raises OSError, and so does any machine that is not Linux,
which is how a run on a Mac reads as nothing connected. IPv4 only, and only these
two interfaces: those are what the Pi has and what a person types.

The Pi's own hotspot (``scripts/pi_hotspot.py``) is shown like any other Wi-Fi
link, ``WLAN 10.42.0.1``: that is the address the Stick and a phone joined to the
hotspot reach the Pi on, so it is the one worth seeing, and telling an access
point from a client would need NetworkManager for no gain on the screen.

The poller is a daemon thread, ``pifilm-network``, shaped like the disk monitor
(``disk.py``): it checks at once when started, then every 10 s on an event that
``close`` sets, so stopping it never waits out the interval. 10 s because a
person who has just plugged in a cable looks at the screen for the address
straight away, and a check is two small file reads and an ioctl. It logs one line
when an address appears or changes (``network: wlan0 192.168.178.56``) and one
when it goes (``network: eth0 gone``), through the session's log, so the journal
says which address the Pi had when. Nothing a read raises leaves the poller: a
failure is one log line per spell of failures and the last addresses stay shown.
The current addresses are read by display threads at every frame under a lock
and never touch the network.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

DEFAULT_INTERVAL_S = 10.0
SYSFS_NET = Path("/sys/class/net")
# (label on the screen, interface), in the order they are shown.
INTERFACES = (("ETH", "eth0"), ("WLAN", "wlan0"))
SIOCGIFADDR = 0x8915

Addresses = tuple[tuple[str, str], ...]


def is_up(name: str, *, sysfs: Path = SYSFS_NET) -> bool:
    """True when the kernel reports the interface's operational state as ``up``."""
    try:
        return (Path(sysfs) / name / "operstate").read_text().strip() == "up"
    except OSError:
        return False


def ipv4_address(name: str) -> str | None:
    """The interface's primary IPv4 address, or None when it has none.

    None too for an interface that does not exist and on a machine that is not
    Linux (the ioctl fails there); never raises for either.
    """
    try:
        import fcntl
        import socket
        import struct

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            request = struct.pack("256s", name.encode()[:15])
            reply = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, request)
        # struct ifreq: 16 bytes of name, then a sockaddr_in whose address is at 4.
        return socket.inet_ntoa(reply[20:24])
    except (OSError, ImportError):
        return None


def addresses(*, sysfs: Path = SYSFS_NET,
              lookup: Callable[[str], str | None] = ipv4_address) -> Addresses:
    """``(label, address)`` for each interface that is up with an address, ETH first."""
    found = []
    for label, name in INTERFACES:
        if is_up(name, sysfs=sysfs):
            address = lookup(name)
            if address:
                found.append((label, address))
    return tuple(found)


class NetworkMonitor:
    """Reads the addresses on its own thread; ``addresses`` never touches the network."""

    def __init__(
        self,
        *,
        log: Callable[[str], None] = print,
        sysfs: Path = SYSFS_NET,
        lookup: Callable[[str], str | None] = ipv4_address,
        read: Callable[[], Addresses] | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._log = log
        self._read = read if read is not None else (
            lambda: addresses(sysfs=sysfs, lookup=lookup))
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._addresses: Addresses = ()
        self._failing = False
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._sleep = sleep if sleep is not None else self._closed.wait

    @property
    def addresses(self) -> Addresses:
        """``(label, address)`` per connected interface, ETH first; empty when none."""
        with self._lock:
            return self._addresses

    def _say(self, message: str) -> None:
        try:
            self._log(message)
        except Exception:
            pass  # a closed stdout must not end the poller

    def poll_once(self) -> None:
        try:
            found = tuple(self._read())
        except Exception as exc:
            if not self._failing:
                self._failing = True
                self._say(f"network: cannot read the addresses: {exc}")
            return
        self._failing = False
        before, after = dict(self.addresses), dict(found)
        for label, name in INTERFACES:
            if label in after and after[label] != before.get(label):
                self._say(f"network: {name} {after[label]}")
            elif label in before and label not in after:
                self._say(f"network: {name} gone")
        with self._lock:
            self._addresses = found

    def _run(self) -> None:
        while not self._closed.is_set():
            self.poll_once()
            self._sleep(self._interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pifilm-network", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
