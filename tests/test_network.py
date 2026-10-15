"""The network addresses behind the viewfinder's ETH / WLAN tag."""

import socket
import threading

from pifilm.capture.network import (
    DEFAULT_INTERVAL_S,
    INTERFACES,
    NetworkMonitor,
    addresses,
    ipv4_address,
    is_up,
)


def _sysfs(tmp_path, **states):
    """A fake /sys/class/net with an operstate file per interface given."""
    for name, state in states.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "operstate").write_text(state + "\n")
    return tmp_path


def test_the_interfaces_are_eth0_then_wlan0():
    assert INTERFACES == (("ETH", "eth0"), ("WLAN", "wlan0"))


def test_the_default_interval_is_10_s():
    assert DEFAULT_INTERVAL_S == 10.0


def test_up_is_operstate_up(tmp_path):
    sysfs = _sysfs(tmp_path, eth0="up", wlan0="down", usb0="dormant")
    assert is_up("eth0", sysfs=sysfs)
    assert not is_up("wlan0", sysfs=sysfs)
    assert not is_up("usb0", sysfs=sysfs)


def test_a_missing_interface_or_operstate_is_not_up(tmp_path):
    (tmp_path / "eth0").mkdir()  # no operstate file
    assert not is_up("eth0", sysfs=tmp_path)
    assert not is_up("wlan0", sysfs=tmp_path)
    assert not is_up("eth0", sysfs=tmp_path / "absent")


def test_addresses_are_eth_first_for_interfaces_that_are_up_with_an_address(tmp_path):
    sysfs = _sysfs(tmp_path, eth0="up", wlan0="up")
    table = {"eth0": "192.168.1.20", "wlan0": "10.42.0.1"}
    assert addresses(sysfs=sysfs, lookup=table.get) == (
        ("ETH", "192.168.1.20"), ("WLAN", "10.42.0.1"))


def test_an_interface_that_is_down_is_left_out_even_with_an_address(tmp_path):
    sysfs = _sysfs(tmp_path, eth0="down", wlan0="up")
    asked = []

    def lookup(name):
        asked.append(name)
        return {"eth0": "192.168.1.20", "wlan0": "192.168.178.56"}[name]

    assert addresses(sysfs=sysfs, lookup=lookup) == (("WLAN", "192.168.178.56"),)
    assert asked == ["wlan0"]  # a down interface is not asked for its address


def test_an_interface_that_is_up_without_an_address_is_left_out(tmp_path):
    sysfs = _sysfs(tmp_path, eth0="up", wlan0="up")
    assert addresses(sysfs=sysfs, lookup={"wlan0": "10.42.0.1"}.get) == (("WLAN", "10.42.0.1"),)


def test_nothing_connected_is_an_empty_tuple(tmp_path):
    assert addresses(sysfs=_sysfs(tmp_path), lookup=lambda name: "1.2.3.4") == ()


def test_an_address_lookup_for_an_interface_that_does_not_exist_is_none():
    assert ipv4_address("pifilm-no-such0") is None


def test_an_address_lookup_never_raises_for_an_ioctl_failure(monkeypatch):
    import fcntl

    def failing(*args):
        raise OSError(99, "Cannot assign requested address")

    monkeypatch.setattr(fcntl, "ioctl", failing)
    assert ipv4_address("eth0") is None


def test_an_address_lookup_reads_the_address_from_the_ioctl_reply(monkeypatch):
    import fcntl

    calls = []

    def ioctl(fd, request, arg):
        calls.append((request, arg[:16]))
        # struct ifreq: 16 bytes of name, then a sockaddr_in (family, port, address).
        return arg[:16] + b"\x02\x00\x00\x00" + socket.inet_aton("10.42.0.1") + bytes(236)

    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    assert ipv4_address("wlan0") == "10.42.0.1"
    assert calls == [(0x8915, b"wlan0" + bytes(11))]


class FakeNet:
    """Stands in for sysfs and the address lookup: {interface: address or None}."""

    def __init__(self, **table):
        self.table = table

    def addresses(self):
        return tuple((label, self.table[name]) for label, name in INTERFACES
                     if self.table.get(name))


def _monitor(net, lines, **kwargs):
    return NetworkMonitor(log=lines.append, read=net.addresses, **kwargs)


def test_nothing_read_yet_is_empty():
    assert _monitor(FakeNet(), []).addresses == ()


def test_an_address_that_appears_is_logged_once():
    net = FakeNet(wlan0="192.168.178.56")
    lines = []
    monitor = _monitor(net, lines)
    monitor.poll_once()
    assert monitor.addresses == (("WLAN", "192.168.178.56"),)
    assert lines == ["network: wlan0 192.168.178.56"]
    monitor.poll_once()
    monitor.poll_once()
    assert len(lines) == 1


def test_a_changed_address_is_logged_with_the_new_one():
    net = FakeNet(eth0="192.168.1.20")
    lines = []
    monitor = _monitor(net, lines)
    monitor.poll_once()
    net.table["eth0"] = "192.168.1.31"
    monitor.poll_once()
    assert monitor.addresses == (("ETH", "192.168.1.31"),)
    assert lines == ["network: eth0 192.168.1.20", "network: eth0 192.168.1.31"]


def test_an_address_that_disappears_is_logged_as_gone():
    net = FakeNet(eth0="192.168.1.20", wlan0="10.42.0.1")
    lines = []
    monitor = _monitor(net, lines)
    monitor.poll_once()
    assert monitor.addresses == (("ETH", "192.168.1.20"), ("WLAN", "10.42.0.1"))
    assert lines == ["network: eth0 192.168.1.20", "network: wlan0 10.42.0.1"]
    net.table["eth0"] = None
    monitor.poll_once()
    assert monitor.addresses == (("WLAN", "10.42.0.1"),)
    assert lines[-1] == "network: eth0 gone"
    monitor.poll_once()
    assert len(lines) == 3


def test_nothing_connected_logs_nothing():
    lines = []
    monitor = _monitor(FakeNet(), lines)
    monitor.poll_once()
    assert monitor.addresses == () and lines == []


def test_a_failing_read_logs_once_per_spell_and_keeps_the_last_addresses():
    net = FakeNet(wlan0="10.42.0.1")
    state = {"fail": False}

    def read():
        if state["fail"]:
            raise RuntimeError("bug")
        return net.addresses()

    lines = []
    monitor = NetworkMonitor(log=lines.append, read=read)
    monitor.poll_once()
    state["fail"] = True
    monitor.poll_once()  # does not raise
    monitor.poll_once()
    assert monitor.addresses == (("WLAN", "10.42.0.1"),)
    cannot = [line for line in lines if "cannot" in line]
    assert cannot == ["network: cannot read the addresses: bug"]
    # A read that works again ends the spell; the next failure is worth another line.
    state["fail"] = False
    monitor.poll_once()
    state["fail"] = True
    monitor.poll_once()
    assert len([line for line in lines if "cannot" in line]) == 2


def test_a_log_that_raises_does_not_stop_the_poll():
    def log(line):
        raise OSError("stdout closed")

    monitor = NetworkMonitor(log=log, read=FakeNet(eth0="192.168.1.20").addresses)
    monitor.poll_once()
    assert monitor.addresses == (("ETH", "192.168.1.20"),)


def test_the_default_read_uses_the_given_sysfs_and_lookup(tmp_path):
    sysfs = _sysfs(tmp_path, eth0="up", wlan0="down")
    monitor = NetworkMonitor(log=lambda line: None, sysfs=sysfs,
                             lookup={"eth0": "192.168.1.20"}.get)
    monitor.poll_once()
    assert monitor.addresses == (("ETH", "192.168.1.20"),)


def test_the_poller_checks_at_once_on_its_own_thread_and_stops_cleanly():
    polled = threading.Event()
    lines = []

    def log(line):
        lines.append(line)
        polled.set()

    waits = []
    monitor = NetworkMonitor(log=log, read=FakeNet(wlan0="10.42.0.1").addresses,
                             interval_s=3600.0)
    original_wait = monitor._closed.wait

    def wait(seconds):
        waits.append((threading.current_thread().name, seconds))
        return original_wait(seconds)

    monitor._sleep = wait
    monitor.start()
    try:
        # The first check does not wait for the first interval.
        assert polled.wait(2.0)
        assert monitor.addresses == (("WLAN", "10.42.0.1"),)
    finally:
        monitor.close()
    assert monitor._thread is not None and not monitor._thread.is_alive()
    assert waits and waits[0] == ("pifilm-network", 3600.0)
    assert monitor._thread.daemon


def test_the_poller_polls_again_after_each_interval():
    three = threading.Event()
    count = [0]
    monitor = NetworkMonitor(log=lambda line: None, read=FakeNet().addresses,
                             interval_s=0.0, sleep=lambda s: None)
    real_poll = monitor.poll_once

    def poll():
        real_poll()
        count[0] += 1
        if count[0] >= 3:
            three.set()

    monitor.poll_once = poll
    monitor.start()
    try:
        assert three.wait(2.0)
    finally:
        monitor.close()


def test_close_without_start_and_twice_is_harmless():
    monitor = NetworkMonitor(read=FakeNet().addresses)
    monitor.close()
    monitor.close()


def test_the_real_machine_can_be_read_without_raising():
    result = addresses()
    assert isinstance(result, tuple)
    for label, address in result:
        assert label in ("ETH", "WLAN")
        socket.inet_aton(address)
