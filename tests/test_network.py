"""The network addresses behind the viewfinder's ETH / WLAN tag."""

import ctypes
import socket
import sys
import threading

from pifilm.capture import network
from pifilm.capture.network import (
    DEFAULT_INTERVAL_S,
    INTERFACES,
    NetworkMonitor,
    addresses,
    ipv4_address,
    ipv4_addresses,
    is_up,
    pick,
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


def test_pick_prefers_the_routable_address_over_the_link_local_one():
    assert pick(["169.254.67.20", "192.168.178.59"]) == "192.168.178.59"


def test_pick_falls_back_to_the_link_local_address_when_there_is_no_other():
    assert pick(["169.254.67.20"]) == "169.254.67.20"


def test_pick_of_nothing_is_none():
    assert pick([]) is None


def test_pick_skips_loopback_and_unspecified():
    assert pick(["127.0.0.1", "10.42.0.1"]) == "10.42.0.1"
    assert pick(["0.0.0.0", "10.42.0.1"]) == "10.42.0.1"
    assert pick(["127.0.0.1"]) is None


def test_pick_skips_an_invalid_string():
    assert pick(["not an address", "192.168.1.5"]) == "192.168.1.5"
    assert pick(["999.1.1.1"]) is None


def test_pick_keeps_the_order_among_routable_addresses():
    assert pick(["10.0.0.2", "192.168.1.5"]) == "10.0.0.2"


def test_an_address_lookup_picks_from_the_interface_s_addresses():
    asked = []

    def list_addresses(name):
        asked.append(name)
        return ["169.254.67.20", "192.168.178.59"]

    assert ipv4_address("eth0", list_addresses=list_addresses) == "192.168.178.59"
    assert asked == ["eth0"]
    assert ipv4_address("eth0", list_addresses=lambda name: []) is None


def test_the_addresses_are_none_on_a_machine_that_is_not_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert ipv4_addresses("eth0") == []
    assert ipv4_address("eth0") is None


def test_the_addresses_read_on_this_machine_without_raising():
    found = ipv4_addresses("lo")
    assert isinstance(found, list)
    assert all(isinstance(address, str) for address in found)
    if sys.platform.startswith("linux"):
        assert "127.0.0.1" in found


def test_the_addresses_are_empty_when_getifaddrs_fails(monkeypatch):
    class FailingLibc:
        def getifaddrs(self, head):
            return -1

        def freeifaddrs(self, head):
            raise AssertionError("nothing to free")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(network, "_libc", lambda: FailingLibc())
    assert ipv4_addresses("eth0") == []


def test_the_addresses_are_empty_without_a_libc(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(network, "_libc", lambda: None)
    assert ipv4_addresses("eth0") == []


def _sockaddr_in(address):
    sin = network._SockaddrIn()
    sin.sin_family = network.AF_INET
    sin.sin_addr = (ctypes.c_ubyte * 4)(*socket.inet_aton(address))
    return sin


def _sockaddr_in6():
    raw = (ctypes.c_ubyte * 28)()
    family = network._Sockaddr.from_buffer(raw)
    family.sa_family = socket.AF_INET6
    return raw


def test_the_addresses_walk_the_getifaddrs_list_for_that_interface(monkeypatch):
    """A fake libc hands back a list laid out as glibc's struct ifaddrs."""
    keep = []
    entries = [
        (b"lo", _sockaddr_in("127.0.0.1")),
        (b"eth0", _sockaddr_in("169.254.67.20")),
        (b"eth0", _sockaddr_in6()),
        (b"eth0", None),  # an entry with no address (an AF_PACKET-less link)
        (b"wlan0", _sockaddr_in("10.42.0.1")),
        (b"eth0", _sockaddr_in("192.168.178.59")),
    ]
    nodes = [network._Ifaddrs() for _ in entries]
    for node, following, (name, addr) in zip(nodes, nodes[1:] + [None], entries,
                                              strict=True):
        buffer = ctypes.create_string_buffer(name)
        keep.append(buffer)
        node.ifa_name = ctypes.cast(buffer, ctypes.c_char_p)
        if addr is not None:
            node.ifa_addr = ctypes.cast(ctypes.pointer(addr), ctypes.POINTER(network._Sockaddr))
        if following is not None:
            node.ifa_next = ctypes.pointer(following)
    freed = []

    class FakeLibc:
        def getifaddrs(self, head):
            head[0] = ctypes.pointer(nodes[0])
            return 0

        def freeifaddrs(self, head):
            freed.append(ctypes.addressof(head.contents))

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(network, "_libc", lambda: FakeLibc())
    assert ipv4_addresses("eth0") == ["169.254.67.20", "192.168.178.59"]
    assert ipv4_addresses("wlan0") == ["10.42.0.1"]
    assert ipv4_addresses("usb0") == []
    assert freed == [ctypes.addressof(nodes[0])] * 3


def test_the_glibc_layout_of_struct_ifaddrs():
    """Pointers, then an unsigned int padded to a pointer boundary (LP64 aarch64)."""
    pointer = ctypes.sizeof(ctypes.c_void_p)
    fields = network._Ifaddrs
    assert fields.ifa_next.offset == 0
    assert fields.ifa_name.offset == pointer
    assert fields.ifa_flags.offset == 2 * pointer
    assert fields.ifa_addr.offset == 3 * pointer
    assert fields.ifa_data.offset == 6 * pointer
    assert ctypes.sizeof(fields) == 7 * pointer
    assert ctypes.sizeof(network._SockaddrIn) == 16
    assert network._SockaddrIn.sin_addr.offset == 4


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
