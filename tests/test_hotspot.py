"""The Pi hotspot script turns the local .env into a NetworkManager keyfile.

A phone or laptop joins the Pi's own access point in the field to SSH in, so
WIFI_SSID and WIFI_PASSWORD in .env are the AP credentials, and the AP address
is PI_HOTSPOT_ADDRESS (10.42.0.1 when it is left out).
"""

import stat

import pytest

from scripts.pi_hotspot import apply_keyfile, hotspot_keyfile, redact_psk

ENV = {
    "WIFI_SSID": "pifilm-cam",
    "WIFI_PASSWORD": "correct-horse-battery",
}


def _section(text: str, name: str) -> dict[str, str]:
    lines = text.splitlines()
    start = lines.index(f"[{name}]") + 1
    body = {}
    for line in lines[start:]:
        if line.startswith("["):
            break
        if "=" in line:
            key, value = line.split("=", 1)
            body[key] = value
    return body


def test_keyfile_is_an_ap_on_wlan0_with_wpa2_psk_and_shared_ipv4_on_the_default_address():
    text = hotspot_keyfile(ENV)

    assert _section(text, "connection")["id"] == "pifilm-ap"
    assert _section(text, "connection")["interface-name"] == "wlan0"
    assert _section(text, "connection")["autoconnect"] == "true"
    assert _section(text, "wifi")["mode"] == "ap"
    assert _section(text, "wifi")["ssid"] == "pifilm-cam"
    assert _section(text, "wifi")["band"] == "bg"
    assert _section(text, "wifi")["powersave"] == "2"
    assert _section(text, "wifi-security")["key-mgmt"] == "wpa-psk"
    assert _section(text, "wifi-security")["proto"] == "rsn"
    assert _section(text, "wifi-security")["psk"] == "correct-horse-battery"
    assert _section(text, "ipv4")["method"] == "shared"
    assert _section(text, "ipv4")["address1"] == "10.42.0.1/24"


def test_keyfile_disables_wifi_power_save_so_the_ap_stays_reliable():
    # brcmfmac enables power management on wlan0 (seen when eth0's carrier drops),
    # which makes the access point drop clients: a client associates, gets a DHCP
    # lease, then keeps re-associating every few minutes and its connections stall.
    # powersave=2 disables NM power save on the AP so the hotspot is steady with
    # or without Ethernet. Measured on a Pi 4 running the migrated SD, 2026-09-19.
    assert _section(hotspot_keyfile(ENV), "wifi")["powersave"] == "2"


def test_keyfile_disables_protected_management_frames_for_the_pi_3b_radio():
    # NetworkManager's default is PMF "optional", which makes hostapd install an
    # AES-CMAC group key at start-up. The Pi 3B / Zero W BCM43430 firmware has
    # no management-frame protection, the driver does not advertise AES-CMAC,
    # and the kernel rejects the key: "key setting validation failed" and the
    # access point never comes up. Measured on Trixie, NM 1.52, fw 7.45.98.
    text = hotspot_keyfile(ENV)

    assert _section(text, "wifi-security")["pmf"] == "1"


def test_keyfile_uuid_is_stable_so_reapplying_replaces_rather_than_duplicates():
    first = _section(hotspot_keyfile(ENV), "connection")["uuid"]
    second = _section(hotspot_keyfile(ENV), "connection")["uuid"]

    assert first == second
    assert len(first) == 36


@pytest.mark.parametrize(
    "password",
    ["short", "x" * 64, "has\nnewline"],
    ids=["under-8", "over-63", "control-char"],
)
def test_rejects_a_passphrase_wpa2_cannot_accept(password):
    with pytest.raises(ValueError, match="WIFI_PASSWORD"):
        hotspot_keyfile({**ENV, "WIFI_PASSWORD": password})


def test_rejects_an_ssid_longer_than_32_bytes():
    with pytest.raises(ValueError, match="WIFI_SSID"):
        hotspot_keyfile({**ENV, "WIFI_SSID": "s" * 33})


def test_hotspot_address_is_read_from_env_when_present():
    text = hotspot_keyfile({**ENV, "PI_HOTSPOT_ADDRESS": "192.168.50.1"})

    assert _section(text, "ipv4")["address1"] == "192.168.50.1/24"


def test_an_empty_hotspot_address_means_the_default():
    text = hotspot_keyfile({**ENV, "PI_HOTSPOT_ADDRESS": ""})

    assert _section(text, "ipv4")["address1"] == "10.42.0.1/24"


@pytest.mark.parametrize(
    "address",
    ["parr.local", "fe80::1", "http://10.42.0.1:8765", "10.42.0.1/24", "10.42.0.256"],
    ids=["hostname", "ipv6", "url", "prefix", "out-of-range"],
)
def test_rejects_a_hotspot_address_that_is_not_a_plain_ipv4_address(address):
    # The hotspot must own exactly this address, so only a plain IPv4 one will do.
    with pytest.raises(ValueError, match="PI_HOTSPOT_ADDRESS"):
        hotspot_keyfile({**ENV, "PI_HOTSPOT_ADDRESS": address})


def test_redacted_keyfile_keeps_everything_except_the_psk():
    text = hotspot_keyfile(ENV)

    shown = redact_psk(text)

    assert "correct-horse-battery" not in shown
    assert _section(shown, "wifi-security")["psk"] == "<redacted>"
    assert _section(shown, "wifi")["ssid"] == "pifilm-cam"


def test_apply_writes_a_root_only_keyfile_then_reloads_and_activates(tmp_path):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)

    target = tmp_path / "pifilm-ap.nmconnection"
    apply_keyfile("keyfile-body\n", target, run=run)

    assert target.read_text() == "keyfile-body\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert calls == [
        ["nmcli", "connection", "reload"],
        ["nmcli", "connection", "up", "pifilm-ap"],
    ]
