"""The Nextcloud sync script pushes ~/Pictures/pifilm to one Nextcloud folder.

It runs on the Pi from a systemd timer, so it must be a quiet no-op unless the
Pi is on the home network (eth0 or wlan0), must copy (never delete) so an
SD-card cleanup cannot erase the cloud archive, and must keep the Nextcloud
password out of every argv. These tests pin those guarantees with injected
subprocess/socket seams and no network.
"""

import fcntl
import subprocess
from pathlib import Path

import pytest

from pifilm.nextcloud import (
    DEFAULT_IFACES,
    REMOTE_NAME,
    NextcloudConfig,
    config_from_env,
    home_lan_ipv4,
    main,
    nextcloud_reachable,
    obscure_password,
    rclone_copy_command,
    rclone_env,
    run_sync,
)

ENV = {
    "NEXTCLOUD_URL": "http://192.168.178.241:8083/",
    "NEXTCLOUD_USER": "george",
    "NEXTCLOUD_PASSWORD": "app-secret-123",
}

GLOBAL_IP_LINE = (
    "2: eth0    inet 192.168.178.59/24 brd 192.168.178.255 "
    "scope global dynamic noprefixroute eth0"
)
LINK_LOCAL_LINE = "2: eth0    inet 169.254.103.155/16 scope link eth0"
WLAN_IP_LINE = (
    "3: wlan0    inet 192.168.178.56/24 brd 192.168.178.255 "
    "scope global dynamic noprefixroute wlan0"
)
HOTSPOT_LINE = "3: wlan0    inet 10.42.0.1/24 brd 10.42.0.255 scope global noprefixroute wlan0"


def _cfg(**over) -> NextcloudConfig:
    base = dict(
        base_url="http://192.168.178.241:8083/",
        user="george",
        password="app-secret-123",
        target_dir="MartinParr",
        source_dir=Path("/home/george/Pictures/pifilm"),
    )
    base.update(over)
    return NextcloudConfig(**base)


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class _ConnOK:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _connect_ok(address, timeout):
    return _ConnOK()


def _router(*, ip_out="", obscure_out="OBSCURED", copy_rc=0, record=None):
    """A fake subprocess.run that dispatches on the command it is given."""

    def run(cmd, **kwargs):
        if cmd[0] == "ip":
            if isinstance(ip_out, dict):
                return _completed(stdout=ip_out.get(cmd[-1], ""))
            return _completed(stdout=ip_out)
        if cmd[1] == "obscure":
            return _completed(stdout=obscure_out + "\n")
        if cmd[1] == "copy":
            if record is not None:
                record.append((cmd, kwargs))
            return _completed(returncode=copy_rc)
        raise AssertionError(f"unexpected command {cmd!r}")

    return run


# --- config ---------------------------------------------------------------


def test_config_from_env_reads_keys_and_builds_webdav_url():
    cfg = config_from_env(ENV)
    assert cfg.base_url == "http://192.168.178.241:8083/"
    assert cfg.user == "george"
    assert cfg.target_dir == "MartinParr"  # default
    assert cfg.webdav_url() == "http://192.168.178.241:8083/remote.php/dav/files/george/"


def test_config_honours_optional_target_and_source():
    cfg = config_from_env(
        {**ENV, "NEXTCLOUD_TARGET_DIR": "Photos/Parr", "NEXTCLOUD_SOURCE_DIR": "/tmp/pics"}
    )
    assert cfg.target_dir == "Photos/Parr"
    assert cfg.source_dir == Path("/tmp/pics")


def test_config_missing_required_key_names_it():
    with pytest.raises(ValueError, match="NEXTCLOUD_PASSWORD"):
        config_from_env({"NEXTCLOUD_URL": "http://h/", "NEXTCLOUD_USER": "g"})


def test_config_rejects_non_http_url():
    with pytest.raises(ValueError, match="http"):
        config_from_env({**ENV, "NEXTCLOUD_URL": "ftp://host/dav"})


def test_host_port_parses_explicit_port():
    assert _cfg().host_port() == ("192.168.178.241", 8083)


def test_host_port_defaults_to_scheme_port():
    assert _cfg(base_url="https://cloud.example/").host_port() == ("cloud.example", 443)
    assert _cfg(base_url="http://cloud.example/").host_port() == ("cloud.example", 80)


# --- network gate --------------------------------------------------------


def test_home_lan_ipv4_returns_a_routable_address():
    assert home_lan_ipv4("eth0", run=lambda *a, **k: _completed(stdout=GLOBAL_IP_LINE)) == (
        "192.168.178.59"
    )


def test_home_lan_ipv4_ignores_link_local():
    assert home_lan_ipv4("eth0", run=lambda *a, **k: _completed(stdout=LINK_LOCAL_LINE)) is None


def test_home_lan_ipv4_none_without_ipv4():
    assert home_lan_ipv4("eth0", run=lambda *a, **k: _completed(stdout="")) is None


def test_nextcloud_reachable_true_when_connect_succeeds():
    assert nextcloud_reachable("h", 8083, connect=_connect_ok) is True


def test_nextcloud_reachable_false_on_oserror():
    def refuse(address, timeout):
        raise OSError("connection refused")

    assert nextcloud_reachable("h", 8083, connect=refuse) is False


# --- password handling ----------------------------------------------------


def test_obscure_password_feeds_stdin_never_argv():
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed(stdout="OBSCURED\n")

    assert obscure_password("app-secret-123", rclone="rclone", run=run) == "OBSCURED"
    cmd, kwargs = calls[0]
    assert cmd == ["rclone", "obscure", "-"]
    assert kwargs["input"] == "app-secret-123"
    assert "app-secret-123" not in " ".join(cmd)


def test_rclone_env_carries_config_out_of_band():
    env = rclone_env(_cfg(), "OBSCURED", base_env={})
    prefix = f"RCLONE_CONFIG_{REMOTE_NAME.upper()}_"
    assert env[prefix + "TYPE"] == "webdav"
    assert env[prefix + "VENDOR"] == "nextcloud"
    assert env[prefix + "USER"] == "george"
    assert env[prefix + "PASS"] == "OBSCURED"
    assert env[prefix + "URL"].endswith("/remote.php/dav/files/george/")


# --- rclone command -------------------------------------------------------


def test_copy_command_copies_not_syncs_with_target_and_min_age():
    cmd = rclone_copy_command(_cfg(), dry_run=False)
    assert "copy" in cmd
    assert "sync" not in cmd
    assert f"{REMOTE_NAME}:MartinParr" in cmd
    assert "/home/george/Pictures/pifilm" in cmd
    assert "--min-age" in cmd
    assert "--dry-run" not in cmd
    assert "app-secret-123" not in " ".join(cmd)


def test_copy_command_skips_lost_and_found_at_the_source_root():
    """The photo folder is an ext4 volume (docs/photo-volume.md) whose root holds a
    root-only lost+found; reading it failed every sync with exit 6."""
    cmd = rclone_copy_command(_cfg(), dry_run=False)
    assert cmd[cmd.index("--exclude") + 1] == "/lost+found/**"


def test_copy_command_dry_run_adds_flag():
    assert "--dry-run" in rclone_copy_command(_cfg(), dry_run=True)


# --- orchestration --------------------------------------------------------


def test_run_sync_is_a_noop_returning_zero_without_ethernet():
    record = []
    rc = run_sync(
        _cfg(),
        apply=True,
        run=_router(ip_out="", record=record),
        connect=_connect_ok,
        which=lambda name: "/usr/bin/rclone",
        log=lambda *a: None,
    )
    assert rc == 0
    assert record == []


def test_run_sync_skips_when_nextcloud_unreachable():
    record = []

    def refuse(address, timeout):
        raise OSError()

    rc = run_sync(
        _cfg(),
        apply=True,
        run=_router(ip_out=GLOBAL_IP_LINE, record=record),
        connect=refuse,
        which=lambda name: "/usr/bin/rclone",
        log=lambda *a: None,
    )
    assert rc == 0
    assert record == []


def test_run_sync_dry_run_when_not_applied():
    record = []
    rc = run_sync(
        _cfg(),
        apply=False,
        run=_router(ip_out=GLOBAL_IP_LINE, record=record),
        connect=_connect_ok,
        which=lambda name: "/usr/bin/rclone",
        log=lambda *a: None,
    )
    assert rc == 0
    (cmd, kwargs), = record
    assert "--dry-run" in cmd
    assert f"RCLONE_CONFIG_{REMOTE_NAME.upper()}_PASS" in kwargs["env"]


def test_run_sync_apply_runs_real_copy_and_propagates_return_code():
    record = []
    rc = run_sync(
        _cfg(),
        apply=True,
        run=_router(ip_out=GLOBAL_IP_LINE, copy_rc=7, record=record),
        connect=_connect_ok,
        which=lambda name: "/usr/bin/rclone",
        log=lambda *a: None,
    )
    assert rc == 7
    (cmd, kwargs), = record
    assert "--dry-run" not in cmd
    assert "app-secret-123" not in " ".join(cmd)
    assert "app-secret-123" not in " ".join(f"{k}={v}" for k, v in kwargs["env"].items())


def test_run_sync_errors_clearly_when_rclone_missing():
    messages = []
    rc = run_sync(
        _cfg(),
        apply=True,
        run=_router(ip_out=GLOBAL_IP_LINE),
        connect=_connect_ok,
        which=lambda name: None,
        log=messages.append,
    )
    assert rc != 0
    assert any("rclone" in m for m in messages)


# --- interface choice: eth0 then wlan0 ------------------------------------


def _sync(ip_out, *, connect=_connect_ok, ifaces=None):
    record, messages = [], []
    kwargs = {} if ifaces is None else {"ifaces": ifaces}
    rc = run_sync(
        _cfg(),
        apply=True,
        run=_router(ip_out=ip_out, record=record),
        connect=connect,
        which=lambda name: "/usr/bin/rclone",
        log=messages.append,
        **kwargs,
    )
    return rc, record, messages


def test_default_ifaces_are_eth0_then_wlan0():
    assert DEFAULT_IFACES == ("eth0", "wlan0")


def test_run_sync_over_wlan0_when_only_wlan0_has_an_address():
    rc, record, messages = _sync({"wlan0": WLAN_IP_LINE})
    assert rc == 0
    assert len(record) == 1
    assert "Nextcloud reachable over wlan0 (192.168.178.56)" in messages


def test_run_sync_prefers_eth0_when_both_have_addresses():
    rc, record, messages = _sync({"eth0": GLOBAL_IP_LINE, "wlan0": WLAN_IP_LINE})
    assert rc == 0
    assert len(record) == 1
    assert "Nextcloud reachable over eth0 (192.168.178.59)" in messages


def test_run_sync_skips_link_local_eth0_for_routable_wlan0():
    rc, record, messages = _sync({"eth0": LINK_LOCAL_LINE, "wlan0": WLAN_IP_LINE})
    assert rc == 0
    assert len(record) == 1
    assert "Nextcloud reachable over wlan0 (192.168.178.56)" in messages


def test_run_sync_in_hotspot_mode_skips_because_nextcloud_is_unreachable():
    def refuse(address, timeout):
        raise OSError("no route to host")

    rc, record, messages = _sync({"wlan0": HOTSPOT_LINE}, connect=refuse)
    assert rc == 0
    assert record == []
    assert messages == [
        "Nextcloud 192.168.178.241:8083 not reachable over wlan0 (10.42.0.1); skipping"
    ]


def test_run_sync_without_any_network_logs_the_exact_skip_line():
    rc, record, messages = _sync({})
    assert rc == 0
    assert record == []
    assert messages == ["no network on eth0 or wlan0; skipping Nextcloud sync"]


def test_run_sync_skip_line_names_a_single_interface():
    rc, record, messages = _sync({"wlan0": WLAN_IP_LINE}, ifaces=["eth0"])
    assert rc == 0
    assert record == []
    assert messages == ["no network on eth0; skipping Nextcloud sync"]


# --- command line ---------------------------------------------------------


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    # The default lock lives under the state directory; keep it in the temp dir.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    path = tmp_path / "test.env"
    path.write_text("".join(f"{k}={v}\n" for k, v in ENV.items()))
    return path


def _capture_main(monkeypatch, argv):
    seen = {}

    def fake_run_sync(cfg, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("pifilm.nextcloud.run_sync", fake_run_sync)
    assert main(argv) == 0
    return seen


def test_main_defaults_to_eth0_then_wlan0(monkeypatch, env_file):
    seen = _capture_main(monkeypatch, ["--env", str(env_file)])
    assert list(seen["ifaces"]) == ["eth0", "wlan0"]


def test_main_iface_replaces_the_default(monkeypatch, env_file):
    seen = _capture_main(monkeypatch, ["--env", str(env_file), "--iface", "wlan0"])
    assert list(seen["ifaces"]) == ["wlan0"]


def test_main_iface_is_repeatable_in_order(monkeypatch, env_file):
    seen = _capture_main(
        monkeypatch, ["--env", str(env_file), "--iface", "wlan0", "--iface", "eth0"]
    )
    assert list(seen["ifaces"]) == ["wlan0", "eth0"]


def test_wrapper_script_runs_the_package_main():
    import pifilm.nextcloud
    import scripts.nextcloud_sync

    assert scripts.nextcloud_sync.main is pifilm.nextcloud.main


# --- one sync at a time ---------------------------------------------------


def _recording_run_sync(monkeypatch):
    calls = []

    def fake_run_sync(cfg, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("pifilm.nextcloud.run_sync", fake_run_sync)
    return calls


def test_main_skips_when_another_sync_holds_the_lock(monkeypatch, env_file, tmp_path, capsys):
    calls = _recording_run_sync(monkeypatch)
    lock = tmp_path / "l"
    with open(lock, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        rc = main(["--env", str(env_file), "--lock", str(lock)])
    assert rc == 0
    assert calls == []
    assert "another Nextcloud sync is running; skipping" in capsys.readouterr().out


def test_main_runs_the_sync_and_frees_the_lock(monkeypatch, env_file, tmp_path):
    calls = _recording_run_sync(monkeypatch)
    lock = tmp_path / "l"
    assert main(["--env", str(env_file), "--lock", str(lock)]) == 0
    assert len(calls) == 1
    with open(lock, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_main_creates_the_lock_directories(monkeypatch, env_file, tmp_path):
    calls = _recording_run_sync(monkeypatch)
    lock = tmp_path / "a" / "b" / "nextcloud-sync.lock"
    assert main(["--env", str(env_file), "--lock", str(lock)]) == 0
    assert len(calls) == 1
    assert lock.exists()


def test_default_lock_is_under_the_state_directory(monkeypatch, tmp_path):
    from pifilm.nextcloud import _default_lock

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert _default_lock() == tmp_path / "pifilm" / "nextcloud-sync.lock"
    monkeypatch.delenv("XDG_STATE_HOME")
    expected = Path.home() / ".local" / "state" / "pifilm" / "nextcloud-sync.lock"
    assert _default_lock() == expected
