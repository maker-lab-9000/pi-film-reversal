"""The Nextcloud sync script pushes ~/Pictures/pifilm to one Nextcloud folder.

It runs on the Pi from a systemd timer, so it must be a quiet no-op unless the
wired LAN is up, must copy (never delete) so an SD-card cleanup cannot erase the
cloud archive, and must keep the Nextcloud password out of every argv. These
tests pin those guarantees with injected subprocess/socket seams and no network.
"""

import subprocess
from pathlib import Path

import pytest

from scripts.nextcloud_sync import (
    REMOTE_NAME,
    NextcloudConfig,
    config_from_env,
    ethernet_ready,
    home_lan_ipv4,
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


# --- ethernet gate --------------------------------------------------------


def test_home_lan_ipv4_returns_a_routable_address():
    assert home_lan_ipv4("eth0", run=lambda *a, **k: _completed(stdout=GLOBAL_IP_LINE)) == (
        "192.168.178.59"
    )


def test_home_lan_ipv4_ignores_link_local():
    assert home_lan_ipv4("eth0", run=lambda *a, **k: _completed(stdout=LINK_LOCAL_LINE)) is None


def test_ethernet_ready_false_without_ipv4():
    assert ethernet_ready("eth0", run=lambda *a, **k: _completed(stdout="")) is False


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
