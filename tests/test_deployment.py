import pytest

from scripts.deploy_remote import (
    CAMERA_OWNERS_COMMAND,
    CAPTURE_PROCESS_COMMAND,
    DeploymentConfig,
    build_capture_command,
    deployment_config,
    inspection_commands,
    open_ssh_client,
    restart_headless_service,
)

REQUIRED_ENV = {
    "PI_HOST": "parr.local",
    "PI_USER": "george",
    "PI_PROJECT_DIR": "/home/george/repos/pi-film-reversal",
    "PI_SSH_PASSWORD": "ssh-secret",
    "PIFILM_REMOTE_TOKEN": "remote-secret",
    "PI_ARTIFACT_DIR": "/home/george/repos/pi-film-reversal/artifacts/personal-v1",
}


def test_listen_address_defaults_to_all_interfaces_when_env_omits_it():
    # The SSH host (Ethernet, mDNS name) and the API bind address (hotspot)
    # are different things in the direct-hotspot topology, so the listen
    # address must not be derived from PI_HOST.
    config = deployment_config(REQUIRED_ENV)

    assert config.listen == "0.0.0.0:8765"


def test_listen_address_is_read_from_env_when_present():
    config = deployment_config({**REQUIRED_ENV, "PIFILM_LISTEN": "10.42.0.1:8765"})

    assert config.listen == "10.42.0.1:8765"


class FakeClient:
    def __init__(self):
        self.calls = []

    def load_system_host_keys(self):
        self.calls.append(("load_system_host_keys",))

    def set_missing_host_key_policy(self, policy):
        self.calls.append(("set_missing_host_key_policy", policy))

    def connect(self, **kwargs):
        self.calls.append(("connect", kwargs))


class FakeParamiko:
    class RejectPolicy:
        pass

    class SSHClient(FakeClient):
        pass


class FakeStream:
    def __init__(self, status=0, output=""):
        self.channel = self
        self.status = status
        self.output = output.encode()

    def recv_exit_status(self):
        return self.status

    def read(self):
        return self.output


class RestartClient:
    """Answers each command from ``replies`` (by exact command), else empty."""

    def __init__(self, replies=None):
        self.commands = []
        self.replies = replies or {}

    def exec_command(self, command):
        self.commands.append(command)
        return None, FakeStream(output=self.replies.get(command, "")), FakeStream()


def test_capture_command_uses_verified_artifact_directory_and_never_includes_token():
    config = DeploymentConfig(
        host="parr.local",
        user="george",
        project_dir="/home/george/repos/pi-film-reversal",
        ssh_password="ssh-secret",
        artifact_dir="/home/george/pifilm artifacts/personal-v1",
        remote_token="remote-secret",
        listen="0.0.0.0:8765",
    )

    command = build_capture_command(config, show_captures=True)

    assert command == (
        "/home/george/repos/pi-film-reversal/.venv/bin/pifilm-capture --no-preview "
        "--show-captures --remote-listen 0.0.0.0:8765 --artifacts "
        "'/home/george/pifilm artifacts/personal-v1'"
    )
    assert "parr.local" not in command
    assert "remote-secret" not in command
    assert "ssh-secret" not in command


def test_open_ssh_client_rejects_unknown_host_keys_before_password_connection():
    config = DeploymentConfig(
        host="192.168.178.56",
        user="george",
        project_dir="/home/george/repos/pi-film-reversal",
        ssh_password="ssh-secret",
        artifact_dir="/home/george/artifacts/personal-v1",
        remote_token="remote-secret",
        listen="0.0.0.0:8765",
    )

    client = open_ssh_client(config, paramiko_module=FakeParamiko)

    assert client.calls[0] == ("load_system_host_keys",)
    assert isinstance(client.calls[1][1], FakeParamiko.RejectPolicy)
    assert client.calls[2] == (
        "connect",
        {
            "hostname": "192.168.178.56",
            "username": "george",
            "password": "ssh-secret",
            "look_for_keys": False,
            "allow_agent": False,
            "timeout": 15,
        },
    )


# What the Pi 4 with a desktop session really reports (2026-09-24): PipeWire and
# WirePlumber keep the video nodes open to monitor them, beside the service.
DESKTOP_OWNERS = "   1468 pipewire\n   1474 wireplumber\n   1830 pifilm-capture\n"
MONITORS_ONLY = "   1468 pipewire\n   1474 wireplumber\n"


def test_restart_stops_the_current_service_before_rechecking_its_camera_owner():
    client = RestartClient()

    restart_headless_service(
        client,
        {
            "camera owners": "   1234 pifilm-capture",
            "capture process": "1234 /home/george/.venv/bin/python pifilm-capture",
            "service pid": "1234",
        },
    )

    assert client.commands == [
        "sudo -n systemctl stop pifilm-capture.service",
        CAPTURE_PROCESS_COMMAND,
        CAMERA_OWNERS_COMMAND,
        "sudo -n systemctl start pifilm-capture.service",
    ]


def test_restart_allows_the_desktop_media_monitors_holding_the_video_nodes():
    """PipeWire/WirePlumber hold every /dev/video* open without capturing; the
    service runs beside them, so they must not block a restart before or after
    the stop (they refused every restart on the Pi 4 with a desktop)."""
    client = RestartClient({CAMERA_OWNERS_COMMAND: MONITORS_ONLY})

    restart_headless_service(
        client,
        {
            "camera owners": DESKTOP_OWNERS,
            "capture process": "1830 /home/george/.venv/bin/python pifilm-capture",
            "service pid": "1830",
        },
    )

    assert client.commands[-1] == "sudo -n systemctl start pifilm-capture.service"


def test_restart_refuses_when_the_capture_process_outlives_the_stop():
    client = RestartClient({CAPTURE_PROCESS_COMMAND: "1830 pifilm-capture"})

    with pytest.raises(RuntimeError, match="capture process remains"):
        restart_headless_service(
            client,
            {"camera owners": "   1830 pifilm-capture",
             "capture process": "1830 pifilm-capture", "service pid": "1830"},
        )
    assert "sudo -n systemctl start pifilm-capture.service" not in client.commands


def test_restart_refuses_a_real_camera_owner_left_after_the_stop():
    client = RestartClient({CAMERA_OWNERS_COMMAND: MONITORS_ONLY + "   2001 rpicam-still\n"})

    with pytest.raises(RuntimeError, match="still owns a camera"):
        restart_headless_service(
            client,
            {"camera owners": DESKTOP_OWNERS,
             "capture process": "1830 pifilm-capture", "service pid": "1830"},
        )
    assert "sudo -n systemctl start pifilm-capture.service" not in client.commands


def test_the_capture_process_is_looked_up_by_its_current_name():
    """The package was renamed parr -> pifilm; searching for ``parr-capture`` found
    nothing, so a stray capture process could never block a restart."""
    assert "[p]ifilm-capture" in CAPTURE_PROCESS_COMMAND   # bracket: never matches itself
    assert "arr-capture" not in CAPTURE_PROCESS_COMMAND


def test_camera_owners_are_named_because_fuser_prints_names_only_to_stderr():
    assert "ps -o pid=,comm=" in CAMERA_OWNERS_COMMAND
    assert "fuser -v" not in CAMERA_OWNERS_COMMAND


def test_presence_checks_print_a_word_so_success_is_not_shown_as_none(tmp_path):
    config = deployment_config(REQUIRED_ENV)
    commands = dict(inspection_commands(config))
    for label in ("project", "artifact"):
        assert "echo present" in commands[label] and "echo missing" in commands[label]


def test_restart_refuses_a_foreign_camera_owner_before_stopping_the_service():
    client = RestartClient()

    with pytest.raises(RuntimeError, match="foreign capture or camera owner"):
        restart_headless_service(
            client,
            {
                "camera owners": "   4321 other-camera-app",
                "capture process": "4321 other-camera-app",
                "service pid": "1234",
            },
        )

    assert client.commands == []
