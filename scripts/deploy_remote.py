"""Inspect and optionally restart the Pi capture service without exposing secrets.

The restart is guarded: it refuses while anything but the service itself holds
the camera, so two processes never compete for it. Two things made that guard
refuse every restart on the Pi 4 (2026-09-24) and are handled here:

- ``fuser -v`` prints its PID/COMMAND table on stderr, and stdout carries bare
  PIDs, so the owners were never named. ``CAMERA_OWNERS_COMMAND`` names them with
  ``ps``.
- A desktop session's PipeWire and WirePlumber keep every ``/dev/video*`` node
  open to monitor it without capturing; the service runs beside them. They are
  allowed by name (``PASSIVE_DEVICE_MONITORS``). If one ever did take the camera,
  the service's own start would fail and say so in its journal.

The capture process is looked up as ``pifilm-capture``; the pre-rename
``parr-capture`` search matched nothing, so it never guarded anything.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firmware.sticks3.scripts.generate_config import parse_env_file

DEFAULT_LISTEN = "0.0.0.0:8765"
SERVICE = "pifilm-capture.service"
CAPTURE_PROCESS_COMMAND = "pgrep -af '[p]ifilm-capture' || true"
# PID and name of every process with a video node open, one per line.
CAMERA_OWNERS_COMMAND = (
    "pids=$(fuser /dev/video* 2>/dev/null | tr -cs '0-9' ',' | sed 's/^,//; s/,$//'); "
    'if [ -n "$pids" ]; then ps -o pid=,comm= -p "$pids"; fi; true'
)
PASSIVE_DEVICE_MONITORS = frozenset({"pipewire", "wireplumber"})


@dataclass(frozen=True)
class DeploymentConfig:
    host: str
    user: str
    project_dir: str
    ssh_password: str
    artifact_dir: str
    remote_token: str
    listen: str


def _required(values: Mapping[str, str], key: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ValueError(f"missing required {key} in local env file")
    return value


def deployment_config(values: Mapping[str, str]) -> DeploymentConfig:
    # PI_HOST is where SSH goes (Ethernet, usually an mDNS name). The API bind
    # address is a separate setting because, with the Pi running its own
    # hotspot, the Stick reaches a different interface than the admin does.
    return DeploymentConfig(
        host=_required(values, "PI_HOST"),
        user=_required(values, "PI_USER"),
        project_dir=_required(values, "PI_PROJECT_DIR"),
        ssh_password=_required(values, "PI_SSH_PASSWORD"),
        artifact_dir=_required(values, "PI_ARTIFACT_DIR"),
        remote_token=_required(values, "PIFILM_REMOTE_TOKEN"),
        listen=values.get("PIFILM_LISTEN") or DEFAULT_LISTEN,
    )


def build_capture_command(config: DeploymentConfig, *, show_captures: bool) -> str:
    """Build the command only after its artifact directory has been inspected."""
    parts = [
        f"{config.project_dir}/.venv/bin/pifilm-capture",
        "--no-preview",
    ]
    if show_captures:
        parts.append("--show-captures")
    parts.extend(
        [
            "--remote-listen",
            config.listen,
            "--artifacts",
            config.artifact_dir,
        ]
    )
    return shlex.join(parts)


def open_ssh_client(config: DeploymentConfig, *, paramiko_module: Any | None = None) -> Any:
    """Open password SSH only when the host is already trusted locally."""
    if paramiko_module is None:
        try:
            import paramiko as paramiko_module
        except ImportError as error:
            message = "install deployment support with: pip install -e '.[deploy]'"
            raise RuntimeError(message) from error
    client = paramiko_module.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko_module.RejectPolicy())
    client.connect(
        hostname=config.host,
        username=config.user,
        password=config.ssh_password,
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
    )
    return client


def inspection_commands(config: DeploymentConfig) -> list[tuple[str, str]]:
    project = shlex.quote(config.project_dir)
    artifact = shlex.quote(config.artifact_dir)
    return [
        ("project", f"if test -d {project}; then echo present; else echo missing; fi"),
        (
            "artifact",
            f"if test -d {artifact} && test -f {artifact}/params.json "
            f"&& test -f {artifact}/pifilm.cube; then echo present; else echo missing; fi",
        ),
        (
            "service pid",
            "systemctl show -p MainPID --value pifilm-capture.service 2>/dev/null || true",
        ),
        ("capture process", CAPTURE_PROCESS_COMMAND),
        ("camera owners", CAMERA_OWNERS_COMMAND),
        ("desktop session", "loginctl list-sessions --no-legend 2>/dev/null || true"),
    ]


def _run(client: Any, command: str) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command)
    return stdout.channel.recv_exit_status(), stdout.read().decode(), stderr.read().decode()


def inspect_target(client: Any, config: DeploymentConfig) -> dict[str, str]:
    results: dict[str, str] = {}
    for label, command in inspection_commands(config):
        status, output, error = _run(client, command)
        if status:
            detail = error.strip() or "command failed"
            raise RuntimeError(f"Pi inspection failed for {label}: {detail}")
        results[label] = output.strip()
    return results


def _process_ids(output: str) -> set[str]:
    return {
        match.group(1)
        for line in output.splitlines()
        if (match := re.match(r"\s*(\d+)\s+", line))
    }


def _camera_owners(output: str) -> dict[str, str] | None:
    """``{pid: command}`` from ``CAMERA_OWNERS_COMMAND``; None if unparseable."""
    owners: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        match = re.match(r"\s*(\d+)\s+(\S+)\s*$", line)
        if match is None:
            return None
        owners[match.group(1)] = match.group(2)
    return owners


def _foreign_camera_owners(output: str, allowed: set[str]) -> bool:
    owners = _camera_owners(output)
    if owners is None:
        return True
    return any(
        pid not in allowed and name not in PASSIVE_DEVICE_MONITORS
        for pid, name in owners.items()
    )


def _has_foreign_initial_owner(inspection: Mapping[str, str]) -> bool:
    service_pid = inspection.get("service pid", "").strip()
    allowed = {service_pid} if service_pid.isdecimal() and service_pid != "0" else set()
    capture_output = inspection.get("capture process", "")
    capture_ids = _process_ids(capture_output)
    if capture_ids - allowed:
        return True
    if capture_output and (not capture_ids or "pifilm-capture" not in capture_output):
        return True
    return _foreign_camera_owners(inspection.get("camera owners", ""), allowed)


def restart_headless_service(client: Any, inspection: Mapping[str, str]) -> None:
    """Stop first, then refuse to start while any process still owns a camera."""
    if _has_foreign_initial_owner(inspection):
        raise RuntimeError("refusing restart: foreign capture or camera owner is active")
    # A known service-owned process may hold the camera initially. Stop it, then
    # recheck before starting a replacement so two processes cannot compete.
    status, _output, error = _run(client, f"sudo -n systemctl stop {SERVICE}")
    if status:
        detail = error.strip() or "command failed"
        raise RuntimeError(f"could not stop {SERVICE}: {detail}")
    status, process_output, _error = _run(client, CAPTURE_PROCESS_COMMAND)
    if status or process_output.strip():
        raise RuntimeError("refusing restart: a capture process remains after service stop")
    status, camera_output, _error = _run(client, CAMERA_OWNERS_COMMAND)
    if status or _foreign_camera_owners(camera_output, set()):
        raise RuntimeError("refusing restart: a process still owns a camera after service stop")
    status, _output, error = _run(client, f"sudo -n systemctl start {SERVICE}")
    if status:
        detail = error.strip() or "command failed"
        raise RuntimeError(f"could not start {SERVICE}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", type=Path, default=Path(".env"), help="local env file (default: .env)"
    )
    parser.add_argument(
        "--restart", action="store_true", help="restart the installed headless systemd service"
    )
    args = parser.parse_args(argv)
    config = deployment_config(parse_env_file(args.env))
    client = open_ssh_client(config)
    try:
        inspection = inspect_target(client, config)
        for label, output in inspection.items():
            print(f"{label}: {output or 'none'}")
        if args.restart:
            restart_headless_service(client, inspection)
            print("headless pifilm-capture service restarted")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
