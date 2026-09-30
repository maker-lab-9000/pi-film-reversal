"""A deliberately small authenticated HTTP surface for remote capture buttons."""

from __future__ import annotations

import hmac
import json
import socket
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import unquote, urlsplit

from .controller import CaptureController, JobSnapshot
from .power import PowerStatus
from .thumbnail import ThumbnailError, fitted_jpeg

MAX_BODY_BYTES = 8 * 1024
MAX_CONNECTIONS = 8
MAX_JOBS = 100
CONNECTION_TIMEOUT_SECONDS = 5


class _BoundedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        request.settimeout(CONNECTION_TIMEOUT_SECONDS)
        super().process_request(request, client_address)

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int],
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class RemoteCaptureServer:
    """Serve only captures submitted through this process and server lifetime."""

    def __init__(
        self,
        controller: CaptureController,
        token: str,
        listen: tuple[str, int],
        power: Callable[[], PowerStatus | None] | None = None,
    ) -> None:
        if not token:
            raise ValueError("a non-empty remote token is required")
        self._controller = controller
        self._token = token
        # Optional UPS reader. Called on the request thread, so it must be a
        # cheap copy of a cached value (PowerMonitor.snapshot), never an I2C read.
        self._power = power
        self.instance_id = str(uuid.uuid4())
        self._jobs: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        handler = self._handler_type()
        self._httpd = _BoundedHTTPServer(listen, handler)
        self._httpd.timeout = CONNECTION_TIMEOUT_SECONDS
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="pifilm-remote-http",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join()
            self._thread = None
        self._httpd.server_close()

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        remote = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                if not remote._authorized(self):
                    return
                remote._get(self)

            def do_POST(self) -> None:  # noqa: N802
                if not remote._authorized(self):
                    return
                remote._post(self)

            def log_message(self, _format: str, *args: object) -> None:
                # The foreground capture UI owns user-visible output.
                return

        return Handler

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        # ``compare_digest`` rejects non-ASCII strings, but accepts UTF-8 bytes.
        expected = f"Bearer {self._token}".encode()
        supplied = handler.headers.get("Authorization", "").encode()
        if hmac.compare_digest(supplied, expected):
            return True
        self._send_json(handler, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def _get(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlsplit(handler.path).path
        if path == "/v1/status":
            self._send_json(handler, HTTPStatus.OK, self._status_payload())
            return
        parts = [unquote(part) for part in path.split("/")]
        if len(parts) == 4 and parts[:3] == ["", "v1", "captures"]:
            request_id = parts[3]
            job = self._job(request_id)
            if job is None:
                self._send_json(handler, HTTPStatus.NOT_FOUND, {"error": "unknown_capture"})
            else:
                self._send_json(handler, HTTPStatus.OK, self._job_payload(job))
            return
        if len(parts) == 5 and parts[:3] == ["", "v1", "captures"] and parts[4] == "image.jpg":
            self._image(handler, parts[3])
            return
        self._send_json(handler, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _post(self, handler: BaseHTTPRequestHandler) -> None:
        if urlsplit(handler.path).path != "/v1/captures":
            self._send_json(handler, HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        body = self._read_body(handler)
        if body is None:
            return
        try:
            decoded = json.loads(body)
            request_id = decoded["request_id"]
            if not isinstance(request_id, str):
                raise ValueError
            uuid.UUID(request_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json(handler, HTTPStatus.BAD_REQUEST, {"error": "invalid_request_id"})
            return

        with self._lock:
            self._trim_jobs_locked()
            known = request_id in self._jobs
            if not known and self._controller.status(request_id) is not None:
                expired = True
                job = None
            else:
                expired = False
                job = self._controller.submit(request_id)
                if job.error_code != "busy" and not known:
                    self._jobs[request_id] = None
                    self._trim_jobs_locked()
        if expired:
            self._send_json(handler, HTTPStatus.NOT_FOUND, {"error": "expired_capture"})
            return
        assert job is not None
        if job.error_code == "busy":
            self._send_json(handler, HTTPStatus.CONFLICT, {"error": "busy"})
            return
        self._send_json(handler, HTTPStatus.ACCEPTED, self._job_payload(job))

    def _read_body(self, handler: BaseHTTPRequestHandler) -> bytes | None:
        raw_length = handler.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(
                handler, HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"},
            )
            return None
        if length > MAX_BODY_BYTES:
            self._send_json(
                handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"},
            )
            return None
        return handler.rfile.read(length)

    def _status_payload(self) -> dict[str, Any]:
        snapshot = self._controller.snapshot()
        battery = None
        if self._power is not None:
            try:
                battery = self._power()
            except Exception:
                # Power reader must not take down /v1/status; degrade to null and continue.
                # The reader is expected to be a cached snapshot, never performing blocking I/O,
                # so any exception indicates a serious bug in the reader implementation. We still
                # degrade rather than crash to keep the response reliable for firmware parsing.
                pass
        return {
            "instance_id": self.instance_id,
            "ready": not snapshot.closed,
            "active_capture_id": snapshot.active_job.request_id if snapshot.active_job else None,
            "last_completed_id": (
                snapshot.last_completed_job.request_id if snapshot.last_completed_job else None
            ),
            "pi_battery": (
                {
                    "percent": battery.percent,
                    "voltage_mv": battery.voltage_mv,
                    "external_power": battery.external_power,
                }
                if battery is not None else None
            ),
            # Ignored by current Stick firmware; lets any client show the 0/2, 1/2 count.
            "double_exposure": {
                "enabled": snapshot.double_exposure,
                "taken": snapshot.exposures_taken,
            },
        }

    def _job(self, request_id: str) -> JobSnapshot | None:
        with self._lock:
            self._trim_jobs_locked()
            if request_id not in self._jobs:
                return None
        return self._controller.status(request_id)

    def _trim_jobs_locked(self) -> None:
        while len(self._jobs) > MAX_JOBS:
            for request_id in self._jobs:
                job = self._controller.status(request_id)
                if job is not None and job.state not in {"queued", "processing"}:
                    self._jobs.pop(request_id)
                    break
            else:
                return

    def _image(self, handler: BaseHTTPRequestHandler, request_id: str) -> None:
        job = self._job(request_id)
        if job is None:
            self._send_json(handler, HTTPStatus.NOT_FOUND, {"error": "unknown_capture"})
            return
        if job.state != "complete":
            self._send_json(handler, HTTPStatus.CONFLICT, {"error": "capture_not_complete"})
            return
        source = getattr(job.result, "pifilm", None)
        if source is None:
            self._send_json(
                handler, HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "image_unavailable"},
            )
            return
        try:
            data = fitted_jpeg(source)
        except ThumbnailError:
            self._send_json(handler, HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_image"})
            return
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)

    def _job_payload(self, job: JobSnapshot) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": job.request_id, "state": job.state}
        if job.state == "complete" and getattr(job.result, "pifilm", None) is not None:
            payload["image_url"] = f"/v1/captures/{job.request_id}/image.jpg"
        if job.error_code is not None:
            payload["error_code"] = job.error_code
        if job.error_message is not None:
            payload["error_message"] = job.error_message
        return payload

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        payload: dict[str, Any],
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)
