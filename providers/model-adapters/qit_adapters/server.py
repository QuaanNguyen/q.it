import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CpuServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, api_key=None):
        if address[0] != "127.0.0.1":
            raise ValueError("the CPU server must bind to loopback")
        super().__init__(address, Handler)
        self.backend = backend
        self.api_key = api_key
        self.inference = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def authorized(self):
        key = self.server.api_key
        if key and not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + key
        ):
            self.json_response(401, {"error": "invalid API key"})
            return False
        return True

    def json_response(self, status, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/v1/qit/model":
            self.json_response(200, self.server.backend.metadata())
        elif self.path == "/health":
            self.json_response(200, {"ok": True})
        else:
            self.json_response(404, {"error": "unknown CPU endpoint"})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != "/v1/qit/invoke":
            self.json_response(404, {"error": "unknown CPU endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError("request must have a bounded content length")
            payload = json.loads(self.rfile.read(size))
            self.server.backend.validate_request(payload)
        except (ValueError, TypeError, KeyError) as error:
            self.json_response(400, {"error": str(error)})
            return
        if not self.server.inference.acquire(blocking=False):
            self.json_response(
                409, {"error": "CPU model is already executing a request"}
            )
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for event in self.server.backend.invoke(payload):
                    self.wfile.write(
                        json.dumps(event, allow_nan=False).encode() + b"\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                self.wfile.write(
                    json.dumps({"type": "error", "error": str(error)}).encode() + b"\n"
                )
                self.wfile.flush()
        finally:
            self.server.inference.release()
