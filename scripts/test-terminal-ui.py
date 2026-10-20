import argparse
import fcntl
import http.server
import json
import os
from pathlib import Path
import pty
import select
import sqlite3
import struct
import subprocess
import tempfile
import termios
import threading
import time

import pyte


class Provider(http.server.BaseHTTPRequestHandler):
    delay = 0.4

    def log_message(self, *args):
        return

    def do_GET(self):
        self.respond({"version": "0.40.3"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/api/show":
            self.respond({"capabilities": ["completion"]})
        else:
            time.sleep(self.delay)
            self.respond(
                '{"response":"raspberry 42 edge","done":false}\n'
                '{"done":true,"eval_count":4,"eval_duration":200000000}\n',
                "application/x-ndjson",
            )

    def respond(self, value, content_type="application/json"):
        data = (value if isinstance(value, str) else json.dumps(value)).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            return


class Terminal:
    def __init__(self, executable, state):
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        self.original = termios.tcgetattr(self.slave)
        self.output = b""
        self.screen = pyte.Screen(80, 24)
        self.stream = pyte.ByteStream(self.screen)
        self.process = subprocess.Popen(
            [executable, "tui"],
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            env={**os.environ, "QIT_HOME": str(state), "TERM": "xterm-256color"},
        )
        self.read_until(b"Benchmarks")

    def read_until(self, marker, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if select.select([self.master], [], [], 0.1)[0]:
                data = os.read(self.master, 65536)
                self.output += data
                self.stream.feed(data)
                if b"\x1b[6n" in data:
                    os.write(self.master, b"\x1b[1;1R")
                if marker.decode() in "\n".join(self.screen.display):
                    return
            if self.process.poll() is not None:
                break
        self.process.kill()
        raise AssertionError(
            f"terminal did not show {marker!r}: {self.screen.display!r}"
        )

    def send(self, value, marker=None):
        os.write(self.master, value)
        if marker is not None:
            self.read_until(marker)

    def close(self, key=b"q"):
        self.send(key)
        self.process.wait(timeout=10)
        restored = termios.tcgetattr(self.slave)
        assert restored == self.original, "terminal settings were not restored"
        os.close(self.master)
        os.close(self.slave)
        assert self.process.returncode == 0, self.output[-1600:]


def database(state):
    return sqlite3.connect(Path(state) / "results.db")


def until_saved(state, count=1):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        with database(state) as connection:
            if (
                connection.execute("select count(*) from benchmark_samples").fetchone()[
                    0
                ]
                >= count
            ):
                return
        time.sleep(0.05)
    raise AssertionError("run did not save a sample")


def start(terminal, endpoint):
    terminal.send(b"2", b"Start run")
    terminal.send(b"\r" + b"\x7f" * 80 + b"core/text-generation-smoke\r")
    terminal.send(b"\x1b[B\x1b[B\rfixture\r")
    terminal.send(b"\x1b[B\r")
    terminal.send(b"\x7f" * 64 + endpoint.encode() + b"\r")
    terminal.send(b"\x1b[B\x1b[B\r\x7f0\r")
    terminal.send(b"\x1b[B" * 4 + b"\r", b"Live progress")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--report")
    args = parser.parse_args()
    executable = str(Path(args.binary).resolve())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    report = {
        "host": os.uname().sysname + " " + os.uname().machine,
        "terminal": "80x24",
    }
    with tempfile.TemporaryDirectory(prefix="qit-terminal-") as directory:
        state = Path(directory)
        terminal = Terminal(executable, state)
        for key, marker in [
            (b"2", b"Start run"),
            (b"3", b"No run started"),
            (b"4", b"Results"),
            (b"5", b"No successful runs"),
        ]:
            terminal.send(key, marker)
        terminal.send(b"1", b"Benchmarks")
        report["idle_process"] = subprocess.check_output(
            ["ps", "-p", str(terminal.process.pid), "-o", "pcpu=,rss=,time="], text=True
        ).strip()
        start(terminal, endpoint)
        until_saved(state)
        terminal.read_until(b"succeeded")
        terminal.send(b"4", b"Results")
        terminal.send(b"\r", b"Run detail")
        terminal.send(b"\x1b")
        terminal.send(b"5", b"Latency ms")
        terminal.close()
        with database(state) as connection:
            run = connection.execute(
                "select status, summary_json from benchmark_runs"
            ).fetchone()
            assert run[0] == "succeeded", run
            assert json.loads(run[1])["sample_count"] == 9, run
        Provider.delay = 3.0
        interrupted_state = state / "interrupted"
        terminal = Terminal(executable, interrupted_state)
        start(terminal, endpoint)
        until_saved(interrupted_state)
        before = subprocess.check_output(
            ["ps", "-p", str(terminal.process.pid), "-o", "time="], text=True
        ).strip()
        began = time.monotonic()
        time.sleep(2)
        after = subprocess.check_output(
            ["ps", "-p", str(terminal.process.pid), "-o", "time=,rss="], text=True
        ).split()

        def seconds(value):
            days, separator, clock = value.rpartition("-")
            return (int(days) * 86400 if separator else 0) + sum(
                float(part) * 60**position
                for position, part in enumerate(reversed(clock.split(":")))
            )

        report["waiting_cpu_percent"] = round(
            100 * (seconds(after[0]) - seconds(before)) / (time.monotonic() - began), 2
        )
        report["waiting_rss_kib"] = int(after[1])
        terminal.close(b"\x03")
        with database(interrupted_state) as connection:
            row = connection.execute(
                "select status,error,summary_json from benchmark_runs"
            ).fetchone()
            assert row[0] == "failed" and "interrupted" in row[1], row
            assert 0 < json.loads(row[2])["sample_count"] < 9, row
        report["checks"] = [
            "five screens",
            "real runner",
            "saved samples",
            "results detail",
            "comparison",
            "Ctrl-C cancellation",
            "terminal restoration",
        ]
    server.shutdown()
    result = json.dumps(report, indent=2)
    print(result)
    if args.report:
        Path(args.report).write_text(result + "\n")


if __name__ == "__main__":
    main()
