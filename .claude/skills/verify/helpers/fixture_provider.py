#!/usr/bin/env python3
"""Loopback Local CPU provider used by the verify skill.

Serves the two routes the runner expects for a `<base-url>/<model>` endpoint:

  GET  /<model>/v1/qit/model
  POST /<model>/v1/qit/invoke

Embeddings are a deterministic hashed bag of words, so MaxSim scores are
computed from real vectors rather than copied from the case expectation.
"""

import argparse
import hashlib
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def embed(words):
    row = [0.0] * 128
    if not words:
        words = ["empty"]
    for word in words:
        for round_number in range(8):
            digest = hashlib.sha256(
                f"{round_number}:{word.lower()}".encode("utf-8")
            ).digest()
            for offset, byte in enumerate(digest[:16]):
                row[round_number * 16 + offset] += (byte - 127.5) / 127.5
    norm = math.sqrt(sum(value * value for value in row))
    if norm == 0:
        row[0] = 1.0
        return row
    return [value / norm for value in row]


def document_vectors(text):
    words = text.split()
    if len(words) < 2:
        return [embed(words), embed(words)]
    middle = max(1, len(words) // 2)
    return [embed(words[:middle]), embed(words[middle:])]


def maxsim(query, documents):
    return [
        sum(
            max(
                sum(left * right for left, right in zip(query_vector, token))
                for token in document
            )
            for query_vector in query
        )
        for document in documents
    ]


def observation(case, inconsistent):
    query = [embed(case["input"].split())]
    documents = [document_vectors(text) for text in case["documents"]]
    scores = maxsim(query, documents)
    ranking = sorted(range(len(documents)), key=lambda index: (-scores[index], index))
    if inconsistent:
        scores = [score + (1.0 if index == 1 else 0.0) for index, score in enumerate(scores)]
    return {
        "schema_version": 1,
        "method": "maxsim",
        "query": query,
        "documents": documents,
        "scores": scores,
        "ranking": ranking,
        "tie_break": "document_order",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(
            json.dumps({"method": self.command, "path": self.path, "detail": format % args})
            + "\n"
        )
        sys.stderr.flush()

    def reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def model(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        return parts[0] if parts and parts[0] else "fixture"

    def do_GET(self):
        model = self.model()
        if not urlparse(self.path).path.endswith("/v1/qit/model"):
            self.reply(404, {"error": "unknown route"})
            return
        self.reply(
            200,
            {
                "schema_version": 1,
                "version": "0.1.0",
                "model": model,
                "configuration": {
                    "device": "cpu",
                    "task": "embedding",
                    "parameter_count": 10,
                    "revision": "fixture-revision",
                    "retrieval": {
                        "schema_version": 1,
                        "method": "maxsim",
                        "dimensions": 128,
                        "tie_break": "document_order",
                    },
                },
            },
        )

    def do_POST(self):
        model = self.model()
        if not urlparse(self.path).path.endswith("/v1/qit/invoke"):
            self.reply(404, {"error": "unknown route"})
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        case = body.get("case") or {}
        if not case.get("input") or not case.get("documents"):
            self.reply(400, {"error": "embedding case requires input and documents"})
            return
        event = {
            "type": "result",
            "retrieval": observation(case, model.endswith("inconsistent")),
        }
        body = (json.dumps(event) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path)
    options = parser.parse_args()
    server = ThreadingHTTPServer((options.host, options.port), Handler)
    if options.port_file:
        options.port_file.write_text(str(server.server_address[1]))
    print(
        f"fixture provider listening on http://{options.host}:{server.server_address[1]}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
