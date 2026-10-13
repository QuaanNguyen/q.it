import argparse
import json
import os

from .catalog import profile_for, profiles


def main():
    parser = argparse.ArgumentParser(prog="qit-cpu")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("list")
    serve = subcommands.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--port", type=int, default=8081)
    serve.add_argument("--threads", type=int, default=4)
    serve.add_argument("--trust-publisher-code", action="store_true")
    options = parser.parse_args()
    if options.command == "list":
        print(json.dumps(profiles(), indent=2))
        return
    if not 0 <= options.port <= 65535:
        parser.error("port must be between zero and 65535")
    from .backend import Backend
    from .server import CpuServer

    backend = Backend(
        profile_for(options.model), options.threads, options.trust_publisher_code
    )
    with CpuServer(
        ("127.0.0.1", options.port), backend, os.getenv("QIT_CPU_API_KEY")
    ) as server:
        print(json.dumps(backend.metadata(), sort_keys=True), flush=True)
        print(
            f"q.it CPU provider listening on http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
