import argparse
import json
import os

from .qvac import fetch, profile_for, profiles, verify


def main():
    parser = argparse.ArgumentParser(prog="qit-nemo")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("list")
    for name in ("verify", "fetch", "serve"):
        command = subcommands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--route", required=True)
        command.add_argument("--model-dir", required=True)
        if name == "serve":
            command.add_argument("--trust-publisher-code", action="store_true")
            command.add_argument("--threads", type=int, default=4)
            command.add_argument("--port", type=int, default=8082)
    options = parser.parse_args()
    if options.command == "list":
        print(json.dumps(profiles(), indent=2))
        return
    try:
        profile = profile_for(options.model)
        if options.command == "serve":
            if not options.trust_publisher_code:
                parser.error(
                    "pinned publisher inference requires --trust-publisher-code"
                )
            if not 1 <= options.threads <= 64 or not 0 <= options.port <= 65535:
                parser.error("threads must be 1-64 and port must be 0-65535")
            from .nemo_runtime import NemoBackend
            from .server import CpuServer

            backend = NemoBackend(
                profile, options.route, options.model_dir, options.threads
            )
            with CpuServer(
                ("127.0.0.1", options.port), backend, os.getenv("QIT_CPU_API_KEY")
            ) as server:
                print(json.dumps(backend.metadata(), sort_keys=True), flush=True)
                print(
                    f"q.it NeMo CPU provider listening on http://127.0.0.1:{server.server_port}",
                    flush=True,
                )
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            return
        operation = fetch if options.command == "fetch" else verify
        result = operation(profile, options.route, options.model_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
