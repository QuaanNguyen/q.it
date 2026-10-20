import argparse
import faulthandler
import json
import os
import traceback

from .artifacts import profile_for, profiles, verify
from .registry import ADAPTERS


def main():
    parser = argparse.ArgumentParser(prog="qit-adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("verify", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--model-path", required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8766)
            command.add_argument("--threads", type=int, default=4)
            command.add_argument("--allow-reviewed-code", action="store_true")
            command.add_argument("--startup-traceback-after", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "list":
            for profile in profiles():
                print(
                    f"{profile['model']}\t{profile['adapter']}\t{profile['precision']}\t{'available' if profile['adapter'] in ADAPTERS else 'unsupported'}"
                )
            return
        profile = profile_for(args.model)
        if args.command == "verify":
            print(
                json.dumps(
                    {
                        "model": profile["model"],
                        "revision": profile["revision"],
                        "artifacts": verify(profile, args.model_path),
                    },
                    indent=2,
                )
            )
            return
        if not 0 <= args.port <= 65535:
            raise ValueError("port must be between zero and 65535")
        from .backend import Backend
        from .server import CpuServer

        if not 0 <= args.startup_traceback_after <= 600:
            raise ValueError(
                "startup traceback interval must be between zero and 600 seconds"
            )
        if args.startup_traceback_after:
            faulthandler.dump_traceback_later(args.startup_traceback_after, repeat=True)
        try:
            backend = Backend(
                profile, args.model_path, args.threads, args.allow_reviewed_code
            )
        finally:
            if args.startup_traceback_after:
                faulthandler.cancel_dump_traceback_later()
        with CpuServer(
            ("127.0.0.1", args.port), backend, os.environ.get("QIT_API_KEY")
        ) as server:
            print(
                f"q.it optional adapter listening on http://127.0.0.1:{server.server_port}",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (ValueError, ImportError, OSError) as error:
        if getattr(args, "startup_traceback_after", 0):
            traceback.print_exc()
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
