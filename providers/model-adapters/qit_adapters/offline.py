import ipaddress
import os
import sys


def forbid_external_connections():
    def audit(event, arguments):
        if event == "socket.connect":
            address = arguments[1]
            if not isinstance(address, tuple):
                return
            host = address[0]
        elif event == "socket.getaddrinfo":
            host = arguments[0]
            if host is None:
                return
        else:
            return
        if host == "localhost":
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise RuntimeError(
            "offline serving forbids external connections; install pinned artifacts explicitly before startup"
        )

    sys.addaudithook(audit)
    for name in (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        os.environ[name] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ[name] = "-1"
