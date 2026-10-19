from pathlib import Path
import platform

from . import __version__
from .artifacts import (
    IMPLEMENTED_ADAPTERS,
    REVIEWED_SOURCE_ADAPTERS,
    locked_environment,
    fingerprint,
    verify,
)
from .decisions import DecisionRuntime
from .models import ModelRuntime
from .speech import SpeechRuntime
from .nemotron import NemotronRuntime
from .onnx_ocr import OnnxOcrRuntime
from .offline import forbid_external_connections


def audit_modules(modules, extra_tensors, torch, precision):
    parameters = {}
    buffers = []
    allowed = (
        {torch.float32, torch.bfloat16} if precision == "bfloat16" else {torch.float32}
    )
    for component, module in modules.items():
        for name, tensor in module.named_parameters():
            if tensor.device.type != "cpu" or tensor.dtype not in allowed:
                raise ValueError(
                    f"loaded tensor has unsupported device or precision: {component}.{name} {tensor.device} {tensor.dtype}"
                )
            parameters[id(tensor)] = {
                "name": component + "." + name,
                "shape": list(tensor.shape),
                "elements": tensor.numel(),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
        for name, tensor in module.named_buffers():
            if tensor.device.type != "cpu" or (
                getattr(tensor, "is_floating_point", lambda: False)()
                and tensor.dtype not in allowed
            ):
                raise ValueError(
                    f"loaded buffer has unsupported device or precision: {component}.{name}"
                )
            buffers.append(
                {
                    "name": component + "." + name,
                    "shape": list(tensor.shape),
                    "elements": tensor.numel(),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
    for name, tensor in extra_tensors.items():
        if tensor.device.type != "cpu" or tensor.dtype not in allowed:
            raise ValueError(
                f"unregistered learned component has unsupported device or precision: {name}"
            )
        parameters[id(tensor)] = {
            "name": name,
            "shape": list(tensor.shape),
            "elements": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
    total = sum(item["elements"] for item in parameters.values())
    if not 0 < total < 1_000_000_000:
        raise ValueError(
            f"complete loaded learned stack has {total} parameters and fails the sub-billion limit"
        )
    return {
        "parameter_count": total,
        "parameters": list(parameters.values()),
        "buffers": buffers,
    }


class Backend:
    def __init__(self, profile, directory, threads=4, allow_reviewed_code=False):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("CPU threads must be between one and sixty-four")
        self.profile = profile
        self.artifacts = verify(profile, directory)
        if profile["adapter"] not in IMPLEMENTED_ADAPTERS:
            raise ValueError(
                profile.get("unsupported_reason")
                or f"unsupported path: {profile['adapter']} still requires its native implementation and CPU qualification; see issue #{profile['issue']}"
            )
        if profile["adapter"] in REVIEWED_SOURCE_ADAPTERS and not allow_reviewed_code:
            raise ValueError(
                "pinned publisher source requires explicit --allow-reviewed-code after source review"
            )
        self.environment = locked_environment(profile)
        self.packages = self.environment["packages"]
        forbid_external_connections()
        if profile["adapter"] == "indra":
            self.runtime = OnnxOcrRuntime(profile, Path(directory).resolve(), threads)
            audit = self.runtime.audit
        else:
            import torch

            if torch.version.cuda is not None:
                raise ValueError(
                    "optional CPU adapters require a CPU-only PyTorch build"
                )
            torch.set_num_threads(threads)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            runtime = {
                "decision_scoring": DecisionRuntime,
                "speech_to_text": SpeechRuntime,
            }.get(profile["task"], ModelRuntime)
            if profile["adapter"] == "nemotron":
                runtime = NemotronRuntime
            self.runtime = runtime(profile, Path(directory).resolve(), torch)
            audit = audit_modules(
                {"model": self.runtime.model},
                self.runtime.extra_tensors,
                torch,
                profile["precision"],
            )
            logical = getattr(self.runtime, "logical_parameters", [])
            audit["logical_quantized_weights"] = logical
            audit["parameter_count"] += sum(entry["elements"] for entry in logical)
            if not 0 < audit["parameter_count"] < 1_000_000_000:
                raise ValueError(
                    "complete logical stack exceeds the under-one-billion parameter limit"
                )
        self.configuration = {
            "model": profile["model"],
            "revision": profile["revision"],
            "task": profile["task"],
            "device": "cpu",
            "dtype": profile["precision"],
            "threads": threads,
            "packages": self.packages,
            "environment": self.environment,
            "parameter_count": audit["parameter_count"],
            "parameter_count_upper_bound": audit.get("parameter_count_upper_bound"),
            "parameter_count_basis": audit.get(
                "parameter_count_basis", "loaded-learned-tensors"
            ),
            "parameter_audit": audit,
            "artifacts": self.artifacts,
            "reviewed_source_revision": profile["revision"]
            if allow_reviewed_code
            else None,
            "license": profile["license"],
            "license_url": f"https://huggingface.co/{profile['model']}/tree/{profile['revision']}",
            "platform": {
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "libc": platform.libc_ver(),
            },
            "decision": self.runtime.configuration,
            "preparation": getattr(self.runtime, "preparation", None),
            "retrieval": getattr(self.runtime, "retrieval", None),
            "configuration_id": fingerprint(
                {
                    "model": profile["model"],
                    "revision": profile["revision"],
                    "precision": profile["precision"],
                    "artifacts": self.artifacts,
                    "packages": self.packages,
                    "threads": threads,
                }
            ),
            "qualification": "experimental; real checkpoint execution remains unverified",
        }
        if profile["adapter"] == "d3":
            self.configuration["precision_admission"] = (
                "d3-native-bf16-backbone-fp32-readout-v1"
            )
        self.configuration["configuration_id"] = fingerprint(
            {
                name: value
                for name, value in self.configuration.items()
                if name != "configuration_id"
            }
        )

    def metadata(self):
        return {
            "schema_version": 1,
            "version": __version__,
            "model": self.profile["model"],
            "configuration": self.configuration,
        }

    def validate_request(self, payload):
        if (
            payload.get("model") != self.profile["model"]
            or payload.get("task") != self.profile["task"]
        ):
            raise ValueError("model or task differs from the loaded adapter")
        if self.profile["task"] != "decision_scoring":
            if not isinstance(payload.get("case"), dict):
                raise ValueError("case is required")
            if (
                type(payload.get("max_output_tokens")) is not int
                or not 1 <= payload["max_output_tokens"] <= 32768
            ):
                raise ValueError("bounded output limit is required")
            return
        case = payload.get("case")
        if not isinstance(case, dict) or not isinstance(case.get("decision"), dict):
            raise ValueError("single-choice case is required")
        if set(case) - {"id", "decision", "media", "media_type"}:
            raise ValueError("decision case contains unsupported fields")
        decision = case["decision"]
        if set(decision) != {"state", "question", "candidates"}:
            raise ValueError(
                "decision requests contain only state, question, and candidates; expected labels are forbidden"
            )
        if any(
            not isinstance(decision.get(key), str) or not decision[key].strip()
            for key in ("state", "question")
        ):
            raise ValueError("state and question must be nonempty text")
        candidates = decision["candidates"]
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 10:
            raise ValueError("single-choice requires two to ten candidates")
        if any(
            not isinstance(candidate, dict)
            or set(candidate) != {"id", "text"}
            or any(
                not isinstance(candidate[key], str) or not candidate[key].strip()
                for key in ("id", "text")
            )
            for candidate in candidates
        ):
            raise ValueError("each candidate needs an identifier and text")
        if len({candidate["id"] for candidate in candidates}) != len(candidates):
            raise ValueError("candidate identifiers must be unique")
        if case.get("media"):
            if self.profile["adapter"] != "d3":
                raise ValueError(
                    "image-backed execution is not yet qualified by this adapter"
                )
            if not case["media"].startswith(
                ("data:image/png;base64,", "data:image/jpeg;base64,")
            ):
                raise ValueError(
                    "decision images must be inline PNG or JPEG data; remote paths are forbidden"
                )
        if "max_output_tokens" in payload:
            raise ValueError("decision scoring does not accept generation limits")

    def invoke(self, payload):
        self.validate_request(payload)
        if self.profile["task"] == "decision_scoring":
            yield self.runtime.invoke(payload["case"])
        else:
            yield self.runtime.invoke(payload["case"], payload["max_output_tokens"])
