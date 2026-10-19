import hashlib
import importlib.metadata
import importlib.util
from importlib.resources import files
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tarfile
import tempfile

from . import __version__
from .media import audio_from_case
from .qvac import artifact_path, verify


def stable_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def environment_identity():
    system = platform.system()
    architecture = platform.machine()
    libc, version = platform.libc_ver()
    if system != "Linux" or architecture != "x86_64" or sys.version_info[:2] != (3, 12):
        raise ValueError(
            f"platform_unsupported: the retained NeMo lock targets Linux x86_64 CPython 3.12; found {system} {architecture} Python {platform.python_version()}; native ARM64 qualification is separate"
        )
    if libc != "glibc" or tuple(int(part) for part in version.split(".")[:2]) < (2, 28):
        raise ValueError(
            f"libc_unsupported: CPU Torch wheels require glibc 2.28 or later; found {libc} {version}"
        )
    lock = files("qit_cpu").joinpath("locks/nemo-linux-x86_64-py312.txt").read_bytes()
    packages = {}
    for line in lock.decode().splitlines():
        requirement = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if requirement is None:
            continue
        name, expected = requirement.groups()
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(
                f"package_missing: {name}=={expected}; install the complete hashed NeMo CPU lock"
            ) from error
        if distribution.version != expected:
            raise ValueError(
                f"package_version: {name} requires {expected}, found {distribution.version}; use an isolated environment"
            )
        record = distribution.read_text("RECORD") or ""
        packages[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        }
    if not packages:
        raise ValueError("package_lock: the NeMo CPU dependency lock is empty")
    package_source = files("qit_cpu")
    source = {
        name: hashlib.sha256(package_source.joinpath(name).read_bytes()).hexdigest()
        for name in (
            "nemo_cli.py",
            "nemo_runtime.py",
            "qvac.py",
            "media.py",
            "server.py",
            "qvac-models.json",
        )
    }
    return {
        "system": system,
        "architecture": architecture,
        "libc": libc,
        "libc_version": version,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "python_abi": sys.implementation.cache_tag,
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "dependencies": packages,
        "provider_source_sha256": source,
    }


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
            "offline_network: NeMo serving cannot connect to external hosts; acquire local artifacts explicitly before startup"
        )

    sys.addaudithook(audit)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"


def archive_inventory(path):
    inventory = []
    with tarfile.open(path) as archive:
        seen = set()
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.name in seen:
                raise ValueError(
                    "archive_path: NeMo archive contains an unsafe or duplicate member"
                )
            seen.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    "archive_path: NeMo archive contains a nonregular member"
                )
            checksum = hashlib.sha256()
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive_member: NeMo archive member cannot be read")
            with source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(chunk)
            inventory.append(
                {
                    "name": member.name,
                    "bytes": member.size,
                    "sha256": checksum.hexdigest(),
                }
            )
    if not inventory:
        raise ValueError("archive_empty: NeMo archive has no regular members")
    return inventory


class NemoBackend:
    def __init__(self, profile, route, directory, threads):
        self.profile = profile
        self.route = route
        self.threads = threads
        self.directory = Path(directory).resolve()
        self.artifacts = verify(profile, route, self.directory)
        self.environment = environment_identity()
        forbid_external_connections()
        try:
            import torch
            import torchaudio
            import nemo.collections.asr as asr
            from omegaconf import OmegaConf, open_dict
        except Exception as error:
            raise ValueError(
                f"native_import: NeMo CPU dependencies could not import: {type(error).__name__}: {error}"
            ) from error
        if (
            torch.version.cuda is not None
            or not torch.__version__.endswith("+cpu")
            or not torchaudio.__version__.endswith("+cpu")
        ):
            raise ValueError(
                "cpu_build: the isolated NeMo environment requires matching CPU-only Torch and Torchaudio wheels"
            )
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.device = torch.device("cpu")
        self.archive_members = archive_inventory(
            artifact_path(self.directory, profile["archive"])
        )
        source_name = "inference_gated.py" if route == "automatic" else "inference.py"
        source_path = artifact_path(self.directory, source_name)
        try:
            specification = importlib.util.spec_from_file_location(
                "qit_qvac_publisher_" + self.artifacts[source_name]["sha256"],
                source_path,
            )
            if specification is None or specification.loader is None:
                raise ValueError("publisher source could not be loaded")
            self.publisher = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(self.publisher)
        except Exception as error:
            raise ValueError(
                f"publisher_import: pinned QVAC source could not import: {error}"
            ) from error
        try:
            self.model = asr.models.ASRModel.restore_from(
                str(self.directory / profile["archive"]),
                map_location=self.device,
                strict=True,
            )
            self.model.to(device=self.device, dtype=torch.float32).eval()
        except Exception as error:
            raise ValueError(
                f"checkpoint_restore: pinned QVAC NeMo archive could not restore on CPU float32: {error}"
            ) from error
        try:
            with open_dict(self.model.cfg.decoding):
                self.model.cfg.decoding.strategy = "greedy_batch"
            self.publisher.safe_decoding(self.model, cuda_graph=False)
            self.decoder = OmegaConf.to_container(
                self.model.cfg.decoding, resolve=True, enum_to_str=True
            )
            if (
                self.decoder["strategy"] != "greedy_batch"
                or self.decoder["greedy"]["use_cuda_graph_decoder"] is not False
                or self.decoder["fused_batch_size"] != -1
            ):
                raise ValueError(
                    "actual decoder differs from the fixed CPU greedy configuration"
                )
            self.preprocessor = OmegaConf.to_container(
                self.model.cfg.preprocessor, resolve=True
            )
        except Exception as error:
            raise ValueError(
                f"decoder_configuration: QVAC CPU greedy decoding is unsupported: {error}"
            ) from error
        self.adapters = []
        self.fixed_handles = []
        self.gate = None
        self.gate_tap = None
        self.gate_threshold = None
        self.adapter_calls = []
        try:
            if route == "multilingual":
                self.publisher.attach_adapters(
                    self.model,
                    str(self.directory / "adapters.safetensors"),
                    self.device,
                    torch.float32,
                )
                self.adapters = list(self.model._adapters)
                self.fixed_handles = list(self.model._adapter_handles)
                self.validate_adapter_shape()
                self.fixed_handles += self.adapter_guards()
            elif route == "automatic":
                self.adapters = self.publisher.make_adapters(
                    str(self.directory / "adapters.safetensors"),
                    self.device,
                    torch.float32,
                )
                self.validate_adapter_shape()
                self.gate, self.gate_tap, self.gate_threshold = (
                    self.publisher.load_gate(
                        str(self.directory / "language_gate.safetensors"), self.device
                    )
                )
                self.gate.to(device=self.device, dtype=torch.float32).eval()
                if (
                    not 0 <= self.gate_tap < len(self.model.encoder.layers)
                    or not math.isfinite(self.gate_threshold)
                    or not 0 <= self.gate_threshold <= 1
                ):
                    raise ValueError("language gate tap or threshold is invalid")
        except Exception as error:
            raise ValueError(
                f"adapter_gate: QVAC adapter or language-gate setup failed: {error}"
            ) from error
        components = [("trunk", self.model)] + [
            (f"adapter_{index}", adapter) for index, adapter in enumerate(self.adapters)
        ]
        if self.gate is not None:
            components.append(("language_gate", self.gate))
        self.components, self.parameter_count = self.audit_components(components)
        if (
            self.components[0]["state_dict_elements"]
            != profile["published_tensor_elements"]
        ):
            raise ValueError(
                "component_count: restored trunk state dictionary differs from the pinned publisher tensor export"
            )
        if not 0 < self.parameter_count < 1_000_000_000:
            raise ValueError(
                "component_count: complete QVAC learned stack fails the under-one-billion limit"
            )
        try:
            tokenizer = self.model.tokenizer
            processor = getattr(tokenizer, "tokenizer", tokenizer)
            serialized = processor.serialized_model_proto()
            self.tokenizer = {
                "class": type(tokenizer).__module__
                + "."
                + type(tokenizer).__qualname__,
                "processor_class": type(processor).__module__
                + "."
                + type(processor).__qualname__,
                "serialized_model_sha256": hashlib.sha256(serialized).hexdigest(),
                "vocabulary_size": tokenizer.vocab_size,
            }
            self.tokenizer["archive_members"] = [
                member["name"]
                for member in self.archive_members
                if member["sha256"] == self.tokenizer["serialized_model_sha256"]
            ]
            if not self.tokenizer["archive_members"]:
                raise ValueError(
                    "restored tokenizer does not match a pinned archive member"
                )
        except Exception as error:
            raise ValueError(
                f"tokenizer_identity: restored QVAC tokenizer identity could not be audited: {error}"
            ) from error
        self.configuration_digest = stable_digest(
            {
                "revision": profile["revision"],
                "route": route,
                "artifacts": self.artifacts,
                "decoder": self.decoder,
                "preprocessor": self.preprocessor,
                "tokenizer": self.tokenizer,
                "lock_sha256": self.environment["lock_sha256"],
                "provider_source_sha256": self.environment["provider_source_sha256"],
            }
        )

    def validate_adapter_shape(self):
        layers = self.publisher.ADAPTER_LAYERS
        if (
            len(self.adapters) != 6
            or len(layers) != 6
            or max(layers) >= len(self.model.encoder.layers)
        ):
            raise ValueError(
                "six adapters must target all six publisher encoder layers"
            )
        if self.route == "multilingual" and len(self.fixed_handles) != 6:
            raise ValueError("six publisher adapter hooks must be attached")
        if self.route == "multilingual":
            self.validate_adapter_handles(self.fixed_handles)

    def validate_adapter_handles(self, handles):
        if len(handles) != 6 or any(
            handle.id not in self.model.encoder.layers[layer]._forward_hooks
            for handle, layer in zip(handles, self.publisher.ADAPTER_LAYERS)
        ):
            raise ValueError(
                "adapter_hook: every publisher adapter hook must remain attached"
            )

    def adapter_guards(self):
        self.adapter_calls = [0] * len(self.adapters)
        handles = []
        for index, (adapter, layer) in enumerate(
            zip(self.adapters, self.publisher.ADAPTER_LAYERS)
        ):

            def guard(
                module,
                arguments,
                output,
                expected=adapter.down.in_features,
            ):
                value = output[0] if isinstance(output, tuple) else output
                if value.dim() != 3 or value.shape[-1] != expected:
                    raise ValueError(
                        "adapter_feature_shape: publisher adapter would silently skip incompatible encoder features"
                    )

            def count_call(module, arguments, output, position=index):
                self.adapter_calls[position] += 1

            handles.append(
                self.model.encoder.layers[layer].register_forward_hook(
                    guard, prepend=True
                )
            )
            handles.append(adapter.register_forward_hook(count_call))
        return handles

    def audit_components(self, components):
        seen = set()
        audited = []
        total = 0
        for name, component in components:
            tensors = []
            count = 0
            state = component.state_dict()
            for kind, entries in (
                ("parameter", component.named_parameters()),
                ("buffer", component.named_buffers()),
            ):
                for tensor_name, tensor in entries:
                    if tensor.device.type != "cpu" or (
                        tensor.is_floating_point()
                        and tensor.dtype != self.torch.float32
                    ):
                        raise ValueError(
                            f"component_dtype: {name}.{tensor_name} is not CPU float32"
                        )
                    if kind == "parameter":
                        if not tensor.is_floating_point():
                            raise ValueError(
                                f"component_dtype: {name}.{tensor_name} is not a floating learned parameter"
                            )
                        count += tensor.numel()
                        if id(tensor) not in seen:
                            total += tensor.numel()
                            seen.add(id(tensor))
                    tensors.append(
                        {
                            "kind": kind,
                            "persistent": tensor_name in state,
                            "name": tensor_name,
                            "shape": list(tensor.shape),
                            "elements": tensor.numel(),
                            "dtype": str(tensor.dtype),
                            "device": str(tensor.device),
                        }
                    )
            audited.append(
                {
                    "name": name,
                    "parameter_count": count,
                    "state_dict_elements": sum(
                        tensor.numel() for tensor in state.values()
                    ),
                    "tensors": tensors,
                }
            )
        return audited, total

    def metadata(self):
        return {
            "schema_version": 1,
            "version": __version__,
            "model": self.profile["model"],
            "configuration": {
                "task": "speech_to_text",
                "device": "cpu",
                "dtype": "float32",
                "threads": self.threads,
                "parameter_count": self.parameter_count,
                "revision": self.profile["revision"],
                "implementation": "qvac_nemo",
                "route": self.route,
                "configuration_digest": self.configuration_digest,
                "pid": os.getpid(),
                "packages": {
                    name: value["version"]
                    for name, value in self.environment["dependencies"].items()
                },
                "environment": self.environment,
                "artifacts": self.artifacts,
                "archive_members": self.archive_members,
                "components": self.components,
                "tokenizer": self.tokenizer,
                "preprocessor": self.preprocessor,
                "decoder": self.decoder,
                "audio_preparation": {
                    "sampling_rate": 16000,
                    "channels": "mono_mean",
                    "resampling": "scipy.signal.resample_poly",
                    "maximum_seconds": 30,
                },
                "language_gate": None
                if self.gate is None
                else {
                    "tap": self.gate_tap,
                    "threshold": self.gate_threshold,
                    "maximum_gate_audio_seconds": 20,
                },
            },
        }

    def validate_request(self, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("model") != self.profile["model"]
            or payload.get("task") != "speech_to_text"
        ):
            raise ValueError(
                "request must name the loaded QVAC model and speech_to_text task"
            )
        if not isinstance(payload.get("case"), dict) or not isinstance(
            payload["case"].get("media"), str
        ):
            raise ValueError("speech request requires one inline audio case")

    def invoke(self, payload):
        self.validate_request(payload)
        import numpy as np
        import soundfile as sf

        samples = audio_from_case(payload["case"], 16000)
        if not np.isfinite(samples).all():
            raise ValueError(
                "audio_preparation: resampled speech must contain finite samples"
            )
        duration = len(samples) / 16000
        routing = {"requested_mode": self.route, "selected_route": self.route}
        handles = []
        with tempfile.TemporaryDirectory(prefix="qit-nemo-audio-") as directory:
            audio = str(Path(directory) / "input.wav")
            sf.write(audio, samples, 16000, subtype="FLOAT")
            with self.torch.inference_mode():
                try:
                    if self.route == "automatic":

                        def gate_guard(module, arguments, output):
                            value = output[0] if isinstance(output, tuple) else output
                            if (
                                value.dim() != 3
                                or value.shape[-1] != self.gate.pool.proj[0].in_features
                            ):
                                raise ValueError(
                                    "gate_feature_shape: language gate received incompatible encoder features"
                                )

                        gate_handle = self.model.encoder.layers[
                            self.gate_tap
                        ].register_forward_hook(gate_guard, prepend=True)
                        try:
                            probabilities = self.publisher.gate_probs(
                                self.model,
                                self.gate,
                                self.gate_tap,
                                [audio],
                                self.device,
                                max_len_s=20.0,
                            )
                        finally:
                            gate_handle.remove()
                        if (
                            len(probabilities) != 1
                            or not math.isfinite(probabilities[0])
                            or not 0 <= probabilities[0] <= 1
                        ):
                            raise ValueError(
                                "gate_probability: language gate returned an invalid distribution"
                            )
                        multilingual = probabilities[0] >= self.gate_threshold
                        routing.update(
                            selected_route="multilingual"
                            if multilingual
                            else "english",
                            p_non_english=probabilities[0],
                            threshold=self.gate_threshold,
                            gate_audio_seconds=min(duration, 20.0),
                        )
                        if multilingual:
                            handles = self.publisher.attach(self.model, self.adapters)
                            self.validate_adapter_handles(handles)
                            handles += self.adapter_guards()
                    elif self.route == "multilingual":
                        self.validate_adapter_handles(self.fixed_handles[:6])
                        self.adapter_calls = [0] * 6
                    output = self.model.transcribe([audio], batch_size=1, verbose=False)
                    if isinstance(output, tuple):
                        output = output[0]
                    if not isinstance(output, list) or len(output) != 1:
                        raise ValueError(
                            "decoder_result: NeMo must return exactly one transcript"
                        )
                    text = getattr(output[0], "text", output[0])
                    if not isinstance(text, str):
                        raise ValueError("decoder_result: NeMo transcript must be text")
                    if routing["selected_route"] == "multilingual":
                        if len(self.adapter_calls) != 6 or not all(self.adapter_calls):
                            raise ValueError(
                                "adapter_hook: at least one loaded adapter did not execute"
                            )
                        routing["adapter_hook_calls"] = list(self.adapter_calls)
                finally:
                    for handle in handles:
                        handle.remove()
        yield {
            "type": "result",
            "output_text": text,
            "audio_duration_seconds": duration,
            "speech_routing": routing,
        }
