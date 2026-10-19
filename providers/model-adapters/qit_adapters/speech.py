import json

from .media import audio_from_case


class SpeechRuntime:
    def __init__(self, profile, directory, torch):
        from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

        self.profile = profile
        self.torch = torch
        self.configuration = None
        self.extra_tensors = {}
        self.logical_parameters = []
        loaded = load_parakeet_tdt(
            str(directory), device="cpu", dtype=torch.float32, local_files_only=True
        )
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.preparation = {
            "audio_rate": 16000,
            "channels": 1,
            "max_seconds": 30,
            "decoder": "publisher native full-utterance TDT",
            "max_tokens": None,
            "runtime": "direct local Kestrel; no Photon client or engine lifecycle",
        }
        if profile["adapter"] == "redux":
            manifest = json.loads((directory / "ternary.json").read_text())
            expected = sum(
                entry["in_features"] * entry["out_features"]
                for entry in manifest["quantized_modules"]
            )
            for name, module in self.model.named_modules():
                if type(module).__name__ != "TernaryLinear":
                    continue
                weight = module.weight
                if (
                    weight.mode != "gemm8"
                    or weight._dense is not None
                    or weight._panels is None
                    or weight.device.type != "cpu"
                ):
                    raise ValueError(
                        f"unsupported Redux resident format for {name}; native CPU GEMM8 panels are required"
                    )
                if (
                    weight.dtype != torch.float32
                    or weight._scales32.dtype != torch.float32
                    or weight._scales32.device.type != "cpu"
                ):
                    raise ValueError(
                        f"unsupported Redux activation or learned-scale precision for {name}"
                    )
                if not torch.equal(module.scales, weight._scales32):
                    raise ValueError(
                        f"Redux native scales differ from loaded scales for {name}"
                    )
                self.extra_tensors[name + ".learned_scales"] = weight._scales32
                self.logical_parameters.append(
                    {
                        "name": name + ".logical_weight",
                        "shape": [module.out_features, module.in_features],
                        "elements": module.out_features * module.in_features,
                        "dtype": "native_ternary",
                        "group_size": module.group_size,
                        "resident_format": weight.mode,
                        "panel_bytes": weight._panels.nbytes,
                    }
                )
            actual = sum(entry["elements"] for entry in self.logical_parameters)
            if not self.logical_parameters or actual != expected:
                raise ValueError(
                    f"Redux fused loaded logical weights ({actual}) do not reconcile with the complete original ternary manifest ({expected})"
                )
            self.preparation.update(
                {
                    "weight_format": "publisher native ternary",
                    "activation_precision": "native groupwise int8 in ternary conformer GEMM; float32 elsewhere",
                    "cache_precision": "native Kestrel configuration with CPU float32 registered tensors",
                }
            )
        elif getattr(self.model, "is_ternary", False):
            raise ValueError("Ultra float32 must not load a ternary checkpoint")

    def invoke(self, case, max_tokens):
        from kestrel.models.parakeet_tdt.features import parakeet_features

        samples = audio_from_case(case, 16000)
        features, mask = parakeet_features(samples)
        with self.torch.inference_mode():
            result = self.model.generate(features, mask, max_tokens=None)
        length = int(result.lengths[0])
        tokens = result.sequences[0, :length].tolist()
        return {
            "type": "result",
            "output_text": self.tokenizer.decode(tokens),
            "audio_duration_seconds": len(samples) / 16000,
            "speech_routing": {
                "route": self.profile["adapter"],
                "decoding": "native full utterance",
                "feature_frames": int(mask.sum()),
                "emitted_token_count": length,
            },
        }
