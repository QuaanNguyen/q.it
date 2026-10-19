from dataclasses import asdict
import hashlib
from pathlib import PurePosixPath
import tarfile

from .media import audio_from_case


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


class NemotronRuntime:
    def __init__(self, profile, directory, torch):
        import nemo.collections.asr as asr
        import torchaudio
        from omegaconf import OmegaConf, open_dict

        if not torchaudio.__version__.endswith("+cpu"):
            raise ValueError("Nemotron requires the pinned CPU-only Torchaudio build")
        archive = directory / "nemotron-3.5-asr-streaming-0.6b.nemo"
        members = archive_inventory(archive)
        self.model = (
            asr.models.ASRModel.restore_from(
                str(archive), map_location=torch.device("cpu"), strict=True
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        if type(self.model).__name__ != "EncDecRNNTBPEModelWithPrompt":
            raise ValueError("unsupported Nemotron restored model class")
        sample_rate = self.model.cfg.get(
            "sample_rate", getattr(self.model, "sample_rate", None)
        )
        if sample_rate != 16000:
            raise ValueError("native Nemotron waveform loading requires a 16 kHz model")
        prompts = OmegaConf.to_container(
            self.model.cfg.model_defaults.prompt_dictionary, resolve=True
        )
        if "en-US" not in prompts:
            raise ValueError("the restored Nemotron prompt dictionary lacks en-US")
        decoder = OmegaConf.create(self.model.cfg.decoding)
        with open_dict(decoder):
            decoder.strategy = "greedy_batch"
            decoder.greedy.use_cuda_graph_decoder = False
            decoder.fused_batch_size = -1
        self.model.change_decoding_strategy(decoder, verbose=False)
        actual_decoder = OmegaConf.to_container(
            self.model.cfg.decoding, resolve=True, enum_to_str=True
        )
        if (
            actual_decoder["strategy"] != "greedy_batch"
            or actual_decoder["greedy"]["use_cuda_graph_decoder"] is not False
            or actual_decoder["fused_batch_size"] != -1
        ):
            raise ValueError(
                "restored Nemotron CPU decoder differs from the requested route"
            )
        tokenizer = getattr(self.model.tokenizer, "tokenizer", self.model.tokenizer)
        tokenizer_sha = hashlib.sha256(tokenizer.serialized_model_proto()).hexdigest()
        if not any(member["sha256"] == tokenizer_sha for member in members):
            raise ValueError(
                "restored Nemotron tokenizer does not match an archive member"
            )
        streaming = getattr(self.model.encoder, "streaming_cfg", None)
        if streaming is None:
            raise ValueError(
                "restored Nemotron encoder does not expose its streaming configuration"
            )
        streaming = asdict(streaming)
        if "chunk_size" not in streaming:
            raise ValueError(
                "restored Nemotron streaming configuration lacks chunk size"
            )
        self.torch = torch
        self.prompt_index = int(prompts["en-US"])
        if not 0 <= self.prompt_index < self.model.num_prompts:
            raise ValueError(
                "Nemotron en-US prompt index exceeds the native prompt width"
            )
        self.extra_tensors = {}
        self.configuration = None
        self.preparation = {
            "route": "nemo-full-utterance",
            "audio_rate": 16000,
            "channels": 1,
            "max_seconds": 30,
            "audio_input": "native in-memory float32 waveform loader",
            "native_min_audio_seconds": 1.0,
            "native_audio_pad_direction": "both",
            "language_prompt_application": "native dynamic prompt; actual one-hot input observed at the prompt projection",
            "language_prompt": {"language": "en-US", "index": prompts["en-US"]},
            "prompt_dictionary": prompts,
            "decoder": actual_decoder,
            "encoder": OmegaConf.to_container(self.model.cfg.encoder, resolve=True),
            "streaming_configuration": streaming,
            "cache_execution": "full utterance; no streaming cache carried between cases",
            "archive_members": members,
            "tokenizer_sha256": tokenizer_sha,
            "latency_scope": "whole invocation; no streaming latency or generation metrics",
        }

    def invoke(self, case, max_tokens):
        samples = audio_from_case(case, 16000)
        observed = []

        def observe_prompt(module, inputs):
            prompt = inputs[0][..., -self.model.num_prompts :]
            if (
                not self.torch.all(prompt[..., self.prompt_index] == 1)
                or not self.torch.all((prompt == 0) | (prompt == 1))
                or not self.torch.all(prompt.sum(dim=-1) == 1)
            ):
                raise ValueError(
                    "native Nemotron did not apply the requested en-US prompt"
                )
            observed.append({"index": self.prompt_index, "frames": prompt.shape[-2]})

        observer = self.model.prompt_kernel.register_forward_pre_hook(observe_prompt)
        try:
            with self.torch.inference_mode():
                hypotheses = self.model.transcribe(
                    audio=[samples],
                    batch_size=1,
                    return_hypotheses=True,
                    num_workers=0,
                    verbose=False,
                    timestamps=False,
                    target_lang="en-US",
                )
        finally:
            observer.remove()
        if not observed:
            raise ValueError(
                "native Nemotron did not expose its effective language prompt"
            )
        if not isinstance(hypotheses, list) or len(hypotheses) != 1:
            raise ValueError("Nemotron must return one complete native hypothesis")
        text = hypotheses[0].text
        if not isinstance(text, str):
            raise ValueError("Nemotron did not return a native transcript")
        return {
            "type": "result",
            "output_text": text,
            "audio_duration_seconds": len(samples) / 16000,
            "speech_routing": {
                "route": "nemo-full-utterance",
                "language_prompt": self.preparation["language_prompt"],
                "effective_prompt": observed,
                "streaming_configuration": self.preparation["streaming_configuration"],
                "cache_execution": self.preparation["cache_execution"],
            },
        }
