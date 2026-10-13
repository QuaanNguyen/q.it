import importlib.metadata
import json
import math
import os
import queue
import threading
import time

from . import __version__
from .catalog import image_settings, prepared_inputs, required_text
from .media import audio_from_case, image_from_case, png_data_url


class Backend:
    def __init__(self, profile, threads=4, trust_publisher_code=False):
        import torch

        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("CPU threads must be between one and sixty-four")
        if profile["trust_publisher_code"] and not trust_publisher_code:
            raise ValueError("this pinned checkpoint requires --trust-publisher-code")
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.profile = dict(profile)
        self.threads = threads
        self.processor = None
        self.tokenizer = None
        self.model = self.load_model(
            trust_publisher_code and profile["trust_publisher_code"]
        )
        self.parameter_count = self.validate_loaded_model()

    def load_model(self, trust):
        mode = self.profile["implementation"]
        model_id = self.profile["model"]
        token = None if model_id.startswith("google/") else False
        revision = self.profile["revision"]
        code_options = (
            {"code_revision": self.profile["code_revision"]}
            if self.profile.get("code_revision")
            else {}
        )
        if mode == "embedding":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_id,
                revision=revision,
                device="cpu",
                trust_remote_code=trust,
                token=token,
                model_kwargs={
                    "torch_dtype": self.torch.float32,
                    "attn_implementation": "eager",
                    **code_options,
                },
                tokenizer_kwargs=code_options,
                config_kwargs=code_options,
            )
            model.float().eval()
            if model.get_sentence_embedding_dimension() != self.profile["dimensions"]:
                raise ValueError(
                    "loaded embedding dimension differs from the pinned profile"
                )
            self.profile["query_prefix"] = (
                model.prompts.get("query") or self.profile["query_prefix"]
            )
            self.profile["document_prefix"] = (
                model.prompts.get("document")
                or model.prompts.get("passage")
                or self.profile["document_prefix"]
            )
            return model
        if mode == "reranking":
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                model_id,
                revision=revision,
                device="cpu",
                trust_remote_code=trust,
                token=token,
                model_kwargs={
                    "torch_dtype": self.torch.float32,
                    "attn_implementation": "eager",
                },
            )
            model.model.float().eval()
            return model
        if mode == "diffusion":
            from diffusers import StableDiffusionPipeline

            model = StableDiffusionPipeline.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=self.torch.float32,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                token=token,
            )
            model.to("cpu")
            model.enable_attention_slicing()
            model.set_progress_bar_config(disable=True)
            return model
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            AutoTokenizer,
            BlipForConditionalGeneration,
        )

        options = {
            "revision": revision,
            "trust_remote_code": trust,
            "token": token,
            **code_options,
        }
        model_options = {
            **options,
            "torch_dtype": self.torch.float32,
            "attn_implementation": "eager",
        }
        if mode in ("chat", "completion"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
        else:
            self.processor = AutoProcessor.from_pretrained(model_id, **options)
            self.tokenizer = self.processor.tokenizer
            if mode == "image_chat":
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id, **model_options
                )
            elif mode == "blip":
                model = BlipForConditionalGeneration.from_pretrained(
                    model_id, **model_options
                )
            elif mode in ("florence", "git"):
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
            else:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, **model_options
                )
        return model.to("cpu").eval()

    def loaded_modules(self):
        mode = self.profile["implementation"]
        if mode == "diffusion":
            return [
                value
                for value in self.model.components.values()
                if isinstance(value, self.torch.nn.Module)
            ]
        if mode == "reranking":
            return [self.model.model]
        return [self.model]

    def validate_loaded_model(self):
        parameters = {}
        for module in self.loaded_modules():
            for parameter in module.parameters():
                if (
                    parameter.device.type != "cpu"
                    or parameter.dtype != self.torch.float32
                ):
                    raise ValueError(
                        "every loaded inference parameter must be CPU float32"
                    )
                parameters[id(parameter)] = parameter.numel()
        total = sum(parameters.values())
        if not 0 < total < 1_000_000_000:
            raise ValueError(
                f"loaded stack has {total} parameters and fails the under-one-billion limit"
            )
        return total

    def decoding_options(self):
        options = {"do_sample": False, "num_beams": 1}
        if self.profile["implementation"] == "florence":
            options["use_cache"] = False
        return options

    def metadata(self):
        packages = {}
        for name in (
            "torch",
            "transformers",
            "sentence-transformers",
            "diffusers",
            "numpy",
            "Pillow",
            "soundfile",
            "scipy",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        configuration = {
            "task": self.profile["task"],
            "device": "cpu",
            "dtype": "float32",
            "threads": self.threads,
            "parameter_count": self.parameter_count,
            "revision": self.profile["revision"],
            "implementation": self.profile["implementation"],
            "packages": packages,
            "pid": os.getpid(),
            "chat_template_kwargs": {"enable_thinking": False}
            if self.profile["implementation"] == "chat"
            else None,
            "decoding": self.decoding_options()
            if self.profile["task"]
            in ("text_generation", "image_to_text", "speech_to_text")
            else None,
            "profile": self.profile,
        }
        if self.profile["task"] == "text_to_image":
            configuration["scheduler"] = dict(self.model.scheduler.config)
            configuration["safety_checker"] = False
        if self.profile["implementation"] == "moonshine":
            configuration["minimum_audio_samples"] = self.minimum_audio_samples()
            configuration["short_audio_padding"] = "right_zero"
        if self.profile["task"] == "embedding":
            configuration["modules"] = [
                type(module).__name__ for module in self.model.children()
            ]
        return {
            "schema_version": 1,
            "version": __version__,
            "model": self.profile["model"],
            "configuration": configuration,
        }

    def validate_request(self, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("model") != self.profile["model"]
        ):
            raise ValueError("request model does not match the loaded checkpoint")
        if payload.get("task") != self.profile["task"]:
            raise ValueError("request task does not match the loaded model")
        case = payload.get("case")
        if not isinstance(case, dict):
            raise ValueError("request requires a benchmark case")
        limit = payload.get("max_output_tokens")
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("output token limit must be between one and 4096")
        task = self.profile["task"]
        if task in ("embedding", "reranking"):
            prepared_inputs(self.profile, case)
        elif task != "speech_to_text":
            prompt = required_text(case, "prompt")
            if self.profile["implementation"] == "florence" and prompt not in (
                "<CAPTION>",
                "<DETAILED_CAPTION>",
                "<MORE_DETAILED_CAPTION>",
                "<OCR>",
                "<OCR_WITH_REGION>",
            ):
                raise ValueError(
                    "Florence requires an explicit caption or OCR task token in the benchmark pack"
                )
        if task == "text_to_image":
            image_settings(case)

    def invoke(self, payload):
        self.validate_request(payload)
        case = payload["case"]
        task = self.profile["task"]
        with self.torch.inference_mode():
            if task == "embedding":
                inputs = prepared_inputs(self.profile, case)
                vectors = self.model.encode(
                    inputs,
                    batch_size=4,
                    normalize_embeddings=True,
                    prompt="",
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                yield {"type": "result", "embeddings": vectors.tolist()}
            elif task == "reranking":
                inputs = prepared_inputs(self.profile, case)
                scores = self.model.predict(
                    [(inputs[0], document) for document in inputs[1:]],
                    batch_size=4,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    activation_fn=self.torch.nn.Identity(),
                )
                values = scores.reshape(-1).tolist()
                if len(values) != len(case["documents"]) or not all(
                    math.isfinite(value) for value in values
                ):
                    raise ValueError(
                        "reranker must return one finite score per document"
                    )
                ranking = sorted(
                    range(len(values)), key=lambda index: (-values[index], index)
                )
                yield {"type": "result", "ranking": ranking, "scores": values}
            elif task == "text_to_image":
                settings = image_settings(case)
                result = self.model(
                    prompt=case["prompt"],
                    width=settings["width"],
                    height=settings["height"],
                    num_inference_steps=settings["steps"],
                    guidance_scale=settings["guidance_scale"],
                    generator=self.torch.Generator(device="cpu").manual_seed(
                        settings["seed"]
                    ),
                )
                if len(result.images) != 1 or result.images[0].size != (
                    settings["width"],
                    settings["height"],
                ):
                    raise ValueError(
                        "diffusion model returned an unexpected image shape"
                    )
                yield {"type": "result", "output_image": png_data_url(result.images[0])}
            elif task == "speech_to_text":
                yield self.transcribe(case, payload["max_output_tokens"])
            else:
                yield from self.generate(case, payload["max_output_tokens"])

    def minimum_audio_samples(self):
        encoder = self.model.get_encoder()
        minimum = 1
        for layer in (encoder.conv3, encoder.conv2, encoder.conv1):
            minimum = (minimum - 1) * layer.stride[0] + layer.kernel_size[0]
        return minimum

    def transcribe(self, case, max_tokens):
        import numpy as np

        rate = self.processor.feature_extractor.sampling_rate
        samples = audio_from_case(case, rate)
        duration = len(samples) / rate
        if self.profile["implementation"] == "moonshine":
            padding = max(0, self.minimum_audio_samples() - len(samples))
            samples = np.pad(samples, (0, padding))
        inputs = self.processor(samples, sampling_rate=rate, return_tensors="pt")
        options = {"max_new_tokens": max_tokens, **self.decoding_options()}
        if self.profile["implementation"] == "moonshine":
            options["max_new_tokens"] = min(
                max_tokens, max(1, math.ceil(duration * 6.5))
            )
        elif getattr(self.model.generation_config, "is_multilingual", False):
            options.update(language="en", task="transcribe")
        generated = self.model.generate(**inputs, **options)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return {
            "type": "result",
            "output_text": text,
            "audio_duration_seconds": duration,
        }

    def generation_inputs(self, case):
        prompt = case["prompt"]
        mode = self.profile["implementation"]
        if mode == "chat":
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if mode == "completion":
            return self.tokenizer(prompt, return_tensors="pt")
        image = image_from_case(case)
        if mode == "image_chat":
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            return self.processor(text=text, images=[image], return_tensors="pt")
        return self.processor(text=prompt, images=image, return_tensors="pt")

    def generate(self, case, max_tokens):
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        started = time.perf_counter_ns()
        inputs = self.generation_inputs(case)
        cancelled = threading.Event()
        result = queue.Queue()

        class TimedStreamer(TextIteratorStreamer):
            first_token_ns = None
            last_token_ns = None
            generated_token_count = 0

            def put(self, value):
                if not self.next_tokens_are_prompt:
                    now = time.perf_counter_ns()
                    if self.first_token_ns is None:
                        self.first_token_ns = now
                    self.last_token_ns = now
                    self.generated_token_count += value.numel()
                super().put(value)

        streamer = TimedStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600
        )

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        def run_generation():
            try:
                with self.torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        **self.decoding_options(),
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([Cancelled()]),
                    )
                result.put((output, None))
            except Exception as error:
                result.put((None, error))
                streamer.end()

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        try:
            for text in streamer:
                if text:
                    yield {"type": "delta", "text": text}
            worker.join(timeout=10)
            if worker.is_alive():
                raise TimeoutError("CPU generation did not terminate")
            output, error = result.get_nowait()
            if error:
                raise error
            input_tokens = inputs["input_ids"].shape[-1]
            encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)
            start = 1 if encoder_decoder else input_tokens
            if self.profile["implementation"] == "blip":
                start = input_tokens - 1
            tokens = output[0, start:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if self.profile["implementation"] == "florence":
                raw = self.tokenizer.decode(output[0], skip_special_tokens=False)
                parsed = self.processor.post_process_generation(
                    raw, task=case["prompt"], image_size=image_from_case(case).size
                )[case["prompt"]]
                text = parsed if isinstance(parsed, str) else json.dumps(parsed)
            duration = None
            first_token_latency = None
            if (
                streamer.generated_token_count == len(tokens)
                and streamer.first_token_ns is not None
            ):
                first_token_latency = streamer.first_token_ns - started
            if (
                streamer.generated_token_count == len(tokens)
                and len(tokens) >= 2
                and streamer.first_token_ns is not None
                and streamer.last_token_ns > streamer.first_token_ns
            ):
                duration = streamer.last_token_ns - streamer.first_token_ns
            yield {
                "type": "result",
                "output_text": text,
                "input_tokens": input_tokens,
                "output_tokens": len(tokens),
                "decode_duration_ns": duration,
                "first_token_latency_ns": first_token_latency,
            }
        finally:
            cancelled.set()
            worker.join(timeout=5)
