import argparse
import base64
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys


def source_module(path):
    name = "native_reference_" + hashlib.sha256(path.read_bytes()).hexdigest()
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def media(case, kind):
    value = case["media"]
    if not value.startswith("data:" + kind):
        raise ValueError("native reference requires retained inline media")
    return base64.b64decode(value.split(",", 1)[1], validate=True)


def image(case):
    from PIL import Image

    return Image.open(io.BytesIO(media(case, "image/"))).convert("RGB")


def audio(case):
    import soundfile

    with soundfile.SoundFile(io.BytesIO(media(case, "audio/"))) as source:
        if source.samplerate != 16000:
            raise ValueError(
                "this independent spoken reference requires original 16 kHz audio"
            )
        return source.read(dtype="float32", always_2d=True).mean(axis=1)


def agrees(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            agrees(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            agrees(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def initialize(adapter, directory, configuration, torch):
    if adapter == "gliclass":
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        model = (
            GLiClassModel.from_pretrained(str(directory), local_files_only=True)
            .to("cpu")
            .float()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        return ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="single-label",
            device="cpu",
            max_length=2048,
            progress_bar=False,
        )
    if adapter == "gliner2":
        from gliner2 import AutoExtractor

        return (
            AutoExtractor.from_pretrained(str(directory), local_files_only=True)
            .to("cpu")
            .float()
            .eval()
        )
    if adapter == "intern":
        module = source_module(directory / "inference.py")
        return module.DecisionEngine(
            checkpoint=str(directory),
            device="cpu",
            dtype="float32",
            attn_implementation="eager",
        )
    if adapter == "frida":
        from frida_decisions import Judge

        return Judge.from_pretrained(
            str(directory),
            device="cpu",
            dtype=torch.float32,
            state_max=512,
            state_cache_mb=0,
        )
    if adapter in {"liquid", "typed-gemma", "d3"}:
        from transformers import AutoModel, AutoTokenizer

        options = (
            {"device": "cpu", "dtype": torch.bfloat16, "verify": "full"}
            if adapter == "d3"
            else {"dtype": torch.float32}
        )
        model = (
            AutoModel.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=True, **options
            )
            .to("cpu")
            .eval()
        )
        if adapter != "d3":
            model.float()
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        if adapter == "liquid":
            model.tokenizer = tokenizer
        return model, tokenizer
    if adapter == "lighton":
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration.from_pretrained(
            str(directory),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cpu").eval(), AutoProcessor.from_pretrained(
            str(directory), local_files_only=True
        )
    if adapter == "ultrabert":
        from sentence_transformers import SentenceTransformer

        return (
            SentenceTransformer(
                str(directory),
                device="cpu",
                local_files_only=True,
                trust_remote_code=True,
                model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
            )
            .float()
            .eval()
        )
    if adapter in {"photon", "redux"}:
        from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

        return load_parakeet_tdt(
            str(directory), device="cpu", dtype=torch.float32, local_files_only=True
        )
    if adapter == "indra":
        from indra_ocr.pipeline import OrtOCR

        return OrtOCR(
            directory,
            threads=configuration["threads"],
            rec_int8=False,
            providers=["CPUExecutionProvider"],
            math=True,
        )
    if adapter == "nemotron":
        import nemo.collections.asr as asr
        from omegaconf import OmegaConf

        model = (
            asr.models.ASRModel.restore_from(
                str(directory / "nemotron-3.5-asr-streaming-0.6b.nemo"),
                map_location=torch.device("cpu"),
                strict=True,
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        model.change_decoding_strategy(
            OmegaConf.create(configuration["preparation"]["decoder"]), verbose=False
        )
        return model
    raise ValueError("no independent publisher reference for " + adapter)


def decision_reference(adapter, native, case):
    decision = case["decision"]
    candidates = decision["candidates"]
    question = {
        "choice": {
            "type": "choice",
            "instructions": decision["question"],
            "criteria": {entry["id"]: entry["text"] for entry in candidates},
        }
    }
    if adapter == "gliclass":
        scores = native(
            decision["state"],
            [entry["text"] for entry in candidates],
            prompt=decision["question"],
            classification_type="single-label",
            return_hierarchical=True,
        )[0]
        probabilities = {entry["id"]: scores[entry["text"]] for entry in candidates}
        selected = max(probabilities, key=probabilities.__getitem__)
    elif adapter == "gliner2":
        return native.classify_text(
            decision["state"],
            {
                "choice": {
                    "labels": {entry["id"]: entry["text"] for entry in candidates},
                    "multi_label": False,
                    "prompt": decision["question"],
                }
            },
            include_confidence=True,
            max_len=None,
        )
    else:
        request = {"state": decision["state"], "questions": question}
        if adapter == "intern":
            answer = native.predict(request)["answers"]["choice"]
        elif adapter == "frida":
            answer = native.judge(request)["answers"]["choice"]
        elif adapter == "typed-gemma":
            answer = native[0].decide(decision["state"], question, tokenizer=native[1])[
                "choice"
            ]
            probabilities, selected = answer.probabilities, answer.label
            return {
                "candidates": [
                    {"id": entry["id"], "probability": probabilities[entry["id"]]}
                    for entry in candidates
                ],
                "selected_id": selected,
            }
        else:
            options = {"images": [image(case)]} if case.get("media") else {}
            answer = native[0].system_one(
                state=decision["state"], questions=question, **options
            )["answers"]["choice"]
        probabilities = answer["probabilities"]
        selected = answer.get("choice", answer.get("decision"))
    return {
        "candidates": [
            {"id": entry["id"], "probability": probabilities[entry["id"]]}
            for entry in candidates
        ],
        "selected_id": selected,
    }


def reference(adapter, native, request, configuration, torch):
    case = request["case"]
    if request["task"] == "decision_scoring":
        return decision_reference(adapter, native, case)
    if adapter == "lighton":
        model, processor = native
        picture = image(case)
        picture.thumbnail((2048, 2048))
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": picture}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cpu")
        generated = model.generate(
            **inputs, max_new_tokens=request["max_output_tokens"], do_sample=False
        )
        return processor.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    if adapter == "ultrabert":
        queries = native.encode(
            [case["input"]],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        documents = native.encode(
            case["documents"],
            prompt_name="document",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return queries.tolist() + documents.tolist()
    if adapter in {"photon", "redux"}:
        from kestrel.models.parakeet_tdt.features import parakeet_features

        features, mask = parakeet_features(audio(case))
        generated = native.model.generate(features, mask, max_tokens=None)
        return native.tokenizer.decode(
            generated.sequences[0, : int(generated.lengths[0])].tolist()
        )
    if adapter == "indra":
        import numpy

        return native(numpy.asarray(image(case).convert("L")), turn=True).text
    if adapter == "nemotron":
        return native.transcribe(
            audio=[audio(case)],
            batch_size=1,
            return_hypotheses=True,
            num_workers=0,
            verbose=False,
            timestamps=False,
            target_lang="en-US",
        )[0].text
    raise ValueError("unsupported native comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--allow-reviewed-code", action="store_true")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    if (evidence / "publisher-parity.json").exists() or (
        evidence / "publisher-native.jsonl"
    ).exists():
        parser.error(
            "native parity evidence already exists; use a fresh evidence directory"
        )
    directory = Path(args.model_path).resolve()
    report = {"status": "failed", "adapter": args.adapter, "comparisons": []}
    try:
        if (
            args.adapter
            in {
                "intern",
                "liquid",
                "typed-gemma",
                "d3",
                "ultrabert",
            }
            and not args.allow_reviewed_code
        ):
            raise ValueError(
                "the independent reference requires explicit --allow-reviewed-code for pinned publisher source"
            )
        configuration = json.loads((evidence / "metadata.json").read_text())[
            "configuration"
        ]
        for name, artifact in configuration["artifacts"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory) or path.is_symlink():
                raise ValueError(
                    "native reference artifact escapes the immutable directory"
                )
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if (
                path.stat().st_size != artifact["bytes"]
                or digest.hexdigest() != artifact["sha256"]
            ):
                raise ValueError("native reference artifact differs: " + name)
        for name in (
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY",
            "DO_NOT_TRACK",
        ):
            os.environ[name] = "1"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        def offline(event, arguments):
            if event == "socket.getaddrinfo" or (
                event == "socket.connect" and isinstance(arguments[1], tuple)
            ):
                raise RuntimeError(
                    "independent native reference forbids network access"
                )

        sys.addaudithook(offline)
        if args.adapter == "indra":
            torch = None
        else:
            import torch

            if torch.version.cuda is not None:
                raise ValueError("native reference requires CPU-only Torch")
            torch.set_num_threads(configuration["threads"])
            torch.set_num_interop_threads(1)
        native = initialize(args.adapter, directory, configuration, torch)
        import contextlib

        with (
            (evidence / "publisher-native.jsonl").open("x") as output,
            torch.inference_mode() if torch else contextlib.nullcontext(),
        ):
            for line in (evidence / "raw-probes.jsonl").read_text().splitlines():
                probe = json.loads(line)
                expected = next(
                    event for event in probe["events"] if event.get("type") == "result"
                )
                observed = reference(
                    args.adapter, native, probe["request"], configuration, torch
                )
                if args.adapter == "gliner2":
                    expected = expected["decision"]["provenance"]["native"][
                        "public_result"
                    ]
                    scope = "publisher public classification object"
                elif probe["request"]["task"] == "decision_scoring":
                    expected = {
                        key: expected["decision"][key]
                        for key in ("candidates", "selected_id")
                    }
                    scope = "complete native candidate distribution and choice"
                elif args.adapter == "ultrabert":
                    expected = expected["embeddings"]
                    scope = "native normalized dense vectors"
                else:
                    expected = expected["output_text"]
                    scope = "complete native transcription"
                matched = agrees(expected, observed)
                output.write(
                    json.dumps(
                        {
                            "case_id": probe["request"]["case"]["id"],
                            "scope": scope,
                            "native": observed,
                            "matches": matched,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                output.flush()
                report["comparisons"].append(
                    {
                        "case_id": probe["request"]["case"]["id"],
                        "matches": matched,
                        "scope": scope,
                    }
                )
                if not matched:
                    raise ValueError(
                        "publisher-native result differs from the installed provider"
                    )
        report["status"] = "publisher-parity-passed"
    except Exception as error:
        report["error"] = str(error)
    (evidence / "publisher-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "publisher-parity-passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
