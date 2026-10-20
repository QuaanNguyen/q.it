import copy
import math

from .artifacts import fingerprint


def distribution(candidates, probabilities, selected):
    identifiers = [candidate["id"] for candidate in candidates]
    if not isinstance(probabilities, dict) or set(probabilities) != set(identifiers):
        raise ValueError(
            "native result did not return one probability for every candidate"
        )
    values = [probabilities[key] for key in identifiers]
    if any(
        (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (not 0 <= value <= 1)
            for value in values
        )
    ):
        raise ValueError(
            "native candidate probabilities must be finite and between zero and one"
        )
    if abs(sum(values) - 1) > 1e-06:
        raise ValueError(
            "native probabilities do not sum to one; q.it does not renormalize them"
        )
    if selected not in identifiers or probabilities[selected] != max(values):
        raise ValueError("native selection does not have maximal probability")
    return [{"id": key, "probability": probabilities[key]} for key in identifiers]


def load_decision(profile, directory, threads):
    import torch

    from .backend import audit_modules

    if torch.version.cuda is not None:
        raise ValueError("optional CPU adapters require a CPU-only PyTorch build")
    if profile["precision"] != "float32":
        raise ValueError("these classifier routes require CPU float32")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    runtime = DecisionRuntime(profile, directory, torch)
    runtime.audit = audit_modules(
        {"model": runtime.model}, runtime.extra_tensors, torch, "float32"
    )
    return runtime


class DecisionRuntime:
    def __init__(self, profile, directory, torch):
        from transformers import AutoTokenizer

        self.profile = profile
        self.directory = str(directory)
        self.torch = torch
        adapter = profile["adapter"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory, local_files_only=True
        )
        self.extra_tensors = {}
        if adapter == "gliclass":
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline

            self.model = (
                GLiClassModel.from_pretrained(self.directory, local_files_only=True)
                .to("cpu")
                .float()
                .eval()
            )
            self.pipeline = ZeroShotClassificationPipeline(
                self.model,
                self.tokenizer,
                classification_type="single-label",
                device="cpu",
                max_length=2048,
                progress_bar=False,
            )
        elif adapter == "gliner2":
            from gliner2 import AutoExtractor

            self.model = (
                AutoExtractor.from_pretrained(self.directory, local_files_only=True)
                .to("cpu")
                .float()
                .eval()
            )
        else:
            raise ValueError(f"unsupported classifier adapter {adapter}")
        self.configuration = {
            "schema_version": 1,
            "inputs": ["text"],
            "calibration": f"publisher-{adapter}-{profile['revision']}",
            "selection_rule": "publisher-native-argmax",
            "preprocessing": f"publisher-{adapter}-single-choice-no-truncation-v1",
        }

    def invoke(self, case):
        decision = case["decision"]
        candidates = decision["candidates"]
        state = decision["state"]
        adapter = self.profile["adapter"]
        prepared = None
        native = {}
        with self.torch.inference_mode():
            if adapter == "gliclass":
                labels = [candidate["text"] for candidate in candidates]
                if len(set(labels)) != len(labels):
                    raise ValueError("GLiClass requires distinct candidate texts")
                pipe = self.pipeline.pipe
                prepared = pipe.prepare_input(
                    state, labels, [], pipe._format_prompt(decision["question"], 0)
                )
                encoded = self.tokenizer(prepared, truncation=False)["input_ids"]
                if len(encoded) > pipe.max_length:
                    raise ValueError("GLiClass input would be truncated")
                scores = self.pipeline(
                    state,
                    labels,
                    prompt=decision["question"],
                    classification_type="single-label",
                    return_hierarchical=True,
                )[0]
                probabilities = {
                    candidate["id"]: scores[candidate["text"]]
                    for candidate in candidates
                }
                selected = max(probabilities, key=probabilities.__getitem__)
                native = {
                    "input_tokens": len(encoded),
                    "tie_rule": "first-maximal-in-candidate-order",
                }
            elif adapter == "gliner2":
                captured = []
                original = getattr(
                    self.model, "_compute_classification_probabilities", None
                )
                if not callable(original):
                    raise ValueError(
                        "GLiNER2 native full-probability observer is unavailable"
                    )
                tasks = {
                    "choice": {
                        "labels": {
                            candidate["id"]: candidate["text"]
                            for candidate in candidates
                        },
                        "multi_label": False,
                        "prompt": decision["question"],
                    }
                }
                schema = self.model._classification_schema(tasks)
                schema_dicts, _ = self.model._build_schema_dicts_and_metadata([schema])
                record = self.model.processor._transform_record(
                    {"text": state, "schema": copy.deepcopy(schema_dicts[0])},
                    max_len=None,
                )
                prepared = {"input_ids": record.input_ids, "schema": schema_dicts[0]}

                def observe(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if result is not None:
                        name, configuration, values = result
                        captured.append(
                            (
                                name,
                                list(configuration["labels"]),
                                values.detach().cpu().tolist(),
                            )
                        )
                    return result

                self.model._compute_classification_probabilities = observe
                try:
                    public_result = self.model.classify_text(
                        state, tasks, include_confidence=True, max_len=None
                    )
                finally:
                    self.model._compute_classification_probabilities = original
                if len(captured) != 1 or captured[0][0] != "choice":
                    raise ValueError(
                        "GLiNER2 did not expose exactly one complete native candidate distribution"
                    )
                _, labels, values = captured[0]
                if labels != [candidate["id"] for candidate in candidates] or len(
                    values
                ) != len(labels):
                    raise ValueError(
                        "GLiNER2 native label order or probability count differs from the requested candidates"
                    )
                probabilities = dict(zip(labels, values, strict=True))
                selected = max(probabilities, key=probabilities.__getitem__)
                native = {
                    "public_result": public_result,
                    "labels": labels,
                    "probabilities": values,
                    "observer": "unmodified native _compute_classification_probabilities return value",
                }
            else:
                raise ValueError(f"unsupported classifier adapter {adapter}")
        ordered = distribution(candidates, probabilities, selected)
        return {
            "type": "result",
            "decision": {
                "schema_version": 1,
                "candidates": ordered,
                "selected_id": selected,
                "provenance": {
                    key: self.configuration[key]
                    for key in ("calibration", "selection_rule", "preprocessing")
                }
                | {
                    "input_digest": fingerprint(prepared),
                    "input_truncated": False,
                    "native": native,
                },
            },
        }
