import copy
import importlib
import importlib.util
import math
import sys

from .artifacts import fingerprint


def distribution(candidates, probabilities, selected):
    identifiers = [candidate["id"] for candidate in candidates]
    if not isinstance(probabilities, dict) or set(probabilities) != set(identifiers):
        raise ValueError(
            "native result did not return one probability for every candidate"
        )
    values = [probabilities[key] for key in identifiers]
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError(
            "native candidate probabilities must be finite and between zero and one"
        )
    if abs(sum(values) - 1) > 0.000001:
        raise ValueError(
            "native probabilities do not sum to one; q.it does not renormalize them"
        )
    if selected not in identifiers or probabilities[selected] != max(values):
        raise ValueError("native selection does not have maximal probability")
    return [{"id": key, "probability": probabilities[key]} for key in identifiers]


def load_module(path, identity):
    name = "qit_publisher_" + identity[:16]
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError("publisher module cannot be imported")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class DecisionRuntime:
    def __init__(self, profile, directory, torch):
        from transformers import AutoModel, AutoTokenizer

        self.profile = profile
        self.directory = str(directory)
        self.torch = torch
        adapter = profile["adapter"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory, local_files_only=True
        )
        self.extra_tensors = {}
        if adapter == "intern":
            self.publisher = load_module(
                directory / "inference.py", profile["revision"]
            )
            self.engine = self.publisher.DecisionEngine(
                checkpoint=self.directory,
                device="cpu",
                dtype="float32",
                attn_implementation="eager",
            )
            self.model = self.engine.backend.model
        elif adapter == "frida":
            from frida_decisions import Judge

            self.engine = Judge.from_pretrained(
                self.directory,
                device="cpu",
                dtype=torch.float32,
                state_max=512,
                state_cache_mb=0,
            )
            self.model = self.engine.model
        elif adapter == "gliclass":
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
        elif adapter in ("liquid", "typed-gemma", "d3"):
            options = {"local_files_only": True, "trust_remote_code": True}
            if adapter == "d3":
                options.update(
                    {"device": "cpu", "dtype": torch.bfloat16, "verify": "full"}
                )
            else:
                options["dtype"] = torch.float32
            self.model = (
                AutoModel.from_pretrained(self.directory, **options).to("cpu").eval()
            )
            if adapter == "d3":
                self.extra_tensors["readout"] = self.model.runtime.readout
            else:
                self.model.float()
            if adapter == "liquid":
                self.model.tokenizer = self.tokenizer
            self.publisher = importlib.import_module(type(self.model).__module__)
        else:
            raise ValueError(f"unsupported decision runtime {adapter}")
        self.configuration = {
            "schema_version": 1,
            "inputs": ["text", "image"] if adapter == "d3" else ["text"],
            "calibration": f"publisher-{adapter}-{profile['revision']}",
            "selection_rule": "publisher-native-argmax",
            "preprocessing": f"publisher-{adapter}-single-choice-no-truncation-v1",
        }

    def invoke(self, case):
        decision = case["decision"]
        candidates = decision["candidates"]
        questions = {
            "choice": {
                "type": "choice",
                "instructions": decision["question"],
                "criteria": {
                    candidate["id"]: candidate["text"] for candidate in candidates
                },
            }
        }
        state = decision["state"]
        image = case.get("media")
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
            elif adapter == "frida":
                if not any("\u0400" <= character <= "\u04ff" for character in state):
                    raise ValueError("FRIDA requires a Russian decision pack")
                request = {"state": state, "questions": questions}
                _, native_candidates, tokenized = self.engine.compile(request)
                for candidate in native_candidates:
                    if (
                        self.engine.text.count(candidate.state) > 512
                        or self.engine.text.count(candidate.instruction)
                        > self.engine.config.instruction_max_tokens
                        or self.engine.text.count(candidate.text)
                        > self.engine.config.option_max_tokens - 1
                    ):
                        raise ValueError("FRIDA input would be truncated")
                prepared = {
                    "state": tokenized.state,
                    "questions": tokenized.questions,
                    "options": tokenized.options,
                }
                result = self.engine.judge(request)
                if result["usage"].get("state_truncated"):
                    raise ValueError("FRIDA reported a truncated state")
                native = result["answers"]["choice"]
                probabilities = native["probabilities"]
                selected = native.get("choice", native.get("decision"))
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
            elif adapter == "intern":
                request = {"state": state, "questions": questions}
                if image:
                    raise ValueError(
                        "Intern image execution awaits installed native processor qualification"
                    )
                compiled, inputs, _ = self.engine.backend.encode(request)
                prepared = {
                    "messages": compiled.messages,
                    "input_ids": inputs["input_ids"].tolist(),
                }
                native = self.engine.predict(request)["answers"]["choice"]
                probabilities = native["probabilities"]
                selected = native.get("decision", native.get("choice"))
            elif adapter == "typed-gemma":
                inputs, _, texts = self.model._build_inputs(
                    [("", state, questions, None)], self.tokenizer
                )
                limit = int(self.model.config.max_seq_length or 32768)
                for text in texts:
                    prompt = (self.model.config.prompt or "") + text
                    if (
                        len(self.tokenizer(prompt, truncation=False)["input_ids"])
                        > limit
                    ):
                        raise ValueError("typed decision input would be truncated")
                prepared = {"texts": texts, "input_ids": inputs["input_ids"].tolist()}
                answer = self.model.decide(state, questions, tokenizer=self.tokenizer)[
                    "choice"
                ]
                native = answer.to_dict()
                probabilities = answer.probabilities
                selected = answer.label
            elif adapter == "liquid":
                if image:
                    raise ValueError(
                        "Liquid image execution awaits installed native processor qualification"
                    )
                module = importlib.import_module(
                    type(self.model).__module__.rsplit(".", 1)[0] + ".prompt"
                )
                question = module.as_question(questions["choice"])
                limit = int(self.model.config.max_length)
                budget = max(96, min(len(candidates) * 24 + 32, limit // 2))
                per_option = max(2, (budget - 3 * len(candidates)) // len(candidates))

                def encode(text):
                    return self.tokenizer(
                        module.escape(text), add_special_tokens=False
                    )["input_ids"]

                if len(encode(question.instructions)) + 1 > max(16, budget):
                    raise ValueError("Liquid question would be truncated")
                if any(
                    len(encode(" " + text)) > per_option
                    for text in module.render_options(question)
                ):
                    raise ValueError("Liquid candidate text would be truncated")
                ids, markers = module.encode(self.tokenizer, state, question, limit)
                state_size = len(encode(module.serialize(state)))
                empty_ids, _ = module.encode(self.tokenizer, "", question, limit)
                if state_size + len(empty_ids) > limit:
                    raise ValueError("Liquid state would be truncated")
                prepared = {"input_ids": ids, "markers": markers}
                native = self.model.system_one(state=state, questions=questions)[
                    "answers"
                ]["choice"]
                probabilities = native["probabilities"]
                selected = native["choice"]
            else:
                images = [image] if image else None
                prepared_native = self.model.runtime.prepare(
                    state, questions, images=images
                )
                if prepared_native.errors:
                    raise ValueError(
                        f"native input preparation failed: {prepared_native.errors}"
                    )
                prepared = {
                    "sequences": prepared_native.sequences,
                    "texts": prepared_native.texts,
                    "lengths": prepared_native.lengths,
                    "media": image,
                }
                probabilities_by_question, tokens = self.model.runtime.run(
                    prepared_native
                )
                result = self.model.runtime.respond(
                    prepared_native, probabilities_by_question, tokens
                )
                native = result["answers"]["choice"]
                probabilities = native["probabilities"]
                selected = native["choice"]
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
