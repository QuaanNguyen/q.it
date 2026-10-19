from .media import image_from_case


class ModelRuntime:
    def __init__(self, profile, directory, torch):
        self.profile = profile
        self.torch = torch
        self.extra_tensors = {}
        self.configuration = None
        if profile["adapter"] == "lighton":
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            self.model = (
                Qwen3_5ForConditionalGeneration.from_pretrained(
                    str(directory),
                    local_files_only=True,
                    dtype=torch.float32,
                    attn_implementation="eager",
                )
                .to("cpu")
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(
                str(directory), local_files_only=True
            )
            self.preparation = {
                "mode": "ocr-transcription",
                "instruction": "",
                "enable_thinking": False,
                "do_sample": False,
                "image_longest_edge": 2048,
            }
        elif profile["adapter"] == "ultrabert":
            from sentence_transformers import SentenceTransformer

            self.model = (
                SentenceTransformer(
                    str(directory),
                    device="cpu",
                    local_files_only=True,
                    trust_remote_code=True,
                    model_kwargs={
                        "dtype": torch.float32,
                        "attn_implementation": "eager",
                    },
                )
                .float()
                .eval()
            )
            if self.model.get_sentence_embedding_dimension() != 768:
                raise ValueError(
                    "UltraBERT must return the native 768-dimensional embeddings"
                )
            self.preparation = {
                "query_prefix": "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
                "document_prefix": "",
                "normalization": "l2",
                "dimensions": 768,
                "pooling": "publisher CLS with prompt included",
                "max_seq_length": self.model.max_seq_length,
            }
        elif profile["adapter"] in ("uibk", "uibk-agent"):
            from pylate.models import ColBERT

            self.model = (
                ColBERT(
                    str(directory),
                    device="cpu",
                    local_files_only=True,
                    trust_remote_code=True,
                )
                .float()
                .eval()
            )
            self.retrieval = {
                "schema_version": 1,
                "method": "maxsim",
                "dimensions": 128,
                "tie_break": "document_order",
                "max_stored_scalars": 1_000_000,
            }
            self.preparation = {
                "query_prompt": self.model.prompts.get("query"),
                "document_prompt": self.model.prompts.get("document"),
                "query_prefix": self.model.query_prefix,
                "document_prefix": self.model.document_prefix,
                "query_length": self.model.query_length,
                "document_length": self.model.document_length,
                "scoring": "pylate native colbert_scores / torch",
                "pool_factor": 1,
                "raw_vectors": "all emitted vectors including dense slots",
            }
        else:
            raise ValueError(f"unsupported native model adapter {profile['adapter']}")

    def invoke(self, case, max_tokens):
        with self.torch.inference_mode():
            if self.profile["adapter"] == "lighton":
                if case.get("prompt") != "<OCR>":
                    raise ValueError(
                        "LightOnOCR accepts only the <OCR> transcription directive; chat, VQA, and grounding cases require separate integrations"
                    )
                image = image_from_case(case)
                image.thumbnail((2048, 2048))
                inputs = self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image", "image": image}]}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to("cpu")
                output = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False
                )
                tokens = output[0, inputs["input_ids"].shape[1] :]
                return {
                    "type": "result",
                    "output_text": self.processor.decode(
                        tokens, skip_special_tokens=True
                    ),
                    "output_tokens": len(tokens),
                    "input_tokens": inputs["input_ids"].shape[1],
                }
            query = case.get("input")
            documents = case.get("documents")
            if (
                not isinstance(query, str)
                or not query.strip()
                or not isinstance(documents, list)
                or not documents
                or not all(
                    isinstance(document, str) and document.strip()
                    for document in documents
                )
            ):
                raise ValueError(
                    "embedding requires a query and nonempty document texts"
                )
            if self.profile["adapter"] in ("uibk", "uibk-agent"):
                from pylate.scores import colbert_scores

                if self.profile["adapter"] == "uibk-agent" and (
                    not query.startswith("Reasoning: ") or "\n\nQuery: " not in query
                ):
                    raise ValueError(
                        "Agent UIBK requires a matching AgentIR reasoning-and-query pack"
                    )
                if (
                    case.get("query_prefix") is not None
                    or case.get("document_prefix") is not None
                ):
                    raise ValueError(
                        "UIBK role preparation is owned by the pinned publisher model"
                    )
                queries = self.model.encode(
                    [query],
                    is_query=True,
                    prompt_name="query",
                    batch_size=1,
                    convert_to_numpy=False,
                    show_progress_bar=False,
                    pool_factor=1,
                )
                encoded = self.model.encode(
                    documents,
                    is_query=False,
                    prompt_name="document",
                    batch_size=1,
                    convert_to_numpy=False,
                    show_progress_bar=False,
                    pool_factor=1,
                )
                query_vectors = self.torch.as_tensor(
                    queries[0], device="cpu", dtype=self.torch.float32
                )
                document_vectors = [
                    self.torch.as_tensor(value, device="cpu", dtype=self.torch.float32)
                    for value in encoded
                ]
                total = query_vectors.numel() + sum(
                    value.numel() for value in document_vectors
                )
                if total > 1_000_000:
                    raise ValueError(
                        "raw retrieval observations exceed one million scalars; use a smaller versioned retrieval case"
                    )
                scores = [
                    float(
                        colbert_scores(
                            query_vectors.unsqueeze(0),
                            value.unsqueeze(0),
                            backend="torch",
                        )[0, 0]
                    )
                    for value in document_vectors
                ]
                ranking = sorted(
                    range(len(documents)), key=lambda index: (-scores[index], index)
                )
                return {
                    "type": "result",
                    "retrieval": {
                        "schema_version": 1,
                        "method": "maxsim",
                        "query": query_vectors.tolist(),
                        "documents": [value.tolist() for value in document_vectors],
                        "scores": scores,
                        "ranking": ranking,
                        "tie_break": "document_order",
                    },
                }
            for field in ("query_prefix", "document_prefix"):
                if (
                    case.get(field) is not None
                    and case[field] != self.preparation[field]
                ):
                    raise ValueError(f"UltraBERT requires its publisher {field}")
            texts = [self.preparation["query_prefix"] + query] + [
                self.preparation["document_prefix"] + document for document in documents
            ]
            for text in texts:
                if (
                    len(self.model.tokenizer(text, truncation=False)["input_ids"])
                    > self.model.max_seq_length
                ):
                    raise ValueError("UltraBERT input would be truncated")
            embeddings = self.model.encode(
                texts,
                batch_size=4,
                normalize_embeddings=True,
                prompt="",
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return {"type": "result", "embeddings": embeddings.tolist()}
