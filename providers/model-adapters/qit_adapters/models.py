def load_image(profile, directory, threads):
    import torch
    from .backend import audit_modules

    if torch.version.cuda is not None:
        raise ValueError("optional CPU adapters require a CPU-only PyTorch build")
    if profile["precision"] != "float32":
        raise ValueError("image routes require CPU float32")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    runtime = ImageRuntime(profile, directory, torch)
    runtime.audit = audit_modules(
        {"model": runtime.model}, runtime.extra_tensors, torch, "float32"
    )
    return runtime


class ImageRuntime:
    def __init__(self, profile, directory, torch):
        if profile["adapter"] != "lighton":
            raise ValueError(f"unsupported native image adapter {profile['adapter']}")
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.profile = profile
        self.torch = torch
        self.configuration = None
        self.extra_tensors = {}
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

    def invoke(self, case, max_tokens):
        from .media import image_from_case

        with self.torch.inference_mode():
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
                "output_text": self.processor.decode(tokens, skip_special_tokens=True),
                "output_tokens": len(tokens),
                "input_tokens": inputs["input_ids"].shape[1],
            }


def load_embedding(profile, directory, threads):
    import torch
    from .backend import audit_modules

    if torch.version.cuda is not None:
        raise ValueError("optional CPU adapters require a CPU-only PyTorch build")
    if profile["precision"] != "float32":
        raise ValueError("embedding routes require CPU float32")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    runtime = EmbeddingRuntime(profile, directory, torch)
    runtime.audit = audit_modules(
        {"model": runtime.model}, runtime.extra_tensors, torch, "float32"
    )
    return runtime


class EmbeddingRuntime:
    def __init__(self, profile, directory, torch):
        self.profile = profile
        self.torch = torch
        self.configuration = None
        self.extra_tensors = {}
        if profile["adapter"] == "ultrabert":
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
        else:
            raise ValueError(
                f"unsupported native embedding adapter {profile['adapter']}"
            )

    def invoke(self, case, max_tokens):
        query = case.get("input")
        documents = case.get("documents")
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(documents, list)
            or not documents
            or not all(
                isinstance(document, str) and document.strip() for document in documents
            )
        ):
            raise ValueError("embedding requires a query and nonempty document texts")
        return self.embed(case, query, documents)

    def embed(self, case, query, documents):
        for field in ("query_prefix", "document_prefix"):
            if case.get(field) is not None and case[field] != self.preparation[field]:
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
        with self.torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=4,
                normalize_embeddings=True,
                prompt="",
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return {"type": "result", "embeddings": embeddings.tolist()}
