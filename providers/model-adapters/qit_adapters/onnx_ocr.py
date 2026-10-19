import hashlib
import math

from .media import image_from_case


def audit_graph(path, onnx):
    model = onnx.load(str(path), load_external_data=False)
    entries = []
    operators = []

    def tensor(value, scope, sparse=False):
        if value.data_location == onnx.TensorProto.EXTERNAL or value.external_data:
            raise ValueError(f"external ONNX tensor data is not admitted: {scope}")
        if value.data_type not in {
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.INT64,
            onnx.TensorProto.INT32,
            onnx.TensorProto.BOOL,
        }:
            raise ValueError(
                f"unsupported ONNX constant precision: {scope} type {value.data_type}"
            )
        if any(dimension < 0 for dimension in value.dims):
            raise ValueError(f"invalid ONNX constant dimensions: {scope}")
        entries.append(
            {
                "scope": scope,
                "shape": list(value.dims),
                "elements": math.prod(value.dims),
                "dtype": onnx.TensorProto.DataType.Name(value.data_type),
                "sha256": hashlib.sha256(value.SerializeToString()).hexdigest(),
                "classification": "static constant; learned status not encoded by ONNX",
                "sparse_storage": sparse,
            }
        )

    def sparse(value, scope):
        entries.append(
            {
                "scope": scope,
                "shape": list(value.dims),
                "elements": math.prod(value.dims),
                "classification": "full logical sparse shape; conservative bound including zero positions",
            }
        )
        tensor(value.values, scope + "/values", True)
        tensor(value.indices, scope + "/indices", True)

    def attributes(values, scope):
        for index, attribute in enumerate(values):
            current = f"{scope}/{index}:{attribute.name}"
            if attribute.ref_attr_name:
                raise ValueError(
                    f"unresolved ONNX function attribute reference: {current}"
                )
            kind = attribute.type
            if kind == onnx.AttributeProto.TENSOR:
                tensor(attribute.t, current)
            elif kind == onnx.AttributeProto.TENSORS:
                for position, value in enumerate(attribute.tensors):
                    tensor(value, f"{current}/{position}")
            elif kind == onnx.AttributeProto.SPARSE_TENSOR:
                sparse(attribute.sparse_tensor, current)
            elif kind == onnx.AttributeProto.SPARSE_TENSORS:
                for position, value in enumerate(attribute.sparse_tensors):
                    sparse(value, f"{current}/{position}")
            elif kind == onnx.AttributeProto.GRAPH:
                graph(attribute.g, current)
            elif kind == onnx.AttributeProto.GRAPHS:
                for position, value in enumerate(attribute.graphs):
                    graph(value, f"{current}/{position}")
            elif kind in {
                onnx.AttributeProto.FLOAT,
                onnx.AttributeProto.INT,
                onnx.AttributeProto.FLOATS,
                onnx.AttributeProto.INTS,
            }:
                count = (
                    len(attribute.floats)
                    if kind == onnx.AttributeProto.FLOATS
                    else len(attribute.ints)
                    if kind == onnx.AttributeProto.INTS
                    else 1
                )
                entries.append(
                    {
                        "scope": current,
                        "elements": count,
                        "classification": "scalar or list attribute; conservative static bound",
                    }
                )

    def nodes(values, scope):
        for index, node in enumerate(values):
            current = f"{scope}/{index}:{node.name or node.op_type}"
            if (
                "Quantiz" in node.op_type
                or "Dequantiz" in node.op_type
                or node.op_type.startswith("QLinear")
                or node.op_type in {"MatMulInteger", "ConvInteger"}
            ):
                raise ValueError(
                    f"quantized ONNX operators are not in the float32 route: {current}"
                )
            operators.append(
                {"scope": current, "domain": node.domain, "operator": node.op_type}
            )
            attributes(node.attribute, current)

    def graph(value, scope):
        for index, value_tensor in enumerate(value.initializer):
            tensor(value_tensor, f"{scope}/initializer/{index}:{value_tensor.name}")
        for index, value_sparse in enumerate(value.sparse_initializer):
            sparse(value_sparse, f"{scope}/sparse/{index}")
        nodes(value.node, scope)

    graph(model.graph, "graph")
    for index, function in enumerate(model.functions):
        scope = f"function/{index}:{function.domain}:{function.name}"
        attributes(function.attribute_proto, scope)
        nodes(function.node, scope)
    return {
        "file": path.name,
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "static_elements_upper_bound": sum(entry["elements"] for entry in entries),
        "constants": entries,
        "operators": operators,
    }


class ObservedSession:
    def __init__(self, session, name, calls):
        self.session = session
        self.name = name
        self.calls = calls

    def run(self, *args, **kwargs):
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        return self.session.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.session, name)


class OnnxOcrRuntime:
    def __init__(self, profile, directory, threads):
        import numpy
        import onnx
        from indra_ocr.pipeline import OrtOCR

        expected = {name for name in profile["files"] if name.endswith(".onnx")}
        if {path.name for path in directory.glob("*.onnx")} != expected:
            raise ValueError(
                "Indra requires exactly the pinned full graph set; unmanifested or reduced-precision graphs are forbidden"
            )
        graphs = [audit_graph(directory / name, onnx) for name in sorted(expected)]
        bound = sum(graph["static_elements_upper_bound"] for graph in graphs)
        if not 0 < bound < 1_000_000_000:
            raise ValueError(
                "full ONNX static tensor bound does not establish the sub-billion limit"
            )
        self.model = OrtOCR(
            directory,
            threads=threads,
            rec_int8=False,
            providers=["CPUExecutionProvider"],
            math=True,
        )
        sessions = {
            name: getattr(self.model, name)
            for name in ("det", "rec", "order", "orient", "script", "rec_cjk")
        }
        sessions.update(
            {
                "math." + name: getattr(self.model.math, name)
                for name in ("cls", "enc", "init", "step")
            }
        )
        for name, session in sessions.items():
            if session is None or session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError(
                    f"Indra full-profile session is absent or is not CPU-only: {name}"
                )
        if self.model.mask is not None:
            raise ValueError("unmanifested Indra mask graph is forbidden")
        self.stage_calls = {}
        for name, session in sessions.items():
            parent, attribute = (
                (self.model.math, name.removeprefix("math."))
                if name.startswith("math.")
                else (self.model, name)
            )
            setattr(parent, attribute, ObservedSession(session, name, self.stage_calls))
        self.numpy = numpy
        self.configuration = None
        self.audit = {
            "parameter_count": None,
            "parameter_count_upper_bound": bound,
            "parameter_count_basis": "onnx-full-static-constant-upper-bound-v1",
            "exact_learned_count": None,
            "graphs": graphs,
            "bound_scope": "all ten loaded graphs including duplicated exports, fixed constants and attributes; no deduplication by equal bytes",
        }
        self.preparation = {
            "route": "publisher OrtOCR full profile including math",
            "input": "bounded RGB image converted to grayscale; no PDF text-layer extraction",
            "orientation": True,
            "math": True,
            "rec_int8": False,
            "providers": ["CPUExecutionProvider"],
            "sessions": list(sessions),
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "cpu_memory_arena": False,
            "license": {
                "model": "https://huggingface.co/datalab-to/indra-ocr",
                "original": f"https://huggingface.co/{profile['model']}/blob/{profile['revision']}/LICENSE",
                "scope": "owner-confirmed lab research/publication; no weight redistribution",
                "output_obligations": "retain attribution, model and original license links, disclose changes, and preserve applicable output share-alike obligations",
                "changes": "q.it local-only serving and measurement wrapper; pinned model graphs unchanged",
            },
        }

    def invoke(self, case, max_tokens):
        if case.get("prompt", "").strip() != "<OCR>":
            raise ValueError(
                "Indra is an OCR route and requires the pack's <OCR> directive"
            )
        image = image_from_case(case).convert("L")
        self.stage_calls.clear()
        page = self.model(self.numpy.asarray(image), turn=True)
        if not isinstance(page.text, str):
            raise ValueError("Indra did not return native OCR text")
        return {
            "type": "result",
            "output_text": page.text,
            "output_metadata": {
                "schema_version": 1,
                "route": "indra-full-ocr",
                "stage_calls": dict(self.stage_calls),
            },
        }
