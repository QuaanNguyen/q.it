from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import onnx
except ImportError:
    onnx = None

from qit_adapters.onnx_ocr import audit_graph


@unittest.skipIf(onnx is None, "install the optional ONNX audit test dependency")
class GraphAuditTests(unittest.TestCase):
    def test_full_graph_bound_includes_nested_constants_and_local_functions(self):
        helper = onnx.helper
        tensor = helper.make_tensor("weight", onnx.TensorProto.FLOAT, [2, 3], [1.0] * 6)
        nested = helper.make_graph(
            [helper.make_node("Constant", [], ["inner"], value=tensor)],
            "nested",
            [],
            [],
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If", ["condition"], ["out"], then_branch=nested, else_branch=nested
                )
            ],
            "root",
            [],
            [],
            [tensor],
        )
        model = helper.make_model(graph)
        function = helper.make_function(
            "local",
            "ConstantFunction",
            [],
            ["result"],
            [helper.make_node("Constant", [], ["result"], value=tensor)],
            [helper.make_opsetid("", 20)],
        )
        model.functions.append(function)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all.onnx"
            onnx.save(model, path)
            audit = audit_graph(path, onnx)
        self.assertEqual(audit["static_elements_upper_bound"], 24)
        self.assertEqual(len(audit["constants"]), 4)
        self.assertTrue(
            any(entry["scope"].startswith("function/") for entry in audit["constants"])
        )

    def test_external_data_reduced_precision_and_quantized_operators_fail_closed(self):
        helper = onnx.helper
        for mode in ("external", "reduced", "quantized"):
            tensor = helper.make_tensor("weight", onnx.TensorProto.FLOAT, [1], [1.0])
            nodes = []
            if mode == "external":
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensor.external_data.add(key="location", value="../outside.bin")
            elif mode == "reduced":
                tensor.data_type = onnx.TensorProto.FLOAT16
            else:
                nodes.append(helper.make_node("MatMulInteger", ["a", "b"], ["c"]))
            model = helper.make_model(
                helper.make_graph(nodes, "root", [], [], [tensor])
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.onnx"
                path.write_bytes(model.SerializeToString())
                with self.assertRaises(ValueError, msg=mode):
                    audit_graph(path, onnx)


if __name__ == "__main__":
    unittest.main()
