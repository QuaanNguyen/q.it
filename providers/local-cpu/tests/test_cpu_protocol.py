import http.client
import json
import sys
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qit_cpu.backend import Backend
from qit_cpu.catalog import image_settings, prepared_inputs, profile_for, profiles
from qit_cpu.server import CpuServer


class Fixture:
    def metadata(self):
        return {"model": "fixture", "configuration": {"device": "cpu"}}

    def validate_request(self, payload):
        if payload.get("model") != "fixture":
            raise ValueError("model mismatch")

    def invoke(self, payload):
        yield {"type": "delta", "text": "hello"}
        if payload.get("fail"):
            raise ValueError("inference failed")
        yield {"type": "result", "output_text": "hello", "output_tokens": 1}


class Protocol(unittest.TestCase):
    def setUp(self):
        self.server = CpuServer(("127.0.0.1", 0), Fixture(), "fixture-key")
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def request(self, path, payload=None, token="fixture-key"):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        headers = {"Authorization": "Bearer " + token}
        connection.request(
            "GET" if payload is None else "POST",
            path,
            body=None if payload is None else json.dumps(payload),
            headers=headers,
        )
        response = connection.getresponse()
        status, content = response.status, response.read()
        connection.close()
        return status, content

    def test_model_identity_and_authentication(self):
        self.assertEqual(self.request("/v1/qit/model", token="wrong")[0], 401)
        status, body = self.request("/v1/qit/model")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model"], "fixture")
        self.assertEqual(self.request("/v1/qit/invoke", {"model": "other"})[0], 400)

    def test_stream_terminal_and_inference_failure(self):
        status, body = self.request("/v1/qit/invoke", {"model": "fixture"})
        self.assertEqual(status, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual([event["type"] for event in events], ["delta", "result"])
        _, body = self.request("/v1/qit/invoke", {"model": "fixture", "fail": True})
        self.assertEqual(json.loads(body.splitlines()[-1])["type"], "error")

    def test_busy_provider_rejects_overlapping_measurements(self):
        self.server.inference.acquire()
        try:
            self.assertEqual(
                self.request("/v1/qit/invoke", {"model": "fixture"})[0], 409
            )
        finally:
            self.server.inference.release()


class Profiles(unittest.TestCase):
    def test_all_canonical_checkpoints_have_pinned_loading_paths(self):
        selected = profiles()
        self.assertEqual(len({profile["model"] for profile in selected}), 43)
        self.assertEqual(
            Counter(profile["task"] for profile in selected),
            {
                "text_generation": 6,
                "embedding": 10,
                "reranking": 6,
                "image_to_text": 8,
                "speech_to_text": 8,
                "text_to_image": 5,
            },
        )
        for profile in selected:
            self.assertRegex(
                profile_for(profile["model"])["revision"], r"^[a-f0-9]{40}$"
            )
            if profile["trust_publisher_code"]:
                self.assertRegex(profile["code_revision"], r"^[a-f0-9]{40}$")

    def test_role_preparation_preserves_raw_dataset_and_order(self):
        case = {"input": "weather", "documents": ["rain", "sun"]}
        original = json.loads(json.dumps(case))
        prepared = prepared_inputs(profile_for("intfloat/e5-small-v2"), case)
        self.assertEqual(prepared, ["query: weather", "passage: rain", "passage: sun"])
        self.assertEqual(case, original)
        case.update(query_prefix="", document_prefix="")
        self.assertEqual(
            prepared_inputs(profile_for("intfloat/e5-small-v2"), case),
            ["weather", "rain", "sun"],
        )
        with self.assertRaises(ValueError):
            prepared_inputs(
                profile_for("intfloat/e5-small-v2"),
                {"input": "weather", "documents": []},
            )

    def test_moonshine_audio_minimum_covers_all_encoder_convolutions(self):
        encoder = SimpleNamespace(
            conv1=SimpleNamespace(kernel_size=(127,), stride=(64,)),
            conv2=SimpleNamespace(kernel_size=(7,), stride=(3,)),
            conv3=SimpleNamespace(kernel_size=(3,), stride=(2,)),
        )
        backend = object.__new__(Backend)
        backend.model = SimpleNamespace(get_encoder=lambda: encoder)
        minimum = backend.minimum_audio_samples()
        self.assertEqual(minimum, 895)
        for length in (minimum - 1, minimum):
            remaining = length
            for layer in (encoder.conv1, encoder.conv2, encoder.conv3):
                remaining = (remaining - layer.kernel_size[0]) // layer.stride[0] + 1
            self.assertEqual(remaining, int(length == minimum))

    def test_image_settings_are_explicit_and_bounded(self):
        self.assertEqual(image_settings({})["width"], 512)
        for update in [
            {"width": 63},
            {"height": 1025},
            {"seed": -1},
            {"steps": 0},
            {"guidance_scale": float("nan")},
        ]:
            value = image_settings({}) | update
            with self.assertRaises(ValueError):
                image_settings({"image_settings": value})


if __name__ == "__main__":
    unittest.main()
