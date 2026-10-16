import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NemoCommand(unittest.TestCase):
    def command(self, *arguments):
        environment = dict(
            os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])
        )
        return subprocess.run(
            [sys.executable, "-m", "qit_cpu.nemo_cli", *arguments],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_lists_pinned_qvac_routes_without_importing_optional_models(self):
        result = self.command("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        profiles = json.loads(result.stdout)
        self.assertEqual(
            [profile["model"] for profile in profiles],
            ["qvac/TranscribePsy-250M-EU", "qvac/TranscribePsy-250M-EU-sub"],
        )
        self.assertEqual(
            profiles[0]["routes"], ["english", "multilingual", "automatic"]
        )
        self.assertEqual(profiles[1]["routes"], ["shared"])
        self.assertEqual(
            profiles[0]["revision"], "49d2b21e72018798a755c0cbe5ebb4fefe2384a1"
        )
        self.assertEqual(
            profiles[1]["revision"], "ccf70de9ff9848f5cd5975adda1370ce4addf6f1"
        )

    def test_verification_reports_the_missing_local_archive_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command(
                "verify",
                "--model",
                "qvac/TranscribePsy-250M-EU",
                "--route",
                "english",
                "--model-dir",
                directory,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("artifact_missing", result.stderr)
        self.assertIn("TranscribePsy-250M-EU.nemo", result.stderr)

    def test_fetch_rejects_an_incompatible_route_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command(
                "fetch",
                "--model",
                "qvac/TranscribePsy-250M-EU-sub",
                "--route",
                "automatic",
                "--model-dir",
                directory,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("route_unsupported", result.stderr)

    def test_serving_requires_explicit_trust_for_pinned_publisher_code(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command(
                "serve",
                "--model",
                "qvac/TranscribePsy-250M-EU",
                "--route",
                "english",
                "--model-dir",
                directory,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--trust-publisher-code", result.stderr)


if __name__ == "__main__":
    unittest.main()
