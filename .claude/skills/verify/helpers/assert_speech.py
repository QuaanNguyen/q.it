#!/usr/bin/env python3
"""Assert the evidence a speech drive captured and write summary.md."""

import base64
import io
import json
import sys
import wave
from pathlib import Path

EXPECTED_MESSAGE = "speech audio duration must be a positive finite number"


def pack_duration(root):
    case = json.loads(
        (root / "qit-runtime/benchmarks/core/speech-to-text-smoke.jsonl")
        .read_text()
        .strip()
    )
    encoded = case["media"].partition(",")[2]
    with wave.open(io.BytesIO(base64.b64decode(encoded))) as audio:
        return audio.getnframes() / audio.getframerate()


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    row = json.loads((out / "export.jsonl").read_text().strip())
    run = row["run"]
    samples = row["samples"]
    assert run["status"] == "succeeded", run["status"]
    assert run["task"] == "speech_to_text", run["task"]
    assert len(samples) == 2, f"expected 2 samples (1 case x 2 iterations), got {len(samples)}"
    duration = pack_duration(root)
    for sample in samples:
        speech = sample["speech"]
        assert speech is not None, f"{sample['case_id']} has no speech observation"
        assert speech["transcript"] == "the birch canoe slid on the smooth planks"
        assert speech["audio_duration_seconds"] == duration, (
            f"stored duration {speech['audio_duration_seconds']} != audio {duration}"
        )
        assert speech["routing"]["selected_route"] == "english"
        assert sample["quality_score"] is None, sample["quality_score"]
    detail = json.loads((out / "dashboard.json").read_text())
    assert len(detail["samples"]) == len(samples), "dashboard returned a different sample count"
    for stored, served in zip(samples, detail["samples"]):
        assert served["speech"] == stored["speech"], "dashboard speech differs from export"
    invalid_status = (out / "invalid.status").read_text().strip()
    assert invalid_status == "2", f"invalid drive exited {invalid_status}, expected 2"
    invalid_output = (out / "invalid.out").read_text() + (out / "invalid.err").read_text()
    assert EXPECTED_MESSAGE in invalid_output, (
        "invalid drive did not report the rejected audio duration"
    )
    history = (out / "results-after-invalid.out").read_text()
    assert history.count("succeeded") >= 1, "successful run missing from history"
    assert history.count("failed") >= 1, "failed run missing from history"
    summary = [
        "# Speech verification",
        "",
        f"- Run: `{run['id']}`",
        f"- Model: `{run['model']}`",
        f"- Samples: {len(samples)}",
        f"- Stored duration matched the case audio ({duration} s) in every sample",
        "- Transcript and routing survived export and the dashboard API unchanged",
        "- Quality stayed null because the smoke pack declares no expected transcript",
        "- Failed run stayed in history",
        f"- Negative audio duration refused with exit 2 and `{EXPECTED_MESSAGE}`",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
