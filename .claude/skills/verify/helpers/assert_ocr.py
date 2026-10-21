#!/usr/bin/env python3
"""Assert the evidence an OCR drive captured and write summary.md."""

import json
import sys
from pathlib import Path

EXPECTED_MESSAGE = "output metadata must be an object of at most 64 KiB"


def main():
    out = Path(sys.argv[2])
    row = json.loads((out / "export.jsonl").read_text().strip())
    run = row["run"]
    samples = row["samples"]
    assert run["status"] == "succeeded", run["status"]
    assert run["task"] == "image_to_text", run["task"]
    assert len(samples) == 2, f"expected 2 samples (1 case x 2 iterations), got {len(samples)}"
    for sample in samples:
        assert sample["quality_score"] == 1.0, sample["quality_score"]
        assert sample["output_excerpt"] == "red and blue", sample["output_excerpt"]
        evidence = sample["output_metadata"]
        assert evidence is not None, f"{sample['case_id']} has no output evidence"
        assert evidence["schema_version"] == 1
        assert evidence["route"] == "indra-full-ocr"
        assert evidence["stage_calls"] == {"det": 1, "rec": 1}
    detail = json.loads((out / "dashboard.json").read_text())
    assert len(detail["samples"]) == len(samples), "dashboard returned a different sample count"
    for stored, served in zip(samples, detail["samples"]):
        assert served["output_metadata"] == stored["output_metadata"], (
            "dashboard evidence differs from export"
        )
        assert served["output_excerpt"] == stored["output_excerpt"]
    invalid_status = (out / "invalid.status").read_text().strip()
    assert invalid_status == "2", f"invalid drive exited {invalid_status}, expected 2"
    invalid_output = (out / "invalid.out").read_text() + (out / "invalid.err").read_text()
    assert EXPECTED_MESSAGE in invalid_output, (
        "invalid drive did not report the malformed output evidence"
    )
    history = (out / "results-after-invalid.out").read_text()
    assert history.count("succeeded") >= 1, "successful run missing from history"
    assert history.count("failed") >= 1, "failed run missing from history"
    summary = [
        "# OCR verification",
        "",
        f"- Run: `{run['id']}`",
        f"- Model: `{run['model']}`",
        f"- Samples: {len(samples)}",
        "- Every sample scored 1.0 on the case's expected colors",
        "- Retained output evidence named the Indra full-OCR route and stage calls",
        "- Dashboard API returned the exported evidence unchanged",
        "- Failed run stayed in history",
        f"- Malformed evidence refused with exit 2 and `{EXPECTED_MESSAGE}`",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
