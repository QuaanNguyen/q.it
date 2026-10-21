#!/usr/bin/env python3
"""Assert the evidence a retrieval drive captured and write summary.md."""

import json
import sys
from pathlib import Path

EXPECTED_MESSAGE = "native retrieval score differs from MaxSim over the retained vectors"


def maxsim(query, documents):
    scores = []
    for document in documents:
        total = 0.0
        for query_vector in query:
            total += max(
                sum(left * right for left, right in zip(query_vector, token))
                for token in document
            )
        scores.append(total)
    return scores


def ranking_of(scores):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    cases = {}
    for line in (root / "qit-runtime/benchmarks/core/embedding-retrieval.jsonl").read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            cases[case["id"]] = case
    row = json.loads((out / "export.jsonl").read_text().strip())
    run = row["run"]
    samples = row["samples"]
    assert run["status"] == "succeeded", run["status"]
    assert run["task"] == "embedding", run["task"]
    assert len(samples) == 6, f"expected 6 samples (3 cases x 2 iterations), got {len(samples)}"
    qualities = {}
    for sample in samples:
        observation = sample["retrieval"]
        assert observation is not None, f"{sample['case_id']} has no retrieval observation"
        assert observation["schema_version"] == 1
        assert observation["method"] == "maxsim"
        assert observation["tie_break"] == "document_order"
        assert len(observation["query"]) == 1
        documents = observation["documents"]
        assert len(documents) == 3, f"{sample['case_id']} retained {len(documents)} documents"
        for vectors in [observation["query"], *documents]:
            assert vectors, f"{sample['case_id']} retained an empty matrix"
            for vector in vectors:
                assert len(vector) == 128, f"{sample['case_id']} vector width {len(vector)}"
                assert all(value == value and abs(value) != float("inf") for value in vector)
        recomputed = maxsim(observation["query"], documents)
        for expected, observed in zip(recomputed, observation["scores"]):
            assert abs(expected - observed) <= 0.0001 * max(abs(expected), 1.0), (
                f"{sample['case_id']} stored score {observed} differs from MaxSim {expected}"
            )
        assert observation["ranking"] == ranking_of(observation["scores"]), sample["case_id"]
        expected_index = cases[sample["case_id"]]["expected_index"]
        position = observation["ranking"].index(expected_index)
        assert sample["quality_score"] == 1.0 / (position + 1), (
            f"{sample['case_id']} quality {sample['quality_score']} != 1/{position + 1}"
        )
        assert (sample["output_excerpt"] or "").startswith("MaxSim scores"), sample["case_id"]
        qualities.setdefault(sample["case_id"], sample["quality_score"])
    detail = json.loads((out / "dashboard.json").read_text())
    assert len(detail["samples"]) == len(samples), "dashboard returned a different sample count"
    for stored, served in zip(samples, detail["samples"]):
        assert served["retrieval"] == stored["retrieval"], "dashboard retrieval differs from export"
    invalid_status = (out / "invalid.status").read_text().strip()
    assert invalid_status == "2", f"invalid drive exited {invalid_status}, expected 2"
    invalid_output = (out / "invalid.out").read_text() + (out / "invalid.err").read_text()
    assert EXPECTED_MESSAGE in invalid_output, "invalid drive did not report the score mismatch"
    summary = [
        "# Retrieval verification",
        "",
        f"- Run: `{run['id']}`",
        f"- Model: `{run['model']}`",
        f"- Samples: {len(samples)}",
        f"- Cases and quality: {json.dumps(qualities, sort_keys=True)}",
        "- Stored scores equal MaxSim recomputed from retained vectors",
        "- Ranking equals score order with document-order ties",
        "- Dashboard API returned the exported observation unchanged",
        f"- Inconsistent observation refused with exit 2 and `{EXPECTED_MESSAGE}`",
    ]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
