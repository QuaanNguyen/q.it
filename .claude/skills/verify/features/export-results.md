# Export results

## Sub-features

- `qit export --format jsonl` streams one JSON object per successful run with its samples.
- `qit export --format csv --output PATH` writes a flat table.
- Failed runs never enter an export; they remain in run history.
- Samples carry metrics, errors, output excerpts, images, decisions, speech detail, and native retrieval observations.

## How to get to it (user POV)

```sh
target/debug/qit export --format jsonl --output results.jsonl
target/debug/qit export --format csv --output results.csv
```

## Driving it with the CLI

The drive writes `export.jsonl` and asserts, for every sample:

- `retrieval.schema_version == 1` and `retrieval.method == "maxsim"`,
- one query vector and one entry per candidate document, all 128-dimensional,
- `scores` equals MaxSim recomputed from the retained vectors,
- `ranking` equals score order with document-order ties,
- `quality_score` equals `1 / (rank of expected document + 1)`.

## Gotchas

- Export defaults to stdout; pass `--output` when evidence must be a file.
- JSONL contains complete vectors, so files grow quickly with long documents; the adapter caps a single observation at one million scalars.
- CSV flattens nested observations; use JSONL for retrieval or decision detail.
