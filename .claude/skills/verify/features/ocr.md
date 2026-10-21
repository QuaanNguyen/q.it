# OCR image to text

## Sub-features

- Run the `core/image-to-text-smoke` pack against a Local CPU provider.
- Retain a bounded `output_metadata` object that records which native route produced the text and which stage calls ran.
- Show the retained evidence in the TUI run detail under `Output evidence`.
- Serve the same evidence from `/api/runs/<id>` and show it in the dashboard's `Output evidence` disclosure.
- Admit the pinned Indra ONNX route by its full static tensor bound because ONNX does not encode which constants are learned.

## How to get to it (user POV)

```sh
target/debug/qit run core/image-to-text-smoke --provider local-cpu \
  --model datalab-to/indra-ocr \
  --base-url http://127.0.0.1:<port>/datalab-to/indra-ocr \
  --iterations 2 --warmups 1
```

The case asks for the two dominant colors of a generated two-color image, so a correct answer is `red and blue`.

## Driving it with the CLI

```sh
.claude/skills/verify/helpers/drive-ocr.sh
```

The fixture provider answers the same metadata the Indra adapter publishes: `image_to_text`, `float32`, the pinned revision, a null exact parameter count, and a `parameter_count_upper_bound` under one billion with basis `onnx-full-static-constant-upper-bound-v1`.

For the failure path the drive starts a second provider in `--mode invalid-metadata`, which returns output evidence as an array. The command must exit `2` and print `output metadata must be an object of at most 64 KiB`.

## Gotchas

- `--model` must be exactly `datalab-to/indra-ocr` for the bound admission to apply. Any other name needs an exact non-zero count.
- Output evidence is capped at 64 KiB after serialization; a large transcript belongs in `output_text`, not in the metadata object.
- The TUI label is `Output evidence`, and the dashboard disclosure has the same title, so a search that misses one still proves the other.
- The failed run is expected to appear in `qit results`; the drive asserts it does.
