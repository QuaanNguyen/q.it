# Benchmark pack guide

Benchmark packs add cases to q.it without adding executable code.
A pack can contain prompts, expected answers, candidate passages, and local image or audio files.

Official dataset-backed evaluations are intended to use this format.
There is currently no online q.it pack registry, so installation starts from a local directory.

## Install a pack

Check the pack before installing it:

```bash
qit benchmark verify /path/to/pack
```

Install it:

```bash
qit benchmark install /path/to/pack
```

List the benchmarks that are now available:

```bash
qit benchmark list
```

Remove the pack later without deleting old results:

```bash
qit benchmark remove pack-name
```

## Minimal layout

A small text benchmark needs one description file and one case file:

```text
my-pack/
├── pack.json
└── prompts.jsonl
```

The description file identifies the pack and the benchmark:

```json
{
  "schema_version": 1,
  "id": "my-pack",
  "name": "My Pi prompts",
  "version": "1.0.0",
  "description": "A small local instruction benchmark.",
  "publisher": "Your name",
  "license": "CC0-1.0",
  "source": "https://example.com/my-pack",
  "benchmarks": [
    {
      "id": "instructions",
      "name": "Instructions",
      "description": "Checks a fixed set of short instructions.",
      "task": "text_generation",
      "cases": "prompts.jsonl",
      "default_max_output_tokens": 64
    }
  ]
}
```

Each line in the case file is one test case:

```json
{"id":"hello","prompt":"Reply with only the word hello.","expected_contains":["hello"]}
```

Run the installed benchmark using its pack and benchmark names:

```bash
qit run my-pack/instructions --provider ollama --model my-model
```

## Supported kinds of cases

The current pack format supports:

- Text generation with optional expected phrases.
- Text embedding retrieval with candidate passages and an expected passage.
- Image-to-text with a prompt, image, and optional expected phrases.
- Speech-to-text with an audio file and optional expected text.
- Text reranking with candidate passages and an expected passage.
- A reserved text-to-image shape for a future server integration.

An image or audio case can refer to a file inside the pack or include a base64 data URL.
Keep normal dataset media as files because large base64 lines are difficult to review.

## Rules for a reproducible pack

A useful pack should:

- Use a stable pack name and version.
- Identify its publisher, source, and license.
- Keep case identifiers stable between runs.
- Keep prompts and expected answers fixed within one version.
- Include only media that the license allows the pack to redistribute.
- Explain whether it is a smoke, standard, or full evaluation.
- Pin the source dataset version and split.
- Avoid random case selection unless the selected case list is stored explicitly.

Increase the pack version whenever a prompt, expected answer, case list, or media file changes.
This prevents results from two different evaluations from appearing comparable.

## Safety and offline use

Packs are data only.
They cannot run Python, shell commands, native libraries, or downloaded model code through q.it.

Installation rejects symbolic links and paths that escape the pack directory.
The pack remains available offline after all of its local files are installed.
