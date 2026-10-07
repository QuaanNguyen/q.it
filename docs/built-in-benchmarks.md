# Built-in benchmarks

The built-in `core` pack contains six small benchmarks with eleven cases in total.
They are designed to confirm that q.it, the model server, and the selected model can complete each kind of request.

They are not official model leaderboards and should not be used as broad claims about model quality.
Install larger dataset-backed benchmark packs when you need a representative quality evaluation.

## Text generation smoke test

Command:

```bash
qit run core/text-generation-smoke --provider ollama --model gemma3:1b
```

This benchmark sends three prompts:

1. Reply with one requested word.
2. Answer a small multiplication question.
3. Explain briefly why low memory use matters on an edge device.

The first two cases check for their expected answer.
The third case measures performance without trying to judge writing quality.

For a streaming server, q.it records request latency, time to first token, and tokens per second when an exact token count is available.
It also records CPU and memory measurements.

Supported servers are Ollama and Transformers Serve.

## Embedding retrieval check

Command:

```bash
qit run core/embedding-retrieval \
  --provider ollama \
  --model embedding-model
```

This benchmark contains three short searches covering weather, Raspberry Pi, and application memory.
Each search has one relevant passage and two unrelated passages.

The server creates an embedding for the search and every passage.
q.it compares those embeddings and checks where the expected passage appears in the ranking.
A first-place match receives the best score.

This is a small semantic sanity check, not a replacement for MTEB or BEIR.

Supported servers are Ollama and Hugging Face Text Embeddings Inference.

## Image-to-text smoke test

Command:

```bash
qit run core/image-to-text-smoke \
  --provider ollama \
  --model vision-model
```

This benchmark sends a generated image with two dominant colors.
It asks the model to name those colors and checks whether both expected color names appear in the response.

The image is included with q.it, so the check works without downloading a dataset.
The benchmark measures request latency, streaming speed, CPU use, and memory use.

Supported servers are Ollama and Transformers Serve.

## Speech-to-text transport check

Command:

```bash
qit run core/speech-to-text-smoke \
  --provider transformers \
  --model speech-model
```

This benchmark uploads a short generated audio tone to the transcription endpoint.
Its purpose is to confirm that q.it can prepare the audio request and receive a valid transcription response.

The audio does not contain spoken words.
This benchmark does not measure speech recognition accuracy and does not produce a word-error score.

Transformers Serve is the only supported server for this check.

## Reranking relevance check

Command:

```bash
qit run core/reranking-relevance \
  --provider tei \
  --model reranking-model
```

This benchmark contains two searches with three candidate passages each.
One search is about edge computing and the other is about checking memory on Raspberry Pi OS.

q.it asks the server to rank each set of passages and checks where the expected passage appears.
A first-place result receives the best score.

Hugging Face Text Embeddings Inference is the only supported server for this check.

## Text-to-image placeholder

The built-in identifier is `core/text-to-image-smoke`.

This entry reserves a benchmark shape for a future image-generation server integration.
None of the current server integrations can run it, so q.it lists it as reserved.

It must not be treated as a working text-to-image benchmark.

## How runs are classified

By default, q.it performs one warmup and three measured passes.
Every case runs during every measured pass.

An incorrect answer is still a completed model response.
It lowers the quality score but does not make the run fail.

A run fails when a required request cannot be completed or validated.
Examples include an unavailable server, a timeout, an unsupported model capability, an invalid response, or an interrupted output stream.
