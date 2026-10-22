# Built-in benchmarks

The built-in `core` pack contains seven small benchmarks with thirteen cases in total.
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

Supported servers are Ollama, Transformers Serve, and the optional local CPU provider.

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

Supported servers are Ollama, Hugging Face Serve, Hugging Face Text Embeddings Inference, and the optional local CPU provider.

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

Supported servers are Ollama, Transformers Serve, and the optional local CPU provider.

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

Transformers Serve and the optional local CPU provider support this check.

## Text-to-speech transport check

Command:

```bash
qit run core/text-to-speech-smoke \
  --provider local-cpu \
  --model gradium/phonon-31m-en
```

This benchmark synthesizes two short phrases and validates the returned audio stream.
The first phrase is a greeting and the second is a short sentence, both with an explicit language and duration limit.
q.it checks the sample rate, the byte count, and the decoded duration of the received PCM audio.

The benchmark records time to first audio, real-time factor, and peak memory for each phrase.
An audio clip that exceeds the case duration limit fails validation.
This is a transport and latency check, not a speech-quality evaluation.

The optional local CPU provider with the Phonon adapter supports this check.
No other provider integration can synthesize audio.

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

Hugging Face Text Embeddings Inference and the optional local CPU provider support this check.

## Text-to-image transport check

The built-in identifier is `core/text-to-image-smoke`.

The optional local CPU provider can run this case through the five pinned BK-SDM profiles.
The default profile generates a 512 by 512 PNG with twenty steps, guidance 7.5, and CPU seed 42.
q.it validates and stores the image, but does not attach an image-quality score.
The other provider integrations do not support this task.

## How runs are classified

By default, q.it performs one warmup and three measured passes.
Every case runs during every measured pass.

An incorrect answer is still a completed model response.
It lowers the quality score but does not make the run fail.

A run fails when a required request cannot be completed or validated.
Examples include an unavailable server, a timeout, an unsupported model capability, an invalid response, or an interrupted output stream.
