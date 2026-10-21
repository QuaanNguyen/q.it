# Speech transcript

## Sub-features

- Run the `core/speech-to-text-smoke` pack against a Local CPU provider.
- Retain the transcript, the audio duration the provider measured, and the routing detail.
- Show duration, transcript, and routing in the TUI run detail.
- Show the same values in the dashboard case row.
- Refuse a duration that is missing, non-finite, or not positive.

## How to get to it (user POV)

```sh
target/debug/qit run core/speech-to-text-smoke --provider local-cpu \
  --model parakeet-fixture \
  --base-url http://127.0.0.1:<port>/parakeet-fixture \
  --iterations 2 --warmups 1
```

The shipped case sends a 0.04 second generated tone, so a correct provider reports that duration rather than a nominal one.

## Driving it with the CLI

```sh
.claude/skills/verify/helpers/drive-speech.sh
```

The fixture decodes the case WAV from the request and reports its true duration, so the assertion compares the stored value with the audio a user actually sent.

For the failure path the drive starts a second provider in `--mode invalid-duration`, which reports `-1`. The command must exit `2` and print `speech audio duration must be a positive finite number`.

## Gotchas

- The smoke pack declares no expected transcript, so `quality_score` is null by design. Use a custom pack with `expected_text` to score transcripts.
- The fixture reports `speech_routing` as a plain object; the CLI stores it without interpreting the route names.
- Speech samples still carry `output_excerpt`, which is the truncated transcript. The TUI prints the speech block instead so the transcript is not duplicated.
