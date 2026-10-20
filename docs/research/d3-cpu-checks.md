# D3 retained CPU checks

The original draft retained installed-command CPU smoke checks and independent publisher comparisons on Sol Linux x86_64.
These historical checks used adapter source `7c5964ec8fe13b259b7b37a8e1e9c941e0e76082` and q.it binary source `3d112569a9d7f28f7185b25d28e2f2fd2b8c4b8f`.
They establish compatibility for those source and pinned runtime combinations.
They do not establish dataset-wide quality, performance rankings, native ARM64 installation, or Raspberry Pi qualification.

| Model | Pinned checkpoint | Stored parameters | Complete loaded parameters |
| --- | --- | ---: | ---: |
| [D3 Edge](https://huggingface.co/vllm-sr/d3-edge/blob/c454ee39b862248d70abd7f84195d2f54ffe6d43/README.md) | `c454ee39b862248d70abd7f84195d2f54ffe6d43` | 594,321,600 | 594,582,720 |
| [D3 Lite](https://huggingface.co/vllm-sr/d3-lite/blob/03560c26f5199c3c04129a2ccb38960c913635e0/README.md) | `03560c26f5199c3c04129a2ccb38960c913635e0` | 852,985,920 | 853,247,040 |

The cards state rounded totals of 0.59B and 0.85B including vision.
Exact stored counts come from Hugging Face safetensors metadata at each pinned revision and are retained in the catalog's card evidence.
The complete loaded counts separately include 261,120 learned readout parameters held outside registered model parameters.
Both complete learned stacks remain strictly below one billion.
The native publisher backbone executes as CPU BF16, and the learned readout remains CPU FP32.

Each model succeeded on the original three English text cases and one image case with four CPU threads.
Independent publisher comparisons matched every candidate probability and the selected answer for all four cases per model.
The retained directories are `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/4` for Edge and `/scratch/qnguye19/qit-optional-cpu-20261019-3d11256/attempts/3/outcomes/5` for Lite.
Each directory's `text` and `image` subdirectories retain metadata, parameter audits, successful exports, the SQLite store, and publisher-comparison output.
The original adapter wheel SHA-256 was `0bc3c34651851d49e1063c72bfb3349beb41eee9a780bb607ceadcd4df11b790`.

This extraction preserves the publisher constructor and exact prepare, run, and respond object flow.
It adds an observed CPU FP32 readout check before complete auditing and native precision metadata admission.
Current Python fixtures check constructor options, source admission, complete counts, tensor rejection, ordered probabilities, preparation errors, and input digests.
Actual Rust CLI fixtures accept both exact pins for text and image and reject changed precision identities, model revisions, devices, and invalid counts before inference.
These fixtures use controlled local endpoints and publisher stand-ins.
No fresh model execution, download, or Sol job is implied by those checks.
