# CPU candidates below one billion parameters for multimodal benchmarks

Research date: 2026-10-09.

This note identifies eight image-to-text, eight speech-to-text, and five text-to-image checkpoints from their original research publishers.
All candidates are available on Hugging Face as original or officially distilled weights.
No community quantization is used to make an oversized model appear eligible.
The cutoff is strictly fewer than 1,000,000,000 parameters for the selected inference stack, including required vision or audio encoders, text decoders, projectors, text encoders, and the complete packaged VAE.

CPU execution here means a documented Hugging Face CPU inference path or a CPU configuration of the publisher's supported PyTorch architecture.
These candidates have not been benchmarked in this research task, and their inclusion does not establish Raspberry Pi 5 speed, memory fit, or successful q.it integration.
Use CPU-only PyTorch, float32, batch size one, and CPU-supported attention instead of copying CUDA or FlashAttention examples.
[Transformers pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines) explicitly allows forcing CPU and overriding checkpoint precision.

## Image to text: eight candidates

The following counts include the visual encoder and language component in the complete checkpoint.
The exact tensor totals come from the original repositories' first-party Hugging Face metadata, rather than an isolated language backbone.
Integer buffers can be included in these totals, making them conservative eligibility checks rather than claims of an exact count of trainable parameters.

| Checkpoint and publisher | Complete checkpoint size | License | CPU route and benchmark fit |
| --- | ---: | --- | --- |
| [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) - Hugging Face | 256,484,928 | Apache-2.0 | Transformers multimodal generation on CPU; instruction-following image questions, including the existing color prompt |
| [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) - Hugging Face | 507,482,304 | Apache-2.0 | Same CPU architecture; compare image-question quality against the smaller sibling |
| [HuggingFaceTB/SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) - Hugging Face | 256,484,928 | Apache-2.0 | Transformers multimodal generation on CPU; use still images for the present image task |
| [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) - Hugging Face | 507,482,304 | Apache-2.0 | Same CPU architecture; still-image comparison of the officially updated training recipe |
| [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) - Microsoft | 231,567,705 | MIT | Publisher example explicitly selects CPU and float32; task-token captioning, OCR, and grounded vision outputs |
| [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) - Microsoft | 776,721,497 | MIT | Same documented CPU fallback; higher-capacity captioning and OCR comparison |
| [microsoft/git-base](https://huggingface.co/microsoft/git-base) - Microsoft | 176,620,287 | MIT | Transformers GIT generation on CPU; captioning baseline rather than a general conversational assistant |
| [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) - Salesforce | 469,733,436 | BSD-3-Clause | Publisher provides a CPU example; conditional or unconditional captioning baseline |

The [official SmolVLM announcement](https://huggingface.co/blog/smolervlm) explains that the small models include a 93M visual encoder alongside their text component.
The SmolVLM2 model cards explicitly support still images as well as videos, so no video benchmark or decoder is required for this selection.
The [SmolVLM architecture documentation](https://huggingface.co/docs/transformers/model_doc/smolvlm) and the [CPU pipeline contract](https://huggingface.co/docs/transformers/main_classes/pipelines) establish the proposed CPU framework path.
For SmolVLM2, replace the model card's CUDA and FlashAttention settings with CPU, float32, and a supported PyTorch attention implementation before validating the server.

The current q.it image provider sends a text question and an image to an OpenAI-compatible streaming chat endpoint, or to Ollama's vision generation endpoint.
[Transformers Serve](https://huggingface.co/docs/transformers/serving) supplies an image chat API, making it the first serving path to validate for the SmolVLM instruction checkpoints.
The available Hugging Face checkpoint does not establish that every Transformers Serve release can load it or stream its output correctly.
Run model discovery, processor loading, one real image request, and streaming termination checks before marking that combination supported.

Florence, GIT, and BLIP need separate architecture handling if the generic chat server cannot represent their processor and generation contract.
The [Florence model card](https://huggingface.co/microsoft/Florence-2-base) uses explicit task tokens such as `<CAPTION>` rather than an arbitrary conversational prompt.
[GIT documentation](https://huggingface.co/docs/transformers/model_doc/git) and the [BLIP CPU example](https://huggingface.co/Salesforce/blip-image-captioning-large) describe caption generation.
For those models, add an appropriate caption or OCR benchmark pack and a processor-aware serving adapter instead of silently translating the existing color-question prompt into a different task.
Keep conversational VQA, captioning, and OCR quality rankings separate.

### Image checkpoint revisions

These first-party metadata revisions were read during the parameter and license audit.
The metadata endpoint has the form `https://huggingface.co/api/models/<checkpoint>`.

| Checkpoint | Audited revision |
| --- | --- |
| HuggingFaceTB/SmolVLM-256M-Instruct | [7e3e67edbbed1bf9888184d9df282b700a323964](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964) |
| HuggingFaceTB/SmolVLM-500M-Instruct | [a7da5b986cb59b408707209984f360a5f4ad7e47](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/tree/a7da5b986cb59b408707209984f360a5f4ad7e47) |
| HuggingFaceTB/SmolVLM2-256M-Video-Instruct | [067788b187b95ebe7b2e040b3e4299e342e5b8fd](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/tree/067788b187b95ebe7b2e040b3e4299e342e5b8fd) |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | [7b375e1b73b11138ff12fe22c8f2822d8fe03467](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467) |
| microsoft/Florence-2-base | [5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac](https://huggingface.co/microsoft/Florence-2-base/tree/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac) |
| microsoft/Florence-2-large | [21a599d414c4d928c9032694c424fb94458e3594](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594) |
| microsoft/git-base | [1f7fe8444292beb4a259e3a5b6eba440cd5999d4](https://huggingface.co/microsoft/git-base/tree/1f7fe8444292beb4a259e3a5b6eba440cd5999d4) |
| Salesforce/blip-image-captioning-large | [353689b859fcf0523410b1806dace5fb46ecdf41](https://huggingface.co/Salesforce/blip-image-captioning-large/tree/353689b859fcf0523410b1806dace5fb46ecdf41) |

## Speech to text: eight candidates

All selected speech checkpoints contain the complete speech encoder and text decoder.
They require no additional learned language model for their normal greedy or sequence-to-sequence decoding path.
The Whisper sizes below use the publisher's nominal complete-model counts, which safely exceed the corresponding stored-tensor totals and remain below the cutoff.

| Checkpoint and publisher | Complete model size | License | CPU route and benchmark fit |
| --- | ---: | --- | --- |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) - OpenAI | 39M nominal | Apache-2.0 in the official HF repository | Transformers ASR pipeline, CPU float32; multilingual speech recognition |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) - OpenAI | 74M nominal | Apache-2.0 in the official HF repository | Same documented CPU fallback; multilingual size comparison |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) - OpenAI | 244M nominal | Apache-2.0 in the official HF repository | Same CPU fallback; multilingual size comparison |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) - OpenAI | 769M nominal | Apache-2.0 in the official HF repository | Same CPU fallback; larger eligible baseline, not a promise of interactive CPU speed |
| [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) - Hugging Face Distil-Whisper team | 166,132,224 | MIT | Publisher explicitly selects CPU float32; English-only official architectural distillation |
| [distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) - Hugging Face Distil-Whisper team | 394,375,168 | MIT | Same documented CPU fallback; English-only speed and accuracy comparison |
| [moonshine-ai/moonshine-tiny](https://huggingface.co/moonshine-ai/moonshine-tiny) - Useful Sensors / Moonshine AI | 27,092,736 | MIT | Publisher explicitly selects CPU float32; English speech, intended for constrained devices |
| [moonshine-ai/moonshine-base](https://huggingface.co/moonshine-ai/moonshine-base) - Useful Sensors / Moonshine AI | 61,513,920 | MIT | Same CPU fallback; English speech size comparison |

The [Whisper model table and CPU pipeline example](https://huggingface.co/openai/whisper-tiny) verify all four selected sizes and their whole encoder-decoder architecture.
The Distil-Whisper cards describe their original distillation method and supply a float32 CPU branch, rather than a community quantization.
The Moonshine publisher's CPU examples describe full raw-audio-to-text inference.
The old `UsefulSensors/moonshine-*` repositories now redirect to the `moonshine-ai/moonshine-*` identifiers above.

q.it already uploads audio as multipart data to `/v1/audio/transcriptions` through its Transformers provider.
The [Transformers Serve transcription API](https://huggingface.co/docs/transformers/serving) matches that transport contract.
Validate each architecture against the pinned server version; a processor-aware CPU ASR wrapper is a fallback if that version lacks Moonshine or another selected architecture.
No current Ollama, TEI, or embedding-only Hugging Face Serve integration should be advertised as an ASR provider.

The current built-in speech transport check contains a tone, not spoken language, and gives no recognition accuracy score.
Add a dataset-backed speech pack with real audio and reference transcripts before comparing model quality.
Use the same English audio subset for the eight-way comparison, with transcription rather than translation and no timestamps in scored text.
Use separate language-specific cohorts if multilingual evaluation is added.
Record audio duration, preprocessing and resampling, decoding settings, word error rate, real-time factor, latency, CPU use, and process memory.

### Speech checkpoint revisions

| Checkpoint | Audited revision |
| --- | --- |
| openai/whisper-tiny | [169d4a4341b33bc18d8881c4b69c2e104e1cc0af](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af) |
| openai/whisper-base | [e37978b90ca9030d5170a5c07aadb050351a65bb](https://huggingface.co/openai/whisper-base/tree/e37978b90ca9030d5170a5c07aadb050351a65bb) |
| openai/whisper-small | [973afd24965f72e36ca33b3055d56a652f456b4d](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) |
| openai/whisper-medium | [abdf7c39ab9d0397620ccaea8974cc764cd0953e](https://huggingface.co/openai/whisper-medium/tree/abdf7c39ab9d0397620ccaea8974cc764cd0953e) |
| distil-whisper/distil-small.en | [9e4a67ca4569c30be43a3fe7fba1621e504f0093](https://huggingface.co/distil-whisper/distil-small.en/tree/9e4a67ca4569c30be43a3fe7fba1621e504f0093) |
| distil-whisper/distil-medium.en | [6e61418885eaf4d5cc9f64e508e80ac5b4c052b7](https://huggingface.co/distil-whisper/distil-medium.en/tree/6e61418885eaf4d5cc9f64e508e80ac5b4c052b7) |
| moonshine-ai/moonshine-tiny | [390624ed33d594443aa4aa221f5b9f283b545b5a](https://huggingface.co/moonshine-ai/moonshine-tiny/tree/390624ed33d594443aa4aa221f5b9f283b545b5a) |
| moonshine-ai/moonshine-base | [7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf](https://huggingface.co/moonshine-ai/moonshine-base/tree/7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf) |

## Text to image: five candidates with an explicit stack

The five candidates below are original Nota AI architectural distillations, not quantized repacks.
The [official BK-SDM repository](https://github.com/Nota-NetsPresso/BK-SDM) documents the removed U-Net blocks, original checkpoint IDs, diffusion pipeline, and CreativeML OpenRAIL-M license.
Hugging Face metadata reports that license for all five audited checkpoints.

The selected inference stack is the official U-Net, full CLIP text encoder, full packaged VAE, tokenizer, and scheduler, with the optional learned safety checker disabled for this controlled benchmark.
No image encoder, IP-Adapter, external upscaler, or learned quality scorer is loaded inside the measured generation process.
This configuration is material to the parameter cutoff: the SD-v1 safety checker alone adds approximately 304M tensors and would put BK-SDM-Base above one billion.
If a checker or another learned component is enabled, recount the resulting stack and identify it as a different configuration.
The [Diffusers pipeline contract](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img) distinguishes the U-Net, text encoder, VAE, and post-generation safety classifier.

| Official checkpoint | U-Net tensors | Text encoder tensors | Full VAE tensors | Audited total | Publisher's rounded whole-model figure |
| --- | ---: | ---: | ---: | ---: | ---: |
| [nota-ai/bk-sdm-base](https://huggingface.co/nota-ai/bk-sdm-base) | 579,384,964 | 123,060,557 | 83,653,863 | **786,099,384** | 0.76B |
| [nota-ai/bk-sdm-small](https://huggingface.co/nota-ai/bk-sdm-small) | 482,346,884 | 123,060,557 | 83,653,863 | **689,061,304** | 0.66B |
| [nota-ai/bk-sdm-tiny](https://huggingface.co/nota-ai/bk-sdm-tiny) | 323,384,964 | 123,060,557 | 83,653,863 | **530,099,384** | 0.50B |
| [nota-ai/bk-sdm-v2-small](https://huggingface.co/nota-ai/bk-sdm-v2-small) | 485,787,524 | 340,387,917 | 83,653,863 | **909,829,304** | 0.88B |
| [nota-ai/bk-sdm-v2-tiny](https://huggingface.co/nota-ai/bk-sdm-v2-tiny) | 326,825,604 | 340,387,917 | 83,653,863 | **750,867,384** | 0.72B |

The counts above are calculated from the original safetensors headers and conservatively include all tensors in each mandatory component, including small buffers and the VAE encoder that is unused in ordinary text-to-image decoding.
Read the initial eight bytes to determine the JSON header length, read only that header using an HTTP byte range, then sum the product of each tensor's dimensions once per component.
Do not add float16 and float32 copies together, or count the same shared component twice.
This audit downloaded headers only, not full model weights.
The publisher's model-card totals are lower than the complete packaged component totals, so the explicit count is used for eligibility.

For CPU, load the ordinary Diffusers Stable Diffusion pipeline in float32 and place the entire pipeline on CPU.
Diffusers [documents CPU device placement](https://huggingface.co/docs/diffusers/using-diffusers/loading); replacing the publisher's CUDA placement with CPU is the proposed execution configuration, not a reported timing result.
Use the same 512 by 512 image resolution, fixed CPU generator seeds, batch size one, scheduler, guidance, prompt subset, warmups, and denoising steps for the common comparison.
Keep any architecture-specific scheduler experiment in its own cohort.

q.it currently exposes text-to-image as a reserved placeholder with no runnable provider.
Add a Diffusers CPU server adapter, output image handling, and saved generation settings before scheduling these models through q.it.
Keep generation timing separate from optional image-quality scoring.
Do not populate the dashboard with invented text-to-image results while that integration is missing.

### Diffusion revisions and exclusions

Each linked revision contains the audited `unet/diffusion_pytorch_model.safetensors`, `text_encoder/model.safetensors`, and `vae/diffusion_pytorch_model.safetensors` headers.

| Checkpoint | Audited revision |
| --- | --- |
| nota-ai/bk-sdm-base | [0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0](https://huggingface.co/nota-ai/bk-sdm-base/tree/0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0) |
| nota-ai/bk-sdm-small | [572238db7ed3a10858900803f3fc8cca53e893e0](https://huggingface.co/nota-ai/bk-sdm-small/tree/572238db7ed3a10858900803f3fc8cca53e893e0) |
| nota-ai/bk-sdm-tiny | [0364108e53b7f7f4d2585e817a0b7a83dc261cfa](https://huggingface.co/nota-ai/bk-sdm-tiny/tree/0364108e53b7f7f4d2585e817a0b7a83dc261cfa) |
| nota-ai/bk-sdm-v2-small | [ad82c51a3b568e855a7f79c5b893ea2f9ebb197f](https://huggingface.co/nota-ai/bk-sdm-v2-small/tree/ad82c51a3b568e855a7f79c5b893ea2f9ebb197f) |
| nota-ai/bk-sdm-v2-tiny | [68277af553777858cd47e133f92e4db47321bc74](https://huggingface.co/nota-ai/bk-sdm-v2-tiny/tree/68277af553777858cd47e133f92e4db47321bc74) |

Exclude [BK-SDM-v2-Base](https://huggingface.co/nota-ai/bk-sdm-v2-base/tree/21915be4c3a83f3fa45f93559750df44943509b1), despite its model card's rounded 0.98B whole-model figure.
Its full packaged stack contains 583,480,964 U-Net tensors, 340,387,917 text encoder tensors, and 83,653,863 VAE tensors: **1,007,522,744** in total.

Exclude [DeciDiffusion-v1-0](https://huggingface.co/Deci/DeciDiffusion-v1-0/tree/99d0c312e8e00e1bfb1aa89ad52b2a34f5829c2d), despite the card's advertised 820M and CPU fallback.
Its actual generation path replaces the default U-Net with `flexible_unet`; that component has 814,746,884 tensors, with another 123,060,480 in the text encoder and 83,653,863 in the VAE: **1,021,461,227** before any optional safety checker.
Counting only its denoiser would incorrectly admit an oversized inference stack.

## q.it implementation evidence

The task and provider compatibility contracts were inspected in [the runtime model definitions](../../qit-runtime/src/model.rs).
The actual image chat, multipart transcription, and provider dispatch paths were inspected in [the HTTP provider implementation](../../qit-runtime/src/provider.rs).
The [built-in benchmark guide](../built-in-benchmarks.md) confirms the image color question, speech tone transport check, and reserved text-to-image entry.
These are source-level compatibility findings, not completed CPU benchmark runs.
