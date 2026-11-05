# Qwen3.5-35B-A3B: llama.cpp Q8_0 versus HF BF16

2026-11-05 KST. Completed on a real Colab G4. Runtime including build/download/reference/comparison: 582.75 seconds.

The comparison uses the same nine prompt builders and 48 teacher-forced
positions per prompt as `scripts/validate_qwen3_5_moe_quality.py` (432 positions).
The HF BF16 cached-greedy reference generates the continuation; llama.cpp
receives the exact prompt and continuation IDs. The first position is the last
prefill output and the next 47 positions are cached one-token decode. No MTP or
speculative calls are made. Full-vocabulary logits are compared using
`KL(HF || llama.cpp)`, top-1 agreement, and a strong mismatch when the HF top-1
minus top-2 logit margin exceeds 0.5.

This is numerical distribution preservation, not task accuracy. Differences
include quantization, execution kernels, and activation/cache precision.

Pinned identities:

- HF reference: `Qwen/Qwen3.5-35B-A3B`, revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- GGUF: `ggml-org/Qwen3.5-35B-A3B-GGUF`, revision
  `3127ef0b7fd4f12626506419ceefde54fa8db53c`,
  `Qwen3.5-35B-A3B-Q8_0.gguf`.
- llama.cpp: `c6a0ff11a0fe16ac1a2eb569190934593a2b3670`.
- Public infer-opt source for the frozen prompt helper:
  `a0a798494c295f1e6a850305944373356e33d982`.

Actual G4: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.970764 GiB,
SM120, CUDA 13.0, PyTorch 2.11.0+cu130. HF uses BF16 with SDPA. llama.cpp uses
full GPU layer offload, Flash Attention, F16 K/V, batch 2048, ubatch 512, and
four CPU threads. All HF-defined tokenizer entries must have matching token IDs
and text in the GGUF before logits are compared.

The reusable harness and commands are documented in
[`scripts/llamacpp_quality/README.md`](../../scripts/llamacpp_quality/README.md).
Before runtime allocation, seven CPU protocol/metric tests and native header
syntax checking passed. These checks alone are not real-model validation.

`launch.py` contains the exact harness payload sent to the dedicated G4;
`poll.py` reads progress. Downloaded model weights and caches are kept under
`/tmp/infer-opt-models` on the VM and excluded from repository artifacts.

## Results

All 432 HF/candidate logit positions were finite. Top-1 agreement was **426/432 (98.6111%)**, mean KL was **0.0021348794**, and strong mismatches were **0**.
The six top-1 differences had HF margins 0, 0, 0.0625, 0.375, 0.25, and 0.5, all within the existing near-tie definition.

| Prompt | Prefix tokens | Mean KL(HF || Q8_0) | Top-1 matches | Strong mismatches |
| --- | ---: | ---: | ---: | ---: |
| short | 5 | 0.00149110 | 48/48 | 0 |
| hello | 1 | 0.00246000 | 46/48 | 0 |
| korean | 20 | 0.00254725 | 47/48 | 0 |
| math | 43 | 0.00208480 | 48/48 | 0 |
| chat | 23 | 0.00052034 | 48/48 | 0 |
| chat_long | 32 | 0.00116035 | 48/48 | 0 |
| prose_1k | 1057 | 0.00082210 | 48/48 | 0 |
| code_1k | 1000 | 0.00211283 | 47/48 | 0 |
| code_1_5k | 1500 | 0.00601515 | 46/48 | 0 |

The nine last-prefill positions had mean KL 0.0125502554 and 9/9 top-1 matches. The 423 single-token decode positions had mean KL 0.0019132757 and 417/423 top-1 matches. Per-step details and HF margins are in `results/quality.json`.

## Historical infer-opt context

| Candidate | Mean KL versus its HF reference | Top-1 matches | Strong mismatches |
| --- | ---: | ---: | ---: |
| Historical infer-opt BF16 | 0.00174283 | 429/432 (99.3056%) | 0 |
| Historical infer-opt INT8 group-128 | 0.00201384 | 429/432 (99.3056%) | 0 |
| This llama.cpp Q8_0 run | 0.00213488 | 426/432 (98.6111%) | 0 |

The historical rows are recomputed from [`../qwen35_moe_g4/quality.json`](../qwen35_moe_g4/quality.json). This is **not a paired comparison**: the original raw HF logits and complete software identity were not retained. This run freshly generated HF logits using Transformers 5.22.0. Same prompt builders/model revision/metric definitions do not establish identical reference logits. No quality winner or isolated quantization effect is claimed.

## Preserved evidence and cleanup

- `results/quality.json`: per-prompt and per-step distribution metrics, argmax IDs, and HF margins.
- `results/reference.json`: exact prompt text/IDs and HF-generated continuation IDs; code prompt SHA-256.
- `results/llama.stderr`: verified 248,077 HF-defined token mappings, 248,320 output vocabulary rows, 41/41 GPU offload, Flash Attention, F16 K/V, and F32 recurrent state. Context capacity was rounded to 1,792 cells for every prompt.
- `results/environment.json`, `manifest.json`, build/install/command logs, and exact harness source: runtime provenance. GGUF SHA-256 matched the previous performance benchmark and pinned Hub LFS metadata.
- `comparison_summary.json`, `verification.json`: locally checked aggregation, phase split, historical context, and source hashes.
- `results.tar.gz`: raw downloaded report archive, excludes weights/full-logit scratch buffers.
- `cleanup.json`: the exact session was stopped after result retrieval; the server reported no active sessions.

No performance measurements, free-running parity checks, task benchmarks, model publication, commit, or push were performed in this run.
