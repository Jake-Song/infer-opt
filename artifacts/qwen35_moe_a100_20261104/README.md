# Qwen3.5 MoE on A100 80GB — 2026-11-04

**Interrupted experiment:** the Colab runtime disappeared during consistent k=1.
The backend subsequently reported no active sessions. The reason is unknown.
Only sequential k=1 and k=2 completed all 15 runs each; those are the table rows below.
Consistent k=1 has 8 partial rows in its log, all with token parity,
but no completed three-repeat aggregate. Parallel and compiled settings were not measured.
`progress.json` is the last remote snapshot, not evidence that a worker remains active.
No final worker exit or after-run device snapshot was retrieved. See `interruption.json`.

## Conditions

- Actual device: **NVIDIA A100-SXM4-80GB**, 79.250732 GiB visible, SM80.
- Torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0; provider default clocks, no clock locking.
- Checkpoint: `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision
  `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`; all 35 payload hashes verified.
- Inference code: `b5ae4228a361fa562063fac7cd46dd19a611e9f2`.
  `source_manifest.json` records the source bundle, including the new experiment runner.
- Batch 1; five prompts (`korean`, `math`, `code_1k`, `prose_1k`, `chat_long`),
  three repetitions per setting, 256 output tokens each.
- Each setting runs in an isolated process with its own matching ordinary-decoder baseline.
  Each decoder is warmed once before measurements; each prompt prefill is warmed before timing.
- Decode uses synchronized host elapsed time, excluding prefill and compilation.
  The first token comes from prefill; the time denominator is 255 subsequent tokens.
  MTP includes round overhead and any final-round overshoot.
- Stock kernel settings, no autotuning or BF16 GEMV overrides.
  The planned compiled MTP stages use `--allow-inexact-verification` for diagnostics.
- The initial 40GB session was stopped before model execution; these results are all from 80GB.

## Decode results

Ranges below are the minimum–maximum of **per-prompt medians across three repetitions**.
Divergence counts compare all 256 output tokens against the ordinary decoder in the same execution mode.
A passing setting means all recorded prompts and repeats passed; it is not a universal exactness guarantee.

| Execution | Verification | k | Plain ms/token | MTP ms/token | Divergent runs | Valid speedup |
|---|---|---:|---:|---:|---:|---:|
| cudagraph | sequential | 1 | 10.68–10.74 | 12.03–13.51 | 0/15 | 0.79–0.89× |
| cudagraph | sequential | 2 | 10.75–10.84 | 12.26–15.96 | 0/15 | 0.68–0.88× |

## Prose example: round cost and accepted output

This compares matched runs within each mode. Tokens/round includes the target token,
not just accepted drafts; small differences from ms/round divided by tokens/round
come from the final round overshooting the requested output length.

| Execution | Verification | k | Plain ms/token | Round ms | Tokens/round | MTP ms/token | Parity |
|---|---|---:|---:|---:|---:|---:|---|
| cudagraph | sequential | 1 | 10.744 | 23.802 | 1.984 | 12.041 | pass |
| cudagraph | sequential | 2 | 10.790 | 35.926 | 2.931 | 12.257 | pass |

## Validation and limits

- All ten expert-kernel cases (1–5 tokens, overlapping/disjoint routes) passed
  eager/CUDA Graph/compiled equality and dynamic-route checks.
  Maximum absolute error versus the independent INT8 reference: 6.103515625e-5.
- Strict checkpoint reload passed for all 41 quantized expert modules, with MTP present.
  The 64-token smoke test passed all four sequential/consistent k=1,2 parity checks.
- Maximum allocated memory in the detailed decode runs: 35.429 GiB.
  Allocated memory is not total device usage or proof of fitting on a 40GB device.
- This run does not include a new HF BF16 numerical comparison or task-level accuracy evaluation.
  Kernel correctness and greedy parity answer narrower questions.
- Historical 27B dense results used the same GPU model but different sessions and tuned settings;
  this is not a paired dense-versus-MoE benchmark.
- `checkpoint-validation.json` contains a short ordinary-decoder profile.
  It is not a profile of the compiled MTP round and does not isolate a divergence cause.
- Failed completed subprocess stages: [].

## Reproduce

```sh
python3 scripts/run_qwen3_5_moe_a100.py --out /path/to/results --repeats 3 --new-tokens 256
```

The runner requires an actual A100 with at least 75 GiB VRAM and downloads the public
pinned checkpoint using `token=False`. It needs Torch, Triton, Transformers, Safetensors,
Hugging Face Hub, and flash-linear-attention 0.6.0. Exact child commands are in `stages.json`.

## Artifacts and cleanup

- `environment.json`, `request.json`, `source_manifest.json`: hardware, software, requested scope, source hashes.
- `checkpoint-integrity.json`, `checkpoint-validation.json`, `moe-kernel-validation.json`: validation evidence.
- `summary.json`: per-prompt medians, divergence counts, and parity-gated speedups.
- `{mode}-{verification}-k{k}.json` and `.log`: raw measurements and execution logs.
- `stages.json`, `progress.json`: completed stage commands and last remote progress.
- `nvidia-smi-before.txt`: device and clock-state snapshot.
- `runtime_cleanup.json`: exact-session termination and backend session check.

Weights stayed in the dedicated Colab runtime; only small evidence files were collected locally.
No HF credentials were transferred to Colab.
