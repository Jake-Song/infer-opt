# Qwen3.5-27B native NVFP4 on G4

Published checkpoint: [Jakemu/Qwen3.5-27B-nvfp4-mtp](https://huggingface.co/Jakemu/Qwen3.5-27B-nvfp4-mtp).
Weight publication commit: `c94aa9d4e6f6a738c2bb06cc6aa51cbb783c15e3`.
License/requirements follow-up: `ae669958096ec926a3229601dc854cb6dba71f86`.
Original Qwen revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`.

This is a **custom infer-opt checkpoint**, including the loader source. It is not
a Transformers or vLLM checkpoint. The 256 large target Linears use native NVFP4
W4A4 with 16-value FP8 block scales and FP32 global scales. The MTP head uses the
same group-128 INT8 recipe as the A100 release; embedding, lm_head, small gates,
normalization and recurrent state retain floating-point precision. No vision tower.

## Verified release properties

- Total weight shard size: **19,253,533,552 bytes (17.931 GiB)** including MTP.
- Saved/reloaded final-position logits are exactly equal on the short release smoke prompt.
- Native kernel checks passed **18 shapes**: M=1/2/3/4/128/1024 across three real
  projection shapes. FP32 matmul of reconstructed FP4 values is the numerical oracle;
  maximum relative L2 error is **0.001729** (BF16 output). Eager, fullgraph compile
  and captured CUDA Graph outputs agree; zero inputs produce zero outputs.
- Explicit FlashInfer 0.6.6 CUTLASS backend on RTX PRO 6000 Blackwell Server Edition,
  SM120, torch 2.11.0+cu130. No W4A16 or floating-point-weight fallback.
- Large-M capture discrepancies were observed without tiling. The verified adapter
  bounds each native GEMM to 128 activation rows and disables PDL.
- ModelOpt 0.47.0 calibration uses 128 UltraChat train_sft conversations, each up to
  512 tokens, independently of the nine evaluation prompts. Revision and token hashes
  are in `calibration.json`; module scales are in `quantization_report.json`.

These are storage/kernel checks, **not the full model accuracy gate**. End-to-end
performance and accuracy evaluation are separate. BF16 general decoding was measured;
BF16-target MTP timing was discontinued to prioritize INT8-versus-NVFP4 measurements.
Every measured MTP configuration uses an INT8 head and BF16 lm_head.

## Artifacts

`native_smoke.json`, `export_validation.json`, `environment.json`, `source_manifest.json`,
`calibration.json` and `quantization_report.json` preserve the release evidence.
The source hashes refer to the actual uploaded runtime, which can differ from later
benchmark/orchestration changes. Predicted latency is not a measurement.

## Final partial evaluation

The user requested finalization before the comparison finished. All G4 sessions
were stopped. Native checkpoint download from the Hub and generation succeeded.

- NVFP4 plain decode: **22.45–22.49 ms/token (44.5 tok/s)** across five prompts,
  256 output tokens, three warmups and five timed repetitions per prompt.
- NVFP4 + INT8 MTP k=1: **12.98–14.28 ms/token**; k=2: **9.71–11.56 ms/token**.
  k=3 completed only chat_long (**10.19 ms/token**) before finalization.
- MTP output agreement failed: same-prefix diagnostics found strong top-1
  mismatches on chat, code and math. MTP timings are diagnostic, not proof of a
  greedy-equivalent speedup. Diagnostics cover up to 64 forced positions.
- BF16 plain decode was **35.94–36.06 ms/token in an earlier G4 session**. This
  is a separate session, not a paired same-device comparison.
- Same-session INT8 comparison, full HF accuracy gate, remaining k=3 cases,
  prefill sweep and kernel profiling were **not completed**. Quality-preserving
  improvement over the existing INT8 path has not been established.

Raw partial results are `nvfp4_measurements.json` and `bf16_measurements.json`.
`report.md` and `result_status.json` distinguish measured and missing evidence.

## Completed follow-up

The [completed same-session evaluation](../qwen3_5_27b_nvfp4_g4_completion_20261028/README.md)
finishes the INT8 comparison, full model accuracy gate, all MTP cases, prefill and
Linear profiling. The partial measurements above remain the historical record.
