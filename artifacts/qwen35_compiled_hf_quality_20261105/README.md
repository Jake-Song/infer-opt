# Qwen3.5-35B-A3B INT8: paired eager/compiled quality against HF BF16

2026-11-05 KST. Completed on the user-approved G4 retry. The first allocation failed with HTTP 503. Runtime for the successful job was 537.81 seconds, including downloads/reference/evaluation.

This experiment uses one shared HF BF16 cached-greedy reference for the original
published custom INT8 checkpoint in eager and full-model
`torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)` decode.
It reproduces the nine prompt builders and 48 teacher-forced positions per prompt
used by the prior numerical comparison. Position zero is an eager last-prefill
output; positions 1–47 use the selected single-token decode mode.

Two different contracts are reported separately:

- HF reference: full-vocabulary `KL(HF || candidate)`, top-1 agreement, and a
  strong mismatch when the HF top-1/top-2 logit margin exceeds 0.5.
- Compiled versus same-checkpoint eager: the same distribution metrics plus
  elementwise `allclose(atol=0.02, rtol=0.02)` at every position.

The experiment does not change the model implementation or relax tolerances.
It does not measure task accuracy or decoding throughput.

## Pinned scope

- HF reference: `Qwen/Qwen3.5-35B-A3B`, revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`, BF16 with SDPA.
- Custom checkpoint: `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision
  `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`. No requantization.
- Inference source: public infer-opt revision
  `a0a798494c295f1e6a850305944373356e33d982`.
- Batch 1, MTP loaded but not called, `fla-delta`, fused projections, static cache
  capacity 8456 to match the recent speed experiment. No BF16 GEMV override or
  max-autotune. Eight warmup steps precede each mode; cache is reset afterwards.
- HF and candidate models run in separate processes. All downloaded weights and
  caches live under `/tmp/infer-opt-models` on the dedicated runtime.
- Transformers 5.22.0, Accelerate 1.15.0, Safetensors 0.8.0, HF Hub 1.33.0,
  flash-linear-attention/fla-core 0.6.0. Actual GPU/PyTorch/Triton identities are
  recorded by the worker.

The harness is `scripts/validate_compiled_qwen3_5_moe_quality.py`, using the
shared metric/reference helpers in `validate_llamacpp_qwen3_5_quality.py`.
Before allocation, nine CPU tests passed. The new tests demonstrate that equal
probability distributions can still fail elementwise logit tolerance.

`worker.py` and `launch.py` preserve the exact runtime orchestration/payload;
`poll.py` reads progress. The worker archives reports, exact inputs and reference
continuation, per-case reference/candidate logit hashes, source, and logs. Weights
and full-logit scratch buffers are excluded.

The first manual allocation attempt used the installed CLI retry behavior and failed before a runtime became READY. See `allocation-failure.json`. The user subsequently authorized one G4 retry; see `allocation-retry.json`.

## Results

Both modes used exactly the same HF reference logit hashes, prompt IDs and forced
continuations within this run. All logits were finite.

| Candidate | Mean KL(HF || candidate) | Top-1 agreement | Strong mismatches |
| --- | ---: | ---: | ---: |
| INT8 eager | 0.0018939932 | 428/432 (99.0741%) | 0 |
| INT8 compile-reduce-overhead | 0.0020585745 | 430/432 (99.5370%) | 0 |

Compiled top-1 differences occurred at `korean` step 29 (HF margin 0.125) and
`code_1_5k` step 38 (HF margin 0.25). Both are near ties under the existing
margin >0.5 strong-mismatch rule. A higher top-1 agreement on this small replay
suite is not a task-accuracy improvement claim. No new KL acceptance threshold
was introduced.

The strict compiled/eager logit test **failed**: all 423 actual decode positions
failed `allclose(atol=rtol=0.02)`. The nine shared eager-prefill positions passed,
so the total was 9/432. Maximum absolute compiled/eager logit difference was
8.6875; their top-1 selections differed at four positions. The maximum is over
all vocabulary logits and is not a probability error.

This completes the HF distribution measurement, but does not establish an
allclose-equivalent compiled decoder. It does not resolve the first diverging
operation or validate a throughput improvement under the strict eager gate.

| Prompt | Eager KL | Compiled KL | Eager matches | Compiled matches | Compiled/eager max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 0.00100881 | 0.00221508 | 48/48 | 48/48 | 8.671875 |
| hello | 0.00169289 | 0.00204632 | 47/48 | 48/48 | 0.960938 |
| korean | 0.00310798 | 0.00277268 | 47/48 | 47/48 | 1.312500 |
| math | 0.00150921 | 0.00183952 | 48/48 | 48/48 | 1.187500 |
| chat | 0.00077444 | 0.00151627 | 48/48 | 48/48 | 2.953125 |
| chat_long | 0.00031433 | 0.00059883 | 48/48 | 48/48 | 1.781250 |
| prose_1k | 0.00049343 | 0.00038823 | 48/48 | 48/48 | 8.687500 |
| code_1k | 0.00170190 | 0.00183423 | 48/48 | 48/48 | 1.203125 |
| code_1_5k | 0.00644296 | 0.00531600 | 46/48 | 47/48 | 1.375000 |

The nine last-prefill outputs are eager in both modes and match each other.
Across the 423 decode positions, mean HF KL was 0.0018142013 for eager and
0.0019822842 for compiled. Preparation including eight warmup steps took
1.79 seconds for eager and 51.22 seconds for compiled; these are setup times,
not decode performance measurements.

## Relation to the preceding llama.cpp result

The preceding llama.cpp run is **not paired** with this experiment. All nine
prompt ID sequences match, but HF-generated continuations differ on `code_1k`
and `code_1_5k`. The earlier HF stderr reports reference PyTorch fallback for
DeltaNet because FLA was absent; the current environment installs FLA 0.6.0.
The effect of this reference implementation difference was not isolated. Do not
rank llama.cpp and compiled infer-opt quality by comparing their aggregate
metrics from these separate runs.

## Evidence and cleanup

- `results/quality.json`: per-mode/per-prompt/per-step HF metrics, margins and
  raw logit hashes. Equal HF hashes across modes verify the paired reference.
- `results/compiled-vs-eager.json`: the separate direct comparison, per-position
  allclose flags and distribution metrics.
- `results/reference.json`, `reference-logit-file-hashes.json`: exact inputs,
  HF continuation, and SHA-256 of temporary HF reference files.
- `results/checkpoint-hashes.json`, `manifest.json`, `load.json`: pinned source,
  all 34 downloaded payload files, 22 verified weight shards, strict load,
  82 fused groups, BF16 activations, and cache capacity 8456. No MTP calls.
- `results/environment.json`: RTX PRO 6000 Blackwell Server Edition,
  94.970764 GiB, CUDA 13.0, PyTorch 2.11.0+cu130, Triton 3.6.0.
- `results/worker.py` and the two validator sources: exact deployed harness.
- `results.tar.gz`: raw recovered archive, excluding model weights and full
  logit scratch buffers. `comparison_summary.json` and `verification.json`
  contain locally checked aggregation and source hashes.
- `connection-recovery.json`: the polling websocket disconnected while the
  independent worker remained active. File transfer recovered the full result;
  worker exit code was zero. No model-execution failure was inferred from the
  polling error.
- `cleanup.json`: the runtime was stopped after recovery; the server reported
  no active sessions. The first failed allocation and its cleanup remain in
  separate `allocation-failure*.json` records.
