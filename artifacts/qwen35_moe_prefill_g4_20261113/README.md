# G4 MoE prefill: stable route sorting, 2026-11-13

Stable route sorting reduced warm batch-one text-model TTFT by **1.23–1.63×**
on eight measured Qwen3.5-35B-A3B INT8 cases. All eight last-position logit
comparisons were bit-exact (KL zero), and all sixteen-token continuations
matched. The optimization keeps the existing expert arithmetic and changes
routing preparation. It is available as an explicit `prefill_backend='sorted'`
option; the default remains `reference`.

The separate [theoretical lower-bound calculation](PREFILL_LOWER_BOUND.md)
estimates a 22 ms optimistic major-GEMM/weight-read roofline at 2048 tokens,
with a conditional bulk-layer stage estimate of 33 ms. It records omitted
costs and routing-coverage assumptions; these are not measured speedups.

## Environment and contract

- RTX PRO 6000 Blackwell Server Edition, 94.970764 GiB, SM 12.0; provider automatic boost.
- PyTorch 2.11.0+cu130, Triton 3.6.0, FLA 0.6.0; eager text-model prefill, MTP off.
- Published `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision
  `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`; loaded weights unchanged.
- Repository base `7ad98bd8f4bee19cb04920eb05d2f831279eed6c`, with hashed experimental sources retained.
- Batch 1, 40 target expert banks, 82 fused projection groups, shared static cache capacity 4120.
- Math/Korean fixtures repeated and truncated to exact 128/512/2048/4096 token lengths;
  these are controlled token workloads, not natural long-document quality tests.
- Two warmups per shape/backend, five timed repetitions. Reference precedes sorted
  within each case; measurements are sequential and clocks were not locked.

TTFT includes the text model, last-position BF16 vocabulary projection and argmax.
Tokenization, cache reset, JIT compilation, warmup, sixteen-token continuation checks,
and profiling are excluded. CUDA-event intervals and synchronized host wall time
are retained; an event interval includes host-induced GPU idle gaps and is not
summed kernel duration. The table uses median synchronized host milliseconds.

| Fixture | Prompt tokens | Reference TTFT ms | Sorted TTFT ms | Speedup |
|---|---:|---:|---:|---:|
| math | 128 | 873.84 | 709.19 | 1.232× |
| math | 512 | 1266.07 | 860.22 | 1.472× |
| math | 2048 | 1383.12 | 915.13 | 1.511× |
| math | 4096 | 1446.11 | 985.30 | 1.468× |
| korean | 128 | 972.70 | 595.39 | 1.634× |
| korean | 512 | 1049.61 | 666.14 | 1.576× |
| korean | 2048 | 1187.17 | 745.16 | 1.593× |
| korean | 4096 | 1268.07 | 821.32 | 1.544× |

Peak allocated memory across model cases was about 34.78–35.37 GiB, including
weights and the shared static cache. Reserved memory reached 36.23 GiB. This
experiment does not establish a memory reduction or fitting on a different GPU.

## Mechanism and numerical checks

The reference performs `torch.where(indices == expert)` for each of 256 experts.
The sorted path sorts flat token/slot assignments stably, reads all 256 counts
once, and slices each expert's assignments. It retains ascending expert order,
token/slot order, the same per-expert `int8_linear` dispatch, SwiGLU rounding and
`index_add_` accumulation. Experts with at most eight assigned rows still use the
existing small-row GEMV; larger matrices still use dequantization plus PyTorch
linear/cuBLAS. The one-to-five-row selected-expert decode/verification path is
unchanged. No new GEMM kernel is used by the sorted-path timings.

Independent profile aggregation confirms, outside timed measurements:

| Scope, 128-token input | Reference `nonzero` | Sorted `nonzero` | Reference stream syncs | Sorted stream syncs |
|---|---:|---:|---:|---:|
| Synthetic expert bank | 256 | 0 | 256 | 1 |
| Full model, math | 10240 | 0 | 10240 | 40 |
| Full model, Korean | 10240 | 0 | 10240 | 40 |

See [profile-summary.json](profile-summary.json). Profiled durations are separate
from the unprofiled TTFT table and are not used as speedups.

Nine real-shape synthetic cases (128/512/2048 tokens × spread/shared/skewed
routing) had exact outputs. For spread routing at 2048 tokens the expert block
changed from 34.236 to 23.280 ms (1.471×); fully shared routing is a favorable
microbenchmark and must not be substituted for model throughput.

The G4 related test selection passed **34 tests**, with 20 unrelated speculative
or verification cases deselected. The local related run passed 24 with 30 CUDA
cases skipped. The recovered experiment independently verifies 90 synthetic
and 80 model timing samples, 256 continuation IDs, input lengths, medians,
ratios, execution-source hashes and 183 internal file hashes.

Exact full-model checks concern last-position prefill logits and one sixteen-token
continuation per backend/case. Cache tensors were not compared independently.
HF BF16 equivalence, task accuracy, batch sizes above one, MTP throughput and
other GPUs were not measured.

## Grouped GEMM candidate: not adopted

`infer_opt/moe_prefill.py` contains a separate explicit experimental GPU route-grouped
W8A16 GEMM. Its first compile failed because an INT64 routing ID changed the type
of an INT32 loop variable in the new combine kernel. This was a new-candidate
error, not an error in the existing prefill path; it was fixed before numerical tests.

Small-shape tests passed, but all 36 real-shape comparisons failed the retained
`atol=2e-4, rtol=.03` criterion. Candidate timings were not accepted and the full
model was not run with this backend. Independent dequantized comparisons also
failed that criterion, so the difference is not solely the legacy small-row GEMV
dispatch. This is a numerical investigation candidate, with no established
full-model parity or speedup. The tolerance was not relaxed to pass it.

The failed runs are retained under [results/attempt2](results/attempt2), including
[the independent diagnosis](results/attempt2/dequant-diagnosis.json). Its nested
`attempt1` records the initial compile failure.

## Reproduce and evidence

Enable the measured route preparation explicitly:

```python
from infer_opt.moe import Int8Experts
for module in model.model.language_model.modules():
    if isinstance(module, Int8Experts):
        module.prefill_backend = 'sorted'
```

Run instructions and source-bundle helpers are in
[scripts/moe_prefill_g4](../../scripts/moe_prefill_g4/README.md).
[validation.json](validation.json) independently checks the aggregation;
[results/model/model.json](results/model/model.json) retains samples and generated IDs;
[results/model/inputs.json](results/model/inputs.json) freezes exact inputs.
Actual executed sources are under [results/executed-source](results/executed-source).
[results.tar.gz](results.tar.gz) retains all eight large raw Chrome traces and
[recovery.json](recovery.json) records the archive size and SHA-256. Aggregate
profiles are extracted for review; raw traces stay compressed to avoid adding
more than a gigabyte of unpacked JSON to the repository.

```sh
python scripts/moe_prefill_g4/validate.py \
  --results artifacts/qwen35_moe_prefill_g4_20261113/results \
  --archive artifacts/qwen35_moe_prefill_g4_20261113/results.tar.gz
```

The initial G4 allocation with `--high-mem` was rejected; retrying without it,
at the user's request, succeeded. Model and compiler caches stayed under
`/tmp/infer-opt-models/moe-prefill-g4-20261113`. Results were recovered and checked
before stopping the exact dedicated session; see `lifecycle.json` for confirmation.
