# G4 MoE prefill: route-grouped experts, 2026-11-14

The route-grouped expert kernel reduced warm batch-one text-model TTFT to
**52–196 ms** on the eight Qwen3.5-35B-A3B INT8 cases, **4.3–15.5×** faster than
stable route sorting and **6.6–22.4×** faster than the reference loop. It
passed the FP64-referenced expert gate in all 45 comparisons and the model gate
in all eight cases: last-position logit KL at most 8.4e-4 and every
sixteen-token continuation equal to the reference's. `grouped` is still an
explicit opt-in; `reference` remains the default.

The same run measured what is left: at 2048 tokens the routed experts are now
about a third of prefill, and GatedDeltaNet (projections plus core) is larger.

## Environment and contract

- RTX PRO 6000 Blackwell Server Edition, 94.97 GiB, SM 12.0; provider automatic boost.
- PyTorch 2.11.0+cu130, Triton 3.6.0, FLA 0.6.0; eager text-model prefill, MTP off.
- Published `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision
  `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`; loaded weights unchanged.
- Sources: uncommitted changes on `9ed8e76af973d6bca0681f6575c0d3fb288cd94a`;
  the executed files and hashes are retained.
- Batch 1, 40 expert banks, 82 fused projection groups, shared static cache capacity 4120.
- Math/Korean fixtures repeated and truncated to exact 128/512/2048/4096 token lengths.
  These are controlled token workloads, not natural long documents; the routing
  loads below describe these repeated fixtures.
- Two warmups, five timed repetitions per shape and backend, in the order
  reference, sorted, grouped; sequential, clocks not locked.

TTFT includes the text model, last-position BF16 vocabulary projection and
argmax, and excludes tokenization, cache reset, compilation, warmup, the
continuation check and all profiling. The table is median synchronized host
milliseconds.

| Fixture | Tokens | Reference | Sorted | Grouped | Grouped vs sorted | KL(ref ‖ grouped) |
|---|---:|---:|---:|---:|---:|---:|
| math | 128 | 896.3 | 729.8 | 52.5 | 13.9× | 8.4e-4 |
| math | 512 | 1292.5 | 889.5 | 57.6 | 15.5× | 6.4e-5 |
| math | 2048 | 1405.5 | 938.2 | 106.0 | 8.8× | 2.2e-5 |
| math | 4096 | 1479.2 | 1004.7 | 196.2 | 5.1× | 4.7e-7 |
| korean | 128 | 996.0 | 610.8 | 52.3 | 11.7× | 1.4e-4 |
| korean | 512 | 1070.9 | 690.1 | 52.9 | 13.0× | 9.4e-5 |
| korean | 2048 | 1216.9 | 764.5 | 105.0 | 7.3× | 3.7e-5 |
| korean | 4096 | 1300.1 | 841.4 | 195.7 | 4.3× | 2.1e-6 |

Sorted stayed bit-exact with the reference in all eight cases. Peak allocated
memory was 34.79–35.37 GiB for every backend. At 2048 tokens the grouped TTFT is
3.2× the 33 ms bulk-stage estimate and 4.8× the 22 ms roofline of the
[lower-bound calculation](../qwen35_moe_prefill_g4_20261113/PREFILL_LOWER_BOUND.md).

## Acceptance

The criterion is the one adopted on 2026-11-14 (see the
[numerical diagnosis](../qwen35_moe_prefill_numerics_20261114/README.md)).

- **Expert level**: on one synthetic real-shape bank, the grouped RMS error to
  FP64 was 0.554–0.561% against 0.564–0.675% for the adopted loop, in all nine
  routing cases and five tile shapes.
- **Model level**: the largest KL, 8.4e-4 at math-128, is under the provisional
  2e-3 ceiling; the smallest reference top-2 logit margin over that
  continuation was 0.25. Maximum last-position logit differences were
  0.72–2.64. Only last-position logits and one sixteen-token continuation per
  case were compared; cache tensors, HF BF16 equivalence, task accuracy, MTP,
  batch sizes above one and other GPUs were not measured.

## The 2026-11-13 difference, explained

The previous grouped kernel was 0.46–0.60% from the dequantized loop at 16–64
rows per expert, which a local GPU did not reproduce. On this GPU the cause is
the adopted path: `F.linear` on the dequantized gate/up matrix is not the
correctly rounded product at some row counts.

| Rows per expert (case) | gate/up within half a BF16 unit | RMS error to its own FP64 value |
|---|---:|---:|
| 4 (128 spread) | 99.99% | 0.166% |
| 16 (512 spread) | 79.8% | 0.202% |
| 63 (2048 spread) | 61.7% | 0.230% |
| 128 (128 shared) | 58.9% | 0.235% |
| 512, 2048 (shared) | 99.90% | 0.166% |

The extra error is unbiased and about the size of one more BF16 rounding. The
down projection (2048×512) is correctly rounded at every row count, and the
grouped kernel is correctly rounded (99.99%) at both projections in every tile.
Which cuBLAS kernel does this was not determined.

## Where the time goes now

Median synchronized section times at 2048 tokens, three passes. Synchronizing
at each boundary attributes cost but removes host/device overlap, so the
sections add up to more than the unprofiled TTFT (123 ms against 106 ms).

| Section | Sorted, math | Grouped, math | Grouped, korean |
|---|---:|---:|---:|
| Routed experts | 871.1 | 42.2 | 41.3 |
| GatedDeltaNet core (30 layers) | 30.6 | 29.4 | 29.4 |
| GatedDeltaNet projections | 19.7 | 19.5 | 19.5 |
| Layer norms | 7.2 | 6.8 | 6.8 |
| Full-attention core (10 layers) | 5.6 | 5.5 | 5.6 |
| Full-attention projections | 5.0 | 5.0 | 5.0 |
| Routing softmax/top-k and sums | 5.7 | 5.3 | 5.3 |
| Shared expert | 4.4 | 4.8 | 4.9 |
| Router, shared gate, glue, lm_head | 5.6 | 5.5 | 5.5 |
| Text model total | 954.5 | 123.4 | 122.7 |

At 128 tokens the grouped sections are 23–26 ms routed experts and 23 ms
GatedDeltaNet core of about 80 ms; at 4096 tokens 72 ms and 63 ms of 213 ms.
The 128-token profile shows 4,179 kernel launches per grouped prefill against
68,255 sorted and 114,688 reference, with 80 grouped GEMM launches.

## Routing loads

Per layer, over 40 layers ([results/model/routes.json](results/model/routes.json)
keeps all counts).

| Fixture | Tokens | Used experts min / median / max | Median rows per used expert | Largest expert, rows | Experts with ≤8 rows, median | Rows in top-8 experts | Tile fill 16 / 32 / 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 128 | 122 / 140.5 / 174 | 4 | 110 | 104.5 | 29% | 0.39 / 0.22 / 0.11 |
| math | 2048 | 155 / 206 / 232 | 39 | 1667 | 49 | 23% | 0.90 / 0.81 / 0.67 |
| math | 4096 | 160 / 211 / 237 | 72.5 | 3536 | 39.5 | 24% | 0.95 / 0.89 / 0.80 |
| korean | 128 | 68 / 126.5 / 145 | 4 | 128 | 89 | 32% | 0.43 / 0.24 / 0.13 |
| korean | 2048 | 87 / 160.5 / 211 | 32 | 2048 | 52 | 31% | 0.92 / 0.84 / 0.71 |
| korean | 4096 | 95 / 171.5 / 213 | 53.5 | 4096 | 45.5 | 30% | 0.96 / 0.91 / 0.82 |

Real routing is neither the uniform nor the fully shared synthetic case: at
2048 tokens a median 63–80% of experts are used per layer, single experts take
up to every token, and a quarter to a third of the used experts still have at
most eight rows.

## Synthetic expert bank

One real-shape bank, median wall milliseconds. All 45 grouped configurations
passed the FP64 gate and were timed.

| Tokens | Routing | Reference | Sorted | Grouped 16×64×64 | Best grouped tile |
|---:|---|---:|---:|---:|---|
| 128 | spread | 32.53 | 24.31 | 0.88 | 0.70 (16×64×128) |
| 512 | spread | 32.77 | 23.87 | 0.89 | 0.73 (16×64×128) |
| 2048 | spread | 34.23 | 23.53 | 1.09 | 1.04 (32×64×64) |
| 2048 | skewed | 35.07 | 23.99 | 1.09 | 1.06 (32×64×64) |
| 2048 | shared | 9.84 | 1.18 | 0.95 | 0.64 (64×64×64) |

The model used the default 16×64×64 tile. Tile choice moves the synthetic layer
by up to 1.5×, in different directions for few-row and many-row experts; no
tile was tuned in this run.

## Reproduce and evidence

```python
from infer_opt.moe import Int8Experts
for module in model.model.language_model.modules():
    if isinstance(module, Int8Experts):
        module.prefill_backend = 'grouped'
```

Run instructions are in [scripts/moe_prefill_g4](../../scripts/moe_prefill_g4/README.md)
(`prepare.py --run 20261114`). [validation.json](validation.json) independently
rechecks 140 file hashes, 435 timing samples, 384 continuation IDs, the 45 + 45
FP64 gates, the eight model gates, route totals and executed-source hashes:

```sh
python scripts/moe_prefill_g4/validate.py \
  --results artifacts/qwen35_moe_prefill_g4_20261114/results \
  --archive artifacts/qwen35_moe_prefill_g4_20261114/results.tar.gz
```

[results/numerics/numerics.json](results/numerics/numerics.json) holds the
stage-wise numerics, [results/model/model.json](results/model/model.json) the
samples, section times, generated IDs and top-2 margins.
The recovered archive `results.tar.gz` (73.9 MB, nine raw Chrome traces, 1.24 GB
unpacked) is not kept in the repository; [recovery.json](recovery.json) records
its size and SHA-256, and the aggregate profiles are extracted under
`results/model` and `results/experts`. Without the archive, omit `--archive`:
everything except the nine raw traces and the remote run's four pytest cache
files is still verified. The G4 was
allocated without `--high-mem` with the user's approval, ran 671 s of worker
time, and was stopped after verified recovery; see [lifecycle.json](lifecycle.json).
