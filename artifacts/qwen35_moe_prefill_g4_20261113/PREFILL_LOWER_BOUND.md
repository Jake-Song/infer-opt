# Optimistic W8A16 prefill lower-bound calculation

For this Qwen3.5-35B-A3B INT8 group128 checkpoint on one G4, the optimistic
major-GEMM/weight-read roofline is **about 22 ms at 2048 tokens** and **46 ms
at 4096 tokens**. A conditional bulk-layer attention/MoE stage model gives
**33 ms** and **47 ms**, respectively. These are modeling floors with omitted
costs, not measured kernel performance or achievable latency promises.

## Hardware rates

[NVIDIA's server specification](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
provides 1597 GB/s GDDR7 bandwidth. The dense BF16 rate is an estimate derived
from [NVIDIA's RTX PRO Blackwell architecture whitepaper, Table 4](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf):
503.8 dense BF16 TFLOP/s at workstation boost 2617 MHz, scaled to the recorded
server maximum boost 2430 MHz, gives 467.8005 TFLOP/s. The server/workstation
have the same enabled core/SM counts. This is a peak-throughput derivation,
not an actual GEMM calibration of the benchmark runtime.

The 2430 MHz server maximum is retained in the preceding same-GPU-type
[hardware report](../qwen35_moe_compiled_g4_20261112/results/gpu-before.stdout).
The prefill run used automatic boost and its actual sampled clocks vary.
W8A16 uses BF16 activations/matrix arithmetic, so dense BF16 performance is used;
INT8 TOPS and structured-sparse Tensor Core peaks are not applicable to this
unchanged checkpoint arithmetic.

## Operation accounting

Configuration: H=2048, expert intermediate M=512, 40 MoE layers, top-k=8 of
256 experts, 30 GatedDeltaNet layers, 10 full-attention layers, Q width 4096,
KV width 512, vocabulary 248320. FMA counts as two FLOPs. Batch one, no prior
context, MTP off, and lm_head is applied to the final prompt position only.

Active linear-weight elements per prompt token:

| Component | Active parameter elements |
|---|---:|
| Routed experts | 1,006,632,960 |
| Shared experts | 125,829,120 |
| GatedDeltaNet input/output projections | 1,010,565,120 |
| Full-attention projections | 272,629,760 |
| Routers and shared-expert gates | 21,053,440 |
| Total | 2,436,710,400 |

This sum is independently checked against the retained tensor metadata. Embedding
lookup does not multiply by the full embedding matrix, and the vocabulary
projection is not performed at every prompt position in this benchmark.

For T prompt tokens, major projection plus useful causal full-attention FLOPs are:

```text
F(T) = 4,873,420,800 T + 81,920 T(T+1) + 1,017,118,720
```

The quadratic term is the useful causal QK/AV work in the ten full-attention
layers, including the diagonal. The formula omits the GatedDeltaNet core,
convolution, normalization, nonlinearities, routing/top-k, softmax, INT8
conversion and padding/masked-out work. Thus it is a deliberately optimistic
subset of the actual prefill work, not an exact total FLOP counter.

## Weight traffic and route coverage

If every expert is used at least once per layer, the minimal single-scan weight
and scale accounting is **35,212,027,136 bytes**:

- Routed expert INT8 banks and BF16 scales: 32,715,571,200 bytes.
- Other target weights, including shared experts/routers/attention and BF16
  lm_head: 2,496,455,936 bytes.
- MTP weights and the full embedding table are excluded.

Routed bytes follow `40 * 256 * 3 * 2048 * 512 * (1 + 2/128)`.
The weights-only memory floor is `35.212 GB / 1597 GB/s = 22.049 ms`.
This assumes ideal fused dequantization and one GDDR read per used weight/scale;
activation traffic, KV/state writes, repeated tile loads and small L2-residency
corrections are omitted.

Actual unique-expert counts were not saved. Full expert coverage is a scenario,
not an observation about these prompts. If only U experts are used per layer,
weight bytes are `2.496 GB + 32.716 GB * U/256`. The extreme all-tokens-share-eight
case gives only 3.519 GB and a 2.203 ms weights-only floor. Under independent
uniform top-8 routing the expected coverage is
`256 * (1 - (1 - 8/256)**T)`: 251.60 experts at T=128 and essentially all 256 at
T>=512. Real learned routers need not follow that distribution.

## Roofline results

Global optimistic floor: `max(F(T)/P_dense, D_used/B_GDDR)`.
The table's global and stage columns assume full expert coverage.
At 2048/4096 tokens the major-GEMM compute floor already dominates even the
full-coverage memory estimate, so those global values also hold for narrower
expert coverage within this model.

| Tokens | Major GEMM TFLOP | Compute floor ms | Global full-coverage floor ms | Bulk-stage model ms | Measured sorted TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.626 | 1.34 | 22.05 | 22.05 | 595–709 |
| 512 | 2.518 | 5.38 | 22.05 | 24.08 | 666–860 |
| 2048 | 10.326 | 22.07 | 22.07 | 33.20 | 745–915 |
| 4096 | 21.337 | 45.61 | 45.61 | 46.64 | 821–985 |

The bulk-stage model sums ideal attention-stage and MoE-stage rooflines across
layers and the final lm_head stage. It permits ideal compute/memory overlap
inside each coarse stage, but assumes those whole-layer stages execute
sequentially. It excludes the same extra operations and activation traffic.
It is a conditional scheduling model, not a universal lower bound for arbitrary
chunk/interlayer pipelining. At 2048 tokens its components are approximately
12.0 ms attention projections/core, 20.6 ms MoE weights/work and 0.64 ms lm_head.

The global full-coverage compute/memory crossover is near 2046 tokens. The
routed expert weights alone take 20.49 ms to read once, and average uniform
rows per expert at 2048 tokens are only `2048*8/256 = 64`. This explains why an
individual expert GEMM can remain memory-limited even near the global crossover.
It does not establish that today's kernels saturate bandwidth.

At T=2048 the current 745–915 ms sorted TTFT is roughly 34–41 times the global
22 ms floor, or 22–28 times the 33 ms bulk-stage estimate. These gaps indicate
substantial distance from ideal hardware utilization; they are not demonstrated
speedups. Reaching them requires addressing launch/dispatch, dequantization,
small expert GEMMs and other omitted costs. Exact arithmetic/output preservation
would still need independent validation.

[Machine-readable results](prefill-lower-bound.json) retain all route scenarios,
input hashes, assumptions and hardware source links. Reproduce without a GPU:

```sh
python scripts/calculate_moe_prefill_lower_bound.py \
  --out artifacts/qwen35_moe_prefill_g4_20261113/prefill-lower-bound.json
```
