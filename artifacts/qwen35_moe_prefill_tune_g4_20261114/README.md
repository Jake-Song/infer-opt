# G4 MoE prefill: tile tuning and what remains, 2026-11-14

A 32×128×128 tile cuts the route-grouped expert kernel's time to 0.79× that of
the 16×64×64 tile it started with, and brings warm text-model TTFT to **97–98 ms
at 2048 tokens and 178 ms at 4096**, 7–10% below the same-session 16×64×64 run.
At 128 tokens TTFT is unchanged at 52 ms. Both tiles passed the model gate in
all eight cases. The tuned tile is now the kernel's default.

Two other candidates were measured and not adopted: splitting experts by row
count into two tile shapes, and a fused INT8 kernel for the dense projections.
A finer breakdown shows that small prompts are now limited by host dispatch,
not by the GPU.

Same environment and contract as the
[route-grouped run](../qwen35_moe_prefill_g4_20261114/README.md): RTX PRO 6000
Blackwell Server Edition, PyTorch 2.11.0+cu130, Triton 3.6.0, FLA 0.6.0, the
published INT8 checkpoint at `331d7db9`, batch one, MTP off, repeated
math/Korean fixtures, two warmups and five repetitions, `reference` first.

## Tile tuning on recorded expert loads

`scripts/tune_moe_prefill_tiles.py` replays the per-layer rows-per-expert counts
of the earlier run on one synthetic real-shape bank and sums each
configuration's time over a case's 40 layers. Only the load per expert is real;
which token goes where is synthetic. 36 configurations: 24 single tiles (row
tile 16/32/64, column tile 64/128, K block 64/128, 4 or 8 warps) and 12 with a
second tile for experts above 16, 32 or 64 rows.

| Configuration | Expert time vs 16×64×64, geometric mean of eight cases |
|---|---:|
| 32×128×128, 4 warps (selected) | 0.793 |
| 32×128×128, 8 warps | 0.794 |
| 32×64×128, 4 warps | 0.818 |
| 64×64×128, 4 warps | 0.828 |
| best two-tile split (16×64×128, above 16 rows 32×128×128) | 0.931 |
| 16×64×64, 4 warps (previous default) | 1.000 |
| 16×64×64, 8 warps | 1.084 |

Summed expert time for the selected tile went from 39.2 to 30.6 ms (math 2048)
and from 69.4 to 51.1 ms (math 4096). It passed the FP64 gate on the first and
middle layer of every case (16 checks, 0.562% at most against 0.602% at least
for the adopted loop). The two-tile split launches each projection twice and
lost to every good single tile.

## Tuned tile in the model

`grouped` is the 16×64×64 tile, `tuned` the 32×128×128 tile, in one session.
Median synchronized host milliseconds.

| Fixture | Tokens | Reference | Grouped 16×64×64 | Tuned 32×128×128 | Tuned vs grouped | KL(ref ‖ tuned) |
|---|---:|---:|---:|---:|---:|---:|
| math | 128 | 889.1 | 52.3 | 52.6 | 0.995× | 7.7e-4 |
| math | 512 | 1294.4 | 57.5 | 53.6 | 1.073× | 4.5e-5 |
| math | 2048 | 1422.2 | 106.0 | 98.3 | 1.078× | 4.1e-5 |
| math | 4096 | 1487.1 | 195.8 | 178.7 | 1.096× | 6.9e-7 |
| korean | 128 | 1007.9 | 52.3 | 52.5 | 0.996× | 1.7e-4 |
| korean | 512 | 1079.3 | 53.4 | 53.5 | 0.998× | 7.6e-5 |
| korean | 2048 | 1223.2 | 105.1 | 96.8 | 1.087× | 6.2e-4 |
| korean | 4096 | 1305.5 | 195.5 | 178.2 | 1.097× | 1.7e-6 |

All sixteen model gates passed: KL under the provisional 2e-3 ceiling and
every sixteen-token continuation equal to the reference's. The grouped column
reproduces the earlier session within 0.5 ms. The expert section shrank in
every case (for example 25.7 to 21.9 ms at math-128), but at 128 tokens that
did not move the unprofiled TTFT.

## Dense projections: fused kernel not adopted

`scripts/bench_w8a16_projection.py` times the six dense projection shapes that
prefill dequantizes on every call. Totals over all layers of one prefill:

| Tokens | Dequantize + `F.linear` (current) | `F.linear` on a cached BF16 weight | Fused INT8 kernel, best tile per shape |
|---:|---:|---:|---:|
| 128 | 9.2 ms | 3.8 ms | 9.6 ms |
| 512 | 13.4 ms | 7.6 ms | 13.2 ms |
| 2048 | 24.3 ms | 19.3 ms | 27.2 ms |
| 4096 | 39.6 ms | 33.5 ms | 45.7 ms |

The fused kernel (`infer_opt.moe_prefill.w8a16_gemm`) wins only on the two
widest projections (GatedDeltaNet q/k/v/z up to 2048 tokens, attention q/k/v up
to 512), which makes the totals a tie up to 512 tokens and a loss from 2048.
It is more accurate (0.166% RMS error to FP64 against 0.24–0.33%), but that is
not a reason to adopt it here. Dequantization alone is 6.5 ms per prefill at
any length; keeping BF16 copies of the dense weights would remove it for about
2.8 GB more memory (1.405 billion elements). That trade is not taken in this run.

## Where the time goes with the tuned tile

Median synchronized section times, math fixture, three passes; host time in
parentheses for leaf sections (time until the call returned, before the device
drained). Sections synchronize at every boundary, so they add up to more than
the unprofiled TTFT.

| Section | 128 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| Routed experts | 21.9 (9.1) | 34.6 (11.0) | 55.0 (11.2) |
| GatedDeltaNet projections | 8.5 (3.4) | 19.6 (3.6) | 29.7 (3.5) |
| GatedDeltaNet delta rule (FLA) | 12.9 (12.6) | 13.8 (13.0) | 19.6 (13.2) |
| GatedDeltaNet activation, splits, gate terms | 7.2 | 17.6 | 28.1 |
| GatedDeltaNet gated norm | 3.1 (2.9) | 5.4 (2.9) | 14.0 (2.9) |
| GatedDeltaNet convolution | 1.0 (0.5) | 5.0 (0.6) | 9.6 (0.6) |
| Layer norms | 6.3 (5.8) | 6.8 (5.8) | 9.2 (5.8) |
| Full attention, projections + core | 5.9 | 10.6 | 19.2 |
| Routing softmax/top-k and sums | 5.1 | 5.3 | 6.4 |
| Shared expert | 4.4 (4.1) | 4.9 (4.0) | 6.7 (4.0) |
| Text model total | 81.3 | 129.4 | 204.7 |
| Unprofiled TTFT | 52.6 | 98.3 | 178.7 |

- **Small prompts are host-bound.** At 128 tokens the leaf sections' host time
  adds up to about 41 ms of the 81 ms. The FLA chunk delta rule spends 12.6 of
  its 12.9 ms before returning, a fixed 0.42 ms per layer at every length; layer
  norms and the shared expert are the same kind.
- **Large prompts are spread out.** At 4096 tokens routed experts are 27% of
  the synchronized total and GatedDeltaNet's parts together 50%, none of
  them dominant.
- The routed experts' own host floor is 9–11 ms per prefill (route sort,
  schedule and four kernel launches per layer).

## Not measured

MTP on, natural long prompts, task accuracy, HF BF16 equivalence, cache
tensors, batch sizes above one and other GPUs, as before. Tile tuning used
loads from the repeated fixtures; a different routing distribution could rank
tiles differently. No GatedDeltaNet or dispatch optimization was attempted.

## Evidence

[validation.json](validation.json) rechecks 138 file hashes, the ranking of
all 36 configurations, the projection totals, 120 model timing samples, 384
continuation IDs and the sixteen model gates:

```sh
python scripts/moe_prefill_g4/validate.py \
  --results artifacts/qwen35_moe_prefill_tune_g4_20261114/results \
  --archive artifacts/qwen35_moe_prefill_tune_g4_20261114/results.tar.gz
```

[results/tune/tile-tuning.json](results/tune/tile-tuning.json),
[results/projections/projections.json](results/projections/projections.json)
and [results/model/model.json](results/model/model.json) hold the raw rows.
The recovered archive `results.tar.gz` (49.3 MB, six raw Chrome traces, 0.8 GB
unpacked) is not kept in the repository; [recovery.json](recovery.json) records
its size and SHA-256. Without it, omit `--archive`: everything except the raw
traces and the remote run's four pytest cache files is still verified. The G4 was allocated
without `--high-mem` with the user's approval, ran 650 s of worker time, and
was stopped after verified recovery; see [lifecycle.json](lifecycle.json).
Reproduce with `prepare.py --run YOUR_RUN --plan tune --routes
artifacts/qwen35_moe_prefill_g4_20261114/results/model/routes.json`.
