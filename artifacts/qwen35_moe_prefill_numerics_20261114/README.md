# MoE prefill numerics: FP64-referenced acceptance, 2026-11-14

The route-grouped prefill candidate had no routing or tiling bug. Its distance
from the adopted loop was BF16 rounding, and the adopted dequantize + matmul
path is itself the less accurate one because it rounds `q * scale` to BF16.
After applying group scales to FP32 partial sums, the candidate passes the new
FP64-referenced criterion in all 45 local comparisons. This is a one-layer
synthetic check on a local RTX 2080 SUPER. The
[G4 run of the same day](../qwen35_moe_prefill_g4_20261114/README.md) confirmed
it on the target GPU and passed the full-model gate.

## Acceptance criterion (changed at the user's decision)

`atol=2e-4, rtol=.03` against another BF16 path is retired for backends that
change reduction or rounding order. It cannot be met by two correct paths: each
BF16 path is about 0.55–0.64% RMS from the unrounded value through its own
roundings, and the two already adopted paths differ from each other by 0.63%.

- **Expert level**: RMS error to an FP64 evaluation of the same INT8 banks
  (weights exactly `q * scale`, no intermediate rounding) must not exceed the
  adopted loop's, with a 1% allowance for rounding ties
  (`infer_opt.moe_prefill.float64_gate`).
- **Model level**: last-position logit KL(reference ‖ candidate) at most 2e-3,
  and the sixteen-token greedy continuation must match
  (`scripts/bench_moe_prefill.py --kl-limit`). The ceiling is the scale of this
  checkpoint's recorded INT8-vs-HF-BF16 mean KL (0.0019–0.0021); it is a
  provisional choice.
- `sorted` keeps the expert arithmetic and still must be bit-exact.

## Environment

RTX 2080 SUPER (SM 7.5, 8 GiB), PyTorch 2.14.0+cu130, Triton 3.8.0. One expert
bank at the real shape: 256 experts, hidden 2048, intermediate 512, group 128,
top-8, BF16. Banks, scales and activations are synthetic (uniform INT8, random
scales, standard-normal activations), as in the 2026-11-13 synthetic benchmark.
SM 7.5 has no BF16 tensor-core instruction and the G4 run used PyTorch 2.11 /
Triton 3.6, so reduction details differ from the G4. No timing is recorded.

## End-to-end RMS error to FP64

`production` is the adopted loop (GEMV up to eight rows, otherwise dequantize +
`F.linear`); `dequant` always dequantizes. Tile 16×64×64; the other four tiles
agree to the digits shown.

| Tokens | Routing | Experts ≤8 rows / used | production | dequant | grouped, rounded weights | grouped, FP32 scales |
|---:|---|---:|---:|---:|---:|---:|
| 128 | spread | 250 / 254 | 0.568% | 0.640% | 0.640% (fail) | 0.566% |
| 128 | shared | 0 / 8 | 0.624% | 0.624% | 0.624% | 0.547% |
| 128 | skewed | 159 / 167 | 0.605% | 0.622% | 0.622% (fail) | 0.547% |
| 512 | spread | 5 / 256 | 0.631% | 0.632% | 0.632% | 0.557% |
| 512 | shared | 0 / 8 | 0.631% | 0.631% | 0.631% | 0.559% |
| 512 | skewed | 243 / 254 | 0.614% | 0.631% | 0.631% (fail) | 0.558% |
| 2048 | spread | 0 / 256 | 0.629% | 0.629% | 0.629% | 0.555% |
| 2048 | shared | 0 / 8 | 0.630% | 0.630% | 0.630% | 0.556% |
| 2048 | skewed | 3 / 256 | 0.628% | 0.628% | 0.628% | 0.555% |

The previous kernel failed 15 of 45 gate checks (three cases × five tiles),
exactly where the adopted loop sends most experts to the more accurate GEMV.
The current kernel passes 45 of 45.

## Findings by stage

Each stage receives identical BF16 inputs (2048 tokens, spread routing shown;
the other cases agree).

| Stage | Implementation | RMS error to FP64 | Within half a BF16 unit |
|---|---|---:|---:|
| gate/up projection | dequantize + `F.linear` | 0.237% | 57.0% |
| gate/up projection | grouped, FP32 scales | 0.166% | 99.99% |
| SwiGLU | torch and grouped (bit-identical) | 0.242% | 72.7% |
| down projection | dequantize + `F.linear` | 0.237% | 57.4% |
| down projection | grouped, FP32 scales | 0.166% | 99.99% |
| weighted sum | loop and grouped (bit-identical) | 0.394% | — |

- **No routing or tiling bug.** The previous kernel equalled the dequantized
  loop in at least 99.98% of elements at both projections, and its end-to-end
  RMS difference to it was at most 0.007%. Error is the same in the first,
  later, partial and full row tiles.
- **Weight rounding is the avoidable part.** Rounding `q * scale` to BF16 costs
  as much as the unavoidable output rounding (0.237% against 0.166% per
  projection). The small-row GEMV already avoids it; the grouped kernel now
  uses the same arithmetic.
- **Pairwise distance is structural.** The FP32-scale kernel is 0.64% from the
  dequantized loop while being closer to FP64 than it. The SwiGLU and
  accumulation kernels were already bit-identical to the eager operations.

## Left open here, closed on the G4

- The 2026-11-13 G4 run showed the previous kernel 0.46–0.60% from the
  dequantized loop at 16–64 rows per expert, but 0.02–0.05% at 4 and at 512+
  rows. Locally the two are bit-equal at every row count. The G4 run traced it
  to the adopted path: `F.linear` on the dequantized gate/up matrix is not
  correctly rounded at those row counts on that GPU.
- Grouped timing, real routing loads and section times need the 35 GiB
  checkpoint and were measured in the G4 run.
- Local grouped timings from a smoke test are not retained: without BF16 tensor
  cores they say nothing about the G4.

## Reproduce

```sh
python scripts/diagnose_moe_prefill_numerics.py \
  --out artifacts/qwen35_moe_prefill_numerics_20261114/local-rtx2080s.json
python -m pytest tests/test_moe_prefill.py -q
```

[local-rtx2080s.json](local-rtx2080s.json) is the current kernel.
[local-rtx2080s-rounded-weight-kernel.json](local-rtx2080s-rounded-weight-kernel.json)
is the same script with the previous `q * scale`-rounding kernel restored; its
exit status 3 is the recorded gate failure.
