# G4 validation evidence

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GiB), **not A100**.
Pinned source revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`.
See [implementation status](../../MOE_STATUS.md) for conclusions and limitations.

| File | Scope |
|---|---|
| `environment.json` | Runtime, source/checkpoint identity and dependencies |
| `moe-kernel-validation.json` | Real expert dimensions, T=1–5, overlapping/disjoint routes, graph/compile checks |
| `moe-expert-tuning.json` | Expert launch microbenchmarks |
| `projection-tuning.json` | Ordinary INT8 projection launch sweep |
| `bf16-tuning.json` | BF16 retained gate/router/head microbenchmarks |
| `bf16-override-end-to-end.json` | Paired compiled INT8 decode, stock vs optional BF16 kernels; includes token parity |
| `quality.json` | Official HF BF16 vs custom eager BF16/INT8, teacher-forced logits |
| `checkpoint-validation.json` | Export reload, safe MTP parity at 64 tokens, decode kernel profile |
| `spec-parallel-repeated.json` | Experimental parallel MTP; failed greedy parity |
| `spec-sequential.json`, `spec-consistent.json` | Conservative CUDA Graph verification; matching but slower |
| `spec-compiled.json` | Compiled ordinary INT8 timing and diagnostic MTP; MTP failed parity |
| `contexts-cold-prefill.json` | Context/memory/parity checks; TTFT includes cold prefill compilation |
| `final-tests.log` | GPU test results at the recorded stage |
| `gpu-tests.txt`, `checkpoint-diagnosis.txt` | Historical dense exact-FP64 equality failures and investigation |
| `summary.json` | Aggregates; divergent MTP runs have no valid speedup |

Timings exclude tokenization and network overhead. Teacher-forced numerical
agreement does not establish task accuracy. Kernel microbenchmark improvements
are not full-model speedups. Different compiled arithmetic can change greedy
outputs near ties; inspect parity separately from timings.

The 34.541 GiB checkpoint was stored on the user-created Colab runtime at
`/content/qwen-moe-int8-g128-mtp-streamed`. That session was lost before HF
upload; model weights are not checked in and are currently inaccessible.
Final remote integration testing reported 34 passed, but its log, the manifest,
and the compiled HF comparison artifacts were not downloaded.
Original export peak process RAM was not recorded. Later converter telemetry
must not be attributed retroactively to that export.
