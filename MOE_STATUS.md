# Qwen3.5-35B-A3B INT8 + MTP

Implemented packed MoE loading, separate MTP expert loading, group-128 W8A16
conversion/export/reload, selected-expert Triton kernels, and greedy speculative
verification. Routers, small gates and `lm_head` retain BF16 weights. This is the
custom infer-opt checkpoint format, not a Transformers quantization format.

## Hardware and checkpoint

Originally validated on the user-created Colab `qwen35-moe-int8-mtp` G4 session:
RTX PRO 6000 Blackwell Server Edition, 94.97 GiB, PyTorch 2.11.0+cu130,
Triton 3.6.0. **These are not A100 measurements.** The separate `cudagemv`
session was not used.

On 2026-11-04, the published checkpoint also passed strict reload and all ten
expert-kernel checks on **A100-SXM4-80GB (79.250732 GiB visible)**, with
PyTorch 2.11.0+cu130 and Triton 3.6.0. The four 64-token sequential/consistent
MTP k=1,2 smoke checks matched ordinary decoding. Evidence is in
[artifacts/qwen35_moe_a100_20261104](artifacts/qwen35_moe_a100_20261104/README.md);
its measurements are separate from G4.

Source: `Qwen/Qwen3.5-35B-A3B`, revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`.
CPU streaming conversion produced 22 shards, 34.541 GiB, including MTP.
Reload validated 41 quantized expert modules. The checkpoint was stored at
`/content/qwen-moe-int8-g128-mtp-streamed`. The Colab session was subsequently
lost (CLI returned 404/401 and no active sessions) before that export was uploaded.

On 2026-11-04 the checkpoint was regenerated and published as
[Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp](https://huggingface.co/Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp),
commit `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`. It contains 22 weight shards
(34.540574 GiB) plus tokenizer/configuration files and release metadata.
The release passed strict reload of 41 expert modules, ordinary 64-token
generation, and MTP k=1,2 token parity with sequential and consistent verification
on G4. All 35 payload files and the manifest were verified at the published
commit. See [release evidence](artifacts/qwen35_moe_hf_recovery_20261104/README.md).
Publishing used local HF authentication; the dedicated Colab runtime was stopped.

The converter processes expert slices and output shards without loading the full
BF16 model on the GPU. This makes conversion independent of A100 VRAM capacity;
host RAM and disk still matter. The actual export predates process-RSS telemetry,
so its peak host RAM was not measured. G4 inference reached 37.235 GiB allocated
at 4096-token context; that is **not proof of fitting or running on A100 40 GB**
(driver allocations, reserved memory and kernels differ).

The 2026-11-04 release regeneration separately measured 26.855965 GiB peak
process RSS during CPU conversion, with CUDA uninitialized in the converter.

## Measured behavior

The original G4 evidence is in [artifacts/qwen35_moe_g4](artifacts/qwen35_moe_g4).
All timing claims below concern batch-one decode, not prefill or task accuracy.

* Ordinary INT8 decode with manual CUDA Graph: roughly 8.6–8.9 ms/token.
* Ordinary INT8 with `compile-reduce-overhead`: roughly 4.91–4.94 ms/token,
  five prompts, five repeats, 256 generated tokens. Compiled and manual graph
  arithmetic can differ; these are separate execution configurations.
* Manual graph MTP with sequential or consistent verification matched the
  baseline on all five 256-token prompts for k=1,2, but was slower
  (roughly 9.2–13.7 ms/token). No MTP speedup is established.
* Parallel verification diverged in 30/50 repeated cases. Compiled consistent
  MTP k=1 diverged in all 25 cases. These timings are diagnostic only.
  Compiled MoE MTP requires explicit `allow_inexact=True` (CLI
  `--allow-inexact-verification`); default MoE verification is sequential.
* Official HF BF16 teacher-forced numerical comparison, nine prompts × 48 tokens:
  custom BF16 mean KL 0.001743, INT8 0.002014; each had three near-tie top-1
  mismatches and no mismatches with reference logit margin >0.5.
  This original check used eager decode and does not establish task accuracy.

`contexts-cold-prefill.json` contains a cold prefill compile in its first long
context measurement. Do not compare its TTFT as warmed performance. The benchmark
now warms each prompt shape before timing.

### A100 80GB, 2026-11-04

The published checkpoint ran on A100-SXM4-80GB with batch 1, five prompts,
three repeats and 256 output tokens. The following ranges are per-prompt
medians; prefill and compilation are excluded from decode timing.

| CUDA Graph setting | Ordinary ms/token | MTP ms/token | Token divergence |
|---|---:|---:|---:|
| Sequential k=1 | 10.68–10.74 | 12.03–13.51 | 0/15 runs |
| Sequential k=2 | 10.75–10.84 | 12.26–15.96 | 0/15 runs |

Neither setting established a speedup. On prose, k=1 produced 1.984 tokens per
23.802 ms round; k=2 produced 2.931 tokens per approximately 35.9 ms round.
High acceptance alone did not offset the sequential verifier's round cost.
This observation does not establish how an optimized batched verifier would perform.

The runtime disappeared during consistent k=1; eight completed partial rows
retained token parity, but the full repeat set was not collected. The backend
reported no active sessions. Parallel and compiled MoE comparisons were not
reached, so this run does not settle their speed or parity on A100. The cause of
runtime loss is unknown. See the linked artifact README and `interruption.json`.

## Kernel tuning scope

| Component | Work / measured result | Adoption |
|---|---|---|
| Routed INT8 experts | New selected-expert kernels; T=1–5, overlap/disjoint and dynamic routes tested; launch sweep ~4% best micro gain | Default stable launch retained |
| Ordinary INT8 projections | 60 launch configurations on actual layer shapes; improvements <1% | Existing defaults retained |
| BF16 small gates / router | Custom GEMV; small-gate micro gains 27–50%, router ~16% | Explicit optional override; full-model evidence recorded separately |
| BF16 `lm_head` | Compared custom GEMV with cuBLAS | cuBLAS retained (faster) |
| GatedDelta recurrence / attention | Profiled; about 1.52% / 0.57% of kernel duration | Existing kernels reused |
| Other pointwise operations | Compiled decode path | No separate hand-tuned kernel claim |

Microbenchmarks use cold-cache measurements. Their gains are not end-to-end
speedups. Optional BF16 overrides preserve checkpoint parameters and dtype but
change reduction order. The paired full-model benchmark measured 4.75–4.78
ms/token versus 4.91–4.99 ms/token, but four of five prompts changed generated
tokens. The override remains experimental and is not enabled by default.

## Reproduce

Start with an already downloaded, pinned source snapshot:

```sh
python scripts/quantize_qwen3_5_moe_streaming.py \
  --source /path/to/pinned-snapshot --out /path/to/empty-int8-output
python -m pytest tests/test_moe.py tests/test_bf16_gemv.py -q
python scripts/bench_qwen3_5_spec_decode.py \
  --weights /path/to/empty-int8-output --configs int8+mtp --ks 1,2 \
  --mode cudagraph --verification sequential --repeats 5 --new-tokens 256 \
  --out moe-safe-mtp.json
```

The streaming converter uses CPU. The legacy `quantize_qwen3_5.py` path loads
full BF16 weights first and is not the low-memory conversion path.

## Validation limits

Real-shape expert tests passed CUDA Graph and compiled/eager equality; maximum
absolute error versus the independent INT8 reference was 3.05e-5. The previous
MoE GPU test run passed 27 tests; the additional BF16 suite passed five.
Historical dense FP64 checkpoint regression tests had two exact-equality
failures at 8.88e-16 logit difference with identical token selections and state;
see `gpu-tests.txt` and `checkpoint-diagnosis.txt`. This is not an entirely green
repository-wide GPU test claim. The later A100 80GB run validates the recorded
INT8 kernel, reload, generation and decoding cases; task-level quality evaluation
remains unvalidated.

The original remote integration testing reported 34 passed before the session was lost.
Its log and the completed compiled-plus-BF16-override HF comparison were not
downloaded. The separate stock compiled INT8 HF comparison had not been
confirmed complete. Only the files present in the artifact directory are retained
evidence; compiled quality was unfinished at that stage. HF publication was completed
separately on 2026-11-04 as described above.

The [2026-11-05 paired HF comparison](artifacts/qwen35_compiled_hf_quality_20261105/README.md)
subsequently completed the stock fused INT8 eager and full compiled decode replay
on G4, with MTP off and cache capacity 8456. Both modes shared the exact HF
reference logits over nine prompts and 432 positions. Eager/compiled mean KL was
0.001894/0.002059, top-1 agreement 428/432 and 430/432, and strong mismatches zero
for both. The strict compiled/eager logit comparison still failed at all 423
actual decode positions (maximum absolute difference 8.6875); the nine shared
eager-prefill positions passed. This completes distribution measurement, not
strict eager equivalence, a root-cause diagnosis, or task accuracy evaluation.

The separate [llama.cpp Q8_0 HF comparison](artifacts/qwen35_llamacpp_hf_quality_20261105/README.md)
had mean KL 0.002135, 426/432 top-1 matches and no strong mismatches. Its HF
environment used DeltaNet PyTorch fallback, whereas the paired custom experiment
installed FLA 0.6.0. The HF continuations differed on two code prompts, so the
aggregate values do not provide a paired ranking across the two engines.

### Compiled parallel MTP on A100 80GB, 2026-11-09

The [completed experiment](artifacts/qwen35_moe_compiled_20261109/README.md)
uses the published INT8 checkpoint with the dense G4 compiler settings:
`compile-reduce-overhead`, fullgraph, static shapes, max-autotune and coordinate
descent enabled. G4 allocation failed twice with 503, then the user-authorized
A100 fallback ran on an actual A100-SXM4-80GB. PyTorch 2.11.0+cu130 and Triton
3.6.0 were pinned. Profiling confirmed CUDA Graph launches for plain and all
three parallel-MTP settings.

Batch 1, five prompts, 256 output tokens, three warmups and five measured repeats:
ordinary compiled decode was 5.45–5.56 ms/token. Parallel MTP k=3 preserved every
generated token on math and code in all five repeats: 4.148/4.526 ms/token,
1.340×/1.224× over their same-session baselines. Math also preserved parity for
k=1 and k=2 (0.996×/1.228×). All chat/prose/Korean settings and code k=1,2
diverged, so their timing ratios are diagnostic.

The MTP-off paired HF BF16 replay covered nine prompts and 432 positions.
Eager/compiled mean KL was 0.001854/0.002055, both matched top-1 at 430/432,
and both had zero strong mismatches. Strict compiled/eager allclose still failed
at all 423 actual decode positions (max absolute difference 7.0); nine shared
eager-prefill positions passed. This is a numerical replay, not task accuracy.
Raw IDs for all 100 timed generations, per-position quality metrics, source
hashes, graph evidence and verified aggregation were recovered before stopping
the dedicated runtime. Model caches stayed under `/tmp/infer-opt-models`.

### A100 round profiling follow-up, 2026-11-09

The [round profiling report](artifacts/qwen35_moe_round_profile_a100_20261109/README.md)
records six original-path traces (off/k=1/k=3 × math/Korean), independent CUDA-event
phase diagnostics, and cold-cache expert sweeps. In the math trace, experts
accounted for 61% of the k=1→3 GPU kernel-time increase; BF16 matmul-containing
fused kernels accounted for 27%. Four-token expert sweeps cost about 3.1×
single-token sweeps, even with forced shared routing. Segmented diagnostics put
the three draft vocabulary projections near 1.8 ms and accept/commit near 0.2 ms.
Profiler times and separately compiled phase times are not original-path wall times.

Manual fullgraph controls reduced frozen-round host time, but some state tensors
differed from the original compiled path. These are optimization candidates, not
validated equivalent speedups. This run's math k=3 generations also diverged in
all five repeats, unlike the earlier experiment. Neither earlier parity nor
one-round output-ID equality establishes general compiled MoE equivalence.

### A100 verification kernel optimization, 2026-11-09

The [optimization report](artifacts/qwen35_moe_verify_opt_a100_20261109/README.md)
records a pair-load expert kernel that preserves reference outputs and uses one warp.
It is enabled for A100 BF16 T=2/4 with the measured 2048/512/top8/group128 shape.
The final automatic-path math checks reduced target-body GPU time by 1.38%/2.67%
for T=2/4, with exact hidden/logits and all 141 cache tensors. GPU tests: 37 passed.
This is fixed-state target-body timing, not end-to-end MTP throughput.
The user requested wrap-up after those two final points; Korean automatic-path
reconfirmation was stopped. Earlier explicit pair-kernel comparisons completed
both prompts with exact state. On 2026-11-11, the user prioritized measuring
llama.cpp expert costs per MTP round before choosing a follow-up implementation.
Route-packed persistent Grouped GEMM remains a subsequent candidate; see the
[next experiment](artifacts/qwen35_moe_verify_opt_a100_20261109/NEXT_EXPERIMENT.md).
All results were recovered and the A100 40GB runtime was stopped.

### Model-free llama.cpp expert comparison, 2026-11-11

The [standalone experiment](artifacts/llamacpp_moe_kernel_a100_20261111/README.md)
ran Qwen3.5-35B-A3B expert shapes on a RunPod A100-SXM4-80GB, without model
downloads. Native Q8_0/Q8_1 W8A8 and the current group128 W8A16 kernels were
measured on the same GPU, with PyTorch 2.9.1+cu130 and Triton 3.5.1. T=1/2/4,
top-k=8, three route-sharing patterns, hot and 128 MiB flush-before-run conditions
were covered. These toolchain versions differ from the earlier model runs.

For partially shared routes after flushing, native llama.cpp down cost 40.90 us
at T=4 versus 55.94 us for current infer-opt. Gate/up was slower, and the full
routed expert block cost 118.64 versus 114.40 us. Native T=1/2 block time was
also slower; fully shared hot T=4 was a favorable exception. No general block
speedup or MTP throughput improvement is established. Clocks were not locked.

Both combined-weight view and native contiguous-weight variants passed 36/36
CPU-reference CUDA cases. Actual captured graphs confirmed native fused gate/up
(two kernels including activation quantization) and the unfused view control
(five kernels). Hardware counters were blocked by ERR_NVGPUCTRPERM, so the
instruction-issue-bound hypothesis remains unresolved. All results and source
hashes were recovered, and the dedicated Pod was deleted. The next step remains
measuring costs in an actual native llama.cpp MTP round.

The separate [HF Jobs A100 permission probe](artifacts/hf_issue_bound_a100_20261111/README.md)
also returned ERR_NVGPUCTRPERM after the user-requested HF CLI update to 2.4.0.
This time diagnostics recorded uid 0, host `RmProfilingAdminOnly=1`, and absent
SYS_ADMIN/PERFMON capabilities in the effective, permitted and bounding sets.
CUDA compilation/execution succeeded, but counter collection did not. The probe
completed and results were recovered; the larger MoE profiling job was not
submitted. The instruction-issue hypothesis remains unresolved.

### A100 down counter diagnosis, 2026-11-11

The [Colab A100 40GB follow-up](artifacts/down_issue_a100_20261111/README.md)
successfully collected eight NCU reports and 12,000 unprofiled synthetic down
timing samples. Current packed one-warp T=4 partial-route issue activity was
62.68% hot / 56.05% after flushing, with achieved occupancy 25.36% / 27.23%.
A one-warp scalar diagnostic reduced dynamic warp instructions by 28.16% but
barely changed latency; it was not bit-exact and was not adopted. These results
do not support a pure issue-throughput saturation diagnosis for the current
packed down kernel. The old four-warp reference had higher hot issue activity
(87.66%). Memory/execution dependency stalls and low achieved occupancy remain
observations, not a uniquely established root cause. Profiling ran near 1.09 GHz
while unprofiled telemetry was generally 1.41 GHz; durations are kept separate.
No production kernel changed. The verified archive, reports and lifecycle record
were recovered, and the dedicated Colab runtime was terminated.

### Matched gate/up and down counter follow-up, 2026-11-11

The [fixed-clock expert follow-up](artifacts/expert_issue_a100_20261111/README.md)
measured fused gate/up + SwiGLU, standalone down and the full routed expert block
on A100 40GB at a locked 1095 MHz. It recovered 25,200 timing samples and nine
NCU reports. Current one-warp partial-hot gate/up issue activity was 75.53%
(71.10% over elapsed cycles), versus down 62.87% (49.74%). Partial-hot gate/up
still moved 45 MB through DRAM because its selected weight set exceeds L2.
Fully shared hot reduced DRAM traffic to 1.39 MB, but issue activity was 81.22%
(76.54% elapsed), not a pure saturation result. A non-exact one-warp old-kernel
diagnostic reduced gate/up instructions 37.10% and NCU latency 24.24%; it was not
adopted. Four-warp results are historical/launch controls, not the current T=2/4
baseline. All recorded block configurations matched their production dispatcher,
but no model quality or MTP throughput was measured. Kernel sources are unchanged;
clocks were restored, results verified, and the dedicated runtime terminated.

### Native llama.cpp MTP measurement, 2026-11-11/12

The [native MTP report](artifacts/llamacpp_mtp_a100_20261111/README.md) records
unmodified llama.cpp at `a0ad695c`, an MTP-inclusive Unsloth Q8_0 GGUF and
A100-SXM4-80GB at a locked 1095 MHz. Off/k=1/2/3 measured
9.13/6.79/5.96/5.64 ms/token on math and 9.19/7.48/7.40/8.40 on Korean.
Every MTP configuration diverged from the same model's MTP-off output in all
five repetitions. Timings are diagnostic; no output-preserving speedup or
paired ranking against historical infer-opt/G4 results is established.
Forty timed requests and 10,240 generated IDs were independently checked.
Eight separate full-request CUDA node traces retain prefill and are kept apart
from unprofiled decode timings; target/draft phase costs are not separately labeled.
CUDA kernels and warp settings were not changed.

### G4 compiled parallel MTP measurement, 2026-11-12

The [completed G4 experiment](artifacts/qwen35_moe_compiled_g4_20261112/README.md)
used the user-provided RTX PRO 6000 Blackwell session (94.97 GiB) after the earlier
503 allocation failures. It compared compiled MTP-off with parallel k=1/2/3 on
the five frozen prompts, cache 2048, 256 output tokens, three warmups and five
repeats. PyTorch 2.11.0+cu130, Triton 3.6.0 and FLA 0.6.0 were pinned; compiler
autotune settings and default kernels/warps were retained. All four modes showed
an actual `cudaGraphLaunch` in separate profiler evidence.

MTP-off measured 4.54–4.65 ms/token. Code preserved all 256 tokens in every repeat
at k=1/2/3: 3.9930/3.4206/3.2078 ms/token, or 1.16×/1.35×/1.44× against its
4.6190 ms/token compiled baseline. Math preserved output at k=1/2, reaching
3.3302 ms/token (1.37×) at k=2. Math k=3 diverged at generated token 18; chat,
prose and Korean diverged at every k. Those timings remain diagnostic. Korean
k=3 was also slower than its MTP-off baseline (4.8707 versus 4.5445 ms/token).

The independent validator checked 100 timing samples and 25,600 generated IDs,
input IDs, acceptance histograms, timing denominators and CUDA Graph evidence.
The archive and 76 internal file hashes were verified. Parity is against the
same compiled MTP-off decoder; HF BF16/eager logit equivalence and task accuracy
were not rerun. The original G4 consistent-verification run is separate evidence.
Results were recovered before stopping the exact session; the server then
reported no active sessions. Earlier failed allocations remain in the artifact history.
