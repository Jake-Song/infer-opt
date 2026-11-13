# Qwen3.5-35B-A3B INT8 prefill experiment

Compare the existing expert loop with stable route sorting on a dedicated
Colab G4 (RTX PRO 6000 Blackwell, about 95 GiB). An additional experimental
route-grouped Triton W8A16 GEMM is available for numerical investigation.
The initial `--high-mem` allocation was rejected; allocation without that flag
succeeded. Do not allocate another runtime automatically.

To reproduce on a fresh dedicated G4, prepare and upload the source bundle:

```sh
python scripts/moe_prefill_g4/prepare.py
colab upload -s YOUR_SESSION /tmp/moe-prefill-g4-bundle/source.tar.gz /tmp/moe-prefill-source.tar.gz
colab exec -s YOUR_SESSION -f /tmp/moe-prefill-g4-bundle/launch.py --timeout 120
colab exec -s YOUR_SESSION -f scripts/moe_prefill_g4/status.py --timeout 120
```

The launcher rejects duplicate workers in the same run directory. After the
worker finishes, run `colab exec -s YOUR_SESSION -f scripts/moe_prefill_g4/finalize.py --timeout 120`.
Download `/tmp/moe-prefill-g4-20261113/recovered-results.tar.gz`, verify its printed
hash and extract it into a repository artifact directory, then stop the exact session.
The recovered run also contains `archive-manifest.json` with SHA-256 hashes.
Large raw Chrome traces may stay inside the archive; retain their aggregate JSON
files alongside the other extracted results. Validate the recovered experiment:

```sh
python scripts/moe_prefill_g4/validate.py --results artifacts/YOUR_RUN/results \
  --archive artifacts/YOUR_RUN/results.tar.gz
```

`Int8Experts.prefill_backend` defaults to `reference`. Set it to `sorted` on
target expert modules to replace repeated `torch.where` queries with a stable
sort and one CPU count read per expert bank. Expert order, assigned row order,
GEMV/matmul dispatch and expert arithmetic are preserved. Exact equality is
required by the synthetic and model benchmarks. The one-to-five-row CUDA
decode/verification dispatcher and checkpoint format are preserved.

`grouped` is a separate experimental option for CUDA BF16/FP16 prefill. It
passed the small-shape GPU tests but failed real-shape numerical checks at
`atol=2e-4, rtol=.03`; it has not been adopted. See the recovered failed run and
independent dequantization diagnostics. Its reduction order must be evaluated
separately from quantization error.

The grouped path sorts token/slot assignments by expert, counts and schedules
tiles on the GPU, dequantizes weights inside tensor-core GEMMs, and combines
weighted outputs in ascending expert order. It retains BF16/FP16 rounding at
projection, activation and accumulation boundaries. Tensor-core reductions and
the small-row reference GEMV can still differ numerically. No BF16 copy of the
expert bank is cached. The intermediate memory grows with tokens × top-k.

Run independent dequantized-reference tests and synthetic real-shape timing
before downloading a model:

```sh
python -m pytest tests/test_moe_prefill.py -q
python scripts/bench_moe_prefill.py --out artifacts/my-prefill/experts \
  --lengths 128,512,2048 --warmup 2 --repeats 5 --profile
```

The synthetic experiment covers uniform, shared and skewed routing. Its
baseline uses the production eager expert loop, including the small per-expert
GEMV dispatch. Exact numerical checks precede sorted-path timing. Use
`--backends reference,grouped` to investigate the four grouped tile configurations;
failed configurations retain their error metrics and cause a nonzero exit.
These are warm, sequential microbenchmarks; they do not imply model throughput.

Use the published checkpoint pinned to
`331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`, under `/tmp/infer-opt-models`:

```sh
python scripts/bench_moe_prefill.py --weights /tmp/infer-opt-models/your-snapshot \
  --out artifacts/my-prefill/model --lengths 128,512,2048,4096 \
  --prompts math,korean --warmup 2 --repeats 5 --profile
```

Model TTFT includes text-model prefill, last-position vocabulary projection and
argmax. It excludes tokenization, cache reset, warmup, JIT compilation, decode,
and profiling. Static cache capacity is shared across all methods and lengths.
CUDA-event intervals include GPU idle time between host launches; they are not
the sum of kernel durations. Both host and CUDA intervals are recorded.

Inputs are exact-length prefixes of repeated frozen token fixtures, saved in
`inputs.json`; these are controlled workloads rather than natural long prompts.
Full-model results retain last-logit error/KL and sixteen-token greedy
continuations; sorted-path logits and generated IDs must match exactly.
This is a limited same-checkpoint numerical comparison, not HF
BF16 equivalence or task accuracy. Do not call a timing ratio output preserving
when generated tokens diverge. Memory statistics report allocated/reserved
peaks, which include the loaded model and shared cache.

`worker.py` installs the previous experiment's PyTorch 2.11.0+cu130, Triton 3.6.0
and FLA 0.6.0 environment; validates actual hardware and source hashes; runs
tests, synthetic experts, then the model comparison. Model/compiler/package
caches stay under `/tmp/infer-opt-models/moe-prefill-g4-20261113`. Results and
traces are snapshotted separately. The worker has a 90-minute process budget;
the CLI runtime must be stopped after verified result recovery.
