# Qwen3.5-35B-A3B INT8 prefill experiment

Compare the existing expert loop with stable route sorting on a dedicated
Colab G4 (RTX PRO 6000 Blackwell, about 95 GiB). An additional experimental
route-grouped Triton W8A16 GEMM is available for numerical investigation.
The initial `--high-mem` allocation was rejected; allocation without that flag
succeeded. Do not allocate another runtime automatically.

To run on a fresh dedicated G4, prepare and upload the source bundle. `--run`
names the remote root `/tmp/moe-prefill-g4-RUN`; the 2026-11-13 run's exact
sources are retained in its artifact, not reproduced by the current worker.
`--plan gate` (default) runs tests, the FP64 diagnosis, synthetic experts and
the three backends in the model. `--plan mtp` runs tests and
`validate_moe_prefill_mtp.py` twice: sequential and consistent verification
(the gate), then parallel verification (diagnostic). `--plan tune --routes ROUTES.json` runs tests,
`tune_moe_prefill_tiles.py` on recorded expert loads, `bench_w8a16_projection.py`,
then reference, grouped (the kernel's current default tile) and
`grouped-tuned` (the newly selected tile) in the model. The tuner's ratios stay
relative to 16×64×64, the tile before the 2026-11-14 tuning:

```sh
python scripts/moe_prefill_g4/prepare.py --run YOUR_RUN
colab upload -s YOUR_SESSION /tmp/moe-prefill-g4-bundle/source.tar.gz /tmp/moe-prefill-source.tar.gz
colab exec -s YOUR_SESSION -f /tmp/moe-prefill-g4-bundle/launch.py --timeout 120
colab exec -s YOUR_SESSION -f scripts/moe_prefill_g4/status.py --timeout 120
```

The launcher rejects duplicate workers in the same run directory. After the
worker finishes, run `colab exec -s YOUR_SESSION -f scripts/moe_prefill_g4/finalize.py --timeout 120`.
Download `/tmp/moe-prefill-g4-YOUR_RUN/recovered-results.tar.gz`, verify its printed
hash and extract it into a repository artifact directory, then stop the exact session.
The recovered run also contains `archive-manifest.json` with SHA-256 hashes.
Large raw Chrome traces stay inside the archive; retain their aggregate JSON
files alongside the other extracted results. Since 2026-11-14 the archive is
not committed (`artifacts/*/results.tar.gz` is ignored): record its size and
SHA-256 in `recovery.json` and keep the file elsewhere if the raw traces
matter. `validate.py` without `--archive` skips and counts the raw traces. `validate.py` covers both the
2026-11-13 layout (reference and sorted) and runs with the grouped backend,
numerics and breakdown outputs. Validate the recovered experiment:

```sh
python scripts/moe_prefill_g4/validate.py --results artifacts/YOUR_RUN/results \
  --archive artifacts/YOUR_RUN/results.tar.gz
```

`Int8Experts.prefill_backend` defaults to `grouped` since the 2026-11-14 MTP
run (`reference` before it). `sorted` replaces the reference loop's repeated
`torch.where` queries with a stable sort and one CPU count read per expert bank. Expert order, assigned row order,
GEMV/matmul dispatch and expert arithmetic are preserved. Exact equality is
required by the synthetic and model benchmarks. The one-to-five-row CUDA
decode/verification dispatcher and checkpoint format are preserved.

`grouped` is the route-grouped kernel for CUDA BF16/FP16 prefill, the default
backend; elsewhere `sorted` runs in its place. It changes reduction and rounding order, so since 2026-11-14
it is judged by distance to FP64 instead of `atol=2e-4, rtol=.03` against
another BF16 path: its RMS error to an FP64 evaluation of the same INT8 banks
must not exceed the adopted loop's (1% tie allowance), and in the model the
last-position logit KL must stay within `--kl-limit` (default 2e-3) with a
matching sixteen-token continuation. See the
[numerical diagnosis](../../artifacts/qwen35_moe_prefill_numerics_20261114/README.md).
A missed grouped gate is a recorded result: all cases are measured and saved,
then the script exits with status 3.

The grouped path sorts token/slot assignments by expert, counts and schedules
tiles on the GPU, multiplies exact INT8 values in tensor-core GEMMs and applies
group scales to FP32 partial sums (as the small-row GEMV does), and combines
weighted outputs in ascending expert order. It retains BF16/FP16 rounding at
projection, activation and accumulation boundaries, but never rounds
`q * scale`. No BF16 copy of the expert bank is cached. The intermediate memory
grows with tokens × top-k.

Run the tests, the stage-wise FP64 diagnosis and synthetic real-shape timing
before downloading a model:

```sh
python -m pytest tests/test_moe_prefill.py -q
python scripts/diagnose_moe_prefill_numerics.py --out artifacts/my-prefill/numerics.json
python scripts/bench_moe_prefill.py --out artifacts/my-prefill/experts \
  --lengths 128,512,2048 --warmup 2 --repeats 5 --profile
```

The synthetic experiment covers uniform, shared and skewed routing. Its
baseline uses the production eager expert loop, including the small per-expert
GEMV dispatch. Exact numerical checks precede sorted-path timing. Use
`--backends reference,sorted,grouped` to time the five grouped tile
configurations; a configuration that misses the FP64 gate keeps its error
metrics, gets no timing and causes exit status 3.
These are warm, sequential microbenchmarks; they do not imply model throughput.

Use the published checkpoint pinned to
`331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`, under `/tmp/infer-opt-models`:

```sh
python scripts/bench_moe_prefill.py --weights /tmp/infer-opt-models/your-snapshot \
  --out artifacts/my-prefill/model --lengths 128,512,2048,4096 \
  --prompts math,korean --warmup 2 --repeats 5 --profile --breakdown 3
```

`--breakdown N` adds two diagnostic passes outside the TTFT samples: per-layer
rows per expert (`routes.json`, summarized in each row) and median synchronized
section times over N passes (`sections_ms`: attention projections and cores,
router, routed experts, shared expert, norms, lm_head). Section times
synchronize the device at every boundary, so they attribute cost but do not sum
to the unprofiled TTFT.

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
tests, the FP64 diagnosis, synthetic experts, then the model comparison with
`--breakdown 3`. All three backends run; if the synthetic grouped gate fails,
the model stage still runs with `reference,sorted` so route loads and section
times are recorded, and `gates.json` keeps each stage's verdict. Model/compiler/package
caches stay under `/tmp/infer-opt-models/moe-prefill-g4-YOUR_RUN`. Results and
traces are snapshotted separately. The worker has a 90-minute process budget;
the CLI runtime must be stopped after verified result recovery.
