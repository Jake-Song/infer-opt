# Decoder-Only Transformer Inference Optimization Template

A compact, correctness-first PyTorch baseline for inference experiments. It uses a modern decoder shape—RMSNorm, RoPE, SwiGLU, grouped-query attention, and a preallocated KV cache—while keeping the optimization boundary explicit.

## Setup

The project targets Python 3.13. Install its dependencies with your preferred environment manager, for example:

```bash
uv sync --group dev
```

## Run an experiment

Check that cached prefill/decode matches a full causal forward pass:

```bash
uv run infer-opt check
```

Measure the paths separately (the final line is a JSON result record):

```bash
uv run infer-opt benchmark --device auto --prompt-len 128 --decode-len 64 --iterations 50
```

Use `--json` for JSON-only output suitable for an experiment runner. CPU is always supported. CUDA timings synchronize before and after each timed region, but results must be measured on the target GPU and with the intended dtype/model shape.

### A100 matrix multiplication profiling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/a100_matmul_profiling.ipynb)

Upload [`notebooks/a100_matmul_profiling.ipynb`](notebooks/a100_matmul_profiling.ipynb)
to Colab and select an A100 GPU. The standalone notebook compares strict FP32,
TF32-enabled FP32, FP16, and BF16 GEMMs across small and large M, checks numerical
error, and exports CUDA-event timings, a PyTorch CPU/CUDA trace, and runtime
metadata in a ZIP. Profiler collection runs separately from timing; hardware
counter metrics require a separate Nsight Compute setup.

### Reproducible 0.5B baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/03_a100_baseline_colab.ipynb)

To run the A100 experiment yourself, upload
[`notebooks/03_a100_baseline_colab.ipynb`](notebooks/03_a100_baseline_colab.ipynb)
to Colab, select an **A100 GPU**, and run all cells. This standalone notebook
includes the current source snapshot, verifies hardware and correctness, measures
all four prompt lengths, displays the results, and downloads a ZIP with raw data,
logs, charts, and source. It needs no Colab CLI, GitHub clone, or Drive mount.
After downloading, disconnect and delete the runtime in Colab.

Regenerate the embedded source snapshot after changing the experiment code:

```bash
uv run python scripts/build_a100_baseline_notebook.py
```

Run the correctness-gated prefill/decode sweep on an existing GPU:

```bash
uv run infer-opt baseline --expected-gpu A100 --output experiments/baseline/my-run
```

Or allocate one Colab A100, run tests and measurements, download the results, and stop
the session with the authenticated `colab` CLI:

```bash
uv run python scripts/run_baseline_colab.py --output experiments/baseline/my-a100-run
```

To use an A100 you already connected from the terminal, pass its session name:

```bash
uv run python scripts/run_baseline_colab.py --session my-a100 --output experiments/baseline/my-existing-a100-run
```

`--session` skips allocation and uses a fresh result directory on that runtime.
Existing sessions stay connected unless `--stop-session` is added. New sessions
created by the runner are always stopped. If any session is already active, the
runner requires `--session` rather than requesting another runtime. The actual GPU
is checked inside Colab in both modes; `colab.log` and `lifecycle.json` record the
commands, CLI executable, failure stage, and final session state.

The default is **494,962,688 random parameters**, BF16, B=1, `eager_sdpa`, contiguous
KV cache, prompts **128/512/2048/8192**, and **128 fixed-token decode steps**.
The model uses vocabulary 32768, hidden size 2048, 8 layers, 16 query heads,
4 KV heads, intermediate size 5632, and maximum sequence length 16384.
Each condition has 3 warmups and 10 measured repetitions. Override the workload with
`--prompt-lengths`, `--decode-len`, `--warmup`, `--iterations`, and `--seed` on the
`baseline` command. Its default device is CUDA; it never silently falls back to CPU.
The Colab runner has a 120-minute budget including cleanup and keeps the runtime's
installed CUDA PyTorch. It executes the bundled source without installing the project.

Prefill includes KV writes and vocabulary logits for **every prompt position**.
Decode measures only the sequential one-token forward calls, with a growing cache;
cache allocation/reset and the prerequisite prefill finish before timing starts.
Model setup, input transfer, RoPE initialization, warmup, validation and sampling
are outside both timing regions. Wall time is the primary measurement; it includes
Python dispatch, CUDA event recording and completion synchronization. CUDA event
elapsed time is also saved and may include host launch gaps.

`results.json` contains every repetition, environment/source identity and correctness
checks; `summary.csv` and the Korean `report.md` show medians and throughput. Decode
ms/token divides each full decode sequence by its step count; p90 is over repetition
averages, **not individual token latencies**. Full logits and every layer's valid K/V
must match the full causal reference (`atol=rtol=0.02` for BF16); cache prefix and
position checks must also pass. Failed conditions have diagnostic records and no
performance results. Existing output files are never overwritten.

Benchmark JSON schema **v2** also applies to the existing `benchmark` command.
Its original scalar fields retain arithmetic-mean semantics; `wall_ms`, `cuda_ms`
and `samples` add distributions and raw observations. Older decode measurements
included prefill/cache setup and must not be compared directly with v2 results.

### Qwen3.5-9B weight-only INT8 on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_9b_int8_colab.ipynb)

[`qwen3_5_9b_int8_colab.ipynb`](notebooks/qwen3_5_9b_int8_colab.ipynb) (Korean) downloads
[`Jakemu/Qwen3.5-9B-int8-g128`](https://huggingface.co/Jakemu/Qwen3.5-9B-int8-g128) -- int8 weights with one bf16
scale per output channel and 128 input columns, lm_head in bf16 -- loads it with
`Qwen3_5ForCausalLM.from_quantized_dir`, and generates through the fixed-shape decoder
(`torch.compile` + CUDA graphs, the one-token matmul in the Triton kernel of `infer_opt/quant_kernels.py`).
On an A100 runtime it decodes at about 10.7 ms/token against 15.1 for bf16. It needs a GPU with
compute capability 8.0 or newer. `scripts/quantize_qwen3_5.py` makes such a checkpoint, and
§5 of [`qwen3_5_9b_benchmark_tables.ipynb`](notebooks/qwen3_5_9b_benchmark_tables.ipynb) has the
measurements and the accuracy gate.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_9b_int8_colab_t4.ipynb)

On a free **T4** runtime, [`qwen3_5_9b_int8_colab_t4.ipynb`](notebooks/qwen3_5_9b_int8_colab_t4.ipynb) loads the same
checkpoint in fp16 (`from_quantized_dir(..., dtype=torch.float16)`; Turing has no bf16). The bf16 model does not fit in a
T4's memory; the int8 one decodes at about 42-44 ms/token there. On an A100, the fp16 run of this checkpoint was as close to
the Hugging Face bf16 model as the bf16 run (mean decode KL 0.00056 vs 0.00071-0.00077, no confident mismatch).

### Qwen3.5-27B native NVFP4 on Colab G4

An experimental native NVFP4 W4A4 path is available for **RTX PRO 6000 Blackwell
(SM120)** in `infer_opt.nvfp4`. It uses FlashInfer 0.6.6's explicit CUTLASS backend,
with 16-value FP8 block scales and calibrated FP32 global scales. Large prefill
inputs are tiled at 128 rows because an untiled large-M CUDA Graph check failed.
The MTP head remains group-128 INT8, and lm_head/embeddings/small gates remain BF16.
This is a custom infer-opt format, not a Transformers or vLLM checkpoint. It loads
through `Qwen3_5ForCausalLM.from_quantized_dir`; original INT8 checkpoints retain
their existing loader behavior. CPU installs do not require FlashInfer or ModelOpt.

Run the bounded G4 experiment (the CLI must already be authenticated):

```bash
python scripts/run_qwen3_5_nvfp4_colab.py --output experiments/nvfp4/my-run --budget-minutes 120
```

The runner verifies native GEMM/compile/graph correctness before downloading 27B,
calibrates on 128 independent UltraChat conversations, exports and reloads a packed
checkpoint, and compares BF16/INT8/NVFP4 plus MTP. It records revisions, calibration
hashes, repeated timings and the existing HF accuracy gate. An explicitly requested
`--publish-repo user/repo` publishes the experimental artifact after the reload
check; the initial card identifies the full accuracy/performance evaluation as
pending. New sessions are always stopped; `--session` explicitly transfers ownership
of an existing session to the runner for cleanup. Predicted speedups are not measured
results, and failed or missing quality checks do not establish an improvement.

The experimental checkpoint is [Jakemu/Qwen3.5-27B-nvfp4-mtp](https://huggingface.co/Jakemu/Qwen3.5-27B-nvfp4-mtp):
17.93 GiB including MTP, 256 native NVFP4 target Linears, with identical saved/reloaded
final-position logits on the short release smoke prompt. The G4 native kernel suite
passed 18 shape cases. Full quality and performance evaluation are separate from
these storage/kernel checks; artifacts are under `notebooks/assets/qwen3_5_27b_nvfp4_g4_20261028/`.
Early NVFP4 MTP tests found same-prefix strong top-1 mismatches against one-token
decoding. Its MTP timing is diagnostic; greedy equivalence has not been established.

### Qwen3.5-27B weight-only INT8 on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_27b_int8_colab.ipynb)

The same recipe applied to the largest dense Qwen3.5. [`qwen3_5_27b_int8_colab.ipynb`](notebooks/qwen3_5_27b_int8_colab.ipynb)
(Korean) runs [`Jakemu/Qwen3.5-27B-int8-g128`](https://huggingface.co/Jakemu/Qwen3.5-27B-int8-g128) (27.8 GiB; the bf16
original is 51.7 GiB). On an A100-SXM4-80GB it decodes at 25.9 ms/token against 37.7 for bf16, and it fits on an A100 40GB.
It passed the accuracy gate in four runs, the last on the checkpoint downloaded from the Hub (mean decode KL
0.53-0.95x the bf16 control's, no confident mismatch). The GEMV launch rule in `infer_opt/quant_kernels.py` was refit
on both models' shapes; see `worklog.md` (2026-10-20).

The current code decodes the same checkpoint at **20.7 ms/token** (bf16 35.2; int8 + lm_head 20.3) after
`model.fuse_projections()`:

- a GEMV kernel that sums 16 int8 weights of a group in registers and scales once, at the speed of a kernel that
  only reads the weights;
- the projections that share an input (gate+up, q+k+v, in_proj_qkv+z) stacked into one GEMV each at load time;
- a split-K Triton kernel for one-token attention over grouped query heads (`infer_opt/attention_kernels.py`).

It passed the gate again (KL ratio 0.67-1.07, no confident mismatch). The 9B checkpoint decodes at 7.3 ms/token with
it. The 27B notebook pins this code (24.1 ms/token on an A100 40GB runtime, peak 29.1 GiB); the 9B notebooks still pin
the earlier commit. Measurements are in `notebooks/assets/qwen3_5_27b_int8_opt_a100_20261020/`.

Since then the fixed-shape decoder compiles with persistent reductions for rows up to 8192 wide
(`infer_opt/inductor_choices.py`): inductor's default limit of 1024 made every 5120-wide RMSNorm a looped kernel. Same-session
A/B on the Hub checkpoint: 20.31 -> 19.86 ms/token; the benchmark above now reads int8 20.0, int8 + lm_head 19.5 ms/token
(bf16 35.1), and the gate passed (KL ratio 0.58). Measurements are in `notebooks/assets/qwen3_5_27b_int8_norm_a100_20261024/`.

**MTP speculative decoding** (`infer_opt/speculative.py`, greedy, batch 1). Every Qwen3.5 checkpoint ships a
multi-token-prediction head (`mtp.*`: one decoder layer on the final hidden state, sharing the embedding and lm_head;
`from_pretrained_dir(..., mtp=True)`). It drafts `k` tokens; one fixed-shape forward of the `k + 1` tokens verifies them,
and the GatedDeltaNet layers keep their state after every token, so a rejected draft is undone by picking the state
instead of recomputing. A whole round (verify, accept, commit, drafts) is one compiled CUDA graph with one host sync. For
the verify forward the int8 GEMV gained a multi-row form (tensor cores, K split for the 5120-row shapes): 2-4 rows cost
1.06-1.26x one row, and the GatedDeltaNet layers verify the `k + 1` tokens in one Triton kernel that keeps the state
after every token (`infer_opt/gated_delta_kernels.py`). On the int8 27B with the MTP head int8 too and lm_head kept in
bf16 as in the published checkpoints (A100-SXM4-80GB, `k = 3`, 256 tokens): 19.8 -> **7.6** ms/token on prose, 7.8 on
math, 9.0 on code, 10.2 on a chat prompt and 13.7 on Korean (`k = 2`), 1.45-2.61x against the plain fixed-shape decoder
in the same session. The MTP head's precision only moves the acceptance, never the output; quantizing lm_head changes the
target model, so it stays opt-in. `scripts/bench_qwen3_5_spec_decode.py` measures it; results and per-kernel profiles of
a round are in `notebooks/assets/qwen3_5_27b_int8_spec_gdn_a100_20261025/` (and `..._spec_a100_20261024/` for the
first version, before the fused GatedDeltaNet kernel).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_27b_int8_spec_colab.ipynb)

[`qwen3_5_27b_int8_spec_colab.ipynb`](notebooks/qwen3_5_27b_int8_spec_colab.ipynb) (Korean) runs the self-contained
[`Jakemu/Qwen3.5-27B-int8-g128-mtp`](https://huggingface.co/Jakemu/Qwen3.5-27B-int8-g128-mtp) checkpoint on Colab.
Its 14 target shards are unchanged from the published INT8 model; one additional 0.402 GiB shard contains the
group-128 INT8 MTP head. `from_quantized_dir` restores it automatically from `quantization.json` (`mtp: true`), with
lm_head still in bf16. The notebook pins both code and weights, generates with both decoders, and compares their
speed and tokens. It needs an A100 (40 GB or more); no original BF16 weights or runtime quantization are needed.

`scripts/publish_qwen3_5_mtp.py` reproduces the release by range-reading only the original MTP tensors, quantizing
the head on CPU, and copying the existing target shards on the Hub. Its manifest records the pinned sources and
file hashes. `scripts/validate_qwen3_5_mtp_checkpoint.py` compares the bundled checkpoint with the original
`attach_mtp` workflow on an A100. Older target-only INT8 checkpoints still load without an MTP head.

The bundled release passed on an A100 40GB: all MTP tensors and final-position prefill logits matched the
separately attached reference, and all 64 generated tokens matched on prose, math and Korean with plain decoding
and MTP `k=1,3`. Peak allocated memory was 29.90 GiB. This validation used CUDA Graph decoding; the notebook's
compile mode was checked statically. [Release report and raw results](notebooks/assets/qwen3_5_27b_int8_mtp_a100_20261025/README.md).

### Qwen3.5-27B Hugging Face BF16 with compile + CUDA Graph

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_27b_hf_colab.ipynb)

Open [`qwen3_5_27b_hf_colab.ipynb`](notebooks/qwen3_5_27b_hf_colab.ipynb) in Colab for a standalone
**HF-original text-model** run. It loads only `Qwen/Qwen3.5-27B` in BF16, uses HF `StaticCache`, compiles
single-token decode with `torch.compile`, then explicitly captures and replays a CUDA Graph. An **80GB-class GPU**
is required; the hardware check rejects insufficient memory before downloading weights.

The notebook checks eager/compiled/graph logits against HF dynamic-cache eager, including cache reset, then
measures 1024-token prompt / 128-step batch-1 decode over five repetitions. Compilation, capture, prefill and
warmup are excluded from decode timing. It also generates text and exports timings, validation and environment
metadata as JSON. No 27B GPU measurements are pre-populated. Regenerate the notebook with
`python scripts/build_qwen3_5_27b_hf_colab_notebook.py`; its decode helpers are embedded, so no repository clone is needed.

### Local A100 result analysis

Open [`04_a100_roofline_analysis.ipynb`](notebooks/04_a100_roofline_analysis.ipynb)
with the repository's Python kernel. It reads local files without running inference
or connecting to a GPU runtime.

```bash
uv sync --group notebooks
# Select .venv as the notebook kernel in VS Code or Jupyter.
```

Set `BASELINE_PATH` to a saved baseline `results.json`. The default points to
`experiments/baseline/a100-0.5b-bf16-20261014-cli/results.json`.
If the raw run is unavailable, explicitly select the tracked summary at
`notebooks/assets/a100_baseline_20261014_summary.json` (no per-repeat samples).
Optionally set `PROFILE_PATH` to a saved `roofline_results.json` for operation GPU
times, measured reference ceilings, and per-operation roofline charts.

The baseline contains whole-phase timings only: missing operation times remain
missing. The notebook separates those measurements from operation FLOP/byte models
and an adjustable **hypothetical** roof; its defaults are not A100 measurements or
specifications. Logical bytes/time is not measured HBM bandwidth. Profiled fixed-prefix
one-token decode is kept separate from the baseline's 128-step average.

Derived CSVs, PNGs, and an input SHA-256 manifest are saved under
`experiments/roofline/local-analysis/`; source result files are not modified.
Regenerate the notebook with `uv run python scripts/build_a100_roofline_analysis_notebook.py`.

## Curriculum

### Interactive website

The Korean **Inference Lab** playground connects workload controls to a roofline chart,
prefill/decode latency estimates, a token timeline, and memory breakdowns. Guided experiments
compare batching, long context, and compute versus bandwidth. Everything runs in the browser;
Python, a GPU, and a backend are not needed to use the site.

With Node.js 22.12+ and npm installed:

```bash
cd web
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. Use **기준 저장** to pin the current configuration for comparison,
or select an experiment to load both its baseline and changed settings. **초기화** restores defaults.

All displayed performance is **analytic, not measured**. Hardware defaults are a hypothetical
100 TFLOP/s and 1000 GB/s profile. Memory traffic follows `infer_opt.profiling`, including its
baseline activation-traffic approximation. Precision changes element size, not the compute input.
The roofline shows modeled ceilings, while latency is a lower-bound estimate within that model.
Open **계산 가정과 한계 알아보기** in the site for accounting conventions and limitations.

Frontend validation and a production preview:

```bash
cd web
npm test
npm run build
npx playwright install chromium  # first browser-test run only
npm run test:browser
npm run preview
```

The static production site is built into `web/dist/`. Fonts are bundled locally.
To regenerate TypeScript test fixtures from the Python helpers, run from the repository root:

```bash
uv run python web/tests/generate_fixtures.py
```

### Notebooks

This repository is both an experiment template and a course: the notebooks in `notebooks/`
use the very model and cache defined here as their subject, so a lesson and a benchmark are
the same artifact. Notebooks are committed with their outputs, measured on the machine named
in each one's first cell -- rerun them to get your own numbers.

```bash
uv sync --group notebooks
uv run jupyter lab notebooks/
```

| Chapter | Notebook | Covers |
|---|---|---|
| 1 | [Inference performance fundamentals](notebooks/01_inference_performance_basics.ipynb) | latency vs throughput, TTFT vs TPOT, prefill vs decode, FLOPs / bandwidth / arithmetic intensity, compute- vs memory-bound, the roofline model, and how batch size, sequence length and hidden dimension move a workload across it |
| 2 | [Operation breakdown and attention](notebooks/02_operation_breakdown_and_attention.ipynb) | splitting a step into embedding / norm / QKV / RoPE / attention / KV read-write / output projection / MLP / LM head / sampling and attributing device time to each, measured share against FLOP share, the causal mask decode does not need, MHA vs MQA vs GQA, FlashAttention-style tiling, KV cache layout, PagedAttention, and sampling cost |
| 2A | [FlashAttention lab (Colab, A100)](notebooks/02b_flash_attention_a100_colab.ipynb) | online softmax derived and verified, the tiled attention loop written out by hand, all four SDPA backends including the real `FLASH_ATTENTION`, memory scaling to the limits of an A100, the FlashAttention 2 kernel loaded from the Hub with `kernels`, and `flash_attn_with_kvcache` for variable-length decode |

`infer_opt.profiling` holds the measurement and analytic-cost helpers the notebooks share:
CUDA-event timing, CUDA-graph capture (for workloads too small to keep the host ahead of the
GPU), measured peak FLOPs and bandwidth, and FLOP/byte counts for any prefill or decode step.

Chapter 2A is the exception to the rule above: it is **committed without outputs**, because the
machine these notebooks are measured on is an RTX 2080 SUPER (sm75) and the kernel the lab exists
to study needs sm80 or newer. It is self-contained -- it imports nothing from `infer_opt` and needs
only the `torch` that Colab already ships -- so open it in Colab on an A100 runtime and the first
numbers in it will be yours. Chapter 2 §7 covers the same ground as far as a Turing card allows.

`infer_opt.opprofile` takes that one step further and splits a single forward step by operation.
`infer_opt.model` names each span through `infer_opt.regions.region` -- free unless a profiler is
running -- and `profile_stages()` attributes device time to those names, while `stage_flops()` and
`stage_bytes()` count what each span was obliged to do. Comparing the two is the chapter-2
diagnostic: a stage whose time share dwarfs its FLOP share is bound by memory or kernel launches,
not by arithmetic. `warmup_device()` is there because an idle GPU clocks down, and the first
measurement of a session lands several times too slow without it.

`infer_opt.paged` implements `PagedKVCache`: block-structured KV storage with a per-sequence block
table and a shared block allocator, interchangeable with `KVCache` under the same
`write`/`read`/`advance` contract. Select it with `--cache paged`. It bounds wasted memory by
`block_size` instead of by `max_seq_len`; per-sequence lengths are tracked but the decode path
still advances the batch as a unit, so continuous batching remains future work.

`infer_opt.sampling` provides greedy, top-k and top-p selection -- deliberately unfused, so the
cost the curriculum measures is the algorithm's rather than one kernel's cleverness.

## Research extension points

`DecoderOnlyTransformer.forward_prefill()` writes an entire prompt into `KVCache`; `forward_decode()` accepts exactly one next token and appends to that same cache. `AttentionStrategy` is the optimization seam: register a uniquely named strategy in `infer_opt.strategies`, then select it through `--strategy` to compare it against `eager_sdpa` under identical settings. Registered alongside it are `sdpa_math` and `sdpa_mem_efficient` (the same call pinned to one SDPA backend, which is how chapter 2 measures the cost of materialising the score matrix), `gqa_native` (grouping handled by the kernel instead of by copying K/V), and `decode_nomask` (a single query at the end of the cache needs no causal mask at all).

The initial weights and inputs are deterministic random tensors. Tokenizers, pretrained checkpoints, serving, and training are intentionally outside this template so that an optimization experiment has a small, reproducible baseline. Sampling is included only as far as the curriculum profiles it (`infer_opt.sampling`); there is no generation loop built on it.
