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
