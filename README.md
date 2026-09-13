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

`infer_opt.profiling` holds the measurement and analytic-cost helpers the notebooks share:
CUDA-event timing, CUDA-graph capture (for workloads too small to keep the host ahead of the
GPU), measured peak FLOPs and bandwidth, and FLOP/byte counts for any prefill or decode step.

## Research extension points

`DecoderOnlyTransformer.forward_prefill()` writes an entire prompt into `KVCache`; `forward_decode()` accepts exactly one next token and appends to that same cache. `AttentionStrategy` is the optimization seam: register a uniquely named strategy in `infer_opt.strategies`, then select it through `--strategy` to compare it against `eager_sdpa` under identical settings.

The initial weights and inputs are deterministic random tensors. Tokenizers, pretrained checkpoints, sampling, serving, and training are intentionally outside this template so that an optimization experiment has a small, reproducible baseline.
