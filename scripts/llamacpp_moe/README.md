# Model-free llama.cpp MoE expert benchmark

This experiment compares native llama.cpp Q8_0/Q8_1 W8A8 operations with infer-opt
group-128 W8A16 kernels on one A100. It uses deterministic synthetic tensors and
does not download a model. The two quantization formats have different arithmetic.

`patch_source.py` requires llama.cpp commit
`a0ad695cb212db3850f163121ddc687513448d42`. It adds cases to the existing
`test-backend-ops` program and host measurement accessors. CUDA kernel code and
dispatch are retained. The patch is for a private benchmark checkout.

Each case uses 256 experts, top-k 8, hidden size 2048 and expert intermediate
size 512. Token counts are 1, 2 and 4. Routes are fully shared, half shared, or
disjoint across tokens, with no repeated expert inside one token. IDs are
contiguous `[8,T]` int32 tensors in ggml, corresponding to `[T,8]` in PyTorch.

| Scope | ggml activation shape | Weight shape / layout |
| --- | --- | --- |
| Gate projection | `[2048,1,T]` | Gate view of `[2048,1024,256]` gate/up buffer |
| Down projection | `[512,8,T]` | `[512,2048,256]` |
| Gate/up + SwiGLU | `[2048,1,T]` | Gate and up views of the same combined buffer |
| Full routed expert block | `[2048,1,T]` | Gate/up, SwiGLU, down, weighted expert sum |

`INFER_MOE_NATIVE_LAYOUT=1` instead allocates separate contiguous gate and up
weights `[2048,512,256]`. They contain the same source values as the combined
buffer views. This preserves native gate/up fusion; the combined-buffer view
control does not. Both variants are kept in the experiment results.

Shapes above list `ne[0]` first. Output/input shapes, byte strides, view offsets,
and all route IDs are emitted as `INFER_LAYOUT` JSON records. Native activation
and output precision differences are recorded; there is no cross-engine exact
equality requirement.

The patched program supports these environment variables:

- `INFER_MOE_CASES=1`: use only the 36 custom correctness cases.
- `INFER_MOE_SIMPLE=1`: keep gate/down cases for standard perf. Repeating just
  the last node of a fused graph would not time the whole graph.
- `INFER_MOE_MEASURE=1`: time the graph captured by the backend, with CUDA events
  on its execution stream. Emit hot and flush-before-run samples separately.
- `INFER_MOE_REPEATS`: samples per cache condition (default 100).
- `INFER_MOE_SELECT`: select a case by substring of its parameter string.
- `INFER_MOE_NATIVE_LAYOUT=1`: use native contiguous gate/up weights.
- `INFER_MOE_DOT_DIR`: export the actual captured CUDA graphs, including kernel
  names, before the temporary CUDA graph is destroyed.

Both native implementations use a 300,000-cycle prelude outside the timed
interval so event and graph launches can be queued before the GPU reaches the
start event. The prelude does not touch benchmark data. Flush-before-run uses a
128 MiB buffer before that prelude and before the start event. Timings include
all kernels captured for the operation, including activation quantization.

`worker.py` builds and runs on a dedicated machine under
`/workspace/infer-opt-llama-moe`. Use a Python virtual environment with access to
the machine's CUDA PyTorch installation. Caches are under `/tmp/infer-opt-models`.
`followup.py` runs after the worker completes: this upstream revision's CSV
printer omits performance columns, so it preserves readable console timings and
collects separate matched counter probes. Counter collection failures are saved.
`native_followup.py` then rebuilds with native-layout cases, reruns correctness
and timing, and exports native and view-control CUDA graphs. The graph evidence
can be collected without performance-counter permission.

`compare.py` times the existing reference and packed infer-opt implementations.
Single gate projection has no paired timing because the existing infer-opt
projection fuses gate/up and SwiGLU. Down, fused gate/up and full expert block
are paired. T=1 uses the ordinary implementation for both baseline labels.

Run `python scripts/llamacpp_moe/analyze.py /path/to/recovered/results` to check
sample counts, route overlap, layouts and replay equality and generate
`analysis.json`. Standalone kernel measurements do not establish MTP throughput,
model quality, or the cause of a bottleneck without supporting hardware counters.
`validate_fixtures.py` checks the C++ and Python source values and native/view
weight indexing agree. It requires a local C++ compiler.
