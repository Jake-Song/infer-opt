# HF BF16 versus llama.cpp Q8_0 numerical comparison

This harness reproduces the nine prompts, 48 cached-greedy positions per prompt,
KL direction `KL(HF || candidate)`, top-1 agreement, and strong-mismatch rule
(`HF top1 - top2 > 0.5`) used by `validate_qwen3_5_moe_quality.py`.
It measures numerical distribution preservation, not task accuracy.

The HF reference generates all forced tokens. llama.cpp consumes those exact IDs
with explicit positions; it does not tokenize, sample, add BOS, or stop at EOS.
The first comparison position is the last prefill output; the remaining 47 are
single-token cached decode. Every token defined by the HF tokenizer is checked
against the GGUF vocabulary before logits are compared. Padded model vocabulary
rows without an HF tokenizer entry are included in the distribution comparison
but have no string mapping to check.

## Build and execute

Pin the original HF snapshot and the llama.cpp/GGUF revisions in `worker.py`.
Put model weights and caches under `/tmp/infer-opt-models`.

```sh
cmake -S scripts/llamacpp_quality -B /tmp/infer-opt-llama-quality-build \
  -DLLAMA_CPP_SOURCE=/path/to/pinned/llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/infer-opt-llama-quality-build -j 12 \
  --target infer-opt-llama-replay

python3 scripts/validate_llamacpp_qwen3_5_quality.py reference \
  --weights /tmp/infer-opt-models/hf-reference-snapshot \
  --reference-dir /tmp/infer-opt-quality-reference
python3 scripts/validate_llamacpp_qwen3_5_quality.py compare \
  --reference-dir /tmp/infer-opt-quality-reference \
  --llama-bin /tmp/infer-opt-llama-quality-build/infer-opt-llama-replay \
  --gguf /tmp/infer-opt-models/Qwen3.5-35B-A3B-Q8_0.gguf \
  --out artifacts/qwen35_llamacpp_hf_quality
```

HF uses BF16 with SDPA. llama.cpp uses full GPU layer offload, Flash Attention,
F16 K/V cache, batch 2048, ubatch 512, four CPU threads, one sequence, and no
MTP/speculative calls. A fresh llama.cpp context is allocated for each prompt.
All contexts use the maximum prompt-plus-continuation capacity across this suite.
The processes are sequential so the models do not share GPU memory.

`worker.py` runs the same procedure on a G4, checks actual GPU/VRAM, downloads
only public sources and models, verifies the GGUF LFS SHA-256, and archives
inputs, metrics, manifests, source, and logs. Full-logit files are temporary
comparison buffers and are excluded from the report archive. No HF credential
or Drive mount is used. The caller must retrieve results and stop the session.

The resulting difference includes both weight quantization and engine/kernel/
cache precision differences. Comparing against historical infer-opt numbers is
not a paired comparison unless the exact HF logits and software are shared.

## Local checks

```sh
.venv/bin/python -m pytest tests/test_llamacpp_quality.py -q
```

The tests cover analytical KL/argmax behavior, the HF-margin rule, teacher-forced
input alignment, and malformed output handling. Native header compilation is
separate from real-model CUDA validation.
