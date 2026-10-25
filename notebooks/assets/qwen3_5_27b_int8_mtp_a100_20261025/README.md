# Qwen3.5-27B INT8 with bundled MTP: release validation

Published model: [Jakemu/Qwen3.5-27B-int8-g128-mtp](https://huggingface.co/Jakemu/Qwen3.5-27B-int8-g128-mtp).
Tested weight revision: `0d68f2ac933cdc988e1875aecceb0a2c8ff473db`.
Code revision: `eaa6d3551670ca110cfcfff19b35a6d64f420ac5`.

The release contains the original 14 INT8 target shards plus one 0.402 GiB MTP shard: 28.197 GiB of weights in total.
Target and MTP Linear weights use INT8 group-128 quantization; lm_head stays BF16. Loading the bundled checkpoint
requires no original BF16 model download. The subsequent reference check explicitly fetches the original MTP tensors.

## Results

| Check | Result |
|---|---|
| Local quantization, MTP checkpoint and speculative-decoding tests | 41 passed; 80 CUDA tests deselected |
| Target shard hashes against the pinned INT8 base | All 14 unchanged |
| Saved/reloaded MTP tensors on CPU | All 23 exactly equal |
| Bundled MTP versus original BF16 head quantized on GPU | All 23 tensor hashes, shapes and dtypes equal |
| Final-position prefill logits: bundled versus attached | Exactly equal on all 3 prompts; maximum absolute difference 0 |
| Same-path generation: bundled versus attached | All token IDs equal for plain, MTP k=1 and MTP k=3 |
| Plain versus speculative generation | All 64 generated token IDs equal on each tested prompt and k |
| Peak allocated GPU memory | 29.897 GiB |

Prompts: `prose_1k` (1057 input tokens), `math` (43), and `korean` (20); each generates 64 new tokens.
Batch size 1; `kernels="fla-delta"`, fused projections, manually captured CUDA Graph decoding (`mode="cudagraph"`).
The reference loads the unchanged target shards with the original target-only metadata, attaches the original
BF16 MTP head, and quantizes its eight Linear modules on GPU. Models are loaded sequentially.

These checks establish checkpoint equivalence for the tested paths. The notebook's compile mode was checked
statically (notebook schema, Python syntax and revision pins), not executed as a complete notebook. This release
run did not repeat the BF16 Transformers accuracy gate or measure throughput. Token agreement is reported for
the tested prompts and lengths; it is not a guarantee against floating-point argmax differences on other inputs.

## Runtime and artifacts

- Requested and allocated: one Colab A100, NVIDIA A100-SXM4-40GB, 39.494 GiB visible memory.
- Python 3.13.16; torch 2.11.0+cu130; transformers 5.20.2; flash-linear-attention 0.6.0;
  huggingface_hub 1.33.0; safetensors 0.8.0.
- Setup, download and validation completed in 277 seconds. The session was stopped afterward;
  the server reported no active sessions.
- [validation.json](validation.json) contains actual token IDs, tensor hashes, comparisons, and runtime versions.
- `publish_manifest.json` and `quantize_info.json` record the release inputs and prepared artifact hashes.
  Later Hub documentation and report updates do not change the tested weight shards.

Reproduce on an A100 after checking out the code revision above and installing the model-card dependencies:

```bash
python scripts/validate_qwen3_5_mtp_checkpoint.py \
  --revision 0d68f2ac933cdc988e1875aecceb0a2c8ff473db \
  --out validation.json
```
