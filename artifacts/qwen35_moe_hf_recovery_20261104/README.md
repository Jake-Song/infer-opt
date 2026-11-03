# Published Qwen3.5 MoE INT8 + MTP checkpoint

Release: [Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp](https://huggingface.co/Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp)

Commit: `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`.

The custom infer-opt W8A16 group-128 checkpoint includes MTP: 22 weight shards,
37,087,658,536 bytes (34.540574 GiB), plus tokenizer/configuration files, license,
and release metadata. It loads through `Qwen3_5ForCausalLM.from_quantized_dir`,
not Transformers AutoModel or vLLM.

## Verification

- Source: `Qwen/Qwen3.5-35B-A3B`, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Inference/conversion code: `b5ae4228a361fa562063fac7cd46dd19a611e9f2`.
- GPU: RTX PRO 6000 Blackwell Server Edition, 94.97 GiB VRAM (Colab G4).
- All 22 regenerated weight shards match the earlier validated export's SHA-256.
- Strict reload succeeded with 41 INT8 expert modules and MTP.
- Ordinary 64-token generation succeeded for `Explain why the sky is blue.`
- MTP k=1,2 matched ordinary decoding with both sequential and consistent
  verification: four token-parity checks passed.
- All 35 payload files passed local and remote size/SHA-256 checks. Remote LFS
  hashes were checked at the exact release commit; small metadata files and the
  manifest were downloaded and hashed.
- CPU conversion peak RSS was 26.855965 GiB; CUDA was not initialized during conversion.

This validates packaging and the recorded generation checks. It is not a new
task-quality evaluation, A100 validation, or an MTP speedup claim.

## Delivery

Publication used local Hugging Face authentication. No HF token was transferred
to Colab. Files were individually verified and pre-uploaded, then published in
one commit. The Colab runtime was stopped after the full local backup passed
verification; the backend reported no active sessions.

The local checkpoint in `experiments/qwen35_moe_hf_rebuild2/checkpoint/` was
deleted at the user's request after publication. The HF release and validation
records remain available. The source and checkpoint weights are not checked into Git.

| File | Evidence |
|---|---|
| `publish_result.json` | Published commit, verified file list and validation summary |
| `release_manifest.json` | SHA-256 and size of every payload file |
| `regeneration_verified.json` | All 22 weights match the preceding export |
| `checkpoint-validation.json` | GPU reload, generation, MTP parity and profiler output |
| `conversion_info.json` | Source hashes, weight bytes and CPU peak RSS |
| `download_complete.json` | Complete local backup verification |
| `environment.json` | Source/code revisions and GPU/software identity |
| `runtime_cleanup.json` | Dedicated runtime termination and backend session check |

## Load the pinned release

With the inference repository and CUDA dependencies installed:

```python
from huggingface_hub import snapshot_download
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM

path = snapshot_download(
    "Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp",
    revision="331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388",
)
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
```

Use `scripts/publish_qwen3_5_moe.py --export-only` on the GPU runtime and
`--publish-existing` on the authenticated local machine to reproduce publication.
For a new release, supply a fresh `--repo` destination to both commands.
