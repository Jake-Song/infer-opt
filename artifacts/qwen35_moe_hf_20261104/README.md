# Qwen3.5 MoE checkpoint regeneration, 2026-11-04

This first regeneration passed validation, but its upload was interrupted and
the temporary local checkpoint was subsequently lost. The
[completed release](../qwen35_moe_hf_recovery_20261104/README.md) regenerated all
22 weight shards with identical SHA-256 values and published them successfully.

Regenerated `Qwen/Qwen3.5-35B-A3B` from source revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, using inference and conversion code
at `b5ae4228a361fa562063fac7cd46dd19a611e9f2`.

The custom infer-opt W8A16 group-128 export includes MTP: **22 weight shards,
37,087,658,536 bytes (34.540574 GiB)**. Tokenizer files, configuration, license,
and release metadata are additional. This is not a Transformers/vLLM format.

## Validation

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GiB VRAM (Colab G4).
- Strict reload succeeded with 41 quantized expert modules and the MTP head.
- Ordinary 64-token generation succeeded for `Explain why the sky is blue.`
- MTP k=1,2 matched ordinary output token-for-token with both sequential and
  consistent verification (four checks).
- CPU conversion peak process RSS: 26.853840 GiB; CUDA was not initialized in
  the converter. This measurement belongs to this regeneration, not the earlier
  export whose RSS was unmeasured.
- All 35 payload files were downloaded locally and checked against SHA-256
  entries in `release_manifest.json`.
- The dedicated Colab runtime was terminated after verified local download;
  the backend reported no active sessions.

These are export/reload and generation checks, not a new task-quality study,
A100 validation, or evidence of an MTP speedup. The recorded single-prompt
decode timing is a validation diagnostic, not a replacement performance study.

## Files

| File | Evidence |
|---|---|
| `environment.json` | Source/code revisions and GPU/software identity |
| `conversion_info.json` | Shards, total weight bytes, source hashes and CPU peak RSS |
| `checkpoint-validation.json` | Strict reload, token parity, sample output and kernel profile |
| `release_manifest.json` | SHA-256 and size of every release payload file |
| `download_complete.json` | Successful local transfer verification |

## Reproduce

On the GPU runtime, without a Hugging Face credential:

```sh
python3 scripts/publish_qwen3_5_moe.py \
  --work-dir /path/to/release \
  --code-revision b5ae4228a361fa562063fac7cd46dd19a611e9f2 \
  --export-only
```

Copy the complete `checkpoint/` directory to the publishing machine. With
local Hugging Face write authentication:

```sh
python3 scripts/publish_qwen3_5_moe.py \
  --work-dir /path/to/release \
  --code-revision b5ae4228a361fa562063fac7cd46dd19a611e9f2 \
  --publish-existing
```

The publisher verifies local hashes before uploading the corresponding files.
After upload, it compares remote LFS SHA-256 values and file sizes, downloads
small metadata files for hashing, and verifies the manifest at the exact commit.
The destination must not already exist, unless `--resume-empty-repo` is used for
an inspected empty repository. `--wait-for-files` can stage individually verified
files while atomic local downloads finish; the final commit includes every file.
