# Qwen3.5-27B MLP-only NVFP4 accuracy ablation

- Execution: finished
- Evidence complete: True
- MLP-only accuracy gate passed: False
- HF reference: cached prompt decode and cached greedy generation; margins from the same generation logits.
- Scope: MLP gate/up/down W4A4; all attention projections BF16; MTP INT8, lm_head BF16.
- Calibration: same 128 UltraChat token sequences as the original full-NVFP4 run.

| Config | Prompt decode KL vs HF | KL / BF16 | Strong / near-tie | Free-run identical |
|---|---:|---:|---:|---:|
| bf16 | 0.00163033 | 1.000 | 0 / 6 | 6/9 |
| nvfp4 | 0.04707331 | 28.874 | 29 / 49 | 0/9 |
| nvfp4-mlp | 0.02373341 | 14.557 | 12 / 33 | 0/9 |

MLP-only weight files: 27.589 GiB (runtime buffers excluded).

This run changes both the reference protocol and the quantization scope. Compare the full and MLP-only checkpoints within this run to attribute scope effects; older KL values use a different HF reference.
No MTP-head-only accuracy test or throughput benchmark is part of this accuracy ablation.
