# A100 BF16 GEMM SASS capture

Captured on NVIDIA A100-SXM4-40GB with PyTorch 2.11.0+cu130 and Nsight Compute 2025.3.1.
Workload: contiguous BF16 `(2048, 4096) @ (4096, 4096)`, preallocated output,
20 warmup iterations, then one GEMM collected with the basic metric set (10 replay passes).

- `sass.txt`: SASS exported from the collected report. Includes 128 static instruction lines containing `HMMA.16816.F32.BF16`; this is not a dynamic execution count.
- `a100_sass.zip`: full capture including `nsight/gemm.ncu-rep`, worker source, capture command, version and logs.

Extract the ZIP and open `nsight/gemm.ncu-rep` in Nsight Compute, select the kernel,
then open Source → SASS. Captured kernel:
`ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`.

NCU replay timings are separate from the notebook's CUDA-event benchmark results.
The allocated Colab runtime was terminated after downloading the capture.
