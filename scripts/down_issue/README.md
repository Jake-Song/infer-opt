# Model-free A100 down instruction-issue diagnosis

This harness diagnoses the existing down projection without changing production
kernels or downloading a model. It requires CUDA PyTorch, Triton, nvcc and Nsight
Compute on an A100, plus working hardware-counter access.

`preflight.py` records the environment and driver/capability state.
`counter_probe.py` compiles and profiles a tiny CUDA kernel. Check
`probe-result.json`: successful CUDA execution alone is insufficient.

`benchmark.py` uses the repository's unmodified `infer_opt/moe_kernels.py`. It
generates seeded synthetic INT8 weights, BF16 scales and activations for experts
256, output 2048, K=512, top-k 8 and group size 128. It times scalar/packed ×
1/4-warp kernels at T=1/2/4. Graph timing uses three shuffled batches of 100
samples for each of 40 conditions, with hot and 128 MiB flush-before-run cases.

```sh
python scripts/down_issue/benchmark.py --out /tmp/down-issue-a100-20261111/timing
```

All variants are checked against an independent tensor reference. Exact equality
to the original four-warp arithmetic is separately recorded. Scalar one-warp
can differ due to reduction order and remains diagnostic; this is not an
automatic production optimization or a model-quality test. Every final replay
must match the same variant's eager output exactly.

`--profile` warms the selected kernel and brackets one eager launch with
cudaProfilerStart/Stop. Use NCU `--profile-from-start off` to exclude compilation,
reference checks and warmup. `--warm-profile` adds 1,000 graph replays first.
Application replay with cache-control none reconstructs the requested cache state
on each pass. Profiling and unprofiled timing are separate runs; measured clocks
can differ even with warmup, so do not mix their durations.

`worker.py` collects the full timing matrix and broad section profiles.
`focused_worker.py --core-only` collects the six primary T=4 partial-route NCU
comparisons using a selected metric set. Both write remote results under
`/tmp/down-issue-a100-20261111`, and Triton caches under
`/tmp/infer-opt-models/down-issue`. They should run as dedicated remote processes.

`launch.py`, `launch_focused.py`, `finish_core.py` and `status.py` are Colab control
helpers for the recorded execution. They use fixed paths and session-specific
state; inspect and adapt them before reuse. `status.py` exports raw metric tables
and snapshots results. The recorded run collected broad hot/cold first, then
stopped after all six primary comparisons rather than completing optional counter
sweeps. Successful and interrupted work are recorded separately.

After downloading and verifying the archive, analyze the recovered directory:

```sh
python scripts/down_issue/analyze.py artifacts/down_issue_a100_20261111/results
```

The analyzer verifies 12,000 samples, batch medians, routes, actual kernel/grid/
block dimensions, and matching output hashes between timing and NCU runs. It
retains original and base-unit raw metrics and computes timing summaries. A
readable interpretation is in the experiment README. Terminate the dedicated
Colab session after recovery.
