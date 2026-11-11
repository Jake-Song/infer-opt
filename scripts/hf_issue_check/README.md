# HF Jobs performance-counter access check

`probe.py` records UID, Linux effective/permitted/bounding capability sets and
the host NVIDIA driver's profiling restriction. It compiles a small CUDA kernel
and attempts one Nsight Compute collection before any larger benchmark is run.
No models or Hugging Face credentials are passed into the container.

`control.py` uses the existing local Hugging Face login to submit an A100 job,
inspect its status and recover a SHA-256-verified result archive from its logs.
Run it with the Python interpreter from the HF CLI environment:

```sh
/home/jake/.hf-cli/venv/bin/python scripts/hf_issue_check/control.py submit \
  --script scripts/hf_issue_check/probe.py --out artifacts/hf_issue_check/probe
/home/jake/.hf-cli/venv/bin/python scripts/hf_issue_check/control.py logs \
  --job JOB_ID --out artifacts/hf_issue_check/probe
```

The job has a ten-minute timeout and one attempt. A completed diagnostic job can
still contain a failed profiler result: inspect `results/result.json`, not only
the job stage. If hardware counters are denied, retain the capability and driver
diagnostics and avoid inferring an instruction bottleneck from the error.
