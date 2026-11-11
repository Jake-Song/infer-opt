# Matched gate/up, down and routed expert block experiment

This model-free harness extends the down-only diagnosis to the existing fused
gate/up + SwiGLU kernel and the full routed expert block. Production kernel code
is unchanged. It reuses the route fixture and kernel loader from
`scripts/down_issue/benchmark.py`.

`preflight.py` records the A100 environment and requests a fixed SM clock of
1095 MHz. `counter_probe.py` tests actual NCU counter access before the larger
measurement. `worker.py` aborts if clock locking or counter access fails, and
restores default GPU clocks in its finalizer.

`benchmark.py` times gate/up, standalone down and the explicit three-kernel block
(gate/up, down, ordered expert combine). The block is separately checked against
the production dispatcher for the same kernel configuration. It uses 256
experts, hidden 2048, intermediate 512, top-k 8 and group 128. Weights are
synthetic INT8, scales and activations BF16, accumulation FP32. No model is loaded.

The main matrix uses T=1/2/4, partial routes, both kernel families at one/four
warps, plus fully shared/disjoint T=4 current-kernel controls. There are 84
hot/flush-before-run conditions, three shuffled batches and 100 samples per
batch, totaling 25,200 samples. Triton caches stay under
`/tmp/infer-opt-models/expert-issue`.

The current A100 T=2/4 path is **packed one-warp**. `scalar4` names the old
`_project` four-warp implementation used for numerical comparison; it is not
the current T=2/4 baseline. `scalar1` and `packed4` are diagnostic overrides.
“Scalar” is a source-family label: the compiler can vectorize its loads. The
default T=1 production path remains the old four-warp implementation.

Every configuration is compared with the old arithmetic, and exactness is
recorded separately from an independent tensor reference tolerance
(`atol=0.002`, `rtol=0.02`, including explicit BF16 rounding). Last graph replay
must equal the same variant's eager result exactly. Non-exact variants remain
diagnostic and do not authorize a production or model-quality claim.

`worker.py` profiles eight T=4 partial-route projection cases, using application
replay, cache-control none and clock-control none while the GPU is externally
locked. The selected kernel is gated by cudaProfilerStart/Stop. Each application
pass reconstructs its cache state; compilation and validation are excluded.

`shared_followup.py` waits for the main worker and its clock reset, locks the
same clock again, and profiles fully shared hot gate/up. This reduces the weight
working set from 40 MiB to 16 MiB while keeping logical work unchanged. It then
restores clocks. `launch.py`, `launch_shared.py` and `status.py` are session-specific
Colab helpers with fixed `/tmp` paths; inspect them before reuse.

After recovering results:

```sh
python scripts/expert_issue/analyze.py artifacts/expert_issue_a100_20261111/results --require-complete
```

The analyzer verifies all samples, shuffled-batch coverage, finite outputs,
production-dispatch block equality, requested clocks, actual kernel/grid/block,
and matching timing/profile output hashes. It exports timing and counter CSVs.

It also computes **ideal lower bounds**, not additive time components:

- Issue: warp instructions / (SM count × peak warp IPC per SM × measured SM clock).
- DRAM: measured bytes / NCU's peak byte rate.
- L2: measured sectors / NCU's peak sector rate.

The maximum of these bounds excludes latency, dependencies, synchronization,
tail effects and other overhead. Exceeding a bandwidth bound alone does not
identify issue saturation. Preserve active-cycle and elapsed-cycle issue metrics
separately. A repeated hot workload need not fit in cache.

The experiment README records interpretation, raw report locations, source
hashes and runtime cleanup. Terminate the dedicated runtime after recovery.
