# Native llama.cpp Qwen3.5 MoE MTP

The worker builds unmodified llama.cpp at commit
`a0ad695cb212db3850f163121ddc687513448d42` and downloads the MTP-inclusive
`unsloth/Qwen3.5-35B-A3B-MTP-GGUF` Q8_0 file at revision
`63af8373893a7a73c6dfcb84cb63d815981da5e0`. Weights and model caches stay under
`/tmp/infer-opt-models`. No CUDA kernel or warp configuration is changed.

The dedicated runtime must be A100 80GB. `preflight.py` checks hardware and tools;
`lock_clock.py` sets SM clocks to 1095 MHz. Restore clocks before stopping it.
`launch.py` embeds only the worker and the two public prompt fixtures and starts
a detached process. It refuses to launch a duplicate worker. Runtime paths and
the Colab session name are specific to the recorded execution; inspect them
before reuse.

In a fresh checkout, restore the raw `.sqlite` and `.nsys-rep` files from the
artifact's `final-results.tar.gz` before running `analyze_traces.py`; expanded
trace copies are intentionally ignored by Git. See the artifact README for the
exact restore command.

Batch 1, context capacity 2048, full GPU offload, flash attention, F16 K/V,
greedy sampling, 256 generated tokens. EOS is ignored to keep output lengths
equal. `/completion` receives fixed token IDs, independently checked against
the native tokenizer. Each request disables prompt caching. There are three
warmups per prompt and configuration per server instance. Two batches reverse
the off/k=1/k=2/k=3 order and contribute three plus two timed samples.

Main timing is the server's synchronized generation wall time after the first
token, divided by 255. It includes native sampling, speculative driver,
accept/rollback and decode overhead. Loading, compilation, warmup, prefill,
first token and HTTP transport are excluded. HTTP wall time is kept separately.
This is a different workload and timing contract from the historical random-token
G4 `llama-bench` measurement.

Raw `/completion` responses preserve generated IDs and draft acceptance counts.
Before/after Prometheus metrics preserve speculative verification counts.
`ms_round` in raw records divides the entire decode duration by that counter;
the counter excludes any ordinary single-token steps, so it is not an isolated
round measurement. Use token latency as the main throughput result.

`status.py` snapshots results and logs without weights, build products or caches.
Download and verify that archive, then run:

```sh
python3 scripts/llamacpp_mtp/analyze.py artifacts/llamacpp_mtp_a100_20261111/results
```

The analyzer checks all forty timed requests, forty-eight warmups, input IDs,
time denominators, source/model hashes, locked clock telemetry and output parity.
An output mismatch keeps the corresponding time diagnostic; output agreement on
these prompts does not establish HF BF16 numerical equivalence or task quality.

`profile_worker.py` waits for the timing worker and collects eight separate
whole-request CUDA node traces with Nsight Systems. `analyze_traces.py` checks
SQLite integrity, graph launches, default MoE block dimensions, and profiled
output agreement with the same unprofiled mode. Traces include prefill; they
do not separately label target/draft phases and do not replace decode timings.
The recorded initial probe timed out while waiting for a long-lived launcher;
the trace worker was changed to use `Popen` before any trace started. The old
source and failed probe are preserved. No timing worker or kernel changed.

`finish.py` requires both workers to have finished, restores GPU clocks, and
creates a final archive with a per-file hash manifest. Recover and verify that
archive before stopping the exact named Colab session. `lite_status.py` checks
progress without recompressing trace files.
