# G4 compiled parallel MoE MTP

Run Qwen3.5-35B-A3B INT8 group128 on one dedicated RTX PRO 6000 Blackwell 96GB.
Compare compiled ordinary decode with compiled `verification=parallel` at k=1/2/3.
This fills the missing G4 compiled parallel experiment; the old G4 compiled
record used `verification=consistent`.

The run uses the published `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp` checkpoint at
`331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`. All weights and model/compiler caches
are under `/tmp/infer-opt-models/moe-compiled-g4-20261112` on the remote runtime.
No BF16 reference weights are downloaded. The five frozen prompts, cache capacity
2048, 256 requested output tokens, three warmups and five repetitions are shared
by all four methods. Generated IDs are compared to the same compiled ordinary
decoder. This is a greedy-parity check, not HF BF16 equivalence or task accuracy.

`torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`,
max_autotune and coordinate_descent_tuning are enabled, matching the earlier
A100 compiled parallel experiment. PyTorch 2.11.0+cu130, Triton 3.6.0 and FLA
0.6.0 are pinned. Kernel/warp settings use the repository defaults; no override.
The dedicated GPU uses provider automatic boost, recorded by one-second telemetry.

`preflight.py` checks actual hardware before model download. `launch.py` embeds
only the reviewed source bundle, worker and prior public prompt fixtures and
starts a detached worker. `worker.py` verifies source/checkpoint hashes, installs
dependencies, and runs `scripts/bench_qwen3_5_moe_compiled.py` unchanged. Results
are snapshotted periodically without model/cache contents. `status.py` reads
the existing worker state. Paths and session names are specific to this run.

Timings exclude prefill, first token, compilation, warmup and profiler activity.
MTP includes final-round overshoot in the emitted-token denominator, following
the earlier compiled benchmark contract. Keep raw IDs, accepted-token histograms
and per-repetition times; mismatching configurations remain diagnostic timings.
CUDA Graph launch evidence is gathered separately after each method.

Recover and hash-check the final result archive, independently validate timings
and IDs, then stop the exact dedicated session and confirm the server assignment
is gone. Allocation failures must not trigger another allocation or another GPU
type without authorization.
