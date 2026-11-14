# G4 MoE prefill backend with MTP on and off, 2026-11-14

MTP on/off greedy parity holds under the route-grouped prefill backend exactly
as under the reference loop: with sequential and consistent verification, all
20 MTP-on runs per backend reproduced their MTP-off 256-token output. On that
basis `grouped` is now the default `Int8Experts.prefill_backend`.

The backend does change long greedy outputs. With MTP off, all five 256-token
continuations left the reference ones, each at a token where the reference's
top two logits were tied or one BF16 step (0.125) apart. Prefill became
11.5–17× faster and decode time did not move.

## Method

RTX PRO 6000 Blackwell Server Edition, PyTorch 2.11.0+cu130, Triton 3.6.0, FLA
0.6.0; published INT8 checkpoint at `331d7db9` with its bundled MTP head; batch
one; CUDA-graph decode (`FixedShapeDecoder`, `MTPSpeculativeDecoder`), cache
capacity 2048. Each backend was set on all 41 expert banks: the target's 40 and
the MTP head's, whose own prefill runs over every prompt row.

Five natural prompts, 256 generated tokens each. Per backend and prompt: MTP
off, and MTP on with sequential and consistent verification at k=1 and 2. Every
configuration was generated twice and gave the same tokens both times.

Prefill and decode are timed separately and never added. Prefill is the call
that yields the first token (median of three after one warmup; with MTP on it
also runs the MTP head's prefill and the first drafts). Decode is one timed run
over the 255 tokens after the first, so it is a check that nothing moved, not a
benchmark. Three prompts are 20–43 tokens long: the TTFT reference remains the
128–4096 token runs.

## MTP off: reference against grouped

| Prompt | Prompt tokens | Prefill, reference | Prefill, grouped | Decode ms/token, reference / grouped | First changed token (0-based) | Reference top-2 margin there |
|---|---:|---:|---:|---:|---:|---:|
| chat_long | 32 | 637.1 ms | 44.1 ms | 8.82 / 8.82 | 121 | 0.125 |
| code_1k | 1000 | 1252.1 ms | 73.6 ms | 8.85 / 8.69 | 219 | 0.000 |
| prose_1k | 1057 | 1199.5 ms | 74.3 ms | 8.87 / 8.77 | 155 | 0.125 |
| math | 43 | 670.0 ms | 44.3 ms | 8.84 / 8.84 | 2 | 0.125 |
| korean | 20 | 504.6 ms | 43.9 ms | 8.83 / 8.83 | 43 | 0.125 |

The gate allowed a changed token only where the reference margin is at most
0.5, the repository's strong-mismatch threshold; all five are at 0.125 or an
exact tie. BF16 logits step by 0.125 at these magnitudes, and the reference
itself has an exact top-2 tie somewhere in four of the five continuations, so
any rounding-level change in the cache eventually picks the other token. After
the first changed token the two continuations are different texts and were not
compared further. Which one is closer to the BF16 model was not measured.

## MTP on

| | Reference | Grouped |
|---|---:|---:|
| Runs equal to MTP off, sequential + consistent (of 20) | 20 | 20 |
| Prefill with MTP on, k=1 | 519–1286 ms | 45.7–76.8 ms |
| Prefill with MTP on, k=2 | 521–1291 ms | 47.1–77.9 ms |
| Decode ms/token, sequential | 9.88–13.65 | 9.85–13.55 |
| Decode ms/token, consistent | 9.27–12.58 | 9.24–12.48 |
| Tokens per round, k=1 / k=2 | 1.77–2.00 / 2.18–2.99 | 1.73–1.99 / 2.18–2.99 |

With the grouped backend MTP adds 1.8–2.5 ms (k=1) or 2.9–3.7 ms (k=2) to the
prefill call; with the reference loop it added 14–39 ms. The MTP head's
numerics affect only drafts, and acceptance is unchanged within a few percent.
Sequential and consistent verification remain slower than MTP off, as recorded
before.

## Parallel verification (diagnostic, added at the user's request)

Parallel verification is not token-exact for this model under either backend:
3 of 15 runs (k=1, 2, 3) equalled MTP off under the reference loop and 3 of 15
under grouped, and not the same three (prose k=1,2 and code k=3 against prose
k=1,2,3). The gate therefore reports one failure, code_1k k=3 leaving MTP off
at token 242 under grouped, alongside prose_1k k=3, which matches under grouped
but not under the reference loop. Decode was 6.6–7.6 ms/token at k=1, 5.0–6.9
at k=2 and 4.1–7.5 at k=3 against 8.8 with MTP off; those ratios are not
output-preserving speedups. This stage ran in the same session after the worker
exited ([parallel-launch.py](parallel-launch.py)); the worker script now
includes it.

## Decode speed by k

One timed run of the 255 tokens after the first, CUDA-graph decode (not
compiled), milliseconds per emitted token with the ratio to MTP off of the same
backend and prompt in parentheses. `†` marks a run whose output left MTP off;
its ratio is not an output-preserving speedup. MTP token counts include the
final round's overshoot.

Parallel verification:

| Backend | Prompt | MTP off | k=1 | k=2 | k=3 | Tokens per round k=1 / 2 / 3 |
|---|---|---:|---:|---:|---:|---:|
| grouped | chat_long | 8.82 | 6.91† (1.28×) | 5.89† (1.50×) | 5.30† (1.66×) | 1.90 / 2.53 / 3.08 |
| grouped | code_1k | 8.84 | 6.68† (1.32×) | 5.32† (1.66×) | 4.67† (1.90×) | 1.97 / 2.79 / 3.45 |
| grouped | prose_1k | 8.85 | 6.59 (1.34×) | 4.95 (1.79×) | 4.06 (2.18×) | 1.99 / 2.99 / 3.98 |
| grouped | math | 8.80 | 6.74† (1.31×) | 5.38† (1.64×) | 4.50† (1.95×) | 1.94 / 2.73 / 3.58 |
| grouped | korean | 8.83 | 7.60† (1.16×) | 6.88† (1.28×) | 7.51† (1.18×) | 1.71 / 2.12 / 2.12 |
| reference | chat_long | 8.82 | 7.01† (1.26×) | 6.06† (1.46×) | 5.19† (1.70×) | 1.88 / 2.46 / 3.15 |
| reference | code_1k | 8.84 | 6.66† (1.33×) | 5.40† (1.64×) | 4.83 (1.83×) | 1.98 / 2.77 / 3.39 |
| reference | prose_1k | 8.86 | 6.62 (1.34×) | 5.04 (1.76×) | 4.42† (2.01×) | 2.00 / 2.99 / 3.72 |
| reference | math | 8.84 | 6.74† (1.31×) | 5.42† (1.63×) | 4.33† (2.04×) | 1.95 / 2.76 / 3.81 |
| reference | korean | 8.83 | 7.39† (1.20×) | 6.56† (1.35×) | 6.38† (1.38×) | 1.77 / 2.26 / 2.54 |

Sequential and consistent verification (every run equals MTP off):

| Backend | Prompt | MTP off | Sequential k=1 | Sequential k=2 | Consistent k=1 | Consistent k=2 |
|---|---|---:|---:|---:|---:|---:|
| grouped | chat_long | 8.82 | 10.27 (0.86×) | 11.77 (0.75×) | 9.75 (0.90×) | 10.71 (0.82×) |
| grouped | code_1k | 8.69 | 9.86 (0.88×) | 10.52 (0.83×) | 9.48 (0.92×) | 9.65 (0.90×) |
| grouped | prose_1k | 8.77 | 9.87 (0.89×) | 9.85 (0.89×) | 9.49 (0.92×) | 9.24 (0.95×) |
| grouped | math | 8.84 | 9.97 (0.89×) | 10.41 (0.85×) | 9.45 (0.94×) | 9.56 (0.92×) |
| grouped | korean | 8.83 | 11.35 (0.78×) | 13.55 (0.65×) | 10.76 (0.82×) | 12.48 (0.71×) |
| reference | chat_long | 8.82 | 10.54 (0.84×) | 11.91 (0.74×) | 10.09 (0.87×) | 11.04 (0.80×) |
| reference | code_1k | 8.85 | 9.95 (0.89×) | 10.59 (0.84×) | 9.52 (0.93×) | 9.81 (0.90×) |
| reference | prose_1k | 8.87 | 9.88 (0.90×) | 9.98 (0.89×) | 9.47 (0.94×) | 9.27 (0.96×) |
| reference | math | 8.84 | 10.09 (0.88×) | 10.58 (0.84×) | 9.67 (0.91×) | 9.81 (0.90×) |
| reference | korean | 8.83 | 11.16 (0.79×) | 13.65 (0.65×) | 10.64 (0.83×) | 12.58 (0.70×) |

Sequential and consistent verification were not run at k=3. These are
CUDA-graph numbers; the compiled parallel runs of 2026-11-12 (MTP off 4.5–4.6,
k=3 2.8–3.5 ms/token) are a different execution mode and are not comparable.

## Not measured

HF BF16 equivalence or task accuracy with the grouped backend, compiled MTP,
k above 3, batch sizes above one, other GPUs, and outputs beyond 256 tokens.

## Evidence

[validation.json](validation.json) recomputes every divergence, count and gate
verdict from the 23,040 retained token IDs and rechecks 125 file hashes:

```sh
python scripts/moe_prefill_g4/validate.py \
  --results artifacts/qwen35_moe_prefill_mtp_g4_20261114/results
```

This run has no raw traces: every file of the recovered archive except the
remote run's four pytest cache files is extracted under `results/`. The archive
itself is not kept in the repository ([recovery.json](recovery.json) records
its size and SHA-256).

[results/mtp/mtp-parity.json](results/mtp/mtp-parity.json) and
[results/mtp-parallel/mtp-parity.json](results/mtp-parallel/mtp-parity.json)
hold the tokens, margins and timings. The executed sources still had
`reference` as the class default; both backends were set explicitly. The G4 was
allocated without `--high-mem` with the user's approval and stopped after
verified recovery; see [lifecycle.json](lifecycle.json). Reproduce with
`prepare.py --run YOUR_RUN --plan mtp`.
