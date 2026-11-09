# A100 MoE MTP round profiling — 2026-11-09

사용자 요청에 따라 **A100-SXM4-80GB에서 직접 실행**했다. 기존 compile + CUDA Graph 경로를 유지한 trace와 별도 구간 계측을 수집했다. 가장 큰 구조적 병목은 routed expert의 다중 토큰 비용이며, 추가 draft마다 반복하는 BF16 vocabulary projection도 크다. CPU 호출 비용도 관측됐지만, 수동 graph 대조 경로에서 일부 상태가 달라 최적화의 동등성은 확정하지 않았다.

## 실행 조건

- GPU: NVIDIA A100-SXM4-80GB, 79.250732 GiB visible. 전용 Colab 세션 `moe-round-profile-a100-20261109`.
- PyTorch 2.11.0+cu130, Triton 3.6.0. 체크포인트 `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`.
- INT8 group128 target/MTP, BF16 router·작은 gate·lm_head, fuse 82 groups. Batch 1, cache 2048.
- 원래 경로: `compile-reduce-overhead`, fullgraph/static, max_autotune + coordinate_descent_tuning. MTP parallel verification, `allow_inexact=True`.
- 수학·한국어, MTP off/k=1/k=3. 각 256토큰 × 5회, warmup 3회. 원래 경로 trace는 각 24 step/round, 별도 host/stream 계측도 각 24 round.
- 구간 계측: 프롬프트별 8 round 이후 상태를 고정하고 매회 복원, 각 20회. Expert micro는 40 target expert block을 모델 순서대로 실행하며 128 MiB L2 flush 후 각 10회.
- 모델·Inductor/Triton 캐시는 원격 `/tmp/infer-opt-models/q35moe-round-profile-20261109`에 두었다. 저장한 결과에는 모델 가중치가 없다.

## 원래 경로의 실제 생성 시간

컴파일·prefill·첫 토큰·warmup·profiler를 제외한 host wall clock 중앙값이다. 분모는 `emitted_tokens − 1`이며, MTP 마지막 round의 초과 출력도 포함한다. 출력 불일치가 있는 행은 속도 진단용이다. 이전 실행의 일치 결과를 이번 실행에 적용하지 않는다.

| 경로 | 입력 | ms/token | ms/round | token/round | 일반 decode와 불일치한 실행 |
|---|---|---:|---:|---:|---:|
| plain | math | 5.593 | — | 1.000 | 0/5 |
| plain | korean | 5.563 | — | 1.000 | 0/5 |
| mtp1 | math | 5.598 | 10.940 | 1.954 | 0/5 |
| mtp1 | korean | 6.346 | 11.160 | 1.759 | 5/5 |
| mtp3 | math | 4.173 | 15.165 | 3.634 | 5/5 |
| mtp3 | korean | 6.723 | 15.307 | 2.277 | 5/5 |

이번 실행에서 MTP k=3은 수학에서도 5/5 불일치했다. 이전 [별도 A100 실행](../qwen35_moe_compiled_20261109/README.md)의 수학 k=3 일치를 일반화할 수 없다. 원래 parallel compiled MoE 경로의 수치 민감성과 품질 한계는 그대로 남는다. 이번에는 HF BF16 정확도 평가를 다시 실행하지 않았다.

## 원래 fullgraph의 GPU 커널 비용

Chrome trace의 CUDA kernel/memcpy/memset duration 합을 호출 수로 나눴다. **profiler 영향이 있으므로 위 wall time과 합산하거나 직접 대체하면 안 된다.** BF16 항목은 이름에 matmul이 포함된 fused kernel도 포함하며 순수 lm_head만 뜻하지 않는다. Top-k/sort는 copy/gather에서 분리했다.

| GPU ms/call, 수학 | MTP off (1 step) | k=1 (1 round) | k=3 (1 round) |
|---|---:|---:|---:|
| Routed experts | 1.667 | 2.836 | 5.155 |
| 일반 INT8 projections | 1.560 | 1.906 | 1.942 |
| BF16 matmul 포함 fused kernels | 1.204 | 2.385 | 3.415 |
| Routing top-k/sort | 0.564 | 0.697 | 0.747 |
| 전체 GPU kernel 합 | 6.205 | 9.577 | 13.365 |

k=1→3의 증가분 3.788 ms 중 expert가 61.2%, BF16 matmul 포함 커널이 27.2%를 차지한다. Expert 비용은 off 대비 k=3에서 약 3.1배다. 일반 INT8 projections는 약 1.2배로 다중 토큰 처리가 상대적으로 잘 상각된다.

6개 trace 모두 24 CUDA Graph launch를 확인했다. Expert 호출 수는 각각 40/41/43 block × (`_project` 2회 + `_combine` 1회)와 일치한다. 각 MTP round의 GPU event 수는 k=1 1271개, k=3 1335개다. 커널 개수 증가만으로 설명되는 현상은 아니다.

## CUDA event로 나눈 구간

구간마다 따로 compile한 뒤 하나의 수동 CUDA Graph 안에서 external CUDA event로 측정했다. 구간 분리는 fusion/tuning 경계를 바꾸므로 원래 production의 정확한 구성요소별 시간으로 간주하지 않는다. 검증 body·lm_head·MTP body·commit의 대략적인 크기를 비교하기 위한 진단이다.

| 고정 상태 GPU ms, 수학 | k=1 | k=3 |
|---|---:|---:|
| Target verify body | 7.135 | 9.363 |
| Verify lm_head + argmax | 0.616 | 0.622 |
| Draft lm_head + argmax 합 | 0.607 | 1.842 |
| MTP body 합 | 0.207 | 0.545 |
| Accept/commit + pack | 0.207 | 0.208 |
| 전체 | 8.772 | 12.581 |

각 행은 각 구간의 중앙값이므로 합이 total 중앙값과 미세하게 다를 수 있다. MTP body 자체보다 반복되는 큰 vocabulary projection의 비용이 크다. accept/commit·pack은 약 0.21 ms여서 첫 최적화 대상으로는 작다.

## CPU와 수동 fullgraph 대조

원래 compiled 함수의 CPU 호출 wall time은 별도 계측에서 약 1.5–2.0 ms/round였다. 뒤이어 `.tolist()`를 기다리는 CPU 시간은 GPU 실행과 겹치므로 GPU 시간에 더하지 않는다. 일반 decode는 루프를 enqueue한 뒤 끝에서 동기화하지만 MTP는 round마다 readout한다.

추가 대조에서는 전체 `spec_round`를 그대로 하나의 함수로 compile하되 자동 CUDA Graph wrapper를 끄고 수동 graph에 capture했다. 아래 시간은 고정된 한 round의 동일한 readout을 포함하며 전체 256토큰 생성 성능이 아니다.

| k / 입력 | 원래 host ms | 수동 fullgraph host ms | 수동 GPU ms | 수동 graph와 원래 경로의 동일 상태 텐서 | segmented/full GPU |
|---|---:|---:|---:|---:|---:|
| 1 / math | 9.625 | 8.793 | 8.750 | 49/145 | 1.0025 |
| 1 / korean | 9.583 | 8.787 | 8.741 | 145/145 | 0.9986 |
| 3 / math | 13.886 | 12.733 | 12.685 | 15/145 | 0.9918 |
| 3 / korean | 14.247 | 12.704 | 12.647 | 99/145 | 0.9906 |

모든 비교는 target IDs·accepted count·다음 input IDs 및 전체 상태 텐서를 별도로 기록했다. **일부 상태가 달라 수동 graph의 시간 차이를 동등한 최적화의 확정된 이득으로 사용할 수 없다.** 상태에는 KV·recurrent·conv 및 speculative buffer 전체가 포함된다. 불일치가 active state인지 scratch인지 이 계측에서는 분해하지 않았다. CPU wrapper를 줄이는 것은 다음 실험 후보이며, 실제 채택에는 상태 차이 원인과 전체 생성 parity를 먼저 해결해야 한다.

1차 diagnostic의 `segmented_over_production`은 CPU dispatch gap이 섞인 stream interval과 GPU 내부 시간을 비교한 값이다. 이를 fusion 개선율로 해석하면 안 된다. 후속 calibrated 단계에서 수동 fullgraph GPU 대조를 추가했다.

## Expert만 분리한 cold-cache 실험

아래는 1차 diagnostic에서 실제 eager replay로 수집한 activation/routes를 사용한 40개 target expert block sweep이다. 각 토큰이 같은 expert 집합을 사용하도록 강제한 대조도 실행했다. 강제 route는 성능 진단용이며 모델 출력·정확도 비교가 아니다.

| 입력 | 토큰 | 실제 route ms | 모든 토큰 동일 route ms | 실제 route 중복 잠재량 |
|---|---:|---:|---:|---:|
| math | 1 | 1.662 | 1.661 | 0.0% |
| math | 2 | 2.862 | 2.811 | 24.2% |
| math | 4 | 5.274 | 5.165 | 34.8% |
| korean | 1 | 1.417 | 1.417 | 0.0% |
| korean | 2 | 2.408 | 2.346 | 25.2% |
| korean | 4 | 4.393 | 4.281 | 43.5% |

중복 잠재량은 레이어별 `1 − unique_experts / total_routes`의 평균이다. T=4에서 실제 route에도 35–44%의 중복 잠재량이 있으나, 모든 토큰에 같은 expert를 강제해도 시간 절감은 약 2–3%다. 현재 `_project`의 토큰·expert별 GEMV 구조가 토큰 간 weight tile을 공유하지 않는다는 코드 관찰과 일치한다. HBM byte·bandwidth·SM occupancy는 측정하지 않았으므로 하드웨어 counter에 근거한 원인 확정은 아니다. 후속 calibrated 단계의 독립 micro 반복도 원본 JSON에 보존했다.

## 다음 최적화 우선순위

1. **Expert의 다중 토큰 weight 재사용**: 같은 expert에 배정된 토큰을 모아 작은 GEMM/tile로 처리하는 방식이 가장 큰 GPU 구조적 후보다. 토큰 grouping 비용까지 포함하고 현재 INT8 기준 결과와 parity를 확인해야 한다.
2. **Draft vocabulary projection**: k=3의 draft lm_head 3회만 약 1.8 ms다. BF16 유지 상태에서 커널/argmax 경로를 개선하거나 별도 양자화 실험을 평가할 가치가 있다. 양자화는 출력 수치가 달라질 수 있으므로 별도 품질 검증이 필요하다.
3. **CPU compiled wrapper/readout**: 반복되는 CPU 호출 비용을 줄이는 후보를 검사한다. 수동 graph 대조의 상태 불일치가 해결되기 전에는 적용된 성능 개선으로 보고하지 않는다.
4. Accept/commit은 현재 약 0.2 ms로 상대적으로 작다.

## 검증·재현 자료

- [validation.json](validation.json): 30회 timed generation의 7,680 token ID, 분모·수락 histogram·불일치 위치, 6개 trace, 구간 sample 합, source hash 검산.
- [analysis_summary.json](analysis_summary.json): 원래 trace 재분류, expert 호출 coverage, CPU/GPU envelope. `analyze_profiles.py`로 재생성한다.
- [results/production](results/production): 각 24회 원래 경로 Chrome trace `.json.gz`, 5회 generation sample/IDs, host/stream 계측.
- [results/calibrated/summary.json](results/calibrated/summary.json): 최종 4개 frozen-point 대조 및 상태 일치/불일치. 세부 samples와 expert micro도 같은 폴더에 있다.
- [results/diagnostic/summary.json](results/diagnostic/summary.json): 1차 구간 진단. 이 단계는 상태 텐서 비교를 수행하지 않았다.
- `results/compiler-python.tar.gz`와 manifest: 생성된 Inductor Python 소스만 보존한다. 모델 캐시·가중치는 포함하지 않는다.
- [source-manifest.json](source-manifest.json), `source.tar.gz`: 실제 production에 전달한 모델·커널 소스. 원래 profiler v1은 `results/profile_qwen3_5_moe_round.py`, 후속 v2는 `results/calibrated-profile-source.py`이며 각 SHA-256을 검산했다.
- `diagnostic-harness.json`의 최초 업로드 설명에는 v2 업로드가 initial diagnostic launch보다 먼저였다는 부정확한 문구가 있다. 실제 initial diagnostic 프로세스는 v1을 로드했고, v2는 별도 fresh calibrated 프로세스가 실행했다. `calibrated-harness.json`에 이를 기록했다.
- CPU parser 테스트 3개 통과. 모델 구현의 수치 동등성·정확도 통과를 뜻하지 않는다.
- 원래 경로 6개 trace와 1차/후속 구간 진단이 모두 exit 0으로 끝났다. 런타임 정리 증거는 `cleanup.json`에 기록한다.
