# A100 down projection: instruction-issue bottleneck check — 2026-11-11

**현재 packed 1-warp down을 순수한 instruction-issue-throughput bound로 보는 가설은 이번 측정에서 지지되지 않았다.** Hot/cold 시간이 비슷한 현상은 재현됐지만, 실제 issue 활성률은 약 63%/56%였다. 같은 1-warp 대조에서 동적 warp 명령 수를 약 28% 줄여도 시간은 거의 줄지 않았다. 기존 4-warp reference는 hot issue 활성률이 약 88%로 더 높았다. 두 경로의 병목을 동일하게 취급하지 않는다.

## 실행 범위와 환경

- Colab **NVIDIA A100-SXM4-40GB**, driver **580.82.07**, CUDA toolkit **13.0.88**, Nsight Compute **2025.3.1**.
- Python 3.13, PyTorch **2.14.1+cu130**, Triton **3.8.0**. 이전 RunPod 비교의 PyTorch 2.9.1/Triton 3.5.1과 다르므로 역사적 절대 시간을 직접 비교하지 않는다.
- 모델 없이 합성 **W8A16 group128 down projection**만 실행했다. Experts 256, hidden/output 2048, K=512, top-k 8, BF16 activation/output, FP32 누산. INT8 weight와 BF16 scale은 seed 901로 생성했다.
- Router, gate/up, shared expert, expert 가중합, 전체 target body 및 MTP round는 포함하지 않는다. 모델 다운로드와 production 커널 수정은 없다.
- T=1/2는 부분 공유 route, T=4는 완전 공유·부분 공유·비공유를 사용했다. Scalar/packed × 1/4 warp의 네 구성을 측정했다. T=1 packed 측정은 진단용 override이며 production의 T=1 dispatch를 뜻하지 않는다.

## 측정 방법

Profiler 없는 시간은 CUDA Graph replay를 같은 stream의 CUDA event로 측정했다. 40개 조건 × 순서를 섞은 3개 배치 × 100회, **12,000개 sample**이다. 각 조건을 먼저 warmup하고 hot은 반복 재생, cold는 매 sample 전 128 MiB buffer를 썼다. Flush와 300,000-cycle prelude는 측정 구간 밖이다. 표는 세 배치 중앙값의 중앙값이며 p10/p90과 개별 sample은 원본에 있다.

카운터는 별도 프로세스에서 `cudaProfilerStart/Stop`으로 down kernel 하나만 선택했다. `--replay-mode application --cache-control none --clock-control none`으로 각 pass에서 warmup/flush를 다시 실행했다. 넓은 hot/cold section 수집 뒤 핵심 metric 대조를 실행했다. 후속 대조는 시작 전 1,000회 graph replay도 추가했다.

**클럭은 고정하지 않았다.** 시간 측정 후 telemetry는 대체로 SM 1410 MHz였지만 NCU에서 실제 측정한 SM 주파수는 약 1.09 GHz였다. 긴 warmup을 추가해도 이 차이는 남았다. 따라서 profiler 없는 지연과 NCU 지연은 서로 다른 표로 유지한다. NCU 지연을 production 시간이나 별도의 가속률로 사용하지 않는다. 카운터에 따른 판정은 기록된 profiling 조건에 대한 것이다.

## Profiler 없는 지연

T=4 부분 공유, 단위 **µs/down**.

| 구성 | Hot | Flush 후 | 기존 4-warp reference와 출력 exact |
|---|---:|---:|---|
| 기존 scalar 4 warp | 55.30 | 64.51 | 예 |
| 현재 packed 1 warp | 56.32 | 55.30 | 예 |
| Scalar 1 warp 대조 | 55.30 | 55.30 | 아니오 |
| Packed 4 warp 대조 | 61.44 | 67.58 | 예 |

현재 packed 1-warp의 완전 공유 hot/cold는 56.32/55.30 µs, 비공유는 55.30/55.30 µs였다. 이전에 가설의 출발점이었던 cache/route 둔감성이 재현됐다. Scalar 4-warp hot의 첫 배치 중앙값은 71.68 µs였고 후속 두 배치는 55.30 µs였다. 클럭을 고정하지 않은 이 실험에서 수 퍼센트 차이를 안정적인 개선으로 확정하지 않는다.

## 현재 경로의 실제 카운터

후속 핵심 metric 수집, T=4 부분 공유 packed 1 warp.

| 지표 | Hot | Flush 후 |
|---|---:|---:|
| NCU 지연, µs | 68.448 | 68.992 |
| SMSP issue active, active-cycle 기준 | 62.68% | 56.05% |
| Eligible warps / scheduler / active cycle | 1.40 | 1.04 |
| Achieved occupancy | 25.36% | 27.23% |
| DRAM throughput / sustained peak | 1.85% | 34.12% |
| L2 throughput / sustained peak | 13.37% | 25.45% |
| DRAM read, decimal MB | 1.840 | 21.340 |
| 동적 warp instruction 수 | 16,056,320 | 16,056,320 |
| Long-scoreboard stall / active warp | 21.00% | 40.26% |

넓은 section 재측정에서도 hot/cold issue active는 62.86%/55.97%였다. 같은 실행의 L2 hit rate는 92.49%/50.33%였다. 즉, hot/cold 시간이 비슷하다고 cache가 영향을 주지 않은 것은 아니다. DRAM traffic과 memory-dependency stall은 실제로 달라졌다.

Hot의 ALU/FMA/LSU/XU active throughput은 각각 약 44.57%/44.08%/42.12%/31.35%였다. HBM/L2 bandwidth 포화나 특정 측정 파이프라인의 지속적인 처리량 포화를 확인하지 못했다. 넓은 수집에서 현재 커널은 40 registers/thread, shared memory 0, 이론 occupancy 50%, 실제 약 25–27%였다. 이론 50%는 1-warp block과 SM의 block 수 한계에 따른 상한이며 register 한계가 더 낮은 상한을 만든 경우는 아니다.

Warp stall 비율은 active warp의 상태 비율이며 전체 wall time의 비율이 아니다. Stall들이 존재한다는 사실만으로 단일 원인을 확정하지 않는다. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)에 따라 issue slot의 공백, eligible warps, pipeline 및 memory 지표를 함께 해석했다.

## 명령 수 대조와 판정

같은 1-warp의 scalar 대조는 동적 warp 명령 수가 **11,534,336개**로 현재 packed의 16,056,320개보다 **28.16% 적었다**. 그런데 NCU 지연은 **68.448 → 67.936 µs**, profiler 없는 hot은 **56.32 → 55.30 µs**였다. issue active는 62.68%에서 52.48%로 오히려 낮아졌다. 명령 수 감소가 시간 감소로 비례해 이어지지 않았다.

이는 명령 발행 용량이 지배적으로 포화됐다는 설명에 반하는 대조다. 다만 두 커널은 load 명령·register 사용·reduction도 달라지므로 완벽하게 명령 수 하나만 바꾼 인과 실험은 아니다. Scalar 1-warp는 bit-exact도 아니어서 최적화로 채택하지 않았다.

기존 scalar 4-warp hot은 **27,459,584개** 명령, issue active **87.66%**, eligible **4.37**, occupancy **79.54%**였다. 현재 packed 경로보다 명령 발행 압력이 높다는 해석은 가능하다. 그렇다고 이 한 수치만으로 기존 커널의 순수 issue-bound까지 확정하지 않는다.

Packed 4-warp hot도 issue active **87.92%**, occupancy **83.51%**였지만 명령 수는 **30,670,848개**, NCU 지연은 **77.088 µs**였다. Warp 수를 늘려 occupancy만 올리는 것으로 개선되지 않았다. 산술 명령·여러 warp 사이 reduction 및 launch 구성을 함께 고려해야 한다.

**판정:** 현재 packed down의 순수 issue-throughput 포화 가설은 지지되지 않는다. 메모리·연산 의존성 대기와 낮은 실제 warp 병렬성이 관측되며 latency hiding/launch 구성은 후속 후보지만, 지배적인 단일 원인은 아직 확정하지 않았다. 특히 block dispatch 포화나 register 병목을 입증했다고 주장하지 않는다. 이 결과를 gate/up 또는 전체 MTP 병목으로 확대하지 않는다.

## 수치 검증과 실행 기록

- Scalar 4-warp, packed 1-warp, packed 4-warp의 모든 시간 조건은 reference 출력과 exact였다. Scalar 1-warp는 작은 reduction 차이가 있어 별도로 기록했다.
- Scalar 1-warp의 전체 측정 조건에서 reference 대비 최대 절댓값 차이는 **0.00048828125**였다. 네 구성 모두 독립 reference 검사의 최대 차이는 **0.0009765625**였다.
- 각 구성은 독립 FP32 tensor reference로 token 0의 8개 expert 전체를 검사했다(`atol=0.002`, `rtol=0.02`, BF16 output). 이 허용오차 검사는 bit-exact를 대신하지 않는다.
- 각 graph의 마지막 replay는 같은 구성의 eager 출력과 exact였고 모든 출력이 finite였다. 이 검사는 모델 품질·logit·state·장기 생성 검증이 아니다.
- 첫 실행은 scalar 1-warp에도 reference exact를 요구하여 정식 timing 전에 중단됐다. 원본을 `attempt-1-results.tar.gz`와 `results/attempt-1/`에 보존했다. 이후 진단 변형의 수치 차이를 명시적으로 기록하도록 harness를 바꿨으며 production 정확도 기준이나 커널은 바꾸지 않았다.
- 모델 가중치·다운로드는 없고 Triton cache는 원격 `/tmp/infer-opt-models/down-issue`에만 두었다.

재현 코드: [scripts/down_issue](../../scripts/down_issue/). [analysis.json](results/analysis.json), [시간 요약](results/timing-summary.csv), [시간 원본](results/timing/timings.json), 각 `.ncu-rep` 및 `.raw.csv`/`.raw-base.csv`, PTX/SASS를 보존했다. `analyze.py`는 sample 수·중앙값·route·실제 kernel/grid/block·profiling/timing output hash를 검산한다. 측정 단계별 source archive와 hash는 `source-manifest.json`에 기록한다.

## 완료와 정리

넓은 hot/cold **2개**와 부분 공유 T=4 핵심 대조 **6개**, 총 **8개 NCU report**를 성공적으로 수집·검산했다. 핵심 질문에 필요한 대조가 끝나 추가 route/T=1/2 counter sweep은 중단했다. 이 조건들의 profiler 없는 시간은 이미 모두 측정했다. 중단된 추가 counter 실행의 부분 로그는 성공한 report에 포함하지 않는다. `triage-interruption.json`과 `core-complete.json`에 범위 변경 및 실제 중단 지점을 보존했다. `worker-complete`/`focused-complete` 전체 sweep 성공을 주장하지 않는다.

최종 archive **2,170,055 bytes**, SHA-256 **`b0ee2e70e1d22463aa821da0b7df47c455e8542af9b35fe158fbdb2c8d1da027`**를 원격 값과 대조했다. 회수·검산 뒤 전용 Colab 세션 `down-issue-a100-20261111`을 종료했고 서버의 활성 세션 **0개**를 확인했다. [archive 검산](archive-validation.json), [최종 검증](validation.json), [정리 기록](cleanup.json).
