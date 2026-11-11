# A100 gate/up + down instruction-issue experiment — 2026-11-11

**Gate/up은 down보다 명령 발행 압력이 높았지만, 현재 packed 경로에서 순수 issue-throughput 포화를 확정하지 못했다.** 같은 1-warp의 기존 커널 대조는 gate/up의 동적 명령 수를 약 37% 줄이고 시간을 약 24–25% 줄였다. 명령 처리 비용이 중요한 최적화 대상이라는 증거지만, 이 변형은 bit-exact가 아니며 load·loop·reduction도 달라 단일 변수 인과 실험이나 채택된 최적화는 아니다.

## 범위와 기준

- Colab **A100-SXM4-40GB**, driver **580.82.07**, CUDA toolkit **13.0.88**, Nsight Compute **2025.3.1**, PyTorch **2.14.1+cu130**, Triton **3.8.0**.
- GPU SM 클럭을 **1095 MHz로 외부 고정**한 상태에서 profiler 없는 시간과 NCU를 모두 측정했다. 모든 시간 조건의 telemetry는 1095 MHz였다. NCU의 실제 elapsed-cycle 기반 주파수도 약 1.08–1.09 GHz였고 검산했다. 이전 실험의 일반 측정/NCU 클럭 차이를 이번에는 줄였다.
- 모델 없는 합성 INT8 group128 W8A16. Experts 256, hidden 2048, intermediate 512, top-k 8. BF16 activation/output, FP32 누산. 가중치 다운로드와 production 커널 변경은 없다.
- **Gate/up은 두 projection과 SwiGLU가 융합된 구간**이다. Down standalone은 독립 합성 activation을 사용한다. 전체 routed expert block은 gate/up·SwiGLU·down·expert 가중합을 직접 실행한다. Router·top-k 생성·shared expert·attention·lm_head·MTP round는 제외한다.

현재 A100 T=2/4의 주 대상은 **`_project_packed`, CTA당 1 warp**다. 보고서와 코드의 `scalar4`/reference는 pair-load 도입 전 `_project`의 4-warp 구현을 뜻하며 현재 T=2/4 baseline이 아니다. `scalar1`과 `packed4`는 대조용 override다. 기본 T=1 decode는 여전히 `_project` 4 warp다. “Scalar”는 소스 계열 이름이며 실제 load는 compiler가 vectorize할 수 있다.

## Profiler 없는 시간

84개 조건 × 섞은 순서의 3개 배치 × 100회, **25,200개 sample**이다. T=1/2/4의 부분 공유 route에서 두 커널 계열 × 1/4 warp × gate/up/down/block을 측정하고, 현재 T=4 경로의 완전 공유·비공유 대조도 추가했다. 같은 stream의 CUDA Graph replay를 CUDA event로 측정했다. Cold는 매 sample 전 128 MiB buffer를 쓰는 조건이며 flush와 300,000-cycle prelude는 측정 밖이다. 표는 세 배치 중앙값의 중앙값이다.

부분 공유 route, 각 셀은 **hot / flush 후 µs**. T=1은 기본 4-warp, T=2/4는 현재 packed 1-warp다.

| T | Gate/up + SwiGLU | Down | 전체 routed expert block |
|---:|---:|---:|---:|
| 1 | 21.504 / 30.720 | 23.552 / 26.624 | 44.032 / 55.296 |
| 2 | 38.912 / 47.104 | 39.936 / 39.936 | 77.824 / 84.992 |
| 4 | 68.608 / 73.728 | 70.656 / 70.656 | 137.216 / 142.336 |

T=4 부분 공유에서 구성별 결과:

| 구성 | Gate/up hot/cold | Down hot/cold | Block hot/cold | 기존 산술과 exact |
|---|---:|---:|---:|---|
| 현재 packed 1 warp | 68.608 / 73.728 | 70.656 / 70.656 | 137.216 / 142.336 | 예 |
| 기존 커널 4 warp | 64.512 / 71.680 | 70.656 / 78.848 | 137.216 / 144.384 | 예 |
| 기존 커널 1 warp 진단 | 51.200 / 60.416 | 70.656 / 70.656 | 120.832 / 130.048 | 아니오 |
| Packed 4 warp 대조 | 70.656 / 76.800 | 78.848 / 84.992 | 150.528 / 155.648 | 예 |

Standalone 중앙값을 더해 block 시간으로 사용하지 않는다. Block의 down 입력은 앞선 gate/up 결과이고 cache 상태도 다르다. Block은 별도로 직접 측정했고 같은 설정의 production dispatcher와 출력 exact를 확인했다. 이 결과를 전체 모델의 tok/s로 확대하지 않는다.

## 현재 커널의 실제 카운터

T=4 부분 공유. NCU는 `cudaProfilerStart/Stop`으로 projection 하나만 선택했고, `--replay-mode application --cache-control none --clock-control none`으로 각 pass에서 cache 상태를 재구성했다. 시간 표와 NCU 지연은 측정 방법이 다르므로 서로 대체하지 않는다.

| 지표 | Gate/up hot | Gate/up cold | Down hot | Down cold |
|---|---:|---:|---:|---:|
| NCU 지연, µs | 67.200 | 70.976 | 68.320 | 69.376 |
| Issue active, active-cycle 기준 | 75.53% | 70.79% | 62.87% | 56.06% |
| Issue active, elapsed-cycle 기준 | 71.10% | 66.76% | 49.74% | 49.35% |
| Eligible warps / scheduler / active cycle | 2.18 | 1.78 | 1.41 | 1.04 |
| Achieved occupancy | 44.68% | 44.99% | 25.45% | 27.49% |
| DRAM throughput / peak | 43.12% | 55.47% | 1.78% | 33.66% |
| L2 throughput / peak | 42.98% | 40.40% | 13.29% | 25.27% |
| 동적 warp instruction 수 | 22,347,776 | 22,347,776 | 16,056,320 | 16,056,320 |

같은 1-warp 구성에서도 gate/up은 CTA 하나가 더 많은 일을 처리하고 실제 warp 병렬성이 높았다. 현재 gate/up의 compiler metadata는 56 registers/thread, shared 0, spills 0이고 down은 40 registers/thread, shared 0, spills 0이다. 같은 warp 수가 같은 병목을 보장하지 않는다.

## 대역폭 시간과 issue 시간의 하한

관측 대역폭 `bytes / 실제 시간`을 다시 분모로 쓰지 않았다. NCU가 제공한 peak rate로 다음의 **이상적 하한**을 계산했다.

- Issue 하한 = 동적 warp 명령 수 / (SM 수 × SM당 peak warp IPC × 측정 SM 주파수). 실제 GPU의 108 SM, SM당 4 warp instructions/cycle 상한을 사용했다.
- DRAM 하한 = 측정 DRAM bytes / NCU peak byte rate.
- L2 sector 하한 = 측정 sectors / NCU peak sector rate.

| 현재 경로 | 실제 NCU µs | Issue 이상 하한 | DRAM 대역폭 하한 | L2 sector 처리량 하한 |
|---|---:|---:|---:|---:|
| Gate/up partial hot | 67.20 | 47.75 | 28.98 | 14.97 |
| Gate/up partial cold | 70.98 | 47.36 | 39.37 | 14.79 |
| Down partial hot | 68.32 | 33.96 | 1.21 | 5.88 |
| Down partial cold | 69.38 | 34.22 | 23.35 | 8.46 |
| Gate/up fully shared hot | 61.79 | 47.27 | 0.89 | 8.95 |

하한들은 더할 수 있는 시간 구성요소가 아니다. 메모리 전송과 명령 실행은 겹칠 수 있고 latency·의존성·동기화·tail 효과 등은 이 식에 포함되지 않는다. Issue 하한과 issue 활성률은 같은 카운터를 다른 방식으로 표현한 부분이 있어 독립적인 두 증거로 세지 않는다.

Gate/up의 가장 큰 이상적 하한은 issue 쪽이지만 실제 지연과 차이가 남았다. **실제 시간이 메모리 대역폭 하한보다 길다는 사실만으로 순수 issue 포화를 확정하지 않는다.** Active-cycle issue 100%와 전체 elapsed 구간의 포화도 구분해야 한다. 특정 실행 파이프라인이 함께 포화됐는지도 확인했다. 현재 partial hot gate/up의 ALU/FMA/XU/LSU active 이용률은 약 62.69%/50.08%/57.43%/37.48%, L1 throughput은 35.79%였다. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)의 scheduler·warp-state·cache 해석을 따른다.

## Hot의 작업 집합과 명령 수 대조

부분 공유 T=4는 20개 고유 expert를 선택한다. Gate/up INT8 weight만 **40 MiB**, scale은 추가 **0.625 MiB**여서 A100 L2 40 MiB를 넘는다. 반복 실행하는 hot에서도 **45,015,296 bytes**의 DRAM traffic이 측정됐다. Hot은 반복 조건이며 모든 weight의 cache 상주를 보장하지 않는다. 완전 공유 route의 weight는 16 MiB로 줄어들어 별도 hot counter 대조를 추가했다.

완전 공유 hot에서는 DRAM traffic이 **1,387,776 bytes**, DRAM 처리량 1.45%, L2 처리량 21.05%로 줄었다. 그래도 NCU 지연은 **61.792 µs**, issue active **81.22%**, elapsed 기준 **76.54%**였다. Dynamic instruction 수는 부분 공유와 같은 22,347,776개였다. Eligible warps는 3.28, occupancy는 44.42%였다. 이 조건에서도 명령 발행 100% 포화는 관측하지 못했다. ALU/FMA/XU/LSU active 이용률은 67.56%/53.98%/61.90%/40.40%였고 long-scoreboard stall은 partial hot의 35.12%에서 8.01%로 줄었다. Stall 비율은 active warp 상태의 비율이며 wall time의 비율이 아니다.

같은 1-warp 기존 커널 진단의 gate/up 동적 명령은 **14,057,472개**로 현재의 22,347,776개보다 **37.10% 적었다**. NCU 시간은 **67.200 → 50.912 µs**로 **24.24% 감소**, profiler 없는 hot은 **68.608 → 51.200 µs**로 **25.37% 감소**했다. 이때 issue active는 63.24%, elapsed 기준 59.01%였다. DRAM 대역폭 하한은 약 28.96 µs로 거의 같았다. 명령 처리 비용을 줄일 여지가 있음을 보여주지만 load·loop·reduction 및 compiler 결과도 함께 달라 단일 원인을 고립한 대조는 아니다.

같은 packed 소스의 4-warp 대조는 동적 명령이 **26,378,240개**로 늘었고 NCU 시간도 **70.048 µs**로 증가했다. Occupancy 92.92%, issue active 83.85%까지 올라도 현재 경로보다 빨라지지 않았다. 기존 4-warp 커널은 22,511,616개 명령, 63.744 µs, issue active 79.55%였다. 이 두 4-warp 구성은 각각 별도의 대조다.

## 검증과 한계

- 모든 25,200 sample의 수·중앙값·3개 배치 coverage와 telemetry를 검산했다. 각 graph의 마지막 출력은 같은 변형 eager와 exact였다.
- 기존 4-warp, 현재 packed 1-warp, packed 4-warp는 모든 측정 조건에서 기존 산술과 exact였다. 기존 커널 1-warp 진단은 exact가 아니었다. 측정 조건 전체 최대 차이는 gate/up **0.000244140625**, down **0.00048828125**, block **0.00006103515625**였다.
- Token 0의 선택된 expert 8개와 전체 block 출력에 대해 독립 FP32 tensor reference 및 명시적 BF16 rounding을 검사했다(`atol=0.002`, `rtol=0.02`). 이것은 exact 검사를 대체하지 않는다.
- 모든 block 구성은 같은 설정의 production dispatcher 출력과 exact였다. 이는 해당 합성 fixture의 검사이며 모델 logits/state/장기 생성 또는 품질 검증이 아니다. 빠른 비동등 변형은 채택하지 않았다.
- 첫 진행 중 snapshot은 갱신 중인 timing JSON을 tar로 읽는 도중 실패했다. GPU 측정은 계속 정상 실행됐다. 회수기는 파일별 bytes snapshot 방식으로 수정했고 이후 archive를 정상 회수했다.

자료: [시간 원본](results/timing/timings.json), [시간 요약](results/timing-summary.csv), [카운터 요약](results/profile-summary.csv), [분석](results/analysis.json), 각 `.ncu-rep`/`.raw.csv`/`.raw-base.csv` 및 PTX/SASS. 재현 코드는 [scripts/expert_issue](../../scripts/expert_issue/README.md)에 있다. 모델·cache는 결과에 포함하지 않으며 Triton cache는 원격 `/tmp/infer-opt-models/expert-issue`에 두었다.

## 판정과 완료

현재 1-warp gate/up은 down보다 instruction pressure가 높고, 명령 수를 줄인 진단 변형의 이득도 컸다. 따라서 명령 처리 비용은 중요한 최적화 대상이다. 그러나 cache 작업 집합을 줄인 완전 공유 hot에서도 issue 슬롯과 개별 측정 파이프라인에 여유가 남아 **순수 issue-throughput 포화가 지배적이라는 결론은 확정하지 않았다**. 대역폭 하한이 작아도 memory latency·의존성·실행 구조의 비용은 남을 수 있다. Literal 100%만을 기계적인 판정 기준으로 삼는 것도 아니며, 여러 지표와 대조를 함께 해석한다.

주요 8개 및 완전 공유 후속 1개, **9개 NCU report**와 25,200개 sample의 검산을 완료했다. 각각 profiling/timing 출력 hash와 실제 kernel/grid/block이 일치했다. 모든 14개 block 구성은 해당 production dispatcher와 exact였다. Production kernel hash는 이전 실험 및 Git HEAD와 같으며 커널을 채택·변경하지 않았다.

최종 archive는 **1,025,582 bytes**, SHA-256 **`0ddca902e05b98e067ff02fd55220f463798b595f74dd7d90b9d6d8ccc1d901d`**이며 원격 값과 일치했다. 두 sweep의 clock reset이 성공했음을 확인한 뒤 archive를 회수·검산하고 전용 세션 `expert-issue-a100-20261111`을 종료했다. 서버 활성 세션 **0개**를 확인했다. [archive 검산](archive-validation.json), [최종 검증](validation.json), [정리 기록](cleanup.json), [소스 manifest](source-manifest.json).
