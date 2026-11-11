# llama.cpp MoE expert 독립 커널 측정 — 2026-11-11

**Down projection은 llama.cpp가 더 빠르지만, 부분 공유 route의 전체 routed expert block은 현재 infer-opt보다 빨라지지 않았다.** Native weight layout에서 gate/up fusion이 실제로 켜진 것도 확인했다. GPU performance counter 접근은 차단되어 instruction-issue bound 가설은 확정하지 못했다.

## 장비와 범위

- RunPod **A100-SXM4-80GB**, visible 79.250732 GiB, GPU 1개. 일반 Colab A100 할당 두 번이 503으로 실패한 뒤 사용자 요청대로 RunPod를 사용했다. Colab high-mem 요청은 사용자 지시에 따라 취소했고 활성 세션이 없음을 확인했다.
- CUDA toolkit 13.0, driver 595.91.07, PyTorch **2.9.1+cu130**, Triton **3.5.1**, Python 3.12.3. 이전 A100 실험의 PyTorch 2.11/Triton 3.6과 환경이 다르다. 비교용 infer-opt 커널을 **이번 같은 Pod에서 다시 측정**했다.
- llama.cpp commit `a0ad695cb212db3850f163121ddc687513448d42`, Release, SM80, CUDA Graph ON. CUDA 연산 커널과 dispatch는 수정하지 않았다. 테스트 케이스, 측정용 stream/graph accessor 및 graph dump를 추가한 private benchmark checkout이다.
- **Qwen3.5-35B-A3B expert shape를 사용하는 합성 tensor 실험**이다. 모델·GGUF·MTP head를 다운로드하거나 실행하지 않았다. Router, top-k 생성, shared expert, attention, lm_head, MTP round 비용은 포함하지 않는다.
- llama.cpp는 native **Q8_0/Q8_1 W8A8**, infer-opt는 **INT8 group128 W8A16**이다. 동일한 양자화 산술이나 모델 품질을 전제하지 않는다.

## 입력과 측정 경계

Experts 256, top-k 8, hidden 2048, expert intermediate 512, T=1/2/4. 각 token 안에서 expert ID는 중복되지 않는다. 토큰 사이의 route는 8개 완전 공유, 4개 공유, 완전 비공유로 나누고 slot 순서를 바꿨다.

| Tensor | ggml `ne[0]`부터 표기 | PyTorch 논리 대응 |
| --- | --- | --- |
| Gate/up 입력 | `[2048,1,T]` | `[T,2048]`, expert 간 공유 |
| Native gate/up weight 각각 | `[2048,512,256]` | `[256,512,2048]` |
| Expert IDs | `[8,T]`, int32, token stride 32 bytes | `[T,8]`, contiguous int32 |
| Down 입력 | `[512,8,T]` | `[T,8,512]`, 실제 버퍼 `[T*8,512]` |
| Down weight | `[512,2048,256]` | `[256,2048,512]` |
| Down 출력 | `[2048,8,T]` | `[T,8,2048]` |

실제 shape·dtype·byte stride·view offset·ID를 원시 `INFER_LAYOUT`에 기록했다. IDs는 두 구현 모두 contiguous로 맞춘 제어 조건이며, llama.cpp production top-k view의 모든 layout을 재현했다는 뜻은 아니다. 양자화 packing과 activation/output dtype은 각 구현의 native 방식을 유지했다.

합성 source 값은 동일한 정수 hash에서 생성했다. C++/Python 값 1,028개와 native/view weight 좌표 96개를 독립 검산했다. Gate/up standalone은 SwiGLU까지, down standalone은 독립적인 route별 activation으로 측정했다. 전체 block은 gate/up·SwiGLU·down·expert weight 곱·합을 포함한다. Block 안의 down 입력은 앞선 결과이므로 standalone 구간 중앙값을 더해 block 시간으로 사용하지 않는다.

각 조건을 warmup한 뒤 backend가 캡처한 CUDA Graph를 직접 재생하고, 같은 stream의 CUDA event로 **100회** 측정했다. Cold 표는 매 실행 전 128 MiB flush buffer를 쓰는 조건이다. Flush와 host enqueue 간격을 가리는 300,000-cycle prelude는 측정 구간 밖이다. 기본 `test-backend-ops perf`의 반복 평균은 별도로 3회 보존했다. 그 pinned 버전의 CSV 출력에는 시간 열이 없어 console 출력도 수집했다.

GPU 클럭은 고정하지 않았고 케이스는 고정 순서로 실행했다. 동일 경로인 T=1의 두 baseline 라벨에도 일부 시간 차이가 있어, 수 퍼센트 차이를 안정적인 개선으로 확정하지 않는다. 원시 sample과 p10/p90을 함께 보존한다.

## 전체 routed expert block — flush 후 중앙값

단위는 **µs/block**. Current는 T=1에서 기존 경로, T=2/4에서 채택된 1-warp pair-load 경로다. Reference는 기존 4-warp 경로다.

| Route | T | Reference | Current | llama.cpp native |
| --- | ---: | ---: | ---: | ---: |
| 완전 공유 | 1 | 50.14 | 45.92 | 57.23 |
| 완전 공유 | 2 | 66.82 | 64.37 | 70.66 |
| 완전 공유 | 4 | 113.06 | 109.31 | 107.09 |
| 부분 공유 | 1 | 45.79 | 45.89 | 57.39 |
| 부분 공유 | 2 | 70.94 | 68.80 | 76.86 |
| 부분 공유 | 4 | 119.02 | 114.40 | 118.64 |
| 비공유 | 1 | 45.76 | 45.63 | 57.31 |
| 비공유 | 2 | 75.55 | 73.34 | 81.66 |
| 비공유 | 4 | 127.74 | 121.34 | 128.77 |

부분 공유에서 llama.cpp의 T=4/T=1은 **2.07배**, current는 **2.49배**다. 그러나 llama.cpp의 T=1이 더 느려 T=4 절대 시간의 우위로 이어지지 않았다. 완전 공유 T=4의 cold 이득은 약 2.1%다. 이 결과를 일반적인 전체 block 가속으로 확대하지 않는다.

## 구간별 결과 — 부분 공유, flush 후

| T | 구간 | Current µs | llama.cpp µs |
| ---: | --- | ---: | ---: |
| 1 | Gate/up + SwiGLU | 25.73 | 32.77 |
| 1 | Down | 22.56 | 19.87 |
| 2 | Gate/up + SwiGLU | 39.65 | 44.42 |
| 2 | Down | 31.71 | 26.21 |
| 4 | Gate/up + SwiGLU | 60.56 | 69.39 |
| 4 | Down | 55.94 | 40.90 |

T=4 down은 current 대비 **1.37배**의 처리 속도지만 gate/up은 느리다. 큰 구간별 차이를 단일 산술 원인으로 단정하지 않는다.

Hot T=4 전체 block의 current/llama.cpp는 완전 공유 **106.91/97.01 µs**, 부분 공유 **109.41/108.46**, 비공유 **112.42/117.09**다. 이득의 방향과 크기는 route·cache 조건에 따라 달라진다.

## Weight layout과 실제 dispatch 확인

처음에는 infer-opt의 `[256,1024,2048]` 결합 gate/up 저장 방식에 대응하는 Q8_0 view를 사용했다. 이 구성은 llama.cpp의 fusion을 유지하지 못했다. Native gate/up 버퍼를 분리해 같은 source weight 값을 채운 대조를 추가했고, 성능 결론은 native 결과를 사용한다. 초기 view 측정도 보존했다.

- Native gate/up 캡처 9개 모두 **activation Q8_1 양자화 + fused matvec 1개**, 총 2개 CUDA 커널이다.
- T=1은 `mul_mat_vec_q<...,true,...>`, T=2/4는 `mul_mat_vec_q_moe<...,true>`다. 템플릿의 `true`는 fusion 활성화다. 실제 grid/block은 DOT에 있다.
- T=4 부분 공유 view 대조는 양자화 2개·unfused matvec 2개·SwiGLU 1개, 총 **5개** 커널이다.
- 부분 공유 gate/up의 view/native 시간은 T=1 **42.82/32.77**, T=2 **49.54/44.42**, T=4 **69.15/69.39 µs**였다. Fusion 활성화만으로 모든 T에서 가속이 생기지는 않았다.

확인 자료: [native T=4 gate/up graph](results/native-graphs/t4-gate_up-partial-native-0.dot), [view 대조 graph](results/view-graphs/t4-gate_up-partial-view-0.dot), [파싱한 kernel 목록](results/analysis.json).

## 검증과 해석의 한계

- View/native 각각 **36/36 CUDA correctness cases**가 CPU reference 비교를 통과했다. Q8_0 테스트의 NMSE 한계는 기존 `5e-4`를 사용했다.
- Native 7,200개, view control 7,200개, infer-opt 10,800개 timing sample을 회수했다. 각 측정 조건의 마지막 graph replay 출력은 같은 엔진의 warm backend/eager 결과와 정확히 일치했다. 두 엔진 사이의 exact equality 검사는 아니다.
- Infer-opt reference/packed 출력 hash 비교 **54/54 일치**. 마지막 replay 및 각 cache 조건의 검사이며, 모델 logits·state·장기 생성 검증이 아니다.
- Nsight Compute는 **`ERR_NVGPUCTRPERM`**으로 실패했다. 해당 실행의 시간 sample은 정식 성능 표에서 제외했다. Instruction issue·DRAM/L2·stall counter는 확보하지 못해 **issue-bound 가설은 미확정**이다. DOT는 실제 kernel dispatch/fusion 증거이며 하드웨어 counter를 대체하지 않는다.
- 이 오류는 해당 Pod에서 counter 접근 권한이 없었다는 증거다. RunPod 전체의 차단 정책을 확인한 것은 아니다. Root 계정으로 실행했지만 host driver의 `RmProfilingAdminOnly`와 container의 Linux capability는 수집하지 않아, driver 설정과 capability 중 실제 원인은 구분하지 못했다. [NVIDIA 설명](https://developer.nvidia.com/ERR_NVGPUCTRPERM)에 따르면 container에서는 host의 접근 허용 또는 적절한 capability가 필요할 수 있다.
- 이 실험으로 native MTP의 ms/round, 생성 tok/s, acceptance, 품질 또는 W8A8의 독립적인 기여도를 확정할 수 없다. 기본 perf, 단일 projection, 융합 구간, block의 측정 경계를 구분한다.

다음 단계는 여기서 빠른 down 경로와 느린 gate/up 경로를 구분해 실제 llama.cpp MTP round에서 차지하는 비중을 재는 것이다. Grouped GEMM이나 W8A8 이식을 자동 채택하지 않았다.

## 재현 자료와 정리

- 실행기: [scripts/llamacpp_moe](../../scripts/llamacpp_moe/README.md). 고정 source patch, fixture, standalone/native follow-up, 분석·검산 코드를 보존했다.
- [analysis.json](results/analysis.json), [paired-timings.csv](results/paired-timings.csv): 검산한 중앙값·배율·sample·kernel 목록.
- [원시 native 결과](results/native-isolated-graph-perf.stdout), [infer-opt 결과](results/infer-opt.json), [기존 view 결과](results/isolated-graph-perf.stdout), `standard-console-*.stdout`: 실제 sample 및 표준 perf 평균.
- `source-v1.tar.gz`, source manifest, native source patch: 첫 실행과 후속 layout 대조를 구분한다. 초기 실행 33개 source hash, 후속 실행 34개 source hash를 모두 검산했다.
- [최종 회수 파일](final-results.tar.gz)의 SHA-256은 `4ec90c6b9c211d9037c26d09b3553c927b36e066996c40faf069eebd3e4214f7`이며 원격 파일과 일치한다.
- 결과 회수 후 전용 RunPod를 **삭제**했고 Pod 목록이 비었음을 확인했다. Colab 활성 세션도 0개다. [cleanup.json](cleanup.json).
