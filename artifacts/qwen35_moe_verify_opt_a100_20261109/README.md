# A100 target verification kernel optimization — 2026-11-09

사용자 요청에 따라 여기서 실험을 정리했다. Expert별 route pack과 persistent Grouped GEMM은 다음 시험으로 남겼다.

## 채택한 변경

`infer_opt/moe_kernels.py`에 인접한 BF16/FP16 활성값 2개를 uint32로, INT8 가중치 2개를 uint16으로 읽는 커널을 추가했다. FP32 누산·BF16 rounding·expert 합산 순서를 보존하고 한 CTA에 1 warp를 사용한다.

기본 선택 범위는 **A100, BF16, T=2/4, hidden=2048, expert intermediate=512, top-k=8, group=128**이다. 일반 decode T=1은 기존 경로다. 명시적 `LAUNCH_CONFIGS` override도 기존 경로를 사용한다. `USE_PACKED_VERIFY=False` 또는 `PACKED_PAIRS_CONFIGS[T]=None`으로 reference를 선택할 수 있다. 설정은 graph capture 전에 바꿔야 한다.

## 실제 target 검증 body 결과

장비는 **A100-SXM4-40GB**, 39.4935 GiB visible, PyTorch 2.11.0+cu130, Triton 3.6.0이다. A100 80GB 고메모리 할당 두 번이 503으로 실패한 뒤 일반 A100 할당을 사용했다. 이전 80GB 실험 시간과 직접 비교하지 않는다.

같은 compiled fullgraph를 reference/후보 커널로 각각 수동 CUDA Graph에 capture했다. 고정 상태를 매회 복원하고 순서를 번갈아 비교했다. Cache 256, max_autotune 및 coordinate_descent, 128 MiB L2 flush. MTP 모듈은 로딩 후 해제하고 target body만 측정했다. 이 결과는 전체 MTP round 또는 생성 tok/s 가속이 아니다.

최종 코드의 자동 선택 경로, 각 100회 중앙값:

| 입력 | 검증 토큰 | 기존 GPU ms | 후보 GPU ms | 시간 감소 |
|---|---:|---:|---:|---:|
| math | 2 | 7.115 | 7.016 | 1.38% |
| math | 4 | 9.478 | 9.225 | 2.67% |

두 측정점 모두 hidden·logits·target ID 및 **캐시 텐서 141/141개가 정확히 일치**했다. 사용자 정리 요청에 따라 이 후속 재확인은 수학 두 점까지 완료한 뒤 중단했다. 한국어 최종 자동 선택 재확인은 미완료다. `final-exit.json`의 성공을 주장하지 않는다.

앞선 동일 연산의 명시적 후보 선택 실험은 수학·한국어 × T=2/4 × 1/2 warp 총 8개 경우를 각 30회 완료했고, 모두 hidden·logits 및 141개 상태가 정확히 일치했다. 채택한 1-warp의 결과:

| 입력 | 검증 토큰 | 기존 GPU ms | 후보 GPU ms | 시간 감소 |
|---|---:|---:|---:|---:|
| math | 2 | 7.357 | 7.275 | 1.11% |
| math | 4 | 9.590 | 9.335 | 2.66% |
| korean | 2 | 7.011 | 6.925 | 1.22% |
| korean | 4 | 9.373 | 9.165 | 2.22% |

## 검증

- 최종 GPU 검사 **37 passed, 24 deselected**: 새 커널 BF16/FP16·T=2/3/4/5·두 실제 크기·1/2 warp·동적 route/graph 및 기존 selected-expert 검사. [원본 로그](results/final-tests.stdout).
- 최종 expert micro는 겹침/부분 겹침/비겹침 6개 경우 × 3개 후보를 각 100회 측정했다. 저장된 모든 후보 출력이 기존 커널과 정확히 일치했다. T=1..5의 compile·CUDA Graph 동적 route 검사도 통과했다.
- 로컬 CPU MoE 검사 20 passed. GPU 항목은 로컬에서 skip했고 원격에서 위와 같이 별도 확인했다.
- 체크포인트는 `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`. 34개 파일의 해시를 기록하고 LFS payload hash를 검증했다. 가중치는 원격 `/tmp/infer-opt-models`에 두었다.
- [validation.json](validation.json)에 sample 중앙값·source hash·동등성 검산을 기록했다. 현재 네 코드 파일은 원격 최종 실행 source hash와 일치한다.
- HF 정확도와 256토큰 전체 MTP 생성은 이번 최적화 이후 다시 측정하지 않았다. 기존 compiled parallel MoE의 일반 decode 대비 불일치 문제가 해결됐다는 뜻은 아니다.

## 제외한 후보

| 후보 | 결과 / 제외 이유 |
|---|---|
| Expert 중복을 찾아 여러 토큰에 weight 공유 | 반복 중복 검사와 비활성 행 계산을 포함한 초안. 실제 측정에서는 느림 |
| 여러 출력 행을 한 CTA로 묶기 | 측정한 타일들이 기존보다 느림 |
| Grouped Tensor Core 초안 | expert별 동적 token gather와 16행 padding을 사용. 느리고 BF16 dequant rounding 차이 발생. Route pack + persistent Grouped GEMM은 아직 미실행 |
| K block/warp 단순 튜닝 | target body 4~8% 개선이지만 hidden/cache 불일치. 채택하지 않음 |
| 32비트 expert offset | 기존 launch에서 수치 일치했지만 느림 |
| 첫 reduction 순서 고정 / unpacked pair | 일부 수치 차이 또는 성능 이득 없음 |

실패한 후보 소스와 sample도 `results/` 및 `moe-kernels-*.py`에 보존했다. 채택하지 않은 구현은 실제 모델 코드에서 제거했다. 원인에 대한 occupancy/HBM counter 측정은 수행하지 않았다.

## 다음 시험

[NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md)에 route pack + Grouped GEMM의 작업 범위와 비교 조건을 남겼다. 새 GPU 실험은 시작하지 않았다.

## 정리 상태

사용자 정리 요청으로 후속 controller를 중단하고 결과를 회수한 뒤 전용 A100 세션을 종료했다. 서버의 active session이 없음을 확인했다. [cleanup.json](cleanup.json), [중단 기록](results/wrap-status.json).
