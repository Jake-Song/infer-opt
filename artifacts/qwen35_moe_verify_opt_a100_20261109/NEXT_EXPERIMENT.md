# 다음 시험: llama.cpp expert의 MTP round 비용 측정

2026-11-11 사용자 요청으로 우선순위를 변경했다. **먼저 `test-backend-ops perf`로 llama.cpp expert 연산을 독립 측정하고, 이후 실제 MTP round에서 비용을 확인한 뒤 route pack + Grouped GEMM 착수 여부를 결정한다.** 독립 측정은 [RunPod A100 결과](../llamacpp_moe_kernel_a100_20261111/README.md)로 완료했다. 이후 [native MTP 생성 측정](../llamacpp_mtp_a100_20261111/README.md)에서 off/k=1/2/3의 속도·수락률·출력 일치와 별도 CUDA trace를 수집했다. 모든 MTP 설정에서 출력이 불일치해 시간은 진단용이며, target/draft expert의 구간별 비용 분해와 Grouped GEMM 구현은 남아 있다.

독립 측정에서 부분 공유·flush 후 T=4 down은 current infer-opt/llama.cpp **55.94/40.90 µs**였지만, 전체 routed expert block은 **114.40/118.64 µs**였다. Native gate/up fusion은 CUDA Graph에서 확인했다. 해당 RunPod에서는 `ERR_NVGPUCTRPERM`으로 counter가 차단됐고, 이후 Colab counter 대조에서도 순수 issue-throughput 포화를 확정하지 않았다. 아래 첫 단계 항목은 실행 조건의 기록으로 유지한다. 두 번째 단계의 native 생성 측정은 위 보고서로 완료했으며, expert별 구간 계측은 아직 남아 있다.

## 첫 단계: 모델 없이 expert 연산 독립 측정

- 도구는 llama.cpp의 `test-backend-ops` **`perf` 모드**, 대상 연산은 `MUL_MAT_ID`다. 합성 weight·activation·expert ID를 사용하므로 전체 모델이나 MTP 포함 GGUF 다운로드 없이 시작할 수 있다.
- 현재 upstream의 기본 명령 형식: `./build/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=q8_0'`. 실제 실행에는 고정 commit의 help·테스트 케이스를 확인한다. 이 필터만으로 Qwen3.5의 정확한 shape가 자동 생성되는 것은 아니다.
- Qwen3.5-35B-A3B의 expert 256개·top-k=8·hidden 2048·intermediate 512에 맞춰 gate/up 및 down 케이스와 T=1/2/4를 추가하거나 지정한다. 토큰 간 expert 공유/비공유 route 대조와 기존 infer-opt 커널 비교를 수행한다.
- 입력 shape와 layout을 명시적으로 검증한다. 아래는 ggml의 `ne[0]`부터 쓰며 마지막 singleton 축은 생략한다. Gate/up 각각의 weight `[2048,512,256]`, activation `[2048,1,T]`(expert 간 broadcast), IDs `[8,T]`, output `[512,8,T]`; down weight `[512,2048,256]`, activation `[512,8,T]`(route별 독립), IDs `[8,T]`, output `[2048,8,T]`다. PyTorch 논리 대응은 gate/up 입력 `[T,2048]`, down 입력 `[T,8,512]`(실제 버퍼 `[T*8,512]`), IDs `[T,8]`다.
- 기본 `test_mul_mat_id`의 IDs는 `[256,T]` backing tensor에서 `[8,T]` view를 만들 수 있어 token stride가 contiguous IDs와 다르다. 비교 케이스는 이 stride를 확인하고 대응 layout을 사용한다. 모든 input/output의 shape·dtype·byte stride·view offset·broadcast 및 route 내용을 manifest에 기록하고 실행 전에 assert한다. Route는 초기화와 perf 반복 재초기화 모두에서 지정한 공유 패턴을 유지한다.
- 현재 infer-opt gate/up weight는 `[256,1024,2048]`로 결합되어 있고 SwiGLU까지 융합한다. 단일 `MUL_MAT_ID`와 비교할 때는 대응 projection만 같은 경계로 측정하고, fused gate/up·SwiGLU와 down·combine을 포함한 expert block은 별도 비교한다. Q8_0와 group128의 packing·scale 및 activation dtype 차이는 유지하고 명시한다.
- `MUL_MAT_ID` backend 실행에는 activation 양자화 등 여러 CUDA 커널이 포함될 수 있다. 연산 전체 지연과 trace에서 분리한 핵심 커널 지연을 구분하고, 반복 측정의 cache 재사용 및 cold-cache 조건을 명시한다. 단독 op의 값을 그대로 전체 expert block 또는 MTP round 비용으로 보고하지 않는다.
- 도구 확인 근거: [upstream test-backend-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp). 이 링크는 도구 확인용이며 실제 측정 commit은 실행 전에 고정한다.

## 두 번째 단계: 실제 MTP round 비용

- 대상: Qwen3.5-35B-A3B Q8_0 + native MTP, A100. llama.cpp commit·모델 revision·GGUF hash·빌드 옵션을 고정하고, MTP tensor 포함 여부와 실제 GPU offload/dispatch를 확인한다. 기존 MTP-off GGUF가 그대로 사용 가능하다고 가정하지 않는다.
- 수학·한국어 입력, batch 1, MTP off/k=1/k=3을 우선 비교한다. Target verification T=2/4와 draft의 T=1 expert 호출을 구분한다. Context·출력 길이·warmup·반복 수를 맞추고 컴파일·prefill을 decode 측정에서 제외한다.
- 원래 실행 경로의 trace에서 **target expert와 draft expert의 GPU ms/round·호출 수**를 각각 집계한다. Gate/up·SwiGLU·down·combine과 activation Q8_1 양자화 등 보조 비용도 분류하고, fusion된 구간은 억지로 분할하지 않는다. Target expert의 T=1/2/4 비용과 증가율을 별도로 비교한다.
- 전체 host ms/round, target verify, draft, lm_head, accept/commit 및 동기화 비용을 구분해 expert가 round에서 차지하는 비중을 확인한다. Profiler 없는 wall time과 trace의 GPU kernel duration 합은 별도 결과로 보고한다.
- Token/round·acceptance·ms/token 및 같은 llama.cpp 설정 내 MTP-off 대비 생성 token 일치 여부도 기록한다. 수락률 차이를 expert 커널 성능 차이로 해석하지 않는다.
- infer-opt 비교는 가능한 한 같은 GPU·입력 token·context·측정 경계에서 다시 측정한다. Q8_0/Q8_1 W8A8과 현재 group128 W8A16은 산술이 다르므로 엔진 간 수치 동등성을 전제하지 않는다. 기존 A100 80GB/40GB 결과와 새 측정의 장비 차이도 표시한다.
- 이 결과로 작은 T에서 llama.cpp expert 비용이 어떻게 증가하는지, 전체 round 차이가 expert에서 오는지 먼저 판단한다. 이후 W8A8 커널 비교 또는 route pack + Grouped GEMM 중 다음 구현 실험을 선택한다.

기존 [llama.cpp 소스 분석](../llamacpp_qwen35_moe_mtp_source_review_20261109/README.md)은 고정 commit의 정적 분석이며, A100 Q8_0+MTP의 round 비용을 측정한 결과는 아니다. 실제 선택 커널은 이번 실행에서 확인한다.

## 후속 후보: expert별 route pack + Grouped GEMM

이전 다음 시험의 범위는 아래에 유지한다. 먼저 위 비용 측정을 완료한다.

- 대상: Qwen3.5-35B-A3B INT8 group128, parallel target verification T=2/4, top-k=8, A100.
- GPU에서 route를 expert별로 정렬/묶고, count·offset·원래 route index를 만든다. 같은 expert의 activation을 연속 버퍼로 pack한다.
- Gate/up와 down에 variable-M grouped GEMM을 적용하고, 유효한 expert/tile만 처리하는 persistent scheduler를 검토한다. Singleton GEMV + 여러 토큰 GEMM의 혼합도 비교한다.
- 결과를 원래 token/slot 순서로 복원하고, 기존 BF16 expert 가중합 순서를 보존한다.
- 비교 시간에는 pack·offset 생성·양자화/역양자화·복원·combine 비용을 모두 포함한다. Expert 단독과 target body 전체를 분리한다.
- 실제 route와 fully-shared/disjoint 대조를 모두 사용한다. T=1..5·동적 routes·compile/CUDA Graph를 확인한다.
- Tensor Core 입력 dtype·scale 적용·reduction 순서가 바뀔 경우 hidden/logits·active state·greedy ID 및 품질을 따로 검증한다. 토큰 ID 몇 개의 일치만으로 동등성을 확정하지 않는다.
- 현재 1-warp pair-load 및 기존 4-warp GEMV를 모두 기준으로 비교한다. 이전 16행 padding/gather 초안의 실패를 Grouped GEMM 전체의 결론으로 취급하지 않는다.

소스 참고: [llama.cpp 분석](../llamacpp_qwen35_moe_mtp_source_review_20261109/README.md). A100 Q8_0의 작은 T는 W8A8 MMVQ 계열이며, 독립 캡처에서 T=1의 일반 matvec 및 T=2/4의 MoE matvec을 확인했다. Expert별 pack/양자화/scatter와 MMQ는 별도 큰 배치 경로다. Grouped GEMM 구현 실험은 아직 시작하지 않았다.
