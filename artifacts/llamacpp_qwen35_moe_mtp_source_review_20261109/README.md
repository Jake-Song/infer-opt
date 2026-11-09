# llama.cpp Qwen3.5 MoE + Q8_0 + MTP source review

기준: 2026-11-09 upstream commit `a0ad695cb212db3850f163121ddc687513448d42`. Context7와 공식 저장소를 확인하고 해당 commit의 소스를 내려받아 정적으로 읽었다. 새 GPU 실험·모델 다운로드는 하지 않았다.

## 핵심

A100의 Q8_0 MoE에서 T=2/4는 기본적으로 Grouped GEMM이 아니라 dedicated multi-token MMVQ 경로를 사용한다. Q8_0는 현재 infer-opt의 W8A16 group128과 다른 산술이다. llama.cpp는 activation을 GPU에서 Q8_1으로 양자화하고 INT8 dot product를 사용한다.

## 작은 배치: T <= 8

- Dispatch: [ggml_cuda_mul_mat_id](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/ggml-cuda.cu#L2351). MMVQ를 MMQ보다 먼저 검사한다.
- [MMVQ 최대 8](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmvq.cuh#L3). A100이 선택하는 Turing+ type table에서 Q8_0는 default 값 8이다.
- [mul_mat_vec_q_moe](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmvq.cu#L883): block=(32,T), token마다 독립 warp 하나. Grid는 출력 행 타일 × top-k slot이다. 해당 token/slot의 expert를 ids로 선택한다.
- [launch](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmvq.cu#L1080): 출력 2행/warp를 처리한다. Shared-memory CTA reduction 대신 warp reduction만 수행한다.
- 같은 expert를 쓰는 토큰을 이 경로에서 명시적으로 deduplicate/pack하여 weight tile을 공동 사용하지 않는다. Warp들은 독립이며 cache를 통한 재사용 가능성과 구분해야 한다.
- Gate/up와 SwiGLU 및 조건부 shared expert 융합을 지원한다. infer-opt도 gate/up·SwiGLU는 이미 융합하므로 이것만으로 차이를 설명할 수 없다.

## 양자화 산술

- [Q8_0 block](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-common.h#L251): 32개 INT8 weight + FP16 scale.
- [activation conversion](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmvq.cu#L1551): F32 activation을 Q8_1으로 변환한다.
- [Q8 dot](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/vecdotq.cuh#L246): DP4A로 INT8 곱·INT32 누산을 하고 weight/activation scale을 곱해 F32 값을 만든다.
- 현재 infer-opt는 INT8 group128 weight와 BF16 activation을 사용하며 FP32 곱·누산을 수행한다. 이번 pair-load 최적화는 memory load를 묶었고 W8A8 연산으로 바꾼 것은 아니다.
- W8A8을 도입하면 activation 양자화가 추가되므로 기존 수치 동등성·품질 결과를 재사용할 수 없다.

## 큰 배치: expert별 compact MMQ

[mmq.cu](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmq.cu#L312), [mmid.cu](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmid.cu#L35):

1. GPU helper가 expert마다 선택된 token/slot을 찾아 연속 compact row를 정한다.
2. ids_src1/ids_dst 및 expert·col_low·col_high의 tile_spec을 만든다.
3. Gate/up의 broadcast 입력은 token당 한 번 양자화하고 expert별 compact slot으로 scatter한다. 순수 양자화의 8회 중복을 피하지만 buffer에는 slot별 복제본을 쓴다.
4. MMQ가 같은 expert의 weight tile과 여러 activation column을 계산한다. Ampere의 Q8 MMA 경로에는 INT8 Tensor Core 연산이 있다.
5. ids_dst로 원래 출력 위치에 write-back한다. 별도 전체 unpack 커널이 항상 필요한 구조가 아니다.

[MMQ scheduling](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/ggml/src/ggml-cuda/mmq.cuh#L1004)은 stream-K 분할을 지원한다. Fusion/설정에 따라 일반 tile launch도 사용하므로 모든 경우가 persistent/stream-K라고 단정하면 안 된다. Column tile을 안쪽 grid 방향으로 두어 동시에 실행되는 block이 weight tile을 L2에서 공유하도록 배치한다.

## MTP driver

- [qwen35moe graph_mtp](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/src/models/qwen35moe.cpp#L532): 단일 MTP block, token embedding과 target hidden 각각 norm -> concat -> projection -> full attention -> MoE+shared expert -> head norm/lm_head.
- [draft-mtp driver](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/common/speculative.cpp#L1528): Qwen은 단일 학습 head를 autoregressive하게 반복해 draft를 만든다. Target와 같은 model 객체로 draft context를 생성할 수 있다.
- [server batch construction](https://github.com/ggml-org/llama.cpp/blob/a0ad695cb212db3850f163121ddc687513448d42/tools/server/server-context.cpp#L633): 마지막 token과 drafted tokens를 하나의 target 검증 batch에 넣는다. Draft 생성의 순차성과 target batch 검증을 구분해야 한다.
- Accept한 prefix에 맞춰 cache를 자르고, recurrent snapshot 범위를 넘으면 checkpoint 복원 경로를 사용한다. MTP pending hidden도 수락 위치로 선택한다.
- CUDA expert 연산은 graph-compatible fast path를 제공하지만, target/draft context 호출과 accept는 host driver가 조정한다. Round 전체가 단일 CUDA graph라는 뜻은 아니다.

## 기존 벤치마크와의 관계

우리의 2026-11-05 llama.cpp G4 Q8_0 결과는 commit `c6a0ff11a0fe16ac1a2eb569190934593a2b3670`, MTP off였다. 당시 로그의 n_layer/n_layer_all은 모두 40이었다. 이를 이번 native MTP 조합의 실행·속도 증거로 쓰지 않는다. MTP를 실행하려면 NextN/MTP tensor가 있는 GGUF 또는 적합한 sidecar가 필요하다.

이 문서는 소스의 dispatch와 연산을 확인한 결과다. A100에서 llama.cpp Q8_0+MTP를 새로 실행하거나 infer-opt와 동일 입력으로 성능 비교한 결과가 아니다.
