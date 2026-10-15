# Transformers 4.57.3: compile 경로와 허용 오차

실험에서 사용한 **v4.57.3**의 공식 소스를 확인했다.
확인한 tag commit은 `47b0e478f324b54f177ea7998a0791870fdd0324`다.
로컬 checkout의 현재 개발 버전과 혼동하지 않도록 tag의 파일을 직접 읽었다.

## 공식 generate의 실행 방식

CUDA와 compilable cache 등 조건을 만족하고 `disable_compile`이 꺼져 있으면
`generate(cache_implementation="static")`가 자동 compile 경로를 사용한다.
일반 greedy/sampling 경로의 첫 prefill은 eager이고, 이후 decode가 compiled call을 사용한다.
Static cache에 필요한 mask와 cache position 준비도 generate가 관리한다.

기본 CompileConfig는 다음과 같다.

```python
CompileConfig(
    backend="inductor",
    mode="reduce-overhead",
    fullgraph=False,
    dynamic=None,
    options=None,
)
```

`get_compiled_call()`은 이 설정을 `torch.compile(self.__call__, ...)`에 전달한다.
BF16 중간 cast 보존 옵션을 기본으로 지정하지 않는다.
원래 실험은 prefill/decode 모두 `model.forward`를 fullgraph=True, dynamic=False로
compile하고 CUDA Graph를 비활성화했으므로 공식 기본 generate 경로와 동일한 설정이 아니다.

근거:
[자동 compile 조건과 prefill/decode 분기](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/utils.py#L2136-L2175),
[실제 생성 루프](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/utils.py#L2756-L2790),
[CompileConfig 기본값](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/configuration_utils.py#L1430-L1476),
[compiled call 구현](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_utils.py#L5804-L5824).

## 공통 허용 오차 하나가 있는 것은 아니다

일반 `generate()`는 eager reference를 따로 실행해 오차를 계산하거나 초과 시 중단하지 않는다.
허용 오차는 테스트 종류·출력 종류·dtype에 따라 정해진다.

| 테스트 | 실제 판정 |
|---|---|
| 공통 compile 생성 테스트 | 생성 token ID가 전부 같으면 통과한다. 이 경우 score/logits/KV를 별도로 비교하지 않는다. |
| 위 테스트에서 token ID가 다른 경우 | 첫 불일치 위치의 score 벡터에 `atol=rtol=1e-5`로 allclose를 적용한다. |
| 공통 BF16 attention 대조 테스트 | 마지막 hidden state 등 해당 출력 tensor에 `atol=rtol=0.04`를 적용한다. 모든 vocabulary logits나 전체 KV의 공통 규칙은 아니다. |

`0.04`는 단순히 전체 상대오차 4%를 뜻하지 않는다. 원소별로
`abs(actual-reference) <= 0.04 + 0.04*abs(reference)`를 검사한다.
Relative L2가 특정 퍼센트 이하면 모든 테스트를 통과한다는 규정은 확인되지 않았다.

공통 compile 생성 테스트는 작은 테스트 모델로 max_new_tokens=5를 사용하고,
fullgraph=True/dynamic=False를 별도로 지정한다. 수치 안정화를 위해 norm epsilon을
1.0으로 바꾸는 helper도 호출한다. 실제 14B pretrained 모델의 BF16 128-token 검증과
동일한 실험이 아니므로 결과를 그대로 일반화하지 않는다.

근거:
[compile 생성 테스트](https://github.com/huggingface/transformers/blob/v4.57.3/tests/generation/test_utils.py#L1620-L1740),
[생성 결과 비교 함수](https://github.com/huggingface/transformers/blob/v4.57.3/tests/generation/test_utils.py#L5030-L5067),
[BF16 attention 대조](https://github.com/huggingface/transformers/blob/v4.57.3/tests/test_modeling_common.py#L3312-L3444),
[테스트용 norm 안정화](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/testing_utils.py#L1652-L1694).

## 이번 실험의 실패 판정 해석

모든 단계의 전체 vocabulary logits와 checkpoint의 모든 KV에 `atol=rtol=0.02`를
적용한 것은 **이 벤치마크가 추가한 검증 정책**이다. Transformers의 공식 공통 기준이 아니다.
숫자만 비교해 0.02가 1e-5보다 느슨하다고 판단할 수 없다. token이 같으면 score를 검사하지
않는 테스트와 모든 tensor 원소를 항상 검사하는 검증은 검사 범위가 다르다.

이번 fixed-token replay의 단계별 argmax 일치만으로 공식 free-generation 테스트를
통과했다고 주장하지 않는다. 공식 방식의 판정에는 실제 `generate()`의 생성 sequence를
비교해야 한다. API 실행 성공, 생성 결과 일치, 내부 tensor 수치 동등성은 각각 구분해서 기록한다.
