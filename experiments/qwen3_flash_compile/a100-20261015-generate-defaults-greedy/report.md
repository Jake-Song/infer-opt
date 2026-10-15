# Transformers generate 기본 옵션 재실행

- NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, Transformers 4.57.3
- Qwen3-14B BF16 / SDPA / 입력 128, 출력 128 / 기본 CompileConfig
- generate가 prefill과 decode 분기 및 cache를 관리한다. 사용자 compile 옵션 없음.
- compile 그래프 관측: True
- **현재 판정: 생성 결과 기준 통과.** 세 경로의 생성 token ID 128개가 모두 일치했다.
- 사용자 요청에 따라 생성 token ID 전체 일치를 통과 기준으로 사용한다.
- logits/KV의 atol=rtol=0.02 비교는 진단 정보로 보존하며 통과 여부를 막지 않는다.
- 토큰 경로가 다르면 이후 tensor 차이는 같은 입력 비교가 아니므로 해석에 제한이 있다.

| 비교 | 생성 토큰 일치 | logits 실패/128 | 최종 KV 실패/전체 |
|---|---|---|---|
| eager_static / eager_dynamic | True | 128/128 | 66/80 |
| compiled_static_defaults / eager_static | True | 127/128 | 67/80 |
| compiled_static_defaults / eager_dynamic | True | 128/128 | 70/80 |

정확도 재실행만 수행했다. compile 및 검증 시간을 성능 결과로 사용하지 않는다.

최종 적용된 세 설정의 do_sample=False를 확인했다. 생성 토큰 128개는 세 경로에서 모두 일치했다.
첫 비교의 logits는 128/128 실패, 같은 StaticCache에서 compile만 바꾼 비교는 prefill을 제외한 decode 127/127 실패다.
활성 Colab 세션 없음과 stop exit code 0을 lifecycle.json에서 확인했다.

생성 결과 기준 재판정은 `generation-evaluation.json`에 저장했다. 원래 실행 시점의
`execution.json`과 `lifecycle.json`은 이전 tensor 기준 판정 기록으로 보존했다. GPU 재실행은 하지 않았다.
