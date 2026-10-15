> 주의: 실행 로그에서 모델 기본값이 do_sample=True를 적용한 것을 확인했다. 이 실행을 greedy 검증 결과로 사용하지 않는다. 수정 실행은 ../a100-20261015-generate-defaults-greedy/ 참조.

# Transformers generate 기본 옵션 재실행

- NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, Transformers 4.57.3
- Qwen3-14B BF16 / SDPA / 입력 128, 출력 128 / 기본 CompileConfig
- generate가 prefill과 decode 분기 및 cache를 관리한다. 사용자 compile 옵션 없음.
- compile 그래프 관측: True
- 모든 단계 logits와 최종 모든 레이어 유효 K/V를 atol=rtol=0.02로 비교.
- 토큰 경로가 다르면 이후 tensor 차이는 같은 입력 비교가 아니므로 해석에 제한이 있다.

| 비교 | 생성 토큰 일치 | logits 실패/128 | 최종 KV 실패/전체 |
|---|---|---|---|
| eager_static / eager_dynamic | True | 128/128 | 66/80 |
| compiled_static_defaults / eager_static | True | 127/128 | 68/80 |
| compiled_static_defaults / eager_dynamic | True | 128/128 | 70/80 |

정확도 재실행만 수행했다. compile 및 검증 시간을 성능 결과로 사용하지 않는다.
