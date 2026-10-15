# A100 40GB에서 확인된 결과

NVIDIA A100-SXM4-40GB, 39.49 GiB, PyTorch 2.11.0+cu130,
Transformers 4.57.3, 공식 Qwen3-14B BF16을 사용했다.
모델 구조를 직접 재구현하거나 가중치를 바꾸지 않았다.
동일한 입력 128토큰과 저장된 토큰 127개를 replay했다.

## BF16 cast 가설은 첫 레이어에서 확인됐지만 전체 오차를 해결하지 못했다

eager StaticCache를 reference로 한 첫 prefill의 첫 레이어 K:

| 설정 | 최대 절대오차 | 허용 오차 초과 원소 |
|---|---:|---:|
| 기본 compile | 0.5 | 7 |
| cast 보존 compile | **0** | **0** |

V도 기본 compile 최대 오차 0.001953125에서 cast 보존 후 0이 됐다.
따라서 로컬에서 찾은 BF16 수치 경로 변화는 GPU의 첫 레이어에서도 확인됐다.
하지만 cast를 보존한 전체 모델도 logits 128/128, KV 384/480개 검사가
eager StaticCache 대비 기준을 넘었다. **이 옵션으로 전체 모델 문제가 해결됐다고 볼 수 없다.**

## compile을 꺼도 cache/backend 변경에 따른 차이가 있다

eager DynamicCache와 eager StaticCache의 비교도 logits 128/128,
KV 396/480개 검사가 기준을 넘었다. 첫 레이어 K/V는 오차 0이지만
후속 레이어에서 차이가 나타났다. Profiler는 다음 선택을 확인했다.

| 설정 | Prefill | Decode |
|---|---|---|
| eager DynamicCache | Flash Attention | Flash Attention |
| eager StaticCache | memory-efficient attention | memory-efficient attention |
| compiled StaticCache, 기본/cast 보존 | memory-efficient attention | memory-efficient attention |

캐시 공간과 backend 선택이 함께 바뀌므로 이것만으로 어느 한 요소에 전체 오차를
귀속하지 않는다. 다섯 비교 모두 같은 토큰을 넣었을 때 단계별 argmax는 일치했다.
실제 free-generation sequence 일치 테스트를 수행한 것은 아니다.

## 미완료 대조군과 정리

추가 `eager DynamicCache + memory-efficient 강제` 대조군은
Qwen3의 Q heads=40 / KV heads=8인 GQA 입력에서 사용 가능한 kernel이 없어 실패했다.
경고에 head 수 불일치가 기록돼 있다. 이 오류로 이후 backend 대조군과
분리 연산 microtest는 실행하지 못했다. 앞서 완료한 네 실행과 다섯 tensor 비교의
원자료는 회수했다. 실제 GPU micro 생성 코드를 회수했다고 주장하지 않는다.

실행기 경과 시간은 약 8분 7초이며, stop exit code 0과 서버 활성 세션 없음을 확인했다.
공식 `generate()`의 compile 경로 및 테스트 허용 오차와 이 검증의 차이는
[Transformers 소스 검토](../transformers-4.57.3-review.md)에 기록했다.

전체 표와 원자료: [report.md](report.md), [diagnosis.json](diagnosis.json),
[benchmark.log](benchmark.log), [lifecycle.json](lifecycle.json).
