# Qwen3-14B StaticCache + torch.compile 측정

실행 상태: **failed**.

Baseline 성능 비교 및 speculative decoding 없이 target 모델 하나만 실행했다.

- GPU: NVIDIA A100-SXM4-80GB; 메모리 79.250732421875 GiB.
- PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 4.57.3.
- 요청 조건: Qwen/Qwen3-14B, BF16, B=1, 고정 합성 토큰 입력, greedy 출력 128토큰.
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- SDPA 정책: `auto`. 자동 선택은 Flash Attention 사용을 보장하지 않는다.
- 전체 forward compile: fullgraph=True, dynamic=False, triton.cudagraphs=False.
- 길이별 최대 cache 용량은 입력 길이 + 128이며, 길이별 별도 컴파일·warmup을 수행한다.

실행 오류:

```text
RuntimeError: logits/KV validation failed at prompt length 128; timing blocked
```

## 측정값

정확도와 greedy 출력 검증을 통과한 길이만 성능값을 표시한다. 설정은 길이별 warmup 2회, 측정 3회이며 완료한 값만 요청별 중앙값으로 표시한다.

| 입력 | Prefill ms | TTFT ms | Decode 총 ms | Decode ms/token | token/s | ITL p50 ms | ITL p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

Prefill은 KV 기록과 마지막 위치의 vocabulary logits 계산까지이며 토큰 선택은 제외한다. TTFT는 미리 할당된 cache의 reset 시작부터 첫 토큰 CPU 수신까지다. Cache 최초 할당·모델 로딩·토큰화·전송·컴파일·정확도 검증·warmup은 정상 상태 지연에서 제외한다.

Decode는 첫 토큰 이후 127회 생성의 동기화된 wall time이다. ITL은 별도 streaming 실행에서 토큰을 하나씩 CPU로 읽은 시각의 차이며, 네트워크 시간은 제외한다. ITL p50/p95는 길이별 3×127개 간격을 모아 계산한다.

## 정확도·준비 비용·실제 attention

### 입력 128

- 수치 검증: False; 전체 vocabulary logits 및 모든 레이어의 유효 K/V 비교 608건.
- 같은 토큰을 넣었을 때 단계별 argmax 일치: True. 이것만으로 free-generation 일치나 logits/KV 정확도를 보장하지 않는다.
- logits 최대 절대오차: 10.5312; relative L2 최댓값: 0.253088.
- compile + 정확도 검증: 108.19998150899994초. 순수 compiler 시간만 분리한 수치는 아니다.
- warmup: 미완료초.
- 측정 중 compile counter 변화: 미확인.
- Prefill attention profiler events: `미수집`.
- Decode attention profiler events: `미수집`.

이 조건은 수치 검증에서 중단되어 warmup·free-generation 검증·지연 측정·profiler 수집을 실행하지 않았다.

수치 기준은 atol=0.02, rtol=0.02이며, reference는 compile을 끈 native Qwen3 + DynamicCache다. 같은 토큰으로 prefill 마지막 위치 및 매 decode 단계의 전체 vocabulary logits를 비교한다. K/V는 prefill과 step 1/4/32/64/127에서 모든 레이어·전체 유효 길이를 확인한다. 추가로 free-generation greedy token 일치와 bulk/streaming 출력 일치를 검사한다. 정확도 검증은 성능 측정 밖이다.

## 실행·정리

- 세션: `qwen3-flash-1015-134612-afa5`.
- 이 실행기 경과 시간: 254.59501792399988초. 기존 세션 재사용 시 최초 할당부터의 총 시간과 다르다.
- 종료 명령 exit code: 0.
- 서버 활성 세션 없음: True.

원자료: [measurements.json](measurements.json), [execution.json](execution.json), [hardware.json](hardware.json), [lifecycle.json](lifecycle.json), [benchmark.log](benchmark.log). Profiler 수집 단계에 도달한 경우에만 trace가 저장된다.
