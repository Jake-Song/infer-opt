# StaticCache + Flash SDPA 호환성 검증

실제 Colab **NVIDIA A100-SXM4-80GB (79.25 GiB)**에서 확인했다.
PyTorch **2.11.0+cu130**, CUDA **13.0**, Transformers **4.57.3**, BF16이다.

이 결과는 작은 무작위 Qwen3 모델의 실행 가능성 검사다. Qwen3-14B의 성능이나 정확도 결과가 아니다.
2개 레이어, hidden size 256, head dimension 64, 4 query heads / 2 KV heads,
입력 16토큰, StaticCache 용량 32로 prefill과 1토큰 decode를 시도했다.

| 설정 | Prefill | Decode |
|---|---|---|
| DynamicCache + 강제 Flash SDPA, eager | 통과 | 통과 |
| StaticCache + SDPA 자동 선택, eager | 통과 | 통과 |
| StaticCache + 강제 Flash SDPA, eager | 실패 | 실행하지 않음 |
| StaticCache + 강제 Flash SDPA, compile | 실패 | 실행하지 않음 |

오류는 `Flash Attention does not support non-null attn_mask`와
`No available kernel. Aborting execution.`이다. StaticCache의 미사용 영역을 가리는
mask를 이 버전의 PyTorch Flash SDPA가 지원하지 않는다. compile을 끄고도 같은
오류가 나므로 compile 자체의 수치 오차 문제는 아니다.

이 조합의 prefill·decode·TTFT·ITL은 **미측정**이다. 14B 가중치는 다운로드하지 않았다.
검사에 성공한 제어 경로에서도 이 단계에서는 latency를 측정하지 않았다.

최초 Colab CLI 실행에서 연결이 끊겼다. 같은 세션에 재연결해 검사를 완료하고
원자료를 회수했다. 추가 GPU 할당은 하지 않았다. 이후 사용자 선택에 따라
같은 세션에서 **StaticCache + compile + SDPA 자동 선택** 측정으로 이어갔다.
세션 종료 상태는 후속 실행의 lifecycle 기록을 확인한다.

원자료: [compatibility.json](compatibility.json), [hardware.json](hardware.json),
[benchmark.log](benchmark.log), [execution.json](execution.json).
실제로 실행한 원본은 이 디렉터리의 `measure_qwen3_flash_compile.py`에 보존했다.
