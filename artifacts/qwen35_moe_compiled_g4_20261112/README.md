# G4 compiled parallel MTP: 할당 실패

Qwen3.5-35B-A3B INT8의 compiled MTP-off와 compiled parallel k=1/2/3 비교를 준비했다.

2026-11-12 KST Colab G4 할당 요청이 `503 Service Unavailable`로 실패했다.
최초 요청 뒤 사용자가 승인한 재시도도 `COLAB_CLI_ASSIGN_RETRIES=0`으로 한 번 수행했으나 동일한 503으로 실패했다.
실제 GPU 확인, 모델 다운로드, benchmark는 실행하지 않았으며 새 성능 결과는 없다.
서버의 활성 세션이 0개임을 확인했다.

사용자 요청에 따라 이번에는 준비 코드와 실패 기록을 저장하고 실측은 다음으로 미뤘다.
재개할 때 G4 할당부터 다시 확인해야 하며, 기존 실행 중인 worker나 모델 cache는 없다.

준비된 조건은 원래 INT8 checkpoint revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`,
기존 다섯 프롬프트, cache 2048, 출력 256토큰, warmup 3회, 측정 5회다.
PyTorch 2.11.0+cu130, Triton 3.6.0, FLA 0.6.0을 고정하고
max_autotune 및 coordinate descent를 활성화한다. CUDA 커널과 warp override는 없다.
시간과 같은 compiled off 대비 greedy 출력 일치를 별도로 검증한다.

Colab CLI 0.7.5에는 503 등에 대한 내부 자동 재시도 기본값 3이 있다.
최초 `colab new` 명령 1회와 명시적으로 승인된 재시도 1회를 실행했다.
승인된 재시도에서는 `COLAB_CLI_ASSIGN_RETRIES=0`으로 내부 재시도를 껐다.

[실험 조건](request.json), [할당 실패](allocation-failure.json),
[승인된 재시도 실패](allocation-retry-1.json), [소스 해시](source-manifest.json), [준비된 harness](../../scripts/moe_compiled_g4/README.md).
