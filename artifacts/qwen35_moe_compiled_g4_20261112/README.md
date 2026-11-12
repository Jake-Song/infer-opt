# G4 compiled parallel MTP 실측, 2026-11-12

사용자가 열어 둔 Colab `qwen35-moe-int8-mtp` 세션에서 보류했던 실험을 완료했다.
실제 GPU는 NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, VRAM 94.9708 GiB다.
새 GPU 할당 없이 기존 세션을 사용했다. 네 실행 조건 모두 profiler에서
`cudaGraphLaunch` 1회를 관측했다. MTP-off는 decode 1 step, MTP는 1 round의 별도 측정이다.

코드 프롬프트에서는 k=1/2/3 모두 MTP-off와 256토큰이 5회 전부 일치했다.
가장 빠른 k=3은 **3.2078 ms/token, 1.4399배**였다.
수학은 k=1/2가 일치하며 k=2는 **3.3302 ms/token, 1.3661배**였다.
수학 k=3과 채팅·산문·한국어의 모든 MTP 조건은 출력이 달라 진단용 시간이다.

## 시간과 출력 일치

각 값은 5회 측정의 중앙값(ms/token). `†`는 5회 모두 greedy 출력이 달랐음을 뜻한다.
일치는 같은 compiled MTP-off 출력을 기준으로 하며 HF BF16/eager logit 동등성이나 task 정확도를 뜻하지 않는다.

| 프롬프트 | 입력 토큰 | MTP-off | k=1 | k=2 | k=3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chat_long | 32 | 4.6488 | 4.1505† | 3.7890† | 3.5182† |
| code_1k | 1000 | 4.6190 | 3.9930 | 3.4206 | 3.2078 |
| prose_1k | 1057 | 4.5625 | 4.1976† | 3.6560† | 2.8154† |
| math | 43 | 4.5494 | 4.0097 | 3.3302 | 2.8848† |
| korean | 20 | 4.5445 | 4.3189† | 4.4853† | 4.8707† |

출력이 일치한 조건에 한정한 MTP-off 대비 배율:

| 프롬프트 | k=1 | k=2 | k=3 |
| --- | ---: | ---: | ---: |
| code_1k | 1.1568× | 1.3504× | 1.4399× |
| math | 1.1346× | 1.3661× | 출력 불일치 |

## 불일치와 수락량

최초 불일치는 1부터 세는 생성 토큰 위치다. 각 조건의 5회 반복에서 같은 위치였다.
수락량은 initial token을 제외한 한 round당 emitted token 수의 중앙값이다.
채팅·산문·한국어와 수학 k=3의 시간 비율은 출력 보존 speedup으로 해석하지 않는다.

| 프롬프트 | 최초 불일치 k=1/2/3 | tokens/round k=1/2/3 |
| --- | --- | --- |
| chat_long | 122 / 122 / 122 | 1.8963 / 2.5248 / 3.1341 |
| code_1k | 일치 / 일치 / 일치 | 1.9692 / 2.7935 / 3.3947 |
| prose_1k | 64 / 166 / 153 | 1.8478 / 2.5960 / 3.9231 |
| math | 일치 / 일치 / 18 | 1.9394 / 2.8444 / 3.8060 |
| korean | 121 / 167 / 167 | 1.7778 / 2.0732 / 2.1983 |

한국어는 k를 늘려도 tokens/round가 1.7778→2.0732→2.1983에 그쳤고,
k=3 시간은 4.8707 ms/token으로 MTP-off 4.5445 ms/token보다 길었다.
이 결과만으로 개별 target/draft 커널의 비용을 분리할 수는 없다.

## 실행 조건

- 체크포인트: `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`.
- 22개 safetensors shard, 34.5406 GiB. HF LFS SHA-256 대조와 strict model reload를 통과했다.
- PyTorch `2.11.0+cu130`, Triton `3.6.0`, FLA/fla-core `0.6.0`, Transformers `5.22.0`, Python `3.13.16`.
- `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`, `max_autotune=True`, `coordinate_descent_tuning=True`.
- Inductor의 자동 CUDA Graph 캡처/재생 경로. 수동 `torch.cuda.CUDAGraph` 캡처 모드는 사용하지 않았다.
- MTP `verification="parallel"`, k=1/2/3, INT8 group128 draft; LM head BF16. 기존 기본 커널·warp 설정 유지.
- 5개 기존 프롬프트, cache capacity 2048, 출력 256토큰, warmup 3회, 측정 5회. 총 20조건·100회.
- CUDA 동기화를 포함한 host wall-clock decode 측정. prefill·첫 토큰·컴파일·warmup·profiler 시간 제외.
- MTP의 마지막 round 초과 생성분은 기존 계약대로 emitted-token 분모에 포함된다. 보관된 generated IDs와 parity 판정은 첫 256토큰이다.
- GPU clock 고정 없이 provider boost 사용; 1초 간격 telemetry 보관. 다른 GPU나 과거 실행과의 paired 순위는 주장하지 않는다.
- 모델·HF·compiler·pip cache는 원격 `/tmp/infer-opt-models/moe-compiled-g4-20261112`에 저장. 모델 weights는 저장소에 없다.
- HF BF16 reference와 task quality는 이번에 다시 측정하지 않았다. source kernel 변경은 없다.

Inductor autotune은 shared-memory 한도를 넘은 일부 후보를 제외하고 유효한 후보를 선택했다.
해당 로그는 컴파일 단계에 속하며 전체 benchmark와 worker는 exit code 0으로 종료했다.
기본 이미지의 사용하지 않는 fastai/timm에는 torchvision 제거에 따른 pip dependency 경고가 남았다.

## 검증과 원시기록

원격 최종 archive SHA-256과 회수한 archive가 일치하고, 내부 76개 파일의 크기·해시를 검증했다.
독립 validator는 100개 timing sample, 25,600개 generated IDs, 입력 일치,
수락 histogram·round 수·분모·중앙값·배율·greedy 일치 판정, worker 해시,
네 모드의 CUDA Graph launch를 확인했다. 원격 source manifest의 32개 파일은 현재 checkout과도 대조했다.
`validation.json`의 `passed`는 기록의 일관성 검증 통과이며 모든 MTP 조건의 출력 일치를 뜻하지 않는다.

```bash
python3 scripts/moe_compiled_g4/analyze.py artifacts/qwen35_moe_compiled_g4_20261112/results
```

- [독립 검증](results/validation.json), [시간 요약](results/benchmark/summary.json), [CUDA Graph 증거](results/benchmark/cuda-graph-evidence.json).
- [고정 입력](results/benchmark/inputs.json), [환경](results/environment.json), [조건](results/contract.json), [실행 명령](results/benchmark.command.json).
- [체크포인트 해시](results/checkpoint-hashes.json), [GPU telemetry](results/gpu-telemetry.csv), [worker 종료](results/exit.json).
- [최종 원시기록 archive](final-results.tar.gz), [회수 해시 검증](archive-verification.json), [내부 파일 해시](results/files-manifest.json).
- [세션 재개 기록](resume.json), [실행 후 종료 확인](execution-cleanup.json), [harness](../../scripts/moe_compiled_g4/README.md).

## A100과 비교

[2026-11-09 A100 실행과의 비교](derived/a100-vs-g4.md)는 같은 입력·설정 및 실제 출력 ID를 대조했다. G4의 MTP-off 시간은 약 16~18% 짧았다. 수학 k=1/2는 GPU 간 출력까지 같았으며 G4에서 각각 1.39배·1.36배 빨랐다. 코드·산문·한국어는 GPU 간 MTP-off 출력도 달라 시간 비율을 동일 출력 workload의 speedup으로 해석하지 않는다.

## 하드웨어 하한 계산

[별도 계산 보고서](derived/README.md)는 원시 측정과 구분한 분석이다. MTP-off의 캐시 허용 가중치 전송 하한은 약 2.00 ms/token이며, 코드 k=3은 약 1.04 ms/token이다. 상태·KV를 포함한 이상적 streaming 시나리오는 각각 2.30 ms/token과 1.34~1.92 ms/token이다. 실제 DRAM counter 측정이나 달성 보장치는 아니다.

## 앞선 할당 실패 이력

같은 날짜 KST의 최초 G4 할당과 사용자 승인 재시도는 `503 Service Unavailable`로 실패해 실험을 보류했었다.
재시도에서는 `COLAB_CLI_ASSIGN_RETRIES=0`을 사용했다. 이 실패 기록은 이번 성공한 기존 세션 실행과 구분해 보존한다.
[최초 실패](allocation-failure.json), [재시도 실패](allocation-retry-1.json), [당시 cleanup](cleanup.json), [원래 요청](request.json).
