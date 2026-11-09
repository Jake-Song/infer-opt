# Qwen3.5-35B-A3B INT8: compile + CUDA Graph와 병렬 MTP

2026-11-09 KST. **A100-SXM4-80GB에서 완료**했다. G4 할당 두 번이 503 오류로 실패한 뒤, 사용자가 승인한 A100 대체 실행을 사용했다.

MTP-off decode는 5.45–5.56 ms/token이었다. 출력이 일치한 k=3 결과는 **수학 4.15 ms/token (1.34×), 코드 4.53 ms/token (1.22×)**다. 채팅·산문·한국어는 모든 k에서 일반 디코딩과 출력이 달랐다. 각 배속의 기준은 같은 세션·같은 프롬프트의 compiled MTP-off decoder다.

## 실행 조건

- GPU: NVIDIA A100-SXM4-80GB, 79.250732 GiB visible; 제공자 기본 클록.
- PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0, Python 3.13.16, Transformers 5.22.0, FLA/fla-core 0.6.0.
- INT8 checkpoint: `Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp`, revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`. 22개 weight shard를 포함한 34개 파일을 해시 확인했다.
- HF BF16 reference: `Qwen/Qwen3.5-35B-A3B`, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- INT8 group-128 W8A16 target 및 MTP. Router·작은 gate·lm_head는 BF16. Fused projections, `fla-delta`, BF16 GEMV override 미사용.
- Dense G4의 compiler 설정을 적용: `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`, `max_autotune=True`, `coordinate_descent_tuning=True`.
- **MTP off 및 k=1/2/3 모두 프로파일러에서 `cudaGraphLaunch`를 관측**했다. [실행 증거](results/benchmark/cuda-graph-evidence.json).
- 배치 1, greedy, 출력 256토큰, 프롬프트·방법별 전체 생성 warmup 3회 및 측정 5회. 모든 target/MTP cache capacity는 2048로 맞췄다.
- 시간은 동기화한 host wall clock이다. Prefill·첫 토큰·컴파일·warmup·프로파일링은 제외한다. MTP는 마지막 round의 초과 생성까지 실제 emitted token 수에 포함한다. 원본의 `decode_ms`, `emitted_tokens`로 분모를 검산할 수 있다.

Dense와 compiler 옵션 및 측정 절차를 맞췄지만, 이번 GPU는 A100이고 historical dense G4와 별도 세션이다. Cache capacity도 이번 실행에 명시한 2048이다.

## Decode 결과

단위는 **ms/token**, 각 항목은 5회 중앙값이다. 굵은 MTP 값은 일반 INT8 compiled decoder와 256토큰 전체가 5회 모두 일치한 항목이다.

| 프롬프트 | 입력 토큰 | MTP off | MTP k=1 | MTP k=2 | MTP k=3 |
|---|---:|---:|---:|---:|---:|
| 채팅 | 32 | 5.5534 | 5.8750 | 5.0664 | 4.8925 |
| 코드 | 1000 | 5.5386 | 5.3982 | 4.5731 | **4.5258** |
| 산문 | 1057 | 5.5531 | 5.5531 | 5.0899 | 3.8331 |
| 수학 | 43 | 5.5575 | **5.5776** | **4.5268** | **4.1480** |
| 한국어 | 20 | 5.4513 | 6.4205 | 5.8693 | 6.2956 |

수학의 MTP k=1/2/3 배속은 각각 0.996× / 1.228× / 1.340×다. 코드는 k=3에서 1.224×로, 출력 일치와 추가 가속을 함께 확인했다. 산문의 k=3 시간은 빠르지만 출력이 달라서 출력 보존 가속으로 계산하지 않는다.

## MTP 출력 일치

일반 INT8 compiled greedy 출력과 비교했다. 첫 차이는 생성 토큰의 **0부터 시작하는 인덱스**다. 각 항목의 5회 반복에서 첫 차이 위치와 일치율이 동일했다.

| 프롬프트 | k=1 | k=2 | k=3 |
|---|---|---|---|
| 채팅 | 0/5회 일치 · 첫 차이 80 | 0/5회 일치 · 첫 차이 237 | 0/5회 일치 · 첫 차이 237 |
| 코드 | 0/5회 일치 · 첫 차이 232 | 0/5회 일치 · 첫 차이 232 | 5/5회 일치 |
| 산문 | 0/5회 일치 · 첫 차이 67 | 0/5회 일치 · 첫 차이 67 | 0/5회 일치 · 첫 차이 165 |
| 수학 | 5/5회 일치 | 5/5회 일치 | 5/5회 일치 |
| 한국어 | 0/5회 일치 · 첫 차이 29 | 0/5회 일치 · 첫 차이 94 | 0/5회 일치 · 첫 차이 29 |

전체 MTP 75회 중 20회가 256토큰 전체 일치했다. 반복은 고정 프롬프트 5개에 대한 것이며 독립적인 문제 75개가 아니다. Free-run이 갈라진 이후의 토큰 일치율은 태스크 정확도나 품질 하락률이 아니다. MTP의 HF 기준 동일-prefix 분포 비교는 이 실험 범위에 포함하지 않았다.

## 정확도 검증: MTP off

HF BF16 cached-greedy reference를 생성한 뒤 **9개 프롬프트 × 48위치 = 432위치**를 teacher forcing으로 재생했다. Eager와 compiled 경로는 같은 HF reference logit hash와 continuation을 공유한다. MTP는 로드되지만 호출하지 않았다. 첫 위치는 eager last-prefill, 나머지 47위치는 각 decoder 경로다.

| MTP-off 경로 | 평균 KL(HF∥INT8) | top-1 일치 | Strong mismatch |
|---|---:|---:|---:|
| eager | 0.001853772 | 430/432 (99.537%) | 0 |
| compile + CUDA Graph | 0.002054525 | 430/432 (99.537%) | 0 |

Strong mismatch는 HF top-1/top-2 logit margin이 0.5보다 크면서 top-1 선택이 다른 경우다. Compiled의 두 불일치는 `hello`의 step 16·19였고 HF margin은 모두 0.125였다. Eager는 `hello` step 16과 `korean` step 29에서 달랐으며 margin은 역시 0.125였다. 모든 logits는 유한값이었다.

**Compiled/eager의 엄격한 logits 동등성 검사는 실패**했다. `allclose(atol=rtol=0.02)`는 실제 decode 423위치에서 0회 통과했고, 공유 eager-prefill 9위치만 통과했다. 최대 절댓값 차이는 7.0이며 top-1 선택은 2위치에서 달랐다. HF 대비 분포 비교와 eager 수치 동등성은 별도 결과다. 이 측정은 지정된 prefix의 수치 비교이며 MMLU·GSM8K 등의 태스크 정확도 평가가 아니다.

## 검증·실행 기록

- [comparison_summary.json](comparison_summary.json): 정확도와 성능 집계.
- [results/benchmark/summary.json](results/benchmark/summary.json): 프롬프트별 속도·일치·첫 차이.
- [results/benchmark/inputs.json](results/benchmark/inputs.json): 실제 입력 텍스트와 토큰 ID. 같은 폴더의 방법별 파일에 반복마다 256개 생성 토큰 ID, baseline ID, 시간, accepted histogram이 있다.
- [results/quality/quality.json](results/quality/quality.json), [compiled-vs-eager.json](results/quality/compiled-vs-eager.json): 위치별 수치, 선택 토큰, margin, hash와 별도 허용오차 검사.
- [verify_results.py](verify_results.py)로 100회 측정·25,600개 생성 토큰 ID·배속·시간 분모·HF paired hash·source hash를 검산했다. [verification.json](verification.json).
- 정확도 검증 도구의 로컬 CPU 테스트는 9개 통과했다. 기존 정확도·allclose 임계값은 그대로 사용했다.
- Worker 완료 코드 0, 실행 2668.4초(할당 대기 제외). 전체 benchmark의 관측 peak allocated memory는 35.879 GiB다.
- 새 런타임의 기본 torch 2.14.1은 추론 시작 전에 2.11.0으로 고정했다. 초기 다운로드를 중단·재개한 경위는 `request.json`과 복구 스크립트에 남겼다. 모니터링 연결 단절 중에도 독립 worker는 계속 실행됐고 중복 실행하지 않았다.
- 모델 가중치·캐시는 원격 `/tmp/infer-opt-models/q35moe-compiled-20261109`에만 두었다. 저장소에는 결과·로그·소스만 회수했다.
- 결과 회수·검산 후 전용 A100 세션을 종료했고 서버에 활성 세션이 없음을 확인했다. [cleanup.json](cleanup.json).

## 재현 자료

[worker.py](worker.py)에 패키지 고정, 공개 revision 다운로드, 해시 검증, HF reference 생성, MTP-off 정확도 평가와 benchmark 명령이 있다. [source.tar.gz](source.tar.gz)와 [source-manifest.json](source-manifest.json)은 실제 사용한 소스 및 SHA-256이다. Base code revision은 `faf3de5cf9f23c2ee2f29ec94ed80721609dc58f`이며 새 benchmark와 autotune CLI 옵션을 포함한다.

새 전용 런타임에서는 `launch.py`로 실행한다. `launch-initial-runtime.py`, `pause-runtime-setup.py`, `restart-pinned-worker.py`는 이번 실행의 초기 환경 수정 기록이다. 회수한 결과의 로컬 검산은 다음과 같다.

```bash
python3 artifacts/qwen35_moe_compiled_20261109/verify_results.py
```

## MTP가 느려지는 경우의 원인 분석

측정된 평균 round 비용을 출력 토큰 수로 나누면 토큰당 시간이 된다. MTP가 빨라지려면 `토큰/round > round 시간 / 일반 decode 토큰 시간`이어야 한다. [전체 15개 설정의 계산](round_cost_analysis.json).

| 프롬프트 / k | MTP off ms/token | Round ms | 실제 토큰/round | 손익분기 토큰/round | MTP ms/token |
|---|---:|---:|---:|---:|---:|
| 수학 / 1 | 5.557 | 10.900 | 1.954 | 1.961 | 5.578 |
| 한국어 / 1 | 5.451 | 10.701 | 1.667 | 1.963 | 6.421 |
| 한국어 / 2 | 5.451 | 12.521 | 2.133 | 2.297 | 5.869 |
| 한국어 / 3 | 5.451 | 14.709 | 2.336 | 2.698 | 6.296 |
| 수학 / 3 | 5.557 | 14.956 | 3.606 | 2.691 | 4.148 |

한국어 k=3은 한 round가 일반 decode 2.70회분인데 실제로는 2.34토큰만 출력해 15.5% 느렸다. k=1 수학은 draft 수락이 95.4%인데도 round가 1.96회분이라 거의 동률이다. 현재 k=1 round 비용을 고정하면 모든 draft가 수락돼도 최대 가속은 프롬프트별 약 1.019–1.035배다. 한국어 출력은 baseline과 불일치하므로 해당 시간은 진단용이다.

**구현에서 확인한 비용:** `moe_kernels.py`의 selected-expert kernel은 `(token × top-k expert, output row)`로 program을 배치하고, 각 program에서 해당 expert의 가중치를 읽어 GEMV를 계산한다. 토큰별 route를 expert로 묶어 가중치 타일을 공유하는 다중 토큰 GEMM은 구현되어 있지 않다. 같은 expert가 겹치면 GPU cache 재사용은 가능하지만, 이를 명시적으로 공유하는 구조는 아니다. 반면 dense `w8a16_group_gemv_rows`는 다중 activation row가 같은 가중치 타일을 사용한다. 실제 HBM 트래픽과 expert 중복률은 이 실행에서 측정하지 않았다.

`spec_round`에는 target 검증 외에 MTP head k회, lm_head k+1회, 후보별 recurrent state 저장과 commit이 있다. MTP head의 첫 호출은 k+1개 후보 위치를 처리한 뒤 마지막 수락 위치를 선택한다. 이는 수락 prefix의 MTP KV 상태를 유지하는 경로이므로, 마지막 위치만 계산하도록 단순히 바꾸면 동등성이 보장되지 않는다. `round()`는 매 round GPU 결과를 `.tolist()`로 호스트에 돌려주며 동기화한다. 일반 decode 벤치마크는 토큰 루프를 enqueue한 뒤 끝에서 동기화한다. CUDA Graph 적용 후에도 이 연산·데이터 이동·호스트 대기 비용은 남는다.

Round 총시간과 수락 토큰 수는 측정값이다. 위 구현 특성은 코드에서 확인했다. Verify, MTP head, lm_head, state traffic, host wait의 각각 몇 ms인지는 이번 profiler가 graph 실행 확인만 수행했기 때문에 아직 분해되지 않았다. 지배적인 병목을 확정하려면 그 구간별 프로파일과 T=1/2/3/4 expert kernel 비용 및 route 중복률을 같은 조건에서 측정해야 한다. 현재 데이터에서는 한국어가 MTP off를 선택하는 편이 빠르며, 출력이 보존된 추가 가속은 수학·코드 k=3에서 확인됐다.

후속 [A100 round profiling](../qwen35_moe_round_profile_a100_20261109/README.md)에서 원래 경로 trace와 구간·expert 계측을 완료했다. Expert가 k=1→3 GPU 커널 시간 증가분의 약 61%를 설명했다. 새 실행에서는 수학 k=3도 출력이 불일치했으므로 위 일치는 이 보고서의 원래 실행에 한정된다. 수동 graph 대조의 일부 상태도 달라 동등한 최적화 효과로 확정하지 않았다.
