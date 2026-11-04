# User HF Qwen3.5-35B-A3B INT8 — G4, MTP OFF

2026-11-05 KST. 사용자가 선택한 원래 체크포인트를 infer-opt에서 실행했다. GGUF 변환이나 재양자화는 하지 않았다.

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.970764 GiB. PyTorch 2.11.0+cu130, Triton 3.6.0, flash-linear-attention/fla-core 0.6.0. 단일 시퀀스, BF16 activations, INT8 group-128, router/small gates/lm_head는 체크포인트의 BF16 유지. GPU 클럭 고정 없음.

모델: https://huggingface.co/Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp (revision `331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388`).
실행 소스: https://github.com/Jake-Song/infer-opt/tree/a0a798494c295f1e6a850305944373356e33d982 .
35개 다운로드 파일의 SHA-256을 보존했으며, Hub LFS 해시가 있는 가중치 파일은 모두 일치했다. strict reload 통과. 가중치 할당 약 34.5405 GiB. 포함된 MTP head는 로드했지만 호출하지 않았다. `fuse_projections()` 82개 그룹을 적용했다.

## Decode

각 조건 워밍업 후 256개 고정 토큰을 7회 처리한 중앙값이다. 단위는 ms/token이고 괄호는 token/s다.

| 기존 문맥 토큰 | infer-opt CUDA Graph | infer-opt compiled (수치 기준 실패, 진단용) | 앞서 측정한 llama-bench Q8_0 |
| ---: | ---: | ---: | ---: |
| 0 | 9.036 (110.66) | 5.562 (179.78) | 4.301 (232.52) |
| 512 | 9.056 (110.42) | 5.544 (180.37) | 4.275 (233.93) |
| 2,048 | 9.061 (110.36) | 5.545 (180.35) | 4.275 (233.90) |
| 8,192 | 9.142 (109.38) | 5.603 (178.47) | 4.427 (225.89) |

CUDA Graph는 수동 캡처, compiled는 `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)`다. max-autotune나 실험적인 BF16 GEMV override는 활성화하지 않았다. 준비 시간은 수동 Graph 약 1.49초, compiled 약 39.69초였으며 측정에서 제외했다.

## Prefill

prefill은 수정하지 않은 eager backbone과 FLA delta kernel을 사용한다. 전체 입력을 한 번에 처리하고 vocabulary logits는 마지막 위치에만 계산했다. 반복 사이 캐시 리셋은 제외했다.

| 입력 토큰 | 중앙값 ms | token/s |
| ---: | ---: | ---: |
| 512 | 1088.71 | 470.3 |
| 2,048 | 1307.68 | 1566.1 |
| 8,192 | 1572.58 | 5209.3 |

현재 MoE prefill 경로는 expert별 Python 순회와 reference 연산을 사용한다 (`infer_opt/moe.py:Int8Experts.forward`). decode의 selected-expert Triton 경로와 다르다. 병목별 소요 비중을 프로파일링하지 않았으므로 그 기여도를 수치로 단정하지 않는다.

## 수치 확인

512토큰 문맥 뒤 고정 토큰 16단계를 같은 체크포인트의 fixed-shape eager와 비교했다. 검증은 fused projection 상태에서 수행했으며 BF16 원본과의 품질 검증이 아니다.

| 경로 | 최대 로그잇 절대 차이 | argmax 일치 | allclose(atol=rtol=0.02) |
| --- | ---: | ---: | --- |
| 수동 CUDA Graph | 0 | 16/16 | 16/16 통과 |
| compiled | 0.8125 | 16/16 | 0/16 통과 |

compiled의 argmax 일치만으로 수치 동등성을 주장하지 않는다. compiled timing은 진단용이고 품질을 유지한 최적화 속도 향상으로 취급하지 않는다. 실패 원인은 이 실험에서 확정하지 않았다. 다른 문맥 길이의 로그잇 비교, 장기 free-running parity, HF BF16 reference 및 작업별 정확도는 평가하지 않았다.

실제 문장 생성은 수동 CUDA Graph에서 48토큰 확인했다. `Explain briefly why the sky is blue.`에 대해 Rayleigh scattering을 설명하는 출력을 얻었다 (`results/generation.json`). 이것은 기능 확인이며 품질 벤치마크가 아니다.

## 측정 범위와 llama.cpp 비교 한계

- 모든 본 측정은 임의 토큰, batch 1, MTP OFF, 워밍업 뒤 7회다. 입력 ID는 `results/input_ids.json`에 보존했다.
- synchronized host wall time을 사용했다. decode는 llama-bench와 같이 매 토큰 GPU 완료를 동기화한다. 모델 로드, 다운로드, 컴파일, 워밍업, 캐시 준비/복원, argmax/sampling, HTTP는 제외했다.
- prefill은 마지막 위치의 vocabulary logits만 포함한다. TTFT/ITL 또는 서비스 요청 종단 간 측정이 아니다.
- infer-opt는 컴파일을 한 번 수행하도록 static cache capacity 8456을 모든 문맥 길이에 공통으로 사용했다. llama-bench는 시험 조건에 맞는 context capacity를 사용했다.
- 두 엔진은 같은 GPU 모델의 별도 Colab 세션에서 측정했다. 같은 입력 ID를 사용한 paired benchmark가 아니며, 양자화 방식(INT8 group-128 vs Q8_0), activation/cache dtype, 프리필 chunking, 실행 커널도 다르다.
- 이 비교로 모델 품질 또는 특정 커널/양자화 하나의 우열을 분리해 결론 내리지 않는다.

## 보존 및 재현

- `worker.py`: 공개 저장소의 고정 revision을 clone하고 HF 고정 revision을 다운로드하여 실행하는 스크립트.
- `launch.py`: Colab 커널에서 worker를 저장하고 분리 실행하는 런처.
- `results/summary.json`: 원시 반복별 ms 및 요약.
- `results/manifest.json`: 전체 파일 해시·모델/코드 리비전·다운로드 경로.
- `results/validation-*.json`: 16단계 수치 비교.
- `results/generation.json`, `load.json`, `environment.json`, `install.log`, `worker.log`: 생성·로드·환경·패키지 로그.
- `results/`: 회수한 원시 결과. 가중치와 clone한 저장소는 포함하지 않는다.
- 모델은 `/tmp/infer-opt-models/huggingface/hub`에서 다운로드/로드했다. 로컬 저장소에는 모델 가중치를 받지 않았다.
- 세션 `q35moe-hf-int8-final` 종료 후 `colab sessions`에서 활성 세션 없음 확인. 앞서 확인 질문 중 중지한 `q35moe-hf-int8`도 종료됐다.
- 프로젝트 `AGENTS.md`에 향후 모델 다운로드 및 캐시를 `/tmp/infer-opt-models`에 두도록 기록했다.

직접 실행 예: G4에서 `python3 worker.py`. 공개 인터넷 및 CUDA PyTorch가 사전 설치된 Colab 환경을 전제로 한다. 원격 작업은 완료까지 실행한 다음 `/content/q35moe-hf-int8-results.tar.gz`를 회수한다.

## 이전 G4 compiled 실험과 조건 차이

이전 [spec-compiled.json](../qwen35_moe_g4/spec-compiled.json)의 일반 디코드도 `compile-reduce-overhead`였다. 5종 실제 문장(20/43/1000/1057/32토큰), 5회 반복, 256개 출력 토큰에서 문장별 중앙값은 4.907–4.936 ms/token이었다. 해당 파일의 `args.repo`는 미사용 기본값(27B)이며, 실제 `--weights` 경로와 환경 기록은 35B MoE다.

이전은 cache capacity 1321(최대 프롬프트 1057 + 출력 256 + 여유 8), greedy argmax 포함, 생성 루프 전후 동기화, 255단계 timing이었다. 이번은 cache capacity 8456, 고정 임의 입력, argmax 제외, 매 토큰 완료 동기화, 256단계 timing이다. 따라서 4.9→5.5 ms/token을 동일 조건 성능 퇴보로 해석할 수 없다. 각 차이의 기여도를 분리 측정하지 않았다.

이전 compiled MTP와 일반 compiled 디코드의 토큰 패리티 검사는 이번의 MTP 없는 compiled/eager 로그잇 비교와 다르다. 이전 기록만으로 이번 `allclose(atol=rtol=0.02)` 검사의 통과 여부를 알 수 없다. 가장 최근의 [2026-11-04 A100 실험](../qwen35_moe_a100_20261104/README.md)은 수동 CUDA Graph로, compiled 단계는 런타임 중단 때문에 미측정이었다.
