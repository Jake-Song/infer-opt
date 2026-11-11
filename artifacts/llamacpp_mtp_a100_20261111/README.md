# llama.cpp Qwen3.5-35B-A3B native MTP on A100

2026-11-11~12 KST. **MTP off / k=1 / k=2 / k=3을 실제 A100 80GB에서 측정했다.**
수학은 k=3, 한국어는 k=1/2에서 시간이 짧았지만, 모든 MTP 설정이 MTP-off와
생성 출력이 달랐다. 따라서 아래 MTP 시간은 진단용이며 출력 보존 가속을 입증하지 않는다.
CUDA 커널과 warp 설정은 변경하지 않았다.

## 실행 조건

- Colab NVIDIA A100-SXM4-80GB, visible 79.250732 GiB.
- Driver 580.82.07, CUDA toolkit 13.0.88, Nsight Systems 2025.3.1.
- SM 클럭 1095 MHz 고정. 두 배치의 모든 설정 종료 후 telemetry는 1095 MHz였다.
  측정된 memory clock은 1593 MHz였다. 기본 boost 클럭의 G4/A100 과거 결과와 직접 비교하지 않는다.
- llama.cpp `a0ad695cb212db3850f163121ddc687513448d42`, Release CUDA SM80,
  CUDA Graph ON. Production 소스 diff가 비어 있음을 빌드 뒤 검사했다.
- `unsloth/Qwen3.5-35B-A3B-MTP-GGUF`, revision
  `63af8373893a7a73c6dfcb84cb63d815981da5e0`, `Qwen3.5-35B-A3B-Q8_0.gguf`.
  파일 37,801,097,376 bytes, SHA-256
  `b08e017b6ffcd1f57343a603a9e6d9e498373ef09da55537a2788af70ceb1d0d`.
  Hub LFS hash와 일치했다. GGUF의 41 blocks, NextN 1개와 MTP 텐서를 직접 확인했다.
- 기존 ggml-org MTP-off GGUF와 다른 파일이다. 이번 off와 MTP에는 모두 위의 동일 파일을 사용했다.
- Batch 1, context capacity 2048, flash attention ON, attention K/V F16,
  GPU offload `-ngl 99`, MTP draft `-ngld 99`, CPU threads 4.
- 수학 43토큰, 한국어 20토큰. 기존 infer-opt 실험의 입력 ID를 그대로 전달하고
  llama.cpp tokenizer의 결과와 exact임을 검사했다. Greedy, 출력 256토큰, EOS 무시.
- Prompt cache OFF. 각 서버에서 입력별 warmup 3회. 첫 배치는 off/1/2/3 순서로
  각 3회, 두 번째는 3/2/1/off 순서로 각 2회 측정했다. 총 timed request 40개,
  출력 ID 10,240개, 본 측정용 warmup request 48개다.

## 측정 경계

표는 profiler 없는 서버 내부 generation wall time 중앙값이다. 서버가 첫 토큰을
샘플링한 시점부터 마지막 생성까지 측정하고 **255개 decode 토큰**으로 나눈다.
다운로드·모델 로드·컴파일·warmup·prefill·첫 토큰과 HTTP 전송은 제외한다.
Native sampling, speculative driver, acceptance/rollback 및 decode의 CPU/GPU 대기는 포함한다.
HTTP wall time과 원래 server timing은 별도로 보존했다.

과거 G4의 임의 토큰 `llama-bench`와 입력·양자화 배포본·GPU·클럭·sampling·동기화 경계가 다르다.
또한 과거 infer-opt A100의 기본 클럭 시간과 paired engine 비교가 아니다.

## Decode 결과

각 항목은 5회 중앙값. 시간비는 `같은 입력 off ms/token / 해당 설정 ms/token`이다.
MTP의 시간비를 품질을 보존한 가속률로 사용하지 않는다.

| 입력 | 설정 | ms/token | token/s | 시간비 | Draft 수락률 | off와 256토큰 전체 일치 |
|---|---|---:|---:|---:|---:|---:|
| 수학 | off | 9.1318 | 109.51 | 1.000 | — | 5/5 |
| 수학 | k=1 | 6.7890 | 147.30 | 1.345 | 93.9% | 0/5 |
| 수학 | k=2 | 5.9576 | 167.85 | 1.533 | 89.6% | 0/5 |
| 수학 | k=3 | 5.6413 | 177.26 | 1.619 | 88.5% | 0/5 |
| 한국어 | off | 9.1887 | 108.83 | 1.000 | — | 5/5 |
| 한국어 | k=1 | 7.4750 | 133.78 | 1.229 | 75.2% | 0/5 |
| 한국어 | k=2 | 7.3962 | 135.20 | 1.242 | 58.8% | 0/5 |
| 한국어 | k=3 | 8.4039 | 118.99 | 1.093 | 44.2% | 0/5 |

한국어 k=1/2의 차이는 약 1%로 작다. 이 제한된 반복만으로 안정적인 우승 설정을 확정하지 않는다.
한국어 k=3은 k=2보다 draft를 더 생성하지만 수락률이 낮고, 토큰당 시간도 길었다.
MTP-off 출력은 입력별 5회 모두 동일했다. MTP의 첫 불일치는 모든 반복에서 동일했다.

| 입력 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| 수학: 첫 불일치 index | 43 | 48 | 43 |
| 한국어: 첫 불일치 index | 43 | 43 | 47 |

Index는 생성 토큰 기준 0부터 센다. 서로 다른 T의 수치 연산, routing 민감도,
상태 처리 등의 기여도를 분리하지 않았으며 원인은 확정하지 않았다.
HF BF16 logits 동등성·분포 품질·태스크 정확도 검사는 이번 범위에 없다.
Free-running 출력 불일치율은 태스크 정확도 하락률이 아니다.

## 별도 CUDA trace

Off/k=1/2/3 × 수학/한국어의 **8개 Nsight Systems CUDA node trace**를 수집했다.
모든 trace에서 CUDA Graph launch와 Q8_0 MMVQ 커널을 확인했다. 다중 토큰
`mul_mat_vec_q_moe`는 k=1/2/3 실행에서 각각 `(32,2,1)`, `(32,3,1)`,
`(32,4,1)` threads/CTA 구성을 확인했다. 기본 dispatcher를 사용했고 warp override는 없다.
낮은 draft 길이와 prefill 때문에 다른 block 크기도 같은 trace에 포함될 수 있다.

[trace-validation.json](results/trace-validation.json)은 8개 SQLite 무결성, kernel/graph 존재,
실행별 block 크기 및 profiler 유무에 따른 같은 설정의 출력 exact를 검사한 결과다.
프로파일링한 8개 실행의 출력은 모두 해당 설정의 profiler 없는 출력과 일치했다.

이 trace는 **prefill을 포함한 요청 전체**를 수집했다. Target/draft 구간에 별도 표지를
넣지 않았으므로 expert의 target/draft별 ms/round 분해를 완료했다고 주장하지 않는다.
GPU kernel duration 합을 본문의 profiler 없는 wall time과 합산하거나 대체하지 않는다.
MoE MMVQ의 선택을 전체 모델의 모든 연산이 Tensor Core를 사용하지 않는다는 뜻으로 확대하지 않는다.

최초 작은 profiler probe는 긴 수명의 `nsys launch` 반환을 기다리다 120초 timeout이 났다.
본 측정 코드는 유지하고, trace 시작 전에 대기 중인 profiler 실행기만 `Popen` 방식으로 바꿨다.
실패 기록과 이전/최종 profiler 소스·해시는 보존했다. 이후 8개 trace는 모두 정상 완료했다.
중간 Colab 연결 단절 때도 기존 세션과 저장된 archive를 확인해 회수했으며 추가 할당은 하지 않았다.

## 검산과 원본

[validation.json](results/validation.json)은 별도 로컬 분석기로 40개 시간·분모·수락률·출력 ID,
입력 tokenizer 일치, 48회 warmup, source/model hash 및 telemetry를 검산한 결과다.
[summary.json](results/summary.json), [timed-requests.json](results/timed-requests.json)에
요약과 전체 원시 응답을 보존했다. 검산기는 [analyze.py](../../scripts/llamacpp_mtp/analyze.py)다.

Before/after Prometheus 원본에는 speculative verification 횟수도 있다.
Raw record의 `ms_round`는 전체 decode 시간을 이 카운터로 나눈 값이며, 카운터가
일반 단일 토큰 step을 제외하므로 독립된 round 지연으로 보고하지 않는다.

실행 코드는 [scripts/llamacpp_mtp](../../scripts/llamacpp_mtp/README.md)에 있다.
가중치와 모델 cache는 원격 `/tmp/infer-opt-models/llama-mtp-20261111`에만 저장했고
저장소 결과에는 포함하지 않는다.

## 회수와 정리

최종 archive **90,714,703 bytes**, SHA-256
`20af124fcbe0eba5e87595815b75b174f06d6076a14588928dca41fbee9aed75`를
원격 값과 대조하고 파일 manifest **670개**의 크기·해시를 모두 검증했다.
[archive-validation.json](archive-validation.json), [files-manifest.json](results/files-manifest.json).
가중치·모델 cache·빌드 바이너리는 결과에 포함하지 않았다.

큰 `.nsys-rep`와 `.sqlite`의 원본은 `final-results.tar.gz`에 보존한다.
풀린 trace 파일은 중복 커밋하지 않는다. 저장소 루트에서 다음 명령으로 복원한 뒤
`analyze_traces.py`를 실행할 수 있다. 원시 로그의 공백과 줄바꿈은 해시 검증을 위해 그대로 유지한다.

```sh
tar -xzf artifacts/llamacpp_mtp_a100_20261111/final-results.tar.gz \
  -C artifacts/llamacpp_mtp_a100_20261111/results --wildcards 'traces/*'
python3 scripts/llamacpp_mtp/analyze_traces.py artifacts/llamacpp_mtp_a100_20261111/results
```

GPU clock reset 성공과 Nsight 활성 세션 없음을 확인한 뒤 전용 Colab 세션
`llama-mtp-a100-20261111`을 종료했다. 서버가 **활성 세션 0개**를 보고했다.
[cleanup.json](cleanup.json), [clock-reset.json](results/clock-reset.json),
[Nsight 종료 상태](results/nsys-final-sessions.json).
