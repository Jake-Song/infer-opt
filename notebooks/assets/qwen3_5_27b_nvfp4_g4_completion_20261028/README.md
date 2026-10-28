# Qwen3.5-27B NVFP4 G4 실험 완료

현재 NVFP4 체크포인트는 전체 모델 정확도 기준을 통과하지 못했다. INT8은 통과했다. NVFP4+MTP 성능 수치는 출력 일치 검증 결과와 함께 해석해야 한다.

## 실행 조건

- RTX PRO 6000 Blackwell Server Edition (SM120), 94.97 GiB VRAM, torch 2.11.0+cu130. 한 G4 세션에서 비교했다.
- FlashInfer 0.6.6 CUTLASS NVFP4 W4A4, block size 16, 최대 128행 단위 GEMM. MTP INT8 group 128, lm_head BF16.
- Decode: 5개 프롬프트, 256개 출력 토큰, warmup 3회 + 측정 5회. 각 행은 반복 측정 중앙값이다.
- BF16 plain 5개, INT8/NVFP4 각각 plain 5개 + MTP k=1/2/3 15개, 총 45개 decode case 완료.
- 각 정밀도에서 512/1024/4096 토큰 prefill 완료. eager model forward와 lm_head 포함, cache 할당·tokenization 제외. 서비스 TTFT는 측정하지 않았다.
- Decode 시간은 prefill과 첫 토큰을 제외한다. MTP 시간 분모에는 마지막 round에서 초과 생성한 토큰도 포함한다.

## 전체 모델 정확도

HF BF16을 기준으로 9개 프롬프트에서 최대 192개 생성 토큰을 비교했다. 입력이 1토큰인 hello는 prompt decode KL 구간이 없으므로 KL 평균은 나머지 8개에 대해 계산한다. 생성 토큰 검증에는 9개 모두 포함한다. 기존 gate의 단일 토큰 입력 집계 오류를 수정했으며, 임계값은 KL 비율 ≤ 2와 strong mismatch 0개 그대로다.

| 정밀도 | 평균 decode KL vs HF | KL / BF16 | strong / near-tie | free-run 완전 일치 | 판정 |
|---|---:|---:|---:|---:|---|
| bf16 | 0.00091986 | 1.000 | 1 / 6 | 5/9 | 대조군 |
| int8 | 0.00094118 | 1.023 | 0 / 9 | 5/9 | 통과 |
| nvfp4 | 0.04764956 | 51.801 | 30 / 49 | 0/9 | 실패 |

BF16 대조군에서도 prose_1k 생성 중 strong mismatch 1개(HF margin 6.5)가 관측됐다. NVFP4는 strong mismatch 30개, 모든 프롬프트에서 free-run 출력 불일치가 관측됐다. 이 gate는 지정된 프롬프트의 수치 검증이며 일반적인 언어 모델 품질 벤치마크는 아니다.

## Decode 성능

| 정밀도 | 방식 | 프롬프트별 중앙값 범위 (ms/token) | tok/s 범위 |
|---|---|---:|---:|
| bf16 | plain | 35.925–36.075 | 27.7–27.8 |
| int8 | plain | 20.894–20.918 | 47.8–47.9 |
| int8 | mtp1 | 12.325–14.584 | 68.6–81.1 |
| int8 | mtp2 | 9.105–13.119 | 76.2–109.8 |
| int8 | mtp3 | 7.414–13.382 | 74.7–134.9 |
| nvfp4 | plain | 22.436–22.463 | 44.5–44.6 |
| nvfp4 | mtp1 | 12.945–14.537 | 68.8–77.3 |
| nvfp4 | mtp2 | 9.518–13.020 | 76.8–105.1 |
| nvfp4 | mtp3 | 7.923–12.942 | 77.3–126.2 |

## MTP 출력 검증

| target | plain과 다른 출력 case | same-prefix strong mismatch 합 |
|---|---:|---:|
| int8 | 6/15 | 0 |
| nvfp4 | 15/15 | 7 |

MTP 수락률과 출력 일치는 전체 모델 실행의 진단이다. 헤드 단독 정확도 검증은 요청 범위에 포함하지 않았다. Same-prefix 진단은 출력 불일치 시 최대 64개 forced position을 확인하며 k별 합계는 독립 오류 수가 아니다.

## 자료와 재현

- 세부 decode·prefill·커널 결과: [report.md](report.md). 원본 JSON의 반복별 시간과 실행 인자를 보존했다.
- Native GEMM 18 shape 및 저장·재로딩 검증은 [이전 release 자료](../qwen3_5_27b_nvfp4_g4_20261028/README.md)에 있다.
- `source_manifest.json`은 실제 업로드한 코드의 SHA-256이다. 로컬 감독 재개는 원격 worker를 중복 실행하지 않았다.
- 첫 실행은 세션 소실로 최종 결과를 회수하지 못했다. 이 폴더는 중간 결과를 주기적으로 회수한 두 번째 세션의 결과만 포함한다.
- session: `nvfp4-d364828a34`; checkpoint revision: `08a9f4929e2033ddd5e81f0ab8b31072e7437e1a`; INT8 revision: `e41d469d00ef99639f592c1cfc5ab6e0a9db75bf`; 원본 revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- 두 로컬 감독 프로세스의 최종 회수·종료 시도가 겹쳤다. 늦은 회수 요청의 오류는 `supervisor_race.json`에 보존했다. 원격 worker의 finished 기록, 최종 ZIP과 측정 파일의 바이트 일치, 서버 active session 0개를 근거로 실행 완료를 판정했다. 세션 종료 상태는 `lifecycle.json`과 `cleanup.json`에 보존했다.

```bash
python3 scripts/run_qwen3_5_nvfp4_colab.py \
  --checkpoint-repo Jakemu/Qwen3.5-27B-nvfp4-mtp \
  --output experiments/nvfp4/checkpoint-evaluation --budget-minutes 120
```

## 정확도 실패 원인 분석 (저장 자료·소스 분석)

현재 확인한 직접 현상은 MTP를 사용하지 않는 target 모델의 수치 오차 증가다. W8A16 INT8에서 W4A4 NVFP4로 가며 가중치와 활성값을 동시에 바꿨으므로, 기존 측정만으로 W4와 A4의 기여를 분리할 수 없다. 아래 원인 후보는 추가 GPU ablation으로 확정한 결과가 아니다.

| 프롬프트 | 입력 토큰 | NVFP4 KL | KL/BF16 | strong / near-tie | 첫 출력 불일치 위치 (0부터) |
|---|---:|---:|---:|---:|---:|
| short | 5 | 0.024284 | 53.0 | 3 / 3 | 40 |
| hello | 1 | 구간 없음 | — | 2 / 10 | 2 |
| korean | 20 | 0.036969 | 89.7 | 5 / 15 | 1 |
| math | 43 | 0.152850 | 444.3 | 0 / 2 | 31 |
| chat | 23 | 0.054745 | 131.0 | 3 / 7 | 71 |
| chat_long | 32 | 0.035066 | 72.4 | 4 / 7 | 9 |
| prose_1k | 1057 | 0.003245 | 24.1 | 3 / 0 | 152 |
| code_1k | 1000 | 0.062859 | 12.7 | 3 / 2 | 30 |
| code_1_5k | 1500 | 0.011177 | 63.5 | 7 / 3 | 0 |

### 자료에서 확인되는 것

- Teacher-forced 1,728개 위치 중 79개(4.57%)가 HF 생성 토큰과 달랐고 strong 분류는 30개다. 동일한 정답 prefix를 넣어도 차이가 있으므로 free-run에서 한 번 갈라진 뒤 누적된 출력 차이만으로 설명되지 않는다.
- 짧은 math 입력(43토큰)에서도 KL 0.15285가 나왔다. 512토큰 보정 길이보다 긴 입력만의 문제로 볼 수 없다. code_1_5k는 첫 생성 토큰부터 다르므로 생성 중 오류 누적 이전의 prefill도 영향을 받는다.
- 64개 layer의 큰 Linear 256개를 모두 W4A4로 바꿨다. 여기에는 GatedDeltaNet의 in_proj_qkvz 48개와 out_proj 48개가 포함된다. 해당 projection의 오차가 recurrent state와 후속 layer로 전달될 가능성이 있으나 layer별 오차를 측정하지 않아 지배적인 위치는 모른다.
- 보정은 UltraChat train_sft 앞 128개 샘플, 각 최대 512토큰의 전체 forward다. 생성 decode를 보정에 포함하지 않는다. 보정 대표성·활성값 범위 초과가 후보지만, clipping/underflow 비율을 기록하지 않았으므로 이를 원인으로 확정할 수 없다.
- 저장된 global scale 512개(256 module × input/weight)는 모두 양수·유한값이다. 값이 정상 범위라는 사실만으로 보정 품질이나 clipping 유무가 검증되지는 않는다.

### 구현 문제와 양자화 오차의 구분

기존 native smoke는 무작위 입력과 가중치로 18개 shape를 검사했다. 복원된 FP4 값의 FP32 matmul 대비 relative L2 오차는 약 0.164–0.173%였고, eager·compile·CUDA Graph 출력도 일치했다. 이는 기본 GEMM/layout 문제의 가능성을 낮추지만 실제 checkpoint의 보정·packing·입력 분포까지 검증하지는 않는다. 저장·재로딩 logits 일치도 저장된 경로의 일관성 검사이며 BF16 대비 정확도 검사는 아니다.

코드의 scale 연결은 s_A=amax_A/(6×448), s_W=amax_W/(6×448) 형태의 global dequantization scale을 받아 quantize에는 역수, GEMM alpha에는 s_A×s_W를 전달한다. 이 방향은 [ModelOpt 0.47.0 NVFP4 scale 정의](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/modelopt/torch/quantization/qtensor/nvfp4_tensor.py)와 [FlashInfer 0.6.6 quantizer API](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.6/flashinfer/fp4_quantization.py)에 부합한다. 정적 코드 확인에서 단순한 reciprocal 방향 오류의 근거는 찾지 못했다. 실제 layer 입력을 고정해 native 출력과 독립 dequantized reference를 비교하는 검증은 아직 필요하다.

### 정확도 판정 자체의 주의점

검증기는 정답 토큰을 HF.generate의 cache 기반 경로에서 가져오고, mismatch의 margin은 HF가 전체 시퀀스를 다시 계산한 logits에서 가져온다. 두 HF 경로의 top-1이 다르면 강한 오류 분류가 왜곡될 수 있다. BF16 prose의 strong 1개(margin 6.5)는 이 자체 일관성 검사를 우선해야 하는 이유다. 이 가설은 아직 재실행으로 검증하지 않았다. 다만 prose를 제외해도 NVFP4 strong은 27개이고, prompt decode KL도 크게 증가하므로 prose 한 사례를 제외하는 것만으로 실패가 사라지지는 않는다.

### 원인을 확정할 다음 비교 순서

1. HF cached teacher-forced logits를 정답 토큰·margin·KL에 일관되게 사용하고 HF cached vs full-sequence 차이를 별도 기록한다. BF16 대조군도 함께 재검사한다.
2. 실제 checkpoint의 동일 layer 입력에서 native W4A4, 독립 복원 FP4 matmul, ModelOpt fake-quant 출력을 비교한다. 서로 다르면 packing/scale/rounding/실행 경로부터 조사한다.
3. 동일 prompt·prefix에서 BF16, 복원 W4A16, W16A4, W4A4를 비교해 가중치·활성값·결합 오차를 분리한다. W4A16/W16A4는 원인 분석용 reference이며 native serving 성능으로 해석하지 않는다.
4. layer별 local output 오차와 recurrent-state 오차, activation clipping/underflow를 기록한다. 민감한 attention projection을 BF16으로 복구하는 구성과 MLP-only W4A4를 비교한다.
5. clipping이나 분포 차이가 확인되면 decode 토큰을 포함한 보정, 데이터 다양화, global-scale sweep을 비교한다. 단순히 보정 샘플 수만 늘리는 것으로 해결된다고 가정하지 않는다.

현재 결론: target의 W4A4 변환에 따른 수치 오차가 가장 유력한 설명이지만, A4가 주원인인지 W4·특정 layer·fake/native 차이가 주원인인지는 미확정이다. MTP 헤드 단독 분석은 수행하지 않았다.
