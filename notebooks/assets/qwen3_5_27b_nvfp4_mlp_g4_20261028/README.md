# Qwen3.5-27B MLP-only NVFP4 정확도 실험

**오차는 줄었지만 정확도 gate는 통과하지 못했다.** Attention을 BF16으로 복구한 MLP-only 구성에서, 전체 W4A4 대비 평균 prompt-decode KL은 약 49.6% 감소하고 strong teacher-forced mismatch는 29개에서 12개로 감소했다.

| 구성 | 평균 decode KL vs HF | KL / BF16 | Strong / near-tie | Free-run 완전 일치 | 판정 |
|---|---:|---:|---:|---:|---|
| bf16 | 0.00163033 | 1.000 | 0 / 6 | 6/9 | 대조군 |
| nvfp4 | 0.04707331 | 28.874 | 29 / 49 | 0/9 | 실패 |
| nvfp4-mlp | 0.02373341 | 14.557 | 12 / 33 | 0/9 | 실패 |

## 변경한 것과 검증 조건

- 원본 Qwen/Qwen3.5-27B BF16에서 새로 보정·저장했다. FP4 가중치를 BF16으로 복원해 원본으로 사용하지 않았다.
- MLP gate/up/down 128개 Linear만 native W4A4로 저장했다. ModelOpt NVFP4_DEFAULT_CFG에 scope 제외 규칙을 적용해 attention input/weight quantizer를 비활성화했다. 큰 attention projection 128개는 저장 파일의 dtype까지 BF16으로 확인했다.
- MTP는 기존 group-128 INT8, embedding·lm_head·작은 gate·norm·state는 기존 정밀도를 유지했다. 이 검증은 MTP를 호출하지 않는 target 전체 모델 decode다.
- UltraChat train_sft revision과 128개 입력의 token hash가 이전 전체 NVFP4 보정과 정확히 같다. 샘플당 최대 512토큰이다. MLP-only 범위로 스케일은 다시 보정했다.
- 가중치 파일은 **29,623,903,776 bytes (27.589 GiB)**다. 기존 전체 NVFP4 17.931 GiB보다 약 9.66 GiB 늘었다. runtime buffer는 포함하지 않는다.
- 저장·재로딩 전후 짧은 smoke prompt의 마지막 logits가 정확히 일치했다.
- G4: RTX PRO 6000 Blackwell Server Edition SM120, 94.97 GiB VRAM, torch 2.11.0+cu130. FlashInfer 0.6.6 CUTLASS, ModelOpt 0.47.0. 한 세션에서 세 구성을 비교했다.
- 기존과 동일한 9개 평가 프롬프트, 각 192개 생성 토큰, 최대 마지막 32개 prompt decode 위치를 검사했다. hello는 입력 1토큰이라 prompt decode KL 구간이 없으며 생성 검증에는 포함한다. 평균 KL은 나머지 8개 prompt의 평균이다.
- 모든 구성은 fla-delta, projection fusion, compile-reduce-overhead를 사용했다. gate는 KL/BF16 ≤ 2 및 strong mismatch 0개 그대로다. Strong은 해당 HF cached logits의 top-1과 top-2 margin이 0.5보다 큰 불일치다.

## HF 기준 경로 수정

프로토콜은 `hf-cached-greedy-v2`다. HF prompt decode logits는 동일 prefix를 prefill하고 한 토큰씩 진행해 얻는다. 생성은 전체 prompt를 prefill한 뒤 raw cached greedy forward로 192개 토큰을 만들고, 그 동일 logits에서 margin을 구한다. 정답은 cached generation에서 가져오면서 margin은 전체 시퀀스 재계산에서 가져오던 혼용을 제거했다. Reference token ID와 margin도 accuracy.json에 보존했다.

이번 BF16 대조군의 strong mismatch는 0개다. 이전 실행의 BF16 strong 1개는 이번에 재현되지 않았지만 이 한 번의 재실행만으로 이전 오류의 원인을 확정하지는 않는다. 기준 경로가 달라졌으므로 과거 KL 51.8배/strong 30개와 현재 MLP-only 값을 직접 비교해 scope 효과로 해석하면 안 된다. 위 표는 같은 새 기준으로 재평가한 전체 NVFP4와 비교한다.

## 프롬프트별 비교

| 프롬프트 | BF16 KL | 전체 W4A4 KL | MLP-only KL | strong 전체 → MLP-only | 첫 free-run 불일치 전체 → MLP-only (0부터) |
|---|---:|---:|---:|---:|---:|
| chat | 0.000475 | 0.045898 | 0.021250 | 2 → 2 | 29 → 47 |
| chat_long | 0.000589 | 0.035575 | 0.024591 | 4 → 0 | 9 → 9 |
| code_1_5k | 0.000139 | 0.015762 | 0.005013 | 5 → 3 | 0 → 0 |
| code_1k | 0.010444 | 0.058865 | 0.052709 | 4 → 2 | 30 → 30 |
| hello | 구간 없음 | 구간 없음 | 구간 없음 | 1 → 1 | 2 → 11 |
| korean | 0.000477 | 0.037591 | 0.024386 | 4 → 0 | 1 → 1 |
| math | 0.000522 | 0.154225 | 0.030796 | 0 → 0 | 31 → 77 |
| prose_1k | 0.000168 | 0.003321 | 0.001574 | 3 → 1 | 152 → 174 |
| short | 0.000228 | 0.025350 | 0.029549 | 6 → 3 | 40 → 57 |

## 해석

양자화 범위를 MLP로 제한하자 평균 KL과 strong mismatch가 줄었으므로, 이 모델에서는 attention까지 W4A4로 양자화한 구성보다 정확도가 개선됐다. 그러나 short처럼 KL이 증가한 prompt도 있고, 모든 prompt에서 free-run 출력은 HF와 달랐다. Attention을 BF16으로 유지하는 것만으로 요구한 정확도에 도달하지는 못했다.

Attention 정밀도 복구와 그 범위에 맞춘 MLP 활성값 스케일 재보정을 함께 한 결과다. 이를 W4와 A4의 개별 원인 분리 또는 특정 attention layer의 민감도 측정으로 해석하지 않는다. 다음 원인 분리에는 같은 MLP 입력·가중치에서 W4A16 reference와 W4A4, native 출력과 독립 복원 reference의 비교가 필요하다. 이 실행에서는 추가 ablation을 진행하지 않았다.

## 자료와 재현

- accuracy.json: 세 구성의 27개 prompt 결과, gate, HF reference token ID와 margin.
- scope_validation.json: native MLP 128개, BF16 attention 128개, 동일 calibration token hash 확인.
- export/: 보정 token hash, 모듈별 scale, 저장·재로딩 검증과 가중치 크기.
- environment.json / source_manifest.json: 실제 장비·패키지와 업로드 코드 해시. analysis_provenance.json에는 현재 측정 코드 일치와 보고서 생성기 해시를 기록했다.
- lifecycle.json / done.json / cleanup.json: worker 완료, 결과 회수와 세션 종료. 최종 ZIP의 원시 JSON과 로컬 파일이 바이트 단위로 일치한다. 서버 active session 0개를 확인했다.
- 처리 속도 측정과 MTP 헤드 단독 검증은 수행하지 않았다. 관련 CPU 회귀 테스트는 18개 통과했다.

```bash
python3 scripts/run_qwen3_5_nvfp4_colab.py --mlp-only \
  --output experiments/nvfp4/mlp-only --budget-minutes 120
```
