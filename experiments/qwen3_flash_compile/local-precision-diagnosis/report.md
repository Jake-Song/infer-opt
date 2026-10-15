# Qwen3 BF16 compile 수치 오차: 로컬 원인 분석

## 결론과 확인 범위

**BF16 연산 fusion에서 중간 반올림을 생략하는 수치 경로 변화가 실제 Qwen3 가중치로 재현됐다.**
RMSNorm과 RoPE의 중간 BF16 변환을 보존하면 로컬 첫 레이어 K 생성 경로가 eager와 정확히 일치한다.
이것은 기존 A100 결과에서 attention 계산 전의 첫 레이어 K부터 오차가 발생한 현상과 일관된다.

다만 **A100 전체 모델 실패의 원인을 모두 확정하거나 수정 효과를 검증한 결과는 아니다.**
새 A100 80GB 할당은 `Service Unavailable`로 실패했고, 사용자가 추가 GPU 없이 로컬 분석만
진행하도록 선택했다. 따라서 cache·attention backend의 독립 영향과 수정 후 전체 logits/KV는
A100에서 재검증하지 못했다. 성능 수치도 추가하지 않았다.

| 구분 | 기존 실패 실행 | 이번 로컬 재현 |
|---|---|---|
| 장치 | A100-SXM4-80GB | CPU |
| PyTorch | 2.11.0+cu130 | 2.14.0+cu130, CPU 실행 |
| 연산 범위 | Qwen3-14B 전체 모델 | 첫 레이어 RMSNorm·RoPE·K 생성 경로 |
| 가중치 | BF16 전체 checkpoint | 동일 revision의 필요한 BF16 가중치 일부 |
| 입력 | 128개 고정 토큰 + 127 decode steps | 저장된 같은 128개 입력 토큰 |

## 기존 A100 기록에서 알 수 있는 것

입력 128토큰, 출력 128토큰의 teacher-forced 검증에서:

- 첫 prefill부터 logits 최대 절대오차 0.15625, relative L2 약 0.01349.
- 첫 레이어 K의 최대 절대오차 0.5, relative L2 약 0.003137, 허용 오차 초과 원소 7개.
- 첫 레이어 V의 최대 절대오차 0.001953125, relative L2 약 0.002785.
- 전체 logits 128/128개 검사와 KV 480개 중 426개 검사가 실패했다.
- cache 길이는 255로 같았고, 같은 토큰을 넣었을 때 단계별 argmax는 같았다.

첫 레이어 K는 해당 레이어의 attention 연산 전에 계산된다.
따라서 attention mask/backend의 차이만으로 첫 레이어 K의 오차를 설명할 수 없다.
다만 최종 logits의 최대 relative L2 약 0.25309 전체를 이 원인 하나로 설명했다고 주장하지 않는다.

## 분리한 연산과 실제 측정

같은 HF revision `40c069824f4251a91eefaf281ebe4c544efd3e18`에서 다음만 HTTP byte range로 읽었다.

- 입력에 등장하는 embedding 30행
- 첫 레이어 input RMSNorm weight
- 첫 레이어 K projection weight
- 첫 레이어 K RMSNorm weight

총 **10,803,456 bytes**이며 전체 14B 모델은 로컬로 내려받지 않았다.
BF16 storage와 Qwen3 원본 RMSNorm·RoPE 공식을 유지했다.
CPU matmul 및 cos/sin 구현은 GPU 구현과 다를 수 있으므로 CPU 결과를 A100 수치로 해석하지 않는다.
생성 코드는 C++이며, A100의 Triton 생성 코드를 회수한 것은 아니다.

| CPU 연산 | 기본 compile 최대 절대오차 | 기본 compile 초과 원소 | cast 보존 최대 절대오차 | cast 보존 초과 원소 |
|---|---:|---:|---:|---:|
| Input RMSNorm | 0.001953125 | 0 | 0 | 0 |
| K RMSNorm | 1.0 | 0 | 0 | 0 |
| RoPE | 0.03125 | 0 | 0 | 0 |
| 첫 레이어 K 전체 경로 | 1.0 | **6** | **0** | **0** |

초과 원소는 기존과 동일한 `abs(actual-reference) > 0.02 + 0.02*abs(reference)` 기준이다.
각 단일 연산이 기준을 통과하더라도, 연산을 연결한 K 생성 경로는 기준을 넘을 수 있었다.
검증 기준은 완화하지 않았다.

## 생성 코드로 확인한 원인

Qwen3 RMSNorm은 FP32로 정규화한 뒤 BF16으로 변환하고, 학습된 weight를 곱한다.

```text
eager:
    normalized_fp32 → BF16 반올림 → weight 곱 → BF16 반올림

기본 compile:
    normalized_fp32 → FP32로 weight 곱 → BF16 반올림
```

기본 생성 코드에서는 정규화 다음의 변환이 `auto tmp12 = (tmp11);`이라는 no-op이고,
weight 곱 뒤에만 BF16 변환이 남아 있다.
`emulate_precision_casts=True`에서는 weight 곱 전에 FP32 → BF16 → FP32 변환이 추가된다.

- [기본 K RMSNorm 생성 코드](cpu-key_rmsnorm-default.txt)
- [cast 보존 K RMSNorm 생성 코드](cpu-key_rmsnorm-preserve_casts.txt)

RoPE에서도 곱셈 결과를 각각 BF16으로 반올림한 뒤 더하는 eager 경로와,
중간값을 FP32로 유지하다 마지막에 BF16으로 변환하는 compile 경로가 달랐다.
eager에서 중간 cast만 의도적으로 제거한 제어 실험은 기본 compiled RoPE와 **오차 0**으로 일치했다.
반대로 cast를 보존한 compiled RoPE는 원래 eager RoPE와 **오차 0**으로 일치했다.
같은 회전·계수·입력에서 BF16 변환 시점만 바꾼 것이므로, 이 로컬 차이는 cast 생략에 귀속할 수 있다.

가중치 저장 형식은 BF16 그대로다. FP32 중간값을 BF16으로 되돌리는 시점이 달라진 것이다.
PyTorch 2.11 소스도 Inductor가 fused 저정밀 연산 사이의 cast를 생략하고,
`emulate_precision_casts`가 이를 보존하는 옵션이라고 설명한다.
[PyTorch 2.11 config](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/config.py)

## 코드와 남은 검증

벤치마크에는 `--emulate-precision-casts`를 추가했다. 이 옵션이 켜지면:

```python
torch.compile(
    model.forward,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False, "emulate_precision_casts": True},
)
```

실패 시 eager StaticCache와도 같은 full-logit/KV 검증을 수행하도록 보완했다.
GPU 진단 스크립트에는 eager DynamicCache / eager StaticCache / 기본 compile /
cast 보존 compile / 동일 attention backend 대조군을 마련했지만, **이번에는 할당 실패로 실행하지 않았다.**
추가 GPU는 사용하지 않았다.

확정되지 않은 부분:

1. A100/PyTorch 2.11에서 cast 보존 옵션으로 전체 128단계 logits와 모든 KV가 기준을 통과하는가.
2. DynamicCache와 StaticCache의 attention 자동 선택 차이가 별도 오차를 만드는가.
3. 수정한 설정의 prefill·decode·TTFT·ITL은 얼마인가.

## 재현과 원자료

저장된 부분 가중치를 사용하면 네트워크나 GPU 없이 재현한다.

```bash
.venv/bin/python scripts/reproduce_qwen3_bf16_casts_cpu.py \
  --output experiments/qwen3_flash_compile/local-precision-diagnosis \
  --fixture experiments/qwen3_flash_compile/a100-20261015-static-compile-auto/measurements.json
```

- [CPU 수치 원자료](cpu_diagnosis.json)
- [부분 가중치 byte range·SHA-256](partial_weights.json)
- [원본 shard tensor metadata](shard-header.json)
- [기존 A100 검증 원자료](../a100-20261015-static-compile-auto/measurements.json)
- [새 A100 할당 실패와 최종 세션 상태](../a100-20261015-compile-diagnosis/lifecycle.json)

로컬 스크립트를 다운로드 없이 한 번 더 실행해 동일 결과를 확인했다.
