# G4 물리적 하한 계산

이번 실행의 모델 형식·현재 target/verify/draft 호출 구조를 유지한다는 조건으로 계산했다.
새 GPU 실행이나 DRAM counter 측정 없이, 고정 checkpoint의 config/quantization metadata,
현재 소스와 저장된 100개 timing sample·수락 통계를 사용했다. 원시 측정값은 `../results/`에 그대로 보존한다.

**MTP-off는 캐시를 매우 유리하게 허용해도 약 2.00 ms/token의 가중치 전송 하한이다.**
코드 k=3은 같은 보수적 가정에서 1.04 ms/token이다. 상태·KV를 포함한 이상적 streaming 시나리오는
각각 2.30 ms/token과 1.34~1.92 ms/token이다. 이 값은 달성 가능한 성능을 보장하지 않는다.

## 하드웨어와 경계

- NVIDIA RTX PRO 6000 Blackwell Server Edition, 실측 VRAM 94.9708 GiB, SM120.
- GDDR7 명목 bandwidth **1597 GB/s = 1.597 TB/s**, FP32 FMA-equivalent peak **120 TFLOP/s**. 실제 지속 bandwidth는 측정하지 않았다.
- Server Edition 사양을 사용했다. Workstation Edition의 1792 GB/s를 대입하지 않았다.
- [NVIDIA Server Edition 사양](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).
- [NVIDIA RTX Blackwell architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)는 full GB202의 L2 128 MiB, 총 L1/shared 24 MiB, register file 48 MiB를 기재한다.
- 여기서는 그 합 200 MiB보다 큰 **256 MiB**를 모든 데이터가 자유롭게 재사용할 수 있는 on-chip budget으로 허용했다. 실제 서버의 L2 용량이나 hit rate를 조회한 값이 아니다.
- 모든 값은 GB=10^9 bytes, MiB=2^20 bytes. GPU 명목 peak를 사용하는 낙관적 하한이다.

## 한 토큰이 사용하는 weights

모델은 hidden=2048, 40 layers(DeltaNet 30 + full attention 10), 256 experts 중 top-8, expert intermediate=512다.
INT8 group-128 weight 하나의 저장비는 1 byte + BF16 scale 2/128 byte, 즉 **1.015625 bytes/parameter**다.
LM head는 BF16 `[248320, 2048]`, 1,017,118,720 bytes다. 전체 35B weight를 매 토큰 읽는 것으로 계산하지 않았다.

| 구분 | 활성 parameter 수 | weight + scale bytes |
| --- | ---: | ---: |
| Target: 공통 INT8 projection/shared expert | 1,405,091,840 | 1,427,046,400 |
| Target: 선택된 8개 expert | 1,006,632,960 | 1,022,361,600 |
| MTP 1-layer: 공통 INT8 projection/shared expert | 38,797,312 | 39,403,520 |
| MTP 1-layer: 선택된 8개 expert | 25,165,824 | 25,559,040 |
| BF16 vocabulary projection 1회 | 508,559,360 | 1,017,118,720 |

Target의 INT8 weights/scales + BF16 LM head 합은 **3,466,526,720 bytes**다.
단순 streaming 시간은 `3,466,526,720 / 1.597e12 = 2.17065 ms`.
이것만으로는 캐시 재사용을 뺀 엄밀한 외부 메모리 하한이라고 할 수 없어 아래에 별도로 계산한다.

## 1. 캐시를 허용한 보수적인 가중치 전송 하한

현재 소스는 target body+LM head를 실행한 뒤 MTP body+LM head를 k회 실행한다.
첫 MTP body는 k+1개 row를 처리하고 이후 k-1회는 한 row씩 처리하지만,
첫 MTP의 출력층은 선택한 hidden 한 row에만 적용한다. 따라서 **LM head 호출은 round당 k+1회**다.
각 단계 시작 때 on-chip에 이미 존재할 수 있는 데이터는 C 이하이므로,
그 단계가 읽는 서로 다른 weight bytes가 W이면 외부에서 적어도 `max(W-C,0)`를 가져와야 한다.
이 계산은 모든 verify row가 같은 8개 expert를 공유한다고 가정하고, 실제로 불가능할 수도 있는 최선의 캐시 재사용을 허용한다.
읽기만 세며 output/state/KV write와 계산·동기화 비용은 제외한다.

```text
C = 256 MiB
Wtarget = 3,466,526,720 bytes
Wdraft  = 1,082,081,280 bytes  # MTP INT8 + scales + BF16 vocabulary
Qround >= max(Wtarget-C, 0) + k * max(Wdraft-C, 0)
Ttoken >= Qround / 1.597e12 / actual_emitted_tokens_per_round
```

실제 계산은 각 raw sample의 `rounds / (emitted_tokens-1)`로 정규화한 후 중앙값을 취한다.
기존 benchmark와 동일하게 마지막 round의 초과 생성분을 포함한다. MTP-off는 step당 계산이다.

## 2. 상태·KV를 포함한 이상적 streaming 시나리오

- Target의 나머지 BF16 weights는 52,290,816 bytes. embedding은 입력 row만 센다.
- DeltaNet FP32 state는 30 layers × 32 heads × 128 × 128 × 4 = **62,914,560 bytes**다.
- MTP-off는 state read+write, 즉 125,829,120 logical bytes. verify는 1회 read + (k+1)개 snapshot write + commit read/write = `(k+4) × state_bytes`.
- Verify row 간 expert 집합이 모두 같아 weight가 완전히 재사용되는 경우와, 각 row가 서로 다른 expert를 고르는 경우를 계산했다. 실제 routing overlap은 이 실행에 저장되지 않았다.
- KV는 실제 source의 64-key chunk read, target 10개 full-attention layer와 MTP layer를 반영했다. 대표 context는 prompt length+128이며, round별 위치 trace가 없어 정확한 평균 DRAM traffic이 아니다.
- 상태의 cache hit, store forwarding, 불필요 snapshot의 cache residency에 따라 실제 DRAM bytes는 논리 bytes보다 작을 수 있다. 따라서 이 시나리오 범위는 엄밀한 물리 하한의 구간이 아니다.
- activation 중간 결과, logits buffer, partial attention merge, 반복 scale load, transaction amplification 등은 완전하게 세지 않아 이상적인 참고값이다.
- 이 범위의 오른쪽도 실제 latency의 상한이 아니다.

## 실측과 계산 비교

모든 시간은 ms/token. `†`는 5회 모두 MTP-off와 생성 토큰이 달랐던 조건이다.
출력이 다른 조건의 시간/하한 비율을 유효한 speedup으로 해석하지 않는다.

| 프롬프트 | 모드 | 실측 | 캐시 허용 가중치 하한 | 이상적 streaming 시나리오 |
| --- | --- | ---: | ---: | ---: |
| chat_long | off | 4.6488 | 2.0026 | 2.2865 |
| chat_long | k=1† | 4.1505 | 1.3247 | 1.6259–1.9719 |
| chat_long | k=2† | 3.7890 | 1.1968 | 1.5055–2.0253 |
| chat_long | k=3† | 3.5182 | 1.1266 | 1.4419–2.0700 |
| code_1k | off | 4.6190 | 2.0026 | 2.2988 |
| code_1k | k=1 | 3.9930 | 1.2756 | 1.5726–1.9058 |
| code_1k | k=2 | 3.4206 | 1.0816 | 1.3660–1.8358 |
| code_1k | k=3 | 3.2078 | 1.0401 | 1.3359–1.9158 |
| prose_1k | off | 4.5625 | 2.0026 | 2.2996 |
| prose_1k | k=1† | 4.1976 | 1.3595 | 1.6764–2.0315 |
| prose_1k | k=2† | 3.6560 | 1.1639 | 1.4703–1.9759 |
| prose_1k | k=3† | 2.8154 | 0.9001 | 1.1563–1.6580 |
| math | off | 4.5494 | 2.0026 | 2.2865 |
| math | k=1 | 4.0097 | 1.2953 | 1.5898–1.9281 |
| math | k=2 | 3.3302 | 1.0623 | 1.3363–1.7977 |
| math | k=3† | 2.8848 | 0.9278 | 1.1874–1.7046 |
| korean | off | 4.5445 | 2.0026 | 2.2865 |
| korean | k=1† | 4.3189 | 1.4130 | 1.7343–2.1034 |
| korean | k=2† | 4.4853 | 1.4574 | 1.8335–2.4665 |
| korean | k=3† | 4.8707 | 1.6063 | 2.0557–2.9512 |

코드 k=3은 round당 3.3947토큰을 emit했다. streaming 모델의 round당 traffic은
7.2424~10.3862 GB, bandwidth 시간은 4.5350~6.5036 ms/round, 토큰당 1.3359~1.9158 ms다.
이 중 출력층 호출 네 번만 4.0685 GB = 2.5476 ms/round = 약 0.7504 ms/token이다.
따라서 이상적인 수락을 가정해도 MTP의 비용을 MTP-off 시간의 1/4로 단순 계산할 수 없다.

코드 k=3 실측 3.2078 ms/token은 해당 streaming 시나리오 시간의 1.67~2.40배다.
이는 모델 기반 headroom이며, 보장된 개선 폭이나 측정된 DRAM 효율이 아니다.

## 산술량 하한과 해석

현재 G4의 selected-expert kernel은 `_project`이고 INT8를 FP32로 변환한 뒤 `x*q*s + acc`를 계산한다.
G4에서는 A100 전용 packed path를 선택하지 않는다. 이 커널은 DP4A/INT8 Tensor Core 연산이 아니다.
소스의 `enable_fp_fusion=False`를 기준으로 weight product당 FP32 multiply 2개 + add 1개를 센다.
120 TFLOP/s FMA peak를 60T scalar ops/s로 환산하면, routed experts의 이 산술만에 대한 하한은
plain **0.0503 ms/token**, code k=3 **0.2089 ms/round, 0.0615 ms/token**이다.
conversion·reduction·address generation·routing·sigmoid·scheduler·occupancy 비용은 제외한다.
이것은 전체 compute floor가 아니라 partial arithmetic floor다.

이상적인 throughput 모델에서는 memory transfer 하한이 위 산술량 하한보다 훨씬 크다.
그러나 실제 커널이 GDDR7 bandwidth에 포화됐거나, 현재 격차가 전부 memory 비효율이라는 뜻은 아니다.
실제 병목을 가리려면 route overlap, DRAM/L2 bytes, instruction issue, occupancy/stall counter가 필요하다.
이번 계산에서는 새로운 GPU를 할당하거나 counter 실험을 수행하지 않았다.

## 재현과 입력 검증

```bash
python3 scripts/moe_compiled_g4/lower_bound.py artifacts/qwen35_moe_compiled_g4_20261112
```

config와 quantization metadata는 실측에 기록된 pinned revision에서 내려받았고 SHA-256을 기존 checkpoint hash와 대조한다.
`tensor-metadata.json`은 해당 config로 meta-device 모델을 만들고 BF16 cast 및 `prepare_quantized`를 적용해 수집한 shape/dtype다.
가중치는 다운로드하지 않았다. 추정 payload는 37,087,512,832 bytes, 원시 shard 크기는 header 포함 37,087,658,536 bytes다.
계산 JSON에 raw sample·metadata·관련 source·calculator SHA-256을 저장한다.
메타데이터 생성은 `.venv/bin/python`에서 `Qwen3_5TextConfig.from_json`,
`with torch.device("meta"): Qwen3_5ForCausalLM(config, mtp=True).to(torch.bfloat16)`,
`prepare_quantized(model, quantization)`을 실행한 뒤 `state_dict()`의 name/shape/dtype/bytes를 기록했다.

[계산 JSON](lower-bound.json), [입력 metadata](inputs/), [원시 실측 보고서](../README.md),
[계산 코드](../../../scripts/moe_compiled_g4/lower_bound.py).
