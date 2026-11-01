# Qwen3.5-27B INT8 1-token GEMV: CUDA C++ vs Triton (A100 80GB, 2026-11-01)

group-128 W8A16 GEMV를 CUDA C++(`infer_opt/csrc/w8a16_gemv.cu`, `infer_opt/quant_cuda.py`)로 새로 작성해서
기존 Triton 커널(`infer_opt/quant_kernels.py`의 `w8a16_group_gemv`)과 비교했다.
Colab A100-SXM4-80GB, torch 2.11.0+cu130, triton 3.6.0, nvcc 13.0. 모든 숫자는 한 세션 안에서 측정했다.

## 결론

- **커널 단위로는 차이가 없다.** 양쪽 다 구성을 sweep해서 shape별 최적끼리 비교하면 모델 전체 합이 CUDA 16.05 ms,
  Triton 16.07 ms(0.999배)다. 두 커널 모두 A100 HBM 대역폭 상한(1.40–1.77 TB/s)에 붙어 있다.
- **decode 전체로는 CUDA가 0.2 ms/token(1%) 빠르다.** int8 19.94 → 19.74, int8+lm_head 19.31 → 19.13.
  앞뒤로 잰 Triton 두 번의 차이는 0.003–0.013 ms라서 측정 잡음보다 크다. 다만 decode A/B에서 CUDA는 shape별
  최적 구성(`gemv_first_run.json`의 `best`)을 썼고 Triton은 규칙 하나 (1, 1024, 2)를 썼다. 0.2 ms의 대부분은
  구성 선택 차이로 보인다. Triton도 shape별 최적 구성을 쓰면 down_proj(2, 1024, 2), lm_head(2, 1024, 2),
  q+k+v(4, 1024, 4)에서 규칙보다 빠르다.
- 정확도는 같다. bf16 대비 KL은 Triton 0.00036–0.00061, CUDA 0.00033–0.00048이다. Triton끼리도 이 정도 폭으로 흔들린다.
  CUDA와 Triton 출력 차이는 상대 오차 2.2e-5 이하다(bf16 반올림 수준).

## 1. decode A/B (`decode_ab.json`)

`bench_qwen3_5_int8_decode.py --repo Qwen/Qwen3.5-27B --kernels fla-delta --group-size 128 --fuse --autotune
--gemv-backends triton,cuda,triton --cuda-configs gemv_first_run.json` (prompt 1024, 128 token, compile-reduce-overhead)

| 구성 | ms/token | KL vs bf16 | argmax 일치 (48 step) |
|---|---|---|---|
| bf16 | 35.07 | 0.00034 | 1.00 |
| int8 [triton] | 19.94 | 0.00061 | 1.00 |
| int8 [cuda] | **19.74** | 0.00033 | 1.00 |
| int8 [triton] 재측정 | 19.94 | 0.00056 | 0.98 |
| int8+lm_head [triton] | 19.31 | 0.00044 | 1.00 |
| int8+lm_head [cuda] | **19.13** | 0.00048 | 0.98 |
| int8+lm_head [triton] 재측정 | 19.32 | 0.00036 | 1.00 |

graph break 0, peak 메모리는 두 backend가 같다(int8 28.8 GiB, int8+lm_head 31.7 GiB).

## 2. 커널 마이크로벤치 (`gemv_sweep.json`, GPU 단독 사용)

`bench_int8_gemv_cuda.py --repo Qwen/Qwen3.5-27B --fuse --triton-sweep`. 랜덤 int8 가중치를 L2보다 큰 복사본으로
돌려가며 CUDA graph로 재생했다. 단위는 us이고, GB/s는 int8 가중치와 scale 바이트 기준이다.

| N x K | 개수 | bf16 cuBLAS | Triton 최적 (구성) | CUDA 규칙 (2, 32, 1) | CUDA 최적 (구성) | CUDA GB/s |
|---|---|---|---|---|---|---|
| 34816 x 5120 (gate+up) | 64 | 243.7 | 106.6 (1, 1024, 2) | 109.7 | 108.0 (4, 32, 1) | 1676 |
| 5120 x 17408 (down) | 64 | 123.0 | 58.5 (2, 1024, 2) | 58.8 | 57.0 (1, 128, 1) | 1589 |
| 16384 x 5120 (qkv+z) | 48 | 122.1 | 52.8 (1, 1024, 2) | 53.0 | 53.0 (2, 32, 1) | 1609 |
| 5120 x 6144 (out/o_proj) | 64 | 46.5 | 22.6 (1, 1024, 2) | 22.8 | 22.8 (2, 32, 1) | 1404 |
| 248320 x 5120 (lm_head) | 1 | 1575.4 | 746.8 (2, 1024, 2) | 730.9 | 729.1 (4, 32, 1) | 1771 |
| 14336 x 5120 (q+k+v) | 16 | 112.1 | 48.6 (4, 1024, 4) | 47.7 | 47.7 (2, 32, 1) | 1563 |
| **모델 합 (ms)** | | 35.68 | **16.07** | 16.28 | **16.05** | |

CUDA 규칙 / Triton 최적 = 1.013, CUDA 최적 / Triton 최적 = 0.999.

### 측정 함정: 처음 재는 변형이 느리게 나온다

이 실행의 "Triton 규칙" 열(shape마다 맨 처음 측정)은 121.6 / 66.5 / 59.2 / 24.9 / 837.4 / 52.7 us였다.
그런데 같은 구성 (1, 1024, 2)를 같은 가중치로 sweep 안에서 다시 재면 34816 x 5120에서 106.6 us가 나왔다(10–14% 차이).
첫 실행(`gemv_first_run.json`)에서 "CUDA가 Triton보다 12% 빠르다"는 결과가 나온 것도 이 함정 때문이다.
그 실행에서는 27B 가중치 다운로드도 동시에 돌고 있었다. 지금 스크립트는 첫 Triton 측정 전에 측정하지 않는 패스를
한 번 돌린다(이 json들은 수정 전 스크립트로 만들었다).

## 3. CUDA 커널 구조와 구성 sweep

- 블록 = `THREADS` 스레드, `ROWS` 출력 행. 스레드가 16 int8을 16-byte streaming 로드(`__ldcs`)로 읽고,
  x 16개는 `__ldg` 2번으로 읽는다. 16개 곱-합은 fp32 레지스터에서 하고, 그룹 scale은 16 weight마다 한 번 곱한다.
  `UNROLL`개 chunk의 로드를 먼저 모두 발행한 뒤 계산하고, 마지막에 warp shuffle과 smem으로 리덕션한다.
- 48개 구성(ROWS 1/2/4 × THREADS 32/64/128/256 × UNROLL 1/2/4/8) 중 (2, 32, 1)이 모든 shape에서 최적의 4% 이내였다.
  1 warp, 행 2개, 스레드당 16-byte 로드 2개를 동시에 띄우는 구성이다. UNROLL을 키우거나 스레드를 늘리면 모든 shape에서 느려졌다.
  처음 기본값 (1, 64, 4)는 1.32배 느렸다.
- 로컬 RTX 2080 SUPER(sm75, 448 GB/s)에서는 양쪽 다 약 460 GB/s로 차이가 없고, 이 GPU에서 (2, 32, 1)은 최적의 1.4배다.

## 파일

- `decode_ab.json`: decode A/B 원본 (`bench_qwen3_5_int8_decode.py` 출력)
- `gemv_sweep.json`: GPU 단독 마이크로벤치. CUDA 48개 구성 전체와 Triton grid 최적 top5 포함
- `gemv_first_run.json`: 다운로드와 동시에 돈 첫 마이크로벤치(첫 측정 bias 있음). decode A/B의 `--cuda-configs`로 쓰였다
