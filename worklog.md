# 작업 기록

## 2026-10-19 — Qwen3.5-9B decode용 weight-only INT8

**목표:** Linear 가중치를 int8로 저장해(W8A16) A100에서 Qwen3.5-9B의 batch 1 decode를 빠르게 만든다. 동시에 양자화 기준(gate)을
통과해야 한다. 기준은 HF 대비 평균 decode KL이 같은 실행의 bf16 모델 KL의 2배 이하이고, teacher-forced strong 불일치가 없는 것이다.

속도 숫자는 모두 다음 조건이다: A100-SXM4-40GB(Colab), `fla-delta:compile-reduce-overhead`, INT8 구성은 inductor max-autotune,
1024-token 프롬프트, decode 128 step 평균. `int8`은 lm_head를 bf16으로 두고 `int8+lm_head`는 lm_head까지 양자화한다.
원시 결과는 `experiments/qwen3_5_int8/`에 있다(git 제외). 노트북이 쓰는 파일은 `notebooks/assets/qwen3_5_9b_int8_a100_20261019/`에
복사했고, `notebooks/qwen3_5_9b_benchmark_tables.ipynb`의 §5에 표로 정리했다.

### Per-channel INT8 (8d49c31, 8204734, 0e9c374)

- `infer_opt.quant`: int8 가중치와 출력 채널당 bf16 scale 1개. `int8_linear`는 inductor가 int8 → bf16 변환을 batch 1 matmul 안으로
  합치도록 작성했다(max-autotune과 coordinate descent를 켰을 때만 합쳐진다).
- 빠르다: 9.80 ms/token, lm_head까지 8.92 (bf16 15.04).
- **gate 실패:** KL 비율 2.14 (lm_head 포함 2.47), strong 불일치 1개 (`prose_1k`, HF margin 1.75).
- 행 단위 클리핑 탐색은 실제 9B MLP 가중치의 오차를 3.7%밖에 줄이지 못했다. 128열마다 scale을 두면 54% 줄었다.

### 그룹 단위 scale (85abb68)

- `quantize_int8(weight, group_size)`: scale은 `[out, in // group_size]`, 그룹마다 absmax, 클리핑 없음.
- 그룹 128은 lm_head를 bf16으로 두면 **gate 통과**(두 번 실행에서 비율 1.55, 1.57, strong 불일치 없음).
  `int8+lm_head`는 경계선이다(한 번 통과, 한 번은 같은 `prose_1k` token에서 실패).
- 하지만 scale이 합(reduction) 방향으로 바뀌어서 출력에 한 번 곱하는 방식이 안 된다. inductor가 합치지 못했다:
  역양자화 후 matmul 14.3 ms/token, reduction 형태(`_group_gemv`) 13.4로 bf16보다 조금 빠를 뿐이었다.
- Qwen3.5-0.8B(로컬 RTX 2080 SUPER)는 그룹 128로 통과하지 못한다(비율 2.1–2.4). 그룹 32에서 통과한다(1.89).

### Triton GEMV (c48a4f8)

- `infer_opt/quant_kernels.py`: `w8a16_group_gemv`. `torch.library.triton_op`으로 등록해서 torch.compile이 커널 하나로 유지하고
  CUDA graph가 캡처한다. `int8_linear`는 그룹 scale을 쓰는 CUDA 1 token 입력을 여기로 보낸다. prefill과 CPU는 여전히 `F.linear` 전에
  역양자화한다.
- v1은 scale을 원소마다 읽었다(모든 k에 대해 `scale[n, k // 128]`). A100에서 약 580 GB/s에 막혀 bf16보다 나을 게 없었다.
  v2는 K를 그룹 하나씩 돌며 행마다 scale을 한 번 읽는다. 큰 shape에서 0.9–1.4 TB/s.
- compile된 decode step 안에서 `triton.autotune`이 나쁜 설정을 골랐다: 12.75 ms/token, **lm_head 포함 25.4**(bf16보다 느림).
  그래서 실행 설정을 고정했다:
  - `scripts/tune_qwen3_5_group_gemv.py`가 모든 Linear shape의 실제 가중치로 설정을 훑는다.
  - 최적 설정은 규칙을 따른다(`launch_config`): warp 8개, grid가 약 256–512 program이 되도록 BLOCK_N. 여섯 shape 중 다섯에서
    측정 최적과 같다.
- 결과(같은 세션): **group-128 int8 10.79 ms/token, int8+lm_head 10.15**. bf16 대비 KL 0.00051 / 0.00057
  (per-channel 0.00085 / 0.00092). 이 커널로 잰 gate: 비율 1.89 / 1.95, strong 불일치 없음.

### 시도했다가 뺀 것

- **down_proj(4096 × 12288)의 split-K:** 행이 적고 길어서 program 수가 적다. K를 두 번째 grid 축으로 나누면 실제 가중치에서
  1회당 48.8 µs vs 50.6 µs였고, decode 전체로는 차이가 없었다(둘 다 10.15). 넣으면서 UNROLL > 1 설정이 느려지기도 해서 되돌렸다.
- **클리핑:** 위 참고(3.7%).

### 알게 된 것

- **gate의 분모가 흔들린다.** bf16 대조의 HF 대비 KL이 실행마다 0.00041–0.00046으로 움직여서, group-128의 KL 자체는
  0.00071–0.00077로 일정한데 비율은 1.55–1.89로 움직였다. 2.0까지의 여유는 첫 실행에서 보인 것보다 작다.
- **단일 shape 마이크로벤치마크는 스크립트마다 숫자가 다르다.** 같은 설정이 한 스크립트에서는 57.3 µs, 다른 스크립트에서는
  50.6 µs였다. 돌려 쓰는 가중치 사본 수 차이 때문이다. 같은 실행 안에서만 비교하고, 결정은 decode 전체로 한다.
- Colab의 dist-packages에 `tests` 패키지가 있어서 저장소의 `tests`(`__init__.py` 없음)를 가린다. 원격에서는 pytest 전에
  `touch tests/__init__.py`를 한다.
- 로컬 GPU(sm75)는 Triton/inductor에서 bf16을 기본 지원하지 않는다. 정확성 확인용으로만 쓰고 속도는 A100에서 잰다.

### 남은 일

- per-channel이 아직 1.2 ms/token 빠르다(lm_head 포함 8.92 vs 10.15). GEMV가 step의 약 7.4 ms를 쓰고, 여섯 shape 전체에
  퍼져 있다(0.4–1.4 TB/s). 한 레이어가 아니라 커널 전반을 개선해야 줄어든다.
- 그룹 scale의 prefill은 matmul 전에 가중치 전체를 역양자화한다(lm_head 포함 시 peak +1.5 GiB, prefill 속도는 재지 않음).
  타일 안에서 역양자화하는 Triton GEMM이면 이 복사를 없앨 수 있다.
- bf16 대조에 덜 민감한 gate(KL 절대 상한, 또는 bf16 여러 번 실행)면 통과/실패가 더 안정적일 것이다.

## 2026-10-20 — 양자화 모델을 Hugging Face에 올리고 Colab 노트북으로 실행

- `infer_opt.quant.save_quantized` / `prepare_quantized`와 `Qwen3_5ForCausalLM.from_quantized_dir`: int8 가중치·scale·나머지 bf16
  텐서를 safetensors shard로 저장하고, `quantization.json`(어느 Linear가 int8인지, group 크기)으로 모듈 구조를 다시 만들어 GPU로 바로 읽는다.
  `from_pretrained_dir`처럼 rope 버퍼(`inv_freq`)도 bf16으로 맞춰야 logits가 같았다.
- `scripts/quantize_qwen3_5.py`: bf16 checkpoint → group-128 INT8(lm_head bf16) → 저장 → 다시 읽어 logits가 완전히 같은지 확인 →
  `--push-to-hub`로 model card와 함께 업로드. Colab A100에서 실행해
  [`Jakemu/Qwen3.5-9B-int8-g128`](https://huggingface.co/Jakemu/Qwen3.5-9B-int8-g128)(공개, Apache-2.0, 6 shard 10.34 GiB)에 올렸다.
- `notebooks/qwen3_5_9b_int8_colab.ipynb`(`scripts/build_qwen3_5_int8_colab_notebook.py`로 생성): Colab A100에서 infer-opt를 고정 커밋으로
  받고 Hub에서 모델을 내려받아 streaming 생성과 decode 속도 측정을 한다.

## 2026-10-20 — T4(fp16)에서 실행

T4(Turing, compute capability 7.5)는 bf16이 없고 메모리가 약 15 GiB다. 원래 bf16 모델은 가중치만 16.7 GiB라 올라가지 않지만,
int8 checkpoint는 10.34 GiB라 그대로 들어간다. 그래서 새 checkpoint를 만들지 않고 같은 checkpoint를 fp16으로 읽게 했다.

- `from_quantized_dir(..., dtype=torch.float16)`: int8 가중치는 그대로 두고 scale·embedding·norm·lm_head 등 실수 텐서만 바꾼다.
- **scale 범위:** 5400만 개 중 최소 1.56e-5. fp16 정규 최솟값(6.1e-5)보다 작은 것이 1866개(0.0035%)로 정밀도만 조금 줄고,
  0이 되는 것은 없다.
- **정확도(A100, 9개 프롬프트, HF bf16 기준):** `validate_qwen3_5_decode_modes.py --dtype float16 --int8-from Jakemu/Qwen3.5-9B-int8-g128`.
  올린 checkpoint를 fp16으로 돌린 HF 대비 KL 0.00056, strong 불일치 0. bf16으로 돌렸을 때(0.00071–0.00077)보다 오히려 작다.
  fp16 대조도 0.00028로 bf16 대조(0.00041)보다 HF에 가깝다(fp16은 가수 비트가 더 많다). KL 비율은 1.98로 기준 2에 가깝지만,
  이것은 분모(fp16 대조)가 작아졌기 때문이다.
- **속도(Colab T4):** compile + CUDA graph decode 42.1 ms/token(`torch` kernels), 41.9(`fla-delta`); 노트북으로 다시 돌렸을 때 44.3. peak 12.2–12.7 GiB.
  T4 대역폭(320 GB/s)으로 잡은 하한 약 28 ms의 67%. FLA는 첫 prefill에서 커널 컴파일에 96초가 걸려서 T4 노트북은 `torch` kernels를 쓴다.
- `notebooks/qwen3_5_9b_int8_colab_t4.ipynb`: 같은 builder(`--target t4`)로 만든다. 원래 모델과의 비교 절은 T4에 들어가지 않아 뺐다.

## 2026-10-20 — Qwen3.5-27B weight-only INT8

**목표:** 9B와 같은 방법(group-128 INT8, lm_head bf16, Triton GEMV, `fla-delta:compile-reduce-overhead`)을 Qwen3.5-27B에 적용한다.
Qwen3.5에는 32B가 없고 dense 모델은 27B가 가장 크다(35B-A3B는 MoE라 따로 구현이 필요하다). 구조는 9B와 같아서 모델 코드는
바꾸지 않았다: hidden 5120, intermediate 17408, 64층(linear 48 / full 16), linear attention head k 16 / v 48, vocab 248320,
lm_head가 embedding과 **묶여 있지 않다**(9B는 묶여 있다). bf16 checkpoint는 51.7 GiB다.

숫자는 모두 Colab **A100-SXM4-80GB**(high-mem 런타임, RAM 167 GiB), torch 2.11, transformers 5.20.2, 1024-token 프롬프트, decode 128 step
평균이다. 원시 결과는 `experiments/qwen3_5_27b_int8/a100_80gb_20261020/`(git 제외), 노트북용 사본은 `notebooks/assets/qwen3_5_27b_int8_a100_20261020/`.

### 결과

| | ms/token | peak GiB | KL vs bf16 |
|---|---:|---:|---:|
| bf16 | 37.65 | 50.9 | – |
| group-128 int8, lm_head bf16 | **25.93** | 28.6 | 0.00038 |
| group-128 int8 + lm_head | 25.22 | 31.7 | 0.00036 |

- bf16 대비 1.45배(9B는 1.40배). 대역폭 하한(`count_qwen3_5_bytes.py`, 2039 GB/s)은 bf16 25.3, int8 lm_head bf16 13.6 ms다.
- **gate 통과(4/4):** HF 대비 평균 decode KL이 같은 실행 bf16의 0.85 / 0.53배(제자리 양자화 두 번), 저장한 checkpoint를 다시 읽은
  실행에서 0.72배, Hub에서 다시 받은 checkpoint로 0.95배. strong 불일치 0. `int8+lm_head`도 0.67 / 1.00배로 통과했다.
- checkpoint: 14 shard 27.80 GiB, 다시 읽은 logits가 완전히 같다. [`Jakemu/Qwen3.5-27B-int8-g128`](https://huggingface.co/Jakemu/Qwen3.5-27B-int8-g128)에 올렸다.
  int8 모델은 약 29 GiB라 A100 40GB에도 들어간다.

### 바꾼 것

- **GEMV 실행 설정 규칙(`launch_config`):** 9B에서 만든 규칙이 27B shape에서 나빴다. sweep(`tune_qwen3_5_group_gemv.py`)에서
  규칙 설정의 decode는 28.1 ms/token, shape별 최적은 24.8(lm_head 포함 int8). 특히 down_proj(5120 × 17408)가 (16, 4)로 104 µs, (8, 1)로 72 µs.
  두 모델의 sweep을 보면 좋은 타일은 (BLOCK_N, UNROLL) = (8, 1|8), (16, 2), (32, 4)뿐이고 (16, 4)는 한 번도 없다. BLOCK_N 경계를
  `cdiv(n, 640)`으로 늦추고 BLOCK_N 16에는 UNROLL 2를 쓰게 바꿨다. 9B 여섯 shape 모두 측정 최적과 같고, 27B 여덟 shape는 최적의 3.5% 안이다.
- **`from_pretrained_dir`:** shard를 CPU를 거치지 않고 `device`로 바로 읽는다(52 GiB를 host RAM에 올리지 않는다).
- **제자리 양자화의 GPU 메모리 부족(9B에서는 드러나지 않았다):**
  - compile된 bf16 decode step이 bf16 가중치를 참조하고 있어서, 양자화가 Linear를 바꿔도 bf16 가중치가 풀리지 않았다(52 + 24 GiB).
    `bench_qwen3_5_int8_decode.py`와 `validate_qwen3_5_decode_modes.py`에서 양자화 전에 `torch._dynamo.reset()`을 부른다.
  - gate 스크립트의 HF 기준값 `full[:, split:T]`가 view라서 프롬프트마다 전체 logits(fp32, 1 token에 약 1 MB)가 남아 있었다. `.clone()`으로 바꿨다.
- model card의 측정 절을 `--repo`별 dict(`MEASURED`)로 옮기고, 남겨 두는 delta-rule gate 폭을 config에서 읽는다(9B 32, 27B 48).
- Colab 노트북 builder에 `--target a100-27b`(`notebooks/qwen3_5_27b_int8_colab.ipynb`). 원래 모델과의 비교 절은 52 + 28 GiB가 80 GB에
  들어가지 않아서 뺐다.

### 알게 된 것

- **gate 평균은 프롬프트 하나(`code_1k`)가 정한다.** bf16 대조의 KL 자체가 0.0106으로 다른 프롬프트(0.0002–0.002)보다 한 자릿수 크고,
  int8은 실행마다 0.0038–0.0089로 움직인다. 이 프롬프트를 빼면 int8/bf16 비율은 0.86–0.94다(제자리 양자화 두 번). 이번에는 bf16 대조가 실행마다 같았다(0.00187).
- 27B decode에서 GEMV는 약 19.3 ms(shape별 최적 기준)이고 나머지 약 5.5 ms는 GEMV 밖이다. 층당 약 86 µs로 9B와 같다.
  64층이라 비중이 커졌다(약 22%).
- GEMV 대역폭은 큰 shape에서 1.25–1.46 TB/s, lm_head만 1.72 TB/s다. MLP(GEMV 시간의 약 65%)를 lm_head 수준으로 올리면 약 2.5 ms가 줄어든다.

### 남은 일

- GEMV 밖의 층당 약 86 µs(norm, conv, delta rule, attention과 작은 kernel들): 27B에서는 decode의 약 22%다.
- MLP GEMV 대역폭(gate/up 1.45, down 1.25 TB/s). gate_proj와 up_proj를 GEMV 하나로 합치는 것도 후보다.
- 노트북을 Colab A100에서 끝까지 실행해 기록 속도를 확인한다.

## 2026-10-20 — Qwen3.5-27B INT8 decode 최적화: GEMV kernel, 투영 합치기, decode attention

**목표:** 27B int8 decode(25.93 ms/token)와 대역폭 하한(lm_head bf16 13.6 ms)의 차이를 줄인다. 체크포인트(`Jakemu/Qwen3.5-27B-int8-g128`)는
바꾸지 않는다.

숫자는 모두 Colab A100-SXM4-80GB, torch 2.11, `fla-delta:compile-reduce-overhead`, inductor max-autotune, 1024-token 프롬프트다.
원시 결과는 `experiments/qwen3_5_27b_int8/a100_80gb_20261020_opt/`(git 제외), 노트북용 사본은
`notebooks/assets/qwen3_5_27b_int8_opt_a100_20261020/`.

### 결과

`bench_qwen3_5_int8_decode.py --fuse`(128 step 평균, 같은 실행의 bf16 포함):

| | 이전 ms/token | 이후 ms/token | peak GiB | KL vs bf16 |
|---|---:|---:|---:|---:|
| bf16 | 37.65 | 35.17 | 50.9 | – |
| group-128 int8, lm_head bf16 | 25.93 | **20.72** | 28.8 | 0.00030 |
| group-128 int8 + lm_head | 25.22 | 20.31 | 31.7 | 0.00033 |

- int8은 bf16 대비 1.70배(이전 1.45배). 하한 13.6 ms의 1.52배다(이전 1.91배).
- 단계별(같은 하니스, Hub checkpoint, `decode_ab.json`): 26.32 → attention·conv 24.68 → 투영 합치기 23.32 → 새 GEMV kernel 20.56
  → Triton decode attention 20.69(실행 간 차이 ±0.15 ms 안. profile의 busy 시간으로는 20.90 → 20.70).

- 9B(`Jakemu/Qwen3.5-9B-int8-g128`, 같은 하니스)도 같은 코드로 9.45 → 7.34 ms/token이다. 9B gate는 다시 돌리지 않았다
  (옛 코드와의 teacher-forced KL 1.1e-4).
- **gate 통과(3/3, 고정한 프롬프트):** Hub checkpoint int8 KL 비율 0.86, 제자리 양자화 int8 1.07 / int8+lm_head 0.67.
  strong 불일치 0. bf16 대조 평균 KL 0.00193(이전 0.00187).

### 어디에 시간을 썼나 (profile, CUDA graph 안 kernel 시간의 합)

CUDA graph로 돌리면 GPU가 step 시간의 99%를 kernel로 채운다(26.07 / 26.2 ms). 그래서 kernel 시간의 합이 곧 ms/token이다.

| | 이전 | 이후 |
|---|---:|---:|
| int8 GEMV | 20.08 ms (400개) | 15.99 ms (256개) |
| lm_head (bf16, inductor) | 1.38 | 1.39 |
| full attention 16층 | 1.75 | 0.23 |
| 나머지 (norm, delta rule, 작은 kernel) | 2.9 | 3.1 |
| 합 (busy) | 26.07 | 20.70 |

### 바꾼 것

- **GEMV kernel (`infer_opt/quant_kernels.py`):** 타일을 `[BLOCK_N, BLOCK_K/16, 16]`로 바꿨다. thread 하나가 한 group 안의 int8 16개를
  16-byte load 한 번으로 읽고 `q * x`를 register에서 더한 뒤 group scale을 한 번 곱한다(이전에는 원소마다 `x * scale`을 곱했다).
  `x`와 scale의 pointer도 `q`와 같은 3차원 모양으로 만들어야 세 load가 같은 layout을 쓴다. 2차원으로 읽으면 Triton이 매 step 부분합을
  shared memory로 옮겼다(`bar.sync`). 실행 설정은 모든 27B shape에서 (BLOCK_N 1, BLOCK_K 1024, 2 warps)가 최적이라 규칙이 상수가 됐다.
  - 27B shape에서 weight만 읽는 kernel(대역폭 천장)의 2–8% 안이고, per-channel scale kernel만큼 빠르다:
    gate+up(합침) 113.7 → 106.5 µs(천장 102.6), down 73.6 → 59.1(54.3), out_proj 28.2 → 22.5(21.0).
  - scale index는 GROUP이 BLOCK_K를 나눌 때 loop 밖에서 계산한다. 매 step 계산하면 9B gate+up에서 13% 느렸다.
- **투영 합치기 (`Qwen3_5ForCausalLM.fuse_projections`):** 같은 입력을 읽는 Linear를 행으로 쌓는다. gate+up(MLP), q+k+v(full attention),
  in_proj_qkv+in_proj_z(GatedDeltaNet). group scale은 행마다라서 int8도 그대로 쌓인다. 로드 뒤에 하는 변환이라 체크포인트 형식은
  같고 state-dict 키만 바뀐다(합친 뒤에는 저장하지 않는다). launch, ramp, tail이 줄어 gate+up 두 GEMV 139 µs가 113 µs가 됐다.
  bench/gate 스크립트에 `--fuse`를 추가했다.
- **decode attention:** `repeat_kv`가 cache를 query head마다 복사하고, SDPA kernel은 query 행(1개)으로 병렬화해 층당 ~110 µs였다
  (4.5 MB cache). query head를 KV head별로 묶어
  - CPU/기타: fp32 reduction (inductor kernel 3개, 층당 39 µs),
  - CUDA: split-K Triton kernel(`infer_opt/attention_kernels.py`, 64 key씩, online softmax 부분합을 PyTorch로 합친다). kernel 7.8 µs, 합치기 ~7 µs.
- **decode conv:** 한 token의 depthwise conv를 kernel 폭 4의 내적으로 바꿨다(`F.conv1d` kernel 대신 inductor가 합친다).
- **gate 프롬프트 고정:** `code_1k`, `code_1_5k` 프롬프트를 살아 있는 `infer_opt/qwen3_5.py`에서 잘라 쓰고 있었다. 이번에 그 파일의
  docstring을 고치자 프롬프트가 바뀌어 bf16 대조 KL이 0.0106 → 0.154로 뛰었다(모델 문제가 아니다). 9e5c3bd의 텍스트를
  `scripts/code_prompt_source.txt`로 고정했다(앞 12 000자는 32c1509부터 같아서 이전 INT8 gate 실행과 같은 프롬프트다).
  `D_gate_hub_fused.*`는 바뀐 프롬프트로 돈 실행이라 비교에 쓰지 않는다.

### 알게 된 것

- **I2F 변환은 병목이 아니었다.** int8 → fp32를 bit trick(`0x4B000000 | (q ^ 0x80)` − 8388736)으로 바꾼 kernel은 두 형태 모두 더 느렸다.
  비용은 원소마다의 scale 곱과 shared memory 왕복이었다.
- **scale group을 키우는 것(group 256, per-channel)은 새 kernel에서 이득이 없다.** 옛 kernel에서는 per-channel이 12–20% 빨랐고 group 256은
  ±5%였다. 새 kernel에서는 group 128이 둘과 같은 속도라 정확도만 잃는다. scale 바이트는 weight의 1.6%다.
- weight만 읽는 kernel도 이 접근 방식으로는 ~1.6 TB/s(89 MB shape)라, GEMV는 이제 천장(15.0 ms)에 가깝다. 하한 12.1 ms와의 차이는 kernel마다의
  ramp/tail이다.

### 남은 일

- norm(5120 폭 한 행): inductor의 persistent reduction 기준(1024)을 넘어 loop 두 번짜리 reduction이 된다. 층당 2개, 6–12 µs씩 약 0.8 ms.
- in_proj_a/b(48행 bf16)가 norm과 합쳐진 reduction kernel(15 µs, 16개).
- Colab 노트북은 아직 9e5c3bd를 고정하고 있고 `fuse_projections()`를 부르지 않는다. 새 kernel은 T4에서 재 보지 않았다.
