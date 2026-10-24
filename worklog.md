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
- 9B Colab 노트북은 아직 04af44f를 고정하고 있다. 새 kernel은 T4에서 재 보지 않았다.
- 27B 노트북은 661f694를 고정하고 `fuse_projections()`를 부르게 다시 만들었다. Colab A100-SXM4-40GB에서 끝까지 실행했고
  decode 24.13 ms/token, peak 29.1 GiB였다(80GB의 20.72와 대역폭 비율로 맞는다).

## 2026-10-24 — Qwen3.5-27B INT8 decode: 넓은 norm을 persistent reduction으로

**목표:** 남은 일의 첫 항목. decode의 RMSNorm은 5120 폭 한 행이라 inductor의 persistent reduction 기준(INNER 1024,
`torch/_inductor/choices.py`의 `should_use_persistent_reduction`)을 넘고, 행을 두 번 읽는 loop kernel을 program 하나가 돌린다.
체크포인트(`Jakemu/Qwen3.5-27B-int8-g128`)는 바꾸지 않는다.

숫자는 모두 Colab A100-SXM4-80GB(high-mem), torch 2.11, `fla-delta:compile-reduce-overhead`, max-autotune, `fuse_projections()`,
1024-token 프롬프트다. 원시 결과와 실험 스크립트(`norm_variant.py`)는 `experiments/qwen3_5_27b_int8/a100_80gb_20261024_norm/`(git 제외),
노트북용 사본은 `notebooks/assets/qwen3_5_27b_int8_norm_a100_20261024/`.

### 결과

변형마다 따로 프로세스와 inductor cache를 쓰고 Hub checkpoint를 읽었다. 128 step × 3회의 중앙값(회차 차이 ±0.01 ms),
kernel 시간은 CUDA graph 20 step profile의 step 평균이다. KL은 base 실행의 logits에 같은 token을 먹인 teacher-forced 값이다.

| 변형 | ms/token | 한 행 norm 128개 | norm·mm reduction (lm_head 포함) | KL vs base |
|---|---:|---:|---:|---:|
| base (661f694와 같은 코드) | 20.31 | 0.87 ms (`red_`) | 1.55 ms | – |
| base 다시 | 20.31 | 0.66 ms (`red_`) | 1.77 ms | 7.0e-5 |
| 한 행만 persistent | 20.24 | 0.40 ms (`per_`) | 1.95 ms | 1.6e-4 |
| 폭 8192 이하 전부 persistent | 19.91 | 0.40 ms | 1.67 ms | 2.2e-4 |
| `triton.multi_kernel = 1` | 20.18 | 64개만 `per_` | 1.63 ms | 1.6e-4 |
| **커밋한 규칙 (폭 8192, 행 1024 이하)** | **19.86** | 0.41 ms | 1.60 ms | 1.8e-4 |

- 같은 세션 A/B로 **20.31 → 19.86 ms/token(−0.45 ms)**. 하한 13.6 ms의 1.46배(이전 1.49배).
- `bench_qwen3_5_int8_decode.py --fuse --autotune --group-size 128`(제자리 양자화, worklog 표와 같은 하니스): bf16 35.11, int8 **19.99**,
  int8+lm_head 19.54 ms/token(2026-10-20에는 35.17 / 20.72 / 20.31). 이 차이(−0.73)는 세션이 달라서 크게 나온다.
  같은 코드의 Hub checkpoint가 그날 20.69, 오늘 20.31이었다.
- **gate 통과(고정 프롬프트, Hub checkpoint):** int8 KL 비율 **0.58**(bf16 대조 0.00200, int8 0.00115), strong 불일치 0.
  이전 같은 gate는 0.86. bench의 int8 KL vs bf16 sliced-eager는 0.00059(이전 0.00030), int8+lm_head는 0.00030(이전 0.00033).

### 바꾼 것

- **`infer_opt/inductor_choices.py`:** `InductorChoices`를 상속한 `WidePersistentReductions`가 inductor 기본 규칙에 더해, INNER reduction이
  폭 8192 이하이고 행 1024 이하이면 persistent로 만든다. `wide_persistent_reductions()` context가 `V.set_choices_handler`와
  `inductor_choices_class` 설정을 함께 건다.
- **`FixedShapeDecoder.step`:** compile 모드에서 이 context 안에서 compile된 함수를 부른다(첫 호출과 재compile이 그 안에서 일어난다).
  bf16 대조와 int8이 같은 규칙을 쓴다.
- 테스트: 5120 폭 한 행은 `persistent_reduction`, 2048행은 loop로 남는지(`tests/test_qwen3_5.py`).

### 알게 된 것

- **행 수도 봐야 한다.** 한 행만 persistent로 바꾸면, inductor가 norm과 합친 in_proj_a/b reduction(48행, 5120 폭)이 31개 4.6 → 16.9 µs,
  16개 7.4 → 12.3 µs로 느려져서 norm에서 번 0.47 ms를 거의 다 잃었다. 그 kernel도 persistent로 하면 7.3 / 3.7 µs다.
  lm_head(248 320행)는 persistent가 35 µs 느려서 행 상한으로 뺐다.
- **choices는 스레드마다 처음 한 번 만들어 캐시된다.** `inductor_choices_class` 설정만 바꾸면 그 전에 compile이 한 번이라도 있었을 때
  효과가 없다. 반대로 handler만 바꾸면 FX graph cache key에 들어가지 않아서 기본 규칙으로 compile한 kernel을 다시 쓸 수 있다.
  그래서 둘 다 건다. 로컬에서 조건을 고친 뒤에도 같은 uuid의 cache가 옛 kernel을 돌려준 적이 있다.
- `multi_kernel`(기준 16배, 런타임에 둘을 재서 고름)은 norm의 절반만 persistent로 골라 20.18 ms였다.
- base를 두 번 돌리면 norm kernel 시간이 0.87 / 0.66 ms로 다르고 KL도 7e-5 나온다. max-autotune과 coordinate descent가 실행마다 다른
  설정을 골라서다. 변형 사이의 KL 1.6–2.2e-4는 그 2–3배다.

### 남은 일

- GEMV 밖 kernel 약 3.3 ms(이번 변경 후). in_proj_a/b(bf16 48행)를 int8 qkvz GEMV에 합치는 것, norm을 GEMV 앞부분에 넣는 것이 후보다.
- 27B 노트북은 아직 661f694를 고정한다. 이 변경으로 다시 고정할지 정해야 한다. 9B와 T4에서는 재 보지 않았다.

## 2026-10-24 — Qwen3.5 MTP self-speculative decoding (greedy)

**목표:** int8 27B의 decode(19.4–20.0 ms/token)는 GEMV가 이미 천장 근처라, 한 번 읽는 weight로 여러 token을 얻는 쪽이 남은 지렛대다.
모든 Qwen3.5 체크포인트에 있는 MTP head(`mtp.*`, decoder layer 1개 + fc + norm 3개, embedding과 lm_head는 target과 공유)를 draft로 쓰고,
greedy로만 검증한다(출력은 target의 greedy와 같아야 한다).

숫자는 모두 Colab A100-SXM4-80GB(high-mem), torch 2.11, `fla-delta:compile-reduce-overhead`, max-autotune, `fuse_projections()`,
256 token 생성, 같은 세션에서 잰 plain fixed-shape decoder(baseline)와 비교했다. bf16 `Qwen/Qwen3.5-27B`를 MTP head와 함께 읽고 제자리
양자화(group 128)했다(Hub int8 체크포인트에는 MTP head가 없다). 원시 결과는 `notebooks/assets/qwen3_5_27b_int8_spec_a100_20261024/`,
로그와 로컬 실험 스크립트는 `experiments/qwen3_5_27b_int8/a100_80gb_20261024_spec/`(git 제외).

### 결과

**기준 구성은 int8+mtp**(target과 MTP head는 int8, lm_head는 bf16)다. lm_head는 공개 체크포인트와 같이 bf16으로 둔다:
MTP head의 정밀도는 acceptance만 바꾸고 출력은 바꾸지 않지만, lm_head를 양자화하면 target 모델 자체가 바뀐다. lm_head int8은 이전 gate를
통과했으나(KL 비율 0.67) 기본이 아니라 선택 사항이다.

k=3(모든 prompt에서 가장 빠름), ms/token(baseline 대비 배수):

| prompt | int8 | **int8+mtp (기준)** | int8+mtp+lm_head (선택) | round당 token (기준) |
|---|---:|---:|---:|---:|
| baseline | 19.92–20.02 | 19.87–19.96 | 19.30–19.39 | 1 |
| prose_1k | 8.82 (2.27x) | **8.72 (2.29x)** | 8.07 (2.40x) | 4.00 |
| math | 9.09 (2.20x) | **8.96 (2.23x)** | 8.24 (2.35x) | 3.89 |
| code_1k | 10.55 (1.90x) | **10.55 (1.89x)** | 9.99 (1.94x) | 3.31 |
| chat_long | 11.78 (1.69x) | **11.07 (1.80x)** | 10.64 (1.81x) | 3.16 |
| korean | 15.00 (1.33x) | **15.99 (1.25x)** | 14.30 (1.36x) | 2.18 |

- int8 = target만 int8, int8+mtp = MTP head도, int8+mtp+lm_head = lm_head(target과 draft가 공유)도 int8.
- 기준 구성의 korean은 k=2가 조금 더 빠르다(15.81, 1.26x). lm_head를 int8로 하면 0.4–0.7 ms/token(5–8%) 더 빨라진다.
- round 시간: 기준 구성 k=1/2/3에서 28.2–28.5 / 32.0–32.3 / 34.9–35.0 ms(step 19.9), lm_head int8이면 27.3 / 30.1 / 32.3 ms(step 19.3).
  lm_head는 draft마다 한 번씩 읽으므로 int8의 이득은 k가 클수록 크다. MTP head int8은 시간·acceptance 모두 거의 그대로다.
- 한국어는 acceptance가 낮다(k=1에서 round 중 31%가 draft 거부). 영어 prose는 k=3에서 모든 round가 4 token을 받았다.
- baseline과 token이 갈라지는 prompt가 있다(chat_long 70번째, korean, code_1k의 일부 k). 0.8B fp16에서 같은 현상을 보면 갈라진 자리의
  top-2 logit 차이가 정확히 0이었다(T-token forward와 1-token forward의 반올림 차이로 동률이 뒤집힘). 27B에서는 아직 확인하지 않았다.

### 바꾼 것

- **`Qwen3_5MTP`, `from_pretrained_dir(..., mtp=True)`, `mtp_hidden()`:** MTP head는 `h' = norm(layer(fc([norm(emb(x_{j+1})),
  norm(h_j)])))`. 기본은 `mtp=None`이라 state dict와 기존 동작은 그대로다.
- **`forward_verify()` + `Qwen3_5Cache.allocate_spec_buffers()/commit(n)`:** T token을 fixed shape로 검증한다. full attention은
  position..position+T-1에 K/V를 쓰고 token t는 position+t까지 본다(거부된 K/V는 position 뒤라 가려지고 나중에 덮인다).
  GatedDeltaNet은 token마다 한 step씩 돌려 conv 입력 창 전체와 token마다의 recurrent state를 spec buffer에 남기고, `commit(n)`이
  device 위에서 n번째 것을 고른다. 되돌리기에 재계산이 없다.
- **`decode_attention(..., tokens)`:** 한 KV head의 G·T query 행이 tl.dot 하나를 공유하고 행마다 key 한계가 다르다. key가 하나도 없는
  행(-inf max)은 빈 partial로 처리한다. T=1 코드 경로는 그대로다.
- **`w8a16_group_gemv_rows`(2–8행):** 두 형태. tensor core 형태는 int8 값이 bf16에서 정확하므로 scale group(128열)마다 `tl.dot`을 하고
  그 부분합에 scale을 곱한다. 출력 5120행 shape(down_proj, o_proj, MTP fc)는 program이 적어 K를 4개로 나눈다(fp32 부분합 + 합).
  multiply-add 형태는 sm80 미만용이다. 규칙은 `rows_launch_config`, 튜닝은 `tune_qwen3_5_group_gemv.py --rows 2,3,4`.
- **`infer_opt/speculative.py`:** `spec_round()`가 verify → accept → commit → MTP catch-up(`(h[p+i], g[i])` T쌍, 받아들인 위치는
  모두 맞는 쌍) → draft k−1개 → 다음 입력을 입력 buffer에 쓰기까지 한 함수다. `MTPSpeculativeDecoder`가 이것을 eager / compile /
  compile-reduce-overhead / cudagraph로 돌리고 round마다 host sync는 한 번(`n`과 `g`를 읽음)이다.
- `quantize_linears(exclude=...)`, `scripts/bench_qwen3_5_spec_decode.py`, `scripts/profile_qwen3_5_spec_round.py`, `tests/test_speculative.py`.

### 알게 된 것

- **MTP 입력 규약은 재서 정했다.** 0.8B에서 teacher-forced depth-1 acceptance: target의 final-norm 뒤 hidden + `[embedding, hidden]`
  순서 0.86, norm 앞 hidden 0.54, 순서를 바꾸면 0.07/0.06. vLLM 규약과 같다.
- **multi-row GEMV를 fp32 multiply-add로 하면 x를 읽는 양이 문제다.** program당 출력 1행이면 program마다 x의 모든 행을 다시 읽는다
  (weight 1바이트당 2M바이트). 2080에서 2/4행이 1행의 1.6/2.4배였고, A100에서는 tensor core 형태보다 2–7배 느렸다.
- **2080(sm75)에서 Triton `tl.dot`은 tensor core를 쓰지 않아** 25배 느렸다. sm75 측정으로 A100 kernel을 고르면 안 된다.
- **split-K는 program 수를 늘려 메모리 병렬성을 올린다.** down_proj(5120×17408): 쪼개지 않으면 BLOCK_N 16에서 program 320개,
  step마다 2 KB tile이라 1.55배. K를 4로 나누고 BLOCK_N 64면 program 수는 같고 tile이 8 KB라 1.13배. 행이 많은 shape는 쪼개면 느려진다.
  3행 GEMV 합계: 1행 15.7 ms에 대해 쪼개기 전 20.3, 후 18.2 ms.
- **round profile(k=3, int8, MTP·lm_head bf16, 모두 수락):** step 20.07 ms, round 33.85 ms(kernel 시간). 증가분 +13.8 ms 중
  multi-row GEMV +2.3, bf16 matmul +6.0(draft lm_head 3회 ~1.4 ms씩, bf16 MTP layer ~0.5 ms × 3), GatedDeltaNet verify 경로 +5.2
  (token별 state 복사 1.7 ms, recurrence kernel 4회 0.34 → 1.22 ms, token별로 나뉜 pointwise·dtype 변환). profile의 분류 정규식이
  `_to_copy`(dtype 변환)도 copy로 셌으니 GatedDeltaNet 몫은 근사다.
- Colab 운영: `pkill -f pipeline.sh`를 subprocess shell에서 부르면 자기 shell을 죽이고 대상은 살아남는다. 첫 bench가 그렇게 13분 남아
  37 GiB를 잡고 있어 재튜닝이 OOM이 났다. PID로 죽인다. 세션 비용 ~10 compute unit(134.1 → 124.0).

### 남은 일

- GatedDeltaNet verify를 layer당 kernel 하나로: state를 register에 두고 T token을 돌며 token마다의 state를 바로 쓰기(최소 151 MB 읽기 +
  T×151 MB 쓰기). round당 3–4 ms, 기준 구성 prose k=3에서 8.7 → ~7.7 ms/token 예상.
- draft용 축소 vocabulary lm_head(자주 나오는 token 행만, FR-Spec 방식): 검증은 전체 vocabulary라 출력은 같다. bf16 lm_head를
  draft마다 읽는 비용(k=3에서 round당 ~4 ms)을 대부분 없애므로, lm_head를 bf16으로 둔 기준 구성에서 효과가 더 크다.
- 27B의 baseline과 갈라진 자리가 near-tie인지 margin을 기록해 확인. sampling(rejection sampling)과 batch > 1은 없다.
- 27B 노트북은 아직 661f694를 고정한다.
