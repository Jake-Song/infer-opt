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
