"""Write notebooks/qwen3_5_9b_int8_colab.ipynb: run the group-128 INT8 Qwen3.5-9B from the Hugging Face Hub on Colab.

    uv run --group notebooks python scripts/build_qwen3_5_int8_colab_notebook.py --ref <git commit>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
COLAB_URL = "https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_9b_int8_colab.ipynb"


def cells(ref: str, model_id: str) -> list:
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return [
        md(f"""# Qwen3.5-9B INT8 (group-128)을 Colab에서 실행하기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

[`{model_id}`](https://huggingface.co/{model_id})에 올린 **weight-only INT8 Qwen3.5-9B**로 글을 생성하고 decode 속도를 잽니다.

- Linear 가중치(lm_head와 출력 32개짜리 delta-rule gate 제외)를 int8로 저장하고, 출력 채널마다·입력 128열마다 bf16 scale을 둡니다.
  가중치가 약 절반이 되어 batch 1 decode가 빨라집니다.
- 1 token decode의 행렬곱은 [infer-opt](https://github.com/Jake-Song/infer-opt)의 Triton 커널(`infer_opt/quant_kernels.py`)이 처리하고,
  decode step 전체는 `torch.compile` + CUDA graph로 돌립니다.
- 같은 A100에서 잰 값: bf16 15.1 ms/token → 이 모델 **10.8 ms/token**. HF 구현 대비 정확도 기준(같은 실행 bf16 KL의 2배 이하,
  확신 있는 불일치 없음)을 통과한 구성입니다.

**런타임:** 런타임 → 런타임 유형 변경 → **A100 GPU**. 커널이 bf16 Triton을 쓰므로 compute capability 8.0 이상이 필요합니다
(T4는 안 됩니다. L4는 될 것으로 보이지만 확인하지 않았습니다). GPU 메모리는 약 11 GiB, 다운로드는 약 10 GB입니다.

이 checkpoint는 `transformers` 형식이 아니라서 infer-opt 코드로 읽습니다. 셀을 위에서부터 차례로 실행하세요."""),
        md("## 1. GPU 확인"),
        code("""import torch

assert torch.cuda.is_available(), "GPU 런타임이 아닙니다: 런타임 → 런타임 유형 변경 → A100 GPU"
name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
print(f"{name}, compute capability {cap[0]}.{cap[1]}, torch {torch.__version__}")
assert cap >= (8, 0), "bf16 Triton 커널은 compute capability 8.0 이상(A100, L4, H100 등)이 필요합니다"
"""),
        md(f"""## 2. 라이브러리와 코드 설치

- `transformers`: tokenizer와 chat template만 씁니다.
- `flash-linear-attention`: delta-rule 커널(`fla-delta`).
- infer-opt는 GitHub에서 받아 커밋 `{ref[:12]}`로 고정합니다(이 노트북을 만들 때의 코드)."""),
        code(f"""!pip install -q -U "transformers>=5" flash-linear-attention
!rm -rf /content/infer-opt && git clone -q https://github.com/Jake-Song/infer-opt.git /content/infer-opt
!git -C /content/infer-opt checkout -q {ref} && git -C /content/infer-opt log -1 --format="%h %s"

import sys
sys.path.insert(0, "/content/infer-opt")
"""),
        md(f"""## 3. 모델 받기와 불러오기

Hub에서 int8 checkpoint(safetensors shard와 `quantization.json`)를 받고, `from_quantized_dir`로 GPU에 바로 올립니다.
`quantization.json`에는 어떤 Linear가 int8인지와 group 크기가 들어 있습니다."""),
        code(f"""import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from infer_opt.quant import Int8Linear
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM

MODEL_ID = "{model_id}"
t0 = time.perf_counter()
path = Path(snapshot_download(MODEL_ID))
size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
print(f"다운로드 {{size / 2**30:.2f}} GiB, {{time.perf_counter() - t0:.0f}} s")

t0 = time.perf_counter()
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
tokenizer = AutoTokenizer.from_pretrained(path)
n_int8 = sum(isinstance(m, Int8Linear) for m in model.modules())
print(f"불러오기 {{time.perf_counter() - t0:.0f}} s, Int8Linear {{n_int8}}개, GPU 메모리 {{torch.cuda.memory_allocated() / 2**30:.2f}} GiB")
"""),
        md("""## 4. Decode 준비

- **static cache**: K/V와 delta-rule 상태를 최대 길이(`MAX_LEN`)만큼 미리 잡아 둡니다. 프롬프트가 바뀌면 `reset()`만 합니다.
  고정 shape 경로의 full attention은 cache 길이 전체를 훑으므로 `MAX_LEN`이 길수록 step이 조금 느려집니다
  (A100에서 4096이면 1211보다 약 1.5 ms/token 느렸습니다).
- **`FixedShapeDecoder(..., "compile-reduce-overhead")`**: 1 token decode step을 `torch.compile` + CUDA graph로 돌립니다.
  처음 step에서 컴파일하므로 **첫 생성은 1–3분** 걸립니다(prefill의 Triton·FLA 커널도 처음 쓸 때 컴파일되어 첫 prefill만 수십 초).
  그 뒤로는 같은 그래프를 재생합니다.
- inductor의 `max_autotune`과 `coordinate_descent_tuning`은 벤치마크를 잴 때와 같은 설정입니다.
- prompt(prefill)는 컴파일 없이 한 번에 처리합니다. 이 경로는 int8 가중치를 bf16으로 풀어서 곱하므로 decode보다 메모리를 더 씁니다."""),
        code("""import torch._inductor.config as inductor_config

from infer_opt.qwen3_5 import FixedShapeDecoder

inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True

MAX_LEN = 2048
cache = model.new_cache(1, MAX_LEN)
decoder = FixedShapeDecoder(model, cache, "compile-reduce-overhead")
stop_ids = {tokenizer.eos_token_id, *tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])} - {None}


@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 256, stream: bool = True) -> dict:
    \"\"\"chat template로 감싼 prompt에 greedy decoding. 생성 글과 prefill / decode 시간을 돌려준다.\"\"\"
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    if ids.shape[1] + max_new_tokens > MAX_LEN:
        raise ValueError(f"prompt {ids.shape[1]} + {max_new_tokens} tokens > MAX_LEN {MAX_LEN}")
    cache.reset()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    token = model(ids, cache)[:, -1].argmax(-1, keepdim=True)  # prefill
    torch.cuda.synchronize(); t1 = time.perf_counter()
    out, shown = [int(token)], ""
    for _ in range(max_new_tokens - 1):
        if out[-1] in stop_ids:
            break
        token = decoder.step(token)[:, -1].argmax(-1, keepdim=True)
        out.append(int(token))  # int()이 GPU를 기다리므로 step마다 동기화된다
        if stream:
            now = tokenizer.decode(out, skip_special_tokens=True)
            if not now.endswith("\ufffd"):  # 여러 token에 걸친 글자는 다 모일 때까지 기다린다
                print(now[len(shown):], end="", flush=True)
                shown = now
    torch.cuda.synchronize(); t2 = time.perf_counter()
    steps = len(out) - 1
    stats = {"prompt_tokens": ids.shape[1], "new_tokens": len(out), "prefill_s": t1 - t0,
             "decode_ms_per_token": (t2 - t1) / max(steps, 1) * 1e3}
    if stream:
        print(f"\\n\\n[prompt {stats['prompt_tokens']} tok, 생성 {stats['new_tokens']} tok, prefill {stats['prefill_s']:.2f} s, "
              f"decode {stats['decode_ms_per_token']:.1f} ms/token]")
    return {"text": tokenizer.decode(out, skip_special_tokens=True), "tokens": out, **stats}
"""),
        md("""## 5. 생성해 보기

첫 실행은 컴파일 때문에 오래 걸리고, 표시되는 decode ms/token에도 컴파일 시간이 섞입니다. 두 번째부터가 실제 속도입니다.
글자를 하나씩 출력하느라 step마다 GPU를 기다리므로, 여기 나오는 ms/token은 6절의 측정보다 조금 큽니다."""),
        code("""_ = generate("Explain in three sentences why batch-1 LLM decoding is limited by memory bandwidth.", max_new_tokens=160)"""),
        code("""_ = generate("가중치를 int8로 양자화하면 언어 모델 추론이 왜 빨라지는지, 정확도는 어떻게 확인하는지 다섯 문장으로 설명해 줘.",
             max_new_tokens=256)"""),
        md("""## 6. Decode 속도 측정

벤치마크 스크립트(`scripts/bench_qwen3_5_int8_decode.py`)와 같은 방식입니다: 랜덤 1024-token 프롬프트를 prefill하고,
warm-up 3 step 뒤 128 step을 이어서 돌린 평균 ms/token입니다. 출력도 동기화도 step 사이에 하지 않습니다.
cache 길이도 그 스크립트와 같게(1211) 따로 잡으므로 decode step을 한 번 더 컴파일합니다(1–2분).
같은 방식으로 A100-SXM4-40GB에서 잰 값은 bf16 15.09, 이 모델 10.79 ms/token입니다."""),
        code("""@torch.no_grad()
def bench_decode(prompt_tokens: int = 1024, steps: int = 128, warmup: int = 3) -> float:
    bench_cache = model.new_cache(1, prompt_tokens + warmup + steps + 56)  # 1211: 벤치마크 스크립트와 같은 cache 길이
    bench_decoder = FixedShapeDecoder(model, bench_cache, "compile-reduce-overhead")
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, prompt_tokens), device="cuda")
    token = model(ids, bench_cache)[:, -1].argmax(-1, keepdim=True)
    for _ in range(warmup):  # 첫 step에서 컴파일된다
        token = bench_decoder.step(token)[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        token = bench_decoder.step(token)[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


ms = bench_decode()
print(f"{torch.cuda.get_device_name()}: {ms:.2f} ms/token ({1000 / ms:.1f} tok/s), "
      f"peak GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
"""),
        md("""## 7. 기존 HF 모델과 비교

원래 모델 [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)를 `transformers`로 bf16 그대로 올려서 같은 GPU에서 비교합니다.
HF 쪽은 기본 설정(동적 cache, `torch.compile` 없음)입니다.

- **7.1 같은 질문**: 같은 chat prompt로 둘 다 greedy 생성하고, 처음 달라지는 token 위치를 봅니다.
- **7.2 decode 속도**: 저장소의 `scripts/bench_qwen3_5.py`와 같은 방식으로 HF decode를 잽니다.
  랜덤 1024-token 프롬프트로 `generate(128)`과 `generate(1)`의 시간 차를 127로 나눈 값이고, 각각 3번 중 최솟값입니다.
  2026-10-17 A100에서 이 방식으로 잰 값은 61.7 ms/token이었습니다.
- **7.3 정확도**: HF가 생성한 token 열을 두 모델에 그대로 넣고(teacher forcing) 위치마다 다음 token 분포를 비교합니다.

INT8 모델을 둔 채 bf16 모델(약 17.5 GiB)을 더 올리므로 **GPU 메모리가 40 GB 이상**이어야 합니다. 그보다 작으면 이 절은 건너뜁니다.
다운로드는 약 19 GB입니다."""),
        code("""from transformers import AutoModelForCausalLM, TextStreamer

total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
RUN_HF = total_gib >= 38
if RUN_HF:
    t0 = time.perf_counter()
    hf_path = snapshot_download("Qwen/Qwen3.5-9B", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    hf = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.bfloat16).cuda().eval()
    print(f"HF bf16 모델 준비 {time.perf_counter() - t0:.0f} s, GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GiB (INT8 모델 포함)")
else:
    print(f"GPU 메모리 {total_gib:.0f} GiB: 두 모델을 함께 올릴 수 없어 7절을 건너뜁니다")
"""),
        md("""### 7.1 같은 질문

HF는 `generate(do_sample=False)`, INT8은 위의 `generate()`입니다. 둘 다 greedy라서 bf16과 int8의 작은 수치 차이가
어느 위치에서 1위 token을 바꾸는지가 그대로 보입니다. 달라진 뒤로는 문장이 다르게 이어지는 것이 정상입니다."""),
        code("""COMPARE_PROMPT = "Explain in three sentences why batch-1 LLM decoding is limited by memory bandwidth."
COMPARE_TOKENS = 160


@torch.no_grad()
def hf_generate(prompt: str, max_new_tokens: int) -> tuple[torch.Tensor, list[int], float]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    full = hf.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                       streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True))
    torch.cuda.synchronize()
    return ids, full[0, ids.shape[1]:].tolist(), time.perf_counter() - t0


if RUN_HF:
    print("=== HF bf16 ===")
    hf_ids, hf_out, hf_s = hf_generate(COMPARE_PROMPT, COMPARE_TOKENS)
    print(f"[{len(hf_out)} tok, {hf_s:.1f} s, prefill 포함 {hf_s / len(hf_out) * 1e3:.1f} ms/token]\\n")
    print("=== INT8 ===")
    q = generate(COMPARE_PROMPT, max_new_tokens=COMPARE_TOKENS)
    q_out = q["tokens"]
    same = next((i for i, (a, b) in enumerate(zip(hf_out, q_out)) if a != b), min(len(hf_out), len(q_out)))
    print(f"\\n처음 {same} token이 같습니다" + ("" if same == min(len(hf_out), len(q_out)) else
          f": {same}번째에서 HF {tokenizer.decode(hf_out[same])!r} / INT8 {tokenizer.decode(q_out[same])!r}"))
"""),
        md("""### 7.2 decode 속도

INT8 숫자는 6절에서 잰 `ms`입니다. HF 쪽은 생성 길이를 고정하려고 `min_new_tokens`를 같이 줍니다."""),
        code("""@torch.no_grad()
def hf_decode_ms(prompt_tokens: int = 1024, new_tokens: int = 128, repeats: int = 3) -> float:
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, prompt_tokens), device="cuda")

    def timed(n):
        best = float("inf")
        for _ in range(repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            hf.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
            torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
        return best

    timed(2)  # warm-up
    return (timed(new_tokens) - timed(1)) / (new_tokens - 1) * 1e3


if RUN_HF:
    hf_ms = hf_decode_ms()
    rows = [("HF transformers bf16 (기본 설정)", hf_ms), ("INT8 group-128 + Triton GEMV + CUDA graph", ms)]
    for label, v in rows:
        print(f"{label:44s} {v:7.2f} ms/token {1000 / v:7.1f} tok/s  HF 대비 {hf_ms / v:5.2f}×")
"""),
        md("""### 7.3 정확도 (teacher forcing)

7.1에서 HF가 만든 token 열(프롬프트 + 생성)을 두 모델에 한 번에 넣고, 생성 구간의 위치마다
KL(HF ‖ INT8)과 1위 token 일치율을 봅니다. INT8 쪽은 여러 token을 한 번에 처리하므로 Triton GEMV가 아니라
가중치를 bf16으로 풀어서 곱하는 경로를 씁니다(같은 int8 값과 scale)."""),
        code("""import torch.nn.functional as F

if RUN_HF:
    seq = torch.cat([hf_ids, torch.tensor([hf_out], device="cuda")], dim=1)
    start = hf_ids.shape[1] - 1  # 첫 생성 token을 예측하는 위치부터
    with torch.no_grad():
        p_hf = F.log_softmax(hf(input_ids=seq).logits[0, start:-1].float(), -1)
        p_q = F.log_softmax(model(seq)[0, start:-1].float(), -1)
    kl = F.kl_div(p_q, p_hf, log_target=True, reduction="none").sum(-1)
    top1 = (p_hf.argmax(-1) == p_q.argmax(-1)).float()
    print(f"{len(kl)}개 위치: KL(HF‖INT8) 평균 {kl.mean():.5f}, 최대 {kl.max():.4f}, 1위 token 일치 {top1.mean():.1%}")
    print("저장소의 정확도 기준(9개 프롬프트)에서는 INT8의 HF 대비 KL이 같은 실행 bf16 구현의 1.55–1.89배였습니다.")
"""),
        md("""이 비교가 끝나면 HF 모델을 내려서 메모리를 돌려줍니다."""),
        code("""if RUN_HF:
    del hf
    torch.cuda.empty_cache()
    print(f"GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
"""),
        md("""## 8. 직접 써 보기

`PROMPT`를 바꿔서 실행하세요. greedy decoding(항상 가장 확률 높은 token)만 지원합니다."""),
        code("""PROMPT = "Qwen3.5의 linear attention(gated delta rule)이 긴 문맥에서 유리한 이유를 설명해 줘."
_ = generate(PROMPT, max_new_tokens=300)"""),
        md("""## 참고

- 정확도 기준과 구현별 속도 비교는 저장소의 `notebooks/qwen3_5_9b_benchmark_tables.ipynb` §5와 `worklog.md`에 있습니다.
- lm_head까지 int8로 하면 10.15 ms/token으로 조금 더 빠르지만 정확도 기준을 한 번 통과하지 못한 적이 있어 이 모델은 lm_head를 bf16으로 둡니다.
- 출력 채널마다 scale이 하나인 per-channel int8은 8.9 ms/token으로 더 빠르지만 정확도 기준을 통과하지 못했습니다."""),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="infer-opt git commit the notebook checks out")
    parser.add_argument("--model-id", default="Jakemu/Qwen3.5-9B-int8-g128")
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "qwen3_5_9b_int8_colab.ipynb"))
    args = parser.parse_args()
    nb = nbformat.v4.new_notebook(cells=cells(args.ref, args.model_id))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nbformat.validate(nb)
    nbformat.write(nb, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
