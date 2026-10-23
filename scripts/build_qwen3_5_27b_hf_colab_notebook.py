"""Build a standalone HF-original 27B notebook with compiled CUDA Graph decode.

    .venv/bin/python scripts/build_qwen3_5_27b_hf_colab_notebook.py
"""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/qwen3_5_27b_hf_colab.ipynb"


def build():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("""# Qwen3.5-27B HF 원본 — torch.compile + CUDA Graph

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/qwen3_5_27b_hf_colab.ipynb)

**HF 원본 `Qwen/Qwen3.5-27B`의 BF16 텍스트 모델**로 생성하고 batch-1 decode 속도를 측정합니다.
HF의 `AutoModelForCausalLM`과 원본 가중치를 사용합니다. INT8 모델이나 infer-opt 모델 구현을 불러오지 않습니다.
이미지·영상 입력은 다루지 않습니다.

- prefill: HF eager, 마지막 위치의 logits만 계산.
- decode: HF `StaticCache` + `torch.compile(fullgraph=True, dynamic=False)` + **명시적인 CUDA Graph capture/replay**.
- 먼저 HF dynamic-cache eager 결과와 비교합니다. 검증 실패 시 성능 측정을 중단합니다.
- 다운로드·컴파일·capture·warmup은 decode 측정에서 제외합니다. 결과와 환경 정보는 JSON으로 저장합니다.

**하드웨어:** 원본은 가중치만 약 52 GiB입니다. 80GB급 A100/H100 GPU가 필요하며, Colab에서 A100을 선택해도
40GB가 배정될 수 있습니다. 아래 셀이 실제 용량을 확인합니다. 40GB/T4/L4에서는 이 BF16 구성을 실행할 수 없습니다.
다운로드·캐시에 여유 디스크 약 65 GiB 이상을 준비하세요. CPU offload는 사용하지 않습니다.

새 GPU 런타임에서 위에서부터 실행하세요. 최초 컴파일은 수 분 이상 걸릴 수 있습니다.
이 노트북은 측정 결과를 미리 채워 넣지 않았습니다."""),
        md("## 1. GPU와 디스크 확인"),
        code('''import shutil
import torch

assert torch.cuda.is_available(), "GPU 런타임을 선택하세요."
gpu = torch.cuda.get_device_properties(0)
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"{gpu.name}: total {total_bytes / 2**30:.1f} GiB, free {free_bytes / 2**30:.1f} GiB")
print("torch", torch.__version__, "CUDA", torch.version.cuda)
assert torch.cuda.get_device_capability() >= (8, 0), "BF16을 지원하는 Ampere 이상 GPU가 필요합니다."
assert free_bytes / 2**30 >= 65, "BF16 원본 + cache/컴파일 여유 공간이 부족합니다. 80GB급 GPU를 사용하세요."
disk_free = shutil.disk_usage("/content").free / 2**30
print(f"disk free: {disk_free:.1f} GiB")
if disk_free < 65:
    print("디스크 여유가 적습니다. 모델이 이미 캐시되어 있지 않다면 다운로드 전에 공간을 확보하세요.")
'''),
        md("""## 2. 라이브러리 설치

cache API를 확인한 `transformers==5.20.2`를 사용합니다. PyTorch는 Colab 설치 버전을 유지합니다.
`flash-linear-attention`은 HF가 사용하는 delta-rule 구현입니다. 별도 모델 코드를 패치하지 않습니다.
`causal-conv1d`가 없으면 HF의 PyTorch conv 경로를 사용하며 이 환경도 결과에 기록됩니다.
이미 Transformers를 import한 런타임이라면 설치 후 런타임을 다시 시작하세요."""),
        code('''import subprocess
import sys

assert "transformers" not in sys.modules, "새 런타임에서 실행하거나 설치 후 런타임을 다시 시작하세요."
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.20.2", "accelerate", "flash-linear-attention"], check=True)
'''),
        md("""## 3. HF BF16 모델 로드

`device_map={"": 0}`으로 모든 가중치를 GPU에 로드합니다. `AutoModelForCausalLM`은 이 체크포인트의
원본 텍스트 모델을 선택하며 vision tower는 사용하지 않습니다. `REVISION`을 커밋 SHA로 지정하면 모델 버전도 고정할 수
있습니다. 기본값은 실행 시점의 main을 SHA로 해석한 뒤 그 SHA로 tokenizer와 모델을 함께 로드합니다."""),
        code('''import importlib.metadata as metadata
import json
import time
from pathlib import Path

import transformers
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-27B"
REVISION = "main"
MODEL_SHA = model_info(MODEL_ID, revision=REVISION).sha
assert transformers.__version__ == "5.20.2", "런타임을 다시 시작해 설치된 버전을 사용하세요."
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_SHA)
start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_SHA, dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa",
).eval().requires_grad_(False)
assert all(p.device.type == "cuda" for p in model.parameters()), "모든 가중치가 GPU에 있어야 합니다."
print(type(model).__name__, "revision", MODEL_SHA)
print(f"load {time.perf_counter() - start:.1f} s, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

environment = {
    "model": MODEL_ID, "revision": MODEL_SHA, "model_class": type(model).__name__,
    "gpu": gpu.name, "gpu_total_gib": total_bytes / 2**30,
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "packages": {name: package_version(name) for name in
                 ("transformers", "triton", "flash-linear-attention", "causal-conv1d", "accelerate")},
    "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
}
print(json.dumps(environment, indent=2))
'''),
        md("""## 4. 고정 cache와 decode 실행기

아래 셀은 노트북에 포함된 실행 보조 코드입니다. HF 모델의 forward나 가중치는 변경하지 않습니다.

`StaticCache`는 full-attention K/V와 linear-attention conv/recurrent state를 유지합니다.
prefill에서 버퍼를 초기화하고, decode + greedy argmax + 위치 갱신을 컴파일합니다.
자동 CUDA Graph 기능은 끈 상태로 컴파일한 후 `torch.cuda.CUDAGraph`로 직접 캡처하므로 graph 사용 여부가 명확합니다.
warmup/capture 뒤에는 cache를 reset하고 다시 prefill합니다. 캡처된 버퍼 주소도 검사합니다.
컴파일이나 캡처가 실패하면 오류를 표시하며 eager로 자동 전환하지 않습니다."""),
        code((ROOT / "scripts/qwen3_5_hf_graph.py").read_text()),
        md("""## 5. 컴파일과 CUDA Graph 캡처

기존 INT8 벤치마크와 맞춰 랜덤 프롬프트 1024 tokens, warmup 3 steps, decode 128 steps,
최대 cache 길이 1211을 기본으로 사용합니다. static attention은 cache 용량의 영향도 받으므로 비교할 때 같은 값을 쓰세요.
설정을 바꾸면 새 실행기를 준비해야 합니다. 반복 실행 시 이전 그래프를 먼저 해제합니다."""),
        code('''import gc

PROMPT_TOKENS = 1024
DECODE_STEPS = 128
WARMUP = 3
REPEATS = 5
MAX_CACHE_LEN = PROMPT_TOKENS + WARMUP + DECODE_STEPS + 56

if "decoder" in globals():
    del decoder
torch._dynamo.reset()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(0)
bench_ids = torch.randint(1000, 100000, (1, PROMPT_TOKENS), device="cuda")
decoder = HFGraphDecoder(model, MAX_CACHE_LEN)
setup = decoder.prepare(bench_ids)
setup["setup_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
print(json.dumps(setup, indent=2))
assert decoder.graph is not None
'''),
        md("""## 6. eager 기준 검증

HF dynamic-cache eager가 생성한 token을 같은 순서로 넣는 teacher forcing으로 검사합니다.
static-cache eager / compile / compile+graph를 각각 비교하고, graph는 cache reset 후 한 번 더 검사합니다.
랜덤 1024-token 입력과 짧은 실제 chat 입력을 모두 검사합니다.

KL(HF dynamic eager ‖ 해당 경로), top-1 일치율, 최대 logits 차이와 cache 위치를 기록합니다.
이 노트북의 smoke gate는 평균 KL ≤ 0.005, HF top-1/2 margin > 0.5인 위치의 불일치 0개입니다.
이는 양자화 모델의 기존 품질 gate나 종합 품질 평가를 대신하지 않습니다."""),
        code('''check_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain why batch-1 LLM decoding is limited by memory bandwidth."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
check_ids = tokenizer(check_text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
validation = []
for name, ids in (("random_1k", bench_ids), ("chat", check_ids)):
    rows = validate_decoder(decoder, ids, steps=16)
    validation.extend({"prompt": name, **row} for row in rows)
print(json.dumps(validation, indent=2))
VALIDATED = all(row["passed"] for row in validation)
assert VALIDATED, "eager 대비 검증 실패: 속도 측정을 중단합니다. 위 수치와 환경을 확인하세요."
'''),
        md("""## 7. Decode 속도 측정

각 반복은 새로운 prefill과 warmup으로 시작합니다. **128회의 1-token decode**만 측정하며 argmax를 포함합니다.
EOS에서 중단하지 않고 step 사이에 CPU token 변환·출력·동기화를 넣지 않습니다.
wall ms/token은 Python replay 호출 비용을 포함합니다. CUDA-event 시간도 함께 저장합니다.
5회 반복 평균이 아니라 **각 반복 ms/token의 중앙값**을 보고합니다. compile/capture와 prefill은 제외됩니다.
이 결과는 `generate()` 전체 지연 시간이나 TTFT가 아닙니다."""),
        code('''import statistics

assert VALIDATED, "먼저 검증 셀을 통과해야 합니다."
samples = benchmark_decoder(decoder, bench_ids, steps=DECODE_STEPS, warmup=WARMUP, repeats=REPEATS)
median_ms = statistics.median(s["wall_ms_per_token"] for s in samples)
summary = {
    "wall_median_ms_per_token": median_ms, "tokens_per_second": 1000 / median_ms,
    "cuda_median_ms_per_token": statistics.median(s["cuda_ms_per_token"] for s in samples),
    "decode_peak_allocated_gib": max(s["decode_peak_allocated_gib"] for s in samples),
}
print(json.dumps(summary, indent=2))
print("wall ms/token per repeat:", [round(s["wall_ms_per_token"], 3) for s in samples])
'''),
        md("""## 8. 직접 생성하기

같은 graph를 새 prompt에 재사용합니다. `enable_thinking=False`, greedy decoding이며 EOS에서 중단합니다.
아래 생성 함수는 종료 판정을 위해 token마다 CPU와 동기화하므로 속도 비교에는 7절을 사용하세요.
프롬프트와 출력의 합이 `MAX_CACHE_LEN`을 넘지 않아야 합니다."""),
        code('''assert VALIDATED, "먼저 검증 셀을 통과해야 합니다."
PROMPT = "가중치를 int8로 양자화하면 언어 모델 추론이 왜 빨라지는지 다섯 문장으로 설명해 줘."
generated = generate_text(decoder, tokenizer, PROMPT, max_new_tokens=160)
print(generated["text"])
print(f"prompt {generated['prompt_tokens']} tokens, generated {generated['new_tokens']} tokens")
'''),
        md("## 9. 결과 저장·다운로드"),
        code('''result = {
    "environment": environment, "setup": setup, "validation": validation,
    "workload": {"prompt_tokens": PROMPT_TOKENS, "decode_steps": DECODE_STEPS,
                 "warmup": WARMUP, "repeats": REPEATS, "max_cache_len": MAX_CACHE_LEN,
                 "seed": 0, "compile_mode": "max-autotune-no-cudagraphs",
                 "explicit_cuda_graph": True, "sampling": "greedy_argmax", "prefill_timed": False},
    "summary": summary, "samples": samples,
}
output = Path("qwen3_5_27b_hf_compile_cudagraph.json")
output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(output.resolve())
from google.colab import files
files.download(str(output))
'''),
        md("""## 참고

- [Qwen3.5-27B 원본](https://huggingface.co/Qwen/Qwen3.5-27B)
- [HF StaticCache와 compile](https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache)
- [PyTorch CUDA Graph capture/replay](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

HF BF16 + compile + graph의 결과입니다. 라이브러리 버전, GPU 종류, cache 길이가 다르면 저장소의 이전 숫자와
직접 비교할 수 없습니다. 완료 후 Colab의 **런타임 → 연결 해제 및 런타임 삭제**로 GPU 사용을 종료하세요."""),
    ]
    for i, cell in enumerate(cells):
        cell.id = f"hf27b-{i:02d}"
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []}})
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
