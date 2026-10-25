"""Write the Colab notebook that runs the INT8 Qwen3.5-27B with MTP speculative decoding.

The checkpoint includes the group-128 INT8 target and MTP head; lm_head stays bf16. No original BF16 weights or
runtime quantization are needed. Writes notebooks/qwen3_5_27b_int8_spec_colab.ipynb.

    uv run --group notebooks python scripts/build_qwen3_5_27b_spec_colab_notebook.py --ref <git commit> --model-revision <HF commit>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
COLAB = "https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/"
FILE = "qwen3_5_27b_int8_spec_colab.ipynb"
MODEL_ID = "Jakemu/Qwen3.5-27B-int8-g128-mtp"
ROWS_TABLE = "notebooks/assets/qwen3_5_27b_int8_spec_a100_20261024/rows_gemv_tune.json"


def cells(ref: str, model_revision: str) -> list:
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return [
        md(f"""# Qwen3.5-27B INT8 + MTP speculative decoding을 Colab에서 실행하기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{FILE})

[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})(weight-only INT8, group 128)에 포함된 **MTP(multi-token prediction) head**로
**self-speculative decoding**으로 생성합니다.

- MTP head(decoder layer 1개)가 다음 token 몇 개(`k`)를 싸게 추측하고, 본 모델이 그 `k + 1`개를 **forward 한 번**으로 검증합니다.
  batch 1 decode는 가중치를 읽는 시간이 대부분이라, token 여러 개를 검증해도 한 token만큼의 시간이 거의 그대로입니다.
- 검증은 greedy입니다: 본 모델의 1위 token과 같은 추측만 받아들이므로 **출력은 보통 decoding과 같습니다**
  (bf16 반올림 차이로 1·2위가 거의 같은 자리에서 갈라질 수는 있습니다).
- 한 round(검증 → 수락 → state 확정 → 다음 추측)를 `torch.compile` + CUDA graph 하나로 돌립니다.
- A100-SXM4-80GB에서 잰 값(k=3, 256 token, 같은 세션의 보통 decoding 19.8 ms/token):
  영어 산문 **7.6**, 수학 7.8, 코드 9.0, chat 10.2, 한국어 13.7(k=2) ms/token(1.45–2.61배).

**런타임:** 런타임 → 런타임 유형 변경 → **A100 GPU**(compute capability 8.0 이상, GPU 메모리 38 GiB 이상).
다운로드는 MTP head까지 포함한 int8 checkpoint 약 30 GB입니다. 셀을 위에서부터 차례로 실행하세요."""),
        md("## 1. GPU 확인"),
        code("""import torch

assert torch.cuda.is_available(), "GPU 런타임이 아닙니다: 런타임 → 런타임 유형 변경 → A100 GPU"
name, cap = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"{name}, compute capability {cap[0]}.{cap[1]}, {total_gib:.0f} GiB, torch {torch.__version__}")
assert cap >= (8, 0), "bf16 커널은 compute capability 8.0 이상(A100, H100 등)이 필요합니다"
assert total_gib >= 38, "GPU 메모리가 38 GiB 이상이어야 합니다(A100 40GB 이상)"
"""),
        md(f"""## 2. 라이브러리와 코드 설치

- `transformers`: tokenizer와 chat template를 씁니다.
- `flash-linear-attention`: prefill과 보통 decoding의 delta-rule 커널(`fla-delta`).
- infer-opt는 GitHub에서 받아 커밋 `{ref[:12]}`로 고정합니다(이 노트북을 만들 때의 코드)."""),
        code(f"""!pip install -q -U "transformers>=5" flash-linear-attention
!rm -rf /content/infer-opt && git clone -q https://github.com/Jake-Song/infer-opt.git /content/infer-opt
!git -C /content/infer-opt checkout -q {ref} && git -C /content/infer-opt log -1 --format="%h %s"

import sys
sys.path.insert(0, "/content/infer-opt")
"""),
        md(f"""## 3. 모델과 MTP head 불러오기

1. Hub의 `{model_revision[:12]}` revision을 받아 `from_quantized_dir`로 GPU에 올립니다.
2. `quantization.json`의 `mtp: true`에 따라 MTP head도 자동으로 복원합니다. 원래 BF16 모델을 받거나 현장에서 양자화할 필요가 없습니다.
3. 본 모델과 MTP head는 group-128 int8이며 공유하는 lm_head는 bf16입니다.
4. `fuse_projections()`로 같은 입력을 읽는 투영을 하나의 행렬곱으로 합칩니다(MTP layer 포함)."""),
        code(f"""import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from infer_opt.quant import Int8Linear
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM

t0 = time.perf_counter()
MODEL_REVISION = "{model_revision}"
path = Path(snapshot_download("{MODEL_ID}", revision=MODEL_REVISION))
size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
print(f"int8 checkpoint {{size / 2**30:.2f}} GiB, {{time.perf_counter() - t0:.0f}} s")

t0 = time.perf_counter()
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path)
assert model.mtp is not None, "MTP가 포함된 checkpoint와 이를 지원하는 infer-opt 코드가 필요합니다"
print("합친 투영 묶음", model.fuse_projections())
n_int8 = sum(isinstance(m, Int8Linear) for m in model.modules())
print(f"MTP 포함, 불러오기 {{time.perf_counter() - t0:.0f}} s, Int8Linear {{n_int8}}개, "
      f"GPU 메모리 {{torch.cuda.memory_allocated() / 2**30:.2f}} GiB")
"""),
        md(f"""## 4. Decode 준비

- **보통 decoding**: `FixedShapeDecoder`가 1 token step을 `torch.compile` + CUDA graph로 돌립니다(비교용).
- **speculative decoding**: `MTPSpeculativeDecoder(model, MAX_LEN, k=SPEC_K)`. round 하나가 `SPEC_K + 1` token을 검증하고
  1 ~ `SPEC_K + 1`개를 내놓습니다. 저장소의 측정에서는 k=3이 모든 프롬프트에서 가장 빨랐습니다(한국어는 k=2와 비슷).
- 검증의 int8 행렬곱(4행)은 여러 행용 Triton 커널을 씁니다. 그 launch 설정은 A100 80GB에서 튜닝한 표(`{ROWS_TABLE}`)를 읽습니다.
- 둘 다 처음 쓸 때 컴파일합니다(inductor `max_autotune`, 벤치마크와 같은 설정). **첫 생성은 각각 4–6분** 걸립니다."""),
        code(f"""import json

import torch._inductor.config as inductor_config

from infer_opt import quant_kernels
from infer_opt.qwen3_5 import FixedShapeDecoder
from infer_opt.speculative import MTPSpeculativeDecoder

inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = True
for m, n, k, group, config in json.loads(Path("/content/infer-opt/{ROWS_TABLE}").read_text())["rows_best"]:
    quant_kernels.ROWS_LAUNCH_CONFIGS[(m, n, k, group)] = tuple(config)

MAX_LEN = 2048
SPEC_K = 3
plain_cache = model.new_cache(1, MAX_LEN)
plain = FixedShapeDecoder(model, plain_cache, "compile-reduce-overhead")
spec = MTPSpeculativeDecoder(model, MAX_LEN, k=SPEC_K, mode="compile-reduce-overhead")
stop_ids = {{tokenizer.eos_token_id, *tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])}} - {{None}}


def encode(prompt: str, max_new_tokens: int) -> torch.Tensor:
    text = tokenizer.apply_chat_template([{{"role": "user", "content": prompt}}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    if ids.shape[1] + max_new_tokens > MAX_LEN:
        raise ValueError(f"prompt {{ids.shape[1]}} + {{max_new_tokens}} tokens > MAX_LEN {{MAX_LEN}}")
    return ids


class Printer:
    \"\"\"생성된 token을 글자로 이어서 출력한다(여러 token에 걸친 글자는 다 모일 때까지 기다린다).\"\"\"

    def __init__(self, on: bool):
        self.on, self.shown = on, ""

    def __call__(self, tokens: list[int]) -> None:
        if self.on:
            now = tokenizer.decode(tokens, skip_special_tokens=True)
            if not now.endswith("\\ufffd"):
                print(now[len(self.shown):], end="", flush=True)
                self.shown = now


def finish(out: list[int], max_new_tokens: int) -> list[int]:
    \"\"\"끝 token(있으면)까지, 최대 max_new_tokens개로 자른다.\"\"\"
    end = next((i + 1 for i, t in enumerate(out) if t in stop_ids), len(out))
    return out[:min(end, max_new_tokens)]


@torch.no_grad()
def generate_plain(prompt: str, max_new_tokens: int = 256, stream: bool = True) -> dict:
    \"\"\"보통 greedy decoding: step마다 1 token.\"\"\"
    ids, show = encode(prompt, max_new_tokens), Printer(stream)
    plain_cache.reset()
    token = model(ids, plain_cache)[:, -1].argmax(-1, keepdim=True)  # prefill
    torch.cuda.synchronize(); t1 = time.perf_counter()
    out = [int(token)]
    while len(out) < max_new_tokens and out[-1] not in stop_ids:
        token = plain.step(token)[:, -1].argmax(-1, keepdim=True)
        out.append(int(token))
        show(out)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    out = finish(out, max_new_tokens)
    return {{"tokens": out, "text": tokenizer.decode(out, skip_special_tokens=True),
            "decode_ms_per_token": (t2 - t1) / max(len(out) - 1, 1) * 1e3}}


@torch.no_grad()
def generate_spec(prompt: str, max_new_tokens: int = 256, stream: bool = True) -> dict:
    \"\"\"MTP speculative greedy decoding: round마다 1 ~ SPEC_K + 1 token.\"\"\"
    ids, show = encode(prompt, max_new_tokens), Printer(stream)
    out = spec.prefill(ids)[0].tolist()  # prefill과 첫 추측
    torch.cuda.synchronize(); t1 = time.perf_counter()
    while len(out) < max_new_tokens and not stop_ids.intersection(out[-SPEC_K - 1:]):
        out += spec.round()
        show(out)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    out = finish(out, max_new_tokens)
    stats = spec.stats.as_dict()
    return {{"tokens": out, "text": tokenizer.decode(out, skip_special_tokens=True), **stats,
            "decode_ms_per_token": (t2 - t1) / max(len(out) - 1, 1) * 1e3}}


def report(r: dict) -> None:
    extra = (f", round {{r['rounds']}}번, round당 {{r['tokens_per_round']:.2f}} token, 수락 분포 {{r['accepted_histogram']}}"
             if "rounds" in r else "")
    print(f"\\n[{{len(r['tokens'])}} token, decode {{r['decode_ms_per_token']:.1f}} ms/token{{extra}}]")
"""),
        md("""## 5. 생성해 보기

첫 실행은 컴파일 때문에 오래 걸리고, 그때 표시되는 ms/token에는 컴파일 시간이 섞입니다. 두 번째부터가 실제 속도입니다.
글자를 출력하느라 round(또는 step)마다 GPU를 기다리므로 6절의 측정보다 조금 느리게 나옵니다."""),
        code("""r = generate_spec("Explain in three sentences why batch-1 LLM decoding is limited by memory bandwidth.", max_new_tokens=160)
report(r)"""),
        code("""r = generate_spec("speculative decoding이 출력은 그대로 두면서 생성을 빠르게 하는 원리를 다섯 문장으로 설명해 줘.",
                  max_new_tokens=256)
report(r)"""),
        md("""## 6. 보통 decoding과 비교

같은 프롬프트를 두 방식으로 출력 없이 생성하고 decode ms/token과 token 열을 비교합니다. 보통 decoding은 처음 쓸 때 컴파일하므로
먼저 짧게 한 번 돌립니다(4–6분).

출력은 대부분 끝까지 같습니다. 갈라지는 경우는 1·2위 logit이 거의 같은 자리입니다: 검증은 `k + 1` token을 한 번에 계산하고 보통
decoding은 1 token씩 계산해서 bf16 반올림이 조금 다르고, 동점에 가까운 1위가 뒤바뀔 수 있습니다. 그 뒤로는 둘 다 자기 앞
token을 따라 정상적인 글을 이어 갑니다."""),
        code("""_ = generate_plain("Hello", max_new_tokens=8, stream=False)  # 컴파일

PROMPTS = {
    "설명(영어)": "Explain how a transformer language model generates text, covering attention, the KV cache, and sampling.",
    "코드": "Write a Python function that merges overlapping intervals, with a docstring and a few tests.",
    "수학": "A train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed? Think step by step.",
    "한국어": "서울의 역사와 한강이 도시 발전에 한 역할을 한 단락으로 설명해 줘.",
}
rows = []
for label, prompt in PROMPTS.items():
    a = generate_plain(prompt, max_new_tokens=256, stream=False)
    b = generate_spec(prompt, max_new_tokens=256, stream=False)
    same = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
    rows.append((label, a["decode_ms_per_token"], b["decode_ms_per_token"], b["tokens_per_round"], same, len(a["tokens"])))

print(f"{'프롬프트':10s} {'보통':>8s} {'spec':>8s} {'배수':>6s} {'token/round':>12s}  같은 token")
for label, ms_a, ms_b, tpr, same, n in rows:
    print(f"{label:10s} {ms_a:8.2f} {ms_b:8.2f} {ms_a / ms_b:5.2f}× {tpr:12.2f}  " + ("전부" if same is None else f"처음 {same}/{n}"))
print(f"peak GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
"""),
        md("""## 7. 직접 써 보기

`PROMPT`를 바꿔서 실행하세요. greedy decoding(항상 가장 확률 높은 token)만 지원합니다."""),
        code("""PROMPT = "Qwen3.5의 linear attention(gated delta rule)이 긴 문맥에서 유리한 이유를 설명해 줘."
r = generate_spec(PROMPT, max_new_tokens=300)
report(r)"""),
        md(f"""## 참고

- 측정과 프로파일의 원시 결과는 저장소의 `notebooks/assets/qwen3_5_27b_int8_spec_gdn_a100_20261025/`(이 코드)와
  `notebooks/assets/qwen3_5_27b_int8_spec_a100_20261024/`(처음 버전, 여러 행 GEMV 튜닝 표), `worklog.md`(2026-10-24, 25)에 있습니다.
  저장소의 측정은 bf16 원래 모델을 같은 방식(group 128)으로 제자리 양자화해서 잰 값입니다.
- 추측이 맞는 비율은 글의 종류에 따라 다릅니다. 영어 산문과 수학은 round마다 거의 4 token을 받고, 한국어는 2 token 정도입니다.
  추측이 계속 틀리면 round가 보통 step보다 비싸서(k=3에서 약 1.5배) 오히려 느려질 수 있습니다.
- lm_head는 공개 checkpoint와 같게 bf16으로 둡니다. lm_head까지 int8로 하면 조금 더 빠르지만(0.4–0.7 ms/token) 본 모델 자체가 바뀝니다.
- GatedDeltaNet layer는 검증한 token마다 state를 남겨 두고(`infer_opt/gated_delta_kernels.py`), 받아들인 위치의 state를 골라
  확정합니다. 거부된 추측을 되돌릴 때 다시 계산하지 않습니다."""),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="infer-opt git commit the notebook checks out")
    parser.add_argument("--model-revision", required=True, help="HF commit of the bundled MTP checkpoint")
    args = parser.parse_args()
    nb = nbformat.v4.new_notebook(cells=cells(args.ref, args.model_revision))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"gpuType": "A100", "provenance": []}
    nbformat.validate(nb)
    for cell in nb.cells:  # a syntax error in a cell only shows on Colab otherwise
        if cell.cell_type == "code":
            ast.parse("\n".join(line for line in cell.source.splitlines() if not line.startswith("!")))
    out = ROOT / "notebooks" / FILE
    nbformat.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
