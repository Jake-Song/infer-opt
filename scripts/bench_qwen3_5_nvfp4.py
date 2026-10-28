"""G4 native NVFP4 preflight, calibration and repeated decoder/MTP measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
from infer_opt.nvfp4 import Nvfp4Linear, quantize_nvfp4, require_sm120
from infer_opt.quant import quantize_linears
from infer_opt.qwen3_5 import FixedShapeDecoder, Qwen3_5ForCausalLM
from infer_opt.speculative import MTPSpeculativeDecoder
from bench_qwen3_5_spec_decode import baseline, speculative, PROMPTS
from validate_qwen3_5_hf import build_prompts, CODE_PROMPT_SOURCE


def calibration_inputs(tok, out: Path, samples=128, revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = revision or HfApi().dataset_info("HuggingFaceH4/ultrachat_200k").sha
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", revision=revision, streaming=True)
    ids, hashes = [], []
    for row in ds:
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        t = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :512]
        if t.shape[1] == 0:
            continue
        hashes.append(hashlib.sha256(t.numpy().tobytes()).hexdigest())
        ids.append(t.cuda())
        if len(ids) == samples:
            break
    if len(ids) != samples:
        raise RuntimeError("calibration dataset exhausted")
    (out / "calibration.json").write_text(json.dumps({"dataset": "HuggingFaceH4/ultrachat_200k",
        "revision": revision, "split": "train_sft", "max_tokens": 512, "token_hashes": hashes}, indent=2))
    return ids


@torch.no_grad()
def smoke():
    import flashinfer
    require_sm120()
    records = []
    for n, k in [(5120, 17408), (34816, 5120), (1024, 5120)]:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ws = (2688 / w.float().abs().amax()).reshape(1)
        q, s = flashinfer.nvfp4_quantize(w, ws)
        for rows in [1, 2, 3, 4, 128, 1024]:
            print(f"smoke M={rows} N={n} K={k}",flush=True)
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            ins = (2688 / x.float().abs().amax()).reshape(1)
            layer = Nvfp4Linear(q, s, ins, 1 / (ins * ws), k, n)
            eager = layer(x)
            aq, asc = flashinfer.nvfp4_quantize(x, ins)
            from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float as unpack
            # Host-side independent oracle; never part of the serving path.
            xd = unpack(aq.cpu(), asc.cpu(), ins.reciprocal().cpu()).cuda()
            wd = unpack(q.cpu(), s.cpu(), ws.reciprocal().cpu()).cuda()
            reference = xd @ wd.T
            relative_error = ((eager.float()-reference).norm()/reference.norm()).item()
            assert relative_error < .005
            torch._dynamo.reset()  # independent shapes must not exhaust the shared code object's limit
            compiled = torch.compile(layer, fullgraph=True, dynamic=False)(x)
            torch.testing.assert_close(eager, compiled, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(eager, captured, rtol=0, atol=0)
            cosine = torch.nn.functional.cosine_similarity(eager.float().flatten(), (x.float() @ w.float().T).flatten(), dim=0).item()
            assert cosine > .95 and torch.isfinite(eager).all()
            assert torch.count_nonzero(layer(torch.zeros_like(x))) == 0
            records.append({"m": rows, "n": n, "k": k, "cosine_vs_bf16": cosine,
                            "relative_error_vs_fp4_reference":relative_error})
    return records


def summary(values):
    return {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values),
            "samples_ms": values, "tokens_per_second": 1000 / statistics.median(values)}


@torch.no_grad()
def verify_agreement(model, ids, forced, k):
    """Compare one-row and multi-row target math on identical prefixes, independent of drafts."""
    width=k+1
    room=ids.shape[1]+forced.numel()+width+16
    single=model.new_cache(1,room)
    multi=model.new_cache(1,room)
    multi.allocate_spec_buffers(width)
    model(ids,single)
    model(ids,multi)
    strong=near=0
    maxdiff=0.
    for start in range(0,min(64,forced.numel())-width+1,width):
        block=forced[start:start+width].reshape(1,width)
        logits=model.lm_head(model.model.language_model.forward_verify(block,multi)).float()
        multi.commit(torch.tensor([width],device=ids.device))
        for i in range(width):
            one=model.lm_head(model.model.language_model.forward_decode(block[:,i:i+1],single)).float()
            wrong=(one.argmax(-1)!=logits[:,i:i+1].argmax(-1)).item()
            top=one.topk(2,-1).values
            margin=(top[...,0]-top[...,1]).item()
            strong+=int(wrong and margin>.5)
            near+=int(wrong and margin<=.5)
            maxdiff=max(maxdiff,(one-logits[:,i:i+1]).abs().max().item())
    return {"strong_mismatches":strong,"near_tie_mismatches":near,"max_logit_diff":maxdiff}


@torch.no_grad()
def measure(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.weights)
    model = (Qwen3_5ForCausalLM.from_quantized_dir(args.nvfp4_from,device="cuda",kernels="fla-delta")
             if args.nvfp4_from else Qwen3_5ForCausalLM.from_pretrained_dir(args.weights, device="cuda", kernels="fla-delta", mtp=True))
    model.fuse_projections()
    quantize_linears(model.mtp, group_size=128)
    quant = None
    if args.precision == "int8":
        quant = quantize_linears(model, group_size=128).summary()
    elif args.precision == "nvfp4" and not args.nvfp4_from:
        quant = quantize_nvfp4(model, calibration_inputs(tok, args.out, args.calibration_samples,
                                                        revision=args.calibration_revision), scope=args.nvfp4_scope)
    (args.out / "quantization.json").write_text(json.dumps(quant, indent=2))
    if args.export:
        import shutil
        from infer_opt.nvfp4 import save_nvfp4
        prompt=tok('The capital of France is',return_tensors='pt').input_ids.cuda()
        before=model(prompt)[:,-1].clone()
        files=save_nvfp4(model,args.export)
        for f in Path(args.weights).iterdir():
            if (f.is_file() and (f.suffix in ['.json','.txt','.jinja'] or f.name=='LICENSE')
                    and 'safetensors.index' not in f.name and f.name != 'quantization.json'):
                shutil.copyfile(f,args.export/f.name)
        # The original config files must not replace our packed-layer metadata.
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()
        model=Qwen3_5ForCausalLM.from_quantized_dir(args.export,device='cuda',kernels='fla-delta')
        torch.testing.assert_close(model(prompt)[:,-1],before,rtol=0,atol=0)
        code=args.export/'code'
        shutil.copytree(ROOT/'infer_opt',code/'infer_opt',ignore=shutil.ignore_patterns('__pycache__'))
        (args.export/'calibration.json').write_bytes((args.out/'calibration.json').read_bytes())
        validation={'reload_logits_identical':True,'weight_bytes':sum(f.stat().st_size for f in files),
                    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (ROOT/'infer_opt').glob('*.py')}}
        (args.export/'validation.json').write_text(json.dumps(validation,indent=2))
        (args.out/'export_validation.json').write_text(json.dumps(validation,indent=2))
        if args.export_only: return validation
    texts = build_prompts(tok, CODE_PROMPT_SOURCE.read_text())
    prompts = {n: tok(texts[n], return_tensors="pt").input_ids.cuda() for n in PROMPTS}
    max_len = max(t.shape[1] for t in prompts.values()) + args.new_tokens + 16
    result = {"precision": args.precision, "args": {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "prompt_tokens": {name:ids.shape[1] for name,ids in prompts.items()},
              "timing_scope": "decode excludes prefill and first token; MTP includes all emitted round tokens, including final overshoot",
              "target_precision": args.precision, "mtp_precision": "int8-g128", "lm_head_precision": "bf16",
              "gpu":torch.cuda.get_device_name(), "torch":str(torch.__version__), "rows":[], "prefill":[]}
    def save():
        temporary=args.out / "measurements.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.out / "measurements.json")
    import torch._inductor.config as ic
    ic.max_autotune = True
    ic.coordinate_descent_tuning = True
    base = FixedShapeDecoder(model, model.new_cache(1, max_len), args.mode)
    expected = {}
    for name, ids in prompts.items():
        for _ in range(args.warmup):
            baseline(model, base, ids, args.new_tokens)
        torch.cuda.reset_peak_memory_stats()
        values = []
        for _ in range(args.repeats):
            tokens, ms = baseline(model, base, ids, args.new_tokens)
            values.append(ms)
        expected[name] = tokens.clone()
        result["rows"].append({"prompt":name,"method":"plain", **summary(values),
                               "peak_gib":torch.cuda.max_memory_allocated()/2**30})
        save()
    del base
    for k in args.ks:
        decoder = MTPSpeculativeDecoder(model, max_len, k=k, mode=args.mode)
        for name, ids in prompts.items():
            for _ in range(args.warmup):
                speculative(decoder, ids, args.new_tokens)
            torch.cuda.reset_peak_memory_stats()
            values = []
            for _ in range(args.repeats):
                tokens, ms, round_ms = speculative(decoder, ids, args.new_tokens)
                values.append(ms)
            same = tokens == expected[name]
            result["rows"].append({"prompt":name,"method":f"mtp{k}",**summary(values),
                "round_ms":round_ms, **decoder.stats.as_dict(), "token_agreement":same.float().mean().item(),
                "first_divergence":None if same.all() else int((~same).nonzero()[0]),
                "peak_gib":torch.cuda.max_memory_allocated()/2**30})
            if not bool(same.all()):
                result["rows"][-1]["same_prefix_verification"] = verify_agreement(model,ids,expected[name],k)
            save()
        del decoder
        torch._dynamo.reset()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    for length in [512,1024,4096]:
        ids = tok((texts["prose_1k"] * 8), return_tensors="pt").input_ids[:, :length].cuda()
        if ids.shape[1] != length:
            raise ValueError(f"prefill requires {length} tokens, got {ids.shape[1]}")
        values=[]
        for i in range(args.warmup + args.repeats):
            cache=model.new_cache(1,length+8)
            torch.cuda.synchronize()
            t=time.perf_counter()
            model(ids,cache)
            torch.cuda.synchronize()
            if i>=args.warmup:
                values.append((time.perf_counter()-t)*1000)
        result["prefill"].append({"tokens":ids.shape[1],"samples_ms":values,"median_ms":statistics.median(values),
                                  "scope":"eager model forward including lm_head, excluding cache allocation and tokenization"})
        save()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision",choices=["bf16","int8","nvfp4"],default="nvfp4")
    p.add_argument("--weights")
    p.add_argument("--nvfp4-from",type=Path)
    p.add_argument("--export",type=Path)
    p.add_argument("--export-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--new-tokens",type=int,default=256)
    p.add_argument("--ks",type=lambda s:[int(k) for k in s.split(',')] if s.strip() else [],default=[1,2,3],
                   help="comma-separated MTP k values; empty string measures plain decode only")
    p.add_argument("--mode",default="compile-reduce-overhead",choices=["eager","cudagraph","compile-reduce-overhead"])
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--calibration-samples",type=int,default=128)
    p.add_argument("--nvfp4-scope",choices=["all","mlp"],default="all")
    p.add_argument("--calibration-revision",default=None)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    require_sm120()
    torch.manual_seed(0)
    if a.smoke:
        (a.out/"smoke.json").write_text(json.dumps(smoke(),indent=2))
    else:
        if not a.weights:
            p.error("--weights is required unless --smoke")
        measure(a)


if __name__ == "__main__":
    main()
