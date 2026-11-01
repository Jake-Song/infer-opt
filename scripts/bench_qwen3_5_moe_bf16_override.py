"""Paired end-to-end plain-INT8 benchmark of explicit BF16 GEMV runtime overrides."""
import argparse,gc,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM,FixedShapeDecoder
from infer_opt.bf16_gemv import install_bf16_gemv
from bench_qwen3_5_spec_decode import baseline
from validate_qwen3_5_hf import build_prompts,CODE_PROMPT_SOURCE

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--configs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.weights)
    texts=build_prompts(tok,CODE_PROMPT_SOURCE.read_text())
    prompts={n:tok(texts[n],return_tensors='pt').input_ids.cuda() for n in ('korean','math','code_1k','prose_1k','chat_long')}
    model=Qwen3_5ForCausalLM.from_quantized_dir(a.weights,device='cuda',kernels='fla-delta');model.fuse_projections()
    table={(n,k):(bn,bk,w) for n,k,bn,bk,w in json.loads(Path(a.configs).read_text())['best']}
    rows=[];reference={};max_len=max(x.shape[1] for x in prompts.values())+280
    for variant in ('stock','bf16-gemv'):
        if variant=='bf16-gemv':install_bf16_gemv(model,table)
        torch._dynamo.reset()
        decoder=FixedShapeDecoder(model,model.new_cache(1,max_len),'compile-reduce-overhead')
        start=time.perf_counter()
        for ids in prompts.values():baseline(model,decoder,ids,2)
        torch.cuda.synchronize();warmup_s=time.perf_counter()-start
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(5):
            for name,ids in prompts.items():
                tokens,ms=baseline(model,decoder,ids,256)
                if variant=='stock' and repeat==0:reference[name]=tokens.cpu()
                row={'variant':variant,'prompt':name,'repeat':repeat,'ms_per_token':ms,
                     'model_ttft_ms':baseline.last_ttft_ms,'warmup_s':warmup_s,
                     'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                     'tokens_equal_stock':torch.equal(tokens.cpu(),reference[name]),'tokens':tokens.cpu().tolist()}
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='tokens'}),flush=True)
                Path(a.out).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'rows':rows,'configs':a.configs},indent=2))
        del decoder;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
