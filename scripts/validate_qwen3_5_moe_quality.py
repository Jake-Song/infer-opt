"""Sequential BF16 HF / BF16 infer-opt / INT8 numerical validation (not task accuracy)."""
import argparse,gc,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, FixedShapeDecoder
from infer_opt.quant import quantize_linears
from infer_opt.moe import quantize_moe_experts
from validate_qwen3_5_hf import build_prompts,CODE_PROMPT_SOURCE

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--tokens',type=int,default=48)
    p.add_argument('--int8-mode',default='eager',choices=['eager','cudagraph','compile-reduce-overhead'])
    p.add_argument('--gemv-configs',default=None)
    p.add_argument('--bf16-configs',default=None)
    p.add_argument('--fuse',action='store_true')
    a=p.parse_args()
    if a.gemv_configs:
        from infer_opt import quant_kernels
        for n,k,g,bn,bk,w in json.loads(Path(a.gemv_configs).read_text())['best']:
            quant_kernels.LAUNCH_CONFIGS[(n,k,g)]=(bn,bk,w)
    from transformers import AutoTokenizer,Qwen3_5MoeForCausalLM
    tok=AutoTokenizer.from_pretrained(a.weights)
    prompts=build_prompts(tok,CODE_PROMPT_SOURCE.read_text())
    hf, loading = Qwen3_5MoeForCausalLM.from_pretrained(a.weights,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys'):
        raise RuntimeError(f'incomplete HF reference: {loading}')
    hf.eval()
    refs={}
    for name,text in prompts.items():
        ids=tok(text,return_tensors='pt').input_ids.cuda()
        prefix=ids.shape[1]
        cache=None;step=ids;logits=[];tokens=[]
        for i in range(a.tokens):
            result=hf(input_ids=step,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values
            last=result.logits[:,-1].float()
            logits.append(last.cpu());step=last.argmax(-1,keepdim=True);tokens.append(step)
        full=torch.cat([ids]+tokens,dim=1).cpu()
        refs[name]={'ids':full,'prefix':prefix,'logits':torch.cat(logits)}
        del result,cache,logits,tokens,ids,step,last
        print('HF',name,flush=True)
    del hf;gc.collect();torch.cuda.empty_cache()
    model=Qwen3_5ForCausalLM.from_pretrained_dir(a.weights,device='cuda',kernels='fla-delta')
    if a.fuse:model.fuse_projections()
    rows=[]
    for precision in ('bf16','int8'):
        if precision=='int8':
            quantize_moe_experts(model,128);quantize_linears(model,group_size=128)
            if a.bf16_configs:
                from infer_opt.bf16_gemv import install_bf16_gemv
                table={(n,k):(bn,bk,w) for n,k,bn,bk,w in json.loads(Path(a.bf16_configs).read_text())["best"]}
                install_bf16_gemv(model,table)
            gc.collect();torch.cuda.empty_cache()
        decoder = (FixedShapeDecoder(model, model.new_cache(1,max(r['ids'].shape[1] for r in refs.values())+1), a.int8_mode)
                   if precision=='int8' and a.int8_mode!='eager' else None)
        for name,ref in refs.items():
            ids=ref['ids'].cuda();prefix=ref['prefix']
            cache=decoder.cache if decoder is not None else model.new_cache(1,ids.shape[1]+1)
            cache.reset()
            actual=[model(ids[:,:prefix],cache)[:,-1].float().cpu()]
            for i in range(a.tokens-1):
                if decoder is not None:
                    if i==0: decoder.prepare()
                    logits=decoder.step(ids[:,prefix+i:prefix+i+1])
                else:
                    logits=model.decode_step(ids[:,prefix+i:prefix+i+1],cache)
                actual.append(logits[:,-1].float().cpu())
            actual=torch.cat(actual);want=ref['logits']
            lp,lq=want.log_softmax(-1),actual.log_softmax(-1)
            mismatch=want.argmax(-1)!=actual.argmax(-1)
            top=want.topk(2,-1).values;margin=top[:,0]-top[:,1]
            row={'precision':precision,'prompt':name,'teacher_forced_tokens':a.tokens,
                 'mean_kl':float((lp.exp()*(lp-lq)).sum(-1).mean()),
                 'top1_agreement':float((~mismatch).float().mean()),
                 'strong_mismatches':int((mismatch & (margin>.5)).sum()),
                 'mismatch_steps':mismatch.nonzero().flatten().tolist()}
            rows.append(row);print(json.dumps(row),flush=True)
            Path(a.out).write_text(json.dumps({'rows':rows,'reference':'HF BF16 cached greedy','tokens':a.tokens,'int8_mode':a.int8_mode,'gemv_configs':a.gemv_configs,'bf16_configs':a.bf16_configs,'fused_projections':a.fuse},indent=2))
            del cache,actual,ids
if __name__=='__main__': main()
