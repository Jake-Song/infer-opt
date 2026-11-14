"""Convert local Qwen3.5 MoE BF16 weights to W8A16 without a full model in RAM/VRAM.

CPU conversion bounds staging to one packed bank plus one output shard. MTP is
included. Output is the custom infer-opt format, not a Transformers checkpoint.
"""
import argparse
import hashlib
import resource
import json
import shutil
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from infer_opt.moe import Qwen3_5Experts
from infer_opt.quant import quantize_int8


def convert(source, destination, group=128, shard_bytes=2 << 30):
    if group <= 0:
        raise ValueError("group size must be positive")
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('destination must be empty')
    config=Qwen3_5TextConfig.from_json(source/'config.json')
    if not config.num_experts or config.tie_word_embeddings:
        raise ValueError('this converter requires an untied Qwen3.5 MoE model')
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    with torch.device('meta'):
        model=Qwen3_5ForCausalLM(config,mtp=True)
    modules={}
    for name,m in model.named_modules():
        if isinstance(m,Qwen3_5Experts):
            if config.hidden_size % group or config.moe_intermediate_size % group:
                raise ValueError('group must divide expert dimensions')
            modules[name]={'type':'int8_experts','group_size':group,'num_experts':config.num_experts,
                           'hidden_size':config.hidden_size,'intermediate_size':config.moe_intermediate_size}
        elif isinstance(m,torch.nn.Linear) and m.out_features>=256 and m.in_features%group==0 and name!='lm_head' and not name.endswith(('mlp.gate','mlp.shared_expert_gate')):
            modules[name]={'in_features':m.in_features,'out_features':m.out_features,'group_size':group,'bias':m.bias is not None}
    destination.mkdir(parents=True,exist_ok=True)
    shard={}; size=0; files=[]
    def flush():
        nonlocal shard,size
        if not shard: return
        name=f'model-int8-{len(files)+1:05d}.safetensors'
        save_file(shard,str(destination/name));files.append(name)
        shard={};size=0
    def put(name,tensor):
        nonlocal size
        n=tensor.numel()*tensor.element_size()
        if size+n>shard_bytes: flush()
        shard[name]=tensor.contiguous();size+=n
    def read(name,expert=None):
        with safe_open(source/index[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name) if expert is None else f.get_slice(name)[expert]
    expected=model.state_dict()
    for name,param in expected.items():
        parent,_,leaf=name.rpartition('.')
        spec=modules.get(parent)
        if spec and spec.get('type')=='int8_experts':
            e,n,k=param.shape
            q=torch.empty((e,n,k),dtype=torch.int8)
            scales=torch.empty((e,n,k//group),dtype=torch.bfloat16)
            for i in range(e):
                if name in index:
                    weight=read(name,i)
                elif leaf=='gate_up_proj':
                    weight=torch.cat([read(f'{parent}.{i}.gate_proj.weight'),read(f'{parent}.{i}.up_proj.weight')])
                else:
                    weight=read(f'{parent}.{i}.down_proj.weight')
                if tuple(weight.shape)!=(n,k): raise ValueError(f'invalid shape: {name}')
                q[i],scales[i]=quantize_int8(weight,group)
            put(name,q)
            put(parent+('.gate_up_scale' if leaf=='gate_up_proj' else '.down_scale'),scales)
        else:
            weight=read(name)
            if tuple(weight.shape)!=tuple(param.shape): raise ValueError(f'invalid shape: {name}')
            if spec and leaf=='weight':
                q,scale=quantize_int8(weight,group)
                put(name,q);put(parent+'.scale',scale)
            else: put(name,weight)
        print(name,flush=True)
    flush()
    for name in ('config.json','tokenizer.json','tokenizer_config.json','generation_config.json','vocab.json','merges.txt','special_tokens_map.json','chat_template.jinja','LICENSE'):
        if (source/name).exists(): shutil.copyfile(source/name,destination/name)
    (destination/'quantization.json').write_text(json.dumps({'format':'infer_opt.int8','modules':modules,'tied':{},'mtp':True},indent=2))
    (destination/'conversion_info.json').write_text(json.dumps({
        'source':str(source.resolve()), 'group_size':group, 'device':'cpu',
        'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
        'source_index_sha256':hashlib.sha256((source/'model.safetensors.index.json').read_bytes()).hexdigest(),
        'shards':files, 'checkpoint_bytes':sum((destination/f).stat().st_size for f in files),
        'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        'cuda_initialized':torch.cuda.is_initialized(),
    },indent=2))
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--group-size',type=int,default=128)
    a=p.parse_args()
    if a.group_size<=0: p.error('group size must be positive')
    convert(a.source,a.out,a.group_size)
