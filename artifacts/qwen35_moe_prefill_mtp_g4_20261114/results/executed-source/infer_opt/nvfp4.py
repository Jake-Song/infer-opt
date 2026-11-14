"""Native W4A4 NVFP4 linears. FlashInfer/ModelOpt are optional, lazily imported.

Weights are packed once; runtime never materializes a floating-point weight.
The CUTLASS SM120 backend is explicit: unsupported hardware fails closed.
"""
from __future__ import annotations

import torch
from torch import nn


def require_sm120() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("native NVFP4 experiment requires an SM120 CUDA GPU")


@torch.library.custom_op("infer_opt::nvfp4_mm", mutates_args=())
def nvfp4_mm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
             input_scale_inv: torch.Tensor, alpha: torch.Tensor, out_features: int) -> torch.Tensor:
    import flashinfer

    shape = x.shape
    flat = x.reshape(-1, shape[-1]).contiguous()
    # FlashInfer 0.6.6 SM120 large-M CUTLASS capture disagrees with eager.
    # Bound each native GEMM to the independently checked M<=128 path.
    outputs = []
    for chunk in flat.split(128):
        a, s = flashinfer.nvfp4_quantize(chunk, input_scale_inv,
                                       sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False, enable_pdl=False)
        outputs.append(flashinfer.mm_fp4(a, weight.T, s, scale.T, alpha, out_dtype=x.dtype,
                         backend="cutlass", block_size=16, use_nvfp4=True, enable_pdl=False))
    y = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return y.reshape(*shape[:-1], out_features)


@nvfp4_mm.register_fake
def _fake(x, weight, scale, input_scale_inv, alpha, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


class Nvfp4Linear(nn.Module):
    def __init__(self, weight, scale, input_scale_inv, alpha, in_features, out_features, bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        for name, value in (("weight", weight), ("scale", scale),
                            ("input_scale_inv", input_scale_inv), ("alpha", alpha), ("bias", bias)):
            self.register_buffer(name, value)

    def forward(self, x):
        y = nvfp4_mm(x, self.weight, self.scale, self.input_scale_inv, self.alpha, self.out_features)
        return y if self.bias is None else y + self.bias


def nvfp4_targets(model: nn.Module, scope: str = "all") -> list[str]:
    """Select native layers explicitly; calibration and packing must use this same set."""
    if scope not in ("all", "mlp"):
        raise ValueError(f"unknown NVFP4 scope: {scope}")
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
            and not n.startswith("mtp.") and not n.endswith("lm_head")
            and m.out_features >= 256 and m.in_features % 16 == 0
            and (scope == "all" or ".mlp." in n)]


@torch.no_grad()
def quantize_nvfp4(model: nn.Module, calibration: list[torch.Tensor], scope: str = "all") -> dict:
    """Calibrate fused BF16 target linears, then replace fake quantization by packed W4A4.

    MTP must already be INT8. Small gates and lm_head stay BF16.
    """
    import copy
    import flashinfer
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.quant_utils import get_activation_scaling_factor, get_weight_scaling_factor_2

    require_sm120()
    names = nvfp4_targets(model, scope)
    if not names or not calibration:
        raise ValueError("NVFP4 requires target linears and calibration inputs")
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    # Support ModelOpt releases with either the legacy dictionary or ordered-list config.
    entries = cfg["quant_cfg"]
    excluded = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n not in names]
    for name in excluded:
        for suffix in ("weight_quantizer", "input_quantizer"):
            pattern = f"{name}.{suffix}"
            if isinstance(entries, dict):
                entries[pattern] = {"enable": False}
            else:
                entries.append({"quantizer_name": pattern, "enable": False})

    def forward_loop(m):
        for ids in calibration:
            m(ids)

    mtq.quantize(model, cfg, forward_loop)
    report = {"format": "nvfp4-w4a4", "scope": scope, "backend": "cutlass", "modules": {}, "bytes": 0}
    for name in names:
        m = model.get_submodule(name)
        ws = get_weight_scaling_factor_2(m).float().reshape(1).clamp_min(1e-30)
        ins = get_activation_scaling_factor(m).float().reshape(1).clamp_min(1e-30)
        q, s = flashinfer.nvfp4_quantize(m.weight.detach().contiguous(), ws.reciprocal(),
                                       sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False, enable_pdl=False)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = Nvfp4Linear(q, s, ins.reciprocal(), ins * ws, m.in_features, m.out_features, m.bias)
        setattr(parent, child, layer)
        report["modules"][name] = {"in": m.in_features, "out": m.out_features,
                                    "input_scale": ins.item(), "weight_scale": ws.item()}
        report["bytes"] += sum(b.numel() * b.element_size() for b in layer.buffers())
    return report


@torch.no_grad()
def save_nvfp4(model, path, shard_bytes=2 << 30):
    """Save a fused native checkpoint, including the INT8 MTP head."""
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    from infer_opt.quant import Int8Linear

    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    # Remove disabled ModelOpt quantizers from floating-point exclusions.
    for name,m in list(model.named_modules()):
        if isinstance(m,nn.Linear) and type(m) is not nn.Linear:
            clean=nn.Linear(m.in_features,m.out_features,bias=m.bias is not None,device='meta')
            clean.weight=nn.Parameter(m.weight.detach(),requires_grad=False)
            if m.bias is not None: clean.bias=nn.Parameter(m.bias.detach(),requires_grad=False)
            parent,_,child=name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model,child,clean)
    modules={}
    for name,m in model.named_modules():
        if isinstance(m,(Nvfp4Linear,Int8Linear)):
            modules[name]={'kind':'nvfp4' if isinstance(m,Nvfp4Linear) else 'int8',
                           'in_features':m.in_features,'out_features':m.out_features,
                           'buffers':{n:{'shape':list(b.shape),'dtype':str(b.dtype).split('.')[-1]}
                                      for n,b in m.named_buffers()}}
    shards=[]
    current={}
    size=0
    for name,t in model.state_dict().items():
        nbytes=t.numel()*t.element_size()
        if current and size+nbytes>shard_bytes:
            file=path/f'model-nvfp4-{len(shards)+1:05d}.safetensors'
            save_file(current,str(file)); shards.append(file); current={}; size=0
        current[name]=t.detach().contiguous().cpu().clone()
        size+=nbytes
    if current:
        file=path/f'model-nvfp4-{len(shards)+1:05d}.safetensors'
        save_file(current,str(file)); shards.append(file)
    (path/'quantization.json').write_text(json.dumps({'format':'infer_opt.nvfp4','fused':True,
        'mtp':model.mtp is not None,'modules':modules},indent=2))
    return shards


def prepare_nvfp4(model,spec):
    from infer_opt.quant import Int8Linear
    model.fuse_projections()
    for name,info in spec['modules'].items():
        buffers={n:torch.empty(s['shape'],dtype=getattr(torch,s['dtype']),device='meta')
                 for n,s in info['buffers'].items()}
        if info['kind']=='nvfp4':
            layer=Nvfp4Linear(buffers['weight'],buffers['scale'],buffers['input_scale_inv'],buffers['alpha'],
                              info['in_features'],info['out_features'],buffers.get('bias'))
        elif info['kind']=='int8':
            layer=Int8Linear(buffers['weight'],buffers['scale'],buffers.get('bias'))
        else: raise ValueError('unknown native checkpoint layer kind')
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,layer)
