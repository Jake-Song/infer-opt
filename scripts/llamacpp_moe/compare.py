"""Same-session native infer-opt fused/down/block baselines; no checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infer_opt import moe_kernels as kernels
from infer_opt.quant import quantize_int8


def synth(count, salt, weight=False, offset=0):
    v = np.arange(offset, offset + count, dtype=np.uint32) + np.uint32(salt)
    v ^= v >> 16
    v *= np.uint32(0x7feb352d)
    v ^= v >> 15
    v *= np.uint32(0x846ca68b)
    v ^= v >> 16
    return ((v & 255).astype(np.int32) - 128).astype(np.float32) * np.float32(1 / (8192 if weight else 64))


def quantized(shape, salt):
    e, n, k = shape
    q = torch.empty(shape, device='cuda', dtype=torch.int8)
    scales = torch.empty(e, n, k // 128, device='cuda', dtype=torch.bfloat16)
    for start in range(0, e, 4):
        stop = min(e, start + 4)
        values = synth((stop-start) * n*k, salt, True, offset=start*n*k)
        raw = torch.from_numpy(values).to(device='cuda', dtype=torch.bfloat16).reshape(-1, k)
        qi, si = quantize_int8(raw, 128)
        q[start:stop] = qi.reshape(stop-start, n, k)
        scales[start:stop] = si.reshape(stop-start, n, k//128)
    return q, scales


def routes(t, routing):
    ids = []
    for token in range(t):
        row = []
        for s in range(8):
            slot = (s + token) % 8
            expert = slot if routing == 'shared' or routing == 'partial' and slot < 4 else 4*token + slot
            if routing == 'disjoint':
                expert = 8*token + slot
            row.append(expert)
        assert len(set(row)) == 8
        ids.append(row)
    return torch.tensor(ids, device='cuda', dtype=torch.int32)


def layout(t):
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'byte_stride': [s*t.element_size() for s in t.stride()]}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--implementation', choices=['reference','packed'])
    args = p.parse_args()
    output = Path(args.out)
    gu, gs = quantized((256,1024,2048), 901)
    down, ds = quantized((256,2048,512), 902)
    flush = torch.empty(128*2**20, device='cuda', dtype=torch.uint8)
    rows = []
    def save():
        output.write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
            'repeats':args.repeats,'prelude_cycles':300000,'rows':rows}, indent=2))

    for t in [1,2,4]:
        for routing in ['shared','partial','disjoint']:
            ids = routes(t, routing)
            x = torch.from_numpy(synth(t*2048,903)).to(device='cuda',dtype=torch.bfloat16).reshape(t,2048)
            dx = torch.from_numpy(synth(t*8*512,903)).to(device='cuda',dtype=torch.bfloat16).reshape(t*8,512)
            weights = torch.tensor([[(i+1)/36 for i in range(8)]]*t,device='cuda',dtype=torch.bfloat16)
            for implementation in ['reference','packed']:
                if args.implementation and implementation != args.implementation:
                    continue
                packed = implementation == 'packed' and t in (2,4)
                kernels.USE_PACKED_VERIFY = False
                kernels.LAUNCH_CONFIGS.clear()
                kernels.PACKED_PAIRS_CONFIGS.clear()
                if packed:
                    kernels.PACKED_PAIRS_CONFIGS[t] = (256,1)
                for kind in ['gate_up','down','block']:
                    if args.profile and (t != 4 or routing != 'shared' or kind != 'gate_up'):
                        continue
                    active_x = dx if kind == 'down' else x
                    y = torch.empty((t*8,2048 if kind == 'down' else 512),device='cuda',dtype=torch.bfloat16)
                    def run():
                        if kind == 'block':
                            return kernels.selected_int8_experts(x,ids,weights,gu,gs,down,ds)
                        q,s = (down,ds) if kind == 'down' else (gu,gs)
                        n,k = (2048,512) if kind == 'down' else (1024,2048)
                        if packed:
                            kernels._project_packed[(t*8,2048 if kind == 'down' else 512)](
                                active_x,q,s,ids,y,n,k,8,128,kind!='down',num_warps=1,enable_fp_fusion=False)
                        else:
                            kernels._project[(t*8,2048 if kind == 'down' else 512)](
                                active_x,q,s,ids,y,n,k,8,128,kind!='down',256,num_warps=4,enable_fp_fusion=False)
                        return y
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(10): run()
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    want = run().clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        got = run()
                    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    for cache in ['hot','cold']:
                        samples=[]
                        for _ in range(args.repeats):
                            if cache == 'cold': flush.zero_()
                            torch.cuda._sleep(300000)
                            a.record();graph.replay();b.record();b.synchronize()
                            samples.append(a.elapsed_time(b)*1000)
                        assert torch.equal(want,got)
                        assert torch.isfinite(got).all()
                        rows.append({'tokens':t,'routing':routing,'implementation':implementation,'packed_active':packed,
                            'kind':kind,'cache':cache,'median_us':statistics.median(samples),'gpu_samples_us':samples,
                            'replay_exact':True,'x':layout(active_x),'ids':layout(ids),'out':layout(got),
                            'ids_values':ids.cpu().tolist(),'flush_bytes':flush.numel() if cache=='cold' else 0,
                            'output_sha256':hashlib.sha256(got.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()})
                        save()
                        print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['gpu_samples_us','ids_values']}),flush=True)


if __name__ == '__main__':
    main()
