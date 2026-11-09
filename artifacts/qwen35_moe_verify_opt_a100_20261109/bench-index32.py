"""A100 selected-expert candidate validation/tuning without model downloads."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
spec = importlib.util.spec_from_file_location("index32_kernels", "/content/q35moe-verify-opt-20261109/moe-kernels-index32.py")
kernels = importlib.util.module_from_spec(spec); sys.modules[spec.name] = kernels; spec.loader.exec_module(kernels)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--kernel', choices=['shared', 'tiled', 'grouped', 'launch', 'index32'], default='shared')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    registry = {'shared': kernels.SHARED_ROUTE_CONFIGS, 'tiled': kernels.TILED_ROUTE_CONFIGS,
                'grouped': kernels.GROUPED_DOT_CONFIGS, 'index32': kernels.INT32_LAUNCH_CONFIGS, 'launch': kernels.LAUNCH_CONFIGS}[args.kernel]
    torch.manual_seed(901)
    e, h, m, group, top = 256, 2048, 512, 128, 8
    gu = torch.randint(-127, 128, (e, 2*m, h), device='cuda', dtype=torch.int8)
    down = torch.randint(-127, 128, (e, h, m), device='cuda', dtype=torch.int8)
    gs = (torch.rand(e, 2*m, h//group, device='cuda') * .0002 + .0001).bfloat16()
    ds = (torch.rand(e, h, m//group, device='cuda') * .0002 + .0001).bfloat16()
    flush = torch.empty(128 * 2**20, device='cuda', dtype=torch.uint8)
    rows, validation = [], []
    def run(x, ids, w):
        return kernels.selected_int8_experts(x, ids, w, gu, gs, down, ds)
    def timing(x, ids, w):
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): run(x, ids, w)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): result = run(x, ids, w)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(args.repeats):
            flush.zero_(); torch.cuda.synchronize()
            a.record(); graph.replay(); b.record(); b.synchronize()
            samples.append(a.elapsed_time(b))
        return {'median_ms': statistics.median(samples), 'samples_ms': samples}
    configs = [(256, 1, 4), (256, 2, 4), (256, 4, 4), (256, 2, 8),
               (128, 2, 4), (512, 1, 4), (512, 2, 4)]
    if args.kernel == 'tiled':
        configs = [(256, 2, 4), (256, 4, 4), (256, 8, 4), (256, 16, 4),
                   (256, 8, 8), (128, 8, 4), (512, 4, 4), (512, 8, 4)]
    if args.kernel == 'grouped':
        configs = [(128, 32, 4), (128, 64, 4), (256, 32, 4), (256, 64, 4), (64, 32, 4)]
    if args.kernel in ['launch', 'index32']:
        configs = [(256, 4), (512, 2), (2048, 2), (2048, 1)]
    for t in [2, 4]:
        x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        weights = torch.softmax(torch.randn(t, top, device='cuda'), -1).bfloat16()
        for routing in ['shared', 'partial', 'disjoint']:
            # Half the selections are shared; slots are intentionally permuted.
            ids = torch.arange(t * top, device='cuda').reshape(t, top)
            if routing == 'shared': ids = (ids % top + torch.arange(t, device='cuda')[:, None]) % top
            if routing == 'partial': ids[:, :4] = torch.arange(4, device='cuda')
            registry.clear()
            baseline = run(x, ids, weights).clone()
            baseline_time = timing(x, ids, weights)
            rows.append({'tokens': t, 'routing': routing, 'config': None, **baseline_time})
            for config in configs:
                registry[t] = config
                got = run(x, ids, weights)
                delta = (got.float() - baseline.float()).abs()
                row = {'tokens': t, 'routing': routing, 'config': config,
                       'exact': bool(torch.equal(got, baseline)), 'max_abs_diff': delta.max().item(),
                       'different_elements': (delta != 0).sum().item(), **timing(x, ids, weights)}
                row['speedup'] = baseline_time['median_ms'] / row['median_ms']
                rows.append(row)
                save(out / 'tuning.json', {'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
                    'flush_bytes': flush.numel(), 'rows': rows})
                print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}), flush=True)
    # One selected candidate must also handle T=1/3/5, changing routes, compile and graphs.
    for t in [1, 2, 3, 4, 5]:
        x = torch.randn(t, h, device='cuda', dtype=torch.bfloat16)
        weights = torch.softmax(torch.randn(t, top, device='cuda'), -1).bfloat16()
        ids = torch.arange(t*top, device='cuda').reshape(t, top) % 16
        config = (128, 32, 4) if args.kernel == 'grouped' else (256, 2, 4)
        if args.kernel in ['launch', 'index32']: config = (256, 2)
        registry.clear()
        baseline = run(x, ids, weights).clone()
        registry[t] = config
        got = run(x, ids, weights)
        compiled = torch.compile(run, fullgraph=True)
        compiled_out = compiled(x, ids, weights)
        torch.testing.assert_close(compiled_out, got, atol=0, rtol=0)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): run(x, ids, weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = run(x, ids, weights)
        ids.copy_((ids + 31) % e)
        changed = run(x, ids, weights)
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(captured, changed, atol=0, rtol=0)
        torch.testing.assert_close(compiled(x, ids, weights), changed, atol=0, rtol=0)
        validation.append({'tokens': t, 'config': config, 'baseline_exact': bool(torch.equal(got, baseline)),
            'max_abs_diff': (got.float()-baseline.float()).abs().max().item(),
            'graph_dynamic_routes_equal': True, 'compiled_dynamic_routes_equal': True})
        save(out / 'validation.json', {'rows': validation})
    save(out / 'complete.json', {'complete': True})


if __name__ == '__main__':
    main()
