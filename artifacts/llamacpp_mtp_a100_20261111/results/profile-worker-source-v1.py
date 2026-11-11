"""Collect separate CUDA node traces from the unmodified native server."""
import json
from pathlib import Path
import subprocess
import time
import traceback

import worker as bench


def main():
    while not (bench.OUT / 'exit.json').exists():
        time.sleep(5)
    assert json.loads((bench.OUT / 'exit.json').read_text())['returncode'] == 0
    model = Path(json.loads((bench.OUT / 'model-manifest.json').read_text())['cache_path'])
    prompts = json.loads((bench.ROOT / 'inputs.json').read_text())
    trace_dir = bench.OUT / 'traces'
    trace_dir.mkdir(exist_ok=True)
    session = None
    try:
        for k in [0, 1, 2, 3]:
            label = f'trace-k{k}'
            session = f'llama-mtp-k{k}'
            bench.save('profile-progress.json', {'stage': label, 'k': k})
            args = json.loads((bench.OUT / f'batch0-k{k}-server.command.json').read_text())
            args[args.index('--port') + 1] = '18892'
            bench.BASE_URL = 'http://127.0.0.1:18892'
            launch = ['nsys', 'launch', '--session-new=' + session, '--trace=cuda,nvtx',
                      '--cuda-graph-trace=node'] + args
            bench.run(launch, label + '-launch', timeout=120)
            for attempt in range(360):
                try:
                    if bench.api('/health', timeout=2).get('status') == 'ok':
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('profile server readiness')
            for name, prompt in prompts.items():
                # Warm the injected process outside the collection window.
                for repeat in range(3):
                    bench.request(prompt, k, f'{label}-{name}-warmup{repeat}')
                report = trace_dir / f'{label}-{name}'
                bench.run(['nsys', 'start', '--session=' + session, '--sample=none',
                           '--cpuctxsw=none', '--force-overwrite=true', '--export=sqlite',
                           '--output=' + str(report)], f'{label}-{name}-start', timeout=120)
                try:
                    bench.request(prompt, k, f'{label}-{name}-profiled')
                finally:
                    bench.run(['nsys', 'stop', '--session=' + session], f'{label}-{name}-stop', timeout=180)
                bench.run(['nsys', 'stats', '--report=cuda_gpu_kern_sum', '--format=csv',
                           str(report) + '.nsys-rep'], f'{label}-{name}-kernel-stats', timeout=180)
            bench.run(['nsys', 'shutdown', '--session=' + session, '--kill=sigterm'], label + '-shutdown')
            session = None
            bench.snapshot()
        bench.save('profile-exit.json', {'returncode': 0, 'traces': 8,
                   'boundary': 'whole request, includes prefill; separate from unprofiled timing'})
    finally:
        if session:
            bench.run(['nsys', 'shutdown', '--session=' + session, '--kill=sigterm'],
                      'profile-failure-shutdown', check=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        bench.save('profile-exit.json', {'returncode': 1, 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        bench.snapshot()
