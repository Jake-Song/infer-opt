"""Pinned public-checkpoint A100 MTP experiment; run one decoder setting per process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--new-tokens', type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stages = []

    def save(name, value):
        temporary = out / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(out / name)

    def progress(stage, **extra):
        record = {'stage': stage, 'elapsed_s': time.monotonic() - started, **extra}
        save('progress.json', record)
        print(json.dumps(record), flush=True)

    def run(name, command, timeout=1200):
        progress(name)
        begin = time.monotonic()
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['TORCHINDUCTOR_COMPILE_THREADS'] = '4'
        with (out / (name + '.log')).open('w') as log:
            try:
                result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                status = {'returncode': result.returncode}
            except subprocess.TimeoutExpired:
                status = {'returncode': None, 'timed_out': True}
        status.update(name=name, command=command, elapsed_s=time.monotonic() - begin)
        stages.append(status)
        save('stages.json', stages)
        return status['returncode'] == 0

    try:
        import torch
        import triton
        from huggingface_hub import snapshot_download

        gpu = torch.cuda.get_device_properties(0)
        if 'A100' not in gpu.name or gpu.total_memory < 75 * 2**30:
            raise RuntimeError('This experiment requires an actual A100 with at least 75 GiB VRAM')
        environment = {
            'gpu': gpu.name, 'memory_gib': gpu.total_memory / 2**30,
            'compute_capability': [gpu.major, gpu.minor],
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'triton': triton.__version__, 'python': sys.version,
            'dependencies': {name: importlib.metadata.version(name) for name in
                             ('transformers', 'huggingface_hub', 'safetensors', 'flash-linear-attention')},
            'checkpoint': 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp',
            'revision': '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388',
            'inference_code_revision': 'b5ae4228a361fa562063fac7cd46dd19a611e9f2',
            'repeats': args.repeats, 'new_tokens': args.new_tokens,
            'prompts': ['korean', 'math', 'code_1k', 'prose_1k', 'chat_long'],
            'batch_size': 1, 'timing': 'synchronized host decode time; prefill and compilation excluded',
            'clock_policy': 'provider default; no clock locking',
        }
        save('environment.json', environment)
        smi = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
        (out / 'nvidia-smi-before.txt').write_text(smi.stdout)

        if not run('expert-kernels', [sys.executable, 'scripts/validate_qwen3_5_moe_kernels.py']):
            raise RuntimeError('A100 expert kernel correctness gate failed')
        (out / 'moe-kernel-validation.json').write_bytes((root / 'moe-kernel-validation.json').read_bytes())

        progress('download')
        weights = Path(snapshot_download(
            environment['checkpoint'], revision=environment['revision'], token=False,
            cache_dir=str(out / 'checkpoint-cache'),
            allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', '*.md', 'LICENSE']))
        progress('verify_checkpoint_hashes')
        manifest = json.loads((weights / 'release_manifest.json').read_text())
        for name, expected in manifest['files'].items():
            path = weights / name
            with path.open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if path.stat().st_size != expected['size'] or actual != expected['sha256']:
                raise RuntimeError('Checkpoint integrity failure: ' + name)
        save('checkpoint-integrity.json', {'revision': environment['revision'],
                                          'verified_files': len(manifest['files']), 'passed': True})
        smoke_ok = run('checkpoint-smoke', [sys.executable, 'scripts/validate_quantized_qwen3_5_moe.py',
                       '--weights', str(weights), '--out', str(out / 'checkpoint-validation.json')])
        if not smoke_ok:
            log = (out / 'checkpoint-smoke.log').read_text()
            if 'out of memory' in log.lower():
                raise RuntimeError('Checkpoint smoke test exceeded A100 memory; see log')
            # A parity failure is an experimental outcome. Detailed runs below
            # retain divergent outputs as failures rather than valid speedups.

        plan = [('cudagraph', verification, k) for verification in
                ('sequential', 'consistent', 'parallel') for k in (1, 2)]
        plan += [('compile-reduce-overhead', 'parallel', k) for k in (1, 2, 3)]
        plan += [('compile-reduce-overhead', 'consistent', 1)]
        summary = []
        for mode, verification, k in plan:
            name = f'{mode}-{verification}-k{k}'
            output = out / (name + '.json')
            command = [sys.executable, 'scripts/bench_qwen3_5_spec_decode.py',
                       '--repo', environment['checkpoint'], '--weights', str(weights),
                       '--configs', 'int8+mtp', '--mode', mode, '--verification', verification,
                       '--ks', str(k), '--new-tokens', str(args.new_tokens),
                       '--repeats', str(args.repeats), '--prompts', ','.join(environment['prompts']),
                       '--out', str(output)]
            if mode.startswith('compile'):
                command.append('--allow-inexact-verification')
            run(name, command)
            if output.exists():
                data = json.loads(output.read_text())
                for prompt in environment['prompts']:
                    rows = [r for r in data['rows'] if 'error' not in r and
                            r.get('prompt', '').split('/repeat-')[0] == prompt and str(k) in r.get('spec', {})]
                    if not rows:
                        continue
                    base = statistics.median(r['baseline_ms_per_token'] for r in rows)
                    spec = statistics.median(r['spec'][str(k)]['ms_per_token'] for r in rows)
                    failures = sum(r['spec'][str(k)]['first_divergence'] is not None for r in rows)
                    complete = len(rows) == args.repeats
                    summary.append({'mode': mode, 'verification': verification, 'k': k, 'prompt': prompt,
                                    'runs': len(rows), 'complete': complete,
                                    'baseline_median_ms': base, 'mtp_median_ms': spec,
                                    'round_median_ms': statistics.median(r['spec'][str(k)]['round_ms'] for r in rows),
                                    'tokens_per_round_median': statistics.median(r['spec'][str(k)]['tokens_per_round'] for r in rows),
                                    'divergent_runs': failures,
                                    'greedy_parity_passed': complete and failures == 0,
                                    'valid_speedup': base / spec if complete and failures == 0 else None,
                                    'timing_ratio': base / spec,
                                    'peak_allocated_gib': max(r['peak_mem_gib'] for r in rows)})
                save('summary.json', summary)
        smi = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
        (out / 'nvidia-smi-after.txt').write_text(smi.stdout)
        progress('complete', failed_stages=[s['name'] for s in stages if s['returncode'] != 0],
                 summary_rows=len(summary))
    except Exception as exc:
        (out / 'error.txt').write_text(traceback.format_exc())
        progress('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
