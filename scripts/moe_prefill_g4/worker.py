"""Bounded dedicated-G4 prefill experiment. Launched from an uploaded source bundle."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

SOURCE = Path(__file__).resolve().parents[2]  # <run root>/source, as extracted by the launcher
ROOT = SOURCE.parent
OUT = ROOT / 'results'
CACHE = Path('/tmp/infer-opt-models') / ROOT.name
GATE_FAILED = 3  # a completed stage whose grouped candidate missed its gate: recorded, not fatal
REV = '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
BACKENDS = 'reference,sorted,grouped'
START = time.monotonic()
os.environ.update(HF_HOME=str(CACHE/'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
    TORCHINDUCTOR_CACHE_DIR=str(CACHE/'inductor'), TRITON_CACHE_DIR=str(CACHE/'triton'),
    CUDA_CACHE_PATH=str(CACHE/'cuda'), PIP_CACHE_DIR=str(CACHE/'pip'),
    TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def save(name, value):
    p = OUT/name
    p.write_text(json.dumps(value, indent=2) + '\n')


def snapshot(include_traces=False):
    with tarfile.open(ROOT/'results.tar.gz.tmp', 'w:gz', compresslevel=1) as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp') and (include_traces or not p.name.endswith('.trace.json')):
                data = p.read_bytes()
                info = tarfile.TarInfo(str(p.relative_to(OUT)))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    (ROOT/'results.tar.gz.tmp').replace(ROOT/'results.tar.gz')


def run(argv, name, timeout=1800, accept=(0,)):
    save('progress.json', {'stage': name, 'elapsed_s': time.monotonic()-START})
    save(name+'.command.json', argv)
    with (OUT/(name+'.stdout')).open('w') as stdout, (OUT/(name+'.stderr')).open('w') as stderr:
        p = subprocess.Popen(argv, cwd=SOURCE, stdout=stdout, stderr=stderr)
        deadline = min(time.monotonic()+timeout, START+5400)
        while p.poll() is None:
            if time.monotonic() > deadline:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                raise TimeoutError(name)
            snapshot()
            time.sleep(10)
    save(name+'.exit.json', {'returncode': p.returncode})
    snapshot()
    if p.returncode not in accept:
        raise RuntimeError(name+': '+(OUT/(name+'.stderr')).read_text()[-6000:])
    return p.returncode


def download():
    from huggingface_hub import snapshot_download
    save('progress.json', {'stage': 'download', 'elapsed_s': time.monotonic()-START})
    weights = snapshot_download('Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp', revision=REV,
        token=False, cache_dir=str(CACHE/'weights'),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    save('checkpoint.json', {'repo': 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp', 'revision': REV})
    return weights


def tune(gates):
    """Tile tuning on recorded expert loads, dense-projection kernels, then the tuned tiles in the model."""
    gates['tune'] = run([sys.executable, 'scripts/tune_moe_prefill_tiles.py',
         '--routes', 'artifacts/routes.json', '--out', str(OUT/'tune')], 'tune', 1500, accept=(0, GATE_FAILED))
    run([sys.executable, 'scripts/bench_w8a16_projection.py', '--out', str(OUT/'projections/projections.json')],
        'projections', 1200)
    save('gates.json', gates)
    weights = download()
    backends = 'reference,grouped,grouped-tuned' if gates['tune'] == 0 else 'reference,grouped'
    tuned = ['--grouped-config', str(OUT/'tune/best-config.json')] if gates['tune'] == 0 else []
    gates['model'] = run([sys.executable, 'scripts/bench_moe_prefill.py', '--weights', weights,
         '--out', str(OUT/'model'), '--lengths', '128,512,2048,4096', '--backends', backends, *tuned,
         '--prompts', 'math,korean', '--warmup', '2', '--repeats', '5', '--breakdown', '3',
         '--profile'], 'model', 2700, accept=(0, GATE_FAILED))
    gates['model_backends'] = backends
    save('gates.json', gates)


def main():
    import torch
    gpu = torch.cuda.get_device_properties(0)
    save('hardware.json', {'gpu': gpu.name, 'vram_gib': gpu.total_memory/2**30,
        'capability': torch.cuda.get_device_capability(), 'initial_torch': str(torch.__version__)})
    assert 'RTX PRO 6000' in gpu.name and gpu.total_memory > 90*2**30
    manifest = json.loads((SOURCE/'source-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest, name
    save('source-manifest.json', manifest)
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], 'remove-av')
    run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'triton==3.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu130'], 'install-torch')
    run([sys.executable, '-m', 'pip', 'install', 'pytest', 'transformers==5.22.0',
         'accelerate==1.15.0', 'safetensors==0.8.0', 'huggingface_hub==1.33.0',
         'flash-linear-attention==0.6.0', 'fla-core==0.6.0'], 'install')
    save('environment.json', {name: importlib.metadata.version(name) for name in
        ['torch', 'triton', 'transformers', 'flash-linear-attention', 'fla-core']})
    telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used',
        '--format=csv', '--loop-ms=1000'], stdout=(OUT/'gpu-telemetry.csv').open('w'))
    plan = json.loads((SOURCE/'run-plan.json').read_text())['plan']
    save('plan.json', {'plan': plan})
    try:
        run([sys.executable, '-m', 'pytest', 'tests/test_moe_prefill.py', 'tests/test_moe.py',
             '-k', 'not speculative and not verify and not compiled and not rejects', '-q'], 'tests', 900)
        gates = {}
        if plan == 'tune':
            tune(gates)
            return
        if plan == 'mtp':
            # MTP on/off greedy parity per prefill backend, target and MTP banks alike.
            weights = download()
            gates['mtp'] = run([sys.executable, 'scripts/validate_moe_prefill_mtp.py', '--weights', weights,
                 '--out', str(OUT/'mtp')], 'mtp', 3600, accept=(0, GATE_FAILED))
            save('gates.json', gates)
            # Diagnostic: parallel verification is not token-exact for this model under any backend.
            gates['mtp-parallel'] = run([sys.executable, 'scripts/validate_moe_prefill_mtp.py',
                 '--weights', weights, '--out', str(OUT/'mtp-parallel'), '--verifications', 'parallel',
                 '--ks', '1,2,3'], 'mtp-parallel', 2400, accept=(0, GATE_FAILED))
            save('gates.json', gates)
            return
        # Stage-wise FP64-referenced numerics of every backend on this GPU, before any timing.
        gates['numerics'] = run([sys.executable, 'scripts/diagnose_moe_prefill_numerics.py',
             '--out', str(OUT/'numerics/numerics.json')], 'numerics', 900, accept=(0, GATE_FAILED))
        gates['experts'] = run([sys.executable, 'scripts/bench_moe_prefill.py', '--out', str(OUT/'experts'),
             '--lengths', '128,512,2048', '--backends', BACKENDS, '--warmup', '2', '--repeats', '5',
             '--profile'], 'experts', 1200, accept=(0, GATE_FAILED))
        save('gates.json', gates)
        # Download only after the model-free correctness and timing experiment succeeds.
        weights = download()
        # The model always runs: route loads and section times do not depend on the grouped verdict.
        backends = BACKENDS if gates['experts'] == 0 else 'reference,sorted'
        gates['model'] = run([sys.executable, 'scripts/bench_moe_prefill.py', '--weights', weights,
             '--out', str(OUT/'model'), '--lengths', '128,512,2048,4096', '--backends', backends,
             '--prompts', 'math,korean', '--warmup', '2', '--repeats', '5', '--breakdown', '3',
             '--profile'], 'model', 2700, accept=(0, GATE_FAILED))
        gates['model_backends'] = backends
        save('gates.json', gates)
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=15)


if __name__ == '__main__':
    try:
        main()
        save('exit.json', {'exit_code': 0, 'elapsed_s': time.monotonic()-START})
    except BaseException:
        save('exit.json', {'exit_code': 1, 'traceback': traceback.format_exc(),
                          'elapsed_s': time.monotonic()-START})
        traceback.print_exc()
    finally:
        snapshot(include_traces=True)
