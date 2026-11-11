"""One G4 run: compiled ordinary decode versus compiled parallel MTP."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/tmp/moe-compiled-g4-20261112')
OUT = ROOT / 'results'
SOURCE = ROOT / 'source'
CACHE = Path('/tmp/infer-opt-models/moe-compiled-g4-20261112')
REPO = 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp'
REV = '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
START = time.time()
os.environ.update(HF_HOME=str(CACHE / 'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
    HF_HUB_DISABLE_TELEMETRY='1', TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'),
    TRITON_CACHE_DIR=str(CACHE / 'triton'), CUDA_CACHE_PATH=str(CACHE / 'cuda'),
    TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
    TORCH_LOGS='perf_hints', PIP_CACHE_DIR=str(CACHE / 'pip'))
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(p)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot():
    temporary = ROOT / 'results.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                data = p.read_bytes()
                entry = tarfile.TarInfo(str(p.relative_to(OUT)))
                entry.size = len(data)
                tar.addfile(entry, io.BytesIO(data))
    temporary.replace(ROOT / 'results.tar.gz')


def stage(name):
    value = {'stage': name, 'elapsed_s': time.time() - START}
    save('progress.json', value)
    print(json.dumps(value), flush=True)
    snapshot()


def run(argv, name, timeout=1800):
    save(name + '.command.json', argv)
    with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
        p = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
        save(name + '.process.json', {'pid': p.pid})
        deadline = time.monotonic() + timeout
        while p.poll() is None:
            if time.monotonic() > deadline:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                raise TimeoutError(name)
            snapshot()
            time.sleep(10)
    save(name + '.exit.json', {'returncode': p.returncode})
    snapshot()
    if p.returncode:
        raise RuntimeError(name + ': ' + (OUT / (name + '.stderr')).read_text()[-6000:])


def main():
    if (OUT / 'launch-guard.json').exists():
        raise RuntimeError('Do not start duplicate workers')
    save('launch-guard.json', {'pid': os.getpid(), 'started_epoch': START})
    for name in ['worker.py', 'preflight.json', 'source-manifest.json', 'expected-inputs.json']:
        shutil.copy2(ROOT / name, OUT / name)
    save('worker-manifest.json', {'worker.py': sha256(ROOT / 'worker.py')})
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha256(SOURCE / name) == expected, name
    stage('hardware')
    run([sys.executable, '-c', 'import torch,json; p=torch.cuda.get_device_properties(0); '
         'assert "RTX PRO 6000" in p.name and p.total_memory>90*2**30; '
         'assert torch.cuda.get_device_capability()==(12,0); '
         'print(json.dumps({"gpu":p.name,"vram_gib":p.total_memory/2**30}))'], 'hardware')
    run(['nvidia-smi', '-q'], 'gpu-before')
    stage('dependencies')
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], 'remove-unused-av')
    run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'triton==3.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu130'], 'install-torch', timeout=1200)
    run([sys.executable, '-m', 'pip', 'install', 'transformers==5.22.0', 'accelerate==1.15.0',
         'safetensors==0.8.0', 'huggingface_hub==1.33.0',
         'flash-linear-attention==0.6.0', 'fla-core==0.6.0'], 'install', timeout=1200)
    packages = {name: importlib.metadata.version(name) for name in
        ['torch', 'triton', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub',
         'flash-linear-attention', 'fla-core', 'numpy']}
    save('environment.json', {'packages': packages, 'python': sys.version,
         'gpu': json.loads((OUT / 'hardware.stdout').read_text()),
         'sm_clock': 'provider automatic boost; no clock lock or warp override',
         'source_revision': manifest['base_revision']})
    save('contract.json', {'checkpoint_repo': REPO, 'checkpoint_revision': REV,
         'mode': 'compile-reduce-overhead', 'verification': 'parallel', 'ks': [1, 2, 3],
         'cache_capacity': 2048, 'new_tokens': 256, 'warmup': 3, 'repeats': 5,
         'max_autotune': True, 'coordinate_descent_tuning': True,
         'allow_inexact': True, 'warp_override': False, 'hf_reference_rerun': False})
    from huggingface_hub import HfApi, snapshot_download
    stage('download-and-verify-int8')
    weights = snapshot_download(REPO, revision=REV, token=False, cache_dir=str(CACHE / 'weights'),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    info = HfApi(token=False).model_info(REPO, revision=REV, files_metadata=True)
    files = []
    for entry in info.siblings:
        p = Path(weights) / entry.rfilename
        if p.is_file():
            actual = sha256(p)
            if entry.lfs:
                assert actual == entry.lfs.sha256, entry.rfilename
            files.append({'file': entry.rfilename, 'bytes': p.stat().st_size,
                          'sha256': actual, 'hub_lfs_verified': entry.lfs is not None})
    save('checkpoint-hashes.json', files)
    stage('compiled-parallel-benchmark')
    with (OUT / 'gpu-telemetry.csv').open('w') as telemetry_file:
        telemetry = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used',
            '--format=csv', '--loop-ms=1000'], stdout=telemetry_file, stderr=subprocess.DEVNULL)
        try:
            run([sys.executable, str(SOURCE / 'scripts/bench_qwen3_5_moe_compiled.py'),
                 '--weights', weights, '--out', str(OUT / 'benchmark'), '--cache-capacity', '2048',
                 '--new-tokens', '256', '--repeats', '5', '--warmup', '3', '--ks', '1,2,3'],
                 'benchmark', timeout=5400)
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=15)
    actual = json.loads((OUT / 'benchmark/inputs.json').read_text())
    expected = json.loads((ROOT / 'expected-inputs.json').read_text())
    for name in expected:
        assert actual[name]['input_ids'] == expected[name]['input_ids'], name
    save('input-validation.json', {'matches_prior_a100_inputs': True,
         'tokens': {name: len(row['input_ids']) for name, row in actual.items()}})
    run(['nvidia-smi', '-q'], 'gpu-after')
    stage('complete')


if __name__ == '__main__':
    try:
        main()
        save('exit.json', {'exit_code': 0, 'elapsed_s': time.time() - START})
    except BaseException:
        save('exit.json', {'exit_code': 1, 'elapsed_s': time.time() - START,
                          'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        snapshot()
