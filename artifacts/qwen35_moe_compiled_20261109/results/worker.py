"""Dedicated G4 or A100 80GB: HF quality and compiled parallel MTP k=1,2,3."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/content/q35moe-compiled-20261109')
OUT = ROOT / 'results'
CACHE = Path('/tmp/infer-opt-models/q35moe-compiled-20261109')
ARCHIVE = Path('/content/q35moe-compiled-20261109-results.tar.gz')
BUNDLE = Path(__file__).resolve().parent
HF_REV = '59d61f3ce65a6d9863b86d2e96597125219dc754'
INT8_REV = '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
START = time.time()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def snapshot():
    temporary = ARCHIVE.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.tmp':
                archive.add(path, arcname=str(path.relative_to(OUT)))
    temporary.replace(ARCHIVE)


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)
    snapshot()


def run(command, name, timeout=2400):
    save(name + '.command.json', command)
    with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
    save(name + '.exit.json', {'returncode': process.returncode})
    snapshot()
    if process.returncode:
        raise RuntimeError(name + ' failed: ' + (OUT / (name + '.stderr')).read_text()[-5000:])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(32 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if ROOT.exists():
        raise RuntimeError('dedicated experiment directory already exists')
    OUT.mkdir(parents=True)
    os.environ.update(HF_HOME=str(CACHE / 'hf-home'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'), TRITON_CACHE_DIR=str(CACHE / 'triton'),
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', TORCH_LOGS='perf_hints')
    shutil.copy2(BUNDLE / 'worker.py', OUT / 'worker.py')
    source = ROOT / 'infer-opt'
    source.mkdir()
    with tarfile.open(BUNDLE / 'source.tar.gz', 'r:gz') as archive:
        archive.extractall(source, filter='data')
    save('source-manifest.json', json.loads((BUNDLE / 'source-manifest.json').read_text()))
    for name in ('bench_qwen3_5_moe_compiled.py', 'validate_compiled_qwen3_5_moe_quality.py',
                 'validate_llamacpp_qwen3_5_quality.py', 'validate_qwen3_5_hf.py'):
        shutil.copy2(source / 'scripts' / name, OUT / name)
    stage('hardware')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    supported = ('RTX PRO 6000' in gpu.name and gpu.total_memory >= 90 * 2**30
                 or 'A100' in gpu.name and gpu.total_memory >= 75 * 2**30)
    if not supported:
        raise RuntimeError('expected RTX PRO 6000 Blackwell >=90GiB or A100 >=75GiB')
    environment = {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                   'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'capability': torch.cuda.get_device_capability(), 'python': sys.version}
    save('environment.json', environment)
    run(['nvidia-smi', '-q'], 'gpu-before')
    stage('dependencies')
    # Fresh runtimes may ship a newer torch; keep dense/MoE compiler versions aligned.
    # Vision/audio packages are unnecessary here and otherwise retain incompatible torch pins.
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], 'remove-unused-av')
    run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'triton==3.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu130'], 'install-torch', timeout=1200)
    run([sys.executable, '-m', 'pip', 'install', 'transformers==5.22.0', 'accelerate==1.15.0',
         'safetensors==0.8.0', 'huggingface_hub==1.33.0',
         'flash-linear-attention==0.6.0', 'fla-core==0.6.0'], 'install', timeout=900)
    environment.update(json.loads(subprocess.check_output([sys.executable, '-c',
        'import json,torch;print(json.dumps({"torch":str(torch.__version__),"cuda":torch.version.cuda}))'], text=True)))
    environment['packages'] = {name: importlib.metadata.version(name) for name in
        ('torch', 'triton', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub',
         'flash-linear-attention', 'fla-core', 'numpy')}
    save('environment.json', environment)
    manifest = {'hf_repo': 'Qwen/Qwen3.5-35B-A3B', 'hf_revision': HF_REV,
        'int8_repo': 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp', 'int8_revision': INT8_REV,
        'cache_root': str(CACHE), 'cache_capacity': 2048, 'batch': 1,
        'quality': {'mtp': False, 'prompts': 9, 'positions': 48,
                    'modes': ['eager', 'compile-reduce-overhead']},
        'benchmark': {'verification': 'parallel', 'ks': [1, 2, 3], 'tokens': 256,
                      'warmup': 3, 'repeats': 5, 'mode': 'compile-reduce-overhead'},
        'compile': {'mode': 'reduce-overhead', 'fullgraph': True, 'dynamic': False},
        'fused_projections': True, 'max_autotune': True, 'coordinate_descent_tuning': True,
        'bf16_gemv_overrides': False}
    save('manifest.json', manifest)
    from huggingface_hub import snapshot_download, HfApi
    stage('download_hf_reference')
    hf_cache = CACHE / 'reference'
    weights = snapshot_download(manifest['hf_repo'], revision=HF_REV, token=False, cache_dir=str(hf_cache),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    reference = CACHE / 'reference-logits'
    stage('hf_bf16_reference')
    run([sys.executable, str(source / 'scripts/validate_llamacpp_qwen3_5_quality.py'),
         'reference', '--weights', weights, '--reference-dir', str(reference),
         '--reference-revision', HF_REV, '--tokens', '48'], 'hf-reference')
    shutil.copy2(reference / 'reference.json', OUT / 'reference.json')
    save('reference-logit-hashes.json', {p.name: sha256(p) for p in sorted(reference.glob('*.npy'))})
    shutil.rmtree(hf_cache)
    stage('download_int8_checkpoint')
    weights = snapshot_download(manifest['int8_repo'], revision=INT8_REV, token=False,
        cache_dir=str(CACHE / 'int8'), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    stage('verify_int8_checkpoint')
    info = HfApi().model_info(manifest['int8_repo'], revision=INT8_REV, files_metadata=True, token=False)
    hashes = []
    for entry in info.siblings:
        path = Path(weights) / entry.rfilename
        if not path.is_file():
            continue
        actual = sha256(path)
        if entry.lfs and actual != entry.lfs.sha256:
            raise RuntimeError('checkpoint hash mismatch: ' + entry.rfilename)
        hashes.append({'file': entry.rfilename, 'bytes': path.stat().st_size, 'sha256': actual,
                       'hub_lfs_verified': entry.lfs is not None})
    save('checkpoint-hashes.json', hashes)
    stage('mtp_off_eager_and_compiled_quality')
    run([sys.executable, str(source / 'scripts/validate_compiled_qwen3_5_moe_quality.py'),
         '--weights', weights, '--reference-dir', str(reference), '--out', str(OUT / 'quality'),
         '--scratch', str(CACHE / 'eager-logits'), '--cache-capacity', '2048', '--max-autotune'], 'quality')
    stage('compiled_parallel_mtp_k123_benchmark')
    run([sys.executable, str(source / 'scripts/bench_qwen3_5_moe_compiled.py'),
         '--weights', weights, '--out', str(OUT / 'benchmark'), '--cache-capacity', '2048',
         '--new-tokens', '256', '--repeats', '5', '--warmup', '3', '--ks', '1,2,3'],
         'benchmark', timeout=3600)
    run(['nvidia-smi', '-q'], 'gpu-after')
    stage('complete')


if __name__ == '__main__':
    try:
        main()
        save('exit.json', {'exit_code': 0, 'elapsed_s': time.time() - START})
    except BaseException:
        save('exit.json', {'exit_code': 1, 'elapsed_s': time.time() - START, 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        snapshot()
