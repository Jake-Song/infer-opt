"""A100 round profiling: production CUDA Graphs and separately measured diagnostic phases."""
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

ROOT = Path('/content/q35moe-round-profile-20261109')
OUT = ROOT / 'results'
CACHE = Path('/tmp/infer-opt-models/q35moe-round-profile-20261109')
ARCHIVE = Path('/content/q35moe-round-profile-20261109-results.tar.gz')
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
        raise RuntimeError('dedicated profiling directory already exists')
    OUT.mkdir(parents=True)
    os.environ.update(HF_HOME=str(CACHE / 'hf-home'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'), TRITON_CACHE_DIR=str(CACHE / 'triton'),
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', TORCH_LOGS='perf_hints')
    shutil.copy2(BUNDLE / 'worker.py', OUT / 'worker.py')
    source = ROOT / 'infer-opt'
    source.mkdir()
    with tarfile.open(BUNDLE / 'source.tar.gz') as archive:
        archive.extractall(source, filter='data')
    save('source-manifest.json', json.loads((BUNDLE / 'source-manifest.json').read_text()))
    stage('hardware')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name or gpu.total_memory < 75 * 2**30:
        raise RuntimeError(f'expected A100 80GB, got {gpu}')
    save('initial-runtime.json', {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
         'torch': str(torch.__version__), 'python': sys.version})
    run(['nvidia-smi', '-q'], 'gpu-before')
    stage('dependencies')
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], 'remove-unused-av')
    run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'triton==3.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu130'], 'install-torch', timeout=1200)
    run([sys.executable, '-m', 'pip', 'install', 'transformers==5.22.0', 'accelerate==1.15.0',
         'safetensors==0.8.0', 'huggingface_hub==1.33.0',
         'flash-linear-attention==0.6.0', 'fla-core==0.6.0'], 'install', timeout=900)
    env_code = ('import json,torch;g=torch.cuda.get_device_properties(0);'
                'print(json.dumps({"gpu":g.name,"vram_gib":g.total_memory/2**30,'
                '"torch":str(torch.__version__),"cuda":torch.version.cuda}))')
    environment = json.loads(subprocess.check_output([sys.executable, '-c', env_code], text=True))
    environment['packages'] = {n: importlib.metadata.version(n) for n in
        ('torch', 'triton', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub',
         'flash-linear-attention', 'fla-core')}
    save('environment.json', environment)
    stage('download_int8_checkpoint')
    from huggingface_hub import snapshot_download, HfApi
    repo = 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp'
    weights = snapshot_download(repo, revision=INT8_REV, token=False, cache_dir=str(CACHE / 'int8'),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    stage('checkpoint_hashes')
    info = HfApi().model_info(repo, revision=INT8_REV, files_metadata=True, token=False)
    hashes = []
    for entry in info.siblings:
        path = Path(weights) / entry.rfilename
        if not path.is_file():
            continue
        digest = sha256(path)
        if entry.lfs and digest != entry.lfs.sha256:
            raise RuntimeError('checkpoint hash mismatch: ' + entry.rfilename)
        hashes.append({'file': entry.rfilename, 'bytes': path.stat().st_size,
                       'sha256': digest, 'hub_lfs_verified': entry.lfs is not None})
    save('checkpoint-hashes.json', hashes)
    stage('await_profile_harness')
    deadline = time.time() + 1200
    while not (BUNDLE / 'profile-ready.json').is_file():
        if time.time() > deadline:
            raise RuntimeError('profiling harness upload deadline exceeded')
        time.sleep(2)
    ready = json.loads((BUNDLE / 'profile-ready.json').read_text())
    profile_source = BUNDLE / 'profile_qwen3_5_moe_round.py'
    if sha256(profile_source) != ready['sha256']:
        raise RuntimeError('profiling source hash mismatch')
    shutil.copy2(profile_source, source / 'scripts' / profile_source.name)
    shutil.copy2(profile_source, OUT / profile_source.name)
    save('profile-harness.json', ready)
    common = [sys.executable, str(source / 'scripts' / profile_source.name),
              '--weights', weights, '--ks', '1,3', '--prompts', 'math,korean', '--cache-capacity', '2048']
    stage('production_profiles')
    run(common + ['--stage', 'production', '--out', str(OUT / 'production'),
                  '--new-tokens', '256', '--repeats', '5', '--warmup', '3', '--profile-rounds', '24'],
        'production', timeout=3600)
    stage('diagnostic_profiles')
    run(common + ['--stage', 'diagnostic', '--out', str(OUT / 'diagnostic'),
                  '--event-repeats', '20', '--micro-repeats', '10'], 'diagnostic', timeout=3600)
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


