"""Bounded G4 job: HF BF16 reference, then paired custom INT8 eager/compile."""
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

ROOT = Path('/content/q35moe-compiled-hf-quality')
OUT = ROOT / 'results'
ARCHIVE = Path('/content/q35moe-compiled-hf-quality-results.tar.gz')
CACHE = Path('/tmp/infer-opt-models/q35moe-compiled-hf-quality')
HF_REV = '59d61f3ce65a6d9863b86d2e96597125219dc754'
INT8_REV = '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
CODE = 'a0a798494c295f1e6a850305944373356e33d982'
START = time.time()
BUNDLE = Path(__file__).resolve().parent
SOURCES = ('worker.py', 'validate_compiled_qwen3_5_moe_quality.py', 'validate_llamacpp_qwen3_5_quality.py')


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)


def run(command, name, timeout=1800):
    save(name + '.command.json', command)
    with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}): ' + (OUT / (name + '.stderr')).read_text()[-4000:])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(32 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if ROOT.exists():
        raise RuntimeError('dedicated job root already exists')
    OUT.mkdir(parents=True)
    os.environ['HF_HOME'] = str(CACHE / 'hf-home')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(CACHE / 'inductor')
    os.environ['TRITON_CACHE_DIR'] = str(CACHE / 'triton')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    stage('hardware')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    if 'RTX PRO 6000' not in gpu.name or gpu.total_memory < 90 * 2**30:
        raise RuntimeError(f'expected G4 with >=90 GiB, got {gpu}')
    environment = {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'capability': torch.cuda.get_device_capability(), 'python': sys.version}
    save('environment.json', environment)
    run(['nvidia-smi', '-q'], 'gpu-before')
    stage('dependencies_and_source')
    run([sys.executable, '-m', 'pip', 'install', 'transformers==5.22.0', 'accelerate==1.15.0',
         'safetensors==0.8.0', 'huggingface_hub==1.33.0',
         'flash-linear-attention==0.6.0', 'fla-core==0.6.0'], 'install')
    environment['packages'] = {name: importlib.metadata.version(name) for name in
        ('torch', 'triton', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub',
         'flash-linear-attention', 'fla-core', 'numpy')}
    save('environment.json', environment)
    source = ROOT / 'infer-opt'
    run(['git', 'init', str(source)], 'source-init')
    run(['git', '-C', str(source), 'remote', 'add', 'origin', 'https://github.com/Jake-Song/infer-opt.git'], 'source-remote')
    run(['git', '-C', str(source), 'fetch', '--depth', '1', 'origin', CODE], 'source-fetch')
    run(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'], 'source-checkout')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != CODE:
        raise RuntimeError('source revision mismatch')
    for name in SOURCES[1:]:
        shutil.copy2(BUNDLE / name, source / 'scripts' / name)
    manifest = {'hf_repo': 'Qwen/Qwen3.5-35B-A3B', 'hf_revision': HF_REV,
        'int8_repo': 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp', 'int8_revision': INT8_REV,
        'code_revision': CODE, 'model_cache_root': str(CACHE), 'mtp_called': False,
        'cache_capacity': 8456, 'tokens_per_prompt': 48, 'modes': ['eager', 'compile-reduce-overhead'],
        'compile': {'fullgraph': True, 'dynamic': False, 'mode': 'reduce-overhead'},
        'kernels': 'fla-delta', 'fuse_projections': True, 'paired_hf_reference': True,
        'scope': 'teacher-forced distribution comparison and eager-logit tolerance; not task accuracy or throughput',
        'harness_sha256': {name: sha256(BUNDLE / name) for name in SOURCES}}
    save('manifest.json', manifest)
    from huggingface_hub import snapshot_download, HfApi
    stage('download_hf_reference')
    hf_cache = CACHE / 'reference'
    weights = snapshot_download(manifest['hf_repo'], revision=HF_REV, token=False, cache_dir=str(hf_cache),
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    reference = ROOT / 'reference-buffers'
    stage('hf_reference')
    run([sys.executable, str(source / 'scripts/validate_llamacpp_qwen3_5_quality.py'),
         'reference', '--weights', weights, '--reference-dir', str(reference),
         '--reference-revision', HF_REV, '--tokens', '48'], 'hf-reference', timeout=2400)
    shutil.copy2(reference / 'reference.json', OUT / 'reference.json')
    save('reference-logit-file-hashes.json', {p.name: sha256(p) for p in sorted(reference.glob('*.npy'))})
    previous = json.loads((BUNDLE / 'previous-llama-reference.json').read_text())
    current = json.loads((reference / 'reference.json').read_text())
    manifest['same_input_ids_as_previous_llama'] = previous['cases'] == current['cases']
    save('manifest.json', manifest)
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
    stage('paired_eager_and_compiled')
    run([sys.executable, str(source / 'scripts/validate_compiled_qwen3_5_moe_quality.py'),
         '--weights', weights, '--reference-dir', str(reference), '--out', str(OUT),
         '--scratch', str(ROOT / 'eager-buffers'), '--cache-capacity', '8456'], 'quality', timeout=2400)
    rows = json.loads((OUT / 'quality.json').read_text())['rows']
    if len(rows) != 18 or any(r['teacher_forced_tokens'] != 48 for r in rows):
        raise RuntimeError('incomplete paired comparison')
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
        for name in SOURCES:
            if (BUNDLE / name).is_file():
                shutil.copy2(BUNDLE / name, OUT / name)
        with tarfile.open(ARCHIVE, 'w:gz') as archive:
            for path in sorted(OUT.iterdir()):
                if path.is_file():
                    archive.add(path, arcname=path.name)
            log = Path('/content/q35moe-compiled-hf-quality-worker.log')
            if log.is_file():
                archive.add(log, arcname='worker.log')
