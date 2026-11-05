"""One G4 job: pinned HF reference, then pinned llama.cpp Q8_0 replay.

Only public sources/models and the accompanying comparison harness are used.
No HF credential, Drive mount, model publication, or performance benchmark.
"""
from __future__ import annotations

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

ROOT = Path('/content/q35moe-llama-hf-quality')
OUT = ROOT / 'results'
ARCHIVE = Path('/content/q35moe-llama-hf-quality-results.tar.gz')
HF_REV = '59d61f3ce65a6d9863b86d2e96597125219dc754'
GGUF_REV = '3127ef0b7fd4f12626506419ceefde54fa8db53c'
LLAMA_REV = 'c6a0ff11a0fe16ac1a2eb569190934593a2b3670'
INFER_REV = 'a0a798494c295f1e6a850305944373356e33d982'
MODEL_CACHE = Path('/tmp/infer-opt-models/q35moe-llama-hf-quality')
START = time.time()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(target)


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)


def run(command, name, timeout=1800):
    save(name + '.command.json', command)
    with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}): ' + (OUT / (name + '.stderr')).read_text()[-4000:])


def clone(repo, revision, destination, label):
    run(['git', 'init', str(destination)], label + '-init')
    run(['git', '-C', str(destination), 'remote', 'add', 'origin', repo], label + '-remote')
    run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', revision], label + '-fetch')
    run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], label + '-checkout')
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError('source revision mismatch')


def main():
    if ROOT.exists():
        raise RuntimeError('dedicated job root already exists; refusing to overwrite')
    OUT.mkdir(parents=True)
    # Set cache paths before importing HF libraries. Credentials are never passed.
    os.environ['HF_HOME'] = str(MODEL_CACHE / 'hf-home')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '4'
    stage('hardware')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    if 'RTX PRO 6000' not in gpu.name or gpu.total_memory < 90 * 2**30:
        raise RuntimeError(f'expected G4 with >=90 GiB; got {gpu}')
    environment = {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'capability': torch.cuda.get_device_capability(), 'python': sys.version}
    save('environment.json', environment)
    run(['nvidia-smi', '-q'], 'gpu-before')
    run(['nvcc', '--version'], 'nvcc')
    stage('dependencies')
    run([sys.executable, '-m', 'pip', 'install', 'transformers>=5,<6', 'accelerate',
         'safetensors', 'huggingface_hub'], 'install')
    environment['packages'] = {name: importlib.metadata.version(name) for name in
                               ('torch', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub', 'numpy')}
    save('environment.json', environment)
    stage('sources_and_build')
    source = ROOT / 'infer-opt'
    llama = ROOT / 'llama.cpp'
    clone('https://github.com/Jake-Song/infer-opt.git', INFER_REV, source, 'infer-opt')
    clone('https://github.com/ggml-org/llama.cpp.git', LLAMA_REV, llama, 'llama')
    # Harness files are uploaded alongside this worker, outside the public checkout.
    bundle = Path(__file__).resolve().parent
    shutil.copy2(bundle / 'validate_llamacpp_qwen3_5_quality.py', source / 'scripts')
    harness = source / 'scripts/llamacpp_quality'
    harness.mkdir(exist_ok=True)
    for name in ('CMakeLists.txt', 'replay.cpp'):
        shutil.copy2(bundle / name, harness / name)
    build = ROOT / 'build'
    run(['cmake', '-S', str(harness), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DLLAMA_CPP_SOURCE=' + str(llama), '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=120',
         '-DGGML_CUDA_GRAPHS=ON', '-DBUILD_SHARED_LIBS=OFF'], 'cmake')
    run(['cmake', '--build', str(build), '-j', str(min(os.cpu_count() or 4, 12)),
         '--target', 'infer-opt-llama-replay'], 'build')
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    manifest = {'hf_repo': 'Qwen/Qwen3.5-35B-A3B', 'hf_revision': HF_REV,
                'gguf_repo': 'ggml-org/Qwen3.5-35B-A3B-GGUF', 'gguf_revision': GGUF_REV,
                'llama_cpp_revision': LLAMA_REV, 'infer_opt_revision': INFER_REV,
                'model_cache_root': str(MODEL_CACHE), 'mtp': False, 'tokens_per_prompt': 48,
                'reference_contract': 'HF cached greedy; same 9 prompts as validate_qwen3_5_moe_quality.py',
                'scope': 'teacher-forced distribution preservation, not task accuracy or throughput',
                'harness_sha256': {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                                   for name in ('worker.py', 'replay.cpp', 'CMakeLists.txt',
                                                'validate_llamacpp_qwen3_5_quality.py')}}
    save('manifest.json', manifest)
    stage('download_hf_reference')
    hf_cache = MODEL_CACHE / 'reference'
    weights = snapshot_download(manifest['hf_repo'], revision=HF_REV, token=False,
        cache_dir=str(hf_cache), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    stage('hf_reference')
    reference = ROOT / 'reference-buffers'
    driver = str(source / 'scripts/validate_llamacpp_qwen3_5_quality.py')
    run([sys.executable, driver, 'reference', '--reference-dir', str(reference), '--weights', weights,
         '--reference-revision', HF_REV, '--tokens', '48'], 'hf-reference', timeout=2400)
    # Release the dedicated HF download cache only after its reference process exits.
    # This keeps disk usage bounded while GGUF is downloaded; no existing cache is touched.
    shutil.rmtree(hf_cache)
    stage('download_gguf')
    filename = 'Qwen3.5-35B-A3B-Q8_0.gguf'
    gguf = hf_hub_download(manifest['gguf_repo'], filename, revision=GGUF_REV,
                           cache_dir=str(MODEL_CACHE / 'gguf'), token=False)
    digest = hashlib.sha256()
    with open(gguf, 'rb') as stream:
        for chunk in iter(lambda: stream.read(32 * 2**20), b''):
            digest.update(chunk)
    info = HfApi().model_info(manifest['gguf_repo'], revision=GGUF_REV, files_metadata=True, token=False)
    entry = next(s for s in info.siblings if s.rfilename == filename)
    expected = entry.lfs.sha256 if entry.lfs else None
    if not expected or digest.hexdigest() != expected:
        raise RuntimeError('GGUF hash differs from pinned Hub LFS metadata')
    manifest.update(gguf_sha256=digest.hexdigest(), gguf_bytes=Path(gguf).stat().st_size)
    save('manifest.json', manifest)
    stage('llama_replay_and_comparison')
    run([sys.executable, driver, 'compare', '--reference-dir', str(reference), '--out', str(OUT),
         '--llama-bin', str(build / 'infer-opt-llama-replay'), '--gguf', gguf], 'comparison', timeout=1800)
    rows = json.loads((OUT / 'quality.json').read_text())['rows']
    if len(rows) != 9 or any(row['teacher_forced_tokens'] != 48 for row in rows):
        raise RuntimeError('incomplete comparison')
    save('aggregate.json', {
        'prompts': len(rows), 'teacher_forced_positions': sum(r['teacher_forced_tokens'] for r in rows),
        'mean_kl': sum(r['mean_kl'] for r in rows) / len(rows),
        'top1_agreement': sum(r['top1_agreement'] for r in rows) / len(rows),
        'strong_mismatches': sum(r['strong_mismatches'] for r in rows),
        'mismatches': sum(len(r['mismatch_steps']) for r in rows),
    })
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
        # Reports/logs/inputs and harness source only; no weights or full-logit buffers.
        for name in ('worker.py', 'replay.cpp', 'CMakeLists.txt', 'validate_llamacpp_qwen3_5_quality.py'):
            origin = Path(__file__).resolve().parent / name
            if origin.is_file():
                shutil.copy2(origin, OUT / name)
        with tarfile.open(ARCHIVE, 'w:gz') as archive:
            for path in sorted(OUT.iterdir()):
                if path.is_file():
                    archive.add(path, arcname=path.name)
            log = Path('/content/q35moe-llama-hf-quality-worker.log')
            if log.is_file():
                archive.add(log, arcname='worker.log')
