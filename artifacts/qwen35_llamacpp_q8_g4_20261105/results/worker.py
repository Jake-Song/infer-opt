"""Standalone Colab benchmark; uses only public llama.cpp and public GGUF weights."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/q35moe-llama-q8')
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)
START = time.time()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)


def run(args, name, timeout=900, env=None):
    print(json.dumps(args), flush=True)
    with (ROOT / (name + '.stdout')).open('w') as out, (ROOT / (name + '.stderr')).open('w') as err:
        result = subprocess.run(args, stdout=out, stderr=err, timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}): ' + (ROOT / (name + '.stderr')).read_text()[-5000:])
    return (ROOT / (name + '.stdout')).read_text()


try:
    stage('hardware')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    assert 'RTX PRO 6000' in gpu.name and gpu.total_memory > 90 * 2**30, str(gpu)
    save('environment.json', {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                              'torch': torch.__version__, 'python': sys.version,
                              'cpu_count': os.cpu_count(), 'cuda_capability': torch.cuda.get_device_capability()})
    run(['nvidia-smi', '-q'], 'gpu-before')
    run(['nvcc', '--version'], 'nvcc')
    stage('source_and_build')
    run(['git', 'clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp.git', str(ROOT / 'llama.cpp')], 'clone')
    commit = run(['git', '-C', str(ROOT / 'llama.cpp'), 'rev-parse', 'HEAD'], 'llama-revision').strip()
    build = ROOT / 'llama.cpp/build'
    run(['cmake', '-S', str(ROOT / 'llama.cpp'), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
         '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=120', '-DGGML_CUDA_GRAPHS=ON',
         '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_CURL=ON'], 'cmake')
    run(['cmake', '--build', str(build), '-j', str(min(os.cpu_count() or 4, 12)),
         '--target', 'llama-bench', 'llama-cli'], 'build', timeout=1800)
    stage('download_model')
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'ggml-org/Qwen3.5-35B-A3B-GGUF'
    filename = 'Qwen3.5-35B-A3B-Q8_0.gguf'
    info = HfApi().model_info(repo, files_metadata=True)
    fileinfo = next(x for x in info.siblings if x.rfilename == filename)
    model = hf_hub_download(repo, filename, revision=info.sha)
    stage('verify_model')
    digest = hashlib.sha256()
    with open(model, 'rb') as inp:
        for chunk in iter(lambda: inp.read(32 * 2**20), b''):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    expected_sha = fileinfo.lfs.sha256 if fileinfo.lfs else None
    if expected_sha:
        assert actual_sha == expected_sha
    save('manifest.json', {'model_repo': repo, 'model_revision': info.sha, 'file': filename,
                           'bytes': Path(model).stat().st_size, 'sha256': actual_sha,
                           'expected_sha256': expected_sha, 'llama_cpp_commit': commit,
                           'mtp': False, 'speculative_decoding': False,
                           'timing': 'llama-bench synchronized engine time; synthetic random tokens; excludes model loading, tokenization, sampling, HTTP, depth prefill and warmup',
                           'repetitions': 7, 'decode_tokens': 256})
    bench = str(build / 'bin/llama-bench')
    run([bench, '--help'], 'bench-help')
    base = [bench, '-m', model, '-ngl', '99', '-fa', 'on', '-t', '4', '-b', '2048', '-ub', '512',
            '-ctk', 'f16', '-ctv', 'f16', '-r', '7', '-o', 'jsonl', '-oe', 'md']
    commands = []
    rows = []
    jobs = [('prefill', ['-p', '512,2048,8192', '-n', '0'])]
    jobs += [(f'decode-d{d}', ['-p', '0', '-n', '256', '-d', str(d)]) for d in [0, 512, 2048, 8192]]
    for name, args in jobs:
        stage(name)
        command = base + args
        commands.append(command)
        save('commands.json', commands)
        output = run(command, name, timeout=600)
        for line in output.splitlines():
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            assert row['n_gpu_layers'] == 99
            tokens = row['n_prompt'] + row['n_gen']
            samples_ms = [x / 1e6 for x in row['samples_ns']]
            rows.append({'test': name, 'prompt_tokens': row['n_prompt'], 'generated_tokens': row['n_gen'],
                         'context_depth': row.get('n_depth', 0), 'median_ms': statistics.median(samples_ms),
                         'median_tokens_s': tokens * 1000 / statistics.median(samples_ms),
                         'median_ms_token': statistics.median(samples_ms) / tokens,
                         'mean_tokens_s': row['avg_ts'], 'std_tokens_s': row['stddev_ts'],
                         'samples_ms': samples_ms})
        save('summary.json', {'manifest': json.loads((ROOT / 'manifest.json').read_text()), 'rows': rows})
    stage('generation_smoke')
    run([str(build / 'bin/llama-cli'), '-m', model, '-ngl', '99', '-fa', 'on', '-c', '2048',
         '-n', '48', '-p', 'Explain briefly why the sky is blue.', '--single-turn', '--temp', '0',
         '--spec-type', 'none', '--chat-template-kwargs', '{"enable_thinking":false}'], 'generation-smoke', timeout=180)
    run(['nvidia-smi', '-q'], 'gpu-after')
    stage('complete')
    save('exit.json', {'exit_code': 0, 'elapsed_s': time.time() - START})
except BaseException:
    save('exit.json', {'exit_code': 1, 'elapsed_s': time.time() - START, 'traceback': traceback.format_exc()})
    traceback.print_exc()
finally:
    import tarfile
    with tarfile.open('/content/q35moe-llama-q8-results.tar.gz', 'w:gz') as tar:
        for path in ROOT.iterdir():
            if path.is_file():
                tar.add(path, arcname=path.name)
