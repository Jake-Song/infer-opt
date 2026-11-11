"""Benchmark pinned native llama.cpp MTP, without modifying CUDA kernels."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path('/tmp/llama-mtp-a100-20261111')
OUT = ROOT / 'results'
CACHE = Path('/tmp/infer-opt-models/llama-mtp-20261111')
REV = 'a0ad695cb212db3850f163121ddc687513448d42'
REPO = 'unsloth/Qwen3.5-35B-A3B-MTP-GGUF'
MODEL_REV = '63af8373893a7a73c6dfcb84cb63d815981da5e0'
MODEL_FILE = 'Qwen3.5-35B-A3B-Q8_0.gguf'
MODEL_SHA = 'b08e017b6ffcd1f57343a603a9e6d9e498373ef09da55537a2788af70ceb1d0d'
PORT = 18891
BASE_URL = f'http://127.0.0.1:{PORT}'
START = time.time()
SERVER = None
os.environ.update(HF_HOME=str(CACHE / 'hf'), HF_HUB_DISABLE_TELEMETRY='1',
                  OMP_NUM_THREADS='4', LLAMA_CACHE=str(CACHE / 'llama'))
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def snapshot():
    tmp = ROOT / 'results.tar.gz.tmp'
    with tarfile.open(tmp, 'w:gz') as tar:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                tar.add(path, arcname=str(path.relative_to(OUT)))
    tmp.replace(ROOT / 'results.tar.gz')


def stage(name):
    info = {'stage': name, 'elapsed_s': time.time() - START}
    save('progress.json', info)
    print(json.dumps(info), flush=True)
    snapshot()


def run(argv, name, timeout=1800, check=True):
    save(name + '.command.json', argv)
    with (OUT / (name + '.stdout')).open('w') as stdout, (OUT / (name + '.stderr')).open('w') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout)
    save(name + '.exit.json', {'returncode': result.returncode})
    if check and result.returncode:
        raise RuntimeError(name + ': ' + (OUT / (name + '.stderr')).read_text()[-6000:])
    return (OUT / (name + '.stdout')).read_text()


def api(path, data=None, timeout=300):
    request = urllib.request.Request(BASE_URL + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
    return raw if path == '/metrics' else json.loads(raw)


def metrics():
    raw = api('/metrics')
    parsed = {}
    for line in raw.splitlines():
        if line and not line.startswith('#'):
            key, value = line.rsplit(None, 1)
            parsed[key] = float(value)
    return raw, parsed


def stop_server():
    global SERVER
    if SERVER is not None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=30)
        except subprocess.TimeoutExpired:
            SERVER.kill()
            SERVER.wait(timeout=10)
        SERVER = None


def download():
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(token=False).model_info(REPO, revision=MODEL_REV, files_metadata=True)
    metadata = next(x for x in info.siblings if x.rfilename == MODEL_FILE)
    assert info.sha == MODEL_REV and metadata.lfs.sha256 == MODEL_SHA
    model = Path(hf_hub_download(REPO, MODEL_FILE, revision=MODEL_REV,
                                cache_dir=str(CACHE / 'hf/hub'), token=False))
    digest = hashlib.sha256()
    with model.open('rb') as source:
        for chunk in iter(lambda: source.read(32 * 2**20), b''):
            digest.update(chunk)
    assert digest.hexdigest() == MODEL_SHA and model.stat().st_size == metadata.size
    save('model-manifest.json', {'repo': REPO, 'revision': MODEL_REV, 'filename': MODEL_FILE,
         'bytes': model.stat().st_size, 'sha256': digest.hexdigest(), 'expected_sha256': MODEL_SHA,
         'cache_path': str(model)})
    return model


def build():
    src = ROOT / 'llama.cpp'
    run(['git', 'init', str(src)], 'git-init')
    run(['git', '-C', str(src), 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git'], 'git-origin')
    run(['git', '-C', str(src), 'fetch', '--depth', '1', 'origin', REV], 'git-fetch')
    run(['git', '-C', str(src), 'checkout', '--detach', 'FETCH_HEAD'], 'git-checkout')
    actual = run(['git', '-C', str(src), 'rev-parse', 'HEAD'], 'source-revision').strip()
    assert actual == REV
    build_dir = src / 'build'
    run(['cmake', '-S', str(src), '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release',
         '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=80', '-DGGML_CUDA_GRAPHS=ON',
         '-DGGML_CUDA_NCCL=OFF', '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_CURL=OFF'], 'cmake')
    run(['cmake', '--build', str(build_dir), '-j', str(min(os.cpu_count() or 4, 12)),
         '--target', 'llama-server'], 'build', timeout=3600)
    shutil.copy2(build_dir / 'CMakeCache.txt', OUT / 'CMakeCache.txt')
    diff = run(['git', '-C', str(src), 'diff', '--exit-code'], 'source-diff')
    assert not diff.strip()
    return build_dir / 'bin/llama-server'


def start_server(binary, model, k, label):
    global SERVER
    stop_server()
    args = [str(binary), '-m', str(model), '-ngl', '99', '-fa', 'on', '-c', '2048',
            '-np', '1', '-b', '2048', '-ub', '512', '-t', '4', '-tb', '4',
            '-ctk', 'f16', '-ctv', 'f16', '--cache-ram', '0', '--no-context-shift',
            '--host', '127.0.0.1', '--port', str(PORT), '--metrics', '--slots',
            '--spec-type', 'none' if k == 0 else 'draft-mtp']
    if k:
        args += ['--spec-draft-n-max', str(k), '--spec-draft-n-min', '0',
                 '--spec-draft-p-min', '0', '-ngld', '99']
    save(label + '-server.command.json', args)
    stdout = (OUT / (label + '-server.stdout')).open('w')
    stderr = (OUT / (label + '-server.stderr')).open('w')
    SERVER = subprocess.Popen(args, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    for _ in range(360):
        if SERVER.poll() is not None:
            raise RuntimeError('Server exited: ' + (OUT / (label + '-server.stderr')).read_text()[-7000:])
        try:
            if api('/health', timeout=2).get('status') == 'ok':
                save(label + '-props.json', api('/props'))
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError('server readiness')


def request(prompt, k, label):
    raw_before, before = metrics()
    body = {'prompt': prompt['input_ids'], 'n_predict': 256, 'temperature': 0,
            'top_k': 1, 'seed': 42, 'ignore_eos': True, 'cache_prompt': False,
            'return_tokens': True, 'stream': False}
    t0 = time.perf_counter()
    response = api('/completion', body)
    elapsed = time.perf_counter() - t0
    raw_after, after = metrics()
    (OUT / (label + '-metrics-before.txt')).write_text(raw_before)
    (OUT / (label + '-metrics-after.txt')).write_text(raw_after)
    timings = response['timings']
    tokens = response['tokens']
    assert len(tokens) == response['tokens_predicted'] == timings['predicted_n'] == 256
    assert timings['cache_n'] == 0, timings
    steps = len(tokens) - 1
    ms_token = timings['predicted_ms'] / steps
    assert math.isclose(ms_token, timings['predicted_per_token_ms'], rel_tol=1e-8)
    assert math.isclose(1000 / ms_token, timings['predicted_per_second'], rel_tol=1e-8)
    delta = {key: after[key] - value for key, value in before.items()
             if key in after and ('spec_decode_' in key or key.endswith('tokens_predicted_total'))}
    rounds = delta.get('llamacpp:spec_decode_num_drafts_total', 0)
    draft_n = timings.get('draft_n', 0)
    accepted = timings.get('draft_n_accepted', 0)
    if k:
        assert draft_n > 0 and rounds > 0, (timings, delta)
        assert accepted <= draft_n
    result = {'k': k, 'response': response, 'request': body,
              'http_wall_ms': elapsed * 1000, 'decode_ms': timings['predicted_ms'],
              'decode_steps': steps, 'ms_token': ms_token, 'tokens_s': 1000 / ms_token,
              'rounds': rounds, 'ms_round': timings['predicted_ms'] / rounds if rounds else None,
              'tokens_round': steps / rounds if rounds else None,
              'draft_n': draft_n, 'draft_accepted': accepted,
              'acceptance': accepted / draft_n if draft_n else None, 'metrics_delta': delta}
    save(label + '.json', result)
    return result


def summarize(rows):
    summary = []
    for prompt in ['math', 'korean']:
        baseline = [r for r in rows if r['prompt'] == prompt and r['k'] == 0]
        if not baseline:
            continue
        baseline_tokens = baseline[0]['response']['tokens']
        base_ms = statistics.median(r['ms_token'] for r in baseline)
        for k in [0, 1, 2, 3]:
            group = [r for r in rows if r['prompt'] == prompt and r['k'] == k]
            if not group:
                continue
            def median(key):
                values = [r[key] for r in group if r[key] is not None]
                return statistics.median(values) if values else None
            diffs = []
            for row in group:
                ids = row['response']['tokens']
                diffs.append(next((i for i, (a, b) in enumerate(zip(ids, baseline_tokens)) if a != b), None))
            ms = median('ms_token')
            summary.append({'prompt': prompt, 'k': k, 'samples': len(group), 'ms_token': ms,
                'tokens_s': 1000 / ms, 'timing_ratio_vs_off': base_ms / ms,
                'acceptance': median('acceptance'), 'rounds': median('rounds'),
                'ms_round': median('ms_round'), 'tokens_round': median('tokens_round'),
                'exact_runs': sum(x is None for x in diffs), 'first_differences': diffs,
                'min_ms_token': min(r['ms_token'] for r in group),
                'max_ms_token': max(r['ms_token'] for r in group)})
    save('summary.json', {'rows': summary, 'timed_requests': len(rows),
         'source_revision': REV, 'model_revision': MODEL_REV,
         'boundary': 'server generation wall time after first token / 255 emitted tokens; HTTP, prefill, first token, loading and warmup excluded',
         'kernel_modifications': False, 'warp_override': False})


def main():
    stage('environment')
    import torch
    gpu = torch.cuda.get_device_properties(0)
    assert 'A100' in gpu.name and gpu.total_memory >= 75 * 2**30
    save('environment.json', {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
         'torch': str(torch.__version__), 'python': sys.version, 'cpu_count': os.cpu_count(),
         'kernel_modifications': False, 'warp_override': False})
    run(['nvidia-smi', '-q'], 'gpu-before')
    run(['nvcc', '--version'], 'nvcc')
    run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', 'gguf'], 'install-deps')
    for filename in ['worker.py', 'inputs.json']:
        shutil.copy2(ROOT / filename, OUT / filename)
    save('harness-manifest.json', {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
         for name in ['worker.py', 'inputs.json']})
    stage('build-and-download')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        build_future = pool.submit(build)
        download_future = pool.submit(download)
        binary = build_future.result()
        model = download_future.result()
    run([str(binary), '--help'], 'server-help')
    from gguf import GGUFReader
    reader = GGUFReader(str(model), mode='r')
    fields = {name: field.contents() for name, field in reader.fields.items()
              if any(s in name for s in ['architecture', 'block_count', 'nextn', 'expert_count', 'expert_used_count'])}
    mtp_tensors = [{'name': t.name, 'shape': t.shape.tolist(), 'type': str(t.tensor_type)}
                   for t in reader.tensors if t.name.startswith('blk.40.') or 'nextn' in t.name]
    save('gguf-audit.json', {'fields': fields, 'mtp_tensors': mtp_tensors,
         'tensor_count': len(reader.tensors)})
    assert any('nextn' in key and int(value) > 0 for key, value in fields.items()), fields
    assert len(mtp_tensors) > 10, mtp_tensors
    del reader
    prompts = json.loads((ROOT / 'inputs.json').read_text())
    rows = []
    # Alternate configurations across two sessions each to expose time-order drift.
    orders = [[0, 1, 2, 3], [3, 2, 1, 0]]
    for batch, order in enumerate(orders):
        for k in order:
            label = f'batch{batch}-k{k}'
            stage('start-' + label)
            start_server(binary, model, k, label)
            for prompt_name, prompt in prompts.items():
                tokenized = api('/tokenize', {'content': prompt['text'], 'add_special': False, 'parse_special': True})
                save(label + '-' + prompt_name + '-tokenize.json', tokenized)
                assert tokenized['tokens'] == prompt['input_ids'], (prompt_name, tokenized)
            for warmup in range(3):
                for prompt_name, prompt in prompts.items():
                    stage(f'warmup-{label}-{prompt_name}-{warmup}')
                    request(prompt, k, f'{label}-{prompt_name}-warmup{warmup}')
            count = 3 if batch == 0 else 2
            jobs = [(name, rep) for rep in range(count) for name in prompts]
            random.Random(901 + batch * 10 + k).shuffle(jobs)
            for prompt_name, rep in jobs:
                stage(f'timing-{label}-{prompt_name}-{rep}')
                record = request(prompts[prompt_name], k, f'{label}-{prompt_name}-sample{rep}')
                record.update(prompt=prompt_name, batch=batch, repeat=rep)
                rows.append(record)
                save('timed-requests.json', rows)
                summarize(rows)
            run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw',
                 '--format=csv'], label + '-telemetry')
            stop_server()
            snapshot()
    run(['nvidia-smi', '-q'], 'gpu-after')
    stage('complete')


if __name__ == '__main__':
    try:
        main()
        save('exit.json', {'returncode': 0, 'elapsed_s': time.time() - START})
    except BaseException:
        save('exit.json', {'returncode': 1, 'elapsed_s': time.time() - START,
                          'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        stop_server()
        snapshot()
