"""Public HF infer-opt checkpoint, no MTP; all downloaded weights live in /tmp."""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/q35moe-hf-int8')
ROOT.mkdir(exist_ok=True)
START = time.time()
REPO = 'Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp'
REVISION = '331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
CODE = 'a0a798494c295f1e6a850305944373356e33d982'
os.environ['HF_HOME'] = '/tmp/infer-opt-models/huggingface'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/infer-opt-models/inductor'
os.environ['OMP_NUM_THREADS'] = '4'


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)


def run(args, name, timeout=600):
    with (ROOT / (name + '.log')).open('w') as log:
        p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode:
        raise RuntimeError(name + ': ' + (ROOT / (name + '.log')).read_text()[-4000:])


try:
    stage('setup')
    run(['git', 'clone', 'https://github.com/Jake-Song/infer-opt.git', str(ROOT / 'code')], 'clone')
    run(['git', '-C', str(ROOT / 'code'), 'checkout', CODE], 'checkout')
    run([sys.executable, '-m', 'pip', 'install', 'safetensors', 'huggingface_hub', 'transformers', 'flash-linear-attention'], 'install')
    sys.path.insert(0, str(ROOT / 'code'))
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_properties(0)
    assert 'RTX PRO 6000' in gpu.name and gpu.total_memory > 90 * 2**30
    save('environment.json', {'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                              'torch': torch.__version__, 'triton': __import__('triton').__version__,
                              'python': sys.version, 'code_revision': CODE})
    stage('download_checkpoint_to_tmp')
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    model_dir = snapshot_download(REPO, revision=REVISION, cache_dir='/tmp/infer-opt-models/huggingface/hub',
                                  allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', 'README.md'])
    stage('verify_checkpoint_hashes')
    hashes = []
    for fileinfo in info.siblings:
        path = Path(model_dir) / fileinfo.rfilename
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as inp:
            for chunk in iter(lambda: inp.read(32 * 2**20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if fileinfo.lfs:
            assert actual == fileinfo.lfs.sha256, path.name
        hashes.append({'file': path.name, 'sha256': actual, 'bytes': path.stat().st_size})
    save('manifest.json', {'repo': REPO, 'revision': info.sha, 'code_revision': CODE, 'path': model_dir,
                           'files': hashes, 'mtp': False, 'runtime': 'infer-opt',
                           'weight_format': 'INT8 group-128; retained BF16 router/gates/lm_head',
                           'input': 'synthetic torch.randint seed 20261105; differs from llama-bench random inputs',
                           'cache_capacity': 8456, 'batch': 1, 'decode_tokens': 256, 'repeats': 7,
                           'timing': 'synchronized host wall time, includes forward dispatch, excludes loading, compilation, warmup, cache reset/restore, sampling; decode synchronizes every step'})
    stage('strict_load')
    from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, FixedShapeDecoder
    model = Qwen3_5ForCausalLM.from_quantized_dir(model_dir, device='cuda', kernels='fla-delta')
    assert model.config.num_hidden_layers == 40 and model.config.num_experts == 256
    fused = model.fuse_projections()
    save('load.json', {'strict_load': True, 'fused_projection_groups': fused, 'mtp_head_present': model.mtp is not None,
                       'mtp_invoked': False, 'allocated_gib': torch.cuda.memory_allocated() / 2**30})
    torch.manual_seed(20261105)
    prompt = torch.randint(1000, 100000, (1, 8192), device='cuda')
    tokens = torch.randint(1000, 100000, (1, 256), device='cuda')
    save('input_ids.json', {'prompt_ids': prompt.cpu().tolist(), 'decode_ids': tokens.cpu().tolist()})
    cache = model.new_cache(1, 8456)

    def sync():
        torch.cuda.synchronize()

    def prefill(ids):
        hidden = model.model.language_model(ids, cache)
        return model.lm_head(hidden[:, -1:])

    def states():
        return [t for group in (cache.keys, cache.values, cache.conv_states, cache.recurrent_states)
                for t in group if t is not None] + [cache.position]

    def snapshot():
        return [t.clone() for t in states()], cache.seen_tokens

    def restore(saved):
        for dst, src in zip(states(), saved[0]):
            dst.copy_(src)
        cache.seen_tokens = saved[1]

    rows = []
    def save_rows():
        save('summary.json', {'rows': rows, 'environment': json.loads((ROOT / 'environment.json').read_text())})

    # Keep prefill unmodified: expert dispatch loops through routed banks.
    # Compute only the final-position vocabulary logits, like llama-bench.
    for depth in [512, 2048, 8192]:
        stage(f'prefill-{depth}')
        ids = prompt[:, :depth]
        cache.reset()
        t0 = time.perf_counter()
        prefill(ids)
        sync()
        warmup_s = time.perf_counter() - t0
        samples = []
        for _ in range(7):
            cache.reset()
            sync()
            t0 = time.perf_counter()
            prefill(ids)
            sync()
            samples.append((time.perf_counter() - t0) * 1000)
        rows.append({'test': 'prefill', 'prompt_tokens': depth, 'samples_ms': samples,
                     'median_ms': statistics.median(samples),
                     'median_tokens_s': depth * 1000 / statistics.median(samples), 'warmup_s': warmup_s})
        save_rows()

    cache.reset()
    # Preparation and compilation happen once, outside all timed regions.
    for mode in ['cudagraph', 'compile-reduce-overhead']:
        stage('prepare-' + mode)
        decoder = FixedShapeDecoder(model, cache, mode)
        t0 = time.perf_counter()
        decoder.prepare()
        for i in range(8):
            decoder.step(tokens[:, i:i + 1])
        sync()
        setup_s = time.perf_counter() - t0
        stage('validate-' + mode)
        cache.reset()
        prefill(prompt[:, :512])
        saved = snapshot()
        expected = []
        eager = FixedShapeDecoder(model, cache, 'eager')
        for i in range(16):
            expected.append(eager.step(tokens[:, i:i + 1]).clone())
        restore(saved)
        diffs, matches, allclose = [], [], []
        for i, want in enumerate(expected):
            got = decoder.step(tokens[:, i:i + 1]).clone()
            diffs.append((got.float() - want.float()).abs().max().item())
            matches.append(bool(torch.equal(got.argmax(-1), want.argmax(-1))))
            allclose.append(bool(torch.allclose(got, want, atol=.02, rtol=.02)))
        save('validation-' + mode + '.json', {'reference': 'same-checkpoint fixed-shape eager',
             'steps': 16, 'max_abs_diff': max(diffs), 'argmax_matches': sum(matches),
             'allclose_atol_002_rtol_002': allclose, 'setup_s': setup_s})
        del expected, saved, eager
        for depth in [0, 512, 2048, 8192]:
            stage(f'{mode}-decode-{depth}')
            cache.reset()
            if depth:
                prefill(prompt[:, :depth])
            saved = snapshot()
            for i in range(256):
                decoder.step(tokens[:, i:i + 1])
                sync()
            samples = []
            for _ in range(7):
                restore(saved)
                sync()
                t0 = time.perf_counter()
                for i in range(256):
                    decoder.step(tokens[:, i:i + 1])
                    sync()
                samples.append((time.perf_counter() - t0) * 1000)
            rows.append({'test': 'decode', 'mode': mode, 'context_depth': depth, 'generated_tokens': 256,
                         'samples_ms': samples, 'median_ms_token': statistics.median(samples) / 256,
                         'median_tokens_s': 256000 / statistics.median(samples), 'setup_s': setup_s,
                         'allclose_passed': all(allclose), 'argmax_matches_16': sum(matches),
                         'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
            save_rows()
            del saved
        del decoder
        gc.collect()
        cache.reset()
    stage('generation_smoke')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = tokenizer('Explain briefly why the sky is blue.', return_tensors='pt').input_ids.cuda()
    output = model.generate_greedy(ids, 48, static_cache=True, decode_mode='cudagraph')
    save('generation.json', {'input': 'Explain briefly why the sky is blue.',
                             'text': tokenizer.decode(output[0].tolist()), 'mtp': False})
    stage('complete')
    save('exit.json', {'exit_code': 0, 'elapsed_s': time.time() - START})
except BaseException:
    save('exit.json', {'exit_code': 1, 'traceback': traceback.format_exc(), 'elapsed_s': time.time() - START})
    traceback.print_exc()
finally:
    import tarfile
    with tarfile.open('/content/q35moe-hf-int8-results.tar.gz', 'w:gz') as tar:
        for path in ROOT.iterdir():
            if path.is_file():
                tar.add(path, arcname=path.name)
