"""One bounded G4 accuracy ablation: BF16, full NVFP4 and MLP-only NVFP4."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_REVISION = 'fc05daec18b0a78c049392ed2e771dde82bdf654'
FULL_REPO = 'Jakemu/Qwen3.5-27B-nvfp4-mtp'
FULL_REVISION = '08a9f4929e2033ddd5e81f0ab8b31072e7437e1a'
CALIBRATION_REVISION = '8049631c405ae6576f93f445c6b8166f76f5505a'
CALIBRATION_TOKEN_HASH = 'a6c4f956c72671382a26c52f17f485033b51437803566dcb44211767dd6338b1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.budget_seconds
    state = {'status': 'running', 'stage': 'install', 'scope': 'mlp',
             'model_revision': MODEL_REVISION, 'full_checkpoint_revision': FULL_REVISION}

    def save():
        temporary = root / 'progress.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / 'progress.json')

    def expired(signum, frame):
        raise TimeoutError('MLP-only G4 budget exhausted')

    def run(command, stage):
        state['stage'] = stage
        save()
        with (root / (stage + '.log')).open('w') as log:
            subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1, deadline-time.monotonic()))

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    try:
        run(['-m', 'pip', 'install', '-r', 'scripts/requirements_nvfp4_g4.txt'], 'install')
        import torch
        from huggingface_hub import snapshot_download
        from infer_opt.nvfp4 import require_sm120
        require_sm120()
        if 'RTX PRO 6000' not in torch.cuda.get_device_name():
            raise RuntimeError('expected RTX PRO 6000 G4')
        (root / 'environment.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}}, indent=2))
        state['stage'] = 'download_original'
        save()
        original = snapshot_download('Qwen/Qwen3.5-27B', revision=MODEL_REVISION,
                                     allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', 'LICENSE'])
        checkpoint = Path(str(root) + '-mlp-checkpoint')
        run(['scripts/bench_qwen3_5_nvfp4.py', '--weights', original,
             '--precision', 'nvfp4', '--nvfp4-scope', 'mlp',
             '--calibration-samples', '128', '--calibration-revision', CALIBRATION_REVISION,
             '--export', str(checkpoint), '--export-only', '--out', str(root / 'export')], 'quantize_mlp')
        calibration = json.loads((root / 'export/calibration.json').read_text())
        token_hash = hashlib.sha256('\n'.join(calibration['token_hashes']).encode()).hexdigest()
        if token_hash != CALIBRATION_TOKEN_HASH or len(calibration['token_hashes']) != 128:
            raise RuntimeError('calibration token hashes differ from original full-NVFP4 run')
        report = json.loads((root / 'export/quantization.json').read_text())
        spec = json.loads((checkpoint / 'quantization.json').read_text())
        packed = [n for n, v in spec['modules'].items() if v['kind'] == 'nvfp4']
        if len(packed) != 128 or not all('.mlp.' in n for n in packed):
            raise RuntimeError('expected exactly 128 native MLP Linears and no native attention Linears')
        if report['scope'] != 'mlp' or set(packed) != set(report['modules']):
            raise RuntimeError('packing and calibration scopes differ')
        from safetensors import safe_open
        attention = []
        for shard in checkpoint.glob('model-nvfp4-*.safetensors'):
            with safe_open(shard, framework='pt', device='cpu') as file:
                for name in file.keys():
                    if name.startswith('mtp.') or not name.endswith('.weight'):
                        continue
                    if '.linear_attn.' not in name and '.self_attn.' not in name:
                        continue
                    tensor = file.get_slice(name)
                    shape = tensor.get_shape()
                    if len(shape) == 2 and shape[0] >= 256 and shape[1] % 16 == 0:
                        if tensor.get_dtype() != 'BF16':
                            raise RuntimeError(f'attention weight is not BF16: {name}')
                        attention.append(name)
        if len(attention) != 128:
            raise RuntimeError(f'expected 128 BF16 attention projections, got {len(attention)}')
        (root / 'scope_validation.json').write_text(json.dumps({
            'scope': 'mlp', 'native_mlp_linears': len(packed), 'native_attention_linears': 0,
            'bf16_attention_linears': len(attention),
            'calibration_token_hashes_identical': True, 'calibration_tokens_sha256': token_hash,
            'mtp_precision': 'int8-g128', 'attention_precision': 'bf16',
            'quantized_modules': packed}, indent=2))
        state['stage'] = 'download_full_nvfp4'
        save()
        full = snapshot_download(FULL_REPO, revision=FULL_REVISION)
        run(['scripts/validate_qwen3_5_decode_modes.py', '--weights', original,
             '--repo', 'Qwen/Qwen3.5-27B', '--fuse', '--autotune',
             '--configs', 'fla-delta:compile-reduce-overhead:bf16,'
             'fla-delta:compile-reduce-overhead:nvfp4,fla-delta:compile-reduce-overhead:nvfp4-mlp',
             '--nvfp4-from', full, '--nvfp4-mlp-from', str(checkpoint),
             '--new-tokens', '192', '--decode-positions', '32',
             '--out', str(root / 'accuracy.json')], 'accuracy')
        accuracy = json.loads((root / 'accuracy.json').read_text())
        state['accuracy_passed'] = next(g['pass'] for g in accuracy['gate']
                                         if g['config'].endswith(':nvfp4-mlp'))
        state['status'] = 'finished'
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (root / 'error.txt').write_text(traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        state['elapsed_seconds'] = time.monotonic() - started
        save()
        with zipfile.ZipFile(str(root) + '.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in root.rglob('*'):
                if file.is_file():
                    archive.write(file, str(file.relative_to(root)))
        (root / 'done.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
