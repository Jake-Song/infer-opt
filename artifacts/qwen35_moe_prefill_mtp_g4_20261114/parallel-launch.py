"""Second stage in the same session: parallel-verification MTP, after the worker has finished."""
import glob, json, os, subprocess, sys
from pathlib import Path
root = Path('/tmp/moe-prefill-g4-20261114c')
out = root/'results'
assert json.loads((out/'exit.json').read_text())['exit_code'] == 0, 'worker must finish first'
assert not (root/'parallel-launch.json').exists(), 'Do not start duplicate stages'
cache = Path('/tmp/infer-opt-models')/root.name
weights = sorted(glob.glob(str(cache/'weights/models--*/snapshots/*')))
assert len(weights) == 1, weights
env = dict(os.environ, HF_HOME=str(cache/'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
    TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), TRITON_CACHE_DIR=str(cache/'triton'),
    CUDA_CACHE_PATH=str(cache/'cuda'), TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
argv = [sys.executable, '-u', 'scripts/validate_moe_prefill_mtp.py', '--weights', weights[0],
        '--out', str(out/'mtp-parallel'), '--verifications', 'parallel', '--ks', '1,2,3']
shell = ' '.join(argv) + f'; echo "{{\\"returncode\\": $?}}" > {out}/mtp-parallel.exit.json'
(out/'mtp-parallel.command.json').write_text(json.dumps(argv))
with (out/'mtp-parallel.stdout').open('w') as o, (out/'mtp-parallel.stderr').open('w') as e:
    p = subprocess.Popen(['bash', '-c', shell], cwd=root/'source', stdout=o, stderr=e, env=env,
                         start_new_session=True)
(root/'parallel-launch.json').write_text(json.dumps({'pid': p.pid, 'argv': argv}))
print(json.dumps({'pid': p.pid}))
