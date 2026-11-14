"""Build a source-only upload bundle and guarded launcher for this G4 run."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/tmp/moe-prefill-g4-bundle')
    p.add_argument('--run', required=True, help='run tag, e.g. 20261114; the remote root is /tmp/moe-prefill-g4-RUN')
    args = p.parse_args()
    remote = '/tmp/moe-prefill-g4-' + args.run
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(set(list(ROOT.glob('infer_opt/*.py')) + list(ROOT.glob('tests/*.py')) + [
        ROOT/'pyproject.toml', ROOT/'scripts/bench_moe_prefill.py',
        ROOT/'scripts/diagnose_moe_prefill_numerics.py',
        ROOT/'scripts/validate_qwen3_5_hf.py', ROOT/'scripts/code_prompt_source.txt',
        ROOT/'scripts/quantize_qwen3_5_moe_streaming.py',
        *list((ROOT/'scripts/moe_prefill_g4').glob('*.py'))]))
    manifest = {'files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in files}}
    manifest_path = out/'source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(out/'source.tar.gz', 'w:gz') as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        tar.add(manifest_path, arcname='source-manifest.json')
    (out/'launch.py').write_text('''from pathlib import Path
import json, subprocess, sys, tarfile
root = Path(''' + repr(remote) + ''')
root.mkdir(parents=True, exist_ok=True)
assert not (root/'launch.json').exists(), 'Do not start duplicate workers'
with tarfile.open('/tmp/moe-prefill-source.tar.gz') as tar:
    tar.extractall(root/'source', filter='data')
with (root/'worker.stdout').open('w') as out, (root/'worker.stderr').open('w') as err:
    p = subprocess.Popen([sys.executable, '-u', str(root/'source/scripts/moe_prefill_g4/worker.py')],
                         stdout=out, stderr=err, start_new_session=True)
(root/'launch.json').write_text(json.dumps({'pid': p.pid}))
print(json.dumps({'pid': p.pid}))
''')
    print(out)


if __name__ == '__main__':
    main()
