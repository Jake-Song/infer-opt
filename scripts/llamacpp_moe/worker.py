"""Run the pinned model-free experiment on a dedicated CUDA machine."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/workspace/infer-opt-llama-moe')
BUNDLE = ROOT / 'source'
OUT = ROOT / 'results'
REV = 'a0ad695cb212db3850f163121ddc687513448d42'
START = time.time()
os.environ.update(PATH='/usr/local/cuda/bin:' + os.environ['PATH'],
                  TORCHINDUCTOR_CACHE_DIR='/tmp/infer-opt-models/llama-moe/inductor',
                  TRITON_CACHE_DIR='/tmp/infer-opt-models/llama-moe/triton', OMP_NUM_THREADS='4')
OUT.mkdir(parents=True, exist_ok=True)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def snapshot():
    dst = ROOT / 'results.tar.gz'
    tmp = dst.with_suffix('.tmp')
    with tarfile.open(tmp, 'w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix != '.tmp':
                tar.add(p, arcname=str(p.relative_to(OUT)))
    tmp.replace(dst)


def stage(name):
    save('progress.json', {'stage': name, 'elapsed_s': time.time() - START})
    print(name, flush=True)
    snapshot()


def run(argv, name, env=None, check=True, timeout=1800):
    save(name + '.command.json', argv)
    with (OUT / (name + '.stdout')).open('w') as out, (OUT / (name + '.stderr')).open('w') as err:
        proc = subprocess.run(argv, stdout=out, stderr=err, env=env, timeout=timeout)
    save(name + '.exit.json', {'returncode': proc.returncode})
    snapshot()
    if check and proc.returncode:
        raise RuntimeError(name + ': ' + (OUT / (name + '.stderr')).read_text()[-6000:])
    return proc.returncode


def main():
    stage('environment')
    run(['nvidia-smi', '-q'], 'gpu-before')
    run(['nvcc', '--version'], 'nvcc')
    run([sys.executable, '-m', 'pip', 'install', 'numpy', 'pytest'], 'install-small-deps')
    import torch
    import triton
    assert 'A100' in torch.cuda.get_device_name()
    save('environment.json', {'gpu': torch.cuda.get_device_name(), 'vram_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
        'torch': str(torch.__version__), 'triton': triton.__version__, 'python': sys.version,
        'cpu_count': os.cpu_count(), 'llama_revision': REV, 'model_downloads': False})
    stage('clone')
    src = ROOT / 'llama.cpp'
    src.mkdir(exist_ok=True)
    for argv, name in [(['git','init',str(src)], 'git-init'),
        (['git','-C',str(src),'remote','add','origin','https://github.com/ggml-org/llama.cpp.git'], 'git-origin'),
        (['git','-C',str(src),'fetch','--depth','1','origin',REV], 'git-fetch'),
        (['git','-C',str(src),'checkout','--detach','FETCH_HEAD'], 'git-checkout')]:
        run(argv, name)
    run([sys.executable, str(BUNDLE / 'scripts/llamacpp_moe/patch_source.py'), str(src)], 'patch')
    run(['git','-C',str(src),'diff'], 'source-patch')
    stage('build')
    build = src / 'build'
    run(['cmake','-S',str(src),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=ON',
         '-DCMAKE_CUDA_ARCHITECTURES=80','-DGGML_CUDA_GRAPHS=ON','-DGGML_CUDA_FA=OFF','-DGGML_CUDA_NCCL=OFF',
         '-DCMAKE_CUDA_FLAGS=-lineinfo','-DBUILD_SHARED_LIBS=OFF','-DLLAMA_BUILD_TESTS=ON',
         '-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF'], 'cmake')
    run(['cmake','--build',str(build),'-j','12','--target','test-backend-ops'], 'build', timeout=1800)
    shutil.copy2(build / 'CMakeCache.txt', OUT / 'CMakeCache.txt')
    bench = str(build / 'bin/test-backend-ops')
    base_env = dict(os.environ, INFER_MOE_CASES='1')
    stage('correctness')
    run([bench,'test','-b','CUDA0'], 'correctness', env=base_env, timeout=1200)
    stage('standard-perf')
    for repeat in range(3):
        env = dict(base_env, INFER_MOE_SIMPLE='1')
        run([bench,'perf','-b','CUDA0','-o','MUL_MAT_ID','--output','csv'], f'standard-perf-{repeat}', env=env, timeout=900)
    stage('isolated-graph-perf')
    run([bench], 'isolated-graph-perf', env=dict(base_env, INFER_MOE_MEASURE='1'), timeout=1200)
    stage('infer-opt-baselines')
    run([sys.executable,str(BUNDLE/'scripts/llamacpp_moe/compare.py'),'--out',str(OUT/'infer-opt.json')], 'infer-opt', timeout=1200)
    stage('profile')
    ncu = shutil.which('ncu')
    if ncu:
        env = dict(base_env, INFER_MOE_MEASURE='1', INFER_MOE_REPEATS='2', INFER_MOE_SELECT='tokens=4,kind=gate,routing=shared')
        run([ncu,'--target-processes','all','--set','full','--launch-skip','10','--launch-count','4','--csv',bench],
            'ncu-llama', env=env, check=False, timeout=600)
    save('source-manifest.json', {str(p.relative_to(BUNDLE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    run(['nvidia-smi','-q'], 'gpu-after')
    stage('complete')


if __name__ == '__main__':
    try:
        main()
        save('exit.json', {'returncode': 0, 'elapsed_s': time.time() - START})
    except BaseException:
        save('exit.json', {'returncode': 1, 'elapsed_s': time.time() - START, 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        snapshot()
