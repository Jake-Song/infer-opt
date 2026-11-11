"""Collect readable standard-perf samples and matched kernel counter probes."""
import os
import sys

import worker

bench = str(worker.ROOT/'llama.cpp/build/bin/test-backend-ops')
env = dict(os.environ, INFER_MOE_CASES='1', INFER_MOE_SIMPLE='1')
worker.stage('standard-perf-console')
# This pinned upstream CSV printer omits performance columns; console keeps them.
for repeat in range(3):
    worker.run([bench,'perf','-b','CUDA0','-o','MUL_MAT_ID'], f'standard-console-{repeat}', env=env, timeout=900)
worker.stage('matched-kernel-counters')
for tokens, routing in [(1,'shared'),(4,'shared'),(4,'disjoint')]:
    prof_env = dict(os.environ, INFER_MOE_CASES='1', INFER_MOE_MEASURE='1',INFER_MOE_REPEATS='2',
                    INFER_MOE_SELECT=f'tokens={tokens},kind=gate_up,routing={routing}')
    worker.run(['ncu','--target-processes','all','--set','full','--kernel-name','regex:.*mul_mat_vec_q_moe.*',
                '--launch-skip','3','--launch-count','1','--csv',bench],
               f'ncu-llama-gate-up-t{tokens}-{routing}', env=prof_env, check=False, timeout=600)
for implementation in ['reference','packed']:
    worker.run(['ncu','--target-processes','all','--set','full','--kernel-name','regex:.*_project.*',
                '--launch-skip','3','--launch-count','1','--csv',sys.executable,
                str(worker.BUNDLE/'scripts/llamacpp_moe/compare.py'),
                '--out',str(worker.OUT/f'ncu-{implementation}-samples.json'),
                '--profile','--implementation',implementation,'--repeats','2'],
               f'ncu-{implementation}', check=False, timeout=600)
worker.stage('followup-complete')
worker.save('followup-exit.json', {'returncode':0})
worker.snapshot()
