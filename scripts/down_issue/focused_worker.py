"""Collect a small, matched metric set after the broad triage reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/tmp/down-issue-a100-20261111')
os.environ['TRITON_CACHE_DIR']='/tmp/infer-opt-models/down-issue/triton'
os.environ['PYTHONUNBUFFERED']='1'
metrics=['gpu__time_duration.sum','smsp__issue_active.avg.pct_of_peak_sustained_active',
         'smsp__warps_eligible.avg.per_cycle_active','smsp__warps_active.avg.per_cycle_active',
         'smsp__inst_executed.sum','sm__warps_active.avg.pct_of_peak_sustained_active',
         'dram__bytes_read.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
         'lts__throughput.avg.pct_of_peak_sustained_elapsed','sm__ctas_launched.avg.pct_of_peak_sustained_elapsed',
         'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active',
         'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
         'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active',
         'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active',
         'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct',
         'smsp__warp_issue_stalled_wait_per_warp_active.pct',
         'smsp__warp_issue_stalled_not_selected_per_warp_active.pct',
         'sm__cycles_elapsed.avg.per_second']

cases=[(4,'partial','packed1','hot'),(4,'partial','packed1','cold'),
       (4,'partial','scalar4','hot'),(4,'partial','scalar4','cold'),
       (4,'partial','scalar1','hot'),(4,'partial','packed4','hot'),
       (4,'shared','packed1','hot'),(4,'disjoint','packed1','hot'),
       (1,'partial','packed1','hot'),(2,'partial','packed1','hot')]
profiles=[]
parser=argparse.ArgumentParser()
parser.add_argument('--core-only',action='store_true',help='Collect only the six T=4 partial-route comparisons.')
args=parser.parse_args()
if args.core_only:cases=cases[:6]
try:
    for t,routing,variant,cache in cases:
        name=f'focused-t{t}-{routing}-{variant}-{cache}'
        command=['ncu','--profile-from-start','off','--replay-mode','application','--cache-control','none',
                 '--clock-control','none','--launch-count','1','--force-overwrite','--export',str(OUT/name),
                 '--metrics',','.join(metrics),'--csv',sys.executable,str(ROOT/'scripts/down_issue/benchmark.py'),
                 '--out',str(OUT/name),'--profile','--tokens',str(t),'--routing',routing,'--variant',variant,'--cache',cache,'--warm-profile']
        (OUT/'progress.json').write_text(json.dumps({'stage':name,'time':time.time(),'command':command}))
        start=time.time()
        with (OUT/(name+'.stdout')).open('w') as stdout,(OUT/(name+'.stderr')).open('w') as stderr:
            proc=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
        entry={'name':name,'returncode':proc.returncode,'seconds':time.time()-start,'command':command}
        profiles.append(entry)
        (OUT/'focused-profiles.json').write_text(json.dumps(profiles,indent=2))
        if proc.returncode!=0:raise RuntimeError(f'Failed: {name}')
    (OUT/'focused-complete.json').write_text(json.dumps({'complete':True,'profiles':len(profiles)}))
    (OUT/'progress.json').write_text(json.dumps({'stage':'complete','time':time.time()}))
except BaseException as error:
    (OUT/'focused-error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2))
    raise
