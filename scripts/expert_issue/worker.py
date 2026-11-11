"""Collect matched fixed-clock gate/up and down counters, plus full-block timings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/tmp/expert-issue-a100-20261111')
OUT.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']='/tmp/infer-opt-models/expert-issue/triton'
os.environ['PYTHONUNBUFFERED']='1'

def stage(name,**extra):
    value={'stage':name,'time':time.time(),**extra}
    (OUT/'progress.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)

def run(name,command,timeout=900):
    stage(name,command=command);start=time.time()
    with (OUT/(name+'.stdout')).open('w') as stdout,(OUT/(name+'.stderr')).open('w') as stderr:
        p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=timeout)
    entry={'name':name,'command':command,'returncode':p.returncode,'seconds':time.time()-start}
    (OUT/(name+'.command.json')).write_text(json.dumps(entry,indent=2))
    if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}')
    return entry

metrics=['gpu__time_duration.sum','smsp__issue_active.avg.pct_of_peak_sustained_active',
         'smsp__issue_active.avg.pct_of_peak_sustained_elapsed','smsp__inst_executed.sum',
         'smsp__warps_eligible.avg.per_cycle_active','sm__warps_active.avg.pct_of_peak_sustained_active',
         'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active',
         'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
         'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active',
         'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active','sm__cycles_elapsed.avg.per_second',
         'dram__bytes.sum','dram__bytes.sum.peak_sustained_elapsed.per_second',
         'dram__throughput.avg.pct_of_peak_sustained_elapsed',
         'lts__t_sectors.sum','lts__t_sectors.sum.peak_sustained_elapsed.per_second',
         'lts__throughput.avg.pct_of_peak_sustained_elapsed',
         'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
         'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct',
         'smsp__warp_issue_stalled_wait_per_warp_active.pct']

try:
    preflight=json.loads((OUT/'preflight.json').read_text())
    if preflight['clock_lock']['returncode']!=0:raise RuntimeError('GPU clock locking failed; inspect preflight before using this protocol.')
    run('counter-probe',[sys.executable,str(ROOT/'scripts/expert_issue/counter_probe.py')])
    probe=json.loads((OUT/'probe-result.json').read_text())
    if not probe['counters_collected']:raise RuntimeError('Performance counters unavailable')
    bench=str(ROOT/'scripts/expert_issue/benchmark.py')
    run('timing',[sys.executable,bench,'--out',str(OUT/'timing')])
    profiles=[]
    cases=[('gate_up','packed1','hot'),('gate_up','packed1','cold'),
           ('down','packed1','hot'),('down','packed1','cold'),
           ('gate_up','scalar4','hot'),('gate_up','scalar4','cold'),
           ('gate_up','scalar1','hot'),('gate_up','packed4','hot')]
    for kind,variant,cache in cases:
        name=f'ncu-t4-partial-{kind}-{variant}-{cache}'
        command=['ncu','--profile-from-start','off','--replay-mode','application','--cache-control','none',
                 '--clock-control','none','--launch-count','1','--force-overwrite','--export',str(OUT/name),
                 '--metrics',','.join(metrics),'--csv',sys.executable,bench,'--out',str(OUT/name),
                 '--profile','--tokens','4','--routing','partial','--kind',kind,'--variant',variant,'--cache',cache]
        entry=run(name,command,timeout=900);profiles.append(entry)
        (OUT/'profiles.json').write_text(json.dumps(profiles,indent=2))
        for suffix,extra in [('raw.csv',[]),('raw-base.csv',['--print-units','base'])]:
            proc=subprocess.run(['ncu','--import',str(OUT/(name+'.ncu-rep')),'--page','raw','--csv',*extra],capture_output=True,text=True,timeout=60)
            (OUT/(name+'.'+suffix)).write_text(proc.stdout)
            if proc.returncode:raise RuntimeError('Raw metric export failed: '+proc.stderr)
    (OUT/'complete.json').write_text(json.dumps({'complete':True,'profiles':len(profiles)}))
    stage('complete')
except BaseException as error:
    (OUT/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2))
    stage('failed',error=str(error))
    raise
finally:
    p=subprocess.run(['nvidia-smi','--reset-gpu-clocks'],capture_output=True,text=True,timeout=30)
    (OUT/'clock-reset.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
