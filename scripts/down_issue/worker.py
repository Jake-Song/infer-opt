"""Run unprofiled timings, then matched down-only NCU reports on Colab."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/tmp/down-issue-a100-20261111')
OUT.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']='/tmp/infer-opt-models/down-issue/triton'
os.environ['PYTHONUNBUFFERED']='1'

def stage(value,**extra):
    payload={'stage':value,'time':time.time(),**extra}
    (OUT/'progress.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload),flush=True)

def run(name,command,timeout=900):
    stage(name,command=command)
    start=time.time()
    with (OUT/(name+'.stdout')).open('w') as stdout,(OUT/(name+'.stderr')).open('w') as stderr:
        try:
            p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=timeout)
            result={'returncode':p.returncode}
        except subprocess.TimeoutExpired:
            result={'returncode':None,'timed_out':True}
    result.update({'command':command,'seconds':time.time()-start})
    (OUT/(name+'.command.json')).write_text(json.dumps(result,indent=2))
    return result

try:
    bench=str(ROOT/'scripts/down_issue/benchmark.py')
    result=run('timing',[sys.executable,bench,'--out',str(OUT/'timing')])
    if result['returncode']!=0: raise RuntimeError('Timing failed')
    run('ncu-sections',['ncu','--list-sections'])
    run('ncu-metrics',['ncu','--query-metrics','--query-metrics-mode','all'])
    sections=['SpeedOfLight','SchedulerStats','WarpStateStats','InstructionStats','ComputeWorkloadAnalysis','MemoryWorkloadAnalysis','Occupancy','LaunchStats']
    cases=[(4,'partial','packed1','hot'),(4,'partial','packed1','cold'),
           (4,'partial','scalar4','hot'),(4,'partial','scalar4','cold'),
           (4,'partial','scalar1','hot'),(4,'partial','packed4','hot'),
           (4,'shared','packed1','hot'),(4,'disjoint','packed1','hot'),
           (1,'partial','packed1','hot'),(2,'partial','packed1','hot')]
    profiles=[]
    for t,routing,variant,cache in cases:
        name=f'ncu-t{t}-{routing}-{variant}-{cache}'
        command=['ncu','--profile-from-start','off','--replay-mode','application','--cache-control','none',
                 '--clock-control','none','--launch-count','1','--force-overwrite','--export',str(OUT/name),'--csv']
        for section in sections: command+=['--section',section]
        command += [sys.executable,bench,'--out',str(OUT/name),'--profile','--tokens',str(t),'--routing',routing,'--variant',variant,'--cache',cache]
        result=run(name,command,timeout=900)
        profiles.append({'name':name,**result})
        (OUT/'profiles.json').write_text(json.dumps(profiles,indent=2))
        if result['returncode']!=0:
            raise RuntimeError(f'Profile failed: {name}')
    stage('complete')
    (OUT/'worker-complete.json').write_text(json.dumps({'complete':True,'profiles':len(profiles)}))
except BaseException as error:
    (OUT/'worker-error.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},indent=2))
    stage('failed',error=str(error))
    raise
