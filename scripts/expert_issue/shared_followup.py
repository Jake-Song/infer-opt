"""After the main sweep, profile a fully shared hot gate/up working set."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/tmp/expert-issue-source')
OUT=Path('/tmp/expert-issue-a100-20261111')
os.environ['TRITON_CACHE_DIR']='/tmp/infer-opt-models/expert-issue/triton'
try:
    deadline=time.time()+2400
    while not ((OUT/'complete.json').exists() and (OUT/'clock-reset.json').exists()):
        if (OUT/'error.json').exists():raise RuntimeError('Main worker failed')
        if time.time()>deadline:raise RuntimeError('Main worker did not finish before follow-up deadline')
        time.sleep(5)
    assert json.loads((OUT/'complete.json').read_text())['profiles']==8
    lock=subprocess.run(['nvidia-smi','--lock-gpu-clocks=1095,1095'],capture_output=True,text=True,timeout=30)
    (OUT/'shared-clock-lock.json').write_text(json.dumps({'returncode':lock.returncode,'stdout':lock.stdout,'stderr':lock.stderr},indent=2))
    if lock.returncode:raise RuntimeError('Clock lock failed')
    tree=ast.parse((ROOT/'scripts/expert_issue/worker.py').read_text())
    metrics=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='metrics' for t in node.targets))
    name='ncu-t4-shared-gate_up-packed1-hot'
    command=['ncu','--profile-from-start','off','--replay-mode','application','--cache-control','none',
             '--clock-control','none','--launch-count','1','--force-overwrite','--export',str(OUT/name),
             '--metrics',','.join(metrics),'--csv',sys.executable,str(ROOT/'scripts/expert_issue/benchmark.py'),
             '--out',str(OUT/name),'--profile','--tokens','4','--routing','shared','--kind','gate_up','--variant','packed1','--cache','hot']
    (OUT/'progress.json').write_text(json.dumps({'stage':name,'time':time.time(),'command':command}))
    start=time.time()
    with (OUT/(name+'.stdout')).open('w') as stdout,(OUT/(name+'.stderr')).open('w') as stderr:
        p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=900)
    entry={'name':name,'returncode':p.returncode,'command':command,'seconds':time.time()-start}
    (OUT/'shared-profiles.json').write_text(json.dumps([entry],indent=2))
    if p.returncode:raise RuntimeError('Shared hot profile failed')
    for suffix,extra in [('raw.csv',[]),('raw-base.csv',['--print-units','base'])]:
        p=subprocess.run(['ncu','--import',str(OUT/(name+'.ncu-rep')),'--page','raw','--csv',*extra],capture_output=True,text=True,timeout=60)
        if p.returncode:raise RuntimeError(p.stderr)
        (OUT/(name+'.'+suffix)).write_text(p.stdout)
    (OUT/'shared-complete.json').write_text(json.dumps({'complete':True,'profiles':1}))
    (OUT/'progress.json').write_text(json.dumps({'stage':'complete-all','time':time.time()}))
except BaseException as error:
    (OUT/'shared-error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2))
    raise
finally:
    # This is reached after the main worker has finished, unless it failed.
    p=subprocess.run(['nvidia-smi','--reset-gpu-clocks'],capture_output=True,text=True,timeout=30)
    (OUT/'shared-clock-reset.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
