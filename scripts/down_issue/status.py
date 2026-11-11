"""Print worker progress and archive the current results for recovery."""
import hashlib
import json
from pathlib import Path
import tarfile
import subprocess
import time

root=Path('/tmp/down-issue-a100-20261111')
for profiles in [root/'profiles.json',root/'focused-profiles.json']:
 if profiles.exists():
    for entry in json.loads(profiles.read_text()):
        report=root/(entry['name']+'.ncu-rep')
        raw=root/(entry['name']+'.raw.csv')
        if entry.get('returncode')==0 and report.exists() and not raw.exists():
            p=subprocess.run(['ncu','--import',str(report),'--page','raw','--csv'],capture_output=True,text=True,timeout=60)
            raw.write_text(p.stdout)
            raw.with_suffix('.export.json').write_text(json.dumps({'returncode':p.returncode,'stderr':p.stderr}))
        base=root/(entry['name']+'.raw-base.csv')
        if entry.get('returncode')==0 and report.exists() and not base.exists():
            p=subprocess.run(['ncu','--import',str(report),'--page','raw','--csv','--print-units','base'],capture_output=True,text=True,timeout=60)
            base.write_text(p.stdout)
            base.with_suffix('.export.json').write_text(json.dumps({'returncode':p.returncode,'stderr':p.stderr}))
for name in ['progress.json','worker-error.json','worker-complete.json','focused-complete.json','focused-error.json','core-complete.json','timing/complete.json']:
    path=root/name
    if path.exists():
        data=json.loads(path.read_text())
        data.pop('command',None)
        if 'time' in data:data['elapsed_seconds']=time.time()-data['time']
        print(name+' '+json.dumps(data),flush=True)
for name in ['worker.stderr','focused-worker.stderr','timing.stderr']:
    path=root/name
    if path.exists(): print(name+'\n'+path.read_text()[-2500:],flush=True)
progress=root/'progress.json'
if progress.exists():
    current=json.loads(progress.read_text()).get('stage','')
    if current.startswith(('ncu-t','focused-t')):
        for suffix in ['.stdout','.stderr']:
            path=root/(current+suffix)
            if path.exists():
                content=path.read_text()
                print(path.name+' '+json.dumps({'application_passes_started':content.count('Connected to process'),'bytes':len(content)}),flush=True)
                print(content[-1200:],flush=True)
archive=Path('/tmp/down-issue-results.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    tar.add(root,arcname='results')
data=archive.read_bytes()
print('ARCHIVE '+json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}),flush=True)
