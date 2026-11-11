"""Recover progress and a checksummed snapshot without touching GPU work."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

root=Path('/tmp/expert-issue-a100-20261111')
for name in ['progress.json','error.json','complete.json','clock-reset.json','shared-complete.json','shared-error.json','shared-clock-reset.json','timing/complete.json']:
    p=root/name
    if p.exists():
        value=json.loads(p.read_text());value.pop('command',None)
        if 'time' in value:value['elapsed_seconds']=time.time()-value['time']
        print(name+' '+json.dumps(value),flush=True)
progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
names=['worker.stderr',progress.get('stage','')+'.stdout',progress.get('stage','')+'.stderr']
if progress.get('stage')=='timing':names+=['timing.stdout','timing.stderr']
for name in dict.fromkeys(names):
    p=root/name
    if p.exists():
        text=p.read_text();print(name+' '+json.dumps({'bytes':len(text),'passes':text.count('Connected to process')}),flush=True)
        print(text[-900:],flush=True)
archive=Path('/tmp/expert-issue-results.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        data=p.read_bytes()
        if p.suffix=='.json':
            for attempt in range(5):
                try:json.loads(data);break
                except json.JSONDecodeError:
                    time.sleep(.02);data=p.read_bytes()
            else:continue
        info=tarfile.TarInfo('results/'+str(p.relative_to(root)))
        info.size=len(data);info.mtime=p.stat().st_mtime
        tar.addfile(info,io.BytesIO(data))
print('ARCHIVE '+json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)
