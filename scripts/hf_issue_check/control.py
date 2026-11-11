"""Submit and recover the user's model-free HF Jobs profiling experiments."""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tarfile

from huggingface_hub import HfApi


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['submit','status','logs','cancel'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--script',type=Path)
    p.add_argument('--job')
    p.add_argument('--timeout',default='10m')
    p.add_argument('--name',default='moe-counter-access-probe')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    if args.action=='submit':
        script=args.script.read_text()
        image='runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster'
        request={'image':image,'flavor':'a100-large','timeout':args.timeout,
                 'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'name':args.name}
        (args.out/'request.json').write_text(json.dumps(request,indent=2))
        job=api.run_job(image=image,command=['python','-u','-c',script],flavor='a100-large',timeout=args.timeout,
                        attempts=1,name=args.name,env={'HF_HOME':'/tmp/infer-opt-models/huggingface',
                            'TORCHINDUCTOR_CACHE_DIR':'/tmp/infer-opt-models/inductor',
                            'TRITON_CACHE_DIR':'/tmp/infer-opt-models/triton'})
        (args.out/'job.json').write_text(json.dumps(asdict(job),indent=2,default=str))
        print(json.dumps({'id':job.id,'url':job.url,'stage':job.status.stage}))
        return
    if args.action=='cancel':
        api.cancel_job(job_id=args.job)
    job=api.inspect_job(job_id=args.job)
    status={'id':job.id,'stage':job.status.stage,'message':job.status.message,'url':job.url}
    for key in ['created_at','started_at','finished_at','durations']:
        value=getattr(job,key,None)
        status[key]=asdict(value) if value is not None and hasattr(value,'__dataclass_fields__') else value
    (args.out/'status.json').write_text(json.dumps(status,indent=2,default=str))
    if args.action=='logs':
        records=list(api.fetch_job_logs(job_id=args.job,follow=False))
        lines=[x['data'] if isinstance(x,dict) else str(x) for x in records]
        (args.out/'logs.json').write_text(json.dumps(records,indent=2,default=str))
        (args.out/'job.log').write_text('\n'.join(lines)+'\n')
        chunks={}
        manifest=None
        for raw in lines:
            for line in raw.splitlines():
                if line.startswith('HF_ARTIFACT_BEGIN '):manifest=json.loads(line[len('HF_ARTIFACT_BEGIN '):])
                if line.startswith('HF_ARTIFACT_CHUNK '):
                    obj=json.loads(line[len('HF_ARTIFACT_CHUNK '):]);chunks[obj['index']]=obj['data']
        recovered=False
        if manifest and len(chunks)==manifest['chunks']:
            data=base64.b64decode(''.join(chunks[i] for i in range(manifest['chunks'])))
            assert hashlib.sha256(data).hexdigest()==manifest['sha256']
            archive=args.out/'results.tar.gz';archive.write_bytes(data)
            with tarfile.open(archive) as a:a.extractall(args.out/'results',filter='data')
            (args.out/'recovery.json').write_text(json.dumps(manifest,indent=2))
            recovered=True
        status.update(log_records=len(records),artifact_recovered=recovered,
                      recent=[x for x in lines[-8:] if 'HF_ARTIFACT_' not in x])
    print(json.dumps(status,default=str))


if __name__=='__main__':main()
