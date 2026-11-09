import hashlib,json,os,traceback
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/content/q35moe-verify-opt-20261109');out=root/'results'
cache=Path('/tmp/infer-opt-models/q35moe-verify-opt-20261109')
def save(n,v):
 p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
try:
 save('download-progress.json',{'stage':'download'})
 repo='Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp';rev='331d7db90ac4b9768c3f4a7bcd64d3b29bb9d388'
 weights=snapshot_download(repo,revision=rev,cache_dir=str(cache/'int8'),token=False,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
 save('download-progress.json',{'stage':'hashes'})
 info=HfApi().model_info(repo,revision=rev,files_metadata=True,token=False)
 rows=[]
 for f in info.siblings:
  p=Path(weights)/f.rfilename
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as s:
   for b in iter(lambda:s.read(32*2**20),b''):h.update(b)
  digest=h.hexdigest()
  if f.lfs and digest!=f.lfs.sha256:raise RuntimeError('hash mismatch: '+f.rfilename)
  rows.append({'file':f.rfilename,'bytes':p.stat().st_size,'sha256':digest,'lfs_verified':f.lfs is not None})
 save('checkpoint.json',{'weights':weights,'repo':repo,'revision':rev,'files':rows})
 save('download-progress.json',{'stage':'complete'})
except BaseException:save('download-error.json',{'traceback':traceback.format_exc()});raise
