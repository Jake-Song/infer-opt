from pathlib import Path
import hashlib,json,tarfile
cache=Path('/tmp/infer-opt-models/q35moe-round-profile-20261109/inductor')
out=Path('/content/q35moe-round-profile-20261109/results')
rows=[]
with tarfile.open(out/'compiler-python.tar.gz','w:gz') as archive:
 for path in sorted(cache.rglob('*.py')):
  data=path.read_bytes();name=str(path.relative_to(cache))
  rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  archive.add(path,arcname=name)
(out/'compiler-python-manifest.json').write_text(json.dumps(rows,indent=2))
print({'python_modules':len(rows),'source_bytes':sum(r['bytes'] for r in rows)})
