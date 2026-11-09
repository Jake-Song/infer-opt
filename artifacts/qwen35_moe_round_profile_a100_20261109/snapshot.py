from pathlib import Path
import tarfile
root=Path('/content/q35moe-round-profile-20261109/results');dest=Path('/content/q35moe-round-profile-20261109-partial-results.tar.gz')
with tarfile.open(dest.with_suffix('.tmp'),'w:gz') as a:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix!='.tmp':a.add(p,arcname=str(p.relative_to(root)))
dest.with_suffix('.tmp').replace(dest);print({'path':str(dest),'bytes':dest.stat().st_size})
