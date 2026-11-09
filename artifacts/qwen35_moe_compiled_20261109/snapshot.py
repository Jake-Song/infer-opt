from pathlib import Path
import tarfile
root=Path('/content/q35moe-compiled-20261109/results')
path=Path('/content/q35moe-compiled-20261109-partial-results.tar.gz')
with tarfile.open(path.with_suffix('.tmp'),'w:gz') as archive:
 for item in sorted(root.rglob('*')):
  if item.is_file() and item.suffix!='.tmp':archive.add(item,arcname=str(item.relative_to(root)))
path.with_suffix('.tmp').replace(path)
print({'archive':str(path),'bytes':path.stat().st_size})
