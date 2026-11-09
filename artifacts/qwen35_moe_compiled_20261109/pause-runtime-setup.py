import json,os,signal,shutil,time
from pathlib import Path
root=Path('/content/q35moe-compiled-20261109')
print('prior_progress', (root/'results/progress.json').read_text() if (root/'results/progress.json').exists() else None)
try:os.killpg(705,signal.SIGTERM)
except ProcessLookupError:pass
time.sleep(1)
if root.exists():shutil.move(str(root),str(root)+'-initial-runtime')
print('stopped initial worker for pinned PyTorch setup')
