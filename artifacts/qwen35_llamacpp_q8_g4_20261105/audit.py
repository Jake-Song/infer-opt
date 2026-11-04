import json
import shutil
import subprocess
import tarfile
from pathlib import Path
root = Path('/content/q35moe-llama-q8')
command = json.loads((root / 'commands.json').read_text())[0]
command = command[:command.index('-p')]
command[command.index('-r') + 1] = '1'
command += ['-p', '0', '-n', '1', '-v']
with (root / 'offload-audit.stdout').open('w') as out, (root / 'offload-audit.stderr').open('w') as err:
    result = subprocess.run(command, stdout=out, stderr=err, timeout=90)
assert result.returncode == 0
shutil.copy(root / 'llama.cpp/build/CMakeCache.txt', root / 'CMakeCache.txt')
log = (root / 'offload-audit.stderr').read_text()
for line in log.splitlines():
    if any(x in line.lower() for x in ['offload', 'flash_attn', 'cuda_graph', 'nextn', 'mtp', 'buffer size', 'ssm', 'n_ctx']):
        print(line)
with tarfile.open('/content/q35moe-llama-q8-results.tar.gz', 'w:gz') as tar:
    for path in root.iterdir():
        if path.is_file():
            tar.add(path, arcname=path.name)
