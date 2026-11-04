import subprocess
import sys
from pathlib import Path
root = Path('/content/q35moe-llama-q8')
code = """from huggingface_hub import HfApi, hf_hub_download
import json
from pathlib import Path
repo='ggml-org/Qwen3.5-35B-A3B-GGUF'
info=HfApi().model_info(repo)
Path('/content/q35moe-llama-q8/download-revision.json').write_text(json.dumps({'repo':repo,'revision':info.sha}))
print(hf_hub_download(repo,'Qwen3.5-35B-A3B-Q8_0.gguf',revision=info.sha,cache_dir='/tmp/infer-opt-models/huggingface/hub'),flush=True)
"""
with (root / 'download.log').open('w') as log:
    p = subprocess.Popen([sys.executable, '-c', code], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print('download pid', p.pid)
