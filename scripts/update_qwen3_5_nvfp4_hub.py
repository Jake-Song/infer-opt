"""Attach provenance and measured evidence to the authorized native NVFP4 release."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='Jakemu/Qwen3.5-27B-nvfp4-mtp')
    p.add_argument('--results',type=Path)
    p.add_argument('--note',help='Append a factual release evaluation note to the model card')
    a=p.parse_args()
    api=HfApi()
    provenance=json.loads(Path(hf_hub_download(a.repo,'source_revision.json')).read_text())
    license_file=hf_hub_download(provenance['model'],'LICENSE',revision=provenance['revision'])
    requirements=b'''# Use an SM120-capable CUDA PyTorch (validated: torch 2.11.0+cu130).
flashinfer-python==0.6.6
flash-linear-attention==0.6.0
transformers==5.20.2
safetensors==0.8.0
huggingface_hub==1.33.0
'''
    operations=[CommitOperationAdd(path_in_repo='LICENSE',path_or_fileobj=license_file),
                CommitOperationAdd(path_in_repo='requirements.txt',path_or_fileobj=requirements)]
    if a.note:
        card=Path(hf_hub_download(a.repo,'README.md')).read_text()
        heading='\n## Evaluation update\n'
        card=card.split(heading)[0]+heading+'\n'+a.note+'\n'
        operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()))
    if a.results:
        allowed=['accuracy.json','result_status.json','report.md','smoke/smoke.json','export/export_validation.json',
                 'bf16/measurements.json','int8/measurements.json','nvfp4/measurements.json']
        for name in allowed:
            file=a.results/name
            if file.exists():
                operations.append(CommitOperationAdd(path_in_repo='evaluation/'+name,path_or_fileobj=str(file)))
    result=api.create_commit(a.repo,operations=operations,commit_message='Add license, pinned inference requirements and evaluation evidence')
    print(result.oid)


if __name__=='__main__':
    main()
