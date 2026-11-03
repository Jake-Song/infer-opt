"""Regenerate, validate, and publish the custom Qwen3.5 MoE INT8+MTP checkpoint.

Export requires a GPU with enough memory for the model and validation caches.
Use --export-only remotely and --publish-existing locally to keep HF credentials
on the publishing machine. Credentials are never included in the artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--repo", default="Jakemu/Qwen3.5-35B-A3B-int8-g128-mtp")
    p.add_argument("--revision", default="59d61f3ce65a6d9863b86d2e96597125219dc754")
    p.add_argument("--code-revision", required=True)
    p.add_argument("--export-only", action="store_true", help="Validate and package without HF authentication or upload")
    p.add_argument("--publish-existing", action="store_true", help="Publish a previously validated export, without a GPU")
    p.add_argument("--resume-empty-repo", action="store_true", help="Resume an interrupted upload to an empty destination")
    p.add_argument("--wait-for-files", action="store_true", help="Verify and pre-upload each file as an atomic local download finishes")
    a = p.parse_args()
    if a.export_only and a.publish_existing:
        p.error("--export-only and --publish-existing are mutually exclusive")
    if a.wait_for_files and not a.publish_existing:
        p.error("--wait-for-files requires --publish-existing")
    from huggingface_hub import HfApi, snapshot_download

    root = Path(__file__).resolve().parents[1]
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint"
    started = time.time()

    def stage(name, **extra):
        result = {"stage": name, "elapsed_s": time.time() - started, **extra}
        (work / "progress.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)

    def run(script, args, log):
        with (work / log).open("w") as f:
            subprocess.run([sys.executable, str(root / "scripts" / script), *args],
                           cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True)

    api = HfApi()
    if not a.export_only:
        assert api.whoami()["name"] == a.repo.split("/")[0]
        if api.repo_exists(a.repo):
            if not a.resume_empty_repo or set(api.list_repo_files(a.repo)) - {".gitattributes"}:
                raise ValueError("Destination exists and is not an explicitly allowed empty repository")
    if a.publish_existing:
        publish(api, a, checkpoint, work, stage, started)
        return
    import torch
    import triton
    environment = {
        "source": "Qwen/Qwen3.5-35B-A3B", "source_revision": a.revision,
        "code_revision": a.code_revision, "torch": str(torch.__version__),
        "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }
    (work / "environment.json").write_text(json.dumps(environment, indent=2))
    stage("download")
    source = snapshot_download(environment["source"], revision=a.revision,
                               cache_dir=str(work / "source-cache"),
                               allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE"],
                               max_workers=8)
    stage("convert")
    run("quantize_qwen3_5_moe_streaming.py",
        ["--source", source, "--out", str(checkpoint)], "conversion.log")
    stage("validate")
    validation_path = work / "checkpoint-validation.json"
    run("validate_quantized_qwen3_5_moe.py",
        ["--weights", str(checkpoint), "--out", str(validation_path)], "validation.log")
    validation = json.loads(validation_path.read_text())
    assert validation["expert_modules"] == 41 and validation["mtp"]
    assert all(row["tokens_equal"] for row in validation["parity"])
    # Publish compact validation evidence; retain the full kernel profile locally.
    validation.pop("kernels", None)
    validation["weights"] = a.repo
    (checkpoint / "validation.json").write_text(json.dumps(validation, indent=2))
    (checkpoint / "environment.json").write_text(json.dumps(environment, indent=2))
    (checkpoint / "README.md").write_text(f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
tags:
- quantized
- int8
- moe
- infer-opt
---
# Qwen3.5-35B-A3B INT8 group-128 + MTP

Custom **infer-opt** weight-only W8A16 checkpoint, including the MTP head.
This is a text-only inference export. Load with infer-opt, not Transformers
AutoModel or vLLM. Routers, small gates, norms, embeddings and lm_head retain
BF16 weights. Quantization uses symmetric round-to-nearest, group size 128,
with BF16 scales and no calibration dataset.

Source revision: `{a.revision}`.
Inference code: [Jake-Song/infer-opt](https://github.com/Jake-Song/infer-opt/tree/{a.code_revision}).

```python
from huggingface_hub import snapshot_download
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM

path = snapshot_download("{a.repo}")
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
```

Install the inference repository and its CUDA dependencies (PyTorch, Triton,
safetensors, transformers and flash-linear-attention). See environment.json
for the versions used for this export and GPU validation.

## Validation and limitations

This regenerated checkpoint passed strict reload with 41 INT8 expert modules,
ordinary 64-token generation, and token parity against ordinary decoding for
MTP k=1,2 using sequential and consistent verification on {environment['gpu']}.
See validation.json. These are packaging/inference checks, not task-level
accuracy evaluation or evidence of an MTP speedup. A100 validation was not run
for this release. Compiled MoE MTP has known parity failures; use the default
sequential verification. The original implementation's quality measurements
are documented in the inference repository's MOE_STATUS.md.

All checkpoint files are covered by SHA-256 entries in release_manifest.json.
The original model's Apache-2.0 license applies.
""")
    stage("hash")
    manifest = {"repo": a.repo, **environment, "files": {}}
    for path in sorted(checkpoint.iterdir()):
        manifest["files"][path.name] = {"size": path.stat().st_size, "sha256": digest(path)}
    (checkpoint / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    (work / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    if a.export_only:
        stage("export_complete")
        return
    publish(api, a, checkpoint, work, stage, started)


def publish(api, a, checkpoint, work, stage, started):
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    manifest = json.loads((checkpoint / "release_manifest.json").read_text())
    assert manifest["repo"] == a.repo and manifest["source_revision"] == a.revision
    assert manifest["code_revision"] == a.code_revision
    validation = json.loads((checkpoint / "validation.json").read_text())
    assert validation["expert_modules"] == 41 and validation["mtp"]
    assert len(validation["parity"]) == 4
    assert all(row["tokens_equal"] for row in validation["parity"])
    def verify_file(name, expected):
        path = checkpoint / name
        if not path.exists() and a.wait_for_files:
            stage("waiting_for_file", file=name)
            while not path.exists():
                if (work / "collect_error.json").exists():
                    raise RuntimeError("Checkpoint download failed; see collect_error.json")
                time.sleep(10)
        assert path.stat().st_size == expected["size"] and digest(path) == expected["sha256"], name

    stage("verify_local")
    if not a.wait_for_files:
        assert {p.name for p in checkpoint.iterdir()} == set(manifest["files"]) | {"release_manifest.json"}
        for name, expected in manifest["files"].items():
            verify_file(name, expected)
    stage("upload")
    api.create_repo(a.repo, private=False, exist_ok=a.resume_empty_repo)
    info = api.model_info(a.repo)
    if {f.rfilename for f in info.siblings} - {".gitattributes"}:
        raise ValueError("Refusing to overwrite a nonempty destination")
    operations = []
    for name, expected in manifest["files"].items():
        if a.wait_for_files:
            verify_file(name, expected)
        stage("upload_file", file=name, completed=len(operations), total=len(manifest["files"]))
        operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(checkpoint / name))
        api.preupload_lfs_files(repo_id=a.repo, additions=[operation], free_memory=True)
        operations.append(operation)
    operations.append(CommitOperationAdd(path_in_repo="release_manifest.json",
                                         path_or_fileobj=str(checkpoint / "release_manifest.json")))
    stage("commit")
    commit = api.create_commit(repo_id=a.repo, operations=operations, parent_commit=info.sha,
                              commit_message="Publish validated INT8 group-128 MoE checkpoint with MTP")
    stage("verify_remote", commit=commit.oid)
    info = api.model_info(a.repo, revision=commit.oid, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    verified = []
    for name, expected in manifest["files"].items():
        entry = remote[name]
        assert entry.size == expected["size"], name
        if entry.lfs is not None:
            assert entry.lfs.sha256 == expected["sha256"], name
        else:
            downloaded = Path(hf_hub_download(a.repo, name, revision=commit.oid,
                                             cache_dir=str(work / "verify-cache")))
            assert digest(downloaded) == expected["sha256"], name
        verified.append(name)
    downloaded_manifest = Path(hf_hub_download(a.repo, "release_manifest.json", revision=commit.oid,
                                              cache_dir=str(work / "verify-cache")))
    assert digest(downloaded_manifest) == digest(checkpoint / "release_manifest.json")
    result = {"repo": a.repo, "url": f"https://huggingface.co/{a.repo}", "commit": commit.oid,
              "verified_files": verified, "manifest_verified": True,
              "checkpoint_bytes": json.loads((checkpoint / "conversion_info.json").read_text())["checkpoint_bytes"],
              "validation": validation, "elapsed_s": time.time() - started}
    (work / "publish_result.json").write_text(json.dumps(result, indent=2))
    stage("complete", commit=commit.oid)


if __name__ == "__main__":
    main()
