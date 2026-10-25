"""Publish a self-contained INT8 Qwen3.5-27B checkpoint without loading the target model.

Only the original MTP tensors are downloaded and quantized on CPU. Existing INT8 shards are copied on the Hub.
Use --prepare-only to inspect the artifacts, then --publish-only to upload the same prepared release.
Publishing uses the cross-repository copy API from huggingface_hub>=2.2; model loading also works with older clients.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer_opt.hub import fetch_hub_tensors  # noqa: E402
from infer_opt.quant import prepare_quantized, quantize_linears, save_quantized  # noqa: E402
from infer_opt.qwen3_5 import Qwen3_5MTP, Qwen3_5TextConfig  # noqa: E402

BASE_REPO = "Jakemu/Qwen3.5-27B-int8-g128"
BASE_REVISION = "8e69ff4b730ad3d372918aee22e094472c1a47ed"
SOURCE_REPO = "Qwen/Qwen3.5-27B"
SOURCE_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
REPO_ID = "Jakemu/Qwen3.5-27B-int8-g128-mtp"
HEAD_FILE = "model-int8-mtp.safetensors"
MANIFEST = "publish_manifest.json"
SUPPORT_FILES = {".gitattributes", "LICENSE", "config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def merge_specs(base: dict, head: dict, *, group_size: int = 128) -> dict:
    if base.get("format") != "infer_opt.int8" or head.get("format") != "infer_opt.int8":
        raise ValueError("expected infer_opt.int8 checkpoints")
    if base.get("mtp") or any(n.startswith("mtp.") for n in base["modules"]):
        raise ValueError("the base already contains MTP")
    if "lm_head" in base["modules"] or not base["modules"]:
        raise ValueError("expected an INT8 target with floating-point lm_head")
    if any(m["group_size"] != group_size for m in base["modules"].values()):
        raise ValueError(f"the published base must use group size {group_size}")
    if not head.get("mtp") or not head["modules"] or any(not n.startswith("mtp.") for n in head["modules"]):
        raise ValueError("expected a head-only MTP checkpoint")
    if any(m["group_size"] != group_size for m in head["modules"].values()):
        raise ValueError("MTP group size differs from the target")
    if set(base["tied"]) & set(head["tied"]):
        raise ValueError("overlapping tied tensor names")
    merged = copy.deepcopy(base)
    merged["modules"].update(head["modules"])
    merged["tied"].update(head["tied"])
    merged["mtp"] = True
    return merged


def quantize_head(config: Qwen3_5TextConfig, state: dict, out: Path, *, group_size=128, min_out_features=256):
    """Keep the target out of memory; validate the saved head against every in-memory tensor."""
    from safetensors.torch import load_file

    with torch.device("meta"):
        holder = nn.Module()
        holder.mtp = Qwen3_5MTP(config)
    holder.load_state_dict(state, strict=True, assign=True)
    del state
    holder.eval()
    report = quantize_linears(holder, group_size=group_size, min_out_features=min_out_features)
    files = save_quantized(holder, out)
    if len(files) != 1:
        raise ValueError("expected the MTP head to fit in one shard")
    files[0].rename(out / HEAD_FILE)
    spec = json.loads((out / "quantization.json").read_text())
    with torch.device("meta"):
        restored = nn.Module()
        restored.mtp = Qwen3_5MTP(config)
    prepare_quantized(restored, spec)
    restored.load_state_dict(load_file(str(out / HEAD_FILE)), strict=True, assign=True)
    expected, actual = holder.state_dict(), restored.state_dict()
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        if tensor.dtype != actual[name].dtype or not torch.equal(tensor, actual[name]):
            raise RuntimeError(f"saved MTP tensor differs: {name}")
    return spec, {"summary": report.summary(), "quantized_modules": report.replaced,
                  "saved_tensors": len(actual), "round_trip_equal": True}


def model_card(repo_id: str, code_revision: str) -> str:
    return f"""---
license: apache-2.0
base_model: {SOURCE_REPO}
base_model_relation: quantized
tags:
- int8
- weight-only
- qwen3.5
- speculative-decoding
---

# {repo_id.split('/')[-1]}

Qwen3.5-27B text model with its **MTP head included**, using weight-only INT8 with symmetric absmax group-128
quantization and BF16 scales. The target shards are unchanged from [{BASE_REPO}](https://huggingface.co/{BASE_REPO}/tree/{BASE_REVISION}).
The eight MTP Linear weights are INT8 too. Embeddings, norms, small delta-rule gates and the shared **lm_head stay BF16**.
The vision tower is omitted. No original BF16 checkpoint or separate MTP download is needed at inference time.

This is the custom **infer-opt checkpoint format**, not a Transformers `AutoModelForCausalLM` checkpoint.
Use an A100 with at least 38 GiB of visible GPU memory (or a compatible larger BF16 GPU).

## Load and generate

```bash
pip install 'transformers>=5' safetensors huggingface_hub flash-linear-attention
git clone https://github.com/Jake-Song/infer-opt.git
git -C infer-opt checkout {code_revision}
```

```python
import sys
sys.path.insert(0, "infer-opt")
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.speculative import MTPSpeculativeDecoder

path = snapshot_download("{repo_id}")  # pin revision=<HF commit SHA> for reproducibility
model = Qwen3_5ForCausalLM.from_quantized_dir(path, device="cuda", kernels="fla-delta")
assert model.mtp is not None
model.fuse_projections()
tokenizer = AutoTokenizer.from_pretrained(path)
prompt = tokenizer.apply_chat_template(
    [{{"role": "user", "content": "Explain speculative decoding in three sentences."}}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
decoder = MTPSpeculativeDecoder(model, max_seq_len=2048, k=3, mode="compile-reduce-overhead")
with torch.no_grad():
    tokens = decoder.generate(ids, max_new_tokens=128)
print(tokenizer.decode(tokens[0, ids.shape[1]:], skip_special_tokens=True))
```

The first compiled generation includes compilation time. This minimal example emits a fixed number of tokens;
the repository's Colab notebook also handles EOS and compares plain and speculative decoding.
Greedy verification uses the target model; different floating-point execution paths can nevertheless change
argmax near tied logits. See the release validation report for observed token agreement; no new speed claim is
made for this packaging change.

## Provenance

- Target: `{BASE_REPO}@{BASE_REVISION}`.
- Original MTP: `{SOURCE_REPO}@{SOURCE_REVISION}` (15 BF16 tensors before quantization).
- Loader: [infer-opt `{code_revision[:12]}`](https://github.com/Jake-Song/infer-opt/tree/{code_revision}).
- `publish_manifest.json` records target-file hashes and prepared artifact hashes; `quantize_info.json` records
  MTP quantization and the exact tensor round-trip check.

License: Apache-2.0, as the original model; see `LICENSE`.
"""


def prepare_release(out: Path, repo_id: str, code_revision: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be empty; use --publish-only to resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    base = api.model_info(BASE_REPO, revision=BASE_REVISION, files_metadata=True)
    source = api.model_info(SOURCE_REPO, revision=SOURCE_REVISION)
    assert base.sha == BASE_REVISION and source.sha == SOURCE_REVISION
    src = out / "_source"
    config = Path(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION, local_dir=src / "base"))
    original = Path(hf_hub_download(SOURCE_REPO, "config.json", revision=SOURCE_REVISION, local_dir=src / "original"))
    cfg = Qwen3_5TextConfig.from_json(config)
    if cfg != Qwen3_5TextConfig.from_json(original):
        raise ValueError("base and MTP source configurations differ")
    spec_path = hf_hub_download(BASE_REPO, "quantization.json", revision=BASE_REVISION, local_dir=src / "base")
    base_spec = json.loads(Path(spec_path).read_text())
    print("Reading the original MTP tensors only", flush=True)
    head_spec, validation = quantize_head(cfg, fetch_hub_tensors(SOURCE_REPO, "mtp.", SOURCE_REVISION), out)
    merged = merge_specs(base_spec, head_spec)
    (out / "quantization.json").write_text(json.dumps(merged, indent=2) + "\n")
    info = {"source": SOURCE_REPO, "source_revision": SOURCE_REVISION, "quantized_base": BASE_REPO,
            "quantized_base_revision": BASE_REVISION, "group_size": 128, "mtp": True, "mtp_quantized": True,
            "lm_head_quantized": False, "code_revision": code_revision, "mtp_validation": validation}
    (out / "quantize_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "README.md").write_text(model_card(repo_id, code_revision))
    copied = [{"path": f.rfilename, "size": f.size, "sha256": f.lfs.sha256 if f.lfs else None,
               "blob_id": f.blob_id} for f in base.siblings
              if f.rfilename in SUPPORT_FILES or f.rfilename.startswith("model-int8-")]
    if len([f for f in copied if f["path"].endswith(".safetensors")]) != 14:
        raise ValueError("expected 14 pinned target shards")
    added = [{"path": name, "size": (out / name).stat().st_size, "sha256": sha256(out / name)}
             for name in [HEAD_FILE, "quantization.json", "quantize_info.json", "README.md"]]
    manifest = {"repo_id": repo_id, "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
                "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION, "code_revision": code_revision,
                "copied_files": copied, "added_files": added}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation), flush=True)
    return manifest


def publish_release(out: Path) -> dict:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download
    from huggingface_hub.errors import FileDuplicationError

    manifest = json.loads((out / MANIFEST).read_text())
    api, repo_id = HfApi(), manifest["repo_id"]
    for entry in manifest["added_files"]:
        if sha256(out / entry["path"]) != entry["sha256"]:
            raise ValueError(f"prepared file changed: {entry['path']}")
    if api.repo_exists(repo_id):
        files = api.list_repo_files(repo_id)
        if any(p != ".gitattributes" for p in files):
            previous = json.loads(Path(hf_hub_download(repo_id, MANIFEST)).read_text())
            if previous != manifest:
                raise ValueError("destination already contains a different release")
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    copies = [CommitOperationCopy(src_path_in_repo=f["path"], path_in_repo=f["path"],
                                  src_repo_id=manifest["base_repo"], src_repo_type="model",
                                  src_revision=manifest["base_revision"]) for f in manifest["copied_files"]]
    additions = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=out / f["path"])
                 for f in manifest["added_files"]]
    additions.append(CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=out / MANIFEST))
    message = "Publish INT8 group-128 Qwen3.5-27B with bundled MTP; keep lm_head BF16"
    try:
        commit = api.create_commit(repo_id, operations=copies + additions, commit_message=message)
    except (FileDuplicationError, ValueError) as exc:
        if isinstance(exc, ValueError) and "no xet hash" not in str(exc):
            raise
        print("Server-side copy unavailable; transferring unchanged source files through disk", flush=True)
        transferred = []
        for f in manifest["copied_files"]:
            path = Path(hf_hub_download(manifest["base_repo"], f["path"], revision=manifest["base_revision"],
                                        local_dir=out / "_transfer"))
            if f["sha256"] and sha256(path) != f["sha256"]:
                raise RuntimeError(f"source hash mismatch: {f['path']}")
            transferred.append(CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=path))
        commit = api.create_commit(repo_id, operations=transferred + additions, commit_message=message)
    remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    by_name = {f.rfilename: f for f in remote.siblings}
    for f in manifest["copied_files"] + manifest["added_files"]:
        actual = by_name[f["path"]]
        if actual.size != f["size"]:
            raise RuntimeError(f"remote size mismatch: {f['path']}")
        if actual.lfs and f["sha256"] and actual.lfs.sha256 != f["sha256"]:
            raise RuntimeError(f"remote hash mismatch: {f['path']}")
        if f.get("blob_id") and not actual.lfs and actual.blob_id != f["blob_id"]:
            raise RuntimeError(f"remote blob mismatch: {f['path']}")
        if f["sha256"] and not actual.lfs:
            checked = Path(hf_hub_download(repo_id, f["path"], revision=commit.oid, local_dir=out / "_verification"))
            if sha256(checked) != f["sha256"]:
                raise RuntimeError(f"remote content mismatch: {f['path']}")
    result = {"repo_id": repo_id, "revision": commit.oid, "url": str(commit.commit_url),
              "copied_files": len(copies), "added_files": len(additions), "verified": True}
    (out / "publish_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--code-revision", help="published infer-opt commit containing MTP checkpoint support")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if not args.publish_only:
        if not args.code_revision or len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
            parser.error("--code-revision must be a full commit SHA")
        prepare_release(args.out, args.repo_id, args.code_revision)
    if not args.prepare_only:
        publish_release(args.out)


if __name__ == "__main__":
    main()
