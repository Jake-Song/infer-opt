"""Paired target-body comparison with the same compiled graph and dynamic expert op.

This measures target verification only, on fixed greedy-continuation tokens.
It is not an end-to-end MTP throughput or ordinary-decoder parity claim.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_opt import moe_kernels as kernels
from infer_opt.qwen3_5 import Qwen3_5ForCausalLM
from infer_opt.inductor_choices import wide_persistent_reductions


def save(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--kernel', choices=['shared', 'tiled', 'grouped', 'launch', 'packed'], default='packed')
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--capacity', type=int, default=256)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    configs = [tuple(map(int, value.split(','))) for value in args.config.split(':')]
    registry = {'shared': kernels.SHARED_ROUTE_CONFIGS, 'tiled': kernels.TILED_ROUTE_CONFIGS,
                'grouped': kernels.GROUPED_DOT_CONFIGS, 'launch': kernels.LAUNCH_CONFIGS,
                'packed': kernels.PACKED_PAIRS_CONFIGS}[args.kernel]
    save(out/'progress.json', {'stage':'load_model'})
    model = Qwen3_5ForCausalLM.from_quantized_dir(args.weights, device='cuda', kernels='fla-delta')
    # Target body only: release the MTP module on A100 40GB.
    model.mtp = None
    gc.collect(); torch.cuda.empty_cache()
    fused = model.fuse_projections()
    rows = []
    for prompt, item in json.loads(Path(args.inputs).read_text()).items():
        for t in [2, 4]:
            save(out/'progress.json', {'stage':'compile_and_capture','prompt':prompt,'tokens':t})
            registry.clear()
            cache = model.new_cache(1, args.capacity); cache.allocate_spec_buffers(t)
            prefix = torch.tensor([item['prefix_ids']], device='cuda')
            model.model.language_model(prefix, cache)
            tokens = torch.tensor([item['verify_ids'][:t]], device='cuda')
            live = [x for group in [cache.keys,cache.values,cache.conv_states,cache.recurrent_states,cache.spec_conv,cache.spec_states]
                    for x in group if x is not None] + [cache.position]
            saved = [x.clone() for x in live]
            def restore():
                for dst,src in zip(live,saved):dst.copy_(src)
            def verify(tokens):return model.model.language_model.forward_verify(tokens,cache)
            fn = torch.compile(verify,fullgraph=True,dynamic=False,
                options={'max_autotune':True,'coordinate_descent_tuning':True,'triton.cudagraphs':False})
            def capture(config):
                registry.clear()
                if config is not None:registry[t]=config
                for _ in range(3):
                    restore()
                    with wide_persistent_reductions():value=fn(tokens)
                restore();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    with wide_persistent_reductions():value=fn(tokens)
                restore();graph.replay();torch.cuda.synchronize()
                return graph,value
            base,base_output=capture(None)
            expected=base_output.clone();expected_state=[x.clone() for x in live]
            expected_logits=model.lm_head(expected).float()
            for config in configs:
                candidate,candidate_output=capture(config)
                comparison={'hidden_exact':bool(torch.equal(candidate_output,expected)),
                    'hidden_max_abs_diff':(candidate_output.float()-expected.float()).abs().max().item(),
                    'state_tensors':len(live),'state_exact_tensors':sum(bool(torch.equal(a,b)) for a,b in zip(live,expected_state)),
                    'state_max_abs_diff':max((a.float()-b.float()).abs().max().item() for a,b in zip(live,expected_state))}
                logits=model.lm_head(candidate_output).float()
                p_ref=expected_logits.log_softmax(-1);p_got=logits.log_softmax(-1)
                comparison.update(target_ids_equal=bool(torch.equal(logits.argmax(-1),expected_logits.argmax(-1))),
                    target_ids_original=expected_logits.argmax(-1).tolist(),target_ids_candidate=logits.argmax(-1).tolist(),
                    mean_kl=((p_ref.exp()*(p_ref-p_got)).sum(-1)).mean().item(),
                    max_abs_logit_diff=(logits-expected_logits).abs().max().item())
                timings={'original':[],'candidate':[]}
                flush=torch.empty(128*2**20,dtype=torch.uint8,device='cuda')
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for i in range(args.repeats):
                    pairs=[('original',base),('candidate',candidate)]
                    if i%2:pairs.reverse()
                    for name,graph in pairs:
                        restore();flush.zero_();torch.cuda.synchronize()
                        a.record();graph.replay();b.record();b.synchronize()
                        timings[name].append(a.elapsed_time(b))
                medians={key:statistics.median(v) for key,v in timings.items()}
                row={'prompt':prompt,'tokens':t,'config':config,'comparison':comparison,
                    'median_gpu_ms':medians,'speedup':medians['original']/medians['candidate'],'samples_ms':timings}
                rows.append(row);save(out/'summary.json',{'rows':rows,'args':vars(args),'gpu':torch.cuda.get_device_name(),
                    'torch':str(torch.__version__),'fused_groups':fused,'scope':'Same fullgraph with automatic CUDA graphs disabled, captured twice; only custom expert kernel changes. Fixed-state target body, not MTP generation.',
                    'kernel_sha256':hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest()})
                print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
                del candidate,candidate_output,logits,p_ref,p_got,flush
            del base,base_output,expected,expected_logits,expected_state,saved,live,cache,fn
            gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
    save(out/'progress.json',{'stage':'complete'})


if __name__=='__main__':main()
