"""Optimistic W8A16 prefill roofline from retained config/tensor metadata; no GPU needed."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT/'artifacts/qwen35_moe_compiled_g4_20261112/derived/inputs'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--config', type=Path, default=INPUTS/'config.json')
    p.add_argument('--metadata', type=Path, default=INPUTS/'tensor-metadata.json')
    p.add_argument('--bandwidth-gbps', type=float, default=1597)
    # NVIDIA workstation dense BF16 rate, scaled by recorded server max boost.
    p.add_argument('--dense-tflops', type=float, default=503.8*2430/2617)
    args = p.parse_args()
    if min(args.bandwidth_gbps, args.dense_tflops) <= 0:
        p.error('peak rates must be positive')
    cfg = json.loads(args.config.read_text())['text_config']
    metadata = json.loads(args.metadata.read_text())
    h, m = cfg['hidden_size'], cfg['moe_intermediate_size']
    layers, experts, top = cfg['num_hidden_layers'], cfg['num_experts'], cfg['num_experts_per_tok']
    n_linear = cfg['layer_types'].count('linear_attention')
    n_full = cfg['layer_types'].count('full_attention')
    key = cfg['linear_num_key_heads']*cfg['linear_key_head_dim']
    value = cfg['linear_num_value_heads']*cfg['linear_value_head_dim']
    q = cfg['num_attention_heads']*cfg['head_dim']
    kv = cfg['num_key_value_heads']*cfg['head_dim']
    params = {
        'routed_experts_active': layers*3*h*m*top,
        'shared_expert': layers*3*h*cfg['shared_expert_intermediate_size'],
        'linear_attention_projections': n_linear*(h*(2*key+2*value)+value*h+2*h*cfg['linear_num_value_heads']),
        'full_attention_projections': n_full*(h*(2*q+2*kv)+q*h),
        'routers_and_shared_gates': layers*h*(experts+1),
    }
    # Independently check the projection operation count against stored tensor shapes.
    meta_active = 0
    byte_groups = defaultdict(int)
    for row in metadata:
        name, shape = row['name'], row['shape']
        if name.startswith('mtp.'):
            continue
        if 'embed_tokens' in name:
            continue  # Only selected embedding rows are touched; not a full table scan.
        if name.startswith('lm_head.'):
            group = 'lm_head'
        elif '.mlp.experts.' in name:
            group = 'routed_experts'
            if name.endswith(('.gate_up_proj', '.down_proj')):
                meta_active += math.prod(shape)//experts*top
        elif '.mlp.shared_expert.' in name:
            group = 'shared_expert'
        elif '.mlp.gate.' in name or '.mlp.shared_expert_gate.' in name:
            group = 'routers'
        elif '.linear_attn.' in name:
            group = 'linear_attention'
        elif '.self_attn.' in name:
            group = 'full_attention'
        else:
            group = 'other'
        byte_groups[group] += row['bytes']
        if len(shape) == 2 and name.endswith('.weight') and group != 'lm_head':
            meta_active += math.prod(shape)
    assert meta_active == sum(params.values())
    compute = args.dense_tflops*1e12
    bandwidth = args.bandwidth_gbps*1e9
    expert_bytes = byte_groups['routed_experts']
    fixed_bytes = sum(byte_groups.values())-expert_bytes
    flop_linear_per_token = 2*sum(params.values())
    lm_flops = 2*h*cfg['vocab_size']  # Only the final prompt position is projected.
    moe_params = sum(params[k] for k in ('routed_experts_active', 'shared_expert', 'routers_and_shared_gates'))
    attn_params = params['linear_attention_projections']+params['full_attention_projections']
    model_rows = json.loads((ROOT/'artifacts/qwen35_moe_prefill_g4_20261113/results/model/model.json').read_text())['rows']
    rows = []
    for t in (128, 512, 2048, 4096):
        causal_flops = 2*n_full*q*t*(t+1)
        flops = flop_linear_per_token*t+causal_flops+lm_flops
        coverage_rows = []
        for label, unique in [('minimum_8_shared', top),
                              ('uniform_expected', experts*(1-(1-top/experts)**t)),
                              ('all_256', experts)]:
            weight_bytes = fixed_bytes+expert_bytes*unique/experts
            coverage_rows.append({'routing': label, 'unique_experts_per_layer': unique,
                'weights_read_bytes': weight_bytes, 'weights_only_memory_ms': weight_bytes/bandwidth*1000,
                'global_roofline_ms': max(flops/compute, weight_bytes/bandwidth)*1000})
        # Conditional model: full-coverage bulk layer execution, with attention and
        # MoE stages sequential and perfect overlap *inside* each coarse stage.
        # This is not a universal bound for cross-layer/chunk-pipelined algorithms.
        attn_stage = max((2*attn_params*t+causal_flops)/compute,
                         (byte_groups['linear_attention']+byte_groups['full_attention'])/bandwidth)
        moe_stage = max(2*moe_params*t/compute,
                        (expert_bytes+byte_groups['shared_expert']+byte_groups['routers'])/bandwidth)
        head_stage = max(lm_flops/compute, byte_groups['lm_head']/bandwidth)
        rows.append({'tokens': t, 'major_gemm_flops': flops,
            'major_gemm_compute_ms': flops/compute*1000, 'routing_scenarios': coverage_rows,
            'bulk_layer_stage_model_ms': (attn_stage+moe_stage+head_stage+byte_groups['other']/bandwidth)*1000,
            'sorted_measured_ms': {r['prompt']: r['sorted']['median_wall_ms'] for r in model_rows if r['tokens'] == t}})
    result = {'hardware': {'memory_bandwidth_gbps': args.bandwidth_gbps,
        'dense_bf16_tflops_estimate': args.dense_tflops,
        'dense_rate_derivation': '503.8 TFLOP/s workstation dense BF16 * server 2430 MHz / workstation 2617 MHz'},
        'active_projection_parameters': params, 'active_projection_parameter_sum': sum(params.values()),
        'major_projection_flops_per_token': flop_linear_per_token,
        'target_weight_bytes_by_group': dict(byte_groups), 'all_experts_target_weight_scan_bytes': sum(byte_groups.values()),
        'expert_block_bytes_per_layer_all_256': expert_bytes/layers,
        'assumptions': ['Batch one, existing INT8 group128 weights with BF16 activations; no W8A8/FP8/FP4 conversion.',
            'FMA counts as two FLOPs. Major GEMMs and useful causal full-attention QK/AV only.',
            'GatedDeltaNet core, convolution, normalization, softmax, nonlinearities, dequantization and routing arithmetic omitted.',
            'Weight-only GDDR traffic; activation/cache traffic, repeated tile reads and host/kernel launch costs omitted.',
            'One off-chip read per used weight/scale; ideal fused dequantization; no whole-bank persistent on-chip residency.',
            'Unique-expert coverage is hypothetical; actual routing IDs were not retained in the benchmark.',
            'All-expert weight scan excludes MTP and full embedding table, includes one BF16 lm_head scan.',
            'Bulk stage model assumes full expert coverage and sequential whole-layer attention/MoE stages, not arbitrary pipelining.',
            'The theoretical estimate does not establish achievable latency or output-preserving speedup.'],
        'sources': {'gpu': 'https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/',
            'architecture': 'https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf'},
        'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.config, args.metadata)}, 'rows': rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
