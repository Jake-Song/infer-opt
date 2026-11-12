"""Offline bounds for the recorded G4 run. No new GPU measurements or weight download.

Keep the conservative cache-capacity bound separate from idealized streaming
traffic scenarios. Neither scenario is an observed DRAM-byte measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

VERSION = 'g4-w8a16-bound-v1'
BW = 1597e9  # Server Edition GDDR7, decimal bytes/s, NVIDIA product specification.
FP32 = 120e12  # Optimistic published FMA-equivalent peak, not an achieved roof.
CACHE_BUDGET = 256 * 2**20  # Generous combined on-chip data budget; not measured L2.
SPEC_URL = 'https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/'
ARCH_URL = 'https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf'


def read(path):
    return json.loads(path.read_text())


def calculate(root):
    inputs, raw = root / 'derived/inputs', root / 'results'
    config = read(inputs / 'config.json')['text_config']
    quant = read(inputs / 'quantization.json')
    tensors = read(inputs / 'tensor-metadata.json')
    expected = {r['file']: r['sha256'] for r in read(raw / 'checkpoint-hashes.json')}
    for name in ['config.json', 'quantization.json']:
        assert hashlib.sha256((inputs / name).read_bytes()).hexdigest() == expected[name]
    h, v, top = config['hidden_size'], config['vocab_size'], config['num_experts_per_tok']
    head = 2 * h * v
    groups = {p: dict(dense_params=0, dense_bytes=0, routed_params=0, routed_bytes=0)
              for p in ['target', 'mtp']}
    for name, module in quant['modules'].items():
        p = 'mtp' if name.startswith('mtp.') else 'target'
        if module.get('type') == 'int8_experts':
            count = 3 * module['hidden_size'] * module['intermediate_size'] * top
            kind = 'routed'
        else:
            count = module['in_features'] * module['out_features']
            kind = 'dense'
        assert count % module['group_size'] == 0
        groups[p][kind + '_params'] += count
        groups[p][kind + '_bytes'] += count + 2 * count // module['group_size']
    for p in groups:
        prefix = 'mtp.' if p == 'mtp' else 'model.language_model.'
        other = sum(t['bytes'] for t in tensors if t['name'].startswith(prefix)
                    and t['dtype'] == 'torch.bfloat16' and 'scale' not in t['name']
                    and 'embed_tokens' not in t['name'])
        groups[p]['other_bf16_bytes'] = other
    state = (config['layer_types'].count('linear_attention') *
             config['linear_num_value_heads'] * config['linear_key_head_dim'] *
             config['linear_value_head_dim'] * 4)
    conv = (config['layer_types'].count('linear_attention') *
            (2 * config['linear_num_key_heads'] * config['linear_key_head_dim'] +
             config['linear_num_value_heads'] * config['linear_value_head_dim']) *
            (config['linear_conv_kernel_dim'] - 1) * 2)
    target, mtp = groups['target'], groups['mtp']
    summary = read(raw / 'benchmark/summary.json')
    rows = []
    kv_token = 2 * config['num_key_value_heads'] * config['head_dim'] * 2
    full_layers = config['layer_types'].count('full_attention')
    for measured in summary['rows']:
        method, prompt = measured['method'], measured['prompt']
        k = 0 if method == 'plain' else int(method.removeprefix('mtp'))
        n = k + 1
        samples = read(raw / f'benchmark/{method}-{prompt}.json')['samples']
        # Cache bound uses ONLY INT8 matrices/scales and BF16 vocabulary matrix.
        # At most CACHE_BUDGET bytes can be carried into each target/draft phase.
        # Each phase consumes the full vocabulary matrix in this fixed call schedule.
        target_unique = target['dense_bytes'] + target['routed_bytes'] + head
        draft_unique = mtp['dense_bytes'] + mtp['routed_bytes'] + head
        cache_bound_bytes = max(0, target_unique - CACHE_BUDGET) + k * max(0, draft_unique - CACHE_BUDGET)
        scenarios = {}
        for overlap in ['all_rows_same_experts', 'rows_disjoint_experts']:
            target_copies = 1 if overlap == 'all_rows_same_experts' else n
            # Initial MTP call has n rows, followed by k-1 single-row calls.
            draft_copies = k if overlap == 'all_rows_same_experts' else 2 * k
            weight_bytes = {
                'target_dense_int8_and_scales': target['dense_bytes'],
                'target_routed_int8_and_scales': target['routed_bytes'] * target_copies,
                'target_other_bf16': target['other_bf16_bytes'],
                'target_lm_head_bf16': head,
                'mtp_dense_int8_and_scales': k * mtp['dense_bytes'],
                'mtp_routed_int8_and_scales': draft_copies * mtp['routed_bytes'],
                'mtp_other_bf16': k * mtp['other_bf16_bytes'],
                'draft_lm_head_bf16': k * head,
                'embedding_rows': (n + 2 * k) * h * 2,
            }
            # Logical state accesses, not mandatory external-memory transfers:
            # plain: initial read + final write. Verify: read + n snapshots +
            # commit's selected-state read + base-state write.
            state_bytes = 2 * state if not k else (n + 3) * state
            # Context center approximates the 256-output sequence; round positions
            # were not recorded, so this is explicitly a scenario, not exact traffic.
            center = measured['prompt_tokens'] + 128
            loaded_context = min(2048, ((center + n + 63) // 64) * 64)
            kv_read = kv_token * loaded_context * (full_layers + k)
            kv_write = kv_token * (full_layers * n + 2 * k)
            total = sum(weight_bytes.values()) + state_bytes + 2 * conv + kv_read + kv_write
            scenarios[overlap] = dict(weight_bytes=weight_bytes,
                recurrent_state_logical_bytes=state_bytes,
                conv_state_reference_bytes=2 * conv, kv_read_reference_bytes=kv_read,
                kv_write_bytes=kv_write, reference_context=center,
                loaded_context=loaded_context, total_logical_bytes=total,
                ideal_streaming_round_ms=total / BW * 1000)
        def amortize(round_ms):
            if not k:
                return round_ms
            # Same normalization as each raw timed sample, including final overshoot.
            return statistics.median(round_ms * s['rounds'] / (s['emitted_tokens'] - 1)
                                     for s in samples)
        for values in scenarios.values():
            values['ideal_streaming_ms_per_token'] = amortize(values['ideal_streaming_round_ms'])
        # Actual expert kernel has two FP32 multiplies plus an add per product,
        # with FMA contraction disabled. 120 TFLOP/s FMA peak = 60T scalar ops/s.
        # Reduction, conversion, sigmoid, routing, and address instructions omitted.
        expert_scalar_ops = 3 * (n * target['routed_params'] + 2 * k * mtp['routed_params'])
        expert_compute_round_ms = expert_scalar_ops / (FP32 / 2) * 1000
        # A separate, deliberately weak math-only lower bound uses a 1 PFLOP/s
        # optimistic upper roof for all useful linear algebra. Not a BF16 dense claim.
        useful_matmul_flops = 2 * (n * (target['dense_params'] + target['routed_params']) +
            2 * k * (mtp['dense_params'] + mtp['routed_params']) + (n + k) * h * v)
        rows.append(dict(method=method, prompt=prompt, rows_per_verify=n,
            emitted_per_round=(measured.get('tokens_per_round', 1)),
            measured_ms_per_token=measured['median_ms'], exact_repeats=5-measured['divergent_runs'],
            conservative_weight_cache_bound_bytes_per_round=cache_bound_bytes,
            conservative_weight_cache_bound_ms_per_token=amortize(cache_bound_bytes / BW * 1000),
            expert_scalar_ops_per_round=expert_scalar_ops,
            expert_fp32_arithmetic_floor_ms_per_round=expert_compute_round_ms,
            expert_fp32_arithmetic_floor_ms_per_token=amortize(expert_compute_round_ms),
            useful_linear_flops_per_round=useful_matmul_flops,
            optimistic_1pflops_linear_math_floor_ms_per_token=amortize(useful_matmul_flops / 1e15 * 1000),
            streaming_scenarios=scenarios))
    dependencies = [inputs / n for n in ['config.json', 'quantization.json', 'tensor-metadata.json']]
    dependencies += [raw / 'benchmark/summary.json', raw / 'checkpoint-hashes.json',
                     raw / 'environment.json', Path(__file__)]
    dependencies += sorted((raw / 'benchmark').glob('plain-*.json'))
    dependencies += sorted((raw / 'benchmark').glob('mtp*-*.json'))
    dependencies += [Path(name) for name in ['infer_opt/speculative.py', 'infer_opt/qwen3_5.py',
        'infer_opt/moe.py', 'infer_opt/moe_kernels.py', 'infer_opt/quant.py',
        'infer_opt/quant_kernels.py', 'infer_opt/attention_kernels.py',
        'infer_opt/gated_delta_kernels.py']]
    return dict(calculator_version=VERSION, hardware=dict(bandwidth_bytes_s=BW,
        fp32_fma_peak_flops_s=FP32, combined_onchip_budget_bytes=CACHE_BUDGET,
        budget_status='generous assumed upper budget, not runtime measurement',
        specification_source=SPEC_URL, architecture_source=ARCH_URL),
        groups=groups, bf16_vocabulary_bytes_per_call=head,
        tensor_metadata_origin='Meta-device model from pinned config, cast to BF16, then prepare_quantized; no weights loaded.',
        meta_tensor_payload_bytes=sum(t['bytes'] for t in tensors),
        checkpoint_shards_with_headers_bytes=sum(r['bytes'] for r in read(raw / 'checkpoint-hashes.json')
            if r['file'].endswith('.safetensors')),
        recurrent_state_bytes=state, conv_state_bytes=conv,
        limitations=['No new hardware run, NCU counters, cache-hit or route-overlap measurements.',
            'Cache-capacity bound assumes fixed current call schedule and unchanged weight storage/dtypes.',
            '256 MiB budget exceeds full GB202 L2 + all L1/shared + register-file data capacity in the cited architecture; server cache capacity was not queried.',
            'Streaming scenarios assume full nominal bandwidth, no transfer amplification, and logical state/KV accesses serviced from GDDR7; they are not strict bounds under arbitrary caching.',
            'Expert overlap scenarios are hypothetical. Routing IDs were not retained in the run.',
            'Expert scalar floor is a partial arithmetic floor, not complete kernel instruction modeling.',
            'Timing/floor ratios are remaining idealized headroom, not validated achievable speedups.',
            'No HF BF16/eager quality equivalence is inferred.'],
        inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_root', type=Path)
    root = parser.parse_args().artifact_root
    result = calculate(root)
    target = root / 'derived/lower-bound.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    for r in result['rows']:
        scenarios = r['streaming_scenarios']
        print(r['prompt'], r['method'], 'observed', round(r['measured_ms_per_token'], 4),
            'cache-bound', round(r['conservative_weight_cache_bound_ms_per_token'], 4),
            'streaming', [round(s['ideal_streaming_ms_per_token'], 4) for s in scenarios.values()],
            'expert-compute-floor', round(r['expert_fp32_arithmetic_floor_ms_per_token'], 4))


if __name__ == '__main__':
    main()
