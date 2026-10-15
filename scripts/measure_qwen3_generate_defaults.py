#!/usr/bin/env python3
"""Validate Qwen3 with the native Transformers generate/static compilation defaults."""
import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

TARGET = 'Qwen/Qwen3-14B'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-memory-gb', type=int, default=40)
    parser.add_argument('--attention', choices=['auto'], default='auto')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'model': TARGET, 'revision': REVISION}
    save(root / 'execution.json', state)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig, GenerationConfig
        from torch._dynamo.utils import counters
        hardware = dict(torch=str(torch.__version__), transformers=transformers.__version__,
                        gpu=torch.cuda.get_device_name(), memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        save(root / 'hardware.json', hardware)
        if transformers.__version__ != '4.57.3' or 'A100' not in hardware['gpu']:
            raise RuntimeError('Unexpected runtime')
        config = CompileConfig()
        save(root / 'options.json', {'compile_config': dataclasses.asdict(config),
             'dtype': 'bfloat16', 'attention': 'sdpa', 'prompt_tokens': 128, 'output_tokens': 128,
             'prefill': 'native generate eager', 'decode': 'native generate automatic compile',
             'custom_torch_compile_options': None, 'atol': 0.02, 'rtol': 0.02,
             'generation_policy': 'greedy, fixed 128 outputs for reproducible comparison; not Qwen sampling recommendation'})
        model = AutoModelForCausalLM.from_pretrained(TARGET, revision=REVISION, dtype=torch.bfloat16,
                   attn_implementation='sdpa', device_map='cuda').eval().requires_grad_(False)
        save(root / 'model.json', {'model': TARGET, 'revision': REVISION,
             'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()})})
        tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=REVISION)
        text = ('한국의 GPU 추론 성능을 설명하시오. Explain how attention and KV caching work. '
                'def add(a, b): return a + b\n') * 32
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[:, :128].to('cuda')
        save(root / 'prompt.json', ids.cpu().tolist())
        outputs = {}
        with torch.inference_mode():
            for name, cache, disabled in [('eager_dynamic', 'dynamic', True),
                                          ('eager_static', 'static', True),
                                          ('compiled_static_defaults', 'static', False)]:
                save(root / 'progress.json', {'stage': name})
                generation = GenerationConfig(max_new_tokens=128, min_new_tokens=128, do_sample=False,
                    use_cache=True, cache_implementation=cache, disable_compile=disabled,
                    compile_config=CompileConfig(), pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                    return_dict_in_generate=True, output_logits=True)
                effective, _ = model._prepare_generation_config(generation, do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50)
                if effective.do_sample or effective.disable_compile != disabled:
                    raise RuntimeError('Effective generation options differ from requested options')
                save(root / (name + '-generation-config.json'), effective.to_dict())
                counters.clear()
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation, do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50)
                torch.cuda.synchronize()
                # Snapshot before the next generate resets the shared StaticCache.
                length = ids.shape[1] + len(result.logits) - 1
                snapshot = {'tokens': result.sequences.cpu(),
                    'logits': torch.stack(result.logits).cpu(),
                    'kv': [(layer.keys[..., :length, :].cpu().clone(),
                            layer.values[..., :length, :].cpu().clone()) for layer in result.past_key_values.layers]}
                save(root / (name + '-tokens.json'), snapshot['tokens'].tolist())
                outputs[name] = snapshot
                torch.save(snapshot, root / (name + '.pt'))
                save(root / (name + '-counters.json'), {k: dict(v) for k, v in counters.items() if v})
                del result
                torch.cuda.empty_cache()
        def compare(a, b):
            a, b = a.float(), b.float()
            error = (a-b).abs()
            good = torch.isfinite(a) & torch.isfinite(b) & (error <= 0.02 + 0.02*b.abs())
            return {'passed': bool(good.all()), 'mismatches': int((~good).sum()),
                    'max_abs_error': float(error.max()),
                    'relative_l2': float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))}
        comparisons = []
        for candidate, reference in [('eager_static', 'eager_dynamic'),
                                     ('compiled_static_defaults', 'eager_static'),
                                     ('compiled_static_defaults', 'eager_dynamic')]:
            a, b = outputs[candidate], outputs[reference]
            equal = torch.equal(a['tokens'], b['tokens'])
            row = {'candidate': candidate, 'reference': reference, 'tokens_equal': equal,
                   'same_input_trajectory': equal,
                   'logits': [compare(x,y) for x,y in zip(a['logits'], b['logits'])],
                   'final_kv': [{'layer': i, 'kind': kind, **compare(x,y)}
                       for i, (aa,bb) in enumerate(zip(a['kv'],b['kv']))
                       for kind,x,y in zip(('keys','values'),aa,bb)]}
            row['generation_passed'] = equal
            row['tensor_diagnostics_passed'] = all(x['passed'] for x in row['logits']+row['final_kv'])
            comparisons.append(row)
        save(root / 'comparisons.json', comparisons)
        compiled_counters = json.loads((root / 'compiled_static_defaults-counters.json').read_text())
        compiled = compiled_counters.get('stats', {}).get('unique_graphs', 0) > 0
        state.update(status='passed' if compiled and all(r['generation_passed'] for r in comparisons) else 'validation_failed',
                     compile_observed=compiled, generation_completed=True, timing_collected=False,
                     validation_policy='exact_generated_token_ids', tensor_checks='diagnostic_only')
        lines = ['# Transformers generate 기본 옵션 재실행', '',
                 f"- {hardware['gpu']}, PyTorch {hardware['torch']}, Transformers {hardware['transformers']}",
                 '- Qwen3-14B BF16 / SDPA / 입력 128, 출력 128 / 기본 CompileConfig',
                 '- generate가 prefill과 decode 분기 및 cache를 관리한다. 사용자 compile 옵션 없음.',
                 f'- compile 그래프 관측: {compiled}',
                 '- 통과 기준: 같은 입력과 greedy 조건에서 생성 token ID 전체 일치. compile 실행도 확인한다.',
                 '- logits/KV의 atol=rtol=0.02 비교는 진단 정보이며 통과 여부를 막지 않는다.',
                 '- 토큰 경로가 다르면 이후 tensor 차이는 같은 입력 비교가 아니므로 해석에 제한이 있다.', '',
                 '| 비교 | 생성 토큰 일치 | logits 실패/128 | 최종 KV 실패/전체 |',
                 '|---|---|---|---|']
        for r in comparisons:
            lines.append(f"| {r['candidate']} / {r['reference']} | {r['tokens_equal']} | {sum(not x['passed'] for x in r['logits'])}/{len(r['logits'])} | {sum(not x['passed'] for x in r['final_kv'])}/{len(r['final_kv'])} |")
        lines += ['', '정확도 재실행만 수행했다. compile 및 검증 시간을 성능 결과로 사용하지 않는다.']
        (root / 'report.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        (root / 'error.txt').write_text(traceback.format_exc())
    finally:
        save(root / 'execution.json', state)
    return 0 if state['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
