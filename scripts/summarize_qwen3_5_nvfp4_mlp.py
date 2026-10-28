"""Summarize the accuracy-only MLP scope ablation without conflating references."""
import json
from pathlib import Path


def summarize(root):
    root = Path(root)

    def read(name):
        file = root / name
        return json.loads(file.read_text()) if file.exists() else {}

    accuracy = read('accuracy.json')
    life = read('lifecycle.json') or read('done.json')
    scope = read('scope_validation.json')
    export = read('export/export_validation.json')
    gates = {g['config'].rsplit(':', 1)[1]: g for g in accuracy.get('gate', [])}
    wanted = {'short', 'hello', 'korean', 'math', 'chat', 'chat_long', 'prose_1k', 'code_1k', 'code_1_5k'}
    rows_by_quant = {q: [r for r in accuracy.get('rows', []) if r.get('quant') == q and not r.get('summary')]
                     for q in ('bf16', 'nvfp4', 'nvfp4-mlp')}
    complete = (life.get('status') == 'finished' and accuracy.get('reference_protocol') == 'hf-cached-greedy-v2'
                and all(len(rows) == 9 and {r['prompt'] for r in rows} == wanted for rows in rows_by_quant.values())
                and set(gates) == {'nvfp4', 'nvfp4-mlp'}
                and all(g.get('complete') and set(g.get('quantized', {}).get('prompt_names', [])) == wanted
                        and set(g.get('bf16', {}).get('prompt_names', [])) == wanted for g in gates.values())
                and scope.get('native_mlp_linears') == 128 and scope.get('native_attention_linears') == 0
                and scope.get('bf16_attention_linears') == 128
                and scope.get('calibration_token_hashes_identical')
                and export.get('reload_logits_identical'))
    passed = bool(complete and gates['nvfp4-mlp'].get('pass'))
    result = {'experiment_complete': bool(complete), 'mlp_accuracy_passed': passed,
              'gates': gates, 'scope_validation': scope, 'export': export}
    lines = ['# Qwen3.5-27B MLP-only NVFP4 accuracy ablation', '',
             f"- Execution: {life.get('status', 'incomplete')}",
             f'- Evidence complete: {bool(complete)}', f'- MLP-only accuracy gate passed: {passed}',
             '- HF reference: cached prompt decode and cached greedy generation; margins from the same generation logits.',
             '- Scope: MLP gate/up/down W4A4; all attention projections BF16; MTP INT8, lm_head BF16.',
             '- Calibration: same 128 UltraChat token sequences as the original full-NVFP4 run.', '',
             '| Config | Prompt decode KL vs HF | KL / BF16 | Strong / near-tie | Free-run identical |',
             '|---|---:|---:|---:|---:|']
    for quant in ('bf16', 'nvfp4', 'nvfp4-mlp'):
        rows = [r for r in accuracy.get('rows', []) if r.get('quant') == quant and not r.get('summary')]
        if not rows:
            continue
        kls = [r['decode_kl_vs_hf'] for r in rows if 'decode_kl_vs_hf' in r]
        ratio = gates.get(quant, {}).get('kl_ratio', 1.0)
        lines.append(f"| {quant} | {sum(kls)/len(kls):.8f} | {ratio:.3f} | "
                     f"{sum(r['tf_mismatch_strong'] for r in rows)} / {sum(r['tf_mismatch_near_tie'] for r in rows)} | "
                     f"{sum(r['gen_first_mismatch'] is None for r in rows)}/{len(rows)} |")
    if export.get('weight_bytes'):
        lines += ['', f"MLP-only weight files: {export['weight_bytes']/2**30:.3f} GiB (runtime buffers excluded)."]
    lines += ['', 'This run changes both the reference protocol and the quantization scope. Compare the full and MLP-only checkpoints within this run to attribute scope effects; older KL values use a different HF reference.',
              'No MTP-head-only accuracy test or throughput benchmark is part of this accuracy ablation.']
    if life.get('error'):
        lines += ['', 'Execution error: ' + life['error']]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    (root / 'result_status.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
