"""Summarize G4 evidence, distinguishing completed negative results from missing work."""
import argparse
import json
from pathlib import Path

PROMPTS = ('chat_long', 'code_1k', 'prose_1k', 'math', 'korean')
ACCURACY_PROMPTS = ('short', 'hello', 'korean', 'math', 'chat', 'chat_long',
                    'prose_1k', 'code_1k', 'code_1_5k')


def summarize(root):
    root = Path(root)

    def read(*paths, default=None):
        for path in paths:
            file = root / path
            if file.exists():
                return json.loads(file.read_text())
        return default

    accuracy = read('accuracy.json', default={})
    gates = [g for g in accuracy.get('gate', []) if g.get('config', '').endswith(':nvfp4')]
    wanted_prompts = set((accuracy.get('args', {}).get('prompts') or ','.join(ACCURACY_PROMPTS)).split(','))
    accuracy_complete = bool(gates) and all(
        g.get('complete') and set(g.get('quantized', {}).get('prompt_names', [])) == wanted_prompts
        and set(g.get('bf16', {}).get('prompt_names', [])) == wanted_prompts for g in gates)
    passed = accuracy_complete and all(g.get('pass') for g in gates)
    accuracy_status = ('not_evaluated' if not gates else 'incomplete' if not accuracy_complete
                       else 'passed' if passed else 'failed')
    all_gates = accuracy.get('gate', [])
    comparison_accuracy_complete = accuracy_complete and any(
        g.get('config', '').endswith(':int8') and g.get('complete')
        and set(g.get('quantized', {}).get('prompt_names', [])) == wanted_prompts
        and set(g.get('bf16', {}).get('prompt_names', [])) == wanted_prompts
        for g in all_gates)
    smoke = read('smoke/smoke.json', 'native_smoke.json', default=[])
    export = read('export/export_validation.json', 'export_validation.json', default={})
    lifecycle = read('lifecycle.json', 'evaluation_lifecycle.json', 'done.json', default={})
    profile = read('linear_profile.json', default={})
    results = {p: read(f'{p}/measurements.json', f'{p}_measurements.json', default={})
               for p in ('bf16', 'int8', 'nvfp4')}
    coverage = {}
    for precision, result in results.items():
        args = result.get('args', {})
        ks = args.get('ks', [] if precision == 'bf16' else [1, 2, 3])
        expected = {(p, m) for p in PROMPTS for m in ['plain', *(f'mtp{k}' for k in ks)]}
        rows = result.get('rows', [])
        observed = {(r['prompt'], r['method']) for r in rows}
        repeats = args.get('repeats', 5)
        complete = (observed == expected and len(rows) == len(expected)
                    and all(len(r.get('samples_ms', [])) == repeats for r in rows))
        prefill = result.get('prefill', [])
        prefill_complete = (len(prefill) == 3 and {r['tokens'] for r in prefill} == {512, 1024, 4096}
                            and all(len(r.get('samples_ms', [])) == repeats for r in prefill))
        coverage[precision] = {'decode_cases': len(rows), 'expected_decode_cases': len(expected),
                               'decode_complete': complete, 'prefill_cases': len(prefill),
                               'prefill_complete': prefill_complete}
    mtp = [r for r in results['nvfp4'].get('rows', []) if r['method'] != 'plain']
    strong = sum(r.get('same_prefix_verification', {}).get('strong_mismatches', 0) for r in mtp)
    divergent = sum(r.get('token_agreement') != 1.0 for r in mtp)
    mtp_expected = coverage['nvfp4']['expected_decode_cases'] - len(PROMPTS)
    mtp_complete = coverage['nvfp4']['decode_complete'] and mtp_expected > 0
    mtp_equivalent = mtp_complete and divergent == 0
    profile_complete = ({r.get('activation_rows') for r in profile.get('rows', [])} == {1, 4}
                        and all(r.get('linear_count') == 256 for r in profile.get('rows', [])))
    complete = (lifecycle.get('status') == 'finished' and comparison_accuracy_complete and profile_complete
                and all(c['decode_complete'] and c['prefill_complete'] for c in coverage.values()))
    status = {'experiment_complete': complete, 'accuracy_passed': passed,
              'accuracy_complete': accuracy_complete, 'accuracy_status': accuracy_status,
              'comparison_accuracy_complete': comparison_accuracy_complete,
              'mtp_strong_mismatches': strong, 'mtp_cases': len(mtp), 'mtp_expected_cases': mtp_expected,
              'mtp_divergent_cases': divergent, 'mtp_greedy_equivalent': mtp_equivalent,
              'coverage': coverage, 'profile_complete': profile_complete,
              'smoke_shapes': len(smoke), 'export': export,
              'measured_precisions': [p for p, r in results.items() if r.get('rows')],
              'checkpoint_repo': lifecycle.get('checkpoint_repo', lifecycle.get('published_repo')),
              'checkpoint_revision': lifecycle.get('checkpoint_revision', lifecycle.get('published_commit'))}
    labels = {'not_evaluated': '미실행', 'incomplete': '미완료', 'passed': '통과', 'failed': '실패 (검증 완료)'}
    lines = ['# Qwen3.5-27B NVFP4 G4 실험', '',
             f"- 실행 상태: {lifecycle.get('status', '미확인')}",
             f"- 비교·검증 작업: {'완료' if complete else '미완료'}",
             f"- NVFP4 정확도 gate: {labels[accuracy_status]}",
             f"- 네이티브 커널 검증: {str(len(smoke)) + '개 shape 통과' if smoke else '이 실행에 증거 없음'}",
             f"- 저장·재로딩 logits 일치: {export.get('reload_logits_identical', '이 실행에 증거 없음')}"]
    if export.get('weight_bytes'):
        lines.append(f"- 가중치 파일 크기: {export['weight_bytes'] / 2**30:.3f} GiB (캐시·실행 버퍼 별도)")
    lines += ['', '## 측정 범위', '',
              'Decode는 prefill과 첫 토큰을 제외한다. MTP는 마지막 round에서 초과 생성한 토큰도 시간 분모에 포함한다.',
              '각 행은 warmup 후 반복 측정의 중앙값이다. 실행 인자·프롬프트 길이는 원본 JSON에 보존한다.', '',
              '| 정밀도 | decode 완료/예정 | prefill 완료/예정 |', '|---|---:|---:|']
    for p, c in coverage.items():
        lines.append(f"| {p} | {c['decode_cases']}/{c['expected_decode_cases']} | {c['prefill_cases']}/3 |")
    lines += ['', '## Decode', '',
              '| 정밀도 | 프롬프트 | 방식 | ms/token 중앙값 | 최소–최대 | tok/s | 토큰/round |',
              '|---|---|---|---:|---:|---:|---:|']
    for precision, result in results.items():
        for r in result.get('rows', []):
            tpr = f"{r['tokens_per_round']:.3f}" if 'tokens_per_round' in r else '—'
            lines.append(f"| {precision} | {r['prompt']} | {r['method']} | {r['median_ms']:.3f} | "
                         f"{r['min_ms']:.3f}–{r['max_ms']:.3f} | {r['tokens_per_second']:.1f} | {tpr} |")
    lines += ['', '## Prefill', '',
              'Eager model forward와 lm_head를 포함하며, cache 할당과 tokenization은 제외한다. 서비스 TTFT 측정은 아니다.', '',
              '| 정밀도 | 입력 토큰 | 중앙값 (ms) |', '|---|---:|---:|']
    for precision, result in results.items():
        for r in result.get('prefill', []):
            lines.append(f"| {precision} | {r['tokens']} | {r['median_ms']:.3f} |")
    lines += ['', '## 정확도', '',
              '| 구성 | 완료 | 평균 KL / BF16 | strong mismatch | 판정 |', '|---|---|---:|---:|---|']
    for g in accuracy.get('gate', []):
        ratio = g.get('kl_ratio')
        lines.append(f"| {g['config']} | {g.get('complete', False)} | "
                     f"{ratio if ratio is not None else '미측정'} | "
                     f"{g.get('quantized', {}).get('strong_mismatches', '미측정')} | {g.get('pass', False)} |")
    lines += ['', f'NVFP4 MTP: {len(mtp)}/{mtp_expected}개 측정, plain greedy와 다른 출력 {divergent}개.',
              f'Same-prefix 진단의 strong mismatch 합은 {strong}이다. k별 진단 합계이며 독립 오류 수가 아니다.',
              'Same-prefix 진단은 출력 불일치 시 최대 64개 forced position을 확인한다.', '',
              '| 프롬프트 | 방식 | 토큰 일치율 | 첫 불일치 | strong / near-tie |',
              '|---|---|---:|---:|---:|']
    for r in mtp:
        diag = r.get('same_prefix_verification', {})
        lines.append(f"| {r['prompt']} | {r['method']} | {r.get('token_agreement', '미측정')} | "
                     f"{r.get('first_divergence', '—')} | {diag.get('strong_mismatches', '—')} / "
                     f"{diag.get('near_tie_mismatches', '—')} |")
    lines += ['', '## Linear 커널 프로파일', '',
              '합성 BF16 입력으로 target Linear 256개를 모델 순서대로 실행한다. norm·attention·recurrent state·lm_head는 제외한다.',
              'CUDA event 시간과 profiler 커널 시간은 end-to-end decode 시간과 별도다.', '',
              '| activation rows | Linear 수 | CUDA event 합계 (ms) |', '|---:|---:|---:|']
    for r in profile.get('rows', []):
        lines.append(f"| {r['activation_rows']} | {r['linear_count']} | {r['aggregate_cuda_ms']:.3f} |")
    lines += ['', '## 판정', '']
    if not passed or not mtp_equivalent:
        lines.append('품질을 유지한 NVFP4+MTP 개선으로 판정하지 않는다. 실패한 정확도 검증과 출력 불일치를 속도 수치와 함께 해석해야 한다.')
    else:
        lines.append('지정된 프롬프트에서 정확도 gate와 MTP 출력 일치를 통과했다. 일반적인 모델 품질 보장은 아니다.')
    if lifecycle.get('error'):
        lines += ['', f"미완료 원인: `{lifecycle['error']}`"]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    (root / 'result_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
