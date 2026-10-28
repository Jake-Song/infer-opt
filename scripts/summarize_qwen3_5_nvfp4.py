"""Summarize complete or partial G4 results without treating missing gates as passes."""
import argparse
import json
from pathlib import Path


def summarize(root):
    def read(path,default):
        file=root/path
        return json.loads(file.read_text()) if file.exists() else default
    accuracy=read('accuracy.json',{})
    gates=[g for g in accuracy.get('gate',[]) if g.get('config','').endswith(':nvfp4')]
    passed=bool(gates) and all(g.get('pass') and g.get('complete') for g in gates)
    accuracy_status='not_evaluated' if not gates else ('passed' if passed else 'failed_or_incomplete')
    smoke=read('smoke/smoke.json',[])
    export=read('export/export_validation.json',{})
    lifecycle=read('lifecycle.json',read('done.json',{}))
    lines=['# Qwen3.5-27B NVFP4 G4 실험','',
           f"- 실행 상태: {lifecycle.get('status','진행 중 또는 미완료')}",
           f"- 네이티브 커널 검증: {len(smoke)}개 shape 통과",
           f"- 저장·재로딩 logits 일치: {export.get('reload_logits_identical','미검증')}",
           f"- NVFP4 정확도 gate: {'통과' if passed else '실패 또는 미완료'}",'']
    if export.get('weight_bytes'):
        lines.append(f"가중치 파일 크기: {export['weight_bytes']/2**30:.3f} GiB. 캐시·실행 버퍼는 별도다.")
    lines+=['','| 정밀도 | 프롬프트 | 방식 | ms/token 중앙값 | 최소–최대 | tok/s | 토큰/round |',
            '|---|---|---|---:|---:|---:|---:|']
    results={}
    for precision in ['bf16','int8','nvfp4']:
        rows=read(f'{precision}/measurements.json',{}).get('rows',[])
        results[precision]=rows
        for r in rows:
            lines.append(f"| {precision} | {r['prompt']} | {r['method']} | {r['median_ms']:.3f} | "
                         f"{r['min_ms']:.3f}–{r['max_ms']:.3f} | {r['tokens_per_second']:.1f} | "
                         f"{r.get('tokens_per_round','—')} |")
    mtp=[r for r in results['nvfp4'] if r['method']!='plain']
    strong=sum(r.get('same_prefix_verification',{}).get('strong_mismatches',0) for r in mtp)
    lines+=['',f'NVFP4 MTP same-prefix 진단의 strong mismatch 합: {strong}. 측정 완료 케이스: {len(mtp)}/15.']
    lines+=['','추정 속도와 실측을 혼용하지 않는다. 정확도 gate와 MTP 검증이 완료되기 전에는 품질을 유지한 개선으로 판정하지 않는다.']
    if lifecycle.get('error'): lines+=['',f"미완료 원인: `{lifecycle['error']}`"]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    status={'accuracy_passed':passed,'accuracy_status':accuracy_status,
            'mtp_strong_mismatches':strong,'mtp_cases':len(mtp),
            'smoke_shapes':len(smoke),'export':export,
            'measured_precisions':[p for p,rows in results.items() if rows],
            'published_repo':lifecycle.get('published_repo'),'published_commit':lifecycle.get('published_commit')}
    (root/'result_status.json').write_text(json.dumps(status,indent=2))
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(summarize(parser.parse_args().root),indent=2))
