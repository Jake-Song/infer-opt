"""Summarize measured MoE results without treating divergent MTP runs as speed wins."""
import argparse,json,statistics,math
from pathlib import Path

def summarize(root):
    result={}
    for filename in ('spec-parallel-repeated.json','spec-sequential.json','spec-consistent.json','spec-compiled.json','contexts.json'):
        path=root/filename
        if not path.exists(): continue
        data=json.loads(path.read_text());groups={}
        for row in data['rows']:
            if 'error' in row: continue
            name=row['prompt'].split('/repeat-')[0]
            group=groups.setdefault(name,{'baseline':[],'spec':{},'peak_gib':[]})
            group['baseline'].append(row['baseline_ms_per_token'])
            group['peak_gib'].append(row['peak_mem_gib'])
            for k,v in row['spec'].items():
                group['spec'].setdefault(k,[]).append(v)
        output=[]
        for name,group in groups.items():
            base=statistics.median(group['baseline'])
            row={'prompt':name,'baseline_median_ms':base,'baseline_p90_ms':sorted(group['baseline'])[math.ceil(.9*len(group['baseline']))-1],
                 'repeats':len(group['baseline']),'peak_gib':max(group['peak_gib']),'spec':{}}
            for k,values in group['spec'].items():
                timings=[v['ms_per_token'] for v in values]
                passed=all(v['first_divergence'] is None for v in values)
                median=statistics.median(timings)
                row['spec'][k]={'median_ms':median,'p90_ms':sorted(timings)[math.ceil(.9*len(timings))-1],
                                'greedy_parity':passed,'valid_speedup':base/median if passed else None,
                                'divergent_runs':sum(v['first_divergence'] is not None for v in values)}
            output.append(row)
        result[filename]=output
    for filename in ('quality.json', 'quality-compiled-bf16.json', 'quality-compiled-stock.json'):
        path=root/filename
        if not path.exists(): continue
        rows=json.loads(path.read_text())['rows'];quality_key='quality' if filename=='quality.json' else filename;result[quality_key]={}
        for precision in ('bf16','int8'):
            selected=[r for r in rows if r['precision']==precision]
            result[quality_key][precision]={'mean_kl':statistics.mean(r['mean_kl'] for r in selected),
                'tokens':sum(r['teacher_forced_tokens'] for r in selected),
                'mismatches':sum(len(r['mismatch_steps']) for r in selected),
                'strong_mismatches':sum(r['strong_mismatches'] for r in selected)}
    path=root/'bf16-override-end-to-end.json'
    if path.exists():
        rows=json.loads(path.read_text())['rows']
        result[path.name]=[]
        for prompt in sorted({r['prompt'] for r in rows}):
            stock=[r for r in rows if r['prompt']==prompt and r['variant']=='stock']
            tuned=[r for r in rows if r['prompt']==prompt and r['variant']=='bf16-gemv']
            if not stock or not tuned: continue
            before=statistics.median(r['ms_per_token'] for r in stock)
            after=statistics.median(r['ms_per_token'] for r in tuned)
            result[path.name].append({'prompt':prompt,'stock_median_ms':before,'override_median_ms':after,
                                     'timing_ratio':before/after,
                                     'tokens_equal_stock':all(r['tokens_equal_stock'] for r in tuned)})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=summarize(a.root);(a.root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
