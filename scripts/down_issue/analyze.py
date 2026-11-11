"""Validate recovered down timings and summarize NCU metric tables."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics


def tables(path):
    text=path.read_text()
    lines=text.splitlines()
    start=next((i for i,line in enumerate(lines) if line.startswith('"ID",')),None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    args=p.parse_args()
    root=args.results
    timing=json.loads((root/'timing/timings.json').read_text())
    assert len(timing['rows'])==120
    grouped=collections.defaultdict(list)
    exact=collections.defaultdict(list)
    for row in timing['rows']:
        assert len(row['samples_us'])==100
        assert row['validation']['finite']
        assert statistics.median(row['samples_us'])==row['median_us']
        assert len(row['validation']['ids'])==row['tokens']
        assert all(len(set(ids))==8 for ids in row['validation']['ids'])
        key=(row['tokens'],row['routing'],row['variant'],row['cache'])
        grouped[key].append(row)
        exact[row['variant']].append(row['validation']['reference_exact'])
    summary=[]
    for key,rows in sorted(grouped.items()):
        assert sorted(r['batch'] for r in rows)==[0,1,2]
        samples=sorted(s for r in rows for s in r['samples_us'])
        summary.append(dict(zip(['tokens','routing','variant','cache'],key))|{
            'median_of_batch_medians_us':statistics.median(r['median_us'] for r in rows),
            'batch_medians_us':[r['median_us'] for r in sorted(rows,key=lambda r:r['batch'])],
            'p10_us':samples[int(.1*(len(samples)-1))],'p90_us':samples[int(.9*(len(samples)-1))],
            'reference_exact':all(r['validation']['reference_exact'] for r in rows)})
    profiles={}
    entries=[]
    for filename in ['profiles.json','focused-profiles.json']:
        if (root/filename).exists(): entries+=json.loads((root/filename).read_text())
    for entry in entries:
        name=entry['name']
        assert entry['returncode']==0,name
        details=tables(root/(name+'.stdout'))
        metrics=[{k:row.get(k) for k in ['Section Name','Metric Name','Metric Unit','Metric Value']}
                 for row in details if row.get('Metric Name')]
        raw_path=root/(name+'.raw.csv')
        if (root/(name+'.raw-base.csv')).exists():raw_path=root/(name+'.raw-base.csv')
        raw=tables(raw_path) if raw_path.exists() else []
        data=[row for row in raw if row.get('ID','').isdigit()]
        assert len(data)==1,(name,len(data))
        units=next((row for row in raw if not row.get('ID','').isdigit()),{})
        if not metrics:
            metrics=[{'Section Name':'raw','Metric Name':k,'Metric Unit':units.get(k),'Metric Value':v}
                     for k,v in data[0].items() if '__' in k]
        kernels={row.get('Kernel Name') for row in data}
        assert len(kernels)==1 and all('_project' in k for k in kernels),kernels
        match=re.search(r'-t(\d+)-(shared|partial|disjoint)-(scalar1|scalar4|packed1|packed4)-(hot|cold)$',name)
        assert match,name
        tokens,routing,variant,cache=match.groups();tokens=int(tokens)
        validation=json.loads((root/name/'profile-validation.json').read_text())
        reference_row=grouped[tokens,routing,variant,cache][0]
        assert validation['output_sha256']==reference_row['validation']['output_sha256'],name
        assert validation['reference_exact']==reference_row['validation']['reference_exact'],name
        grid=[int(v) for v in re.findall(r'\d+',data[0]['Grid Size'])]
        block=[int(v) for v in re.findall(r'\d+',data[0]['Block Size'])]
        assert grid==[tokens*8,2048,1],(name,grid)
        assert block==[32*int(variant[-1]),1,1],(name,block)
        profiles[name]={'seconds':entry['seconds'],'kernels':sorted(kernels),'metrics':metrics,'raw':raw,
                        'validation':validation,'profile_matches_timing_hash':True,
                        'report_sha256':hashlib.sha256((root/(name+'.ncu-rep')).read_bytes()).hexdigest()}
    result={'timing_samples':sum(len(r['samples_us']) for r in timing['rows']),
            'conditions':len(grouped),'reference_exact_by_variant':{k:all(v) for k,v in exact.items()},
            'timings':summary,'profiles':profiles,'source_sha256':timing['kernel_sha256']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2))
    with (root/'timing-summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps({k:v for k,v in result.items() if k not in ['profiles','timings']},indent=2))
    for name,profile in profiles.items():
        print('\n'+name)
        for metric in profile['metrics']:
            if metric['Metric Name'] in ['smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','gpu__time_duration.sum','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__inst_executed.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed']:
                print(json.dumps(metric))


if __name__=='__main__':main()
