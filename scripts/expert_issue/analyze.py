"""Verify matched timings/counters and compute ideal bandwidth lower bounds."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from down_issue.analyze import tables


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();root=args.results
    timing=json.loads((root/'timing/timings.json').read_text())
    assert len(timing['rows'])==252
    assert timing['clock_lock']['returncode']==0
    grouped=collections.defaultdict(list)
    for row in timing['rows']:
        assert len(row['samples_us'])==100
        assert statistics.median(row['samples_us'])==row['median_us']
        assert row['validation']['finite']
        assert row['telemetry'].startswith('1095 MHz'),row['telemetry']
        ids=row['validation']['ids']
        assert len(ids)==row['tokens'] and all(len(set(x))==8 for x in ids)
        if row['kind']=='block':assert row['validation']['production_dispatch_exact']
        key=tuple(row[k] for k in ['tokens','routing','kind','variant','cache'])
        grouped[key].append(row)
    summary=[]
    for key,rows in sorted(grouped.items()):
        assert sorted(r['batch'] for r in rows)==[0,1,2]
        samples=sorted(s for r in rows for s in r['samples_us'])
        summary.append(dict(zip(['tokens','routing','kind','variant','cache'],key))|{
            'median_us':statistics.median(r['median_us'] for r in rows),
            'batch_medians_us':[r['median_us'] for r in sorted(rows,key=lambda r:r['batch'])],
            'p10_us':samples[int(.1*(len(samples)-1))],'p90_us':samples[int(.9*(len(samples)-1))],
            'reference_exact':all(r['validation']['reference_exact'] for r in rows),
            'max_abs_diff':max(r['validation']['max_abs_diff'] for r in rows)})
    profiles={}
    entries=json.loads((root/'profiles.json').read_text()) if (root/'profiles.json').exists() else []
    if (root/'shared-profiles.json').exists():entries+=json.loads((root/'shared-profiles.json').read_text())
    for entry in entries:
        assert entry['returncode']==0
        name=entry['name'];raw_path=root/(name+'.raw-base.csv')
        if not raw_path.exists():continue
        raw=tables(raw_path)
        data=[r for r in raw if r.get('ID','').isdigit()]
        assert len(data)==1,(name,len(data))
        units=next(r for r in raw if not r.get('ID','').isdigit());row=data[0]
        match=re.search(r'-t(\d+)-(shared|partial|disjoint)-(gate_up|down)-(scalar1|scalar4|packed1|packed4)-(hot|cold)$',name)
        assert match,name
        tokens,routing,kind,variant,cache=match.groups();tokens=int(tokens)
        validation=json.loads((root/name/'validation.json').read_text())
        want=grouped[tokens,routing,kind,variant,cache][0]['validation']
        assert validation['output_sha256']==want['output_sha256']
        assert validation['reference_exact']==want['reference_exact']
        assert '_project' in row['Kernel Name']
        grid=[int(v) for v in re.findall(r'\d+',row['Grid Size'])]
        block=[int(v) for v in re.findall(r'\d+',row['Block Size'])]
        assert grid==[tokens*8,512 if kind=='gate_up' else 2048,1],grid
        assert block==[32*int(variant[-1]),1,1],block
        metrics={}
        for k,v in row.items():
            if '__' not in k:continue
            try:metrics[k]={'value':float(v.replace(',','')),'unit':units.get(k)}
            except (ValueError,AttributeError):continue
        def value(key):return metrics[key]['value']
        assert metrics['gpu__time_duration.sum']['unit']=='ns'
        duration_us=value('gpu__time_duration.sum')/1000
        dram_lower=value('dram__bytes.sum')/value('dram__bytes.sum.peak_sustained_elapsed.per_second')*1e6
        l2_lower=value('lts__t_sectors.sum')/value('lts__t_sectors.sum.peak_sustained_elapsed.per_second')*1e6
        sms=float(row['device__attribute_multiprocessor_count'].replace(',',''))
        ipc=float(row['device__attribute_max_ipc_per_multiprocessor'].replace(',',''))
        issue_lower=value('smsp__inst_executed.sum')/(sms*ipc*value('sm__cycles_elapsed.avg.per_second'))*1e6
        assert 1.05e9<=value('sm__cycles_elapsed.avg.per_second')<=1.11e9,(name,value('sm__cycles_elapsed.avg.per_second'))
        profiles[name]={'tokens':tokens,'routing':routing,'kind':kind,'variant':variant,'cache':cache,'duration_us':duration_us,
                        'dram_ideal_bandwidth_lower_bound_us':dram_lower,'l2_ideal_bandwidth_lower_bound_us':l2_lower,
                        'issue_ideal_throughput_lower_bound_us':issue_lower,
                        'issue_bound_model':{'sms':sms,'max_warp_instructions_per_sm_cycle':ipc},
                        'metrics':metrics,'validation':validation,'profile_matches_timing_hash':True,
                        'report_sha256':hashlib.sha256((root/(name+'.ncu-rep')).read_bytes()).hexdigest()}
    if args.require_complete:
        assert json.loads((root/'complete.json').read_text())=={'complete':True,'profiles':8}
        assert json.loads((root/'shared-complete.json').read_text())=={'complete':True,'profiles':1}
        assert len(profiles)==9
        assert json.loads((root/'clock-reset.json').read_text())['returncode']==0
        assert json.loads((root/'shared-clock-reset.json').read_text())['returncode']==0
    analysis={'timing_samples':25200,'timing_conditions':len(grouped),'timings':summary,'profiles':profiles,
              'production_kernel_sha256':timing['source_sha256']}
    (root/'analysis.json').write_text(json.dumps(analysis,indent=2))
    with (root/'timing-summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    profile_summary=[]
    for name,p in profiles.items():
        get=lambda key:p['metrics'][key]['value']
        profile_summary.append({'name':name,'kind':p['kind'],'routing':p['routing'],'variant':p['variant'],'cache':p['cache'],
            'duration_us':p['duration_us'],'issue_ideal_us':p['issue_ideal_throughput_lower_bound_us'],
            'dram_ideal_us':p['dram_ideal_bandwidth_lower_bound_us'],'l2_ideal_us':p['l2_ideal_bandwidth_lower_bound_us'],
            'issue_active_pct':get('smsp__issue_active.avg.pct_of_peak_sustained_active'),
            'issue_elapsed_pct':get('smsp__issue_active.avg.pct_of_peak_sustained_elapsed'),
            'eligible_warps':get('smsp__warps_eligible.avg.per_cycle_active'),
            'occupancy_pct':get('sm__warps_active.avg.pct_of_peak_sustained_active'),
            'warp_instructions':get('smsp__inst_executed.sum'),
            'dram_bytes':get('dram__bytes.sum'),'dram_throughput_pct':get('dram__throughput.avg.pct_of_peak_sustained_elapsed'),
            'l2_throughput_pct':get('lts__throughput.avg.pct_of_peak_sustained_elapsed'),
            'sm_frequency_hz':get('sm__cycles_elapsed.avg.per_second'),'reference_exact':p['validation']['reference_exact']})
    if profile_summary:
        with (root/'profile-summary.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(profile_summary[0]));writer.writeheader();writer.writerows(profile_summary)
    print(json.dumps({'timing_samples':25200,'conditions':len(grouped),'verified_profiles':len(profiles)}))
    for name,p in profiles.items():
        keys=['smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__issue_active.avg.pct_of_peak_sustained_elapsed',
              'smsp__warps_eligible.avg.per_cycle_active','sm__warps_active.avg.pct_of_peak_sustained_active',
              'smsp__inst_executed.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed',
              'lts__throughput.avg.pct_of_peak_sustained_elapsed','sm__cycles_elapsed.avg.per_second']
        print(name,json.dumps({'duration_us':p['duration_us'],'issue_lower_us':p['issue_ideal_throughput_lower_bound_us'],'dram_lower_us':p['dram_ideal_bandwidth_lower_bound_us'],
                              'l2_lower_us':p['l2_ideal_bandwidth_lower_bound_us'],
                              **{key:p['metrics'][key]['value'] for key in keys}}))


if __name__=='__main__':main()
