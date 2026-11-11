"""Validate saved microbenchmark samples and aggregate matched scopes."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    llama, layouts = [], []
    native = (root/'native-isolated-graph-perf.stdout').exists()
    timing_path = root/('native-isolated-graph-perf.stdout' if native else 'isolated-graph-perf.stdout')
    for line in timing_path.read_text().splitlines():
        if line.startswith('INFER_LAYOUT '):
            layouts.append(json.loads(line.removeprefix('INFER_LAYOUT ')))
        elif line.startswith('INFER_RESULT '):
            row = json.loads(line.removeprefix('INFER_RESULT '))
            assert len(row['gpu_samples_us']) == 100
            assert all(x > 0 for x in row['gpu_samples_us']) and row['replay_exact']
            row['median_us'] = statistics.median(row['gpu_samples_us'])
            row['p10_us'], row['p90_us'] = statistics.quantiles(row['gpu_samples_us'],n=10)[0::8]
            llama.append(row)
    assert len(llama) == 72 and len(layouts) == 36, (len(llama),len(layouts))
    for row in layouts:
        tensors = {t['name']:t for t in row['tensors']}
        ids = tensors['ids']
        assert ids['ne'] == [8,row['tokens'],1,1] and ids['nb'][1] == 32
        n = row['tokens']
        for token in range(n):
            selected = row['ids'][token*8:token*8+8]
            assert len(set(selected)) == 8 and all(0 <= x < 256 for x in selected)
        groups = [set(row['ids'][t*8:t*8+8]) for t in range(n)]
        if n>1:
            shared = set.intersection(*groups)
            assert len(shared) == {'shared':8,'partial':4,'disjoint':0}[row['routing']]
        x = tensors['x']
        assert x['ne'] == ([512,8,n,1] if row['kind']=='down' else [2048,1,n,1])
        expected_out = [2048,n,1,1] if row['kind']=='block' else [2048 if row['kind']=='down' else 512,8,n,1]
        assert tensors['out']['ne'] == expected_out
        for tensor in row['tensors']:
            assert len(tensor['ne'])==4 and len(tensor['nb'])==4
    baseline = json.loads((root/'infer-opt.json').read_text())
    assert len(baseline['rows']) == 108
    parity=[]
    for row in baseline['rows']:
        if row['implementation'] != 'packed':
            continue
        ref = next(x for x in baseline['rows'] if x['implementation']=='reference' and
                   all(x[k]==row[k] for k in ['tokens','routing','kind','cache']))
        parity.append(row['output_sha256']==ref['output_sha256'])
    paired=[]
    for row in baseline['rows']:
        assert len(row['gpu_samples_us']) == 100 and row['replay_exact']
        assert row['median_us'] == statistics.median(row['gpu_samples_us'])
        target = next(x for x in llama if all(x[k]==row[k] for k in ['tokens','routing','kind','cache']))
        paired.append({**{k:row[k] for k in ['tokens','routing','kind','cache','implementation','packed_active']},
            'infer_opt_us':row['median_us'],'llama_us':target['median_us'],
            'infer_opt_over_llama':row['median_us']/target['median_us']})
    standard=[]
    for path in sorted(root.glob('standard-console-*.stdout')):
        matches = re.findall(r'MUL_MAT_ID\((.*?)\):\s+(\d+) runs -\s+([\d.]+) us/run',path.read_text())
        assert len(matches)==18, (path,len(matches))
        for params,runs,time_us in matches:
            standard.append({'file':path.name,'params':dict(x.split('=',1) for x in params.split(',')),
                             'n_runs':int(runs),'mean_us':float(time_us)})
    graphs=[]
    for directory in ['native-graphs','view-graphs']:
        for path in sorted((root/directory).glob('*.dot')):
            symbols=[]
            for line in path.read_text().splitlines():
                if '| {ID |' in line and '_Z' in line:
                    symbols.append(line.split(' | ',2)[2].split('\\<\\<\\<',1)[0])
            names = subprocess.check_output(['c++filt'],input='\n'.join(symbols)+'\n',text=True).splitlines()
            graphs.append({'file':str(path.relative_to(root)),'kernel_count':len(names),'kernels':names})
    if native:
        assert len([x for x in graphs if x['file'].startswith('native-graphs/')])==36
        fused=[x for x in graphs if x['file'].startswith('native-graphs/') and '-gate_up-' in x['file']]
        assert len(fused)==9 and all(x['kernel_count']==2 for x in fused)
        assert all(any('mul_mat_vec_q' in name and 'true' in name for name in x['kernels']) for x in fused)
        view=[x for x in graphs if x['file'].startswith('view-graphs/')]
        assert len(view)==1 and view[0]['kernel_count']==5
    result={'environment':json.loads((root/'environment.json').read_text()),'llama':llama,
            'paired':paired,'standard_perf':standard,
            'captured_graphs':graphs,
            'llama_layout':'native' if native else 'view',
            'counter_permission_denied':[p.name for p in root.glob('ncu-*.stdout') if 'ERR_NVGPUCTRPERM' in p.read_text()],
            'validation':{'llama_sample_rows':len(llama),'layout_cases':len(layouts),'infer_opt_sample_rows':len(baseline['rows']),
                          'llama_samples':sum(len(x['gpu_samples_us']) for x in llama),
                          'infer_opt_samples':sum(len(x['gpu_samples_us']) for x in baseline['rows']),
                          'all_replay_exact':True,'shape_routes_checked':True,
                          'infer_opt_reference_packed_equal':sum(parity),'infer_opt_reference_packed_comparisons':len(parity)},
            'limits':['Native Q8_0/Q8_1 W8A8 versus group128 W8A16; different numerical arithmetic.',
                      'Synthetic standalone expert cases, not full model or MTP round throughput.',
                      'Profiler counter runs are separate from timing samples.']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2))
    with (root/'paired-timings.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(paired[0]))
        writer.writeheader();writer.writerows(paired)
    print(json.dumps(result['validation'],indent=2))
    for t in [1,2,4]:
        for kind in ['gate_up','down','block']:
            rows = [x for x in paired if x['tokens']==t and x['kind']==kind and x['cache']=='cold' and x['routing']=='partial']
            print(t,kind,rows)


if __name__=='__main__':
    main()
