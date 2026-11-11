"""Validate complete-request CUDA traces without assigning speculative phases."""
import argparse
import json
from pathlib import Path
import sqlite3


def analyze(root):
    assert json.loads((root / 'profile-exit.json').read_text())['returncode'] == 0
    timed = json.loads((root / 'timed-requests.json').read_text())
    reference = {(r['prompt'], r['k']): r['response']['tokens'] for r in timed}
    traces = []
    for k in [0, 1, 2, 3]:
        for prompt in ['math', 'korean']:
            label = f'trace-k{k}-{prompt}'
            db = root / 'traces' / (label + '.sqlite')
            report = db.with_suffix('.nsys-rep')
            assert db.is_file() and report.is_file() and report.stat().st_size > 0
            connection = sqlite3.connect(db)
            assert connection.execute('pragma quick_check').fetchone()[0] == 'ok'
            graph_count = connection.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
                join StringIds s on s.id=r.nameId where s.value like '%GraphLaunch%'""").fetchone()[0]
            kernels = connection.execute('''select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k
                join StringIds s on s.id=k.demangledName
                group by k.demangledName,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ''').fetchall()
            assert graph_count > 0 and kernels
            native = [{'name': name, 'grid': [gx, gy, gz], 'block': [bx, by, bz],
                       'calls': count, 'gpu_ms': duration / 1e6}
                      for name, gx, gy, gz, bx, by, bz, count, duration in kernels
                      if 'mul_mat_vec_q_moe<' in name]
            if k:
                assert any(r['block'] == [32, k + 1, 1] for r in native), (label, native)
            generic = [r for r in kernels if 'mul_mat_vec_q<' in r[0]]
            assert generic
            profiled = json.loads((root / (label + '-profiled.json')).read_text())
            same_ids = profiled['response']['tokens'] == reference[prompt, k]
            for stage in ['start', 'stop', 'kernel-stats']:
                assert json.loads((root / f'{label}-{stage}.exit.json').read_text())['returncode'] == 0
            assert json.loads((root / f'trace-k{k}-shutdown.exit.json').read_text())['returncode'] == 0
            traces.append({'prompt': prompt, 'k': k, 'cuda_graph_launches': graph_count,
                           'kernel_calls': sum(r[7] for r in kernels),
                           'gpu_kernel_sum_ms_including_prefill': sum(r[8] for r in kernels) / 1e6,
                           'native_moe_kernels': native, 'generic_mmvq_present': True,
                           'profiled_ids_match_same_mode_unprofiled': same_ids,
                           'boundary': 'whole request including prefill; target and draft are not separately labeled'})
            connection.close()
    result = {'passed': True, 'traces': traces,
              'all_profiled_ids_match_same_mode_unprofiled': all(r['profiled_ids_match_same_mode_unprofiled'] for r in traces)}
    (root / 'trace-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'traces': len(traces),
                     'same_mode_output_match': result['all_profiled_ids_match_same_mode_unprofiled']}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    analyze(parser.parse_args().results)
