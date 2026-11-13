"""Independently check recovered hashes, timing aggregation and sorted-path parity."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--archive', help='Read retained raw traces from the recovered archive when not extracted')
    args = p.parse_args()
    root = Path(args.results)
    read = lambda name: json.loads((root/name).read_text())
    archive = read('archive-manifest.json')
    packed = tarfile.open(args.archive) if args.archive else None
    for name, metadata in archive['files'].items():
        path = root/name
        if path.exists():
            assert path.stat().st_size == metadata['bytes'], name
            stream = path.open('rb')
        else:
            assert packed is not None, name
            assert packed.getmember(name).size == metadata['bytes'], name
            stream = packed.extractfile(name)
        digest = hashlib.sha256()
        with stream:
            for chunk in iter(lambda: stream.read(8*2**20), b''):
                digest.update(chunk)
        assert digest.hexdigest() == metadata['sha256'], name
    if packed:
        packed.close()
    assert read('exit.json')['exit_code'] == 0
    assert read('experts/complete.json')['complete']
    assert read('model/complete.json')['complete']
    count = 0
    def check_timing(row):
        nonlocal count
        samples = row['samples']
        assert len(samples) == 5
        for kind in ('wall', 'cuda'):
            values = [s[f'{kind}_ms'] for s in samples]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert statistics.median(values) == row[f'median_{kind}_ms']
        count += len(samples)
    expert_rows = read('experts/experts.json')['rows']
    assert len(expert_rows) == 9
    for row in expert_rows:
        assert row['backend'] == 'sorted' and row['numerics']['exact']
        assert row['numerics']['max_abs'] == 0
        for backend in ('reference', 'sorted'):
            check_timing(row[backend])
        assert row['timing_ratio'] == row['reference']['median_wall_ms'] / row['sorted']['median_wall_ms']
    expert_samples = count
    inputs = read('model/inputs.json')
    model_rows = read('model/model.json')['rows']
    assert {(r['prompt'], r['tokens']) for r in model_rows} == set(itertools.product(
        ['math', 'korean'], [128, 512, 2048, 4096]))
    assert len(model_rows) == 8
    rows = []
    for row in model_rows:
        assert len(inputs[f"{row['prompt']}-{row['tokens']}"]) == row['tokens']
        for backend in ('reference', 'sorted'):
            check_timing(row[backend])
            assert len(row[backend]['generated_ids']) == 16
        candidate = row['sorted']
        assert candidate['numerics']['exact'] and candidate['numerics']['max_abs'] == 0
        assert candidate['kl_reference_to_candidate'] == 0
        assert candidate['first_divergence'] is None and candidate['greedy_parity']
        assert candidate['generated_ids'] == row['reference']['generated_ids']
        ratio = row['reference']['median_wall_ms'] / candidate['median_wall_ms']
        assert ratio == candidate['timing_ratio']
        rows.append({'prompt': row['prompt'], 'tokens': row['tokens'],
            'reference_ms': row['reference']['median_wall_ms'],
            'sorted_ms': candidate['median_wall_ms'], 'speedup': ratio})
    for folder in ('experts', 'model'):
        for name, digest in read(folder+'/manifest.json')['source_sha256'].items():
            path = root/'executed-source'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    summary = {'archive_files_verified': len(archive['files']), 'expert_cases': len(expert_rows),
        'expert_timing_samples': expert_samples, 'model_cases': len(rows),
        'model_timing_samples': count-expert_samples, 'continuation_ids_checked': len(rows)*2*16,
        'all_last_logits_exact': True, 'all_continuations_equal': True, 'model_rows': rows}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
