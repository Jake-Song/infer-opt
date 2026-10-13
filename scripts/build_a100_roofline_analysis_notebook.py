#!/usr/bin/env python3
"""Build a local-files-only Korean analysis notebook (no GPU execution)."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, name, source):
    result = dict(cell_type=kind, id=name, metadata={}, source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(destination):
    cells = []
    def md(name, source): cells.append(cell('markdown', name, source))
    def code(name, source): cells.append(cell('code', name, source))
    md('intro', '''
        # 로컬 A100 결과 분석: 시간과 연산별 roofline

        저장소의 Python 환경에서 실행하세요. GPU는 필요하지 않습니다.
        `uv sync --group notebooks` 후 VS Code/Jupyter에서 `.venv` 커널을 선택합니다.
        아래 입력 경로를 지정하고 모든 셀을 실행하면 로컬 JSON을 읽고 표·PNG·CSV를 만듭니다.
        모델 추론, GPU 측정, 원격 연결, 다운로드는 수행하지 않습니다.

        기본 입력은 2026-10-14 A100 베이스라인 원본입니다. 원본이 없는 checkout에서는
        **명시적으로 지정한** `notebooks/assets/a100_baseline_20261014_summary.json`을 사용할 수 있습니다.
        자동 대체는 하지 않습니다. 이 요약 파일에는 반복 측정 샘플이 없습니다.

        기존 baseline에는 연산별 시간이 없습니다. 이를 전체 시간에서 임의로 나누지 않습니다.
        연산별 시간은 선택 입력 `roofline_results.json`이 있을 때만 표시합니다.
    ''')
    code('paths', '''
        import csv
        import hashlib
        import json
        import sys
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display, Image

        ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / 'infer_opt' / 'roofline.py').is_file()), None)
        if ROOT is None:
            raise RuntimeError('저장소 루트 또는 notebooks 폴더에서 실행하세요.')
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from infer_opt.config import ModelConfig
        from infer_opt.roofline import operation_costs, assess, plot_results

        BASELINE_PATH = ROOT / 'experiments/baseline/a100-0.5b-bf16-20261014-cli/results.json'
        # 원본이 없으면 위 경로를 ROOT / 'notebooks/assets/a100_baseline_20261014_summary.json'으로 변경
        PROFILE_PATH = None  # 예: ROOT / 'experiments/roofline/my-run/roofline_results.json'
        OUTPUT_DIR = ROOT / 'experiments/roofline/local-analysis'

        # 민감도 분석을 위한 가상 기준선. A100 사양/실측 성능이 아닙니다.
        # 아래 모형 표에만 사용되며, profiler 파일의 실측 calibration과 섞지 않습니다.
        ASSUMED_COMPUTE_TFLOPS = 100.0
        ASSUMED_BANDWIDTH_GBPS = 1000.0
        assert min(ASSUMED_COMPUTE_TFLOPS, ASSUMED_BANDWIDTH_GBPS) > 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        def table(rows, columns):
            def fmt(value):
                if value is None: return '—'
                if isinstance(value, float): return f'{value:.4f}'
                return str(value).replace('|', '/')
            display(Markdown('| ' + ' | '.join(columns) + ' |\\n|' + '|'.join(['---'] * len(columns)) + '|\\n' +
                '\\n'.join('| ' + ' | '.join(fmt(row.get(k)) for k in columns) + ' |' for row in rows)))

        def save_csv(name, rows):
            if rows:
                with (OUTPUT_DIR / name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    ''')
    code('load', '''
        def read_result(path):
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'입력 파일이 없습니다: {path}. 경로 설정을 수정하세요.')
            raw = path.read_bytes()
            data = json.loads(raw)
            if data.get('status') != 'passed':
                raise ValueError(f'실험 전체가 passed인 파일만 분석합니다: {path}')
            return data, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

        baseline, baseline_input = read_result(BASELINE_PATH)
        is_summary = 'origin' in baseline and 'source_commit' in baseline
        config = ModelConfig(**baseline['model'])
        run = baseline['run']
        decode_len = run['decode_len']
        print('입력:', baseline_input['path'])
        print('종류:', '저장된 요약 (샘플 없음)' if is_summary else '원본 results.json')
        print('환경:', baseline['environment'])
        print('설정:', run)
        print('소스 커밋:', baseline.get('source', {}).get('git_commit', baseline.get('source_commit')))
        baseline_rows = []
        for condition in baseline['conditions']:
            valid = condition.get('validation_passed') if is_summary else condition.get('validation', {}).get('passed')
            if condition.get('status') != 'passed' or valid is not True:
                raise ValueError(f"검증 미통과 조건: {condition['prompt_len']}")
            if is_summary:
                prefill = condition['prefill_wall_median_ms']
                decode = condition['decode_wall_median_ms_per_token']
            else:
                prefill = condition['benchmark']['prefill']['wall_ms']['median']
                decode = condition['benchmark']['decode']['wall_ms']['median'] / decode_len
            baseline_rows.append(dict(prompt_len=condition['prompt_len'], prefill_wall_ms=prefill,
                                      decode_wall_ms_per_token=decode, validation='passed'))
        table(baseline_rows, list(baseline_rows[0]))
        save_csv('baseline-times.csv', baseline_rows)
    ''')
    md('timing-scope', '''
        ## 실제 측정 시간

        Prefill은 전체 prompt 처리, decode는 cache가 증가하는 고정 토큰 replay의
        **128단계(설정값 `decode_len`) 총 wall 시간 / 단계 수**입니다.
        개별 토큰 latency 분포나 고정 context 한 토큰 측정으로 해석하지 않습니다.
        아래 산점도는 각 반복 실행의 평균 ms/token입니다. Wall에는 host launch·동기화가 포함됩니다.
        준비·warmup·검증 시간은 제외된 저장값을 읽습니다.
    ''')
    code('timing-plot', '''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, phase, key in zip(axes, ['prefill', 'decode'], ['prefill_wall_ms', 'decode_wall_ms_per_token']):
            ax.plot([r['prompt_len'] for r in baseline_rows], [r[key] for r in baseline_rows], 'o-', label='Wall median')
            if not is_summary:
                for condition in baseline['conditions']:
                    samples = condition['benchmark'][phase].get('samples', [])
                    values = [s['wall_ms'] / (decode_len if phase == 'decode' else 1) for s in samples]
                    ax.scatter([condition['prompt_len']] * len(values), values, alpha=.25, color='gray')
            ax.set(xlabel='Prompt tokens', ylabel='ms/token (sequence average)' if phase == 'decode' else 'ms', title=phase)
            ax.set_xscale('log', base=2)
            ax.grid(alpha=.3)
            ax.legend()
        fig.savefig(OUTPUT_DIR / 'baseline-times.png', dpi=160)
        plt.show()
        plt.close(fig)
    ''')
    md('model-scope', r'''
        ## 연산량으로 병목을 예측하기

        $I=F/Q$ (FLOP/byte), $P=\min(C,I B)$, $I_{ridge}=C/B$.
        TFLOP/s와 GB/s 단위에서는 `ridge = 1000*C/B`입니다.
        GEMM의 FLOPs는 `2MNK`, 논리적 bytes는 `(MK+KN+MN)*dtype_bytes`입니다.
        아래 표는 **가상 C/B 기준선**에 대한 민감도 분석입니다. 실측 병목 판정이 아닙니다.
        C/B 설정을 바꿔 예측이 얼마나 바뀌는지 확인하세요.

        Prefill은 입력 길이 전체, decode는 **해당 prefix 뒤의 다음 한 토큰**을 모델링합니다.
        128단계 평균 실측 시간과 조합해 연산별 TFLOP/s를 만들지 않습니다.
        연산량·bytes는 모든 레이어 합계이며 KV view는 0, attention은 유효 causal QK/AV와
        현재 구현의 GQA 복제·mask를 포함합니다. Softmax, 반복 tile load, 일부 FP32 중간값은 제외됩니다.
        따라서 논리적 bytes/time은 effective GB/s이며 HBM counter 실측이 아닙니다.
        RMSNorm·RoPE·SwiGLU·복사는 Tensor Core roof로 판정하지 않습니다.
    ''')
    code('model-table', '''
        element_bytes = {'bfloat16': 2, 'float16': 2, 'float32': 4}[run['dtype']]
        model_rows = []
        for length in sorted(r['prompt_len'] for r in baseline_rows):
            for phase, tokens, context in [('prefill', length, length), ('decode', 1, length + 1)]:
                costs = operation_costs(config, tokens=tokens, context=context,
                                        batch=run['batch_size'], element_bytes=element_bytes)
                for op, cost in costs.items():
                    prediction = assess(cost, 0, ASSUMED_COMPUTE_TFLOPS, ASSUMED_BANDWIDTH_GBPS)
                    model_rows.append(dict(prompt_len=length, phase=phase, operation=op, kind=cost['kind'],
                        flops=cost['flops'], logical_bytes=cost['logical_bytes'],
                        intensity=prediction['arithmetic_intensity'], gpu_ms=None,
                        predicted_bound=prediction['predicted_bound'], accounting=cost['accounting']))
        print(f'가상 기준선: {ASSUMED_COMPUTE_TFLOPS} TFLOP/s, {ASSUMED_BANDWIDTH_GBPS} GB/s; 실측 아님')
        SELECTED_PROMPT = max(r['prompt_len'] for r in baseline_rows)
        selected = [r for r in model_rows if r['prompt_len'] == SELECTED_PROMPT]
        table(selected, ['phase', 'operation', 'flops', 'logical_bytes', 'intensity', 'gpu_ms', 'predicted_bound'])
        save_csv('operation-model.csv', model_rows)
    ''')
    code('model-plot', '''
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ridge = ASSUMED_COMPUTE_TFLOPS * 1000 / ASSUMED_BANDWIDTH_GBPS
        xs = np.geomspace(.1, 1e5, 300)
        for ax, phase in zip(axes, ['prefill', 'decode']):
            ax.loglog(xs, np.minimum(ASSUMED_COMPUTE_TFLOPS, xs * ASSUMED_BANDWIDTH_GBPS / 1000), label='Hypothetical roof')
            ax.axvline(ridge, color='gray', linestyle=':', label='Hypothetical ridge')
            for row in selected:
                if row['phase'] != phase or row['kind'] not in ('gemm', 'attention'): continue
                # Vertical lines encode AI only; no measured performance point is invented.
                ax.axvline(row['intensity'], alpha=.3, label=row['operation'])
            ax.set(xlabel='Logical FLOP/byte', ylabel='Hypothetical ceiling (TFLOP/s)', title=f'{phase}, prefix={SELECTED_PROMPT}')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle('MODEL ONLY: vertical lines are operation intensities, not measured throughput')
        fig.savefig(OUTPUT_DIR / 'hypothetical-roofline.png', dpi=160)
        plt.show()
        plt.close(fig)
    ''')
    md('profile-scope', '''
        ## 선택 입력: 저장된 연산별 profiler 결과

        `PROFILE_PATH`에 `infer_opt.roofline`이 저장한 `roofline_results.json`을 지정하면
        그 파일의 환경·실측 calibration·검증 결과를 사용합니다. 위 baseline과는 별도 실험입니다.
        이 파일의 decode는 고정 prefix 다음 한 토큰이며, baseline의 128단계 평균과 다릅니다.
        GPU 시간은 profiler를 켠 별도 실행의 kernel 시간 합계입니다. 재귀 CPU parent 시간을
        중복 합산하지 않았는지 accounting audit를 확인합니다. Profiler OFF wall 시간과도 구분합니다.
        Roof 아래에 멀리 떨어진 점은 launch·병렬성·cache 등 추가 진단이 필요합니다.
    ''')
    code('profile-results', '''
        profile_input = None
        if PROFILE_PATH is None:
            display(Markdown('**연산별 실측 시간 없음.** `gpu_ms`는 결측값입니다. 기존 전체 시간만으로 복원할 수 없습니다.'))
        else:
            profile_data, profile_input = read_result(PROFILE_PATH)
            print('Profiler 입력:', profile_input)
            print('별도 실험 환경:', profile_data['environment'])
            print('별도 실험 모델:', profile_data['model'])
            print('측정 범위:', profile_data.get('decode_scope'))
            print('실측 기준선:', profile_data['calibration'])
            measured_rows = []
            for condition in profile_data['conditions']:
                if (condition['status'] != 'passed' or not condition['validation']['passed']
                        or not condition['profile']['accounting_passed']):
                    raise ValueError('검증 또는 kernel accounting audit 미통과 profiler 조건')
                measured_rows.extend(condition['operations'])
            if not measured_rows:
                raise ValueError('연산별 측정값이 없는 profiler 파일입니다.')
            table(measured_rows, ['prompt_len', 'phase', 'operation', 'gpu_ms', 'gpu_share_pct',
                                 'arithmetic_intensity', 'achieved_tflops', 'predicted_bound', 'evidence'])
            save_csv('measured-operations.csv', measured_rows)
            for path in plot_results(profile_data, OUTPUT_DIR):
                display(Image(filename=str(path)))
    ''')
    code('manifest', '''
        manifest = dict(baseline_input=baseline_input, profile_input=profile_input,
                        assumed_compute_tflops=ASSUMED_COMPUTE_TFLOPS,
                        assumed_bandwidth_gbps=ASSUMED_BANDWIDTH_GBPS,
                        note='Model assumptions are not measured A100 performance; missing operation times stay missing.')
        (OUTPUT_DIR / 'analysis-inputs.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print('분석 결과 저장:', OUTPUT_DIR)
        print('입력 JSON은 수정하지 않았습니다. 각 입력의 SHA-256은 analysis-inputs.json에 기록했습니다.')
    ''')
    md('interpretation', '''
        ## 결과를 해석할 때

        Prefill의 여러 토큰은 가중치를 재사용하므로 GEMM의 intensity가 커집니다.
        B=1 decode는 가중치 재사용이 작아 memory 측에 가까워질 수 있습니다.
        이는 모형의 예측이며 실제 메모리 포화 여부는 연산별 시간과 counter 없이는 확정할 수 없습니다.
        전체 wall 시간이 길어도 kernel이 길다는 뜻은 아닙니다. Host dispatch와 launch 간격도 포함됩니다.

        파일에 없는 정보는 결측으로 남깁니다. 연산별 kernel 시간·DRAM counter는
        집계된 baseline JSON만으로 새로 만들 수 없습니다.
        배경: [NVIDIA Roofline 가이드](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts).
    ''')
    # Keep notebook metadata local and kernel-neutral.
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3 (infer-opt)', language='python', name='python3'),
        language_info=dict(name='python', version='3.13')), nbformat=4, nbformat_minor=5)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    build_notebook(ROOT / 'notebooks/04_a100_roofline_analysis.ipynb')
