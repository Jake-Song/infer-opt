#!/usr/bin/env python3
"""Build a standalone Colab notebook containing the current baseline source."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import textwrap
from pathlib import Path

from run_baseline_colab import ROOT, build_bundle


def cell(kind: str, name: str, source: str, *, hidden: bool = False) -> dict:
    result = {
        "cell_type": kind, "id": name, "metadata": {"id": name},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        if hidden:
            result["metadata"]["cellView"] = "form"
    return result


def build_notebook(destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="a100-notebook-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        source = build_bundle(archive)
        payload = archive.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    literal = "\n".join(f'    "{part}"' for part in textwrap.wrap(encoded, 100))
    cells = [
        cell("markdown", "intro", """
            # A100에서 직접 실행하는 0.5B 추론 베이스라인

            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/infer-opt/blob/main/notebooks/03_a100_baseline_colab.ipynb)

            **사용법:** 이 `.ipynb` 파일을 Colab의 **파일 → 노트북 업로드**로 엽니다.
            **런타임 → 런타임 유형 변경**에서 GPU와 **A100**을 선택한 뒤,
            위에서 아래로 실행하거나 **모두 실행**을 누르세요.
            GPU 종류는 첫 셀에서 실제로 확인합니다.

            노트북 안에 현재 저장소의 소스와 테스트를 포함했습니다.
            별도 GitHub clone, 소스 업로드, Colab CLI, Drive 연결 없이 실행합니다.
            Colab에 설치된 CUDA PyTorch를 그대로 사용합니다.

            | 항목 | 기본 설정 |
            |---|---|
            | 모델 | 저장소의 decoder-only Transformer, 랜덤 가중치 494,962,688개 |
            | 구조 | RMSNorm, RoPE, SwiGLU, GQA, contiguous KV cache |
            | 크기 | 8 layers, hidden 2048, FFN 5632, Q heads 16 / KV heads 4, vocab 32768 |
            | 실행 | B=1, BF16, eager PyTorch SDPA 자동 backend 선택 |
            | 입력 길이 | 128 / 512 / 2048 / 8192 tokens |
            | Decode | 고정 토큰 replay 128단계, 매 단계 KV cache 증가 |
            | 반복 | 조건마다 warmup 3회 + 측정 10회 |

            이는 **랜덤 모델의 실행 성능 실험**이며 생성 품질 평가가 아닙니다.
            Prefill과 decode를 따로 측정합니다. 이 노트북은 eager 기준선이며,
            `torch.compile`이나 별도 최적화 라이브러리와의 비교는 포함하지 않습니다.

            첫 셀 실행부터 최대 120분을 사용하고, 마지막 5분은 결과 회수용으로 남깁니다.
            오류나 중단이 발생해도 마지막 다운로드 셀을 실행하면 이미 저장된 결과를 받을 수 있습니다.
        """),
        cell("code", "hardware", """
            #@title 1. 실제 A100 / BF16 환경 확인
            import json
            import sys
            import time
            from pathlib import Path
            import torch

            NOTEBOOK_STARTED = time.monotonic()
            TOTAL_BUDGET_MINUTES = 120
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU가 없습니다. 런타임 유형에서 A100 GPU를 선택하세요.")
            gpu = torch.cuda.get_device_properties(0)
            if "A100" not in gpu.name:
                raise RuntimeError(f"A100 실험입니다. 현재 장치: {gpu.name}")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("현재 GPU / PyTorch 환경에서 BF16을 사용할 수 없습니다.")
            print(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 2**30:.2f} GiB")
            print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Python: {sys.version.split()[0]}")
            print("A100 / BF16 확인 완료. CUDA PyTorch는 재설치하지 않습니다.")
        """),
        cell("markdown", "source-info", f"""
            ## 2. 포함된 실험 코드 준비

            다음 셀은 접혀 있는 소스 묶음을 풀고 SHA-256을 확인합니다.
            모델·측정·검증 코드는 터미널 베이스라인과 동일합니다.
            반복 실행 시 새 작업 폴더를 만들므로 이전 결과를 덮어쓰지 않습니다.

            소스 SHA-256: `{source['sha256']}`

            검증 기준은 전체 logits, 모든 레이어의 유효 K/V, cache prefix 보존,
            최종 cache 위치입니다. BF16 비교는 `atol=rtol=0.02`를 사용합니다.
            오차 계산에만 FP32를 사용하고 검증 시간은 성능 측정에서 제외합니다.
        """),
        cell("code", "unpack", "\n".join([
            "#@title 2. 실험 소스 복원 (포함된 코드 — 펼쳐서 확인 가능)",
            "import base64, hashlib, io, shutil, tarfile, tempfile",
            "ARCHIVE_B64 = (", literal, ")",
            f"ARCHIVE_SHA256 = {hashlib.sha256(payload).hexdigest()!r}",
            "archive_bytes = base64.b64decode(ARCHIVE_B64)",
            "assert hashlib.sha256(archive_bytes).hexdigest() == ARCHIVE_SHA256, '소스 묶음 손상'",
            "work_parent = '/content' if Path('/content').is_dir() else None",
            "WORK_ROOT = Path(tempfile.mkdtemp(prefix='infer-opt-a100-', dir=work_parent))",
            "with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as archive:",
            "    archive.extractall(WORK_ROOT, filter='data')",
            "SOURCE = json.loads((WORK_ROOT / 'source_manifest.json').read_text())",
            "for name, expected in SOURCE['files'].items():",
            "    assert hashlib.sha256((WORK_ROOT / name).read_bytes()).hexdigest() == expected, name",
            "RESULT_DIR = WORK_ROOT / 'results'",
            "RESULT_DIR.mkdir()",
            "(RESULT_DIR / 'source.tar.gz').write_bytes(archive_bytes)",
            "shutil.copyfile(WORK_ROOT / 'source_manifest.json', RESULT_DIR / 'source_manifest.json')",
            "TESTS_PASSED = False",
            "print('작업 폴더:', WORK_ROOT)",
            "print('소스 SHA-256:', SOURCE['sha256'])",
        ]), hidden=True),
        cell("code", "settings", """
            #@title 3. 실험 설정 — 기본값으로 먼저 실행하세요
            PROMPT_LENGTHS = [128, 512, 2048, 8192]
            DECODE_LEN = 128
            WARMUP = 3
            ITERATIONS = 10
            SEED = 0

            assert PROMPT_LENGTHS and all(isinstance(n, int) and n > 0 for n in PROMPT_LENGTHS)
            assert len(set(PROMPT_LENGTHS)) == len(PROMPT_LENGTHS)
            assert DECODE_LEN > 0 and ITERATIONS > 0 and WARMUP >= 0
            assert max(PROMPT_LENGTHS) + DECODE_LEN <= 16384
            RUN_SETTINGS = dict(prompt_lengths=PROMPT_LENGTHS, decode_len=DECODE_LEN,
                                warmup=WARMUP, iterations=ITERATIONS, seed=SEED,
                                device="cuda", dtype="bfloat16", expected_gpu="A100",
                                total_budget_minutes=TOTAL_BUDGET_MINUTES)
            (RESULT_DIR / 'notebook_settings.json').write_text(json.dumps(RUN_SETTINGS, indent=2))
            print(json.dumps(RUN_SETTINGS, indent=2))
        """),
        cell("code", "tests", """
            #@title 4. 회귀 테스트 — 통과한 경우에만 측정
            import importlib.util
            import subprocess

            TESTS_PASSED = False
            if importlib.util.find_spec('pytest') is None:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest>=8'],
                               check=True, timeout=120)
            tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=WORK_ROOT,
                                   capture_output=True, text=True, timeout=300)
            (RESULT_DIR / 'tests.txt').write_text(tests.stdout + tests.stderr)
            print(tests.stdout + tests.stderr)
            TESTS_PASSED = tests.returncode == 0
            if not TESTS_PASSED:
                raise RuntimeError("회귀 테스트 실패. 측정을 진행하지 않고 tests.txt를 확인하세요.")
        """),
        cell("markdown", "measurement-info", """
            ## 5. 정확성 검증 후 측정

            각 입력 길이마다 `validating → measuring → passed` 순서로 진행됩니다.
            정확성 검증 실패 조건에는 성능 수치를 기록하지 않습니다.

            - **Prefill:** 모든 prompt 위치의 vocabulary logits와 KV 쓰기까지 포함합니다.
            - **Decode:** 준비용 prefill이 끝난 뒤, cache를 늘리는 한 토큰 forward 128회를 측정합니다.
            - 모델 로딩, 입력 전송, cache 할당·초기화, warmup, 검증, 토큰 선택은 측정에서 제외합니다.
            - 주 지표는 동기화한 wall time입니다. Python dispatch와 최종 동기화를 포함합니다.
              CUDA event 시간도 저장하지만, host launch 사이의 빈 시간이 포함될 수 있어
              순수 GPU kernel 시간의 합으로 해석하지 않습니다.

            셀을 중단한 경우에도 완료된 조건은 `results.json`에 남습니다.
            이 셀을 다시 실행하려면 **2번 소스 준비 셀부터** 실행해 새 결과 폴더를 만드세요.
        """),
        cell("code", "measure", """
            #@title 5. 베이스라인 실행 — 진행 로그 출력, 시간 제한 및 결과 회수
            import zipfile

            def pack_results():
                archive_path = WORK_ROOT / (WORK_ROOT.name + '-results.zip')
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
                    for path in sorted(RESULT_DIR.rglob('*')):
                        if path.is_file():
                            archive.write(path, path.relative_to(RESULT_DIR))
                return archive_path

            assert TESTS_PASSED, "4번 테스트 셀을 먼저 통과해야 합니다."
            if (RESULT_DIR / 'results.json').exists():
                raise FileExistsError("이 폴더에는 실행 기록이 있습니다. 재실행은 2번 셀부터 진행하세요.")
            remaining = TOTAL_BUDGET_MINUTES * 60 - (time.monotonic() - NOTEBOOK_STARTED) - 300
            if remaining < 60:
                raise RuntimeError("실행 예산이 부족합니다. 결과를 다운로드한 뒤 1번 셀부터 시작하세요.")
            command = [sys.executable, '-u', '-m', 'infer_opt', 'baseline',
                       '--output', str(RESULT_DIR), '--device', 'cuda', '--dtype', 'bfloat16',
                       '--expected-gpu', 'A100', '--prompt-lengths', *map(str, PROMPT_LENGTHS),
                       '--decode-len', str(DECODE_LEN), '--warmup', str(WARMUP),
                       '--iterations', str(ITERATIONS), '--seed', str(SEED),
                       '--budget-minutes', str(remaining / 60)]
            execution = {'status': 'running', 'command': command,
                         'source_archive_sha256': ARCHIVE_SHA256}
            log_path = RESULT_DIR / 'benchmark.log'
            process = None
            deadline = time.monotonic() + remaining
            try:
                with log_path.open('w') as log, log_path.open() as reader:
                    process = subprocess.Popen(command, cwd=WORK_ROOT, stdout=log, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        print(reader.read(), end='', flush=True)
                        if time.monotonic() >= deadline:
                            raise TimeoutError('노트북 실행 시간 제한에 도달했습니다.')
                        time.sleep(0.5)
                    print(reader.read(), end='', flush=True)
                execution['returncode'] = process.returncode
                saved_path = RESULT_DIR / 'results.json'
                saved_status = json.loads(saved_path.read_text())['status'] if saved_path.exists() else 'error'
                execution['status'] = saved_status if process.returncode == 0 else (
                    saved_status if saved_status in ('error', 'validation_failed') else 'error')
            except (Exception, KeyboardInterrupt) as exc:
                execution['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
                execution['error'] = type(exc).__name__ + ': ' + str(exc)
                print(execution['error'])
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                execution['notebook_elapsed_seconds'] = time.monotonic() - NOTEBOOK_STARTED
                (RESULT_DIR / 'notebook_execution.json').write_text(json.dumps(execution, indent=2))
                RESULTS_ZIP = pack_results()
                print('실행 상태:', execution['status'])
                print('결과 ZIP:', RESULTS_ZIP)
            if execution['status'] != 'passed':
                print('전체 실험이 통과하지 않았습니다. 아래 결과/로그를 확인하고 ZIP을 다운로드하세요.')
        """),
        cell("code", "summary", """
            #@title 6. 결과 표와 지연시간 그래프
            import csv
            from IPython.display import Markdown, display

            result_path = RESULT_DIR / 'results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                print('베이스라인 상태:', result['status'])
                print('조건별 상태:', [(c['prompt_len'], c['status']) for c in result['conditions']])
                print('환경:', json.dumps(result.get('environment', {}), ensure_ascii=False, indent=2))
            report_path = RESULT_DIR / 'report.md'
            if report_path.exists():
                display(Markdown(report_path.read_text()))
            else:
                print('측정 보고서가 없습니다. benchmark.log 또는 tests.txt를 확인하세요.')
            summary_path = RESULT_DIR / 'summary.csv'
            rows = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
            measured = [row for row in rows if row['status'] == 'passed']
            if measured:
                import matplotlib.pyplot as plt
                prompts = [int(row['prompt_len']) for row in measured]
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
                for ax, key, label in [
                    (axes[0], 'prefill_wall_median_ms', 'Prefill latency (ms)'),
                    (axes[1], 'decode_wall_median_ms_per_token', 'Decode latency (ms/token)'),
                ]:
                    ax.plot(prompts, [float(row[key]) for row in measured], marker='o')
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(prompts, [str(n) for n in prompts])
                    ax.set_xlabel('Prompt tokens')
                    ax.set_ylabel(label)
                    ax.grid(alpha=0.25)
                fig.suptitle('B=1 BF16 eager SDPA — wall-time medians (passed conditions only)')
                fig.savefig(RESULT_DIR / 'latency.png', dpi=160)
                plt.show()
            print('Decode ms/token은 전체 decode 시간 / 단계 수입니다.')
            print('p90은 반복별 평균 지연시간의 분포이며 개별 토큰 지연시간 p90이 아닙니다.')
        """),
        cell("markdown", "download-info", """
            ## 7. 결과 다운로드

            ZIP에는 원시 반복값과 정확성 검사(`results.json`), 요약(`summary.csv`),
            보고서(`report.md`), 그래프(`latency.png`), 테스트·실행 로그,
            재현용 소스(`source.tar.gz`, `source_manifest.json`)가 들어갑니다.
            실행 실패 시에는 그때까지 생성된 파일만 포함됩니다.
            실행 완료 여부는 `notebook_execution.json`과 조건별 상태를 함께 확인하세요.
            표에 없는 조건이나 진행 중인 조건을 완료된 측정값으로 사용하지 마세요.

            **다운로드를 확인한 뒤 Colab 메뉴에서 런타임 연결 해제 및 삭제를 실행하세요.**
            노트북은 결과를 받을 수 있도록 런타임을 자동으로 종료하지 않습니다.
        """),
        cell("code", "download", """
            #@title 7. 결과 ZIP 다운로드 — 중간 실패 시에도 실행 가능
            import zipfile
            from IPython.display import FileLink, display

            RESULTS_ZIP = WORK_ROOT / (WORK_ROOT.name + '-results.zip')
            with zipfile.ZipFile(RESULTS_ZIP, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
                for path in sorted(RESULT_DIR.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(RESULT_DIR))
            print(f'{RESULTS_ZIP.name} ({RESULTS_ZIP.stat().st_size / 1024:.1f} KiB)')
            try:
                from google.colab import files
            except ImportError:
                display(FileLink(str(RESULTS_ZIP)))
            else:
                files.download(str(RESULTS_ZIP))
        """),
    ]
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "infer_opt": {"source_sha256": source["sha256"],
                          "source_archive_sha256": hashlib.sha256(payload).hexdigest(),
                          "hardware_execution": "not run"},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_notebook(ROOT / "notebooks" / "03_a100_baseline_colab.ipynb")
